"""Sparse (keyword) retrieval using BM25, implemented from scratch."""

import math
import re
from collections import Counter

from config import BM25_B, BM25_K1, BM25_REMOVE_STOPWORDS

_TOKEN_RE = re.compile(r"\w+")

# Common English words that carry little meaning for keyword matching.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers him his how i if in into is it its
    just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their theirs them then there these they this
    those through to too under until up us very was we were what when where which while
    who whom why will with would you your yours
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = BM25_REMOVE_STOPWORDS) -> list:
    """Lowercase, split into word tokens, and (optionally) drop stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever:
    """
    BM25 score of a chunk d for a query q:

        score(q, d) = sum over terms t in q of
            IDF(t) * f(t,d) * (k1 + 1) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

        IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

    f(t,d)  = how often term t appears in chunk d
    |d|     = chunk length in tokens, avgdl = average chunk length
    N       = number of chunks, n(t) = number of chunks containing t
    """

    def __init__(self, chunks: list, k1: float = BM25_K1, b: float = BM25_B):
        self.chunks = chunks
        self.k1, self.b = k1, b

        tokens = [tokenize(c["text"]) for c in chunks]
        self.term_freqs = [Counter(t) for t in tokens]
        self.doc_len = [len(t) for t in tokens]
        self.n_docs = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.avgdl = self.avgdl or 1.0  # avoid division by zero

        doc_freq = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def scores(self, query: str) -> list:
        """BM25 score of every chunk for the query."""
        result = [0.0] * self.n_docs
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:  # term never appears in any chunk
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                result[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return result

    def search(self, query: str, k: int) -> list:
        """Top-k chunks by BM25 score. Chunks sharing no keyword with the query are dropped."""
        scores = self.scores(query)
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.chunks[i], "score": scores[i]} for i in ranked if scores[i] > 0]