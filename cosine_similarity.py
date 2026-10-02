"""Dense retrieval using cosine similarity between embedding vectors."""

import numpy as np


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between one query vector and many document vectors.

        cos(q, d) = (q . d) / (||q|| * ||d||)

    Returns values in [-1, 1]; higher means more semantically similar.
    """
    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    dots = doc_vecs @ q
    norms = np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q)
    return dots / np.maximum(norms, 1e-12)  # guard against division by zero


def cosine_search(store, query_vector: np.ndarray, k: int) -> list:
    """Rank all chunks by cosine similarity to the query; return the top-k with scores."""
    sims = cosine_similarity(query_vector, store.get_vectors())
    top_ids = np.argsort(-sims)[:k]
    return [{**store.chunks[i], "score": float(sims[i])} for i in top_ids]