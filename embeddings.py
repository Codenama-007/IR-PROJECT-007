"""Embedding generation with all-MiniLM-L6-v2 (sentence-transformers)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the model once and reuse it."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts, batch_size: int = 64) -> np.ndarray:
    """Embed a list of strings -> float32 array of shape (n, 384), L2-normalized."""
    vectors = get_embedder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # normalized + inner product = cosine similarity
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query -> array of shape (1, 384)."""
    return embed_texts([query])