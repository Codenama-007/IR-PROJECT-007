"""
Vector store: FAISS for storing vectors + JSON for human-readable chunk storage.

Two files are written to the data/ folder:
  - faiss.index  : binary FAISS index holding the embedding vectors
  - chunks.json  : the chunk texts + metadata (source file, page, id) and a signature

Row i in the FAISS index corresponds to chunks[i] in chunks.json.
"""

import json

import faiss
import numpy as np

from config import CHUNKS_JSON_PATH, DATA_DIR, INDEX_PATH


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.signature = None
        self._vectors = None  # lazy cache of all vectors

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def build(self, chunks, vectors: np.ndarray, signature: str):
        """Create a fresh FAISS index from chunk vectors."""
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors
        self.index.add(vectors)
        self.chunks = chunks
        self.signature = signature
        self._vectors = None

    def get_vectors(self) -> np.ndarray:
        """All stored vectors as an (n, dim) array, read back from the FAISS index."""
        if self._vectors is None:
            self._vectors = self.index.reconstruct_n(0, self.index.ntotal)
        return self._vectors

    def save(self):
        """Persist FAISS index (binary) and chunks (JSON) to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"signature": self.signature, "num_chunks": len(self.chunks), "chunks": self.chunks},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load(self) -> bool:
        """Load a previously saved index + JSON. Returns True on success."""
        if not (INDEX_PATH.exists() and CHUNKS_JSON_PATH.exists()):
            return False
        try:
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(CHUNKS_JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.signature = data.get("signature")
            self._vectors = None
            return self.is_ready and self.index.ntotal == len(self.chunks)
        except Exception:
            return False

    def search(self, query_vector: np.ndarray, k: int):
        """FAISS-native top-k search (same ranking as cosine_retriever on normalized vectors)."""
        scores, ids = self.index.search(query_vector, min(k, len(self.chunks)))
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]