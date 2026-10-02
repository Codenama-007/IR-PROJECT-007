"""Central configuration for the RAG fact-check chatbot."""

from pathlib import Path

# Embeddings
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional sentence-transformers model

# LLM (served locally by Ollama)
DEFAULT_LLM = "llama3.2:3b"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1000   # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters
MIN_CHUNK_LENGTH = 50

# Retrieval
DEFAULT_TOP_K = 4
RETRIEVER_COSINE = "Cosine similarity (embeddings)"
RETRIEVER_BM25 = "BM25 (keywords)"

# BM25 parameters
BM25_K1 = 1.5   # term-frequency saturation
BM25_B = 0.75   # document-length normalization
BM25_REMOVE_STOPWORDS = True  # drop common words ("what", "is", "the"...) before scoring

# Persistence: FAISS binary index + JSON metadata
DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_JSON_PATH = DATA_DIR / "chunks.json"

# Web verification
WEB_MAX_RESULTS = 5