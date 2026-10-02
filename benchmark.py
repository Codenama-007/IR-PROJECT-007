"""
Times each indexing stage separately so you can see which one is slow.

Usage:  python benchmark.py path/to/your.pdf
"""

import sys
import time


def stamp(label, start):
    print(f"  -> {label}: {time.perf_counter() - start:.1f}s", flush=True)


if len(sys.argv) < 2:
    sys.exit("Usage: python benchmark.py path/to/your.pdf")

pdf_path = sys.argv[1]
total = time.perf_counter()

print("1. Importing libraries (torch, sentence-transformers, faiss)...", flush=True)
t = time.perf_counter()
from document_processor import extract_pages, process_documents  # noqa: E402
from embeddings import get_embedder  # noqa: E402
import faiss  # noqa: E402,F401
stamp("imports", t)

print("2. Loading embedding model (first run downloads ~90 MB)...", flush=True)
t = time.perf_counter()
embedder = get_embedder()
stamp("model load", t)

print("3. Extracting text from the PDF...", flush=True)
t = time.perf_counter()
with open(pdf_path, "rb") as f:
    data = f.read()
pages = extract_pages(data)
chars = sum(len(text) for _, text in pages)
stamp(f"text extraction ({len(pages)} pages, {chars:,} characters)", t)

print("4. Chunking...", flush=True)
t = time.perf_counter()
chunks = process_documents([(pdf_path, data)], 1000, 200)
stamp(f"chunking ({len(chunks)} chunks)", t)

print("5. Embedding (progress every 64 chunks)...", flush=True)
t = time.perf_counter()
texts = [c["text"] for c in chunks]
for i in range(0, len(texts), 64):
    embedder.encode(texts[i : i + 64], normalize_embeddings=True, show_progress_bar=False)
    print(f"     {min(i + 64, len(texts))}/{len(texts)} chunks", flush=True)
stamp("embedding", t)

print(f"\nTOTAL: {time.perf_counter() - total:.1f}s")