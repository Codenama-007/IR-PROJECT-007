"""Document processing: PDF -> pages -> text chunks."""

import hashlib
import io
import re

from pypdf import PdfReader

from config import MIN_CHUNK_LENGTH


def extract_pages(file_bytes: bytes):
    """Return a list of (page_number, text) for a PDF given as bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text: str, size: int, overlap: int):
    """Sliding-window chunker that tries to cut at sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    overlap = min(overlap, size // 2)  # large overlap vs small size would make the window crawl
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size * 0.5:  # snap to sentence end without shrinking too much
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > MIN_CHUNK_LENGTH:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def process_documents(files, chunk_size: int, overlap: int):
    """
    files: list of (filename, bytes).
    Returns a list of chunk dicts: {id, text, source, page}.
    """
    chunks = []
    for name, data in files:
        for page_no, text in extract_pages(data):
            for piece in chunk_text(text, chunk_size, overlap):
                chunks.append(
                    {"id": len(chunks), "text": piece, "source": name, "page": page_no}
                )
    return chunks


def make_signature(files, chunk_size: int, overlap: int) -> str:
    """Hash of file contents + chunk settings, used to detect when re-indexing is needed."""
    h = hashlib.md5()
    for name, data in files:
        h.update(name.encode())
        h.update(hashlib.md5(data).digest())
    h.update(f"{chunk_size}-{overlap}".encode())
    return h.hexdigest()