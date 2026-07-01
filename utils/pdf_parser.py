"""
PDF Parser — Extract and chunk text from PDF files.

Phase 2 of the learning path: basic PDF parsing with pdfplumber.
Chunks are stored in memory — no vector DB needed at this scale.
"""

import pdfplumber
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text of the document with page markers.
    """
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                full_text += f"\n\n--- Page {i + 1} ---\n{text.strip()}"

    return full_text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks for search.

    Uses word-level chunking with configurable overlap so that
    context isn't lost at chunk boundaries.

    Args:
        text: Full document text.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of dicts: { "text": ..., "index": ... }
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    if step < 1:
        step = 1  # prevent infinite loop

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({"text": chunk_text, "index": len(chunks)})

    return chunks


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 300,
    overlap: int = 30,
) -> List[Dict[str, str]]:
    """
    One-shot: load a PDF and return chunked text.

    Convenience wrapper that chains extract + chunk.
    """
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)
