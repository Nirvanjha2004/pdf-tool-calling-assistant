"""
PDF Parser — Extract and chunk text from PDF files.

Phase 2 of the learning path: basic PDF parsing with pdfplumber.
Falls back to pypdfium2 for PDFs where pdfplumber returns empty text
(e.g. digitally-created PDFs with non-standard encoding).
Chunks are stored in memory — no vector DB needed at this scale.
"""

import pdfplumber
import pypdfium2 as pdfium
from typing import List, Dict


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                full_text += f"\n\n--- Page {i + 1} ---\n{text.strip()}"
    return full_text.strip()


def _extract_with_pypdfium2(pdf_path: str) -> str:
    """Extract text using pypdfium2 (fallback for tricky PDFs)."""
    full_text = ""
    doc = pdfium.PdfDocument(pdf_path)
    for i, page in enumerate(doc):
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        if text and text.strip():
            full_text += f"\n\n--- Page {i + 1} ---\n{text.strip()}"
    return full_text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, page by page.

    Tries pdfplumber first (best for most PDFs). If that returns
    empty or near-empty text, falls back to pypdfium2 which handles
    a wider range of PDF encodings.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text of the document with page markers.
    """
    text = _extract_with_pdfplumber(pdf_path)

    # If pdfplumber got very little text (under 50 chars), try pypdfium2
    if len(text) < 50:
        text = _extract_with_pypdfium2(pdf_path)

    return text


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
