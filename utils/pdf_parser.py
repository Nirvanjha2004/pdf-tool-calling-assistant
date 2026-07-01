"""
PDF Parser — Extract and chunk text from PDF files.

Three-tier extraction pipeline (ordered by speed):
  1. pypdfium2   — fastest, handles most PDFs including tricky encodings
  2. pdfplumber  — fallback for edge cases pypdfium2 misses
  3. rapidocr    — OCR for scanned / image-only PDFs (lazy-loaded)

The RapidOCR engine is pre-warmed at server startup in a background thread
so it is ready by the time an image PDF is uploaded.

--- Why RapidOCR instead of easyocr ---
  easyocr is a full PyTorch model — on CPU (no CUDA build, or no GPU at
  all) that overhead is the actual bottleneck, not page count. RapidOCR
  runs PP-OCR mobile models via ONNXRuntime, which is built for exactly
  this: fast CPU inference, no torch dependency, much smaller install.
  Typically 3-5x faster than easyocr for the same scanned document on CPU.

--- OCR speed optimizations ---
  - Pages are OCR'd concurrently across a thread pool — ONNXRuntime
    sessions are safe to call from multiple threads, so pages don't
    have to be processed one after another.
  - Pages are rendered at a lower default DPI (150 instead of 200) and
    downscaled to a max width cap, since OCR accuracy plateaus well
    before 200 DPI but compute cost grows ~quadratically with pixels.
  - Images are converted to grayscale before being handed to the model.

pip install rapidocr-onnxruntime
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import pypdfium2 as pdfium
import pdfplumber

logger = logging.getLogger(__name__)

# ─── Tunables (env-overridable) ──────────────────────────────────────────────

OCR_DPI = int(os.getenv("OCR_DPI", "150"))                     # render DPI for OCR pages
OCR_MAX_WIDTH_PX = int(os.getenv("OCR_MAX_WIDTH_PX", "1600"))  # hard cap on rendered width
OCR_WORKERS = int(os.getenv("OCR_WORKERS", str(max(os.cpu_count() or 1, 1))))

# ─── RapidOCR lazy singleton ─────────────────────────────────────────────────
# Loaded once in a background thread at startup — never blocks an upload.

_ocr_engine = None
_ocr_lock = threading.Lock()
_ocr_ready = threading.Event()  # set once the model is loaded


def prewarm_ocr() -> None:
    """
    Load the RapidOCR engine in a background thread at server startup.
    Call this once from the FastAPI lifespan handler — it returns immediately.
    """
    def _load():
        global _ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Pre-warming RapidOCR engine (background thread)...")
            engine = RapidOCR()
            with _ocr_lock:
                _ocr_engine = engine
            _ocr_ready.set()
            logger.info("RapidOCR engine ready.")
        except Exception as e:
            logger.warning("RapidOCR pre-warm failed: %s", e)
            _ocr_ready.set()  # unblock waiters even on failure

    t = threading.Thread(target=_load, daemon=True, name="ocr-prewarm")
    t.start()


def _get_ocr_engine():
    """Return the RapidOCR engine, waiting up to 60 s for pre-warm to finish."""
    _ocr_ready.wait(timeout=60)
    with _ocr_lock:
        return _ocr_engine


# ─── Tier 1: pypdfium2 (primary — fastest) ───────────────────────────────────

def _extract_with_pypdfium2(pdf_path: str) -> str:
    """Extract text using pypdfium2 — fast and handles most PDF types."""
    full_text = ""
    doc = pdfium.PdfDocument(pdf_path)
    for i, page in enumerate(doc):
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        if text and text.strip():
            full_text += f"\n\n--- Page {i + 1} ---\n{text.strip()}"
    return full_text.strip()


# ─── Tier 2: pdfplumber (fallback) ───────────────────────────────────────────

def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (good for complex layouts/tables)."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                full_text += f"\n\n--- Page {i + 1} ---\n{text.strip()}"
    return full_text.strip()


# ─── Tier 3: rapidocr (OCR — image PDFs) ─────────────────────────────────────

def _render_pages_for_ocr(pdf_path: str):
    """
    Render every page to a downscaled grayscale numpy array, ready for OCR.
    Lower DPI + grayscale + width cap = far less data for the model to chew on,
    with negligible accuracy loss for typical scanned documents.
    """
    import numpy as np
    from PIL import Image

    doc = pdfium.PdfDocument(pdf_path)
    images = []
    scale = OCR_DPI / 72

    for page in doc:
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil().convert("L")  # grayscale, not RGB

        if pil_image.width > OCR_MAX_WIDTH_PX:
            ratio = OCR_MAX_WIDTH_PX / pil_image.width
            new_size = (OCR_MAX_WIDTH_PX, max(int(pil_image.height * ratio), 1))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        images.append(np.array(pil_image))

    return images


def _ocr_single_page(engine, img) -> str:
    """Run RapidOCR on one page image and join the recognized lines."""
    result, _elapse = engine(img)
    if not result:
        return ""
    # result is a list of [box, text, confidence] triples
    return "\n".join(line[1] for line in result).strip()


def _run_ocr(engine, images) -> List[str]:
    """
    OCR all page images concurrently. ONNXRuntime sessions are safe to call
    from multiple threads, so this fans pages out across a thread pool
    instead of processing them one after another — a big win on multi-core
    CPUs since each page's OCR is independent of the others.
    """
    page_texts: List[str] = [""] * len(images)

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        futures = {
            pool.submit(_ocr_single_page, engine, img): i
            for i, img in enumerate(images)
        }
        for future in futures:
            i = futures[future]
            try:
                page_texts[i] = future.result()
            except Exception as e:
                logger.warning("OCR failed on page %d: %s", i + 1, e)

    return page_texts


def _extract_with_rapidocr(pdf_path: str) -> str:
    """
    OCR every page, fanned out across threads instead of a sequential loop.
    """
    engine = _get_ocr_engine()
    if engine is None:
        logger.warning("RapidOCR engine unavailable — OCR skipped.")
        return ""

    t0 = time.perf_counter()
    images = _render_pages_for_ocr(pdf_path)
    if not images:
        return ""

    page_texts = _run_ocr(engine, images)

    full_text = ""
    for i, page_text in enumerate(page_texts):
        if page_text:
            full_text += f"\n\n--- Page {i + 1} (OCR) ---\n{page_text}"

    logger.info(
        "OCR'd %d pages in %.2fs (workers=%d, dpi=%d)",
        len(images), time.perf_counter() - t0, OCR_WORKERS, OCR_DPI,
    )
    return full_text.strip()


# ─── Main orchestrator ────────────────────────────────────────────────────────

def _is_equation_heavy(pdf_path: str, extracted_text: str) -> bool:
    """
    Heuristic: returns True when a PDF likely has embedded equation images
    that text extraction missed.

    Signal: chars-per-page is low relative to what a text-only page would have,
    AND the text contains question/answer patterns (Q., options like (A) (B))
    with suspiciously short or absent option content.
    """
    try:
        doc = pdfium.PdfDocument(pdf_path)
        num_pages = max(len(doc), 1)
    except Exception:
        num_pages = 1

    chars_per_page = len(extracted_text) / num_pages

    # A typical text page has 1500-3000 chars. Math exam PDFs with embedded
    # equations often come in at <800 chars/page because the equations are gone.
    if chars_per_page > 1200:
        return False  # Plenty of text — no need for OCR

    # Secondary signal: option lines with nothing after them, e.g. "(A) \n(B)"
    import re
    empty_options = re.findall(r'\([A-D]\)\s*\n', extracted_text)
    return len(empty_options) >= 2  # 2+ blank options = almost certainly equation PDF


def extract_text_from_pdf(pdf_path: str) -> tuple[str, str]:
    """
    Extract all text from a PDF using a smart 3-tier pipeline.

    For PDFs that contain embedded image objects (e.g. MathType equations
    exported from Word), text extraction alone misses the math. In that case
    we run OCR on every page and merge it with the extracted text so both
    the prose AND the equations are captured.

    Returns:
        (text, method) where method is one of:
        'pypdfium2', 'pdfplumber', 'rapidocr', 'pypdfium2+ocr', or 'none'
    """
    # Tier 1 — pypdfium2 (fastest)
    text = _extract_with_pypdfium2(pdf_path)
    has_text = len(text) >= 50

    # Tier 2 — pdfplumber fallback if pypdfium2 got nothing
    if not has_text:
        logger.info("pypdfium2 got too little text, trying pdfplumber...")
        text = _extract_with_pdfplumber(pdf_path)
        has_text = len(text) >= 50

    # Check whether the PDF has embedded image objects (equations, figures)
    # If so, merge OCR output even though we already have text — this picks
    # up MathType/Word equation images that text extraction silently skips.
    if has_text and _is_equation_heavy(pdf_path, text):
        logger.info("Equation-heavy PDF detected — merging OCR for math expressions...")
        ocr_text = _extract_with_rapidocr(pdf_path)
        if ocr_text:
            merged = text + "\n\n--- OCR pass (equations/images) ---\n" + ocr_text
            logger.info("Merged text+OCR (%d chars)", len(merged))
            return merged, "pypdfium2+ocr"
        return text, "pypdfium2"

    if has_text:
        method = "pypdfium2" if "pypdfium2" not in locals() else "pypdfium2"
        logger.info("Extracted via text (%d chars)", len(text))
        return text, "pypdfium2"

    # Tier 3 — pure OCR (fully image-based PDF)
    logger.info("Both text tiers failed, falling back to pure RapidOCR...")
    text = _extract_with_rapidocr(pdf_path)
    if text:
        logger.info("Extracted via rapidocr (%d chars)", len(text))
        return text, "rapidocr"

    logger.warning("All extraction tiers failed.")
    return "", "none"


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 30,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping word-level chunks for search.

    Args:
        text: Full document text.
        chunk_size: Words per chunk.
        overlap: Overlapping words between adjacent chunks.

    Returns:
        List of dicts: { "text": ..., "index": ... }
    """
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunks.append({
            "text": " ".join(words[i : i + chunk_size]),
            "index": len(chunks),
        })
    return chunks


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 300,
    overlap: int = 30,
) -> tuple[List[Dict[str, str]], str]:
    """
    Load a PDF and return (chunks, extraction_method).

    The extraction_method tells the caller how the text was obtained
    so it can surface that info to the user (e.g. "via OCR").
    """
    text, method = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap), method