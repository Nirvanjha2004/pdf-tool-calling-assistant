"""
PDF Search Tool — Search within extracted PDF text.

Uses simple keyword overlap scoring — no vector embeddings needed.
Supports async background loading so uploads return instantly.
"""

import uuid
import threading
from typing import List, Dict, Optional
from enum import Enum

from utils.pdf_parser import load_and_chunk_pdf


# ─── In-Memory Document Store ────────────────────────────────────────────────

_chunks: List[Dict[str, str]] = []
_document_loaded: bool = False
_extraction_method: str = "none"

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "with", "would",
}


# ─── Job Tracking ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    ERROR     = "error"


_jobs: Dict[str, dict] = {}  # job_id → {status, message, chunks_count, method, error}


def _run_load_job(job_id: str, pdf_path: str) -> None:
    """Worker function that runs in a background thread."""
    global _chunks, _document_loaded, _extraction_method

    _jobs[job_id]["status"] = JobStatus.RUNNING
    try:
        chunks, method = load_and_chunk_pdf(pdf_path)
        _chunks = chunks
        _document_loaded = True
        _extraction_method = method

        _jobs[job_id].update({
            "status":       JobStatus.DONE,
            "chunks_count": len(chunks),
            "method":       method,
            "message": (
                f"Document loaded: {len(chunks)} chunks via {method}."
            ),
        })
    except Exception as e:
        _jobs[job_id].update({
            "status": JobStatus.ERROR,
            "error":  str(e),
            "message": f"Failed to process PDF: {e}",
        })
    finally:
        # Clean up the temp file regardless of success/failure
        try:
            import os
            os.unlink(pdf_path)
        except OSError:
            pass


def load_document_async(pdf_path: str) -> str:
    """
    Start loading a PDF in a background thread.

    Returns a job_id immediately — the caller can poll
    get_job_status(job_id) to know when it's done.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":       JobStatus.PENDING,
        "chunks_count": 0,
        "method":       "none",
        "message":      "Processing...",
        "error":        None,
    }
    t = threading.Thread(
        target=_run_load_job,
        args=(job_id, pdf_path),
        daemon=True,
        name=f"pdf-load-{job_id[:8]}",
    )
    t.start()
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    """Return the current status dict for a job, or None if not found."""
    return _jobs.get(job_id)


# ─── Kept for backwards compat (used by chat_loop terminal mode) ──────────────

def load_document(pdf_path: str) -> str:
    """Synchronous load — blocks until done. Used by CLI / tests only."""
    global _chunks, _document_loaded, _extraction_method
    chunks, method = load_and_chunk_pdf(pdf_path)
    _chunks = chunks
    _document_loaded = True
    _extraction_method = method
    return f"Document loaded: {len(chunks)} chunks created from {pdf_path}"


def is_document_loaded() -> bool:
    return _document_loaded

def get_chunk_count() -> int:
    return len(_chunks)

def get_extraction_method() -> str:
    return _extraction_method


# ─── Search ───────────────────────────────────────────────────────────────────

def search_document(query: str) -> str:
    """
    Search the loaded document for chunks relevant to the query.

    Returns top 3 scoring chunks. If nothing scores > 0 (e.g. a broad
    "tell me about this" query), returns all chunks so the LLM can summarize.
    """
    if not _document_loaded:
        return (
            "No document loaded. Please upload a PDF first."
        )
    if not _chunks:
        return "Document is empty — no text was extracted from the PDF."

    query_words = [
        w for w in query.lower().split()
        if len(w) > 2 and w not in _STOP_WORDS
    ]
    if not query_words:
        return "Query too short. Please provide a more detailed question."

    scored: List[tuple] = []
    for chunk in _chunks:
        chunk_lower = chunk["text"].lower()
        score = sum(1 for w in query_words if w in chunk_lower)
        scored.append((score, chunk["text"], chunk["index"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Cap total text sent to LLM at ~6000 chars to avoid hitting token limits
    MAX_CHARS = 6000

    if scored[0][0] == 0:
        # No keyword match — return all chunks for summarization queries
        parts = ["Here is the full content extracted from the document:"]
        total = 0
        for _, text, idx in scored:
            if total + len(text) > MAX_CHARS:
                break
            parts.append(f"[Chunk {idx + 1}]\n{text}")
            total += len(text)
    else:
        parts = []
        total = 0
        for score, text, idx in scored[:3]:
            if total + len(text) > MAX_CHARS:
                break
            parts.append(f"[Chunk {idx + 1} — relevance: {score}]\n{text}")
            total += len(text)

    return "\n\n".join(parts)


# ─── Tool Definition ──────────────────────────────────────────────────────────

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search within the loaded PDF document for content relevant to "
            "the user's question. Use this whenever the user asks about "
            "the document they uploaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find in the document",
                }
            },
            "required": ["query"],
        },
    },
}


def handle_tool_call(arguments: dict) -> str:
    return search_document(arguments.get("query", ""))
