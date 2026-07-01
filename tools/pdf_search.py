"""
PDF Search Tool — Search within extracted PDF text.

Phase 3 of the learning path: the LLM decides when to
search the document vs. answer from knowledge vs. calculate.

Uses simple keyword overlap scoring — no vector embeddings needed.
"""

from typing import List, Dict

from utils.pdf_parser import load_and_chunk_pdf


# ─── In-Memory Document Store ────────────────────────────────────────────────
# Simple global store. For production, use a proper DB or vector store.

_chunks: List[Dict[str, str]] = []
_document_loaded: bool = False

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


def load_document(pdf_path: str) -> str:
    """
    Load a PDF document into memory for searching.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Status message with chunk count.
    """
    global _chunks, _document_loaded

    _chunks = load_and_chunk_pdf(pdf_path)
    _document_loaded = True

    return f"Document loaded: {len(_chunks)} chunks created from {pdf_path}"


def is_document_loaded() -> bool:
    """Check if a document is currently loaded in memory."""
    return _document_loaded


def get_chunk_count() -> int:
    """Return the number of chunks in the loaded document."""
    return len(_chunks)


def search_document(query: str) -> str:
    """
    Search the loaded document for chunks relevant to the query.

    Uses simple keyword overlap scoring (word frequency matching).
    Returns the top 3 most relevant chunks.

    Args:
        query: Natural language search query.

    Returns:
        Formatted string with up to 3 matching chunks.
    """
    if not _document_loaded:
        return (
            "No document loaded. Please upload a PDF first using the "
            "/upload endpoint or /load command."
        )

    if not _chunks:
        return "Document is empty - no chunks to search."

    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2 and w not in _STOP_WORDS]

    if not query_words:
        return "Query too short. Please provide a more detailed question."

    # Score each chunk by counting how many query words appear in it.
    # Keep the best chunk even if nothing matches exactly, so the caller
    # still gets a useful excerpt instead of a dead-end response.
    scored_chunks: List[tuple] = []
    for chunk in _chunks:
        chunk_lower = chunk["text"].lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        scored_chunks.append((score, chunk["text"], chunk["index"]))

    # Sort by relevance score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    if not scored_chunks:
        return "Document is empty - no searchable text was extracted from the PDF."

    # Return top 3 chunks, even if the best score is 0.
    top_chunks = scored_chunks[:3]

    if top_chunks[0][0] == 0:
        # No keyword match — return ALL chunks so the LLM has the full document
        # content to summarize or answer "tell me about this document" queries.
        result_parts = ["Here is the full content extracted from the document:"]
        result_parts.extend(
            f"[Chunk {idx + 1}]\n{text}" for _, text, idx in scored_chunks
        )
    else:
        result_parts = [
            f"[Chunk {idx + 1} — relevance score: {score}]\n{text}"
            for score, text, idx in top_chunks
        ]

    return "\n\n".join(result_parts)


# ─── Tool Definition (JSON Schema for the LLM) ───────────────────────────────

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
    """Execute the PDF search tool with the given arguments."""
    query = arguments.get("query", "")
    return search_document(query)
