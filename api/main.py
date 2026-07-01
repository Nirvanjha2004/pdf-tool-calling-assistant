"""
FastAPI Endpoint — PDF Q&A Assistant API

Serves both the API endpoints and the static frontend.
"""

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.chat_loop import ask
from tools.pdf_search import load_document, is_document_loaded, get_chunk_count

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Q&A Assistant API",
    description=(
        "Upload a PDF and ask natural-language questions. "
        "The LLM automatically decides whether to use the calculator tool, "
        "search the document, or answer from its own knowledge."
    ),
    version="1.0.0",
)

# ─── Static Files ────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "PDF Q&A Assistant with Tool-Calling",
        "endpoints": {
            "POST /upload": "Upload a PDF document (multipart/form-data)",
            "POST /ask": 'Ask a question (JSON: {"question": "..."})',
            "GET /health": "Health check",
        },
    }


# ─── Pydantic Models ─────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    history: Optional[list] = None


class AskResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    message: str
    chunks_count: int


class HealthResponse(BaseModel):
    status: str
    document_loaded: bool
    chunks_count: int


# ─── API Endpoints ───────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and whether a document is loaded."""
    return HealthResponse(
        status="ok",
        document_loaded=is_document_loaded(),
        chunks_count=get_chunk_count(),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to ask questions about.

    The PDF is parsed, text is extracted, and chunked into
    overlapping segments for search. All stored in memory.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = load_document(tmp_path)

        # Extract chunk count from the result message
        match = re.search(r"(\d+) chunks?", result)
        chunks = int(match.group(1)) if match else 0

        return UploadResponse(message=result, chunks_count=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        # Clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a natural-language question.

    The LLM automatically decides whether to:
    - Use the calculator tool (for math/computations)
    - Search the loaded document (for document questions)
    - Answer from its own knowledge (for general questions)

    Optionally include a 'history' array with previous messages
    to continue a conversation.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(request.question, request.history)
        return AskResponse(answer=result["content"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


# ─── Run (for development) ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Note: .env is loaded automatically by core/llm_client.py on import
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
