"""
FastAPI Endpoint — PDF Q&A Assistant API

Key design decisions:
- /upload   → saves the file and starts a background job, returns instantly
- /job/{id} → client polls this to know when processing is done
- easyocr is pre-warmed at startup in a background thread so it's
  ready by the time an image PDF is uploaded
"""

import os
import re
import sys
import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.chat_loop import ask
from tools.pdf_search import (
    load_document_async,
    get_job_status,
    is_document_loaded,
    get_chunk_count,
    get_extraction_method,
    JobStatus,
)
from utils.pdf_parser import prewarm_ocr


# ─── Lifespan: pre-warm OCR model at startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off easyocr model load in background — doesn't block startup
    prewarm_ocr()
    yield
    # (shutdown logic here if needed)


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Q&A Assistant API",
    description="Upload a PDF and ask natural-language questions.",
    version="2.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "PDF Q&A Assistant"}


# ─── Models ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    history: Optional[list] = None

class AskResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    job_id: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # pending | running | done | error
    message: str
    chunks_count: int
    method: str          # pypdfium2 | pdfplumber | easyocr | none
    error: Optional[str]

class HealthResponse(BaseModel):
    status: str
    document_loaded: bool
    chunks_count: int
    extraction_method: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        document_loaded=is_document_loaded(),
        chunks_count=get_chunk_count(),
        extraction_method=get_extraction_method(),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF — returns a job_id immediately.

    The file is saved to a temp location and processing starts in the
    background. Poll GET /job/{job_id} to track progress.
    For normal text PDFs this completes in < 1s.
    For scanned/image PDFs, easyocr OCR runs in the background (~10-30s).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a persistent temp file (background thread will clean it up)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()

    job_id = load_document_async(tmp_path)

    return UploadResponse(
        job_id=job_id,
        message="Upload received. Processing started.",
    )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    """
    Poll this endpoint to check if PDF processing is complete.

    Status values:
      pending  — queued, not started yet
      running  — actively extracting / OCR-ing
      done     — ready for questions
      error    — extraction failed
    """
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        chunks_count=job.get("chunks_count", 0),
        method=job.get("method", "none"),
        error=job.get("error"),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a natural-language question about the loaded document."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = ask(request.question, request.history)
        return AskResponse(answer=result["content"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


# ─── Dev runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
