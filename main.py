"""
FastAPI backend for Signal Transcript Knowledge Base.
Fully database-backed Level 1 architecture:
  - Non-blocking staged async workers
  - Resilient PostgreSQL job & chunk state machine
  - Cosine vector search with pgvector

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import shutil
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import db
import pipeline
import workers

logger = logging.getLogger("signal.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes DB schema and launches concurrent in-process background worker loops."""
    logger.info("Initializing database schema and indices...")
    db.init_db()

    logger.info("Spawning in-process Level 1 stage worker tasks...")
    asr_task = asyncio.create_task(workers.asr_stage_worker())
    groq_task = asyncio.create_task(workers.groq_stage_worker())
    embed_task = asyncio.create_task(workers.embedding_stage_worker())

    yield

    logger.info("Shutting down background stage workers...")
    asr_task.cancel()
    groq_task.cancel()
    embed_task.cancel()


app = FastAPI(title="Signal — Transcript Knowledge Base", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Healthcheck endpoint for container probes."""
    db_ok = False
    try:
        with db.get_db_cursor(commit=False) as cur:
            cur.execute("SELECT 1")
            db_ok = bool(cur.fetchone())
    except Exception:
        db_ok = False
    return {"status": "ok", "database_connected": db_ok}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    model: str = Form("base"),
    chunk_seconds: float = Form(30.0),
    skip_correction: bool = Form(False),
):
    """
    Accepts an audio/video upload, stores it in staging, creates a 'queued'
    job record in PostgreSQL, and returns the job_id immediately.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    temp_id = uuid.uuid4().hex[:8]
    sanitized_filename = f"{temp_id}_{file.filename}"
    dest_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Persist queued job in PostgreSQL
    job_id = db.create_job(
        source_filename=file.filename,
        file_path=dest_path,
        model_name=model,
        chunk_seconds=chunk_seconds,
        skip_correction=skip_correction,
    )

    logger.info(f"Enqueued job {job_id[:8]} for '{file.filename}' (model={model}, skip_corr={skip_correction})")
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    """Returns the live state and progress of a pipeline job from PostgreSQL."""
    job_info = db.get_job_status(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_info


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search")
def search(req: SearchRequest):
    """Semantic vector search using cosine distance (<=>)."""
    if not req.query.strip():
        return {"results": []}
    try:
        results = pipeline.search_kb(query=req.query, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Returns database chunk and file counts."""
    return db.get_stats()


# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")