"""
FastAPI backend for the transcript KB pipeline.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Requires (see requirements.txt):
    fastapi uvicorn python-multipart openai-whisper sentence-transformers
    psycopg2-binary requests

Env vars required:
    OPENROUTER_API_KEY
    PGDATABASE, PGUSER, PGPASSWORD, PGHOST, PGPORT  (or defaults in pipeline.py)
"""

import os
import shutil
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pipeline

app = FastAPI(title="Transcript KB Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for real production use
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job store. For real production, swap this for Redis/DB-backed
# job tracking (e.g. Celery/RQ) so jobs survive a server restart.
JOBS = {}


def update_job(job_id: str, message: str, status: str = "running"):
    JOBS[job_id]["status"] = status
    JOBS[job_id]["messages"].append(message)
    JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()


def run_pipeline_job(job_id: str, file_path: str, source_name: str,
                      model_name: str, chunk_seconds: float, skip_correction: bool):
    try:
        def progress_cb(msg):
            update_job(job_id, msg)

        num_chunks = pipeline.process_file(
            file_path=file_path,
            source_name=source_name,
            model_name=model_name,
            chunk_seconds=chunk_seconds,
            skip_correction=skip_correction,
            progress_cb=progress_cb,
        )
        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["result"] = {"chunks_saved": num_chunks}
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("base"),
    chunk_seconds: float = Form(30.0),
    skip_correction: bool = Form(False),
):
    job_id = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "status": "queued",
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "source_file": file.filename,
    }

    background_tasks.add_task(
        run_pipeline_job, job_id, saved_path, file.filename, model, chunk_seconds, skip_correction
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search")
async def search(req: SearchRequest):
    try:
        results = pipeline.search_kb(req.query, req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")