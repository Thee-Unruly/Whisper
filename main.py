"""
FastAPI backend for Signal Transcript Knowledge Base.
Fully database-backed Level 1 architecture:
  - Non-blocking staged async workers
  - Enterprise Submodules routing & isolation
  - Resilient PostgreSQL job & chunk state machine
  - Cosine vector search with pgvector HNSW & full-text search

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import shutil
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any
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
    logger.info("Initializing database schema, submodules, and indices...")
    db.init_db()

    logger.info("Spawning in-process Level 1 stage worker tasks...")
    asr_task = asyncio.create_task(workers.asr_stage_worker())
    groq_task = asyncio.create_task(workers.groq_stage_worker())
    embed_task = asyncio.create_task(workers.embedding_stage_worker())
    synth_task = asyncio.create_task(workers.synthesis_stage_worker())

    yield

    logger.info("Shutting down background stage workers...")
    asr_task.cancel()
    groq_task.cancel()
    embed_task.cancel()
    synth_task.cancel()


app = FastAPI(title="Signal — Enterprise Neural Audio Knowledge Base", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DBConfigRequest(BaseModel):
    host: str
    port: int = 5432
    dbname: str = "postgres"
    user: str = "postgres"
    password: str


class SubmoduleCreateRequest(BaseModel):
    code: str
    name: str
    description: str = ""
    db_target: Optional[str] = None


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
    return {"status": "ok", "database_connected": db_ok, "config": db.get_current_db_config()}


@app.get("/api/db/config")
def get_db_config():
    """Returns current active database connection configuration."""
    return db.get_current_db_config()


@app.post("/api/db/test")
def test_db_config(req: DBConfigRequest):
    """Tests connection to a specified PostgreSQL or Supabase instance."""
    ok, message = db.test_connection_params(req.dict())
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/db/config")
def set_db_config(req: DBConfigRequest):
    """Dynamically switches the active database connection pool in runtime."""
    ok, message = db.update_db_config(req.dict())
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message, "config": db.get_current_db_config()}


@app.post("/api/db/purge")
def purge_db():
    """
    Purges all existing records (jobs, chunks), drops tables,
    and re-executes the clean restructured schema with all 12 enterprise submodules.
    """
    ok, message = db.purge_and_reinit_db()
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    
    # Clean staging files
    try:
        for f in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    except Exception as e:
        logger.warning(f"Could not purge uploads directory: {e}")

    return {
        "ok": True, 
        "message": message, 
        "stats": db.get_stats(),
        "submodules": db.get_submodules()
    }


# ==========================================
# Submodules Management Endpoints
# ==========================================

@app.get("/api/submodules")
def list_submodules():
    """Returns all available submodules with file and chunk counts."""
    return {"submodules": db.get_submodules()}


@app.post("/api/submodules")
def create_submodule(req: SubmoduleCreateRequest):
    """Creates a new custom submodule or updates an existing one."""
    sub = db.add_or_update_submodule(
        code=req.code,
        name=req.name,
        description=req.description,
        db_target=req.db_target
    )
    return {"ok": True, "submodule": sub}


# ==========================================
# Ingestion & State Machine Endpoints
# ==========================================

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    model: str = Form("base"),
    chunk_seconds: float = Form(30.0),
    skip_correction: bool = Form(False),
    submodule: str = Form("02_finance"),
):
    """
    Accepts an audio/video upload, stores it in staging, creates a 'queued'
    job record in PostgreSQL routed to the selected submodule, and returns the job_id.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    temp_id = uuid.uuid4().hex[:8]
    sanitized_filename = f"{temp_id}_{file.filename}"
    dest_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    # Persist queued job in PostgreSQL with target submodule
    job_id = db.create_job(
        source_filename=file.filename,
        file_path=dest_path,
        model_name=model,
        chunk_seconds=chunk_seconds,
        skip_correction=skip_correction,
        submodule_code=submodule,
        file_size_bytes=file_size,
    )

    logger.info(f"Enqueued job {job_id[:8]} for '{file.filename}' -> [{submodule}] (model={model})")
    return {"job_id": job_id, "submodule": submodule}


@app.get("/status/{job_id}")
def status(job_id: str):
    """Returns the live state and progress of a pipeline job from PostgreSQL."""
    job_info = db.get_job_status(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_info


@app.get("/api/jobs/{job_id}/transcript")
def get_job_transcript(job_id: str):
    """Returns full clean transcript and timestamped chunks for PDF / document export."""
    job = db.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    chunks = db.get_job_chunks(job_id)
    full_text = db.get_full_job_transcript(job_id)
    return {
        "job_id": job_id,
        "source_filename": job["source_filename"],
        "model_name": job.get("model_name", "base"),
        "submodule_code": job.get("submodule_code", "02_finance"),
        "submodule_name": job.get("submodule_name", "02. Finance"),
        "duration_seconds": job.get("duration_seconds"),
        "summary": job["summary"],
        "action_items": job["action_items"],
        "full_text": full_text,
        "chunks": chunks
    }


# ==========================================
# Discovery & Search Endpoints
# ==========================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    submodule: Optional[str] = "all"
    client: Optional[str] = None


class AskRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    top_k: int = 5
    submodule: Optional[str] = "all"
    client: Optional[str] = None


@app.post("/search")
@app.post("/api/search")
def search(req: SearchRequest):
    """Semantic vector search using cosine distance (<=>), optionally scoped to a submodule and client."""
    if not req.query.strip():
        return {"results": []}
    try:
        results = pipeline.search_kb(
            query=req.query, 
            top_k=req.top_k, 
            submodule=req.submodule, 
            client=req.client
        )
        return {"results": results, "submodule": req.submodule, "client": req.client}
    except Exception as e:
        logger.error(f"Search query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
@app.post("/api/ask")
async def ask(req: AskRequest):
    """Answers user question using retrieved transcript chunks and Groq LLM synthesis, optionally scoped to a submodule and client."""
    q = (req.question or req.query or "").strip()
    if not q:
        return {"answer": "Please provide a valid question.", "sources": []}
    try:
        response = await pipeline.ask_kb(
            query=q, 
            top_k=req.top_k, 
            submodule=req.submodule, 
            client=req.client
        )
        return response
    except Exception as e:
        logger.error(f"Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Returns database chunk and file counts, plus submodules breakdown."""
    return db.get_stats()


# ==========================================
# Evaluation UI Endpoints
# ==========================================

class EvalRequest(BaseModel):
    test_cases: List[Dict[str, Any]]
    top_k: int = 5


@app.post("/api/eval/run")
def run_evaluation(req: EvalRequest):
    """
    Runs IR retrieval metrics evaluation (Precision, Recall, F1, Hit Rate, MRR)
    over a set of ground truth test cases supporting doc IDs, filenames, or expected keywords.
    """
    import evaluate_retrieval
    if not req.test_cases:
        raise HTTPException(status_code=400, detail="Test cases cannot be empty.")

    total_queries = len(req.test_cases)
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_hit_rate = 0.0
    sum_mrr = 0.0

    query_results = []

    for item in req.test_cases:
        query = item.get("query", "")
        # Accepts 'expected', 'expected_keywords', 'expected_files', or 'relevant_doc_ids'
        relevant_specs = item.get("expected") or item.get("expected_keywords") or item.get("expected_files") or item.get("relevant_doc_ids") or []
        if isinstance(relevant_specs, (str, int)):
            relevant_specs = [relevant_specs]

        client = item.get("client")
        module = item.get("module")

        retrieved = evaluate_retrieval.retrieve_top_k(query, top_k=req.top_k, client=client, module=module)

        metrics = evaluate_retrieval.calculate_metrics_flexible(relevant_specs, retrieved, req.top_k)

        sum_precision += metrics["precision"]
        sum_recall += metrics["recall"]
        sum_f1 += metrics["f1_score"]
        sum_hit_rate += metrics["hit_rate"]
        sum_mrr += metrics["mrr"]

        query_results.append({
            "query": query,
            "expected": relevant_specs,
            "retrieved": retrieved,
            "metrics": metrics
        })

    summary = {
        "mean_precision": round(sum_precision / total_queries, 4),
        "mean_recall": round(sum_recall / total_queries, 4),
        "mean_f1_score": round(sum_f1 / total_queries, 4),
        "mean_hit_rate": round(sum_hit_rate / total_queries, 4),
        "mrr": round(sum_mrr / total_queries, 4),
        "total_queries": total_queries,
        "top_k": req.top_k
    }

    return {
        "summary": summary,
        "details": query_results,
        "explanation": evaluate_retrieval.explain_metrics(summary, query_results),
    }



# ==========================================
# Agentic Conversational AI Endpoints
# ==========================================

class AgentChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    submodule: Optional[str] = None


@app.post("/api/agent/chat")
async def agent_chat(req: AgentChatRequest):
    """
    Conversational Agent turn:
    - Asks follow-up clarification if user question doesn't specify module
    - Performs scoped Vector RAG retrieval on PostgreSQL
    - Synthesizes natural answer citing exact timestamps and files
    """
    import agent
    try:
        response = await agent.process_agent_turn(messages=req.messages, forced_submodule=req.submodule)
        return response
    except Exception as e:
        logger.error(f"Agent chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent")
def read_agent_ui():
    """Serves the dedicated white & black Comic Sans Conversational Agent UI."""
    return FileResponse("static/agent.html")


# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")