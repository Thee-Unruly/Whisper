"""
Database layer for Signal Transcript Knowledge Base.
Provides connection pooling, schema migrations, and short-lived atomic transaction
helpers for the Level 1 task queue and state machine.
"""

import os
import uuid
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("signal.db")

# ---- DATABASE CONFIG ----
DB_CONFIG = {
    "dbname": os.environ.get("PGDATABASE", "transcripts_agile"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
}
EMBEDDING_DIM = 384

_db_pool = None


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        import psycopg2.pool
        _db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            **DB_CONFIG
        )
    return _db_pool


@contextmanager
def get_db_connection():
    pool = get_db_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as conn:
        try:
            with conn.cursor() as cur:
                yield cur
            if commit:
                conn.commit()
        except Exception:
            if commit:
                conn.rollback()
            raise


def _vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def init_db():
    """Idempotent database initialization and schema migration."""
    with get_db_cursor(commit=True) as cur:
        # 1. Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Create ENUM types safely
        cur.execute("""
            DO $$ BEGIN
                CREATE TYPE job_status AS ENUM (
                    'queued', 'transcribing', 'transcribed', 
                    'correcting', 'embedding', 'completed', 
                    'partial_failure', 'failed', 'cancelled'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)

        cur.execute("""
            DO $$ BEGIN
                CREATE TYPE chunk_status AS ENUM (
                    'raw', 'correcting', 'corrected', 
                    'embedding', 'indexed', 'failed'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)

        # 3. Create jobs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_filename TEXT NOT NULL,
                file_path TEXT,
                model_name TEXT NOT NULL DEFAULT 'base',
                chunk_seconds NUMERIC(5,2) NOT NULL DEFAULT 30.0,
                skip_correction BOOLEAN NOT NULL DEFAULT FALSE,
                status job_status NOT NULL DEFAULT 'queued',
                total_chunks INT DEFAULT NULL,
                processed_chunks INT NOT NULL DEFAULT 0,
                error_message TEXT,
                summary TEXT,
                action_items TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Migration safe check if columns exist on older tables
        cur.execute("""
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS summary TEXT;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS action_items TEXT;
        """)

        # 4. Create transcript_chunks table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
                source_file TEXT,
                chunk_index INT NOT NULL,
                start_time NUMERIC(8,2) NOT NULL,
                end_time NUMERIC(8,2) NOT NULL,
                text_raw TEXT NOT NULL,
                text_corrected TEXT,
                text TEXT,
                embedding VECTOR({EMBEDDING_DIM}),
                status chunk_status NOT NULL DEFAULT 'raw',
                retry_count INT NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # 5. Create Indices
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created 
            ON jobs(status, created_at ASC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_job_status 
            ON transcript_chunks(job_id, status);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_status_claim 
            ON transcript_chunks(status, created_at ASC);
        """)

        # Cosine distance HNSW index for semantic vector search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine_hnsw 
            ON transcript_chunks USING hnsw (embedding vector_cosine_ops);
        """)


# ==========================================
# Short-Lived Atomic Transaction Helpers
# ==========================================

def create_job(source_filename: str, file_path: str, model_name: str = "base",
               chunk_seconds: float = 30.0, skip_correction: bool = False) -> str:
    """Inserts a new job in 'queued' state."""
    job_id = str(uuid.uuid4())
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, source_filename, file_path, model_name, chunk_seconds, skip_correction, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'queued')
            RETURNING id;
            """,
            (job_id, source_filename, file_path, model_name, chunk_seconds, skip_correction)
        )
    return job_id


def claim_next_job() -> Optional[Dict[str, Any]]:
    """
    Atomic claim of the next queued job using FOR UPDATE SKIP LOCKED.
    Commits immediately (~2ms), setting status to 'transcribing'.
    """
    with get_db_cursor(commit=True) as cur:
        cur.execute("""
            UPDATE jobs
            SET status = 'transcribing', updated_at = NOW()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, source_filename, file_path, model_name, chunk_seconds, skip_correction;
        """)
        row = cur.fetchone()
        if row:
            return {
                "id": str(row[0]),
                "source_filename": row[1],
                "file_path": row[2],
                "model_name": row[3],
                "chunk_seconds": float(row[4]),
                "skip_correction": bool(row[5]),
            }
    return None


def save_raw_chunks(job_id: str, source_filename: str, chunks: List[Dict[str, Any]],
                    skip_correction: bool = False):
    """
    Bulk inserts transcribed chunks and updates job total count and next stage.
    Runs in a short transaction (~5ms).
    """
    total = len(chunks)
    initial_status = 'corrected' if skip_correction else 'raw'
    next_job_status = 'embedding' if skip_correction else 'correcting'

    with get_db_cursor(commit=True) as cur:
        for idx, c in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO transcript_chunks (
                    job_id, source_file, chunk_index, start_time, end_time,
                    text_raw, text_corrected, text, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::chunk_status);
                """,
                (
                    job_id,
                    source_filename,
                    idx,
                    c["start"],
                    c["end"],
                    c["text"],
                    c["text"] if skip_correction else None,
                    c["text"],
                    initial_status
                )
            )

        cur.execute(
            """
            UPDATE jobs
            SET total_chunks = %s, status = %s::job_status, updated_at = NOW()
            WHERE id = %s;
            """,
            (total, next_job_status, job_id)
        )


def claim_raw_chunks(batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Claims up to `batch_size` raw chunks for Groq correction using SKIP LOCKED,
    fetching the preceding chunk's text for context-aware stitching.
    """
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            WITH to_claim AS (
                SELECT id FROM transcript_chunks
                WHERE status = 'raw'
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE transcript_chunks tc
            SET status = 'correcting', updated_at = NOW()
            FROM to_claim
            WHERE tc.id = to_claim.id
            RETURNING tc.id, tc.job_id, tc.chunk_index, tc.text_raw,
                      (
                          SELECT COALESCE(prev.text_corrected, prev.text_raw, prev.text)
                          FROM transcript_chunks prev
                          WHERE prev.job_id = tc.job_id AND prev.chunk_index = tc.chunk_index - 1
                          LIMIT 1
                      ) AS prev_text;
            """,
            (batch_size,)
        )
        rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "job_id": str(r[1]),
                "chunk_index": r[2],
                "text_raw": r[3],
                "prev_text": r[4],
            }
            for r in rows
        ]


def save_corrected_chunk(chunk_id: str, corrected_text: str):
    """Saves corrected text and marks chunk as 'corrected'."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE transcript_chunks
            SET text_corrected = %s, text = %s, status = 'corrected', updated_at = NOW()
            WHERE id = %s;
            """,
            (corrected_text, corrected_text, chunk_id)
        )


def fail_chunk(chunk_id: str, error_msg: str, max_retries: int = 3):
    """Records failure and either resets to retry or marks failed."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE transcript_chunks
            SET retry_count = retry_count + 1,
                last_error = %s,
                status = CASE WHEN retry_count + 1 >= %s THEN 'failed'::chunk_status ELSE 'raw'::chunk_status END,
                updated_at = NOW()
            WHERE id = %s;
            """,
            (error_msg, max_retries, chunk_id)
        )


def claim_corrected_chunks(batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Claims up to `batch_size` corrected chunks for batch MiniLM embedding.
    """
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            WITH to_claim AS (
                SELECT id FROM transcript_chunks
                WHERE status = 'corrected'
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE transcript_chunks
            SET status = 'embedding', updated_at = NOW()
            FROM to_claim
            WHERE transcript_chunks.id = to_claim.id
            RETURNING transcript_chunks.id, transcript_chunks.job_id, 
                      transcript_chunks.chunk_index, 
                      COALESCE(transcript_chunks.text_corrected, transcript_chunks.text_raw, transcript_chunks.text) AS text_to_embed;
            """,
            (batch_size,)
        )
        rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "job_id": str(r[1]),
                "chunk_index": r[2],
                "text": r[3],
            }
            for r in rows
        ]


def save_chunk_embeddings_and_update_job(chunk_ids: List[str], embeddings: List[List[float]], job_ids: List[str]):
    """
    Bulk saves embeddings, marks chunks as 'indexed', and atomically checks
    if any associated jobs have completed.
    """
    unique_job_ids = list(set(job_ids))
    with get_db_cursor(commit=True) as cur:
        # 1. Update chunks with embeddings
        for chunk_id, emb in zip(chunk_ids, embeddings):
            cur.execute(
                """
                UPDATE transcript_chunks
                SET embedding = %s, status = 'indexed', updated_at = NOW()
                WHERE id = %s;
                """,
                (_vector_literal(emb), chunk_id)
            )

        # 2. Atomic job progress and completion update CTE
        for jid in unique_job_ids:
            cur.execute(
                """
                WITH job_stats AS (
                    SELECT 
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE status = 'indexed') AS indexed_count,
                        COUNT(*) FILTER (WHERE status = 'failed') AS failed_count
                    FROM transcript_chunks
                    WHERE job_id = %s
                )
                UPDATE jobs j
                SET 
                    processed_chunks = s.indexed_count,
                    status = CASE 
                        WHEN (s.indexed_count + s.failed_count) = s.total AND s.failed_count > 0 THEN 'partial_failure'::job_status
                        WHEN s.indexed_count = s.total AND s.total > 0 THEN 'completed'::job_status
                        ELSE j.status
                    END,
                    updated_at = NOW()
                FROM job_stats s
                WHERE j.id = %s;
                """,
                (jid, jid)
            )


def fail_job(job_id: str, error_message: str):
    """Marks a job as failed."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = 'failed', error_message = %s, updated_at = NOW()
            WHERE id = %s;
            """,
            (error_message, job_id)
        )


def get_full_job_transcript(job_id: str) -> str:
    """Returns the concatenated clean transcript for the entire job."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT COALESCE(text_corrected, text_raw, text)
            FROM transcript_chunks
            WHERE job_id = %s
            ORDER BY chunk_index ASC;
            """,
            (job_id,)
        )
        rows = cur.fetchall()
        return "\n\n".join(r[0] for r in rows if r[0])


def save_job_summary(job_id: str, summary: str, action_items: str):
    """Saves the executive summary and key action items for a completed job."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE jobs
            SET summary = %s, action_items = %s, updated_at = NOW()
            WHERE id = %s;
            """,
            (summary, action_items, job_id)
        )


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full job status, summary, and progress formatted for API responses."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, source_filename, model_name, status, total_chunks, 
                   processed_chunks, error_message, summary, action_items, created_at, updated_at
            FROM jobs
            WHERE id = %s;
            """,
            (job_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        status_str = row[3]
        total = row[4]
        processed = row[5]
        error = row[6]
        summary = row[7]
        action_items = row[8]

        # Map to frontend expectations
        if status_str == "completed":
            frontend_status = "complete"
        elif status_str in ("failed", "partial_failure"):
            frontend_status = "error"
        elif status_str == "queued":
            frontend_status = "queued"
        else:
            frontend_status = "running"

        messages = [
            f"Job {job_id[:8]} initialized for {row[1]}",
            f"Current status: {status_str}",
        ]
        if total is not None:
            messages.append(f"Progress: {processed}/{total} chunks processed")
        if summary:
            messages.append("Executive synthesis & action items generated")
        if error:
            messages.append(f"Error: {error}")

        return {
            "id": str(row[0]),
            "source_filename": row[1],
            "status": frontend_status,
            "raw_status": status_str,
            "total_chunks": total,
            "processed_chunks": processed,
            "summary": summary,
            "action_items": action_items,
            "messages": messages,
            "result": {
                "chunks_saved": processed,
                "summary": summary,
                "action_items": action_items,
            },
            "error": error,
        }


def get_stats() -> Dict[str, int]:
    """Returns total indexed chunks and distinct processed files."""
    with get_db_cursor(commit=False) as cur:
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'indexed' OR embedding IS NOT NULL) AS total_chunks,
                COUNT(DISTINCT COALESCE(source_file, 'unknown')) AS total_files
            FROM transcript_chunks;
        """)
        row = cur.fetchone()
        return {
            "total_chunks": row[0] if row else 0,
            "total_files": row[1] if row else 0,
        }


def search_chunks(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs cosine vector search (<=>) on indexed transcript chunks."""
    vec_str = _vector_literal(query_embedding)
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT 
                COALESCE(source_file, 'unknown') AS source_file,
                start_time,
                end_time,
                COALESCE(text_corrected, text_raw, text) AS chunk_text,
                embedding <=> %s::vector AS distance
            FROM transcript_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
            """,
            (vec_str, vec_str, top_k)
        )
        rows = cur.fetchall()
        return [
            {
                "source_file": r[0],
                "start_time": float(r[1]),
                "end_time": float(r[2]),
                "text": r[3],
                "distance": float(r[4]),
            }
            for r in rows
        ]
