"""
Database layer for Signal Transcript Knowledge Base.
Provides connection pooling, schema migrations, submodules management,
and short-lived atomic transaction helpers for the Level 1 task queue and state machine.
"""

import os
import uuid
import json
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("signal.db")

# ---- DATABASE CONFIG ----
# Default to port 5433 (exposed port in docker-compose.yml) or 5432
_DEFAULT_PORT = 5433 if os.environ.get("PGPORT") is None else int(os.environ.get("PGPORT", 5433))

DB_CONFIG = {
    "dbname": os.environ.get("PGDATABASE", "transcripts_agile"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "host": os.environ.get("PGHOST", "localhost"),
    "port": _DEFAULT_PORT,
}
EMBEDDING_DIM = 384

# Default Enterprise Submodules Catalog
DEFAULT_SUBMODULES = [
    {"code": "01_credit", "name": "01. Credit", "description": "Credit facilities, appraisals, approvals, and loan policies"},
    {"code": "02_credit_portal", "name": "02. Credit Portal", "description": "Customer and agent digital credit portal workflows"},
    {"code": "02_finance", "name": "02. Finance", "description": "Accounting periods, financial ledgers, journals, and fiscal policies"},
    {"code": "03_e_recruitment", "name": "03. E-Recruitment", "description": "Hiring requisitions, candidate evaluation, and interviews"},
    {"code": "04_procurement", "name": "04. Procurement", "description": "Requisitions, purchase orders, vendor evaluations, and contracts"},
    {"code": "05_e_procurement", "name": "05. E-Procurement", "description": "Supplier portal, digital tendering, RFQs, and bids"},
    {"code": "06_treasury", "name": "06. Treasury", "description": "Cash flow, banking reconciliations, liquidity, and forex"},
    {"code": "07_grc", "name": "07. GRC", "description": "Governance, risk management, internal audit, and compliance policies"},
    {"code": "08_hr", "name": "08. HR", "description": "Employee relations, talent management, onboarding, and leave"},
    {"code": "09_payroll", "name": "09. Payroll", "description": "Salary structures, statutory deductions, benefits, and payslips"},
    {"code": "10_edms", "name": "10. EDMS", "description": "Electronic Document Management System, archives, and records"},
    {"code": "11_power_bi", "name": "11. Power BI", "description": "BI reports, executive dashboards, KPIs, and data analytics"},
]

_db_pool = None


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        import psycopg2.pool
        try:
            _db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                **DB_CONFIG
            )
        except Exception as e:
            # If port 5433 failed and was the default, try 5432 fallback
            if DB_CONFIG["port"] == 5433:
                logger.warning(f"Connection pool on port 5433 failed ({e}). Attempting fallback to port 5432...")
                try:
                    fallback_config = dict(DB_CONFIG, port=5432)
                    _db_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,
                        maxconn=20,
                        **fallback_config
                    )
                    DB_CONFIG["port"] = 5432
                    logger.info("Successfully connected to fallback PostgreSQL port 5432.")
                    return _db_pool
                except Exception as fallback_e:
                    logger.error(f"Fallback to 5432 also failed: {fallback_e}")
            raise e
    return _db_pool


def test_connection_params(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Tests connecting to PostgreSQL/Supabase with the given parameters and checks pgvector."""
    import psycopg2
    try:
        conn = psycopg2.connect(
            dbname=config.get("dbname", "postgres"),
            user=config.get("user", "postgres"),
            password=config.get("password", "postgres"),
            host=config.get("host", "localhost"),
            port=int(config.get("port", 5432)),
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            # Check vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        conn.close()
        return True, "Successfully connected and verified pgvector extension."
    except Exception as e:
        logger.warning(f"Database test connection failed: {e}")
        return False, str(e)


def update_db_config(new_config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Tests and updates runtime DB configuration, re-establishes connection pool,
    and runs idempotent schema migrations.
    """
    global _db_pool, DB_CONFIG
    
    ok, msg = test_connection_params(new_config)
    if not ok:
        return False, f"Connection test failed: {msg}"
    
    if _db_pool is not None:
        try:
            _db_pool.closeall()
        except Exception as e:
            logger.warning(f"Error closing old pool: {e}")
        _db_pool = None
    
    DB_CONFIG["dbname"] = new_config.get("dbname", "postgres")
    DB_CONFIG["user"] = new_config.get("user", "postgres")
    DB_CONFIG["password"] = new_config.get("password", "postgres")
    DB_CONFIG["host"] = new_config.get("host", "localhost")
    DB_CONFIG["port"] = int(new_config.get("port", 5432))
    
    try:
        get_db_pool()
        init_db()
        return True, f"Connected to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']} successfully!"
    except Exception as e:
        logger.error(f"Failed to re-initialize database with new config: {e}")
        return False, str(e)


def get_current_db_config() -> Dict[str, Any]:
    """Returns safe masked database configuration information."""
    host = DB_CONFIG.get("host", "localhost")
    is_supabase = "supabase" in host.lower()
    return {
        "host": host,
        "port": DB_CONFIG.get("port", _DEFAULT_PORT),
        "dbname": DB_CONFIG.get("dbname", "transcripts_agile"),
        "user": DB_CONFIG.get("user", "postgres"),
        "is_supabase": is_supabase,
        "is_connected": _db_pool is not None
    }


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

        # 3. Create submodules table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS submodules (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                code VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                db_target TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Seed default submodules
        for sub in DEFAULT_SUBMODULES:
            cur.execute("""
                INSERT INTO submodules (code, name, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                SET name = EXCLUDED.name, description = EXCLUDED.description;
            """, (sub["code"], sub["name"], sub["description"]))

        # 4. Create jobs table with submodules & extended metadata
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_filename TEXT NOT NULL,
                file_path TEXT,
                file_size_bytes BIGINT DEFAULT 0,
                duration_seconds NUMERIC(10,2) DEFAULT NULL,
                language VARCHAR(10) DEFAULT 'en',
                model_name TEXT NOT NULL DEFAULT 'base',
                chunk_seconds NUMERIC(5,2) NOT NULL DEFAULT 30.0,
                skip_correction BOOLEAN NOT NULL DEFAULT FALSE,
                submodule_id UUID REFERENCES submodules(id) ON DELETE SET NULL,
                submodule_code VARCHAR(50) NOT NULL DEFAULT '02_finance',
                submodule_name VARCHAR(100) NOT NULL DEFAULT '02. Finance',
                status job_status NOT NULL DEFAULT 'queued',
                total_chunks INT DEFAULT NULL,
                processed_chunks INT NOT NULL DEFAULT 0,
                failed_chunks INT NOT NULL DEFAULT 0,
                error_message TEXT,
                summary TEXT,
                action_items TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ DEFAULT NULL
            );
        """)

        # Migration-safe column checks on jobs
        cur.execute("""
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS submodule_id UUID REFERENCES submodules(id) ON DELETE SET NULL;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS submodule_code VARCHAR(50) NOT NULL DEFAULT '02_finance';
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS submodule_name VARCHAR(100) NOT NULL DEFAULT '02. Finance';
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT DEFAULT 0;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS duration_seconds NUMERIC(10,2) DEFAULT NULL;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en';
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS failed_chunks INT NOT NULL DEFAULT 0;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ DEFAULT NULL;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS summary TEXT;
            ALTER TABLE jobs ADD COLUMN IF NOT EXISTS action_items TEXT;
        """)

        # 5. Create transcript_chunks table with full-text search tsvector and submodules
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
                source_file TEXT NOT NULL,
                chunk_index INT NOT NULL,
                start_time NUMERIC(10,2) NOT NULL,
                end_time NUMERIC(10,2) NOT NULL,
                submodule_id UUID REFERENCES submodules(id) ON DELETE SET NULL,
                submodule_code VARCHAR(50) NOT NULL DEFAULT '02_finance',
                submodule_name VARCHAR(100) NOT NULL DEFAULT '02. Finance',
                speaker VARCHAR(100) DEFAULT NULL,
                text_raw TEXT NOT NULL,
                text_corrected TEXT,
                text TEXT,
                embedding VECTOR(384),
                tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, text_corrected, text_raw, ''))) STORED,
                status chunk_status NOT NULL DEFAULT 'raw',
                retry_count INT NOT NULL DEFAULT 0,
                last_error TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Migration-safe column checks on transcript_chunks
        cur.execute("""
            ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS submodule_id UUID REFERENCES submodules(id) ON DELETE SET NULL;
            ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS submodule_code VARCHAR(50) NOT NULL DEFAULT '02_finance';
            ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS submodule_name VARCHAR(100) NOT NULL DEFAULT '02. Finance';
            ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS speaker VARCHAR(100) DEFAULT NULL;
            ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
        """)

        # 6. Create Indices
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created 
            ON jobs(status, created_at ASC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_submodule 
            ON jobs(submodule_code);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_job_status 
            ON transcript_chunks(job_id, status);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_job_index 
            ON transcript_chunks(job_id, chunk_index ASC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_submodule 
            ON transcript_chunks(submodule_code);
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

        # GIN index for full-text lexical search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin 
            ON transcript_chunks USING gin (tsv);
        """)


def purge_and_reinit_db() -> Tuple[bool, str]:
    """
    Completely purges all existing tables (jobs, chunks, submodules),
    clears types, and recreates the restructured schema with all 12 submodules pre-seeded.
    """
    global _db_pool
    try:
        with get_db_cursor(commit=True) as cur:
            logger.warning("PURGING DATABASE: Dropping transcript_chunks, jobs, submodules...")
            cur.execute("DROP TABLE IF EXISTS transcript_chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS jobs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS submodules CASCADE;")
            cur.execute("DROP TYPE IF EXISTS chunk_status CASCADE;")
            cur.execute("DROP TYPE IF EXISTS job_status CASCADE;")
        
        # Re-initialize clean schema
        init_db()
        logger.info("DATABASE PURGE & RESTRUCTURE COMPLETE: All tables and 12 submodules initialized.")
        return True, "Database successfully purged and restructured with all 12 submodules initialized."
    except Exception as e:
        logger.error(f"Failed to purge and reinit DB: {e}", exc_info=True)
        return False, str(e)


# ==========================================
# Submodules Management
# ==========================================

def get_submodules() -> List[Dict[str, Any]]:
    """Returns all submodules with aggregated total jobs and indexed chunk counts."""
    with get_db_cursor(commit=False) as cur:
        cur.execute("""
            SELECT 
                s.id, 
                s.code, 
                s.name, 
                s.description, 
                s.db_target,
                COUNT(DISTINCT j.id) AS total_jobs,
                COUNT(c.id) FILTER (WHERE c.status = 'indexed' OR c.embedding IS NOT NULL) AS indexed_chunks
            FROM submodules s
            LEFT JOIN jobs j ON j.submodule_code = s.code
            LEFT JOIN transcript_chunks c ON c.submodule_code = s.code
            GROUP BY s.id, s.code, s.name, s.description, s.db_target
            ORDER BY s.code ASC;
        """)
        rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "code": r[1],
                "name": r[2],
                "description": r[3] or "",
                "db_target": r[4],
                "total_jobs": r[5],
                "indexed_chunks": r[6],
            }
            for r in rows
        ]


def get_submodule_info(code: str) -> Optional[Dict[str, Any]]:
    """Looks up a submodule by its unique code."""
    with get_db_cursor(commit=False) as cur:
        cur.execute("""
            SELECT id, code, name, description, db_target 
            FROM submodules 
            WHERE code = %s OR LOWER(name) = LOWER(%s)
            LIMIT 1;
        """, (code, code))
        row = cur.fetchone()
        if row:
            return {
                "id": str(row[0]),
                "code": row[1],
                "name": row[2],
                "description": row[3] or "",
                "db_target": row[4],
            }
    return None


def add_or_update_submodule(code: str, name: str, description: str = "", db_target: Optional[str] = None) -> Dict[str, Any]:
    """Adds a new custom submodule or updates an existing one."""
    with get_db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO submodules (code, name, description, db_target)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (code) DO UPDATE
            SET name = EXCLUDED.name, description = EXCLUDED.description, db_target = EXCLUDED.db_target
            RETURNING id, code, name, description, db_target;
        """, (code, name, description, db_target))
        r = cur.fetchone()
        return {
            "id": str(r[0]),
            "code": r[1],
            "name": r[2],
            "description": r[3] or "",
            "db_target": r[4],
        }


# ==========================================
# Short-Lived Atomic Transaction Helpers
# ==========================================

def create_job(
    source_filename: str, 
    file_path: str, 
    model_name: str = "base",
    chunk_seconds: float = 30.0, 
    skip_correction: bool = False,
    submodule_code: str = "02_finance",
    file_size_bytes: int = 0,
    duration_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Inserts a new job in 'queued' state with target submodule routing."""
    job_id = str(uuid.uuid4())
    sub_info = get_submodule_info(submodule_code)
    sub_id = sub_info["id"] if sub_info else None
    sub_name = sub_info["name"] if sub_info else submodule_code

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                id, source_filename, file_path, model_name, chunk_seconds, skip_correction,
                submodule_id, submodule_code, submodule_name, file_size_bytes, duration_seconds,
                metadata, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued')
            RETURNING id;
            """,
            (
                job_id, source_filename, file_path, model_name, chunk_seconds, skip_correction,
                sub_id, submodule_code, sub_name, file_size_bytes, duration_seconds,
                json.dumps(metadata or {})
            )
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
            RETURNING id, source_filename, file_path, model_name, chunk_seconds, skip_correction,
                      submodule_code, submodule_name, submodule_id;
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
                "submodule_code": row[6],
                "submodule_name": row[7],
                "submodule_id": str(row[8]) if row[8] else None,
            }
    return None


def save_raw_chunks(
    job_id: str, 
    source_filename: str, 
    chunks: List[Dict[str, Any]],
    skip_correction: bool = False,
    submodule_code: str = "02_finance",
    submodule_name: str = "02. Finance",
    submodule_id: Optional[str] = None
):
    """
    Bulk inserts transcribed chunks tagged with target submodule and updates job state.
    """
    total = len(chunks)
    initial_status = 'corrected' if skip_correction else 'raw'
    next_job_status = 'embedding' if skip_correction else 'correcting'

    # Compute duration if available from chunks
    max_duration = max([c["end"] for c in chunks], default=0.0) if chunks else 0.0

    with get_db_cursor(commit=True) as cur:
        for idx, c in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO transcript_chunks (
                    job_id, source_file, chunk_index, start_time, end_time,
                    submodule_id, submodule_code, submodule_name, speaker,
                    text_raw, text_corrected, text, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::chunk_status);
                """,
                (
                    job_id,
                    source_filename,
                    idx,
                    c["start"],
                    c["end"],
                    submodule_id,
                    submodule_code,
                    submodule_name,
                    c.get("speaker"),
                    c["text"],
                    c["text"] if skip_correction else None,
                    c["text"],
                    initial_status
                )
            )

        cur.execute(
            """
            UPDATE jobs
            SET total_chunks = %s, 
                status = %s::job_status, 
                duration_seconds = COALESCE(duration_seconds, %s),
                updated_at = NOW()
            WHERE id = %s;
            """,
            (total, next_job_status, max_duration if max_duration > 0 else None, job_id)
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
                    failed_chunks = s.failed_count,
                    status = CASE 
                        WHEN (s.indexed_count + s.failed_count) = s.total AND s.failed_count > 0 THEN 'partial_failure'::job_status
                        WHEN s.indexed_count = s.total AND s.total > 0 THEN 'completed'::job_status
                        ELSE j.status
                    END,
                    completed_at = CASE
                        WHEN s.indexed_count = s.total AND s.total > 0 THEN NOW()
                        ELSE j.completed_at
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


def get_job_chunks(job_id: str) -> List[Dict[str, Any]]:
    """Returns all chunks with timestamps and text for PDF and document export."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT chunk_index, start_time, end_time, COALESCE(text_corrected, text_raw, text) AS text,
                   submodule_code, submodule_name, speaker
            FROM transcript_chunks
            WHERE job_id = %s
            ORDER BY chunk_index ASC;
            """,
            (job_id,)
        )
        rows = cur.fetchall()
        return [
            {
                "chunk_index": r[0],
                "start_time": float(r[1]),
                "end_time": float(r[2]),
                "text": r[3],
                "submodule_code": r[4],
                "submodule_name": r[5],
                "speaker": r[6],
            }
            for r in rows
        ]


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
                   processed_chunks, failed_chunks, error_message, summary, action_items,
                   submodule_code, submodule_name, duration_seconds, file_size_bytes,
                   created_at, updated_at, completed_at
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
        failed = row[6]
        error = row[7]
        summary = row[8]
        action_items = row[9]
        sub_code = row[10]
        sub_name = row[11]
        duration = float(row[12]) if row[12] is not None else None
        fsize = row[13]

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
            f"Job {job_id[:8]} routed to [{sub_name}] for {row[1]}",
            f"Current status: {status_str}",
        ]
        if total is not None:
            messages.append(f"Progress: {processed}/{total} chunks processed ({failed} failed)")
        if summary:
            messages.append("Executive synthesis & action items generated")
        if error:
            messages.append(f"Error: {error}")

        return {
            "id": str(row[0]),
            "source_filename": row[1],
            "model_name": row[2],
            "status": frontend_status,
            "raw_status": status_str,
            "submodule_code": sub_code,
            "submodule_name": sub_name,
            "duration_seconds": duration,
            "file_size_bytes": fsize,
            "total_chunks": total,
            "processed_chunks": processed,
            "failed_chunks": failed,
            "summary": summary,
            "action_items": action_items,
            "messages": messages,
            "result": {
                "chunks_saved": processed,
                "summary": summary,
                "action_items": action_items,
                "submodule_code": sub_code,
                "submodule_name": sub_name,
            },
            "error": error,
        }


def get_stats() -> Dict[str, Any]:
    """Returns database chunk and file counts, plus full submodule inventory."""
    with get_db_cursor(commit=False) as cur:
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'indexed' OR embedding IS NOT NULL) AS total_chunks,
                COUNT(DISTINCT COALESCE(source_file, 'unknown')) AS total_files
            FROM transcript_chunks;
        """)
        row = cur.fetchone()
        submodules = get_submodules()
        return {
            "total_chunks": row[0] if row else 0,
            "total_files": row[1] if row else 0,
            "submodules": submodules,
        }


def search_chunks(
    query_embedding: List[float], 
    top_k: int = 5, 
    submodule_code: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs cosine vector search (<=>) on indexed transcript chunks,
    optionally filtering by specific enterprise submodule.
    """
    vec_str = _vector_literal(query_embedding)
    
    # Check if submodule filtering applies
    filter_clause = ""
    params = [vec_str]
    if submodule_code and submodule_code.lower() not in ("all", "*", ""):
        filter_clause = "AND (submodule_code = %s OR LOWER(submodule_name) = LOWER(%s))"
        params.extend([submodule_code, submodule_code])
    params.extend([vec_str, top_k])

    with get_db_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT 
                COALESCE(source_file, 'unknown') AS source_file,
                start_time,
                end_time,
                COALESCE(text_corrected, text_raw, text) AS chunk_text,
                embedding <=> %s::vector AS distance,
                submodule_code,
                submodule_name,
                speaker
            FROM transcript_chunks
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
            """,
            tuple(params)
        )
        rows = cur.fetchall()
        return [
            {
                "source_file": r[0],
                "start_time": float(r[1]),
                "end_time": float(r[2]),
                "text": r[3],
                "distance": float(r[4]),
                "submodule_code": r[5],
                "submodule_name": r[6],
                "speaker": r[7],
            }
            for r in rows
        ]
