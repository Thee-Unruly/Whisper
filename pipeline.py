"""
Core pipeline logic: transcribe -> chunk -> correct -> embed -> store/search.
Importable by the FastAPI app (no CLI/argparse here).
Uses Groq LLM for fast transcript correction.
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Windows + OneDrive-synced project folders cause numba (a Whisper dependency)
# to fail with "[Errno 22] Invalid argument" when it tries to write its JIT
# compile cache, because OneDrive's Files On-Demand layer doesn't support the
# file operations numba needs. Redirect the cache to a local, non-synced temp
# folder to avoid this. Must be set before whisper (and numba) are imported.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "numba_cache"),
)

# ---- DATABASE CONFIG ----
DB_CONFIG = {
    "dbname": os.environ.get("PGDATABASE", "transcripts_agile"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---- LLM CONFIG (Groq) ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CORRECTION_SYSTEM_PROMPT = (
    "You are correcting a raw speech-to-text transcript chunk. "
    "Fix grammar, punctuation, obvious mis-transcribed words, and sentence "
    "boundaries by reasoning about what was most likely actually said. "
    "Do NOT change the meaning, do NOT add information, do NOT summarize. "
    "Return ONLY the corrected text, with no preamble, labels, or commentary."
)

# Loaded lazily and cached across requests so we don't reload models every call
_whisper_models = {}
_embedding_model = None
_db_pool = None


def get_llm_config():
    """Returns active Groq LLM config if GROQ_API_KEY is configured."""
    groq_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if groq_key:
        return {
            "provider": "Groq",
            "url": GROQ_URL,
            "key": groq_key,
            "model": os.environ.get("GROQ_MODEL", GROQ_MODEL),
        }
    return None


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        import psycopg2.pool
        _db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
    return _db_pool


@contextmanager
def get_db_cursor(commit=False):
    pool = get_db_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def get_whisper_model(model_name: str):
    import whisper
    if model_name not in _whisper_models:
        _whisper_models[model_name] = whisper.load_model(model_name)
    return _whisper_models[model_name]


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def transcribe(audio_path: str, model_name: str = "base", progress_cb=None):
    model = get_whisper_model(model_name)
    if progress_cb:
        progress_cb("Transcribing audio with Whisper...")
    result = model.transcribe(audio_path)
    return result["segments"]


def chunk_segments(segments, chunk_seconds: float = 30.0):
    chunks = []
    current_text = []
    chunk_start = None
    chunk_end = None

    for seg in segments:
        if chunk_start is None:
            chunk_start = seg["start"]
        current_text.append(seg["text"].strip())
        chunk_end = seg["end"]

        if chunk_end - chunk_start >= chunk_seconds:
            chunks.append({"start": chunk_start, "end": chunk_end, "text": " ".join(current_text).strip()})
            current_text = []
            chunk_start = None

    if current_text:
        chunks.append({"start": chunk_start, "end": chunk_end, "text": " ".join(current_text).strip()})

    return chunks


def correct_chunk_text(text: str) -> str:
    import requests
    llm = get_llm_config()
    if not llm:
        return text

    try:
        response = requests.post(
            llm["url"],
            headers={
                "Authorization": f"Bearer {llm['key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": llm["model"],
                "messages": [
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return text  # fall back to original on API error

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return text


def correct_chunks(chunks, progress_cb=None, max_workers=6):
    if not chunks:
        return chunks

    llm = get_llm_config()
    if not llm:
        if progress_cb:
            progress_cb("Skipping LLM correction: No GROQ_API_KEY found.")
        for chunk in chunks:
            chunk["text_raw"] = chunk["text"]
        return chunks

    total = len(chunks)
    if progress_cb:
        progress_cb(f"Correcting {total} chunks via {llm['provider']} ({llm['model']})...")

    def _correct_single(item):
        idx, chunk = item
        original = chunk["text"]
        chunk["text_raw"] = original
        chunk["text"] = correct_chunk_text(original)
        return idx

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_correct_single, (i, c)) for i, c in enumerate(chunks, 1)]
        for f in as_completed(futures):
            completed += 1
            if progress_cb:
                progress_cb(f"[{completed}/{total}] Corrected via {llm['provider']}")

    return chunks


def embed_chunks(chunks, progress_cb=None):
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    if progress_cb:
        progress_cb(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def _vector_literal(embedding):
    return "[" + ",".join(str(x) for x in embedding) + "]"


def ensure_table(cur):
    # Enable pgvector extension if not already enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create the chunks table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS transcript_chunks (
            id SERIAL PRIMARY KEY,
            source_file TEXT NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            text TEXT NOT NULL,
            text_raw TEXT,
            embedding VECTOR({EMBEDDING_DIM})
        );
    """)
    
    # Create HNSW index for ultra-fast approximate nearest neighbor cosine vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS transcript_chunks_embedding_hnsw_idx 
        ON transcript_chunks USING hnsw (embedding vector_cosine_ops);
    """)


def save_to_postgres(chunks, source_file: str):
    with get_db_cursor(commit=True) as cur:
        ensure_table(cur)

        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO transcript_chunks (source_file, start_time, end_time, text, text_raw, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    source_file,
                    chunk["start"],
                    chunk["end"],
                    chunk["text"],
                    chunk.get("text_raw"),
                    _vector_literal(chunk["embedding"]),
                ),
            )


def process_file(file_path: str, source_name: str, model_name="base", chunk_seconds=30.0,
                  skip_correction=False, progress_cb=None):
    """
    Full pipeline: transcribe -> chunk -> (optionally correct) -> embed -> store.
    progress_cb(str) is called with human-readable status updates for job tracking.
    """
    segments = transcribe(file_path, model_name, progress_cb)
    chunks = chunk_segments(segments, chunk_seconds)

    if not skip_correction:
        chunks = correct_chunks(chunks, progress_cb)
    else:
        for chunk in chunks:
            chunk["text_raw"] = chunk["text"]

    chunks = embed_chunks(chunks, progress_cb)

    if progress_cb:
        progress_cb("Saving to database...")
    save_to_postgres(chunks, source_file=source_name)

    if progress_cb:
        progress_cb("Done")

    return len(chunks)


def search_kb(query: str, top_k: int = 5):
    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()

    with get_db_cursor(commit=False) as cur:
        # Ensure table/extension exist before querying
        ensure_table(cur)
        cur.execute(
            """
            SELECT source_file, start_time, end_time, text,
                   embedding <=> %s::vector AS distance
            FROM transcript_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (_vector_literal(query_embedding), _vector_literal(query_embedding), top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "source_file": r[0],
            "start_time": r[1],
            "end_time": r[2],
            "text": r[3],
            "distance": float(r[4]),
        }
        for r in rows
    ]