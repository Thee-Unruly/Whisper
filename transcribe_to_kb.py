"""
Enterprise Neural Audio Knowledge Base Ingestion Script.
Optimized for the provisioned PostgreSQL pgvector database (rag_poc @ 172.20.0.11:4357).

Features:
  - Video/Audio Ingestion (mp4, mkv, mov, mp3, wav, m4a)
  - Faster-Whisper (CTranslate2) or standard Whisper ASR
  - Dynamic LLM Transcript Correction (Groq / OpenRouter)
  - 1536-Dimensional Embeddings mapped to public.documents
  - Multi-Client (e.g. TMRC) & Enterprise Module Partitioning
  - Semantic Search CLI for testing retrieval

Usage:
  # Ingest single file:
  python transcribe_to_kb.py "Accounting Periods.mp4" --client TMRC --module "02. Finance"

  # Ingest entire folder:
  python transcribe_to_kb.py "C:/path/to/02. Finance" --client TMRC --module "02. Finance"

  # Search remote knowledge base:
  python transcribe_to_kb.py --search "accounting period configuration" --client TMRC --module "02. Finance"
"""

import argparse
import os
import sys
import glob
import time
import tempfile
import requests
import psycopg2
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Workaround for Numba caching on Windows / synced directories
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

# ---- TARGET DATABASE CONFIG (Provisioned Remote DB) ----
DEFAULT_REMOTE_HOST = os.environ.get("PGHOST", "172.20.0.11")
DEFAULT_REMOTE_PORT = int(os.environ.get("PGPORT", 4357))
DEFAULT_REMOTE_DB = os.environ.get("PGDATABASE", "rag_poc")
DEFAULT_REMOTE_USER = os.environ.get("PGUSER", "agile_ai_uno")
DEFAULT_REMOTE_PASS = os.environ.get("PGPASSWORD", "")

DB_CONFIG = {
    "dbname": os.environ.get("REMOTE_PGDATABASE") or DEFAULT_REMOTE_DB,
    "user": os.environ.get("REMOTE_PGUSER") or DEFAULT_REMOTE_USER,
    "password": os.environ.get("REMOTE_PGPASSWORD") or DEFAULT_REMOTE_PASS,
    "host": os.environ.get("REMOTE_PGHOST") or DEFAULT_REMOTE_HOST,
    "port": int(os.environ.get("REMOTE_PGPORT") or DEFAULT_REMOTE_PORT),
}

# Vector dimension required by public.documents in rag_poc
TARGET_VECTOR_DIM = 1536

# ---- LLM PROVIDER & CORRECTION CONFIG ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

CORRECTION_SYSTEM_PROMPT = (
    "You are an expert technical transcript editor. You are correcting a raw speech-to-text chunk "
    "from an enterprise software walkthrough/training session. "
    "Fix spelling of enterprise domain terms, grammar, punctuation, and sentence boundaries. "
    "Maintain high technical accuracy. Do NOT summarize. Do NOT add hallucinated details. "
    "Return ONLY the clean corrected text with no commentary or markdown wrappers."
)


# ==========================================
# 1. Audio / Video Transcription
# ==========================================

def transcribe(audio_path: str, model_name: str = "base") -> List[Dict[str, Any]]:
    """Transcribes media file using faster-whisper (preferred) or openai-whisper."""
    print(f"\n[ASR] Transcribing '{os.path.basename(audio_path)}' using model '{model_name}'...")
    t0 = time.monotonic()
    
    try:
        from faster_whisper import WhisperModel
        print("  Using faster-whisper (CTranslate2 engine)...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments_generator, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        
        segments = []
        for s in segments_generator:
            segments.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
            })
        duration = info.duration
    except ImportError:
        try:
            import whisper
            print("  Using standard openai-whisper...")
            model = whisper.load_model(model_name)
            result = model.transcribe(audio_path)
            segments = [
                {
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ]
            duration = segments[-1]["end"] if segments else 0.0
        except ImportError:
            print("Error: Please install faster-whisper or openai-whisper.")
            print("Run: pip install faster-whisper")
            sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"  ASR Complete in {elapsed:.1f}s. Extracted {len(segments)} raw segments (Total length: {duration:.1f}s).")
    return segments


def chunk_segments(segments: List[Dict[str, Any]], chunk_seconds: float = 30.0) -> List[Dict[str, Any]]:
    """Groups short segments into contextual windows of ~chunk_seconds."""
    chunks = []
    current_text = []
    chunk_start = None
    chunk_end = None

    for seg in segments:
        if chunk_start is None:
            chunk_start = seg["start"]
        if seg["text"]:
            current_text.append(seg["text"])
        chunk_end = seg["end"]

        if (chunk_end - chunk_start) >= chunk_seconds and current_text:
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": " ".join(current_text).strip(),
            })
            current_text = []
            chunk_start = None

    if current_text and chunk_start is not None:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": " ".join(current_text).strip(),
        })

    return chunks


# ==========================================
# 2. LLM Grammar & Domain Correction
# ==========================================

def correct_text_llm(text: str, prev_context: str = "") -> str:
    """Sends transcript chunk to Groq or OpenRouter for grammar and context correction."""
    if not text.strip():
        return text

    # Prefer Groq if key is available
    if GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        model = GROQ_MODEL
    elif OPENROUTER_API_KEY:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        model = OPENROUTER_MODEL
    else:
        return text

    user_content = text
    if prev_context:
        user_content = f"[PREVIOUS CHUNK CONTEXT: {prev_context[-200:]}]\n\n[CHUNK TO CORRECT]:\n{text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return text
    except Exception:
        return text


def correct_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Iteratively corrects chunks with context preservation."""
    provider = "Groq" if GROQ_API_KEY else ("OpenRouter" if OPENROUTER_API_KEY else "None")
    if provider == "None":
        print("  [Correction] No LLM API key detected. Skipping LLM correction step.")
        for c in chunks:
            c["text_raw"] = c["text"]
        return chunks

    print(f"\n[Correction] Correcting {len(chunks)} chunks via {provider}...")
    prev_ctx = ""
    for i, c in enumerate(chunks, 1):
        original = c["text"]
        c["text_raw"] = original
        corrected = correct_text_llm(original, prev_context=prev_ctx)
        c["text"] = corrected
        prev_ctx = corrected
        if i % 5 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] chunks corrected")
    return chunks


# ==========================================
# 3. 1536-Dimensional Embeddings
# ==========================================

_embed_model = None

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates 1536-dimensional embeddings matching public.documents schema.
    Uses FastEmbed / SentenceTransformers with exact zero-padding to 1536.
    Zero-padding strictly preserves normalized cosine distance for vector search.
    """
    global _embed_model
    if not texts:
        return []

    print(f"\n[Embedding] Generating {TARGET_VECTOR_DIM}-dim embeddings for {len(texts)} chunks...")
    try:
        from fastembed import TextEmbedding
        if _embed_model is None:
            _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        raw_embs = [e.tolist() for e in _embed_model.embed(texts)]
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer
            if _embed_model is None:
                _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            raw_embs = [e.tolist() for e in _embed_model.encode(texts, show_progress_bar=False)]
        except ImportError:
            print("Error: Please install fastembed or sentence-transformers.")
            print("Run: pip install fastembed")
            sys.exit(1)

    # Pad/Project to target 1536 dimensions
    padded = []
    for vec in raw_embs:
        if len(vec) < TARGET_VECTOR_DIM:
            vec = vec + [0.0] * (TARGET_VECTOR_DIM - len(vec))
        elif len(vec) > TARGET_VECTOR_DIM:
            vec = vec[:TARGET_VECTOR_DIM]
        padded.append(vec)

    return padded


# ==========================================
# 4. Database Ingestion (public.documents)
# ==========================================

def get_db_connection():
    """Connects to target PostgreSQL database."""
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        connect_timeout=10,
        sslmode="prefer",
    )


def save_to_documents_table(
    chunks: List[Dict[str, Any]],
    source_filename: str,
    client: str = "TMRC",
    module: str = "02. Finance"
) -> int:
    """
    Saves chunks directly into the provisioned public.documents table.
    Encodes client, module, filename, and timestamps into the source column.
    """
    if not chunks:
        print("No chunks to save.")
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    print(f"\n[Database] Connecting to PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']})...")
    conn = get_db_connection()
    cur = conn.cursor()

    inserted_count = 0
    for chunk, emb in zip(chunks, embeddings):
        # Format structured metadata string into 'source'
        source_meta = (
            f"client:{client} | module:{module} | file:{source_filename} | "
            f"time:[{chunk['start']:.1f}s - {chunk['end']:.1f}s]"
        )
        vec_literal = "[" + ",".join(str(x) for x in emb) + "]"

        cur.execute("""
            INSERT INTO public.documents (source, chunk_text, embedding)
            VALUES (%s, %s, %s::vector)
            RETURNING id;
        """, (source_meta, chunk["text"], vec_literal))

        doc_id = cur.fetchone()[0]
        inserted_count += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"[Database] Successfully saved {inserted_count} chunks into 'public.documents' (Client: {client}, Module: {module}).")
    return inserted_count


# ==========================================
# 5. Semantic Search Tester
# ==========================================

def search_documents(query: str, client: Optional[str] = None, module: Optional[str] = None, top_k: int = 5):
    """Executes cosine distance semantic search directly on public.documents."""
    print(f"\n[Search] Searching for: '{query}' (Client={client or 'Any'}, Module={module or 'Any'})...")
    query_emb = get_embeddings([query])[0]
    q_literal = "[" + ",".join(str(x) for x in query_emb) + "]"

    conn = get_db_connection()
    cur = conn.cursor()

    where_clauses = []
    params = [q_literal]

    if client:
        where_clauses.append(f"source LIKE %s")
        params.append(f"%client:{client}%")
    if module:
        where_clauses.append(f"source LIKE %s")
        params.append(f"%module:{module}%")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(top_k)

    sql = f"""
        SELECT id, source, chunk_text, (embedding <=> %s::vector) AS distance
        FROM public.documents
        {where_sql}
        ORDER BY distance ASC
        LIMIT %s;
    """

    cur.execute(sql, tuple(params))
    results = cur.fetchall()

    print(f"\nFound {len(results)} matches:")
    for idx, r in enumerate(results, 1):
        doc_id, source, text, dist = r
        score = max(0.0, 1.0 - dist) * 100
        print(f"\n[{idx}] Match Score: {score:.1f}% (Distance: {dist:.4f}) | ID: {doc_id}")
        print(f"    Source: {source}")
        print(f"    Text:   {text}")

    cur.close()
    conn.close()


# ==========================================
# 6. Main Runner
# ==========================================

def process_file(file_path: str, client: str, module: str, model_name: str, chunk_seconds: float, skip_correction: bool):
    """Processes an individual video/audio file end-to-end."""
    filename = os.path.basename(file_path)
    print(f"\n=======================================================")
    print(f" Processing: {filename}")
    print(f" Client:     {client}")
    print(f" Module:     {module}")
    print(f"=======================================================")

    segments = transcribe(file_path, model_name=model_name)
    if not segments:
        print(f"Warning: No speech detected in {filename}")
        return

    chunks = chunk_segments(segments, chunk_seconds=chunk_seconds)
    if not skip_correction:
        chunks = correct_chunks(chunks)
    else:
        for c in chunks:
            c["text_raw"] = c["text"]

    save_to_documents_table(chunks, source_filename=filename, client=client, module=module)


def main():
    parser = argparse.ArgumentParser(description="Transcribe enterprise media into PostgreSQL (rag_poc.documents).")
    parser.add_argument("input", nargs="?", help="Path to video/audio file or folder containing media files")
    parser.add_argument("--client", default="TMRC", help="Client name (e.g., TMRC)")
    parser.add_argument("--module", default="02. Finance", help="Module name (e.g., 01. Credit, 02. Finance, 04. Procurement)")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Target chunk duration in seconds")
    parser.add_argument("--skip-correction", action="store_true", help="Skip LLM grammar correction")
    parser.add_argument("--search", type=str, default=None, help="Query string to perform semantic search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results to retrieve")
    args = parser.parse_args()

    # Search Mode
    if args.search:
        search_documents(args.search, client=args.client, module=args.module, top_k=args.top_k)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Ingestion Mode
    valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".mp3", ".wav", ".m4a", ".aac", ".webm"}

    if os.path.isdir(args.input):
        print(f"Scanning directory: {args.input}")
        media_files = [
            f for f in glob.glob(os.path.join(args.input, "**", "*.*"), recursive=True)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if not media_files:
            print(f"No media files found in {args.input}")
            return
        print(f"Found {len(media_files)} media files to process.")
        for idx, mf in enumerate(media_files, 1):
            print(f"\n--- [{idx}/{len(media_files)}] ---")
            process_file(mf, client=args.client, module=args.module, model_name=args.model,
                         chunk_seconds=args.chunk_seconds, skip_correction=args.skip_correction)
    elif os.path.isfile(args.input):
        process_file(args.input, client=args.client, module=args.module, model_name=args.model,
                     chunk_seconds=args.chunk_seconds, skip_correction=args.skip_correction)
    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()