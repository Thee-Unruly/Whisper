"""
Pipeline: video/audio -> Whisper transcript -> chunked segments ->
local embeddings (sentence-transformers) -> Postgres (pgvector) KB.

Usage:
    python transcribe_to_kb.py path/to/video.mp4
    python transcribe_to_kb.py path/to/video.mp4 --model base --chunk-seconds 30

Requires:
    pip install openai-whisper sentence-transformers psycopg2-binary requests
    ffmpeg installed and on PATH
    Postgres with the pgvector extension enabled:
        CREATE EXTENSION IF NOT EXISTS vector;
    An OpenRouter API key set as an environment variable:
        export OPENROUTER_API_KEY="sk-or-..."
"""

import argparse
import os
import sys

# ---- CONFIG: edit these for your DB ----
DB_CONFIG = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": 5432,
}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, 384-dim, good default
EMBEDDING_DIM = 384

# OpenRouter config (used for transcript correction step)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # set this in your env
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"  # pick any model OpenRouter supports
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CORRECTION_SYSTEM_PROMPT = (
    "You are correcting a raw speech-to-text transcript chunk. "
    "Fix grammar, punctuation, obvious mis-transcribed words, and sentence "
    "boundaries by reasoning about what was most likely actually said. "
    "Do NOT change the meaning, do NOT add information, do NOT summarize. "
    "Return ONLY the corrected text, with no preamble, labels, or commentary."
)


def transcribe(audio_path: str, model_name: str = "base"):
    try:
        import whisper
    except ImportError:
        print("Install whisper first: pip install -U openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing '{audio_path}' (Whisper reads mp4 directly via ffmpeg)...")
    result = model.transcribe(audio_path)
    return result["segments"]  # list of {start, end, text, ...}


def chunk_segments(segments, chunk_seconds: float = 30.0):
    """
    Group raw Whisper segments (a few seconds each) into larger chunks
    of ~chunk_seconds, so each KB entry has enough context to be useful
    for retrieval instead of being a fragment like 'and then we' .
    """
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
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": " ".join(current_text).strip(),
            })
            current_text = []
            chunk_start = None

    # leftover
    if current_text:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": " ".join(current_text).strip(),
        })

    return chunks


def correct_chunk_text(text: str) -> str:
    """
    Send one transcript chunk to an LLM via OpenRouter to fix grammar,
    punctuation, and likely mis-transcribed words through reasoning,
    without changing meaning or adding/removing information.
    """
    import requests

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    if response.status_code != 200:
        print(f"OpenRouter error ({response.status_code}): {response.text}")
        print("Falling back to original (uncorrected) text for this chunk.")
        return text

    data = response.json()
    corrected = data["choices"][0]["message"]["content"].strip()
    return corrected


def correct_chunks(chunks):
    print(f"Correcting {len(chunks)} chunks via OpenRouter ({OPENROUTER_MODEL})...")
    for i, chunk in enumerate(chunks, 1):
        original = chunk["text"]
        chunk["text_raw"] = original  # keep the original for reference/debugging
        chunk["text"] = correct_chunk_text(original)
        print(f"  [{i}/{len(chunks)}] corrected")
    return chunks


def embed_chunks(chunks):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers first: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def ensure_table(cur):
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


def save_to_postgres(chunks, source_file: str):
    try:
        import psycopg2
    except ImportError:
        print("Install psycopg2 first: pip install psycopg2-binary")
        sys.exit(1)

    print("Connecting to Postgres...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
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
                chunk["embedding"],
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Saved {len(chunks)} chunks to Postgres (table: transcript_chunks).")


def main():
    parser = argparse.ArgumentParser(description="Transcribe media into a searchable Postgres KB.")
    parser.add_argument("input", help="Path to audio/video file (mp4, mp3, wav, etc.)")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Target chunk length in seconds")
    parser.add_argument("--skip-correction", action="store_true", help="Skip the LLM correction step (use raw Whisper text)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    segments = transcribe(args.input, args.model)
    chunks = chunk_segments(segments, args.chunk_seconds)
    if not args.skip_correction:
        chunks = correct_chunks(chunks)
    chunks = embed_chunks(chunks)
    save_to_postgres(chunks, source_file=os.path.basename(args.input))


if __name__ == "__main__":
    main()