"""
Core pipeline logic: transcribe -> chunk -> Groq correct -> batch embed -> search.
Decoupled functions for Level 1 stage workers and API queries.
"""

import os
import sys
import tempfile
import asyncio
import logging
from typing import List, Dict, Any, Optional

import db

logger = logging.getLogger("signal.pipeline")

# Windows + OneDrive-synced project folders cause numba (a Whisper dependency)
# to fail with "[Errno 22] Invalid argument" when it tries to write its JIT
# compile cache. Redirect the cache to a local, non-synced temp folder.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "numba_cache"),
)

# ---- EMBEDDING CONFIG ----
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---- GROQ LLM CONFIG ----
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

_whisper_models = {}
_embedding_model = None


def get_whisper_model(model_name: str):
    """Loads Whisper model lazily and caches across calls."""
    import whisper
    if model_name not in _whisper_models:
        logger.info(f"Loading Whisper model '{model_name}'...")
        _whisper_models[model_name] = whisper.load_model(model_name)
    return _whisper_models[model_name]


def get_embedding_model():
    """Loads SentenceTransformer model lazily and caches across calls."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ==========================================
# Pipeline Stage Functions
# ==========================================

def transcribe_file(audio_path: str, model_name: str = "base") -> List[Dict[str, Any]]:
    """Stage 1: Transcribe audio using Whisper."""
    model = get_whisper_model(model_name)
    result = model.transcribe(audio_path)
    return result.get("segments", [])


def chunk_segments(segments: List[Dict[str, Any]], chunk_seconds: float = 30.0) -> List[Dict[str, Any]]:
    """Groups Whisper segments into timestamped chunks."""
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
                "text": " ".join(current_text).strip()
            })
            current_text = []
            chunk_start = None

    if current_text:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": " ".join(current_text).strip()
        })

    return chunks


async def async_correct_text(
    client,
    text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3
) -> str:
    """
    Stage 2: Async Groq LLM correction pass with rate-limit & backoff header handling.
    """
    key = api_key or os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return text  # Fallback to raw text if no key provided

    model_name = model or os.environ.get("GROQ_MODEL") or GROQ_MODEL

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }

    for attempt in range(max_retries):
        try:
            resp = await client.post(GROQ_URL, json=payload, headers=headers, timeout=25.0)

            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            elif resp.status_code == 429:
                # Dynamic backoff obeying Groq headers
                retry_after_str = resp.headers.get("retry-after")
                sleep_duration = float(retry_after_str) if retry_after_str else (1.5 * (2 ** attempt) + 0.2)
                logger.warning(f"Groq rate limit hit (429). Backing off for {sleep_duration:.2f}s...")
                await asyncio.sleep(sleep_duration)
            else:
                logger.warning(f"Groq API error {resp.status_code}: {resp.text}")
                await asyncio.sleep(1.0)
        except Exception as e:
            logger.warning(f"Exception during Groq request attempt {attempt + 1}: {e}")
            await asyncio.sleep(1.0)

    return text  # Fallback to raw text on persistent failure


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Stage 3: Vectorized batch embedding via all-MiniLM-L6-v2."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts)
    return [emb.tolist() for emb in embeddings]


def search_kb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embeds query and queries PostgreSQL using Cosine Distance (<=>)."""
    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()
    return db.search_chunks(query_embedding, top_k=top_k)