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
    "You are an expert transcript editor and thought-structuring assistant. "
    "Your objective is to transform raw spoken audio chunks into clean, coherent, and well-structured written text.\n\n"
    "Rules:\n"
    "1. Remove verbal fillers (e.g., 'um', 'uh', 'like', 'you know', 'sort of', stuttering, false starts).\n"
    "2. Correct speech-to-text phonetic mis-transcriptions, grammar, punctuation, and capitalization.\n"
    "3. Align conversational wandering into concise, logically coherent sentences.\n"
    "4. Maintain absolute factual fidelity: do NOT hallucinate facts, omit technical terms, or alter speaker intent.\n"
    "5. Use the provided PREVIOUS CONTEXT (if available) to ensure seamless continuity across sentence and thought boundaries.\n"
    "6. Return ONLY the final polished text with no introduction, explanations, or quotes."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are an executive synthesis assistant. Analyze the full clean meeting/audio transcript provided "
    "and generate two structured sections:\n\n"
    "## Executive Summary\n"
    "A concise 2-3 paragraph synthesis summarizing the core topics, perspectives, and main outcomes discussed.\n\n"
    "## Key Decisions & Action Items\n"
    "A clear bulleted list of decisions made, identified next steps, and action items with owners if mentioned.\n\n"
    "Format in clean markdown. Do not include introductory conversational filler."
)

_whisper_models = {}
_embedding_model = None


def get_whisper_model(model_name: str):
    """Loads faster-whisper WhisperModel lazily with optimal quantization."""
    from faster_whisper import WhisperModel
    import torch
    if model_name not in _whisper_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading faster-whisper model '{model_name}' on device='{device}' (compute_type='{compute_type}')...")
        _whisper_models[model_name] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.path.join(tempfile.gettempdir(), "whisper_cache"),
        )
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
    """Stage 1: Transcribe audio using faster-whisper with Silero VAD filter."""
    model = get_whisper_model(model_name)
    segments_generator, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    
    segments = []
    for seg in segments_generator:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
    return segments


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
    prev_context: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3
) -> str:
    """
    Stage 2: Async Groq LLM correction and thought-structuring pass
    with context-aware stitching and backoff handling.
    """
    key = api_key or os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return text  # Fallback to raw text if no key provided

    model_name = model or os.environ.get("GROQ_MODEL") or GROQ_MODEL

    user_prompt = text
    if prev_context:
        user_prompt = f"PREVIOUS CHUNK CONTEXT (for continuity only):\n\"{prev_context}\"\n\nCURRENT SPOKEN RAW CHUNK TO EDIT AND STRUCTURE:\n\"{text}\""

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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

    return text


async def async_generate_summary_and_action_items(
    client,
    full_transcript: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, str]:
    """
    Job-Level Synthesis: Generates Executive Summary and Action Items from the full transcript.
    """
    if not full_transcript.strip():
        return {"summary": "", "action_items": ""}

    key = api_key or os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return {"summary": "", "action_items": ""}

    model_name = model or os.environ.get("GROQ_MODEL") or GROQ_MODEL

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Full Clean Transcript:\n\n{full_transcript[:25000]}"},
        ],
        "temperature": 0.3,
    }

    for attempt in range(max_retries):
        try:
            resp = await client.post(GROQ_URL, json=payload, headers=headers, timeout=45.0)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                
                # Split into Executive Summary and Action Items sections if formatted
                summary_part = content
                action_part = ""
                if "## Key Decisions" in content or "## Action Items" in content:
                    split_header = "## Key Decisions" if "## Key Decisions" in content else "## Action Items"
                    parts = content.split(split_header, 1)
                    summary_part = parts[0].replace("## Executive Summary", "").strip()
                    action_part = f"{split_header}\n{parts[1]}".strip()
                
                return {
                    "summary": summary_part,
                    "action_items": action_part,
                }
            elif resp.status_code == 429:
                await asyncio.sleep(2.0 * (attempt + 1))
        except Exception as e:
            logger.warning(f"Failed to generate summary on attempt {attempt + 1}: {e}")
            await asyncio.sleep(1.0)

    return {"summary": "", "action_items": ""}


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