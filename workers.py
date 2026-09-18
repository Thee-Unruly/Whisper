"""
Background stage worker loops for Level 1 architecture.
Decoupled concurrent execution of:
  - Stage 1: Whisper ASR (GPU / CPU executor)
  - Stage 2: Groq LLM correction (Non-blocking async HTTP with Token Bucket)
  - Stage 3: MiniLM batch embedding (Vectorized CPU SIMD)
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional
import httpx

import db
import pipeline

logger = logging.getLogger("signal.workers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


class AsyncTokenBucket:
    """Non-blocking token bucket rate limiter: computes delay inside lock, sleeps outside lock."""
    def __init__(self, rate_per_minute: float = 30.0):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.capacity = float(rate_per_minute)
        self.tokens = float(rate_per_minute)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        wait_time = 0.0
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens < 1.0:
                deficit = 1.0 - self.tokens
                wait_time = deficit / self.rate
                self.tokens = 0.0
                self.last_update += wait_time
            else:
                self.tokens -= 1.0

        if wait_time > 0:
            await asyncio.sleep(wait_time)


# ==========================================
# Stage 1: Whisper ASR Worker Loop
# ==========================================

async def asr_stage_worker(poll_interval: float = 1.0):
    """Claims queued jobs, runs Whisper ASR in an executor, persists raw chunks."""
    logger.info("Starting Stage 1 ASR Worker Loop...")
    loop = asyncio.get_running_loop()

    while True:
        try:
            job = db.claim_next_job()
            if not job:
                await asyncio.sleep(poll_interval)
                continue

            job_id = job["id"]
            file_path = job["file_path"]
            source_filename = job["source_filename"]
            model_name = job["model_name"]
            chunk_seconds = job["chunk_seconds"]
            skip_correction = job["skip_correction"]
            submodule_code = job.get("submodule_code", "02_finance")
            submodule_name = job.get("submodule_name", "02. Finance")
            submodule_id = job.get("submodule_id")

            logger.info(f"[Stage 1 ASR] Processing job {job_id[:8]} ({source_filename}) for [{submodule_name}]...")

            try:
                # Heavy Whisper inference outside any DB transaction
                segments = await loop.run_in_executor(
                    None,
                    pipeline.transcribe_file,
                    file_path,
                    model_name
                )

                chunks = pipeline.chunk_segments(segments, chunk_seconds)
                logger.info(f"[Stage 1 ASR] Generated {len(chunks)} chunks for job {job_id[:8]}. Persisting...")

                # Short transaction to bulk insert chunks with submodule metadata
                db.save_raw_chunks(
                    job_id, 
                    source_filename, 
                    chunks, 
                    skip_correction=skip_correction,
                    submodule_code=submodule_code,
                    submodule_name=submodule_name,
                    submodule_id=submodule_id
                )

            except Exception as exc:
                logger.error(f"[Stage 1 ASR] Job {job_id[:8]} failed: {exc}", exc_info=True)
                db.fail_job(job_id, str(exc))
            finally:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error(f"[Stage 1 ASR Loop Error]: {e}", exc_info=True)
            await asyncio.sleep(poll_interval)


# ==========================================
# Stage 2: Groq LLM Correction Worker Loop
# ==========================================

async def groq_stage_worker(rate_limit_per_minute: float = 25.0, poll_interval: float = 0.5):
    """Claims raw chunks and fans out async Groq requests with rate limiting."""
    logger.info("Starting Stage 2 Groq Correction Worker Loop...")
    limiter = AsyncTokenBucket(rate_per_minute=rate_limit_per_minute)

    async with httpx.AsyncClient(timeout=35.0) as client:
        while True:
            try:
                raw_chunks = db.claim_raw_chunks(batch_size=4)
                if not raw_chunks:
                    await asyncio.sleep(poll_interval)
                    continue

                logger.info(f"[Stage 2 Groq] Correcting batch of {len(raw_chunks)} chunks...")

                async def _process_chunk(chunk):
                    chunk_id = chunk["id"]
                    text_raw = chunk["text_raw"]
                    prev_text = chunk.get("prev_text")
                    try:
                        await limiter.acquire()
                        corrected = await pipeline.async_correct_text(
                            client, 
                            text=text_raw, 
                            prev_context=prev_text
                        )
                        db.save_corrected_chunk(chunk_id, corrected)
                    except Exception as exc:
                        logger.warning(f"[Stage 2 Groq] Failed chunk {chunk_id[:8]}: {exc}")
                        db.fail_chunk(chunk_id, str(exc))

                # Fan out concurrently within rate limit
                await asyncio.gather(*[_process_chunk(c) for c in raw_chunks])

            except Exception as e:
                logger.error(f"[Stage 2 Groq Loop Error]: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)


# ==========================================
# Stage 3: MiniLM Batch Embedding Worker Loop
# ==========================================

async def embedding_stage_worker(batch_size: int = 32, poll_interval: float = 0.5):
    """Claims corrected chunks, micro-batches into MiniLM, bulk saves embeddings & updates job state."""
    logger.info("Starting Stage 3 Embedding Worker Loop...")
    loop = asyncio.get_running_loop()

    while True:
        try:
            chunks = db.claim_corrected_chunks(batch_size=batch_size)
            if not chunks:
                await asyncio.sleep(poll_interval)
                continue

            chunk_ids = [c["id"] for c in chunks]
            job_ids = [c["job_id"] for c in chunks]
            texts = [c["text"] for c in chunks]

            logger.info(f"[Stage 3 Embedding] Encoding batch of {len(texts)} chunks...")

            # Vectorized inference in executor
            embeddings = await loop.run_in_executor(
                None,
                pipeline.embed_texts,
                texts
            )

            # Atomic bulk vector update & job status CTE
            db.save_chunk_embeddings_and_update_job(chunk_ids, embeddings, job_ids)
            logger.info(f"[Stage 3 Embedding] Indexed {len(texts)} chunks successfully.")

        except Exception as e:
            logger.error(f"[Stage 3 Embedding Loop Error]: {e}", exc_info=True)
            await asyncio.sleep(poll_interval)


# ==========================================
# Stage 4: Executive Synthesis & Action Items Worker Loop
# ==========================================

async def synthesis_stage_worker(poll_interval: float = 1.0):
    """
    Stage 4 (Synthesis): Automatically generates Executive Summary and Action Items
    for completed jobs once all chunks have been indexed.
    """
    logger.info("Starting Stage 4 Executive Synthesis Worker Loop...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                if not db.table_exists("jobs"):
                    await asyncio.sleep(poll_interval * 5)
                    continue

                # Find a completed job that hasn't had synthesis generated yet
                job_id = None
                with db.get_db_cursor(commit=False) as cur:
                    cur.execute("""
                        SELECT id FROM jobs 
                        WHERE status = 'completed' AND summary IS NULL 
                        LIMIT 1;
                    """)
                    row = cur.fetchone()
                    if row:
                        job_id = str(row[0])

                if not job_id:
                    await asyncio.sleep(poll_interval)
                    continue

                full_transcript = db.get_full_job_transcript(job_id)
                if full_transcript.strip():
                    logger.info(f"[Stage 4 Synthesis] Generating Executive Summary for job {job_id[:8]}...")
                    synth = await pipeline.async_generate_summary_and_action_items(client, full_transcript)
                    db.save_job_summary(job_id, synth["summary"], synth["action_items"])
                    logger.info(f"[Stage 4 Synthesis] Summary & Action Items saved for job {job_id[:8]}.")
                else:
                    db.save_job_summary(job_id, "No transcript available.", "")

            except Exception as e:
                logger.error(f"[Stage 4 Synthesis Loop Error]: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)


# ==========================================
# Master Runner & Standalone Entrypoint
# ==========================================

async def run_all_workers():
    """Runs all 4 stage worker loops concurrently inside the current process."""
    db.init_db()
    await asyncio.gather(
        asr_stage_worker(),
        groq_stage_worker(),
        embedding_stage_worker(),
        synthesis_stage_worker(),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Signal Pipeline Stage Workers")
    parser.add_argument("--stage", choices=["all", "asr", "groq", "embed", "synthesis"], default="all",
                        help="Select which stage worker loop to run")
    args = parser.parse_args()

    db.init_db()
    if args.stage == "asr":
        asyncio.run(asr_stage_worker())
    elif args.stage == "groq":
        asyncio.run(groq_stage_worker())
    elif args.stage == "embed":
        asyncio.run(embedding_stage_worker())
    elif args.stage == "synthesis":
        asyncio.run(synthesis_stage_worker())
    else:
        asyncio.run(run_all_workers())
