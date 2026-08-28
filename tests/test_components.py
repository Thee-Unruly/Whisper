"""
Unit and integration tests for Signal Level 1 architecture.
Tests rate limiting, chunking, embeddings, and database atomic helpers.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import db
import pipeline
import workers
from workers import AsyncTokenBucket


class TestAsyncTokenBucket(unittest.TestCase):
    def test_non_blocking_schedule(self):
        async def run_limiter_test():
            # 60 tokens per minute = 1 token per second
            limiter = AsyncTokenBucket(rate_per_minute=60.0)
            
            # First token should be instant
            t0 = time.monotonic()
            await limiter.acquire()
            t1 = time.monotonic()
            self.assertLess(t1 - t0, 0.1)

            # Consume capacity quickly
            limiter.tokens = 0.0
            t_start = time.monotonic()
            await limiter.acquire()
            t_end = time.monotonic()
            # Should wait approximately 1.0 second (0.8s - 1.3s)
            self.assertGreaterEqual(t_end - t_start, 0.8)

        asyncio.run(run_limiter_test())

    def test_concurrent_fanout(self):
        async def run_concurrent():
            limiter = AsyncTokenBucket(rate_per_minute=600.0) # 10 tokens/sec
            results = []

            async def worker(wid):
                await limiter.acquire()
                results.append(wid)

            # Spawn 10 workers concurrently
            tasks = [worker(i) for i in range(10)]
            await asyncio.gather(*tasks)
            self.assertEqual(len(results), 10)

        asyncio.run(run_concurrent())


class TestChunkingLogic(unittest.TestCase):
    def test_chunk_segments_exact_interval(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Hello world"},
            {"start": 10.0, "end": 25.0, "text": "this is a test"},
            {"start": 25.0, "end": 35.0, "text": "of transcript chunking"},
            {"start": 35.0, "end": 50.0, "text": "and final segment"},
        ]
        chunks = pipeline.chunk_segments(segments, chunk_seconds=30.0)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["text"], "Hello world this is a test of transcript chunking")

    def test_empty_segments(self):
        chunks = pipeline.chunk_segments([], chunk_seconds=30.0)
        self.assertEqual(chunks, [])


class TestEmbeddingLogic(unittest.TestCase):
    def test_embed_texts_output_dimensions(self):
        texts = ["This is a test sentence for vector search.", "Another sample chunk."]
        embeddings = pipeline.embed_texts(texts)
        self.assertEqual(len(embeddings), 2)
        # Verify 384 dimensions for all-MiniLM-L6-v2
        self.assertEqual(len(embeddings[0]), 384)
        self.assertEqual(len(embeddings[1]), 384)
        self.assertIsInstance(embeddings[0][0], float)


class TestGroqCorrection(unittest.TestCase):
    def test_async_correct_text_fallback_on_error(self):
        async def run_fallback():
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network down"))
            
            raw_text = "raw transcript with typo"
            res = await pipeline.async_correct_text(
                mock_client,
                raw_text,
                api_key="gsk_test",
                max_retries=1
            )
            # Should safely fallback to original text
            self.assertEqual(res, raw_text)

        asyncio.run(run_fallback())

    def test_async_correct_text_context_aware_stitching(self):
        async def run_context():
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "We discussed the budget."}}]
            }
            mock_client.post = AsyncMock(return_value=mock_resp)

            prev_ctx = "In the previous meeting"
            res = await pipeline.async_correct_text(
                mock_client,
                text="we discussed the budget",
                prev_context=prev_ctx,
                api_key="gsk_test"
            )
            self.assertEqual(res, "We discussed the budget.")
            
            # Verify payload included previous context
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            user_msg = payload["messages"][1]["content"]
            self.assertIn("PREVIOUS CHUNK CONTEXT", user_msg)
            self.assertIn(prev_ctx, user_msg)

        asyncio.run(run_context())

    def test_summary_synthesis_parsing(self):
        async def run_synth():
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "## Executive Summary\nThe team discussed Q3 goals.\n\n## Key Decisions & Action Items\n- John to finish deployment by Friday."
                    }
                }]
            }
            mock_client.post = AsyncMock(return_value=mock_resp)

            synth = await pipeline.async_generate_summary_and_action_items(
                mock_client,
                full_transcript="Full meeting transcript text...",
                api_key="gsk_test"
            )
            self.assertIn("The team discussed Q3 goals.", synth["summary"])
            self.assertIn("John to finish deployment by Friday.", synth["action_items"])

        asyncio.run(run_synth())


class TestDBFormatting(unittest.TestCase):
    def test_vector_literal_formatting(self):
        vec = [0.123, -0.456, 0.789]
        formatted = db._vector_literal(vec)
        self.assertEqual(formatted, "[0.123,-0.456,0.789]")


if __name__ == "__main__":
    unittest.main()
