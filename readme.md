# Signal — Transcript Knowledge Base

Turns audio/video recordings into a semantically searchable knowledge base:

```
media file → Whisper transcription → chunking → LLM grammar correction
           → local embeddings → Postgres (pgvector) → semantic search
```

Built so an AI (or you) can later query "what did they say about X" and get
back the relevant moment from a recording, not just keyword matches.

---

## Architecture

```
kb_app/
├── main.py            FastAPI backend — upload, background jobs, search API
├── pipeline.py         Core pipeline logic (importable, no CLI)
├── static/index.html   Frontend — plain HTML/CSS/JS, no build step
└── requirements.txt
```

**Flow:**
1. `POST /process` — upload a file, kicks off a background job
2. `GET /status/{job_id}` — poll for progress (transcribing → correcting → embedding → saving)
3. `POST /search` — semantic search over everything stored so far

Whisper and the embedding model are loaded once and cached in memory across
requests, so repeated calls don't reload them.

---

## Setup

### 1. Install dependencies

```bash
cd kb_app
pip install -r requirements.txt
```

You also need **ffmpeg** installed and on your `PATH` (Whisper uses it to
decode audio from video containers).

```bash
ffmpeg -version   # should print a version, not "command not found"
```

### 2. Set up Postgres with pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The `transcript_chunks` table is created automatically on first run if it
doesn't exist.

### 3. Set environment variables

```bash
cp .env.example .env
```

Then edit `.env` with your real values:

```
OPENROUTER_API_KEY=sk-or-...
PGDATABASE=your_db
PGUSER=your_user
PGPASSWORD=your_password
PGHOST=localhost
PGPORT=5432
```

`main.py` loads `.env` automatically on startup via `python-dotenv`. Never
commit the real `.env` file — only `.env.example` should go in version control.

### 4. Run it

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in a browser.

---

## Using the UI

**Input Deck** (left) — upload a file, pick a Whisper model size, set chunk
length, and hit **Process**. A tape-counter log shows live progress
(transcribing → correcting chunk N/M → embedding → saving).

- **Whisper model**: `tiny`/`base` are fast but less accurate; `small`/`medium`
  give noticeably better transcripts at the cost of speed.
- **Chunk length**: how many seconds of transcript get grouped into one KB
  entry. Longer chunks = more context per entry, shorter = more precise
  retrieval. 30s is a reasonable default.
- **Skip LLM correction**: bypasses the grammar-correction pass (raw Whisper
  output goes straight to embedding). Useful for quick tests without burning
  API calls.

**Search Deck** (right) — type a natural-language query and get back the
most semantically similar chunks, with source file, timestamp range, and
distance score (lower = more similar).

---

## API reference

### `POST /process`
Multipart form upload.

| Field | Type | Default |
|---|---|---|
| `file` | file | required |
| `model` | string | `"base"` |
| `chunk_seconds` | float | `30.0` |
| `skip_correction` | bool | `false` |

Returns `{ "job_id": "..." }`.

### `GET /status/{job_id}`
Returns job state:
```json
{
  "status": "running | complete | error | queued",
  "messages": ["Transcribing audio...", "Corrected chunk 3/12", "..."],
  "result": { "chunks_saved": 12 },
  "error": null
}
```

### `POST /search`
```json
{ "query": "what did they say about pricing?", "top_k": 5 }
```
Returns:
```json
{
  "results": [
    {
      "source_file": "meeting.mp4",
      "start_time": 142.5,
      "end_time": 172.1,
      "text": "...",
      "distance": 0.31
    }
  ]
}
```

---

## Database schema

```sql
CREATE TABLE transcript_chunks (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,        -- LLM-corrected transcript
    text_raw TEXT,             -- original Whisper output, kept for audit
    embedding VECTOR(384)      -- all-MiniLM-L6-v2 embeddings
);
```

`text_raw` is preserved so you can always see what the correction step
changed, or roll back if a correction ever drifts from what was actually said.

---

## How to verify it's actually working

It's reasonable not to just trust the UI's log messages — here's how to
confirm real work is happening at each stage:

1. **Watch the terminal running `uvicorn`.** Whisper and sentence-transformers
   print their own progress/download output there (e.g. model download bars
   on first run, torch device info). Faked progress wouldn't produce this.

2. **Check the live counter in the header.** It calls `GET /stats`, which
   runs a real `SELECT COUNT(*)` against Postgres — not something derived
   from the job log. Process a file and watch the number increase after
   completion.

3. **Query Postgres directly**, independent of the app entirely:
   ```sql
   SELECT source_file, start_time, end_time, LEFT(text, 80) AS preview
   FROM transcript_chunks
   ORDER BY id DESC
   LIMIT 10;
   ```
   If rows show up with your actual filename and plausible timestamps, the
   full pipeline ran end to end.

4. **Compare `text` vs `text_raw`** on a row. If the correction step ran,
   these should differ slightly (punctuation, fixed words) but say the same
   thing. If you used `--skip-correction` equivalent (the checkbox), they'll
   be identical or `text_raw` will be null.

5. **Search for something specific** you know was said in the recording,
   using different wording than what was actually spoken. Semantic search
   finding it despite the wording mismatch is strong evidence the embedding
   + pgvector similarity search is functioning, not just doing keyword match.

6. **Try a short, throwaway clip first** (10–20 seconds) rather than a long
   recording — it makes the whole pipeline finish in under a minute, so you
   can confirm each stage quickly before committing to a long file.

## Known limitations (read before deploying anywhere but localhost)

This is solid for local/internal use. Before putting it on the open internet:

- **Job state is in-memory** (`JOBS` dict in `main.py`) — restarting the
  server loses in-flight job status. Swap for Redis or a DB table, and
  consider Celery/RQ instead of `BackgroundTasks` for real job durability.
- **No authentication** — anyone who can reach the API can upload files and
  query the KB.
- **CORS is wide open** (`allow_origins=["*"]`) — restrict this to your
  actual frontend origin.
- **No upload validation** — file size/type isn't checked before processing
  starts.
- **Correction step costs money per chunk** — one OpenRouter call per chunk.
  A long recording with a 30s chunk size can mean dozens of API calls; batch
  multiple chunks per call if this becomes a cost/speed issue.

---

## Tech stack

- **Transcription**: [OpenAI Whisper](https://github.com/openai/whisper) (local)
- **Correction**: any model via [OpenRouter](https://openrouter.ai) (default: `anthropic/claude-3.5-sonnet`)
- **Embeddings**: [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` (local, 384-dim)
- **Storage/search**: PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- **Backend**: FastAPI
- **Frontend**: static HTML/CSS/JS, no framework