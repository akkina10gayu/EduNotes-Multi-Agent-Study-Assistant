# EduNotes v2 Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate EduNotes from Streamlit + JSON/ChromaDB to Next.js + Supabase at `/Users/ganatejaakula/Desktop/Repos/Edunotesv2`.

**Architecture:** Monorepo with `frontend/` (Next.js on Netlify) and `backend/` (FastAPI on Railway). All storage moves to Supabase (Postgres + pgvector + Auth + Storage). Agent pipeline carries over from v1 unchanged.

**Tech Stack:** Next.js 14+, Tailwind CSS, Supabase (DB/Auth/Storage/pgvector), FastAPI, sentence-transformers, Groq API.

**Design Doc:** `docs/plans/2026-02-22-edunotesv2-migration-design.md`

---

## Phase 1: Project Scaffold & Supabase Schema

### Task 1: Initialize Monorepo

**Files:**
- Create: `Edunotesv2/` root directory
- Create: `Edunotesv2/.env.example`
- Create: `Edunotesv2/README.md`

**Step 1: Create project root and structure**

```bash
mkdir -p /Users/ganatejaakula/Desktop/Repos/Edunotesv2
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2
git init
mkdir -p backend/src/{agents,api,db,utils,models,knowledge_base}
mkdir -p backend/config backend/scripts
mkdir -p frontend
mkdir -p supabase/migrations
```

**Step 2: Create `.env.example`**

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# LLM
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_PROVIDER=groq
ENABLE_LOCAL_FALLBACK=false

# Backend
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Frontend
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: initialize Edunotesv2 monorepo structure"
```

---

### Task 2: Write Supabase Migration

**Files:**
- Create: `Edunotesv2/supabase/migrations/001_initial_schema.sql`

**Step 1: Create the migration file**

Write the full SQL schema from the design doc (Section 2). This includes:
- `vector` extension
- Tables: `documents`, `document_chunks`, `flashcard_sets`, `flashcards`, `quizzes`, `quiz_questions`, `quiz_attempts`, `quiz_answers`, `study_activities`, `study_streaks`
- All indexes including IVFFlat for pgvector
- RLS policies on every table

Reference: design doc Section 2 has the exact SQL.

**Step 2: Create seed file**

Create `Edunotesv2/supabase/seed.sql` with sample topics:

```sql
-- Seed data is inserted after user signup via the app.
-- This file is a placeholder for future seeding needs.
```

**Step 3: Commit**

```bash
git add supabase/
git commit -m "feat: add Supabase schema migration with pgvector and RLS"
```

---

## Phase 2: Backend — Copy & Refactor

### Task 3: Copy Agent Code (Unchanged)

**Files:**
- Copy: `src/agents/*.py` -> `Edunotesv2/backend/src/agents/`
- Copy: `src/utils/*.py` -> `Edunotesv2/backend/src/utils/`
- Copy: `src/models/*.py` -> `Edunotesv2/backend/src/models/`
- Copy: `src/knowledge_base/document_processor.py` -> `Edunotesv2/backend/src/knowledge_base/`
- Copy: `config/settings.py` -> `Edunotesv2/backend/config/`
- Copy: `src/__init__.py`, `src/agents/__init__.py`, etc.

**Step 1: Copy all source files**

```bash
# From EduNotes-Multi-Agent-Study-Assistant root
cp -r src/agents/* /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/agents/
cp -r src/utils/* /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/utils/
cp -r src/models/* /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/models/
cp src/knowledge_base/document_processor.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/knowledge_base/
cp src/__init__.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/
cp src/agents/__init__.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/agents/
cp src/utils/__init__.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/utils/
cp src/models/__init__.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/models/
touch /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/knowledge_base/__init__.py
touch /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/db/__init__.py
cp config/* /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/config/
cp src/api/*.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/src/api/
cp scripts/*.py /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend/scripts/
```

**Step 2: Commit raw copy**

```bash
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2
git add backend/
git commit -m "chore: copy v1 backend source files (pre-refactor)"
```

---

### Task 4: Update `config/settings.py`

**Files:**
- Modify: `Edunotesv2/backend/config/settings.py`

**Step 1: Add Supabase settings, remove file-based storage settings**

Add these new settings:
```python
# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Feature flags
ENABLE_LOCAL_FALLBACK = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000")
```

Remove these settings (no longer needed):
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `EMBEDDING_DIMENSION`
- `KB_PATH`
- `FLASHCARD_STORAGE`, `PROGRESS_STORAGE`
- `LANGCHAIN_CACHE`, `LANGCHAIN_CACHE_PATH`, `LANGCHAIN_VERBOSE`

Keep: `EMBEDDING_MODEL` (still used for generating embeddings), all LLM settings, search settings, scraper settings, cache settings.

**Step 2: Commit**

```bash
git add backend/config/settings.py
git commit -m "feat: update settings for Supabase, add ENABLE_LOCAL_FALLBACK flag"
```

---

### Task 5: Create Supabase Client

**Files:**
- Create: `Edunotesv2/backend/src/db/supabase_client.py`

**Step 1: Write the client**

```python
"""Singleton Supabase client for backend."""
import os
from supabase import create_client, Client

_client: Client | None = None


def get_supabase_client() -> Client:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
            )
        _client = create_client(url, key)
    return _client
```

**Step 2: Commit**

```bash
git add backend/src/db/supabase_client.py
git commit -m "feat: add Supabase client singleton"
```

---

### Task 6: Create Document Store (replaces VectorStore + DocumentProcessor)

**Files:**
- Create: `Edunotesv2/backend/src/db/document_store.py`

**Step 1: Write the store**

This class replaces both `src/knowledge_base/vector_store.py` and parts of `src/knowledge_base/document_processor.py`. It must implement:

- `add_document(user_id, title, topic, content, source, url, metadata)` — inserts into `documents` table, chunks text using `RecursiveCharacterTextSplitter` (same settings as v1: chunk_size=1000, overlap=200), generates embeddings via `sentence-transformers`, inserts chunks with embeddings into `document_chunks`.
- `search(user_id, query, top_k=5, threshold=0.7)` — generates query embedding, runs pgvector cosine similarity via Supabase RPC or raw SQL.
- `list_documents(user_id)` — returns all documents for user (id, title, topic, source, created_at).
- `get_document(user_id, doc_id)` — returns full document content.
- `delete_document(user_id, doc_id)` — deletes document and cascaded chunks.
- `get_collection_stats(user_id)` — returns doc count, chunk count.
- `get_unique_topics(user_id)` — returns distinct topics.
- `search_documents_semantic(user_id, query, limit)` — searches document-level (not chunk-level) by matching against chunk embeddings, returns unique parent documents.

Use `sentence_transformers.SentenceTransformer` for embedding generation (same model: `all-MiniLM-L6-v2`). The embedding model is loaded once and reused.

For pgvector search, create a Supabase RPC function in a new migration:

```sql
-- supabase/migrations/002_search_function.sql
create or replace function match_document_chunks(
    query_embedding vector(384),
    match_threshold float,
    match_count int,
    p_user_id uuid
)
returns table (
    id uuid,
    document_id uuid,
    chunk_index int,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    join documents d on dc.document_id = d.id
    where d.user_id = p_user_id
        and 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;
```

**Step 2: Commit**

```bash
git add backend/src/db/document_store.py supabase/migrations/002_search_function.sql
git commit -m "feat: add DocumentStore with pgvector search"
```

---

### Task 7: Create Flashcard Store

**Files:**
- Create: `Edunotesv2/backend/src/db/flashcard_store.py`

**Step 1: Write the store**

Same interface as current `src/utils/flashcard_store.py` (methods: `save_set`, `load_set`, `delete_set`, `list_sets`, `update_card_review`, `export_to_anki`, `export_all_to_anki`, `get_sets_by_topic`, `get_statistics`), but all operations go through `get_supabase_client()` instead of JSON files.

Key mapping:
- `save_set(set_data)` → insert into `flashcard_sets` + bulk insert into `flashcards`
- `load_set(set_id)` → select from `flashcard_sets` join `flashcards`
- `list_sets(user_id)` → select from `flashcard_sets` where `user_id` matches
- `update_card_review(set_id, card_id, correct)` → update `flashcards` row (increment `times_reviewed`, conditionally increment `times_correct`, set `last_reviewed`)
- `export_to_anki(set_id)` → query cards, format as tab-separated text
- `get_statistics(user_id)` → aggregate queries across `flashcard_sets` and `flashcards`

All methods take `user_id` as first parameter (new vs v1).

**Step 2: Commit**

```bash
git add backend/src/db/flashcard_store.py
git commit -m "feat: add FlashcardStore backed by Supabase"
```

---

### Task 8: Create Quiz Store

**Files:**
- Create: `Edunotesv2/backend/src/db/quiz_store.py`

**Step 1: Write the store**

Consolidates quiz storage from `study_routes.py`. Methods:

- `save_quiz(user_id, quiz_data)` → insert into `quizzes` + `quiz_questions`
- `load_quiz(user_id, quiz_id)` → select quiz + questions
- `list_quizzes(user_id)` → select from `quizzes`
- `delete_quiz(user_id, quiz_id)` → delete (cascades to questions, attempts, answers)
- `start_attempt(user_id, quiz_id)` → insert into `quiz_attempts`, return attempt_id
- `submit_answer(attempt_id, question_id, answer)` → check correctness against `quiz_questions.correct_answer`, insert into `quiz_answers`
- `complete_attempt(attempt_id)` → calculate score, update `quiz_attempts` with `completed_at`, `score`, `correct_count`
- `get_attempt_results(attempt_id)` → return detailed results with per-question breakdown

**Step 2: Commit**

```bash
git add backend/src/db/quiz_store.py
git commit -m "feat: add QuizStore backed by Supabase"
```

---

### Task 9: Create Progress Store

**Files:**
- Create: `Edunotesv2/backend/src/db/progress_store.py`

**Step 1: Write the store**

Same interface as current `src/utils/progress_store.py`. Methods:

- `record_activity(user_id, activity_type, topic, metadata)` → insert into `study_activities`, update `study_streaks` (upsert: if `last_activity_date` is yesterday, increment `current_streak`; if today, no change; otherwise reset to 1; update `best_streak` if current > best)
- `record_note_generated(user_id, topic)` → calls `record_activity` with type `note_generated`
- `record_flashcard_review(user_id, topic, correct)` → calls `record_activity`
- `record_quiz_completed(user_id, topic, score, total)` → calls `record_activity`
- `get_statistics(user_id)` → aggregate: total notes, flashcards reviewed, quizzes completed, topics studied, flashcard accuracy, quiz accuracy
- `get_topic_rankings(user_id)` → group by topic, count activities, calculate mastery level
- `get_weekly_summary(user_id)` → filter activities to current week, count by type, count distinct active days
- `get_recent_activities(user_id, limit=5)` → select recent from `study_activities`
- `reset_progress(user_id)` → delete all activities and streak for user

**Step 2: Commit**

```bash
git add backend/src/db/progress_store.py
git commit -m "feat: add ProgressStore backed by Supabase"
```

---

### Task 10: Create File Storage

**Files:**
- Create: `Edunotesv2/backend/src/db/file_storage.py`

**Step 1: Write the storage wrapper**

Two Supabase Storage buckets: `pdfs`, `exports`.

Methods:
- `upload_pdf(user_id, filename, content_bytes)` → uploads to `pdfs/{user_id}/{filename}`, returns public URL
- `upload_export(user_id, filename, content)` → uploads to `exports/{user_id}/{filename}`, returns signed URL
- `get_pdf_url(user_id, filename)` → returns signed URL
- `delete_pdf(user_id, filename)` → removes from storage

**Step 2: Commit**

```bash
git add backend/src/db/file_storage.py
git commit -m "feat: add Supabase Storage wrapper for PDFs and exports"
```

---

### Task 11: Create Auth Middleware

**Files:**
- Create: `Edunotesv2/backend/src/api/auth.py`

**Step 1: Write JWT validation dependency**

```python
"""Supabase JWT authentication for FastAPI."""
import os
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate Supabase JWT and return user_id."""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no sub claim")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
```

Add `SUPABASE_JWT_SECRET` to `.env.example` and `config/settings.py`.

**Step 2: Commit**

```bash
git add backend/src/api/auth.py
git commit -m "feat: add Supabase JWT auth middleware"
```

---

### Task 12: Refactor `app.py` — Swap Stores, Add Auth & CORS

**Files:**
- Modify: `Edunotesv2/backend/src/api/app.py`

**Step 1: Update imports and initialization**

Replace:
```python
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.document_processor import DocumentProcessor
from src.utils.progress_store import ProgressStore
```

With:
```python
from src.db.document_store import DocumentStore
from src.db.flashcard_store import FlashcardStore
from src.db.progress_store import ProgressStore
from src.db.quiz_store import QuizStore
from src.api.auth import get_current_user
from fastapi.middleware.cors import CORSMiddleware
```

**Step 2: Add CORS middleware**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Step 3: Replace global store variables**

Replace `vector_store = VectorStore()` and `doc_processor = DocumentProcessor()` with:
```python
document_store = DocumentStore()
```

**Step 4: Add `user_id` dependency to all endpoints**

Every endpoint gets `user_id: str = Depends(get_current_user)` and passes it through to store methods.

Example for `generate_notes`:
```python
async def generate_notes(request: Request, body: GenerateNotesRequest, user_id: str = Depends(get_current_user)):
```

Then pass `user_id` to `orchestrator.process(...)` and `progress_store.record_note_generated(user_id, topic)`.

**Step 5: Update all KB endpoints**

- `update_knowledge_base` → use `document_store.add_document(user_id, ...)`
- `search_knowledge_base` → use `document_store.search(user_id, ...)`
- `list_documents` → use `document_store.list_documents(user_id)`
- `get_document` → use `document_store.get_document(user_id, doc_id)`
- `get_topics` → use `document_store.get_unique_topics(user_id)`
- `get_stats` → use `document_store.get_collection_stats(user_id)`

**Step 6: Commit**

```bash
git add backend/src/api/app.py
git commit -m "feat: refactor app.py for Supabase stores, auth, and CORS"
```

---

### Task 13: Refactor `study_routes.py`

**Files:**
- Modify: `Edunotesv2/backend/src/api/study_routes.py`

**Step 1: Add auth dependency to all routes**

Every route gets `user_id: str = Depends(get_current_user)`.

**Step 2: Replace store usage**

- Flashcard routes: use `FlashcardStore` from `src.db.flashcard_store`
- Quiz routes: use `QuizStore` from `src.db.quiz_store`
- Progress routes: use `ProgressStore` from `src.db.progress_store`

All store method calls now pass `user_id` as first argument.

**Step 3: Commit**

```bash
git add backend/src/api/study_routes.py
git commit -m "feat: refactor study_routes for Supabase stores and auth"
```

---

### Task 14: Refactor Orchestrator for `user_id`

**Files:**
- Modify: `Edunotesv2/backend/src/agents/orchestrator.py`

**Step 1: Thread `user_id` through**

- `Orchestrator.__init__` → replace `self.vector_store = VectorStore()` and `self.doc_processor = DocumentProcessor()` with `self.document_store = DocumentStore()`
- `Orchestrator.process(self, query, ..., user_id=None)` → add `user_id` parameter
- `process_topic_query` → pass `user_id` to `self.retriever.process()` and `self.document_store.search()`
- All retriever calls → pass `user_id`

**Step 2: Update RetrieverAgent**

- Modify: `Edunotesv2/backend/src/agents/retriever.py`
- `RetrieverAgent.process(input_data)` → `input_data` now includes `user_id` key
- Replace `self.vector_store.search(...)` with `self.document_store.search(user_id, ...)`

**Step 3: Commit**

```bash
git add backend/src/agents/orchestrator.py backend/src/agents/retriever.py
git commit -m "feat: thread user_id through orchestrator and retriever"
```

---

### Task 15: Implement Flan-T5 Feature Flag

**Files:**
- Modify: `Edunotesv2/backend/src/utils/llm_client.py`
- Modify: `Edunotesv2/backend/src/agents/summarizer.py`

**Step 1: Guard local model in LLMClient**

In `LLMClient._init_client()`, replace the fallback block:

```python
except Exception as e:
    logger.error(f"Failed to initialize {self.provider} client: {e}")
    if settings.ENABLE_LOCAL_FALLBACK:
        logger.info("Falling back to local model")
        self.provider = "local"
        self._init_local()
    else:
        raise RuntimeError(
            f"Failed to initialize LLM provider '{self.provider}': {e}. "
            "Set ENABLE_LOCAL_FALLBACK=true to use local models as fallback."
        )
```

In `LLMClient._init_local()`:
```python
def _init_local(self):
    if not settings.ENABLE_LOCAL_FALLBACK:
        raise RuntimeError(
            "Local model disabled. Set ENABLE_LOCAL_FALLBACK=true in .env to enable."
        )
    self.client = None
    self.model = "local"
    logger.info("Using local model mode (BART/Flan-T5)")
```

Guard the `USE_LOCAL_MODEL` check in `__init__`:
```python
use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
if use_local:
    if not settings.ENABLE_LOCAL_FALLBACK:
        raise RuntimeError(
            "USE_LOCAL_MODEL=true but ENABLE_LOCAL_FALLBACK=false. "
            "Set ENABLE_LOCAL_FALLBACK=true to use local models."
        )
    self.provider = "local"
```

**Step 2: Guard local model in SummarizerAgent**

In `SummarizerAgent._initialize()`, replace the fallback:
```python
except Exception as e:
    self.logger.error(f"Error initializing summarizer: {e}")
    if settings.ENABLE_LOCAL_FALLBACK:
        self.logger.info("Falling back to local model")
        self.use_local = True
        self._initialize_local_model()
    else:
        raise
```

In `SummarizerAgent.__init__`, guard `self.use_local`:
```python
self.use_local = settings.USE_LOCAL_MODEL and settings.ENABLE_LOCAL_FALLBACK
```

**Step 3: Make torch/transformers imports conditional**

In `summarizer.py`, wrap local model imports:
```python
if settings.ENABLE_LOCAL_FALLBACK:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
```

**Step 4: Commit**

```bash
git add backend/src/utils/llm_client.py backend/src/agents/summarizer.py
git commit -m "feat: gate Flan-T5 fallback behind ENABLE_LOCAL_FALLBACK flag"
```

---

### Task 16: Update `requirements.txt`

**Files:**
- Create: `Edunotesv2/backend/requirements.txt`

**Step 1: Write updated requirements**

```
# Core Framework
fastapi
uvicorn[standard]
pydantic
python-multipart

# Supabase
supabase
pyjwt

# ML Models - Embeddings (always needed)
sentence-transformers
numpy

# Free LLM APIs
groq
huggingface_hub

# Web Scraping
newspaper3k
beautifulsoup4
lxml
requests
aiohttp

# Web Search
ddgs
duckduckgo_search
googlesearch-python

# Caching
diskcache

# Data Processing
PyPDF2
pymupdf4llm
pymupdf

# Text Processing
langchain
langchain-community

# Utilities
python-dotenv
pyyaml
loguru
tqdm

# API Extensions
slowapi

# Development & Testing
pytest
pytest-asyncio
```

Create optional `Edunotesv2/backend/requirements-local.txt`:
```
# Only needed when ENABLE_LOCAL_FALLBACK=true
transformers
torch
tokenizers
```

**Step 2: Commit**

```bash
git add backend/requirements.txt backend/requirements-local.txt
git commit -m "feat: update requirements - add supabase, split local model deps"
```

---

### Task 17: Delete Replaced Files

**Files:**
- Delete: `Edunotesv2/backend/src/utils/flashcard_store.py`
- Delete: `Edunotesv2/backend/src/utils/progress_store.py`
- Delete: `Edunotesv2/backend/src/knowledge_base/vector_store.py` (if copied)

**Step 1: Remove replaced files**

```bash
rm -f backend/src/utils/flashcard_store.py
rm -f backend/src/utils/progress_store.py
rm -f backend/src/knowledge_base/vector_store.py
```

**Step 2: Update `__init__.py` files to remove old imports**

Check `backend/src/utils/__init__.py` and `backend/src/knowledge_base/__init__.py` for any imports of the deleted modules and remove them.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove replaced file-based stores"
```

---

### Task 18: Create Backend Dockerfile

**Files:**
- Create: `Edunotesv2/backend/Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create `.dockerignore`**

```
__pycache__
*.pyc
.env
.git
venv
data/
logs/
```

**Step 3: Commit**

```bash
git add backend/Dockerfile backend/.dockerignore
git commit -m "feat: add Dockerfile for backend deployment"
```

---

## Phase 3: Frontend — Next.js

### Task 19: Initialize Next.js App

**Step 1: Create Next.js project**

```bash
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --no-import-alias
```

**Step 2: Install dependencies**

```bash
cd frontend
npm install @supabase/supabase-js @supabase/ssr react-markdown remark-gfm rehype-highlight
```

**Step 3: Create `netlify.toml`**

```toml
[build]
  command = "npm run build"
  publish = ".next"

[[plugins]]
  package = "@netlify/plugin-nextjs"
```

**Step 4: Commit**

```bash
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2
git add frontend/
git commit -m "feat: initialize Next.js app with Tailwind and Supabase"
```

---

### Task 20: Supabase Client Setup (Frontend)

**Files:**
- Create: `frontend/src/lib/supabase/client.ts`
- Create: `frontend/src/lib/supabase/server.ts`
- Create: `frontend/src/lib/supabase/middleware.ts`
- Create: `frontend/src/middleware.ts`

**Step 1: Browser client**

```typescript
// frontend/src/lib/supabase/client.ts
import { createBrowserClient } from '@supabase/ssr'

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
```

**Step 2: Server client**

```typescript
// frontend/src/lib/supabase/server.ts
import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

export async function createClient() {
  const cookieStore = await cookies()
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return cookieStore.getAll() },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options)
          )
        },
      },
    }
  )
}
```

**Step 3: Auth middleware (protects routes)**

```typescript
// frontend/src/middleware.ts
import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({ request })
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return request.cookies.getAll() },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          supabaseResponse = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )
  const { data: { user } } = await supabase.auth.getUser()

  if (!user && !request.nextUrl.pathname.startsWith('/auth')) {
    const url = request.nextUrl.clone()
    url.pathname = '/auth/login'
    return NextResponse.redirect(url)
  }

  return supabaseResponse
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|auth).*)'],
}
```

**Step 4: Commit**

```bash
git add frontend/src/lib/supabase/ frontend/src/middleware.ts
git commit -m "feat: add Supabase client and auth middleware"
```

---

### Task 21: API Client Layer

**Files:**
- Create: `frontend/src/lib/api/client.ts`
- Create: `frontend/src/lib/api/notes.ts`
- Create: `frontend/src/lib/api/study.ts`
- Create: `frontend/src/lib/api/kb.ts`
- Create: `frontend/src/lib/api/progress.ts`
- Create: `frontend/src/types/index.ts`

**Step 1: Create TypeScript types**

Define types matching the Pydantic models: `GenerateNotesRequest`, `GenerateNotesResponse`, `FlashcardSet`, `Flashcard`, `Quiz`, `QuizQuestion`, `StudyProgress`, `Document`, `SearchResult`, etc.

**Step 2: Create base API client**

```typescript
// frontend/src/lib/api/client.ts
import { createClient } from '@/lib/supabase/client'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1'

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(session ? { 'Authorization': `Bearer ${session.access_token}` } : {}),
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `API error: ${response.status}`)
  }

  return response.json()
}
```

**Step 3: Create domain-specific API modules**

Each module (`notes.ts`, `study.ts`, `kb.ts`, `progress.ts`) wraps `apiClient` calls matching the FastAPI endpoints. For example:

```typescript
// frontend/src/lib/api/notes.ts
export async function generateNotes(params: GenerateNotesRequest) {
  return apiClient<GenerateNotesResponse>('/generate-notes', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export async function getTopics() {
  return apiClient<{ topics: string[] }>('/topics')
}
```

**Step 4: Commit**

```bash
git add frontend/src/lib/api/ frontend/src/types/
git commit -m "feat: add typed API client layer"
```

---

### Task 22: Auth Pages

**Files:**
- Create: `frontend/src/app/auth/login/page.tsx`
- Create: `frontend/src/app/auth/callback/route.ts`

**Step 1: Login page**

Simple page with email/password form using Supabase Auth. Include "Sign Up" toggle. Style to match the EduNotes blue (`#6CA0DC`) branding.

**Step 2: OAuth callback route**

```typescript
// frontend/src/app/auth/callback/route.ts
import { createClient } from '@/lib/supabase/server'
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  if (code) {
    const supabase = await createClient()
    await supabase.auth.exchangeCodeForSession(code)
  }
  return NextResponse.redirect(`${origin}/`)
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/auth/
git commit -m "feat: add login page and OAuth callback"
```

---

### Task 23: Root Layout & Navigation

**Files:**
- Modify: `frontend/src/app/layout.tsx`
- Create: `frontend/src/components/layout/Navbar.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Modify: `frontend/src/app/globals.css`

**Step 1: Root layout**

Layout with:
- Navbar at top with "EduNotes Study Assistant" header in `#6CA0DC`
- Sidebar (collapsible) with: API health indicator, font size control (Small/Medium/Large stored in localStorage), System Stats accordion, Note History accordion, Help & Setup Guide accordion
- Main content area
- Footer: "EduNotes v2.0 | Multi-Agent Study Assistant"

**Step 2: Global CSS**

Add the `.notes-container` style (dark bg `#1E1E1E`, blue left border `#6CA0DC`), flashcard gradient styles, and font size CSS variables that match the Streamlit CSS exactly.

**Step 3: Sidebar**

Replicate every sidebar feature from the Streamlit app:
- API health check with 30s polling
- Font size radio (Small/Medium/Large) → updates CSS variable, persists in localStorage
- System Stats expander: LLM model info, KB doc count, agent statuses
- Note History expander: last 6 notes, truncated queries (35 chars), type icons (topic/url/text/pdf), View button
- Help & Setup Guide expander: getting started, API key setup, how to use, troubleshooting

**Step 4: Commit**

```bash
git add frontend/src/app/layout.tsx frontend/src/app/globals.css frontend/src/components/layout/
git commit -m "feat: add root layout, navbar, and sidebar with full feature parity"
```

---

### Task 24: Generate Notes Page (Main Feature)

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Create: `frontend/src/components/notes/TopicChips.tsx`
- Create: `frontend/src/components/notes/InputPanel.tsx`
- Create: `frontend/src/components/notes/NoteViewer.tsx`
- Create: `frontend/src/components/notes/NoteEditor.tsx`
- Create: `frontend/src/components/notes/ResearchResults.tsx`
- Create: `frontend/src/components/notes/SaveToKBModal.tsx`

This is the largest task. It must replicate every feature from Streamlit Tab 1 (lines 1032-2092 of `streamlit_app.py`).

**Step 1: TopicChips component**

- Fetches topics from API (5-min cache via SWR or manual timer)
- Merges KB topics with defaults (Machine Learning, Deep Learning, etc.)
- Displays 5 per row, truncated at 14 chars with tooltip
- "Show more" / "Show less" toggle (5 default, 10 expanded)
- Click fills input

**Step 2: InputPanel component**

- Two-column layout: textarea (topic/URL/text) + PDF file drop zone
- Auto-detection with inline hints ("URL detected", "Text detected", "Topic detected")
- Research Mode toggle with help text
- Search Mode radio (Auto/KB Only/Web Search/Both) — only visible for topic input
- Output Format radio (Paragraph Summary/Important Points/Key Highlights)
- Summary Length radio (Auto/Brief/Medium/Detailed) — only for Paragraph Summary
- Generate button with:
  - Conflict check (both PDF and text)
  - Progress bar with 4 stages and status messages
  - Rate limit error handling with model switching dialog

**Step 3: NoteViewer component (reusable for recent + history)**

- Close button
- Metadata bar: Query Type, Sources Used, Source, Search Mode (4 cols for topic, 3 for others)
- Dark markdown container with `react-markdown` + `remark-gfm`
- Action buttons: Download (.md), Copy (via `navigator.clipboard.writeText`), Save to KB, Edit
- Undo Last Edit button (visible after save)
- Dismissible success messages

**Step 4: NoteEditor component**

- Textarea with `useReducer` for undo/redo stack
- Save/Undo/Redo/Cancel buttons
- On save: updates note in state + history

**Step 5: ResearchResults component**

- Collapsible section for vision/figure data
- Displays base64 images with `<img>` tags
- Cleaned markdown descriptions (heading downshift, remove "No X present" lines)

**Step 6: SaveToKBModal component**

- Auto-filled Title, Topic, Source, URL fields based on query metadata
- PDF duplicate detection (checks API)
- Validation (title + topic required)
- Save/Cancel buttons

**Step 7: First-time welcome banner**

- Conditional on `localStorage` flag
- Dismissible with "Got it!" button

**Step 8: Quick Stats bar**

- 4 metric cards: Notes (session count), Flashcard sets, Quizzes, Streak (with best delta)

**Step 9: History Note Viewer section**

- Below recent notes, uses same `NoteViewer` component
- Triggered from sidebar history selection

**Step 10: Commit**

```bash
git add frontend/src/app/page.tsx frontend/src/components/notes/
git commit -m "feat: add Generate Notes page with full Streamlit parity"
```

---

### Task 25: Knowledge Base Page

**Files:**
- Create: `frontend/src/app/kb/page.tsx`
- Create: `frontend/src/components/kb/DocumentList.tsx`
- Create: `frontend/src/components/kb/DocumentViewer.tsx`
- Create: `frontend/src/components/kb/SemanticSearch.tsx`
- Create: `frontend/src/components/kb/AddDocumentForm.tsx`

Must replicate Streamlit Tabs 2 + 3 (Search KB + Update KB).

**Step 1: Two-mode toggle**

"Browse Documents" / "Search Vector DB" radio toggle at top.

**Step 2: Browse Documents mode**

- Search input with button
- Document list (semantic when keyword provided, full list otherwise)
- Selectbox/dropdown to pick document
- Info bar (topic, date, word count)
- Scrollable content in dark `.notes-container`
- Download/Copy/View as Text buttons

**Step 3: Search Vector DB mode**

- Query input
- Num results slider (1-10)
- Similarity threshold slider (0.0-1.0, step 0.05)
- Expandable result cards with score, chunk content, metadata caption

**Step 4: Add Document section (replaces Tab 3)**

- Form: Title, Topic (default "general"), Source (default "manual"), Content textarea
- Submit button with validation

**Step 5: Commit**

```bash
git add frontend/src/app/kb/ frontend/src/components/kb/
git commit -m "feat: add Knowledge Base page with browse, search, and add"
```

---

### Task 26: Study Mode Page

**Files:**
- Create: `frontend/src/app/study/page.tsx`
- Create: `frontend/src/components/study/FlashcardGenerator.tsx`
- Create: `frontend/src/components/study/FlashcardReview.tsx`
- Create: `frontend/src/components/study/QuizGenerator.tsx`
- Create: `frontend/src/components/study/QuizTaker.tsx`
- Create: `frontend/src/components/study/AnkiExport.tsx`

Must replicate Streamlit Tab 4 (Study Mode) with its 3 sub-tabs.

**Step 1: Flashcards tab — left column**

- Generate: topic input, card count slider (5-20, default 10), content textarea, generate button with spinner
- Load Existing: dropdown of existing sets, load button
- Export to Anki: dropdown for specific set export, "Export All" link with total card count

**Step 2: Flashcards tab — right column (FlashcardReview)**

- Progress bar + "Card N of M"
- Gradient flashcard: purple (`linear-gradient(135deg, #667eea 0%, #764ba2 100%)`) for Q, green (`linear-gradient(135deg, #11998e 0%, #38ef7d 100%)`) for A
- "Show Answer" button → flips to answer
- "Got it!" (records correct) / "Review Again" (records incorrect) buttons with API tracking
- Previous / Shuffle / Next navigation

**Step 3: Quizzes tab — left column**

- Generate: topic, question count slider (3-15, default 5), content textarea, generate button
- Load Existing: dropdown, load button

**Step 4: Quizzes tab — right column (QuizTaker)**

- Quiz title + question count header
- Per-question: question text, radio options (index=None initially)
- "Submit Quiz" button (validates all answered)
- Results view:
  - Color-coded score: green (>=80%), blue (>=60%), yellow (<60%)
  - Correct/Incorrect metric cards
  - Detailed per-question review with green/red highlighting
  - Explanation text
  - "Try Again" / "New Quiz" buttons

**Step 5: Commit**

```bash
git add frontend/src/app/study/ frontend/src/components/study/
git commit -m "feat: add Study Mode page with flashcards, quizzes, and Anki export"
```

---

### Task 27: Progress Page

**Files:**
- Create: `frontend/src/app/progress/page.tsx`
- Create: `frontend/src/components/progress/StreakDisplay.tsx`
- Create: `frontend/src/components/progress/TopicRankings.tsx`
- Create: `frontend/src/components/progress/WeeklyActivity.tsx`

Must replicate Streamlit Study Mode → Progress sub-tab.

**Step 1: Overall Statistics**

4 metric cards: Notes Generated, Flashcards Reviewed, Quizzes Completed, Topics Studied

**Step 2: Streak + Accuracy**

Two-column:
- Left: Current Streak (days), Longest Streak (days)
- Right: Flashcard Accuracy %, Quiz Accuracy %

**Step 3: Topic Mastery**

Top 5 topics table: Topic name, Mastery Level, Notes count

**Step 4: This Week**

3 metric cards: Notes This Week, Flashcards Reviewed, Active Days (X/7)

**Step 5: Recent Activity**

Last 5 activities: summary text + date

**Step 6: Commit**

```bash
git add frontend/src/app/progress/ frontend/src/components/progress/
git commit -m "feat: add Progress page with streaks, rankings, and activity"
```

---

### Task 28: Shared UI Components

**Files:**
- Create: `frontend/src/components/ui/LoadingSpinner.tsx`
- Create: `frontend/src/components/ui/ErrorBoundary.tsx`
- Create: `frontend/src/components/ui/ConfirmDialog.tsx`
- Create: `frontend/src/lib/utils/markdown.ts`
- Create: `frontend/src/lib/utils/formatters.ts`

**Step 1: LoadingSpinner**

Reusable spinner component with optional message text.

**Step 2: ErrorBoundary**

React error boundary that catches and displays errors gracefully.

**Step 3: ConfirmDialog**

Reusable confirmation dialog for destructive actions.

**Step 4: Markdown utilities**

Helper for cleaning vision descriptions (heading downshift, remove "No X present" lines) — same logic as Streamlit's inline cleaning.

**Step 5: Formatters**

Date formatting, score formatting, streak display formatting.

**Step 6: Commit**

```bash
git add frontend/src/components/ui/ frontend/src/lib/utils/
git commit -m "feat: add shared UI components and utility functions"
```

---

## Phase 4: Integration & Deployment

### Task 29: End-to-End Testing

**Step 1: Start Supabase locally**

```bash
npx supabase init  # if not done
npx supabase start
npx supabase db push  # apply migrations
```

**Step 2: Start backend**

```bash
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2/backend
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

**Step 3: Start frontend**

```bash
cd /Users/ganatejaakula/Desktop/Repos/Edunotesv2/frontend
npm run dev
```

**Step 4: Test each flow**

- [ ] Sign up / login
- [ ] Generate notes from topic (all search modes)
- [ ] Generate notes from URL
- [ ] Generate notes from pasted text
- [ ] Generate notes from PDF upload
- [ ] Research mode with vision analysis
- [ ] Edit notes inline (undo/redo)
- [ ] Save notes to KB
- [ ] Copy notes to clipboard
- [ ] Download notes as .md
- [ ] Browse KB documents
- [ ] Search KB semantically
- [ ] Add document to KB manually
- [ ] Generate flashcards
- [ ] Review flashcards (Got it / Review Again)
- [ ] Export flashcards to Anki
- [ ] Generate quiz
- [ ] Take quiz and submit
- [ ] View quiz results
- [ ] Check progress page (streaks, rankings, activity)
- [ ] Font size toggle
- [ ] Note history in sidebar
- [ ] Rate limit error handling

**Step 5: Fix any issues found**

**Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

### Task 30: Deployment Configuration

**Step 1: Set up Supabase project**

1. Create project at [supabase.com](https://supabase.com)
2. Run migrations via Supabase CLI or dashboard SQL editor
3. Enable pgvector extension
4. Create storage buckets: `pdfs`, `exports`
5. Note down: URL, anon key, service role key, JWT secret

**Step 2: Deploy backend to Railway**

1. Connect Railway to the git repo
2. Set root directory to `backend/`
3. Set all environment variables
4. Verify health check at `/api/v1/health`

**Step 3: Deploy frontend to Netlify**

1. Connect Netlify to the git repo
2. Set build directory to `frontend/`
3. Set environment variables (NEXT_PUBLIC_* only)
4. Update `CORS_ORIGINS` on backend to include Netlify URL
5. Verify at deployed URL

**Step 4: Commit any deployment tweaks**

```bash
git add -A
git commit -m "chore: finalize deployment configuration"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | 1-2 | Project scaffold + Supabase schema |
| Phase 2 | 3-18 | Backend copy, Supabase stores, auth, feature flag, Dockerfile |
| Phase 3 | 19-28 | Next.js frontend with full Streamlit parity |
| Phase 4 | 29-30 | Integration testing + deployment |

**Total: 30 tasks across 4 phases.**
