# EduNotes v2 Migration Design

**Date:** 2026-02-22
**Target Location:** `/Users/ganatejaakula/Desktop/Repos/Edunotesv2`
**Approach:** Full Parallel Rewrite (Approach 1)

## Overview

Migrate EduNotes from Streamlit + JSON/ChromaDB to Next.js + Supabase (Full: DB + Auth + Storage + pgvector), keeping FastAPI as the backend deployed separately.

### What Changes

| Layer | Current | Target |
|-------|---------|--------|
| Frontend | Streamlit (single Python file, port 8501) | Next.js 14+ App Router (Netlify) |
| Database | JSON files + ChromaDB | Supabase Postgres + pgvector |
| Auth | None (single user) | Supabase Auth (multi-user, RLS) |
| File Storage | Local filesystem (`data/`) | Supabase Storage buckets |
| Vector Search | ChromaDB + LangChain | pgvector extension in Supabase |
| Local Fallback | Always available (Flan-T5/BART) | Behind `ENABLE_LOCAL_FALLBACK` flag |
| Backend | FastAPI (localhost:8000) | FastAPI (Railway/Render) |

### What Stays the Same

- FastAPI backend framework
- All agent code (orchestrator, content_agent, web_search, scraper, summarizer, note_maker)
- Groq API as primary LLM provider
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- DuckDuckGo + Google for web search
- PDF processing pipeline (pymupdf4llm + PyPDF2)
- Academic search (arXiv + Semantic Scholar)

---

## 1. Project Structure

```
Edunotesv2/
├── frontend/                  # Next.js app (deploys to Netlify)
│   ├── src/
│   │   ├── app/               # App Router pages
│   │   │   ├── layout.tsx     # Root layout - nav, auth provider, theme
│   │   │   ├── page.tsx       # Generate Notes (main feature)
│   │   │   ├── auth/
│   │   │   │   ├── login/page.tsx
│   │   │   │   └── callback/route.ts
│   │   │   ├── study/page.tsx # Flashcards + Quizzes
│   │   │   ├── kb/page.tsx    # Knowledge Base
│   │   │   └── progress/page.tsx
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Navbar.tsx
│   │   │   │   └── Sidebar.tsx
│   │   │   ├── notes/
│   │   │   │   ├── TopicChips.tsx
│   │   │   │   ├── InputPanel.tsx
│   │   │   │   ├── NoteViewer.tsx
│   │   │   │   ├── NoteEditor.tsx
│   │   │   │   ├── ResearchResults.tsx
│   │   │   │   └── SaveToKBModal.tsx
│   │   │   ├── study/
│   │   │   │   ├── FlashcardGenerator.tsx
│   │   │   │   ├── FlashcardReview.tsx
│   │   │   │   ├── QuizGenerator.tsx
│   │   │   │   ├── QuizTaker.tsx
│   │   │   │   └── AnkiExport.tsx
│   │   │   ├── kb/
│   │   │   │   ├── DocumentList.tsx
│   │   │   │   ├── DocumentViewer.tsx
│   │   │   │   ├── SemanticSearch.tsx
│   │   │   │   └── AddDocumentForm.tsx
│   │   │   ├── progress/
│   │   │   │   ├── StreakDisplay.tsx
│   │   │   │   ├── TopicRankings.tsx
│   │   │   │   └── WeeklyActivity.tsx
│   │   │   └── ui/
│   │   │       ├── LoadingSpinner.tsx
│   │   │       ├── ErrorBoundary.tsx
│   │   │       └── ConfirmDialog.tsx
│   │   ├── lib/
│   │   │   ├── supabase/
│   │   │   │   ├── client.ts
│   │   │   │   └── server.ts
│   │   │   ├── api/
│   │   │   │   ├── client.ts
│   │   │   │   ├── notes.ts
│   │   │   │   ├── study.ts
│   │   │   │   ├── kb.ts
│   │   │   │   └── progress.ts
│   │   │   └── utils/
│   │   │       ├── markdown.ts
│   │   │       └── formatters.ts
│   │   └── types/
│   ├── public/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── netlify.toml
│
├── backend/
│   ├── src/
│   │   ├── agents/            # Carried over from v1
│   │   │   ├── orchestrator.py
│   │   │   ├── content_agent.py
│   │   │   ├── web_search.py
│   │   │   ├── scraper.py
│   │   │   ├── summarizer.py
│   │   │   ├── note_maker.py
│   │   │   ├── retriever.py
│   │   │   └── base.py
│   │   ├── api/
│   │   │   ├── app.py
│   │   │   ├── study_routes.py
│   │   │   └── auth.py        # NEW - JWT validation
│   │   ├── db/                # NEW - Supabase stores
│   │   │   ├── supabase_client.py
│   │   │   ├── document_store.py
│   │   │   ├── flashcard_store.py
│   │   │   ├── progress_store.py
│   │   │   ├── quiz_store.py
│   │   │   └── file_storage.py
│   │   ├── utils/
│   │   │   ├── llm_client.py
│   │   │   ├── search_provider.py
│   │   │   ├── text_utils.py
│   │   │   ├── quiz_generator.py
│   │   │   ├── flashcard_generator.py
│   │   │   ├── cache_utils.py
│   │   │   ├── academic_search.py
│   │   │   ├── pdf_processor.py
│   │   │   └── logger.py
│   │   ├── models/
│   │   │   ├── schemas.py
│   │   │   ├── flashcard.py
│   │   │   ├── quiz.py
│   │   │   └── progress.py
│   │   └── knowledge_base/
│   │       └── document_processor.py  # Refactored for Supabase
│   ├── config/
│   │   └── settings.py
│   ├── scripts/
│   │   ├── setup_kb.py
│   │   └── seed_data.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── supabase/
│   ├── migrations/
│   │   └── 001_initial_schema.sql
│   └── seed.sql
│
├── .env.example
└── README.md
```

---

## 2. Supabase Database Schema

```sql
-- Enable pgvector extension
create extension if not exists vector;

-- ==================== KNOWLEDGE BASE ====================

create table documents (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    title text not null,
    topic text,
    source text,              -- 'topic', 'url', 'pdf', 'text'
    url text,
    content text not null,
    metadata jsonb default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    chunk_index integer not null,
    content text not null,
    embedding vector(384),    -- all-MiniLM-L6-v2 = 384 dimensions
    metadata jsonb default '{}',
    created_at timestamptz default now()
);

-- ==================== FLASHCARDS ====================

create table flashcard_sets (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    topic text not null,
    source_content text,
    card_count integer default 0,
    created_at timestamptz default now()
);

create table flashcards (
    id uuid primary key default gen_random_uuid(),
    set_id uuid references flashcard_sets(id) on delete cascade,
    front text not null,
    back text not null,
    difficulty text default 'medium',
    times_reviewed integer default 0,
    times_correct integer default 0,
    last_reviewed timestamptz,
    created_at timestamptz default now()
);

-- ==================== QUIZZES ====================

create table quizzes (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    topic text not null,
    source_content text,
    question_count integer default 0,
    created_at timestamptz default now()
);

create table quiz_questions (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid references quizzes(id) on delete cascade,
    question text not null,
    question_type text default 'multiple_choice',
    options jsonb,
    correct_answer text not null,
    explanation text,
    order_index integer
);

create table quiz_attempts (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid references quizzes(id) on delete cascade,
    user_id uuid references auth.users(id) on delete cascade,
    started_at timestamptz default now(),
    completed_at timestamptz,
    score numeric,
    total_questions integer,
    correct_count integer
);

create table quiz_answers (
    id uuid primary key default gen_random_uuid(),
    attempt_id uuid references quiz_attempts(id) on delete cascade,
    question_id uuid references quiz_questions(id),
    user_answer text,
    is_correct boolean,
    answered_at timestamptz default now()
);

-- ==================== PROGRESS ====================

create table study_activities (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    activity_type text not null,
    topic text,
    metadata jsonb default '{}',
    created_at timestamptz default now()
);

create table study_streaks (
    user_id uuid primary key references auth.users(id) on delete cascade,
    current_streak integer default 0,
    best_streak integer default 0,
    last_activity_date date,
    updated_at timestamptz default now()
);

-- ==================== INDEXES ====================

create index idx_documents_user on documents(user_id);
create index idx_documents_topic on documents(topic);
create index idx_document_chunks_doc on document_chunks(document_id);
create index idx_flashcard_sets_user on flashcard_sets(user_id);
create index idx_quizzes_user on quizzes(user_id);
create index idx_activities_user on study_activities(user_id);
create index idx_activities_date on study_activities(created_at);

create index idx_chunks_embedding on document_chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ==================== ROW LEVEL SECURITY ====================

alter table documents enable row level security;
alter table document_chunks enable row level security;
alter table flashcard_sets enable row level security;
alter table flashcards enable row level security;
alter table quizzes enable row level security;
alter table quiz_questions enable row level security;
alter table quiz_attempts enable row level security;
alter table quiz_answers enable row level security;
alter table study_activities enable row level security;
alter table study_streaks enable row level security;

-- Users can only access their own data
create policy "Users access own data" on documents
    for all using (auth.uid() = user_id);
create policy "Users access own data" on flashcard_sets
    for all using (auth.uid() = user_id);
create policy "Users access own data" on quizzes
    for all using (auth.uid() = user_id);
create policy "Users access own data" on quiz_attempts
    for all using (auth.uid() = user_id);
create policy "Users access own data" on study_activities
    for all using (auth.uid() = user_id);
create policy "Users access own data" on study_streaks
    for all using (auth.uid() = user_id);

-- Child tables: accessible via parent ownership
create policy "Users access own chunks" on document_chunks
    for all using (
        document_id in (select id from documents where user_id = auth.uid())
    );
create policy "Users access own flashcards" on flashcards
    for all using (
        set_id in (select id from flashcard_sets where user_id = auth.uid())
    );
create policy "Users access own questions" on quiz_questions
    for all using (
        quiz_id in (select id from quizzes where user_id = auth.uid())
    );
create policy "Users access own answers" on quiz_answers
    for all using (
        attempt_id in (select id from quiz_attempts where user_id = auth.uid())
    );
```

---

## 3. Backend Architecture

### 3.1 Supabase Client Layer (`backend/src/db/`)

**`supabase_client.py`** - Singleton connection using `supabase-py` with `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`. Exposes `get_supabase_client()`.

**`document_store.py`** - Merges current `VectorStore` + `DocumentProcessor`:
- `add_document(user_id, title, topic, content, ...)` - inserts document, chunks text, generates embeddings via `sentence-transformers`, stores chunks + vectors in `document_chunks`
- `search(user_id, query, top_k)` - generates query embedding, calls Supabase RPC for cosine similarity search
- `list_documents(user_id)`, `get_document(user_id, doc_id)`, `delete_document(...)` - CRUD

**`flashcard_store.py`** - Same interface as current `FlashcardStore`: `save_set`, `load_set`, `delete_set`, `list_sets`, `update_card_review`, `export_to_anki`, `get_statistics`

**`progress_store.py`** - Same interface: `record_activity`, `get_statistics`, `get_topic_rankings`, `get_weekly_summary`

**`quiz_store.py`** - Consolidates quiz logic from `study_routes.py` into a proper store.

**`file_storage.py`** - Supabase Storage wrapper. Two buckets: `pdfs` and `exports`.

### 3.2 Auth Middleware (`src/api/auth.py`)

- FastAPI dependency that extracts Supabase JWT from `Authorization: Bearer <token>`
- Validates against Supabase JWT secret
- Injects `user_id` into request state
- All endpoints gain `user_id` parameter

### 3.3 Flan-T5 Feature Flag

In `config/settings.py`:
```python
ENABLE_LOCAL_FALLBACK = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
```

Behavior when `false` (default):
- `LLMClient._init_local()` raises `ConfigError`
- `transformers` and `torch` never imported
- Docker image ~2GB smaller
- If Groq fails, error returned to user (no silent fallback)

Behavior when `true`:
- Current behavior preserved
- Loads Flan-T5/BART as fallback

### 3.4 Agent Code Changes

| Module | Changes |
|--------|---------|
| `agents/orchestrator.py` | Replace `vector_store.search()` with `document_store.search()`. Thread `user_id`. |
| `agents/content_agent.py` | No changes |
| `agents/web_search.py` | No changes |
| `agents/summarizer.py` | Guard local model fallback behind `ENABLE_LOCAL_FALLBACK` |
| `agents/scraper.py` | No changes |
| `agents/note_maker.py` | No changes |
| `agents/retriever.py` | Replace `vector_store` calls with `document_store` |
| `api/app.py` | Swap store instantiation, add auth dependency, add CORS middleware |
| `api/study_routes.py` | Swap store calls, add `user_id` from auth |

~70% of backend code carries over unchanged.

### 3.5 Files Removed

- `src/utils/flashcard_store.py` - replaced by `src/db/flashcard_store.py`
- `src/utils/progress_store.py` - replaced by `src/db/progress_store.py`
- `src/knowledge_base/vector_store.py` - replaced by `src/db/document_store.py`
- `data/` directory - all data in Supabase

---

## 4. Frontend Architecture (Next.js)

### 4.1 Tech Stack

| Concern | Choice |
|---------|--------|
| Framework | Next.js 14+ (App Router) |
| Styling | Tailwind CSS |
| Auth | `@supabase/ssr` |
| API calls | Fetch with typed wrapper |
| State | React hooks + context |
| Markdown | `react-markdown` + `remark-gfm` |
| Code highlighting | `rehype-highlight` |

### 4.2 Auth Flow

1. User lands on app -> `layout.tsx` checks Supabase session
2. No session -> redirect to `/auth/login`
3. Login via Supabase Auth UI (email/password + optional OAuth)
4. Session cookie set via `@supabase/ssr`
5. Every page is protected (middleware checks session)
6. JWT passed to FastAPI on every API call
7. Backend validates JWT, extracts `user_id`, scopes all queries

### 4.3 API Client

All FastAPI calls go through a typed client that attaches the Supabase JWT:
```typescript
const response = await fetch(`${API_BASE_URL}/api/generate-notes`, {
  headers: {
    'Authorization': `Bearer ${session.access_token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(payload),
});
```

---

## 5. Frontend Feature Parity (Streamlit -> Next.js)

Every feature from the 2852-line `ui/streamlit_app.py` must be replicated exactly.

### 5.1 Global Layout & Sidebar

- Main header: "EduNotes Study Assistant" in `#6CA0DC`
- Sidebar: API health indicator (polled, 30s cache)
- Sidebar: Font size control (Small/Medium/Large) persisted in `localStorage`
- Sidebar: System Stats accordion (LLM model, KB doc count, agent statuses)
- Sidebar: Note History accordion (last 6 notes, truncated queries, type icons, View button)
- Sidebar: Help & Setup Guide accordion
- First-time welcome banner (dismissible, persisted in `localStorage`)
- Quick Stats bar (4 metrics: Notes, Flashcard sets, Quizzes, Streak with best delta)
- Custom notes container: dark background `#1E1E1E`, blue left border `#6CA0DC`
- Footer: "EduNotes v2.0"

### 5.2 Generate Notes Page

- **Quick Topics** - KB + default topics merged, 5 per row, truncated with tooltip, show more/less
- **Input Panel** - 2-column: textarea (topic/URL/text) + PDF uploader, auto-detection captions
- **Research Mode** toggle with help text
- **Search Mode** radio (Auto/KB Only/Web Search/Both) - only visible for topic input
- **Output Format** radio (Paragraph Summary/Important Points/Key Highlights)
- **Summary Length** radio (Auto/Brief/Medium/Detailed) - only for Paragraph Summary
- **Generate button** with conflict check (both PDF and text), progress bar (4 stages), status messages
- **PDF handling** - 10MB validation, cached text re-use, form data upload
- **Rate limit handling** - detection, dialog with wait time, model switching UI
- **Recently Generated Notes** - close button, metadata bar (4 metrics), dark markdown container
- **Research/Vision data** - collapsible figures with base64 images, cleaned descriptions
- **Action buttons** - Download (.md) / Copy to Clipboard / Save to KB / Edit
- **Inline editing** - textarea with undo/redo stacks, Save/Undo/Redo/Cancel
- **Undo Last Edit** in view mode
- **Save to KB form** - auto-filled fields, PDF duplicate detection, validation
- **Copy to clipboard** - use `navigator.clipboard.writeText()` (improvement over Streamlit text_area workaround)
- **History Note Viewer** - identical features to recent notes (metadata, edit, download, copy, save)

### 5.3 Search Knowledge Base Page

- **Two modes** via radio/toggle: "Browse Documents" / "Search Vector DB"
- **Browse** - keyword search, semantic document list, selectbox picker, info bar (topic/date/words), scrollable content, Download/Copy/View as Text
- **Search Vector DB** - query input, num results slider (1-10), similarity threshold slider (0.0-1.0), expandable results with score/chunk/metadata

### 5.4 Update Knowledge Base Page

- Manual document form: Title, Topic (default "general"), Source (default "manual"), Content textarea, submit

### 5.5 Study Mode (3 sub-tabs)

**Flashcards:**
- 2-column: left (generate + load + Anki export), right (study cards)
- Generate: topic, card count slider (5-20), content textarea, spinner
- Load: selectbox of existing sets
- Export: specific set or all, download links with card counts
- Study: progress bar, card N of M, gradient cards (purple Q / green A), Got it!/Review Again with API tracking, Previous/Shuffle/Next nav

**Quizzes:**
- 2-column: left (generate + load), right (quiz)
- Generate: topic, question count (3-15), content textarea
- Load: selectbox of existing quizzes
- Taking: questions with radio options, Submit (validates all answered)
- Results: color-coded score (green/blue/yellow), correct/incorrect counts, detailed review per question with green/red highlighting + explanation, Try Again/New Quiz

**Progress:**
- Overall stats (4 metrics)
- Study Streak (current + longest)
- Accuracy (flashcard % + quiz %)
- Topic Mastery (top 5 with level + note count)
- This Week (notes, flashcards, active days/7)
- Recent Activity (last 5 with summary + date)

### 5.6 Session State Mapping

| Streamlit `session_state` | Next.js |
|---|---|
| `note_history` (last 6) | `useState` (in-memory, resets on reload - same behavior) |
| `last_generated_notes`, `last_vision_data`, `last_generation_metadata` | `useState` in Generate Notes page |
| `edit_mode_*`, `edit_undo_stack_*`, `edit_redo_stack_*` | `useReducer` with undo/redo |
| `current_flashcard_set`, `current_card_index`, `show_answer` | `useState` in Flashcard component |
| `current_quiz`, `quiz_attempt_id`, `quiz_answers`, `quiz_submitted` | `useState` in Quiz component |
| `font_size` | `useLocalStorage` (persists - improvement) |
| `dismissed_welcome`, `first_time_user` | `useLocalStorage` |
| PDF cache (`last_pdf_*`) | `useRef` (in-memory) |

---

## 6. Deployment

### 6.1 Architecture

```
Netlify (Next.js)  ──HTTPS──>  Railway (FastAPI)  ──>  Supabase (DB + Auth + Storage)
```

### 6.2 Frontend (Netlify)

```toml
# netlify.toml
[build]
  command = "npm run build"
  publish = ".next"

[[plugins]]
  package = "@netlify/plugin-nextjs"

[build.environment]
  NEXT_PUBLIC_SUPABASE_URL = ""
  NEXT_PUBLIC_SUPABASE_ANON_KEY = ""
  NEXT_PUBLIC_API_BASE_URL = ""
```

### 6.3 Backend (Railway)

Dockerfile-based deployment. `sentence-transformers` model cached in Docker layer. Health check endpoint for monitoring. Environment variables in Railway dashboard.

### 6.4 Environment Variables

```
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# LLM
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_PROVIDER=groq
ENABLE_LOCAL_FALLBACK=false

# Backend
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=https://your-app.netlify.app,http://localhost:3000

# Frontend
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_PUBLIC_API_BASE_URL=https://your-api.railway.app/api/v1
```

---

## 7. Difficulty Assessment

| Work Item | Difficulty | % of Total |
|---|---|---|
| Supabase schema + migrations | Easy | 5% |
| `src/db/` store layer (4 stores + client + auth + storage) | Medium | 15% |
| Backend refactoring (swap stores, add auth, CORS, flag) | Medium | 10% |
| Next.js frontend (full Streamlit parity) | **Hard** | 45% |
| Supabase Auth integration (frontend + backend) | Medium | 8% |
| pgvector search (embedding gen + RPC queries) | Medium | 7% |
| Deployment config (Dockerfile, netlify.toml, env) | Easy | 3% |
| Testing & integration | Medium | 7% |

**Overall difficulty: Hard.** Frontend rewrite is the dominant effort. Backend changes are moderate since the agent pipeline carries over unchanged.
