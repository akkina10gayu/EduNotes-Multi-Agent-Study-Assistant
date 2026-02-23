# EduNotes v2 — Setup & Deployment Guide

Step-by-step instructions to get EduNotes v2 running locally and deployed to production.

---

## Architecture

```
Netlify (Next.js frontend)  ──HTTPS──>  Render (FastAPI backend)  ──>  Supabase (DB + Auth + Storage)
```

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| **Supabase** | Postgres DB, pgvector, Auth, Storage | Yes (500MB DB, 1GB storage) |
| **Render** | FastAPI backend (Docker) | Yes (spins down after inactivity) |
| **Netlify** | Next.js frontend (SSR) | Yes (100GB bandwidth) |
| **Groq** | LLM API (Llama 3.3 70B, Llama 4 Scout) | Yes (rate limited) |

---

## Environment Variables Reference

### Backend (Render / local `.env`)

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | Yes | `https://abc123.supabase.co` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | `eyJhbGciOi...` | Supabase anon/public key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | `eyJhbGciOi...` | Supabase service role key (bypasses RLS) |
| `SUPABASE_JWT_SECRET` | Yes | `super-secret-jwt-token` | Supabase JWT secret for token validation |
| `GROQ_API_KEY` | Yes | `gsk_abc123...` | Groq API key from console.groq.com |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Primary LLM model (default shown) |
| `GROQ_VISION_MODEL` | No | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision model for research mode |
| `LLM_PROVIDER` | No | `groq` | LLM provider (default: groq) |
| `ENABLE_LOCAL_FALLBACK` | No | `false` | Enable Flan-T5/BART local fallback (default: false) |
| `API_HOST` | No | `0.0.0.0` | Server bind host |
| `API_PORT` | No | `8000` | Server bind port |
| `CORS_ORIGINS` | Yes | `https://edunotes.netlify.app,http://localhost:3000` | Comma-separated allowed origins |

### Frontend (Netlify / local `.env.local`)

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | `https://abc123.supabase.co` | Same as backend SUPABASE_URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | `eyJhbGciOi...` | Same as backend SUPABASE_ANON_KEY |
| `NEXT_PUBLIC_API_BASE_URL` | Yes | `https://edunotes-api.onrender.com/api/v1` | Backend API URL (include `/api/v1`) |

---

## Step 1: Set Up Supabase

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Wait for the project to finish provisioning

### Run Migrations

Go to **SQL Editor** in the Supabase dashboard and run these two files in order:

1. **`supabase/migrations/001_initial_schema.sql`** — Creates all 10 tables with RLS policies
2. **`supabase/migrations/002_search_function.sql`** — Creates the pgvector search RPC function

### Enable pgvector

Go to **Database > Extensions**, search for `vector`, and enable it. (The migration enables it too, but verify it's active.)

### Create Storage Buckets

Go to **Storage** and create two buckets:
- `pdfs` — Set to **Private**
- `exports` — Set to **Private**

### Collect Your Keys

Go to **Settings > API** and note down:

| What | Where to Find |
|------|--------------|
| Project URL | Settings > API > Project URL |
| Anon Key | Settings > API > Project API keys > `anon` `public` |
| Service Role Key | Settings > API > Project API keys > `service_role` `secret` |
| JWT Secret | Settings > API > JWT Settings > JWT Secret |

### Configure Auth

Go to **Authentication > URL Configuration**:
- **Site URL**: Set to your Netlify URL (or `http://localhost:3000` for local dev)
- **Redirect URLs**: Add your Netlify URL and `http://localhost:3000`

---

## Step 2: Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Go to **API Keys** and create a new key
4. Copy the key — you'll need it for the backend

---

## Step 3: Local Development

### Backend

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and fill in your Supabase + Groq keys
# Set CORS_ORIGINS=http://localhost:3000

# Start the server
uvicorn src.api.app:app --reload --port 8000
```

Verify: Open `http://localhost:8000/api/v1/health` — should return `{"status": "ok"}`

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local
cat > .env.local << 'EOF'
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...your-anon-key
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
EOF
# Edit .env.local with your actual values

# Start dev server
npm run dev
```

Verify: Open `http://localhost:3000` — should show the login page

### Test the Full Flow

1. Sign up with email/password on the login page
2. Check your email for the confirmation link (Supabase sends it)
3. After confirming, log in
4. Try generating notes from a topic
5. Check Knowledge Base, Study Mode, and Progress pages

---

## Step 4: Deploy Backend to Render

1. Push this repo to GitHub (if not already)
2. Go to [render.com](https://render.com) > **New > Web Service**
3. Connect your GitHub repo: `akkina10gayu/EduNotes-Multi-Agent-Study-Assistant`
4. Set **Branch** to `edunotesv2-migration`
5. Set **Root Directory** to `backend`
6. Render will auto-detect the Dockerfile
7. Add all **Environment Variables** from the table above
   - Set `CORS_ORIGINS` to `http://localhost:3000` for now (update after Netlify deploy)
8. Click **Deploy**
9. Wait for the build (first build takes ~5-10 min due to `sentence-transformers` model download)
10. Verify: Visit `https://your-service.onrender.com/api/v1/health`

Note your Render URL (e.g., `https://edunotes-api.onrender.com`).

---

## Step 5: Deploy Frontend to Netlify

1. Go to [netlify.com](https://netlify.com) > **Add new site > Import from Git**
2. Connect your GitHub repo: `akkina10gayu/EduNotes-Multi-Agent-Study-Assistant`
3. Set **Branch** to `edunotesv2-migration`
4. Set **Base directory** to `frontend`
5. Build command: `npm run build` (should auto-detect)
6. Publish directory: `frontend/.next`
7. Add **Environment Variables**:
   - `NEXT_PUBLIC_SUPABASE_URL` = your Supabase project URL
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = your Supabase anon key
   - `NEXT_PUBLIC_API_BASE_URL` = `https://your-service.onrender.com/api/v1`
8. Click **Deploy**

Note your Netlify URL (e.g., `https://edunotes.netlify.app`).

---

## Step 6: Connect Everything

After both services are deployed:

1. **Update Render CORS**: Go to Render > your service > Environment, update `CORS_ORIGINS` to your Netlify URL:
   ```
   https://edunotes.netlify.app
   ```
   (Add `http://localhost:3000` too if you still want local dev to work: `https://edunotes.netlify.app,http://localhost:3000`)

2. **Update Supabase Auth URLs**: Go to Supabase > Authentication > URL Configuration:
   - Set **Site URL** to your Netlify URL
   - Add your Netlify URL to **Redirect URLs**

3. **Redeploy Render** to pick up the CORS change

---

## Step 7: Verify Production

- [ ] Visit your Netlify URL
- [ ] Sign up with email/password
- [ ] Confirm email
- [ ] Log in
- [ ] Generate notes from a topic (try "Machine Learning")
- [ ] Save notes to Knowledge Base
- [ ] Browse Knowledge Base
- [ ] Search Knowledge Base semantically
- [ ] Generate flashcards
- [ ] Review flashcards
- [ ] Generate a quiz
- [ ] Take the quiz
- [ ] Check Progress page
- [ ] Test font size toggle in sidebar
- [ ] Test sign out

---

## Troubleshooting

### "Missing SUPABASE_URL" error on build
Make sure environment variables are set in Netlify/Render dashboard, not just locally.

### CORS errors in browser console
Check that `CORS_ORIGINS` on Render matches your exact Netlify URL (no trailing slash).

### "Token expired" or 401 errors
The Supabase JWT expires after 1 hour by default. The frontend auto-refreshes tokens, but if you see persistent 401s, check that `SUPABASE_JWT_SECRET` on the backend matches the one in your Supabase project.

### Backend health check fails on Render
First deploy takes ~5-10 min. The free tier spins down after 15 min of inactivity — first request after sleep takes ~30-60s.

### pgvector search returns no results
Make sure you ran `002_search_function.sql` and that documents have been added (chunks need embeddings generated by the backend).

### Render build fails on sentence-transformers
The Dockerfile pre-downloads the `all-MiniLM-L6-v2` model at build time. This needs ~500MB. Make sure Render's free tier has enough disk space.
