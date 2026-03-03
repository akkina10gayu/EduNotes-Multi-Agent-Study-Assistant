# EduNotes

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A multi-agent study assistant that transforms topics, articles, PDFs, and documents into structured notes, flashcards, and quizzes — with an AI chat for interactive learning — powered by free LLM APIs.

## What It Does

- **Generate Notes** from any topic, URL, PDF, or pasted text with customizable formats and length
- **Chat with AI** in 7 modes — free-form chat, exam answer writing, concept explanations, topic comparison, Socratic tutoring, research paper analysis, and guided research writing
- **Research Mode** analyzes plots, figures, and tables from PDFs, and finds related academic references automatically
- **Create Flashcards** for active recall practice (exportable to Anki)
- **Take Quizzes** to test your understanding with auto-generated questions
- **Build a Knowledge Base** that grows with your learning
- **Search & Download** from your knowledge base — find stored notes and download them as text file
- **Track Progress** with streaks and performance stats
- **Free to Use** with Groq, Gemini, and Cerebras free-tier APIs — no credit card required

## Quick Start

### Prerequisites
- Python 3.9 or higher
- 4GB RAM minimum

### Installation

```bash
git clone <repo-url>
cd EduNotes
python -m venv venv

# Activate environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install and initialize
pip install -r requirements.txt
python scripts/setup_kb.py --init
python scripts/seed_data.py --sample
```

### API Key Setup

Add your API keys to `.env`:

```
# Note generation (required)
GROQ_API_KEY=your_key_here

# Chat with AI (at least one required for chat)
GEMINI_API_KEY=your_key_here
CEREBRAS_API_KEY=your_key_here
```

- **Groq** (free) — [console.groq.com](https://console.groq.com) — powers note generation, flashcards, and quizzes
- **Gemini** (free) — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — primary provider for AI Chat
- **Cerebras** (free) — [cloud.cerebras.ai](https://cloud.cerebras.ai) — fallback for AI Chat if Gemini is unavailable

Alternatively, for note generation you can use HuggingFace instead of Groq: set `LLM_PROVIDER=huggingface` and `HF_TOKEN=your_token` in `.env`.

### Run the Application

Open two terminal windows:

```bash
# Terminal 1 - Start API
uvicorn src.api.app:app --reload

# Terminal 2 - Start UI
streamlit run ui/streamlit_app.py
```

Visit [localhost:8501](http://localhost:8501) to start using the app.

## How to Use

1. **Generate Notes** — Enter a topic, paste a URL, upload a PDF, or paste text directly. Choose a search mode (Auto, KB Only, Web Search, or KB + Web) and select your preferred format. Enable Research Mode for academic papers and figure analysis. Edit, download, or copy notes inline.

2. **Chat with AI** — Pick a mode and start a conversation. Toggle Auto Search to let the assistant pull context from your knowledge base and the web automatically.
   - **Chat** — open-ended Q&A on any topic
   - **Exam Answer** — model answers in brief, standard, or detailed depth
   - **Explain Concept** — explanations at your level (ELI5 to advanced) in technical, analogy, or visual style
   - **Compare & Contrast** — structured side-by-side comparison of two topics
   - **Socratic Tutor** — guides you with questions instead of direct answers
   - **Paper Analysis** — paste a research paper and ask follow-up questions about it
   - **Research Writer** — conversational workflow that gathers context, builds an outline, and writes a paper section by section

   Conversations are saved automatically. Search past chats by keyword, load, export, or delete them from the Chat History panel. You can also edit or regenerate any message in the conversation.

3. **Study Mode** — Generate flashcards and quizzes from your notes. Review flashcards with accuracy tracking, take quizzes with instant feedback, and export to Anki.

4. **Knowledge Base** — Browse stored documents, search semantically, add new content, and download documents.

5. **Progress** — Track your learning streak, view topic rankings, and monitor weekly activity.

## Output Formats

| Format | Description |
|--------|-------------|
| Paragraph Summary | Flowing paragraphs (brief, medium, or detailed) |
| Important Points | Numbered key points, each self-contained |
| Key Highlights | Essential terms with concise definitions |
| Research Mode | All of the above plus analysis of plots, figures, and tables from PDFs with related academic references |

## Architecture

EduNotes uses a multi-agent architecture with two main pipelines — a note generation pipeline coordinated by an orchestrator, and a conversational chat pipeline for interactive learning.

### Notes Pipeline

```
                         User Input
                             |
              +--------------+--------------+
              |              |              |
           Topic          URL / PDF       Text
              |              |              |
              v              v              v
     +----------------+  +-----------+     |
     | Web Search     |  | Scraper / |     |
     | Agent          |  | PDF Proc. |     |
     | (build query,  |  | (extract  |     |
     |  search, rank, |  |  content) |     |
     |  scrape, check |  |           |     |
     |  quality)      |  |           |     |
     +-------+--------+  +-----+-----+     |
             |                  |           |
             +--------+---------+-----------+
                      |
                      v
              +---------------+
              | Content Agent |
              | (classify,    |
              |  strategize,  |
              |  summarize,   |
              |  evaluate)    |
              +-------+-------+
                      |
                      v
              +---------------+
              |  Note Maker   |
              | (format as    |
              |  markdown)    |
              +-------+-------+
                      |
                      v
              Formatted Notes
                      |
         +------------+------------+
         |   (if Research Mode)    |
         v                         v
  Academic References      Figures & Tables
  (OpenAlex, arXiv,        (Vision analysis
   Semantic Scholar)        of PDF pages)
```

### Chat Pipeline

```
                    User Message + Mode
                           |
                    +------+------+
                    |  ChatAgent  |
                    +------+------+
                           |
            +--------------+--------------+
            |                             |
      Mode Router                  Smart Context
      (chat, exam answer,          KB Search
       explain, compare,               |
       socratic, paper           Sufficiency Check
       analysis)                       |
            |                     Web Search
            |                    (if needed)
            +--------------+--------------+
                           |
                           v
                  +------------------+
                  | Gemini 2.5 Flash |
                  | (Cerebras / Groq |
                  |    fallback)     |
                  +--------+---------+
                           |
                +----------+----------+
                |          |          |
           Response     Sources  Suggestions
```

Research Writer follows a multi-stage flow within the chat:
Topic → Sufficiency Analysis → Follow-up Questions → Outline → Section Writing → Assembly.

### Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Routes input to the right pipeline and coordinates all agents |
| **ChatAgent** | Conversational AI with 7 modes, assembles context from KB and web, generates inline follow-up suggestions |
| **ResearchWriter** | Iterative information gathering with sufficiency analysis, builds outlines, and writes research papers section by section |
| **WebSearchAgent** | Generates optimized search queries, finds web pages, ranks them by educational value, scrapes the best ones, and filters out low-quality content |
| **ContentAgent** | Identifies content type (academic, tutorial, etc.), applies tailored processing instructions, delegates to the Summarizer, and checks output quality |
| **Retriever** | Searches the knowledge base using semantic similarity |
| **Scraper** | Extracts readable content from web pages |
| **Summarizer** | Generates summaries via LLM with automatic fallback if rate-limited |
| **NoteMaker** | Structures the final output into formatted markdown notes |

### Multi-Model Setup

Five models across three providers, each with independent rate limits to maximize free-tier availability:

**Note Generation (Groq)**

| Model | What It Handles |
|-------|-----------------|
| Llama 3.3 70B | Main summarization — generates the actual notes |
| Llama 3.1 8B | Lightweight tasks — optimizing search queries, ranking web results, classifying content type, and checking output quality |
| Llama 4 Scout 17B | Vision — analyzing plots, figures, tables, and equations from PDF pages |

**AI Chat (Gemini / Cerebras)**

| Model | What It Handles |
|-------|-----------------|
| Gemini 2.5 Flash | Primary — all chat modes, research writing, paper analysis |
| Llama 3.3 70B (Cerebras) | Fallback — used automatically when Gemini is unavailable |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Notes LLM | Groq API (Llama 3.3 70B / 3.1 8B / 4 Scout) + HuggingFace fallback |
| Chat LLM | Google Gemini 2.5 Flash + Cerebras Llama 3.3 70B fallback |
| Vision | Llama 4 Scout 17B via Groq (Research Mode) |
| Web Search | DuckDuckGo + Google fallback |
| PDF Processing | pymupdf4llm + PyPDF2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| Caching | DiskCache |
| Conversations | JSON file-based session storage |
| Academic Search | OpenAlex + arXiv + Semantic Scholar APIs |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API errors | Add `GROQ_API_KEY` to `.env`, or set `LLM_PROVIDER=huggingface` with `HF_TOKEN` |
| Chat not working | Add `GEMINI_API_KEY` to `.env`; optionally add `CEREBRAS_API_KEY` as fallback |
| Rate limit reached | Wait for daily reset, or the app auto-falls back to a lighter model |
| Port conflicts | Stop processes on ports 8000 or 8501 |
| Database issues | Run `python scripts/setup_kb.py --reset` |
| PDF extraction fails | Ensure `pymupdf4llm` is installed: `pip install pymupdf4llm` |

## License

MIT License. Please respect website terms of service when processing external content.
