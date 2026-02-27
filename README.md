# EduNotes

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A multi-agent study assistant that transforms topics, articles, PDFs, and documents into structured notes, flashcards, and quizzes — powered by free LLM APIs.

## What It Does

- **Generate Notes** from any topic, URL, PDF, or pasted text with customizable formats and length
- **Research Mode** analyzes plots, figures, and tables from PDFs, and finds related academic references automatically
- **Create Flashcards** for active recall practice (exportable to Anki)
- **Take Quizzes** to test your understanding with auto-generated questions
- **Build a Knowledge Base** that grows with your learning
- **Search & Download** from your knowledge base — find stored notes and download them as text file
- **Track Progress** with streaks and performance stats
- **Free to Use** with Groq (primary) and HuggingFace (backup) APIs — no credit card required

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

Get a free API key from [console.groq.com](https://console.groq.com) and add it to `.env`:

```
GROQ_API_KEY=your_key_here
```

Alternatively, get a free HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set `LLM_PROVIDER=huggingface` and `HF_TOKEN=your_token` in `.env`.

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

2. **Study Mode** — Generate flashcards and quizzes from your notes. Review flashcards with accuracy tracking, take quizzes with instant feedback, and export to Anki.

3. **Knowledge Base** — Browse stored documents, search semantically, add new content, and download documents.

4. **Progress** — Track your learning streak, view topic rankings, and monitor weekly activity.

## Output Formats

| Format | Description |
|--------|-------------|
| Paragraph Summary | Flowing paragraphs (brief, medium, or detailed) |
| Important Points | Numbered key points, each self-contained |
| Key Highlights | Essential terms with concise definitions |
| Research Mode | All of the above plus analysis of plots, figures, and tables from PDFs with related academic references |

## Architecture

EduNotes uses a multi-agent pipeline where specialized agents handle different tasks, coordinated by a central orchestrator.

### How It Works

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

### Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Routes input to the right pipeline and coordinates all agents |
| **WebSearchAgent** | Generates optimized search queries, finds web pages, ranks them by educational value, scrapes the best ones, and filters out low-quality content |
| **ContentAgent** | Identifies content type (academic, tutorial, etc.), applies tailored processing instructions, delegates to the Summarizer, and checks output quality |
| **Retriever** | Searches the knowledge base using semantic similarity |
| **Scraper** | Extracts readable content from web pages |
| **Summarizer** | Generates summaries via LLM with automatic fallback if rate-limited |
| **NoteMaker** | Structures the final output into formatted markdown notes |

### Multi-Model Setup

Three separate models with independent rate limits to maximize free-tier availability:

| Model | What It Handles |
|-------|-----------------|
| Llama 3.3 70B | Main summarization — generates the actual notes |
| Llama 3.1 8B | Lightweight tasks — optimizing search queries, ranking web results, classifying content type, and checking output quality |
| Llama 4 Scout 17B | Vision — analyzing plots, figures, tables, and equations from PDF pages |

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (Llama 3.3 70B / 3.1 8B / 4 Scout) + HuggingFace fallback |
| Vision | Llama 4 Scout 17B via Groq (Research Mode) |
| Web Search | DuckDuckGo + Google fallback |
| PDF Processing | pymupdf4llm + PyPDF2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| Caching | DiskCache |
| Academic Search | OpenAlex + arXiv + Semantic Scholar APIs |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API errors | Add `GROQ_API_KEY` to `.env`, or set `LLM_PROVIDER=huggingface` with `HF_TOKEN` |
| Rate limit reached | Wait for daily reset, or the app auto-falls back to a lighter model |
| Port conflicts | Stop processes on ports 8000 or 8501 |
| Database issues | Run `python scripts/setup_kb.py --reset` |
| PDF extraction fails | Ensure `pymupdf4llm` is installed: `pip install pymupdf4llm` |

## License

MIT License. Please respect website terms of service when processing external content.
