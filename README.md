# EduNotes

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A smart study assistant that transforms topics, articles, and documents into structured notes, flashcards, and quizzes.

## What It Does

EduNotes takes your learning materials and creates study-ready content automatically:

- **Generate Notes** from any topic, URL, PDF, or pasted text (articles, research content, lengthy paragraphs) with customizable formats
- **Create Flashcards** for active recall practice (exportable to Anki)
- **Take Quizzes** to test your understanding
- **Build a Knowledge Base** that grows with your learning
- **Track Progress** with streaks and performance stats
- **Works Offline** using local models when no API key is configured

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

Get a free API key from [console.groq.com](https://console.groq.com) for faster performance. Add it to your `.env` file:

```
GROQ_API_KEY=your_key_here
```

The app works without it using local models, but responses will be slower.

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

1. **Generate Notes** - Enter a topic, paste a URL, or upload a PDF. Choose a search mode (Auto, KB Only, Web Search, or KB + Web Search) and select your preferred format. Edit, download, or copy generated notes inline.

2. **Search Knowledge Base** - Two modes:
   - **Browse Documents** - View full original documents stored in the KB. Search semantically by meaning, select from a dropdown, and view/download the raw text or open it in a new browser tab.
   - **Search Vector DB** - Semantic similarity search across document chunks for precise matching.

3. **Update Knowledge Base** - Add new documents with title, topic, and content.

4. **Study Mode** - Create flashcards from your notes, take quizzes, and monitor your learning streak.

## Output Formats

| Format | Description |
|--------|-------------|
| Paragraph Summary | Flowing paragraphs (brief, medium, or detailed) |
| Important Points | Numbered key points, each self-contained |
| Key Highlights | Essential terms with concise definitions |

## Architecture

EduNotes uses a multi-agent system with each specialized components handling different tasks:

| Component | Role |
|-----------|------|
| Retriever | Searches knowledge base using semantic similarity |
| Web Search | Finds and evaluates educational content from the web using LLM-powered search |
| Scraper | Extracts and processes content from URLs |
| Summarizer | Generates summaries using LLM |
| Note-Maker | Structures output into formatted notes |

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (Llama 3.3 70B) / Local Flan-T5 |
| Web Search | DuckDuckGo (ddgs) |
| Embeddings | MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API errors | Add `GROQ_API_KEY` to `.env` or set `USE_LOCAL_MODEL=true` |
| Port conflicts | Stop processes on ports 8000 or 8501 |
| Database issues | Run `python scripts/setup_kb.py --reset` |

## License

MIT License. Please respect website terms of service when processing external content.
