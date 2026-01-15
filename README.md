# EduNotes - AI-Powered Study Assistant

A multi-agent educational assistant that creates structured study notes and interactive learning materials from any source using RAG (Retrieval-Augmented Generation) and state-of-the-art LLMs.

## 🚀 What is EduNotes?

EduNotes is your intelligent study companion that helps you learn faster and retain more:

- **📚 Smart Note Generation**: Automatically creates comprehensive study notes from topics, URLs, or text
- **🌐 Web Content Processing**: Scrapes and summarizes educational articles and blogs
- **🃏 AI Flashcards**: Generates interactive flashcards from your notes for active recall practice
- **📋 Adaptive Quizzes**: Creates personalized quizzes to test your understanding
- **📊 Progress Tracking**: Monitor your study habits, streaks, and mastery levels

## 🤖 How It Works

EduNotes uses an intelligent async pipeline that coordinates four specialized agents:

**1. Retriever Agent** → Searches your knowledge base using AI-powered semantic similarity (ChromaDB + MiniLM embeddings)

**2. Scraper Agent** → Extracts clean content from web URLs using newspaper3k with smart fallback to BeautifulSoup

**3. Summarizer Agent** → Generates summaries using hybrid LLM approach:
   - **Primary**: Groq API with Llama-3.1-70B (free, fast, high-quality)
   - **Fallback**: Local Flan-T5 model (works offline)

**4. Note-Maker Agent** → Formats everything into structured markdown notes with metadata and sources

The system automatically detects your input type (topic/URL/text), routes it through the right agents, and delivers comprehensive study materials in seconds.

## ✨ Key Features

**Core Capabilities:**
- **🧠 Smart Input Detection**: Automatically detects topics, URLs, or direct text input
- **📊 Semantic Search**: AI-powered vector search through your knowledge base
- **⚡ Hybrid LLM**: Free cloud API (Groq) for speed, local models for offline use
- **🔄 Auto KB Updates**: Web content automatically added to your knowledge base

**Study Features:**
- **🃏 Flashcard Generation**: AI creates question-answer pairs from your notes
- **📋 Quiz Mode**: Generate multiple-choice quizzes with instant feedback
- **📊 Progress Dashboard**: Track study streaks, accuracy, and topic mastery
- **📤 Export Options**: Download notes and flashcards as markdown

**Technical:**
- **🎨 Clean UI**: 4-tab Streamlit interface (Generate, Search, Update, Study Mode)
- **💾 Persistent Storage**: ChromaDB vector database with SQLite backend
- **🛡️ Production Ready**: Rate limiting, caching, error handling, logging

## 🛠️ Technology Stack

- **LLM Strategy**: Hybrid approach for best of both worlds
  - **Primary**: Groq API (Llama-3.1-70B) - Free, fast, high-quality
  - **Fallback**: Local Flan-T5-base (990MB) - Works offline
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (80MB, local)
- **Vector Store**: ChromaDB with persistent SQLite backend
- **Framework**: LangChain for agent coordination
- **Backend**: FastAPI with OpenAPI docs and rate limiting
- **Frontend**: Streamlit with 4-tab interface
- **Web Scraping**: newspaper3k + BeautifulSoup fallback
- **Caching**: DiskCache for performance

## 📋 Prerequisites

- **Python**: 3.9 or higher (3.10 recommended)
- **Memory**: 4GB RAM minimum (8GB recommended) 
- **Storage**: 5GB free disk space
- **Internet**: Required for initial setup and web scraping

## 🚀 Quick Setup

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd EduNotes

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Get Free API Key (Optional but Recommended)

For best performance, get a free Groq API key (no credit card required):

1. Visit https://console.groq.com
2. Sign up with your email
3. Copy your API key
4. Add to `.env` file:
```env
GROQ_API_KEY=your_key_here
```

> **Note**: Without API key, system automatically uses local models (slower but works offline)

### 3. Initialize Knowledge Base

```bash
python scripts/setup_kb.py --init
python scripts/seed_data.py --sample  # Adds 30+ educational documents
```

### 4. Start the Application

Open two terminals:

**Terminal 1 - API Server:**
```bash
uvicorn src.api.app:app --reload
```

**Terminal 2 - Web Interface:**
```bash
streamlit run ui/streamlit_app.py
```

### 5. Open Your Browser

- **Main App**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📚 How to Use

### Tab 1: Generate Notes
**From Topics:** Type "machine learning" → Get comprehensive notes from your knowledge base

**From URLs:** Paste any article URL → System scrapes, summarizes, and adds to KB

**From Text:** Paste long text → Get structured summary with key points

All notes can be downloaded as markdown files.

### Tab 2: Search Knowledge Base
Find specific information in your KB with semantic search. Adjust similarity threshold to control relevance.

### Tab 3: Update Knowledge Base
Manually add documents to expand your knowledge base. Enter title, topic, and content.

### Tab 4: Study Mode

**Flashcards:**
- Generate flashcards from any content
- Interactive flip cards with Q&A format
- Track review count and accuracy
- Shuffle and filter by difficulty

**Quizzes:**
- AI generates multiple-choice questions
- Instant feedback on answers
- Score tracking with explanations
- Review incorrect answers

**Progress Dashboard:**
- Study streaks and daily activity
- Topic mastery levels
- Accuracy metrics for flashcards and quizzes
- Recent activity feed

## 🔧 Configuration

Key settings in `.env` file:

```env
# LLM Configuration (Use free API or local models)
LLM_PROVIDER=groq              # groq, huggingface, or local
GROQ_API_KEY=your_key_here     # Free from console.groq.com
USE_LOCAL_MODEL=false          # Set true for offline mode

# Local Models (Fallback)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SUMMARIZATION_MODEL=google/flan-t5-base

# Knowledge Base
KB_CHUNK_SIZE=512
KB_CHUNK_OVERLAP=50

# Study Features
MAX_FLASHCARDS_PER_NOTE=20
QUIZ_QUESTIONS_COUNT=10
```

## 📁 Project Structure

```
EduNotes/
├── src/
│   ├── agents/          # 4 specialized agents (retriever, scraper, summarizer, note-maker)
│   ├── api/             # FastAPI backend + study feature routes
│   ├── knowledge_base/  # ChromaDB vector store
│   ├── models/          # Data models (notes, flashcards, quizzes, progress)
│   └── utils/           # LLM client, generators, storage, caching
├── ui/                  # Streamlit app (4 tabs)
├── scripts/             # Setup and seed scripts
├── data/
│   ├── knowledge_base/  # Document storage
│   ├── flashcards/      # Flashcard sets
│   ├── quizzes/         # Quiz data
│   └── progress/        # Study progress tracking
├── config/              # Settings and configuration
└── logs/                # Application logs
```

## 🚨 Troubleshooting

**"API key not found" error:**
- Add `GROQ_API_KEY` to `.env` file, OR
- Set `USE_LOCAL_MODEL=true` to use offline mode

**Port 8000/8501 already in use:**
```bash
# Kill the process using the port
netstat -ano | findstr :8000    # Windows
lsof -i :8000                   # Mac/Linux
```

**ChromaDB errors:**
```bash
python scripts/setup_kb.py --reset  # Reinitialize database
```

**Slow performance:**
- Get a free Groq API key (10x faster than local models)
- Reduce `KB_CHUNK_SIZE` in `.env`

## 🎯 Use Cases

- **Students**: Auto-generate notes, flashcards, and quizzes from lectures and readings
- **Researchers**: Summarize academic papers and build searchable knowledge bases
- **Professionals**: Create and maintain technical documentation libraries
- **Self-Learners**: Track progress, build study habits, and master new topics
- **Educators**: Quickly create study materials and assessments from content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please ensure responsible use of the scraping features and respect website terms of service.

---

**EduNotes v2.0** - Your AI-powered study companion with smart note generation, flashcards, quizzes, and progress tracking.