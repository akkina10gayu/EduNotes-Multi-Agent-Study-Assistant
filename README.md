# EduNotes - Multi-Agent Study Assistant

A production-ready multi-agent educational assistant that creates structured study notes from various sources using RAG (Retrieval-Augmented Generation) and advanced NLP models.

## 🚀 What is EduNotes?

EduNotes is an intelligent study companion that automatically generates comprehensive, structured notes from multiple sources:

- **📚 Knowledge Base Search**: Retrieves information from your local knowledge repository
- **🌐 Web Content Processing**: Scrapes and summarizes educational articles and blogs
- **📝 Text Summarization**: Creates key points from direct text input
- **🎯 Smart Note Generation**: Produces well-organized markdown notes with bullet points

## 🤖 Multi-Agent Architecture

EduNotes uses a coordinated multi-agent system where specialized AI agents work together:

### **Orchestrator Agent**
- **Role**: Coordinates all other agents and manages the workflow
- **Function**: Detects input type (topic/URL/text) and routes to appropriate agents

### **Retriever Agent** 
- **Role**: Searches the local knowledge base using semantic similarity
- **Technology**: ChromaDB vector database with sentence transformers
- **Function**: Finds relevant documents based on user queries

### **Scraper Agent**
- **Role**: Extracts content from web URLs
- **Technology**: newspaper3k with BeautifulSoup fallback
- **Function**: Intelligently scrapes educational content from websites

### **Summarizer Agent**
- **Role**: Creates concise summaries and extracts key points
- **Technology**: Google's Flan-T5-small model (CPU-optimized)
- **Function**: Generates bullet-point summaries from long texts

### **Note-Maker Agent**
- **Role**: Formats final structured study notes
- **Function**: Creates clean markdown notes with metadata and sources

## ✨ Key Features

- **🧠 Intelligent Query Processing**: Automatically detects whether input is a topic, URL, or direct text
- **📊 Vector Search**: Semantic search through your knowledge base using embeddings
- **⚡ Fast Processing**: CPU-optimized models for quick note generation
- **🎨 Clean UI**: Intuitive Streamlit interface with real-time processing
- **💾 Persistent Storage**: ChromaDB for efficient document storage and retrieval
- **🔄 Automatic KB Updates**: Web-scraped content automatically added to knowledge base
- **📤 Export Options**: Download notes as markdown files
- **🛡️ Rate Limiting**: Built-in API protection and error handling

## 🛠️ Technology Stack

- **Framework**: LangChain for agent orchestration
- **Vector Database**: ChromaDB with SQLite backend
- **ML Models**: 
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (80MB)
  - Summarization: `google/flan-t5-small` (242MB)
- **Backend**: FastAPI with automatic OpenAPI documentation
- **Frontend**: Streamlit for interactive web interface
- **Caching**: DiskCache for performance optimization
- **Web Scraping**: newspaper3k with BeautifulSoup fallback

## 📋 Prerequisites

- **Python**: 3.9 or higher (3.10 recommended)
- **Memory**: 4GB RAM minimum (8GB recommended) 
- **Storage**: 5GB free disk space
- **Internet**: Required for initial setup and web scraping

## 🚀 Quick Setup

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd EduNotes

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: This will download ~1.5GB of ML models on first run.

### 2. Initialize Knowledge Base

```bash
# Initialize the vector database
python scripts/setup_kb.py --init

# Add sample educational content (recommended)
python scripts/seed_data.py --sample

# Optional: Fetch additional content from web
python scripts/seed_data.py --web --topics machine-learning deep-learning
```

### 3. Start the Application

**Terminal 1 - Start API Server:**
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Start Web Interface:**
```bash
streamlit run ui/streamlit_app.py
```

### 4. Access the Application

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/api/v1/health

## 📚 How to Use

### Generate Notes from Topics
1. Enter a topic like "machine learning" or "neural networks"
2. The system searches your knowledge base
3. Generates structured notes with key points
4. Download as markdown file

### Process Web Articles
1. Paste any educational blog/article URL
2. System automatically scrapes and summarizes content
3. Content gets added to your knowledge base
4. Receive structured notes with source links

### Summarize Direct Text
1. Paste long text (research papers, articles, etc.)
2. System extracts key points and creates summary
3. Get organized notes in markdown format

### Search Knowledge Base
1. Use the "Search Knowledge Base" tab
2. Enter keywords to find relevant documents
3. Adjust similarity threshold and result count
4. Browse matching content with scores

### Update Knowledge Base
1. Go to "Update Knowledge Base" tab
2. Manually add documents with title, topic, and content
3. System processes and adds to searchable database

## 🔧 Configuration

Key settings in `.env` file:

```env
# API Configuration
API_PORT=8000
DEBUG_MODE=False

# Model Settings (CPU-optimized)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SUMMARIZATION_MODEL=google/flan-t5-small

# Knowledge Base Settings
KB_CHUNK_SIZE=512
KB_CHUNK_OVERLAP=50

# Performance Settings
RATE_LIMIT_PER_MINUTE=100
MAX_CONCURRENT_REQUESTS=10
```

## 📁 Project Structure

```
EduNotes/
├── src/
│   ├── agents/          # Multi-agent implementations
│   ├── api/             # FastAPI backend
│   ├── knowledge_base/  # Vector store management
│   ├── models/          # Pydantic schemas
│   └── utils/           # Utilities (logging, caching)
├── ui/                  # Streamlit interface
├── scripts/             # Setup and data scripts
├── config/              # Configuration files
├── data/                # Knowledge base storage
├── models/              # Downloaded ML models
├── cache/               # Performance caching
└── logs/                # Application logs
```

## 🚨 Troubleshooting

### Common Issues

**Models not downloading:**
```bash
# Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

**ChromaDB errors:**
```bash
# Reset knowledge base
python scripts/setup_kb.py --reset
```

## 🎯 Use Cases

- **Students**: Generate study notes from research papers and online articles
- **Researchers**: Quickly summarize and organize academic papers
- **Professionals**: Create knowledge bases from industry documentation  
- **Content Creators**: Extract key points from various educational sources
- **Lifelong Learners**: Build personal knowledge repositories with smart search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please ensure responsible use of the scraping features and respect website terms of service.

---

**EduNotes v1.0** - Transforming how you create and manage study materials with AI-powered multi-agent assistance.