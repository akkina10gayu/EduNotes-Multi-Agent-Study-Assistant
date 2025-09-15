"""
Configuration settings for EduNotes application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = os.getenv("API_VERSION", "v1")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
QA_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(MODELS_DIR / "downloaded")))

# LangChain Settings
LANGCHAIN_CACHE = os.getenv("LANGCHAIN_CACHE", "sqlite")
LANGCHAIN_CACHE_PATH = os.getenv("LANGCHAIN_CACHE_PATH", "./.langchain.db")
LANGCHAIN_VERBOSE = os.getenv("LANGCHAIN_VERBOSE", "False").lower() == "true"

# ChromaDB Settings
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "vector_db" / "chroma")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "edunotes_kb")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

# Knowledge Base
KB_PATH = Path(os.getenv("KB_PATH", str(DATA_DIR / "knowledge_base")))
KB_UPDATE_BATCH_SIZE = int(os.getenv("KB_UPDATE_BATCH_SIZE", 100))
KB_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", 512))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", 50))

# Scraping Settings
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", 10))
SCRAPER_RETRY_ATTEMPTS = int(os.getenv("SCRAPER_RETRY_ATTEMPTS", 3))
SCRAPER_USER_AGENT = os.getenv("SCRAPER_USER_AGENT", "EduNotes/1.0")

# Cache Settings
CACHE_EMBEDDING_TTL = int(os.getenv("CACHE_EMBEDDING_TTL", 86400))
CACHE_SUMMARY_TTL = int(os.getenv("CACHE_SUMMARY_TTL", 604800))
CACHE_SCRAPE_TTL = int(os.getenv("CACHE_SCRAPE_TTL", 3600))

# API Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 100))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 1000))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", str(LOGS_DIR / "edunotes.log")))

# Performance
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))