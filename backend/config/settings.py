"""
Configuration settings for EduNotes application
Version 2.0 - With FREE LLM API support
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

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# Feature flags
ENABLE_LOCAL_FALLBACK = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000")

# =============================================================================
# LLM API Settings (FREE APIs - No cost)
# =============================================================================
# Provider: groq (recommended), huggingface, local
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

# Groq API (FREE - 14,400 requests/day)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# HuggingFace API (FREE - 30,000 requests/month)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# =============================================================================
# Local Model Settings (Fallback when USE_LOCAL_MODEL=true)
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "google/flan-t5-base")
QA_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(MODELS_DIR / "downloaded")))

# Knowledge Base
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

# =============================================================================
# Web Search Settings
# =============================================================================
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", 10))
SEARCH_REGION = os.getenv("SEARCH_REGION", "wt-wt")
SEARCH_SAFE_SEARCH = os.getenv("SEARCH_SAFE_SEARCH", "moderate")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SEARCH_FALLBACK_ENABLED = os.getenv("SEARCH_FALLBACK_ENABLED", "true").lower() == "true"

# Web Search Agent
WEB_SEARCH_MAX_URLS_TO_SCRAPE = int(os.getenv("WEB_SEARCH_MAX_URLS_TO_SCRAPE", 3))
WEB_SEARCH_MIN_CONTENT_LENGTH = int(os.getenv("WEB_SEARCH_MIN_CONTENT_LENGTH", 200))
WEB_SEARCH_CACHE_TTL = int(os.getenv("WEB_SEARCH_CACHE_TTL", 3600))

# =============================================================================
# Study Features Settings
# =============================================================================
MAX_FLASHCARDS_PER_NOTE = int(os.getenv("MAX_FLASHCARDS_PER_NOTE", 20))
QUIZ_QUESTIONS_COUNT = int(os.getenv("QUIZ_QUESTIONS_COUNT", 10))