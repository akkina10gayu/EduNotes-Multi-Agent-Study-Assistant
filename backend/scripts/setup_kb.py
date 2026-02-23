"""
Setup script for initializing the knowledge base
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.document_processor import DocumentProcessor
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

def init_knowledge_base(reset: bool = False):
    """Initialize the knowledge base"""
    try:
        logger.info("Initializing knowledge base...")
        
        # Create necessary directories
        settings.KB_PATH.mkdir(parents=True, exist_ok=True)
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        vector_store = VectorStore()
        
        if reset:
            logger.warning("Resetting knowledge base...")
            vector_store.delete_collection()
            vector_store = VectorStore()  # Reinitialize
        
        # Get stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Knowledge base initialized: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup EduNotes Knowledge Base")
    parser.add_argument("--init", action="store_true", help="Initialize knowledge base")
    parser.add_argument("--reset", action="store_true", help="Reset and reinitialize KB")
    
    args = parser.parse_args()
    
    if args.init or args.reset:
        success = init_knowledge_base(reset=args.reset)
        if success:
            print("✅ Knowledge base initialized successfully!")
        else:
            print("❌ Failed to initialize knowledge base")
            sys.exit(1)
    else:
        print("Usage: python setup_kb.py --init [--reset]")

if __name__ == "__main__":
    main()