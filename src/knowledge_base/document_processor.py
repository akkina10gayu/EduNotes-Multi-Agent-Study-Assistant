"""
Document processing for knowledge base
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib
from datetime import datetime

from config import settings
from src.utils.logger import get_logger
from src.utils.text_utils import clean_text, extract_keywords

logger = get_logger(__name__)

class DocumentProcessor:
    """Process and prepare documents for vector store"""
    
    def __init__(self):
        self.processed_dir = settings.KB_PATH / "processed"
        self.raw_dir = settings.KB_PATH / "raw"
        self.metadata_file = settings.KB_PATH / "metadata.json"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID from content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_document(self, content: str, source: str, title: str = None, 
                        url: str = None, topic: str = None) -> Optional[Dict[str, Any]]:
        """Process a single document"""
        try:
            # Clean content
            cleaned_content = clean_text(content)
            
            # Generate document ID
            doc_id = self._generate_doc_id(cleaned_content)
            
            # Check if document already exists
            if doc_id in self.metadata:
                logger.info(f"Document already exists: {doc_id}")
                return None
            
            # Extract keywords
            keywords = extract_keywords(cleaned_content)
            
            # Create document object
            document = {
                'id': doc_id,
                'content': cleaned_content,
                'source': source,
                'title': title or f"Document_{doc_id[:8]}",
                'url': url or '',
                'topic': topic or 'general',
                'keywords': keywords,
                'date_added': datetime.now().isoformat(),
                'word_count': len(cleaned_content.split())
            }
            
            # Save raw content
            raw_file = self.raw_dir / f"{doc_id}.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save processed document
            processed_file = self.processed_dir / f"{doc_id}.json"
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2)
            
            # Update metadata
            self.metadata[doc_id] = {
                'title': document['title'],
                'source': document['source'],
                'date_added': document['date_added'],
                'topic': document['topic'],
                'keywords': document['keywords'][:5]  # Store top 5 keywords
            }
            self._save_metadata()
            
            logger.info(f"Processed document: {document['title']}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        processed = []
        
        for doc in documents:
            result = self.process_document(
                content=doc['content'],
                source=doc.get('source', 'unknown'),
                title=doc.get('title'),
                url=doc.get('url'),
                topic=doc.get('topic')
            )
            
            if result:
                processed.append(result)
        
        logger.info(f"Processed {len(processed)} out of {len(documents)} documents")
        return processed
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all processed documents"""
        documents = []
        
        for file_path in self.processed_dir.glob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(json.load(f))
        
        return documents
    
    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search documents by topic"""
        documents = []
        
        for doc_id, meta in self.metadata.items():
            if meta.get('topic', '').lower() == topic.lower():
                file_path = self.processed_dir / f"{doc_id}.json"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents.append(json.load(f))
        
        return documents