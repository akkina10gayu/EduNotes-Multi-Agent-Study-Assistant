"""
Vector store management using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """Manages ChromaDB vector store for document embeddings"""
    
    def __init__(self):
        """Initialize vector store"""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.KB_CHUNK_SIZE,
            chunk_overlap=settings.KB_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(
                name=settings.CHROMA_COLLECTION
            )
            logger.info(f"Loaded existing collection: {settings.CHROMA_COLLECTION}")
        except:
            self.collection = self.client.create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "hnsw:search_ef": 100
                }
            )
            logger.info(f"Created new collection: {settings.CHROMA_COLLECTION}")
        
        # Initialize Langchain Chroma
        self.vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
            client=self.client
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = None) -> bool:
        """Add documents to vector store"""
        try:
            batch_size = batch_size or settings.KB_UPDATE_BATCH_SIZE
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare texts and metadata
                texts = []
                metadatas = []
                
                for doc in batch:
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(doc['content'])
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadatas.append({
                            'source': doc.get('source', 'unknown'),
                            'title': doc.get('title', 'Untitled'),
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'url': doc.get('url', ''),
                            'date_added': doc.get('date_added', ''),
                            'topic': doc.get('topic', 'general')
                        })
                
                # Add to vector store
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
                logger.info(f"Added batch of {len(texts)} chunks to vector store")
            
            # Persist changes
            self.vectorstore.persist()
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(self, query: str, k: int = 5, score_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by threshold and format results
            # Note: ChromaDB cosine similarity returns distance, lower scores = better matches
            formatted_results = []
            for doc, score in results:
                if score <= score_threshold:  # Lower scores are better for cosine distance
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score
                    })
            
            logger.info(f"Found {len(formatted_results)} relevant documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=settings.CHROMA_COLLECTION)
            logger.info(f"Deleted collection: {settings.CHROMA_COLLECTION}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': settings.CHROMA_COLLECTION,
                'persist_directory': str(settings.CHROMA_PERSIST_DIR)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def get_unique_topics(self) -> List[str]:
        """Get list of unique topics from the knowledge base"""
        try:
            # Get all documents with metadata
            results = self.collection.get()

            if not results or not results.get('metadatas'):
                return []

            # Extract unique topics
            topics = set()
            for metadata in results['metadatas']:
                if metadata and 'topic' in metadata:
                    topic = metadata['topic']
                    if topic:
                        # Clean up topic name
                        topic = topic.replace('-', ' ').replace('_', ' ').title()
                        topics.add(topic)

            # Sort alphabetically
            sorted_topics = sorted(list(topics))
            logger.info(f"Found {len(sorted_topics)} unique topics in KB")
            return sorted_topics

        except Exception as e:
            logger.error(f"Error getting unique topics: {e}")
            return []