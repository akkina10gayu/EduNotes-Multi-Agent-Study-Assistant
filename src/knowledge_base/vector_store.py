"""
Vector store management using ChromaDB with built-in ONNX embeddings.
No torch/transformers required — uses the same MiniLM-L6-v2 model via ONNX runtime.
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib
import time

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache TTL for topics (5 minutes) - Phase 3 optimization
TOPICS_CACHE_TTL = 300

class VectorStore:
    """Manages ChromaDB vector store for document embeddings"""

    def __init__(self):
        """Initialize vector store"""
        # ONNX embeddings (same MiniLM-L6-v2 model, no torch required)
        self.embedding_fn = ONNXMiniLM_L6_V2()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.KB_CHUNK_SIZE,
            chunk_overlap=settings.KB_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize ChromaDB client (PersistentClient auto-persists)
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
                "hnsw:search_ef": 100
            }
        )
        logger.info(f"Collection ready: {settings.CHROMA_COLLECTION} ({self.collection.count()} documents)")

        # Topics cache (Phase 3 optimization)
        self._topics_cache = None
        self._topics_cache_time = 0

    @staticmethod
    def _make_id(text: str, chunk_index: int) -> str:
        """Generate a deterministic ID for a text chunk."""
        return hashlib.md5(f"{text[:200]}:{chunk_index}".encode()).hexdigest()

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = None) -> bool:
        """Add documents to vector store"""
        try:
            batch_size = batch_size or settings.KB_UPDATE_BATCH_SIZE

            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Prepare texts, metadata, and IDs
                texts = []
                metadatas = []
                ids = []

                for doc in batch:
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(doc['content'])

                    for chunk_idx, chunk in enumerate(chunks):
                        texts.append(chunk)
                        ids.append(self._make_id(chunk, chunk_idx))
                        metadatas.append({
                            'source': doc.get('source', 'unknown'),
                            'title': doc.get('title', 'Untitled'),
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'url': doc.get('url', ''),
                            'date_added': doc.get('date_added', ''),
                            'topic': doc.get('topic', 'general')
                        })

                # Upsert to collection (handles duplicates gracefully)
                self.collection.upsert(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"Added batch of {len(texts)} chunks to vector store")

            # Invalidate topics cache (Phase 3 optimization)
            self._topics_cache = None
            self._topics_cache_time = 0
            logger.debug("Topics cache invalidated after adding documents")

            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search(self, query: str, k: int = 5, score_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            # Format results (ChromaDB returns parallel lists under [0])
            formatted_results = []
            if results and results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                distances = results['distances'][0]

                for content, metadata, distance in zip(docs, metas, distances):
                    if distance <= score_threshold:
                        formatted_results.append({
                            'content': content,
                            'metadata': metadata,
                            'score': distance
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
        """Get list of unique topics from the knowledge base (with caching)"""
        try:
            # Phase 3 optimization: Check cache first
            current_time = time.time()
            if self._topics_cache is not None and (current_time - self._topics_cache_time) < TOPICS_CACHE_TTL:
                logger.debug(f"Returning cached topics ({len(self._topics_cache)} topics)")
                return self._topics_cache

            # Cache miss or expired - fetch from database
            logger.info("Topics cache miss/expired, fetching from database...")

            # Get all documents with metadata
            results = self.collection.get()

            if not results or not results.get('metadatas'):
                self._topics_cache = []
                self._topics_cache_time = current_time
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

            # Update cache
            self._topics_cache = sorted_topics
            self._topics_cache_time = current_time

            logger.info(f"Found {len(sorted_topics)} unique topics in KB (cached for {TOPICS_CACHE_TTL}s)")
            return sorted_topics

        except Exception as e:
            logger.error(f"Error getting unique topics: {e}")
            return []
