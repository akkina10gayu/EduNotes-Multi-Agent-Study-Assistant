"""
Retriever agent for searching knowledge base
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.knowledge_base.vector_store import VectorStore
from src.utils.cache_utils import cached
from config import settings

class RetrieverAgent(BaseAgent):
    """Agent for retrieving documents from knowledge base"""
    
    def __init__(self):
        super().__init__("RetrieverAgent")
        self.vector_store = VectorStore()
    
    @cached("retriever", ttl=settings.CACHE_EMBEDDING_TTL)
    def search_knowledge_base(self, query: str, k: int = 5, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant documents"""
        try:
            results = self.vector_store.search(
                query=query,
                k=k,
                score_threshold=threshold
            )
            
            if results:
                self.logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            else:
                self.logger.warning(f"No relevant documents found for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieval request"""
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))
            
            query = input_data.get('query', '')
            k = input_data.get('k', 5)
            threshold = input_data.get('threshold', 0.7)
            
            if not query:
                return self.handle_error(ValueError("No query provided"))
            
            # Search knowledge base
            results = self.search_knowledge_base(query, k, threshold)
            
            # Format response
            return {
                'success': True,
                'query': query,
                'results': results,
                'count': len(results),
                'agent': self.name
            }
            
        except Exception as e:
            return self.handle_error(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.vector_store.get_collection_stats()