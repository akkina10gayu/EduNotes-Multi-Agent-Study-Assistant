"""
Retriever agent for searching knowledge base
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.db import document_store
from src.utils.cache_utils import cached
from config import settings

class RetrieverAgent(BaseAgent):
    """Agent for retrieving documents from knowledge base"""
    
    def __init__(self):
        super().__init__("RetrieverAgent")
    
    @cached("retriever", ttl=settings.CACHE_EMBEDDING_TTL)
    def search_knowledge_base(self, query: str, k: int = 5, threshold: float = 1.0, user_id: str = None) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant documents"""
        try:
            results = document_store.search(
                user_id=user_id,
                query=query,
                k=k,
                threshold=threshold
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
            user_id = input_data.get('user_id')

            if not query:
                return self.handle_error(ValueError("No query provided"))

            # Search knowledge base
            results = self.search_knowledge_base(query, k, threshold, user_id=user_id)
            
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
        """Get knowledge base statistics (user-independent summary)"""
        return {'status': 'active'}