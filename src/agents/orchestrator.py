"""
Main orchestrator for coordinating all agents
"""
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

from src.agents.retriever import RetrieverAgent
from src.agents.scraper import ScraperAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.note_maker import NoteMakerAgent
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.document_processor import DocumentProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryType(Enum):
    TOPIC = "topic"
    URL = "url"
    TEXT = "text"

class Orchestrator:
    """Main orchestrator for managing agent pipeline"""
    
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.scraper = ScraperAgent()
        self.summarizer = SummarizerAgent()
        self.note_maker = NoteMakerAgent()
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        self.logger = logger
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        # Check if it's a URL
        if query.startswith(('http://', 'https://', 'www.')):
            return QueryType.URL
        
        # Check if it's direct text (long input)
        if len(query) > 500:
            return QueryType.TEXT
        
        # Otherwise, it's a topic query
        return QueryType.TOPIC
    
    async def process_topic_query(self, query: str) -> Dict[str, Any]:
        """Process a topic-based query"""
        try:
            self.logger.info(f"Processing topic query: {query[:50]}...")
            
            # Search knowledge base
            retrieval_result = await self.retriever.process({
                'query': query,
                'k': 5,
                'threshold': 1.0
            })
            
            if retrieval_result['success'] and retrieval_result['count'] > 0:
                # Found relevant documents
                self.logger.info(f"Found {retrieval_result['count']} relevant documents")
                
                # Extract content from results
                documents = [doc['content'] for doc in retrieval_result['results']]
                sources = [
                    {
                        'title': doc['metadata'].get('title', 'Unknown'),
                        'url': doc['metadata'].get('url', '#')
                    }
                    for doc in retrieval_result['results']
                ]
                
                # Extract key points (primary focus)
                bullet_result = await self.summarizer.process({
                    'content': ' '.join(documents[:3]),  # Use top 3 documents
                    'mode': 'bullet_points',
                    'num_points': 8  # More key points for comprehensive notes
                })
                
                # Create concise summary for context only
                summary_result = await self.summarizer.process({
                    'content': ' '.join(documents[:2]),  # Shorter summary from top 2 docs
                    'mode': 'summary'
                })
                
                # Create notes (emphasize key points over summary)
                notes_result = await self.note_maker.process({
                    'mode': 'create',
                    'title': f"{query.title()}",
                    'summary': bullet_result.get('summary', ''),  # Use bullet points as main content
                    'key_points': [],  # Key points already in summary
                    'sources': sources,
                    'topic': query
                })
                
                return {
                    'success': True,
                    'query_type': 'topic',
                    'query': query,
                    'notes': notes_result.get('notes', ''),
                    'sources_used': len(sources),
                    'from_kb': True
                }
            
            else:
                # No relevant documents found - need to fetch from web
                self.logger.info("No relevant documents in KB, fetching from web...")
                return await self.fetch_and_process_topic(query)
                
        except Exception as e:
            self.logger.error(f"Error processing topic query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def fetch_and_process_topic(self, topic: str) -> Dict[str, Any]:
        """Fetch information about a topic from web and process"""
        try:
            # For now, return a message that KB needs updating
            # In production, this would trigger web scraping
            return {
                'success': True,
                'query_type': 'topic',
                'query': topic,
                'notes': f"# {topic}\n\nNo information found in knowledge base.\n\nPlease run the KB update script to fetch latest information about this topic.",
                'sources_used': 0,
                'from_kb': False,
                'message': 'Knowledge base needs updating for this topic'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching topic from web: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_url_query(self, url: str) -> Dict[str, Any]:
        """Process a URL-based query"""
        try:
            self.logger.info(f"Processing URL: {url}")
            
            # Scrape the URL
            scrape_result = await self.scraper.process({'url': url})
            
            if not scrape_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to scrape URL: {url}"
                }
            
            # Extract key points (primary focus)
            bullet_result = await self.summarizer.process({
                'content': scrape_result['content'],
                'mode': 'bullet_points',
                'num_points': 8
            })
            
            # Create notes (emphasize key points)
            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': scrape_result.get('title', 'Web Article'),
                'summary': bullet_result.get('summary', ''),  # Use bullet points as main content
                'key_points': [],  # Key points already in summary
                'sources': [{'title': scrape_result.get('title', 'Web Article'), 'url': url}],
                'topic': 'Web Content'
            })
            
            # Optionally add to knowledge base
            if scrape_result['content']:
                doc = self.doc_processor.process_document(
                    content=scrape_result['content'],
                    source='web',
                    title=scrape_result.get('title'),
                    url=url,
                    topic='web_content'
                )
                if doc:
                    self.vector_store.add_documents([doc])
                    self.logger.info("Added scraped content to knowledge base")
            
            return {
                'success': True,
                'query_type': 'url',
                'query': url,
                'notes': notes_result.get('notes', ''),
                'sources_used': 1,
                'from_kb': False
            }
            
        except Exception as e:
            self.logger.error(f"Error processing URL: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_text_query(self, text: str) -> Dict[str, Any]:
        """Process direct text input"""
        try:
            self.logger.info("Processing direct text input")
            
            # Extract key points (primary focus)
            bullet_result = await self.summarizer.process({
                'content': text,
                'mode': 'bullet_points',
                'num_points': 8
            })
            
            # Create notes (emphasize key points)
            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': 'Key Points Summary',
                'summary': bullet_result.get('summary', ''),  # Use bullet points as main content
                'key_points': [],  # Key points already in summary
                'sources': [],
                'topic': 'Direct Input'
            })
            
            return {
                'success': True,
                'query_type': 'text',
                'notes': notes_result.get('notes', ''),
                'sources_used': 0,
                'from_kb': False
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process(self, query: str) -> Dict[str, Any]:
        """Main processing method"""
        try:
            # Detect query type
            query_type = self.detect_query_type(query)
            
            self.logger.info(f"Detected query type: {query_type.value}")
            
            # Process based on type
            if query_type == QueryType.URL:
                return await self.process_url_query(query)
            elif query_type == QueryType.TEXT:
                return await self.process_text_query(query)
            else:  # TOPIC
                return await self.process_topic_query(query)
                
        except Exception as e:
            self.logger.error(f"Error in orchestrator: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        kb_stats = self.retriever.get_stats()
        return {
            'knowledge_base': kb_stats,
            'agents': {
                'retriever': 'active',
                'scraper': 'active',
                'summarizer': 'active',
                'note_maker': 'active'
            }
        }