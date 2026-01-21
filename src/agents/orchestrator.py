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
    
    async def process_topic_query(self, query: str, summarization_mode: str = "paragraph_summary") -> Dict[str, Any]:
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

                # Remove duplicate sources by URL
                seen_urls = set()
                sources = []
                for doc in retrieval_result['results']:
                    url = doc['metadata'].get('url', '#')
                    if url not in seen_urls and url != '#':
                        seen_urls.add(url)
                        sources.append({
                            'title': doc['metadata'].get('title', 'Unknown'),
                            'url': url
                        })

                # Create comprehensive technical summary using ALL available documents
                all_content = ' '.join(documents[:5])  # Use top 5 documents for maximum content
                self.logger.info(f"Processing {len(documents)} documents with total length: {len(all_content)}")

                # Pass the summarization_mode directly to the summarizer
                detailed_summary = await self.summarizer.process({
                    'content': all_content,  # Use all available content
                    'mode': summarization_mode
                })

                # Determine title based on summarization mode
                title_suffix_map = {
                    'paragraph_summary': '',
                    'important_points': ' - Important Points',
                    'key_highlights': ' - Key Highlights'
                }
                note_title = f"{query.title()}{title_suffix_map.get(summarization_mode, '')}"

                # Create notes with detailed technical content
                notes_result = await self.note_maker.process({
                    'mode': 'create',
                    'title': note_title,
                    'summary': detailed_summary.get('summary', ''),
                    'key_points': [],
                    'sources': sources,
                    'topic': query,
                    'summarization_mode': summarization_mode  # Pass mode for header formatting
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
                return await self.fetch_and_process_topic(query, summarization_mode)
                
        except Exception as e:
            self.logger.error(f"Error processing topic query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def fetch_and_process_topic(self, topic: str, summarization_mode: str = "paragraph_summary") -> Dict[str, Any]:
        """Fetch information about a topic from web and process"""
        try:
            self.logger.info(f"No KB results for '{topic}', attempting web search...")

            # Build search URLs for the topic
            search_urls = self._build_search_urls(topic)

            if not search_urls:
                return {
                    'success': True,
                    'query_type': 'topic',
                    'query': topic,
                    'notes': f"# {topic}\n\nNo information found in knowledge base.\n\nPlease run the seed script to populate the KB:\n```\npython scripts/seed_data.py --sample\n```",
                    'sources_used': 0,
                    'from_kb': False,
                    'message': 'Knowledge base needs updating for this topic'
                }

            # Try to scrape content from educational sources
            scraped_content = []
            sources = []

            for url in search_urls[:3]:  # Try up to 3 URLs
                try:
                    scrape_result = await self.scraper.process({'url': url})
                    if scrape_result.get('success') and scrape_result.get('content'):
                        content = scrape_result['content']
                        if len(content) > 200:  # Only use substantial content
                            scraped_content.append(content)
                            sources.append({
                                'title': scrape_result.get('title', 'Web Source'),
                                'url': url
                            })
                            self.logger.info(f"Successfully scraped: {url}")
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {url}: {e}")
                    continue

            if not scraped_content:
                return {
                    'success': True,
                    'query_type': 'topic',
                    'query': topic,
                    'notes': f"# {topic}\n\nNo information found in knowledge base and web scraping was unsuccessful.\n\nPlease run the seed script to populate the KB:\n```\npython scripts/seed_data.py --sample\n```",
                    'sources_used': 0,
                    'from_kb': False,
                    'message': 'Could not fetch information from web'
                }

            # Combine scraped content and summarize
            combined_content = "\n\n".join(scraped_content[:3])
            self.logger.info(f"Combined {len(scraped_content)} web sources, total length: {len(combined_content)}")

            # Pass the summarization_mode directly to the summarizer
            summary_result = await self.summarizer.process({
                'content': combined_content,
                'mode': summarization_mode
            })

            # Determine title based on summarization mode
            title_suffix_map = {
                'paragraph_summary': '',
                'important_points': ' - Important Points',
                'key_highlights': ' - Key Highlights'
            }
            note_title = f"{topic.title()}{title_suffix_map.get(summarization_mode, '')}"

            # Create notes
            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': note_title,
                'summary': summary_result.get('summary', ''),
                'key_points': [],
                'sources': sources,
                'topic': topic,
                'summarization_mode': summarization_mode  # Pass mode for header formatting
            })

            # Optionally add to KB for future queries
            if combined_content:
                try:
                    doc = self.doc_processor.process_document(
                        content=combined_content,
                        source='web',
                        title=f"Web content: {topic}",
                        url=sources[0]['url'] if sources else '',
                        topic=topic.lower().replace(' ', '-')
                    )
                    if doc:
                        self.vector_store.add_documents([doc])
                        self.logger.info(f"Added web content about '{topic}' to KB")
                except Exception as e:
                    self.logger.warning(f"Failed to add to KB: {e}")

            return {
                'success': True,
                'query_type': 'topic',
                'query': topic,
                'notes': notes_result.get('notes', ''),
                'sources_used': len(sources),
                'from_kb': False,
                'message': 'Generated from web sources'
            }

        except Exception as e:
            self.logger.error(f"Error fetching topic from web: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _build_search_urls(self, topic: str) -> List[str]:
        """Build educational URLs for a topic"""
        # Common educational sources for tech/ML topics
        topic_slug = topic.lower().replace(' ', '-')
        topic_query = topic.lower().replace(' ', '+')

        urls = []

        # Educational sites that work well with scraping
        educational_sources = [
            f"https://www.geeksforgeeks.org/{topic_slug}/",
            f"https://www.tutorialspoint.com/{topic_slug}/index.htm",
            f"https://realpython.com/tutorials/{topic_slug}/",
        ]

        # Only add URLs that are likely to exist
        for url in educational_sources:
            urls.append(url)

        return urls[:3]  # Return top 3
    
    async def process_url_query(self, url: str, summarization_mode: str = "paragraph_summary") -> Dict[str, Any]:
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

            # Log content length for debugging
            content_length = len(scrape_result['content'])
            self.logger.info(f"Scraped content length: {content_length} characters")

            if content_length < 100:
                self.logger.error(f"Scraped content too short: {scrape_result['content'][:200]}")
                return {
                    'success': False,
                    'error': f"Scraped content too short ({content_length} chars). Check URL accessibility."
                }

            # Pass the summarization_mode directly to the summarizer
            detailed_summary = await self.summarizer.process({
                'content': scrape_result['content'],
                'mode': summarization_mode
            })

            # Determine title based on summarization mode
            base_title = scrape_result.get('title', 'Web Article')
            title_suffix_map = {
                'paragraph_summary': '',
                'important_points': ' - Important Points',
                'key_highlights': ' - Key Highlights'
            }
            note_title = f"{base_title}{title_suffix_map.get(summarization_mode, '')}"

            # Create notes with detailed content
            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': note_title,
                'summary': detailed_summary.get('summary', ''),
                'key_points': [],
                'sources': [{'title': scrape_result.get('title', 'Web Article'), 'url': url}],
                'topic': 'Web Content',
                'summarization_mode': summarization_mode  # Pass mode for header formatting
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
    
    async def process_text_query(self, text: str, summarization_mode: str = "paragraph_summary") -> Dict[str, Any]:
        """Process direct text input"""
        try:
            self.logger.info(f"Processing direct text input of length: {len(text)}")

            # Validate input
            if not text or len(text.strip()) < 10:
                return {
                    'success': False,
                    'error': 'Input text too short. Please provide at least 10 characters.'
                }

            # Pass the summarization_mode directly to the summarizer
            detailed_summary = await self.summarizer.process({
                'content': text.strip(),
                'mode': summarization_mode
            })

            # Check if summarization was successful
            if not detailed_summary.get('success', False):
                self.logger.error(f"Summarization failed: {detailed_summary.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Failed to summarize text: {detailed_summary.get('error', 'Unknown error')}"
                }

            # Determine title based on summarization mode
            title_map = {
                'paragraph_summary': 'Summary',
                'important_points': 'Important Points',
                'key_highlights': 'Key Highlights'
            }
            note_title = title_map.get(summarization_mode, 'Summary')

            # Create notes with detailed content
            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': note_title,
                'summary': detailed_summary.get('summary', ''),
                'key_points': [],
                'sources': [],
                'topic': 'Direct Input',
                'summarization_mode': summarization_mode  # Pass mode for header formatting
            })

            # Check if note creation was successful
            if not notes_result.get('success', False):
                self.logger.error(f"Note creation failed: {notes_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Failed to create notes: {notes_result.get('error', 'Unknown error')}"
                }

            return {
                'success': True,
                'query_type': 'text',
                'query': text[:100] + "..." if len(text) > 100 else text,
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
    
    async def process(self, query: str, summarization_mode: str = "paragraph_summary") -> Dict[str, Any]:
        """Main processing method"""
        try:
            # Detect query type
            query_type = self.detect_query_type(query)

            self.logger.info(f"Detected query type: {query_type.value}")

            # Process based on type
            if query_type == QueryType.URL:
                return await self.process_url_query(query, summarization_mode)
            elif query_type == QueryType.TEXT:
                return await self.process_text_query(query, summarization_mode)
            else:  # TOPIC
                return await self.process_topic_query(query, summarization_mode)
                
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