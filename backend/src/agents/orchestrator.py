"""
Main orchestrator for coordinating all agents
"""
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

from src.agents.retriever import RetrieverAgent
from src.agents.scraper import ScraperAgent
from src.agents.content_agent import ContentAgent
from src.agents.note_maker import NoteMakerAgent
from src.agents.web_search import WebSearchAgent
from src.db import document_store
from src.utils.logger import get_logger
from src.utils.academic_search import get_academic_search

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
        self.content_agent = ContentAgent()
        self.note_maker = NoteMakerAgent()
        self.web_search = WebSearchAgent()
        self.academic_search = get_academic_search()
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
    
    async def process_topic_query(self, query: str, summarization_mode: str = "paragraph_summary", output_length: str = "auto", search_mode: str = "auto", research_mode: bool = False, user_id: str = None) -> Dict[str, Any]:
        """Process a topic-based query with configurable search mode.

        Args:
            query: The user's topic query
            summarization_mode: Format for notes output
            output_length: Length of summary output
            search_mode: 'auto' (KB first, web fallback), 'kb_only', 'web_search' (web only)
            user_id: Optional user ID for scoping KB searches and storage
        """
        try:
            self.logger.info(f"Processing topic query: {query[:50]}... (search_mode={search_mode})")

            # Web search only mode - skip KB entirely
            if search_mode in ("web_search", "web_only"):
                self.logger.info("Search mode: web_search - going directly to web search")
                result = await self._web_search_and_process(query, summarization_mode, output_length, user_id=user_id)
                if research_mode and result.get('success'):
                    result = await self._append_related_papers(result, query)
                return result

            # Both mode - search KB and web, combine results
            if search_mode == "both":
                self.logger.info("Search mode: both - searching KB and web")
                result = await self._process_both_sources(query, summarization_mode, output_length, user_id=user_id)
                if research_mode and result.get('success'):
                    result = await self._append_related_papers(result, query)
                return result

            # Search knowledge base (for 'auto' and 'kb_only' modes)
            retrieval_result = await self.retriever.process({
                'query': query,
                'k': 5,
                'threshold': 1.0,
                'user_id': user_id
            })

            if retrieval_result['success'] and retrieval_result['count'] > 0:
                # Found relevant documents in KB
                self.logger.info(f"Found {retrieval_result['count']} relevant documents in KB")

                # Extract content from results
                documents = [doc['content'] for doc in retrieval_result['results']]

                # Remove duplicate sources by URL - filter out empty/invalid URLs
                seen_urls = set()
                sources = []
                for doc in retrieval_result['results']:
                    url = doc['metadata'].get('url', '')
                    if url and url not in seen_urls and url.startswith(('http://', 'https://')):
                        seen_urls.add(url)
                        sources.append({
                            'title': doc['metadata'].get('title', 'Unknown'),
                            'url': url
                        })

                # Create comprehensive technical summary using ALL available documents
                all_content = ' '.join(documents[:5])
                self.logger.info(f"Processing {len(documents)} documents with total length: {len(all_content)}")

                detailed_summary = await self.content_agent.process({
                    'content': all_content,
                    'mode': summarization_mode,
                    'output_length': output_length
                })

                # Gap resolution
                detailed_summary = await self._resolve_gaps(
                    detailed_summary, all_content, query, summarization_mode, output_length
                )

                # Determine title based on summarization mode
                title_suffix_map = {
                    'paragraph_summary': '',
                    'important_points': ' - Important Points',
                    'key_highlights': ' - Key Highlights'
                }
                note_title = f"{query.title()}{title_suffix_map.get(summarization_mode, '')}"

                notes_result = await self.note_maker.process({
                    'mode': 'create',
                    'title': note_title,
                    'summary': detailed_summary.get('summary', ''),
                    'key_points': [],
                    'sources': sources,
                    'topic': query,
                    'summarization_mode': summarization_mode
                })

                result = {
                    'success': True,
                    'query_type': 'topic',
                    'query': query,
                    'notes': notes_result.get('notes', ''),
                    'sources_used': len(sources),
                    'from_kb': True
                }
                if research_mode:
                    result = await self._append_related_papers(result, query)
                return result

            else:
                # No relevant documents found in KB
                if search_mode == "kb_only":
                    self.logger.info("Search mode: kb_only - no results found, not searching web")
                    return {
                        'success': True,
                        'query_type': 'topic',
                        'query': query,
                        'notes': f"# {query}\n\nNo information found in the knowledge base for this topic.\n\nTry switching to **Auto** or **Web Search** mode to search the internet.",
                        'sources_used': 0,
                        'from_kb': False,
                        'message': 'No results found in knowledge base'
                    }

                # Auto mode: fall back to web search
                self.logger.info("No relevant documents in KB, falling back to web search...")
                result = await self._web_search_and_process(query, summarization_mode, output_length, user_id=user_id)
                if research_mode and result.get('success'):
                    result = await self._append_related_papers(result, query)
                return result

        except Exception as e:
            self.logger.error(f"Error processing topic query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _web_search_and_process(
        self,
        query: str,
        summarization_mode: str = "paragraph_summary",
        output_length: str = "auto",
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Use WebSearchAgent to find content, then summarize and make notes.
        Replaces the old _build_search_urls() + fetch_and_process_topic().
        """
        try:
            self.logger.info(f"Starting web search for: {query}")

            # Step 1: Web search via WebSearchAgent
            search_result = await self.web_search.search_and_get_content(query)

            if not search_result:
                return {
                    'success': True,
                    'query_type': 'topic',
                    'query': query,
                    'notes': f"# {query}\n\nNo information found via web search. The topic may be too specific or the search service may be temporarily unavailable.\n\nTry rephrasing your query or using a more specific topic name.",
                    'sources_used': 0,
                    'from_kb': False,
                    'message': 'Web search returned no results'
                }

            # Step 2: Summarize the content
            summary_result = await self.content_agent.process({
                'content': search_result['content'],
                'mode': summarization_mode,
                'output_length': output_length
            })

            # Gap resolution
            summary_result = await self._resolve_gaps(
                summary_result, search_result['content'], query, summarization_mode, output_length
            )

            # Step 3: Create notes with source references
            title_suffix_map = {
                'paragraph_summary': '',
                'important_points': ' - Important Points',
                'key_highlights': ' - Key Highlights'
            }
            note_title = f"{query.title()}{title_suffix_map.get(summarization_mode, '')}"

            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': note_title,
                'summary': summary_result.get('summary', ''),
                'key_points': [],
                'sources': search_result.get('sources', []),
                'topic': query,
                'summarization_mode': summarization_mode
            })

            # Step 4: Add to KB for future queries
            if user_id and search_result.get('content'):
                try:
                    document_store.add_document(
                        user_id=user_id,
                        title=f"Web search: {query}",
                        topic=query.lower().replace(' ', '-'),
                        content=search_result['content'],
                        source='web_search',
                        url=search_result['sources'][0]['url'] if search_result.get('sources') else '',
                    )
                    self.logger.info(f"Added web search content about '{query}' to KB")
                except Exception as e:
                    self.logger.warning(f"Failed to add web search content to KB: {e}")

            return {
                'success': True,
                'query_type': 'topic',
                'query': query,
                'notes': notes_result.get('notes', ''),
                'sources_used': len(search_result.get('sources', [])),
                'from_kb': False,
                'message': f"Generated from web search ({search_result.get('results_scraped', 0)} sources)"
            }

        except Exception as e:
            self.logger.error(f"Error in web search and process: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_both_sources(
        self,
        query: str,
        summarization_mode: str = "paragraph_summary",
        output_length: str = "auto",
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Search both KB and web, combine results, then summarize.

        Used when search_mode='both' to get comprehensive coverage by
        pulling information from the local knowledge base AND the web.
        """
        try:
            self.logger.info(f"Processing with both KB and web search for: {query[:50]}")

            kb_content = ""
            kb_sources = []
            web_content = ""
            web_sources = []

            # Step 1: Search knowledge base
            try:
                retrieval_result = await self.retriever.process({
                    'query': query,
                    'k': 5,
                    'threshold': 1.0,
                    'user_id': user_id
                })
                if retrieval_result['success'] and retrieval_result['count'] > 0:
                    documents = [doc['content'] for doc in retrieval_result['results']]
                    kb_content = ' '.join(documents[:5])

                    # Build KB sources (deduplicated by URL)
                    seen_urls = set()
                    for doc in retrieval_result['results']:
                        url = doc['metadata'].get('url', '')
                        if url and url not in seen_urls and url.startswith(('http://', 'https://')):
                            seen_urls.add(url)
                            kb_sources.append({
                                'title': doc['metadata'].get('title', 'Unknown'),
                                'url': url
                            })
                    self.logger.info(f"KB returned {len(documents)} documents, {len(kb_sources)} sources")
                else:
                    self.logger.info("KB returned no relevant documents")
            except Exception as e:
                self.logger.warning(f"KB search failed in 'both' mode: {e}")

            # Step 2: Web search (always, regardless of KB results)
            try:
                search_result = await self.web_search.search_and_get_content(query)
                if search_result:
                    web_content = search_result.get('content', '')
                    web_sources = search_result.get('sources', [])
                    self.logger.info(f"Web search returned {len(web_content)} chars, {len(web_sources)} sources")
                else:
                    self.logger.info("Web search returned no results")
            except Exception as e:
                self.logger.warning(f"Web search failed in 'both' mode: {e}")

            # Step 3: Check if we got anything
            if not kb_content and not web_content:
                return {
                    'success': True,
                    'query_type': 'topic',
                    'query': query,
                    'notes': f"# {query}\n\nNo information found from either the knowledge base or web search.\n\nTry rephrasing your query or using a more specific topic name.",
                    'sources_used': 0,
                    'from_kb': False,
                    'message': 'No results from either source'
                }

            # Step 4: Combine content from both sources
            combined_parts = []
            if kb_content:
                combined_parts.append(kb_content)
            if web_content:
                combined_parts.append(web_content)
            combined_content = '\n\n---\n\n'.join(combined_parts)

            self.logger.info(f"Combined content: {len(combined_content)} chars from {len(combined_parts)} sources")

            # Step 5: Summarize the combined content
            summary_result = await self.content_agent.process({
                'content': combined_content,
                'mode': summarization_mode,
                'output_length': output_length
            })

            # Gap resolution
            summary_result = await self._resolve_gaps(
                summary_result, combined_content, query, summarization_mode, output_length
            )

            # Step 6: Build deduplicated source list (KB + Web)
            all_sources = kb_sources + web_sources
            seen = set()
            unique_sources = []
            for s in all_sources:
                if s['url'] not in seen:
                    seen.add(s['url'])
                    unique_sources.append(s)

            # Step 7: Create notes
            title_suffix_map = {
                'paragraph_summary': '',
                'important_points': ' - Important Points',
                'key_highlights': ' - Key Highlights'
            }
            note_title = f"{query.title()}{title_suffix_map.get(summarization_mode, '')}"

            notes_result = await self.note_maker.process({
                'mode': 'create',
                'title': note_title,
                'summary': summary_result.get('summary', ''),
                'key_points': [],
                'sources': unique_sources,
                'topic': query,
                'summarization_mode': summarization_mode
            })

            # Step 8: Add web content to KB for future queries
            if user_id and web_content:
                try:
                    document_store.add_document(
                        user_id=user_id,
                        title=f"Web search: {query}",
                        topic=query.lower().replace(' ', '-'),
                        content=web_content,
                        source='web_search',
                        url=web_sources[0]['url'] if web_sources else '',
                    )
                    self.logger.info(f"Added web content from 'both' mode to KB")
                except Exception as e:
                    self.logger.warning(f"Failed to add web content to KB: {e}")

            # Build descriptive message
            source_parts = []
            if kb_content:
                source_parts.append(f"KB ({len(kb_sources)} sources)")
            if web_content:
                source_parts.append(f"Web ({len(web_sources)} sources)")

            return {
                'success': True,
                'query_type': 'topic',
                'query': query,
                'notes': notes_result.get('notes', ''),
                'sources_used': len(unique_sources),
                'from_kb': bool(kb_content),
                'message': f"Combined from {' and '.join(source_parts)}"
            }

        except Exception as e:
            self.logger.error(f"Error in both-sources processing: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _append_related_papers(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Append related academic papers to notes when research_mode is enabled."""
        try:
            self.logger.info(f"Research Mode: searching for related papers on '{query[:50]}'")
            papers = self.academic_search.search_papers(query, max_results=5)

            if papers:
                papers_md = self.academic_search.format_papers_markdown(papers)
                result['notes'] = result.get('notes', '') + papers_md
                result['related_papers'] = papers
                self.logger.info(f"Research Mode: appended {len(papers)} related papers")
            else:
                result['related_papers'] = []
                self.logger.info("Research Mode: no related papers found")
        except Exception as e:
            self.logger.warning(f"Research Mode: paper discovery failed: {e}")
            result['related_papers'] = []

        return result

    async def _fill_content_gaps(self, original_query: str, gap_queries: List[str]) -> Optional[str]:
        """Use WebSearchAgent to fill knowledge gaps identified by ContentAgent."""
        all_content = []
        for gap_query in gap_queries[:2]:  # Max 2 gap queries to limit API calls
            search_query = f"{original_query} {gap_query}"
            self.logger.info(f"Gap resolution: searching for '{search_query}'")
            result = await self.web_search.search_and_get_content(search_query)
            if result and result.get('content'):
                all_content.append(result['content'][:3000])  # Limit each gap fill

        return "\n\n".join(all_content) if all_content else None

    async def _resolve_gaps(self, result: Dict[str, Any], content: str, query: str,
                            summarization_mode: str, output_length: str) -> Dict[str, Any]:
        """Check ContentAgent result for gaps and resolve them if needed (max 1 iteration)."""
        if result.get('needs_more_info') and result.get('gap_queries'):
            self.logger.info(f"ContentAgent identified gaps: {result['gap_queries']}")

            gap_content = await self._fill_content_gaps(query, result['gap_queries'])

            if gap_content:
                enriched = content + "\n\n--- Additional Information ---\n\n" + gap_content
                result = await self.content_agent.process({
                    'content': enriched,
                    'mode': summarization_mode,
                    'output_length': output_length
                })

        return result

    async def process_url_query(self, url: str, summarization_mode: str = "paragraph_summary", output_length: str = "auto", research_mode: bool = False, user_id: str = None) -> Dict[str, Any]:
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

            # Pass the summarization_mode and output_length directly to the content agent
            # skip_evaluation=True: URL content is authoritative, no gap resolution needed
            detailed_summary = await self.content_agent.process({
                'content': scrape_result['content'],
                'mode': summarization_mode,
                'output_length': output_length,
                'skip_evaluation': True
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
            if user_id and scrape_result['content']:
                try:
                    document_store.add_document(
                        user_id=user_id,
                        title=scrape_result.get('title', 'Web Article'),
                        topic='web_content',
                        content=scrape_result['content'],
                        source='web',
                        url=url,
                    )
                    self.logger.info("Added scraped content to knowledge base")
                except Exception as e:
                    self.logger.warning(f"Failed to add scraped content to KB: {e}")
            
            result = {
                'success': True,
                'query_type': 'url',
                'query': url,
                'notes': notes_result.get('notes', ''),
                'sources_used': 1,
                'from_kb': False
            }
            if research_mode:
                # Use page title as search query for paper discovery
                paper_query = scrape_result.get('title', url)
                result = await self._append_related_papers(result, paper_query)
            return result

        except Exception as e:
            self.logger.error(f"Error processing URL: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_text_query(self, text: str, summarization_mode: str = "paragraph_summary", output_length: str = "auto", research_mode: bool = False, user_id: str = None) -> Dict[str, Any]:
        """Process direct text input"""
        try:
            self.logger.info(f"Processing direct text input of length: {len(text)}")

            # Validate input
            if not text or len(text.strip()) < 10:
                return {
                    'success': False,
                    'error': 'Input text too short. Please provide at least 10 characters.'
                }

            # Pass the summarization_mode and output_length directly to the content agent
            # skip_evaluation=True: Direct text/PDF input is authoritative, no gap resolution needed
            detailed_summary = await self.content_agent.process({
                'content': text.strip(),
                'mode': summarization_mode,
                'output_length': output_length,
                'skip_evaluation': True
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

            result = {
                'success': True,
                'query_type': 'text',
                'query': text[:100] + "..." if len(text) > 100 else text,
                'notes': notes_result.get('notes', ''),
                'sources_used': 0,
                'from_kb': False
            }
            if research_mode:
                # Clean markdown formatting for better academic search results
                paper_query = text.strip()[:200]
                for ch in '#*[](){}|`~>\n\r':
                    paper_query = paper_query.replace(ch, ' ')
                paper_query = ' '.join(paper_query.split()).strip()[:80]
                result = await self._append_related_papers(result, paper_query)
            return result

        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process(self, query: str, summarization_mode: str = "paragraph_summary", output_length: str = "auto", search_mode: str = "auto", research_mode: bool = False, user_id: str = None) -> Dict[str, Any]:
        """Main processing method"""
        try:
            # Detect query type
            query_type = self.detect_query_type(query)

            self.logger.info(f"Detected query type: {query_type.value}, output_length: {output_length}, search_mode: {search_mode}, research_mode: {research_mode}")

            # Process based on type
            if query_type == QueryType.URL:
                return await self.process_url_query(query, summarization_mode, output_length, research_mode=research_mode, user_id=user_id)
            elif query_type == QueryType.TEXT:
                return await self.process_text_query(query, summarization_mode, output_length, research_mode=research_mode, user_id=user_id)
            else:  # TOPIC
                return await self.process_topic_query(query, summarization_mode, output_length, search_mode=search_mode, research_mode=research_mode, user_id=user_id)
                
        except Exception as e:
            self.logger.error(f"Error in orchestrator: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        # Get LLM provider info directly from the LLM client singleton for accuracy
        try:
            from src.utils.llm_client import get_llm_client
            llm_client = get_llm_client()
            llm_info = llm_client.get_provider_info() if llm_client else {}
        except Exception:
            # Fallback to content_agent if direct access fails
            llm_info = self.content_agent.get_provider_info()

        return {
            'agents': {
                'retriever': 'active',
                'scraper': 'active',
                'content_agent': 'active',
                'note_maker': 'active',
                'web_search': 'active'
            },
            'llm': {
                'provider': llm_info.get('provider', 'unknown'),
                'model': llm_info.get('model', 'unknown'),
                'is_local': llm_info.get('is_local', False)
            }
        }