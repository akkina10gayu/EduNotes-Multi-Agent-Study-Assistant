"""
WebSearchAgent - LLM-powered intelligent web search agent.

This is a genuine AGENT because it uses LLM reasoning at 3 decision points:
1. Query Generation: Reformulates user topic into effective search queries
2. Result Evaluation: Ranks and filters search results for educational relevance
3. Content Quality Assessment: Validates scraped content is useful

Uses DuckDuckGo search API and ScraperAgent as TOOLS, LLM (Groq) as the BRAIN.

Pipeline (5 stages):
    User Topic
        -> [Stage 1: Query Generation]      (LLM)
        -> [Stage 2: Web Search]             (DuckDuckGo tool)
        -> [Stage 3: Result Evaluation]      (LLM)
        -> [Stage 4: Content Extraction]     (Scraper tool)
        -> [Stage 5: Quality Assessment]     (LLM)
        -> Return curated content + sources
"""
import re
import time
from typing import Dict, Any, List, Optional

from src.agents.base import BaseAgent
from src.agents.scraper import ScraperAgent
from src.utils.search_provider import get_search_provider, get_fallback_provider, SearchResult
from src.utils.llm_client import get_llm_client
from src.utils.cache_utils import cache, get_cache_key
from config import settings


class WebSearchAgent(BaseAgent):
    """
    LLM-powered web search agent.

    This is a genuine agent because it uses LLM reasoning at 3 points:
    1. Query generation: Reformulates user topic into effective search queries
    2. Result evaluation: Ranks and filters search results for relevance
    3. Content quality assessment: Validates scraped content is useful

    Uses search API and scraper as TOOLS, LLM as the BRAIN.
    """

    def __init__(self):
        super().__init__("WebSearchAgent")
        self.llm_client = get_llm_client()
        self.search_provider = get_search_provider()
        self.fallback_provider = get_fallback_provider()
        self.scraper = ScraperAgent()

    # =================================================================
    # STAGE 1: QUERY GENERATION (LLM-powered)
    # =================================================================

    def generate_search_queries(self, topic: str) -> List[str]:
        """
        Use LLM to generate 2-3 effective search queries from user topic.

        Why this matters:
        - User says "CNN" -> LLM generates "convolutional neural network
          tutorial", "CNN deep learning architecture explained"
        - User says "overfitting" -> LLM generates "overfitting machine
          learning prevention", "how to avoid overfitting techniques"
        - Better queries = better search results = better notes
        """
        # Cache check — avoid redundant LLM calls for the same topic (24h TTL)
        normalized_topic = topic.lower().strip()
        cache_key = get_cache_key("ws_queries", normalized_topic)
        cached_queries = cache.get(cache_key)
        if cached_queries is not None:
            self.logger.info(f"Stage 1 cache HIT for topic: {topic[:50]}")
            return cached_queries

        if self.llm_client.is_local_mode():
            return self._generate_fallback_queries(topic)

        system_prompt = (
            "You are a search query optimizer for an educational "
            "study assistant. Given a topic, generate 2-3 search queries that will "
            "find the best educational content about that topic. Focus on tutorials, "
            "explanations, and comprehensive guides."
        )

        prompt = f"""Generate exactly 3 search queries to find educational content about this topic: "{topic}"

RULES:
- Each query should target different aspects of the topic
- Include words like "tutorial", "explained", "guide", "fundamentals"
- Make queries specific enough to avoid irrelevant results
- Focus on EDUCATIONAL content, not news or products

FORMAT (one query per line, nothing else):
query 1
query 2
query 3

EXAMPLE for topic "neural networks":
neural networks fundamentals tutorial beginners
how neural networks work architecture explained
neural network training backpropagation guide

YOUR QUERIES FOR "{topic}":"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,
                system_prompt=system_prompt
            )

            if not response:
                return self._generate_fallback_queries(topic)

            queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present (1. or 1) or - or *)
                line = re.sub(r'^[\d]+[.\)]\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                line = line.strip('"\'')
                if line and len(line) > 5:
                    queries.append(line)

            if not queries:
                return self._generate_fallback_queries(topic)

            self.logger.info(f"LLM generated {len(queries[:3])} search queries")
            # Cache successful LLM-generated queries (24h TTL)
            cache.set(cache_key, queries[:3], expire=86400)
            return queries[:3]

        except Exception as e:
            self.logger.error(f"LLM query generation failed: {e}")
            return self._generate_fallback_queries(topic)

    def _generate_fallback_queries(self, topic: str) -> List[str]:
        """Simple fallback when LLM is unavailable"""
        self.logger.info("Using fallback query generation")
        return [
            f"{topic} tutorial explained",
            f"{topic} fundamentals guide",
            f"what is {topic} comprehensive overview"
        ]

    # =================================================================
    # STAGE 2: WEB SEARCH (Tool - no LLM)
    # =================================================================

    def execute_search(self, queries: List[str], original_topic: str = None) -> List[SearchResult]:
        """
        Execute search queries using the search provider tool.
        This is a TOOL invocation, no intelligence here.

        Deduplicates results by URL across all queries.
        Adds a small delay between queries to avoid rate-limiting.

        If the primary provider returns 0 results and a fallback provider
        is available, automatically retries with the fallback (Google).
        """
        all_results = []
        seen_urls = set()

        for i, query in enumerate(queries):
            try:
                # Per-query cache check (1h TTL from WEB_SEARCH_CACHE_TTL)
                normalized_query = query.lower().strip()
                query_cache_key = get_cache_key("ws_search", normalized_query)
                cached_results = cache.get(query_cache_key)

                if cached_results is not None:
                    # Cache hit — skip delay and API call
                    self.logger.info(f"Stage 2 cache HIT for query: {query[:50]}")
                    results = cached_results
                else:
                    # Cache miss — delay between queries to avoid rate-limiting
                    if i > 0:
                        time.sleep(1)

                    results = self.search_provider.search(
                        query=query,
                        max_results=settings.SEARCH_MAX_RESULTS
                    )

                    # Cache non-empty search results
                    if results:
                        cache.set(query_cache_key, results, expire=settings.WEB_SEARCH_CACHE_TTL)

                for result in results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        all_results.append(result)

            except Exception as e:
                self.logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # --- Retry: If primary returned 0, try raw topic on primary provider ---
        # LLM-generated queries can be too decorated for DDG. The raw topic
        # (e.g. "Bi-Encoders and cross encoders in RAG") sometimes works better.
        if not all_results and original_topic:
            topic_clean = original_topic.strip()
            if topic_clean.lower() not in [q.lower().strip() for q in queries]:
                self.logger.info(
                    f"Retrying primary provider with raw topic: {topic_clean[:50]}"
                )
                try:
                    time.sleep(1)
                    results = self.search_provider.search(
                        query=topic_clean,
                        max_results=settings.SEARCH_MAX_RESULTS
                    )
                    if results:
                        norm = topic_clean.lower().strip()
                        cache.set(
                            get_cache_key("ws_search", norm),
                            results,
                            expire=settings.WEB_SEARCH_CACHE_TTL
                        )
                    for result in results:
                        if result.url not in seen_urls:
                            seen_urls.add(result.url)
                            all_results.append(result)
                    if all_results:
                        self.logger.info(
                            f"Raw topic retry found {len(all_results)} results"
                        )
                except Exception as e:
                    self.logger.warning(f"Raw topic retry failed: {e}")

        # --- Fallback: If still 0 results, try Google scraper ---
        if not all_results and self.fallback_provider:
            self.logger.warning(
                f"Primary search ({settings.SEARCH_PROVIDER}) returned 0 results "
                f"across all queries. Trying fallback provider (Google)..."
            )

            # Build fallback queries: original queries + raw topic for broader coverage
            fallback_queries = list(queries)
            if original_topic:
                topic_clean = original_topic.strip()
                if topic_clean and topic_clean not in fallback_queries:
                    fallback_queries.append(topic_clean)

            for i, query in enumerate(fallback_queries):
                try:
                    # Separate cache namespace for fallback results
                    normalized_query = query.lower().strip()
                    fb_cache_key = get_cache_key("ws_search_fb", normalized_query)
                    cached_results = cache.get(fb_cache_key)

                    if cached_results is not None:
                        self.logger.info(f"Fallback cache HIT for query: {query[:50]}")
                        results = cached_results
                    else:
                        # Longer delay for Google to avoid rate-limiting
                        if i > 0:
                            time.sleep(2)

                        results = self.fallback_provider.search(
                            query=query,
                            max_results=settings.SEARCH_MAX_RESULTS
                        )

                        if results:
                            cache.set(fb_cache_key, results, expire=settings.WEB_SEARCH_CACHE_TTL)

                    for result in results:
                        if result.url not in seen_urls:
                            seen_urls.add(result.url)
                            all_results.append(result)

                except Exception as e:
                    self.logger.warning(f"Fallback search failed for query '{query}': {e}")
                    continue

            if all_results:
                self.logger.info(
                    f"Fallback provider found {len(all_results)} results "
                    f"(primary had 0)"
                )
            else:
                self.logger.warning(
                    "Fallback provider also returned 0 results"
                )

        self.logger.info(
            f"Search returned {len(all_results)} unique results "
            f"from {len(queries)} queries"
        )
        return all_results

    # =================================================================
    # STAGE 3: RESULT EVALUATION (LLM-powered)
    # =================================================================

    def evaluate_results(
        self,
        topic: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Use LLM to evaluate and rank search results for educational value.

        Why this matters:
        - DuckDuckGo returns a mix of everything: blogs, forums, shops,
          documentation, tutorials, news articles
        - The LLM reads each title + snippet and judges educational quality
        - Returns only the top 3 most promising results
        """
        if not results:
            return []

        # If using local model or too few results, skip LLM evaluation
        if self.llm_client.is_local_mode() or len(results) <= 3:
            return results[:settings.WEB_SEARCH_MAX_URLS_TO_SCRAPE]

        # Cache check — avoid redundant LLM evaluation for same topic + result set
        normalized_topic = topic.lower().strip()
        sorted_urls = ",".join(sorted(r.url for r in results[:15]))
        eval_cache_key = get_cache_key("ws_eval", normalized_topic, sorted_urls)
        cached_selected = cache.get(eval_cache_key)
        if cached_selected is not None:
            self.logger.info(f"Stage 3 cache HIT for topic: {topic[:50]}")
            return cached_selected

        # Build a numbered list of results for the LLM to evaluate
        results_text = ""
        for i, r in enumerate(results[:15]):  # Max 15 for prompt size
            results_text += (
                f"{i+1}. Title: {r.title}\n"
                f"   URL: {r.url}\n"
                f"   Snippet: {r.snippet}\n\n"
            )

        system_prompt = (
            "You are an expert at evaluating web search results "
            "for educational quality. You identify which results are most likely to contain "
            "comprehensive, accurate, and educational content about a given topic. You "
            "avoid product pages, news articles, forums, and shallow content."
        )

        prompt = f"""Evaluate these search results for the topic: "{topic}"

Pick the TOP 3 results that are most likely to contain high-quality EDUCATIONAL content. Consider:
- Educational sites (universities, tutorials, documentation) > blogs > forums
- Comprehensive guides > short articles
- Technical accuracy indicators in the snippet
- Reputable sources (GeeksforGeeks, Wikipedia, official docs, Medium tutorials)
- Avoid: product pages, news, forums, Q&A sites, paywalled content

SEARCH RESULTS:
{results_text}

RESPOND WITH ONLY the numbers of your top 3 picks, separated by commas.
Example: 1, 4, 7

YOUR TOP 3 PICKS:"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1,
                system_prompt=system_prompt
            )

            if not response:
                return results[:settings.WEB_SEARCH_MAX_URLS_TO_SCRAPE]

            # Parse the response to get indices
            selected_indices = []
            numbers = re.findall(r'\d+', response)
            for num_str in numbers:
                idx = int(num_str) - 1  # Convert 1-based to 0-based
                if 0 <= idx < len(results):
                    selected_indices.append(idx)

            if not selected_indices:
                return results[:settings.WEB_SEARCH_MAX_URLS_TO_SCRAPE]

            selected = [results[i] for i in selected_indices]
            self.logger.info(
                f"LLM selected {len(selected)} results from "
                f"{len(results)} candidates"
            )
            # Cache successful LLM evaluation (1h TTL from WEB_SEARCH_CACHE_TTL)
            result_to_return = selected[:settings.WEB_SEARCH_MAX_URLS_TO_SCRAPE]
            cache.set(eval_cache_key, result_to_return, expire=settings.WEB_SEARCH_CACHE_TTL)
            return result_to_return

        except Exception as e:
            self.logger.error(f"LLM result evaluation failed: {e}")
            return results[:settings.WEB_SEARCH_MAX_URLS_TO_SCRAPE]

    # =================================================================
    # STAGE 4: CONTENT EXTRACTION (Tool - ScraperAgent)
    # =================================================================

    async def extract_content(
        self,
        results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """
        Use the Scraper tool to fetch content from selected URLs.
        This is a TOOL invocation, no intelligence here.
        """
        scraped = []

        for result in results:
            try:
                scrape_result = await self.scraper.process({
                    'url': result.url
                })

                if (scrape_result.get('success')
                        and scrape_result.get('content')
                        and len(scrape_result['content'])
                            >= settings.WEB_SEARCH_MIN_CONTENT_LENGTH):
                    scraped.append({
                        'url': result.url,
                        'title': scrape_result.get('title', result.title),
                        'content': scrape_result['content'],
                        'snippet': result.snippet,
                        'source': result.source
                    })
                    self.logger.info(
                        f"Scraped: {result.url} "
                        f"({len(scrape_result['content'])} chars)"
                    )
                else:
                    self.logger.warning(
                        f"Scrape returned insufficient content: {result.url}"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Failed to scrape {result.url}: {e}"
                )
                continue

        return scraped

    # =================================================================
    # STAGE 5: CONTENT QUALITY ASSESSMENT (LLM-powered)
    # =================================================================

    def assess_content_quality(
        self,
        topic: str,
        scraped_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to assess whether scraped content is actually useful.

        Why this matters:
        - Some pages scrape successfully but the content is garbage:
          * Navigation menus and footer text
          * Cookie consent / paywall messages
          * Completely off-topic content
          * Very thin content dressed up with ads
        - The LLM reads the first ~500 chars of each scraped page
          and decides if it's worth using
        """
        if not scraped_items:
            return []

        # If local mode, skip quality check (just check length)
        if self.llm_client.is_local_mode():
            return [
                item for item in scraped_items
                if len(item['content']) >= settings.WEB_SEARCH_MIN_CONTENT_LENGTH
            ]

        quality_items = []

        for item in scraped_items:
            # Take first 500 chars for quick assessment
            preview = item['content'][:500]

            prompt = f"""Is this content useful educational material about "{topic}"? Read the preview and answer YES or NO with a brief reason.

CONTENT PREVIEW:
{preview}

Answer (YES/NO + reason):"""

            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0.1
                )

                if response and 'YES' in response.upper():
                    quality_items.append(item)
                    self.logger.info(
                        f"Quality check PASSED: {item['url']}"
                    )
                else:
                    self.logger.info(
                        f"Quality check FAILED: {item['url']} "
                        f"- Reason: {response}"
                    )

            except Exception as e:
                # If quality check fails, include the item anyway
                self.logger.warning(
                    f"Quality check error for {item['url']}: {e}, "
                    f"including anyway"
                )
                quality_items.append(item)

        # If all items failed quality check, return all anyway
        # (better to have something than nothing)
        if not quality_items and scraped_items:
            self.logger.warning(
                "All items failed quality check, "
                "returning all scraped content as fallback"
            )
            return scraped_items

        return quality_items

    # =================================================================
    # MAIN PROCESS METHOD (Full Pipeline)
    # =================================================================

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full web search pipeline.

        This is the main entry point called by the Orchestrator.

        Args:
            input_data: Dict with keys:
                - 'query': The user's topic/query string (required)

        Returns:
            Dict with keys:
                - 'success': bool
                - 'content': Combined content from all sources
                - 'sources': List of source dicts (title, url)
                - 'search_queries': Queries that were used
                - 'results_found': Total search results found
                - 'results_selected': Results selected by LLM
                - 'results_scraped': Successfully scraped count
                - 'agent': "WebSearchAgent"
        """
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))

            topic = input_data.get('query', '')
            if not topic:
                return self.handle_error(ValueError("No query provided"))

            self.logger.info(f"Starting web search for: {topic}")

            # STAGE 1: Generate search queries (LLM)
            self.logger.info("Stage 1: Generating search queries...")
            queries = self.generate_search_queries(topic)
            self.logger.info(f"Generated {len(queries)} queries: {queries}")

            # STAGE 2: Execute search (Tool, with automatic fallback)
            self.logger.info("Stage 2: Executing web search...")
            search_results = self.execute_search(queries, original_topic=topic)
            if not search_results:
                return {
                    'success': False,
                    'error': 'No search results found. The search API may be temporarily unavailable.',
                    'agent': self.name,
                    'search_queries': queries
                }
            self.logger.info(
                f"Found {len(search_results)} unique results"
            )

            # STAGE 3: Evaluate results (LLM)
            self.logger.info("Stage 3: Evaluating results with LLM...")
            selected_results = self.evaluate_results(topic, search_results)
            self.logger.info(
                f"Selected {len(selected_results)} top results"
            )

            # STAGE 4: Extract content (Tool/Scraper)
            self.logger.info("Stage 4: Extracting content from URLs...")
            scraped_items = await self.extract_content(selected_results)
            if not scraped_items:
                return {
                    'success': False,
                    'error': 'Failed to extract content from search results. The websites may be inaccessible.',
                    'agent': self.name,
                    'search_queries': queries,
                    'results_found': len(search_results)
                }
            self.logger.info(
                f"Successfully scraped {len(scraped_items)} pages"
            )

            # STAGE 5: Assess quality (LLM)
            self.logger.info("Stage 5: Assessing content quality...")
            quality_items = self.assess_content_quality(
                topic, scraped_items
            )
            self.logger.info(
                f"Quality check: {len(quality_items)}/{len(scraped_items)} "
                f"items passed"
            )

            # Combine content from all quality sources
            combined_content = "\n\n---\n\n".join(
                item['content'] for item in quality_items
            )

            # Build sources list (title + url for reference in notes)
            sources = [
                {'title': item['title'], 'url': item['url']}
                for item in quality_items
            ]

            self.logger.info(
                f"Web search complete. "
                f"Returning {len(quality_items)} sources, "
                f"{len(combined_content)} chars"
            )

            return {
                'success': True,
                'content': combined_content,
                'sources': sources,
                'search_queries': queries,
                'results_found': len(search_results),
                'results_selected': len(selected_results),
                'results_scraped': len(quality_items),
                'agent': self.name
            }

        except Exception as e:
            self.logger.error(f"Web search agent error: {e}")
            return self.handle_error(e)

    # =================================================================
    # CONVENIENCE METHOD FOR DIRECT SEARCH
    # =================================================================

    async def search_and_get_content(
        self, topic: str
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method for quick search.
        Called directly by orchestrator.

        Args:
            topic: Topic to search for

        Returns:
            Dict with 'content' and 'sources', or None if failed
        """
        result = await self.process({'query': topic})
        if result.get('success'):
            return result
        return None
