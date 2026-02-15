"""
Search provider abstraction layer.
This is a TOOL, not an agent - it takes a query and returns raw search results.
No intelligence, no filtering, no decisions.

Supports:
- DuckDuckGo (free, no API key, default)
- Serper.dev (premium, Google results, stubbed for future)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Single search result"""
    title: str
    url: str
    snippet: str
    source: str  # "duckduckgo" or "serper"


class BaseSearchProvider(ABC):
    """Abstract base for search providers"""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Execute a search query and return results"""
        pass


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider (free, no API key).

    Uses the `ddgs` package (v9+) which has multi-engine support
    (Google, Brave, Wikipedia, Yahoo, DuckDuckGo, etc.).
    Falls back to the deprecated `duckduckgo_search` package if `ddgs` is not installed.
    """

    def __init__(self):
        try:
            from ddgs import DDGS
            self._ddgs_class = DDGS
            self._is_v9 = True
            logger.info("DDGS v9+ search provider initialized (multi-engine)")
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                self._ddgs_class = DDGS
                self._is_v9 = False
                logger.info("DuckDuckGo search provider initialized (legacy v8)")
            except ImportError:
                logger.error("Neither ddgs nor duckduckgo-search installed. Run: pip install ddgs")
                raise

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search using DDGS and return results.

        Creates a fresh DDGS instance per call to avoid stale session state
        and cookie contamination that causes silent 0-result failures.

        Returns list of dicts: [{"title": "...", "href": "...", "body": "..."}, ...]
        """
        # wt-wt is a DDG-specific region code not recognized by other engines.
        # Convert to us-en (ddgs v9 default) for multi-engine compatibility.
        region = settings.SEARCH_REGION
        if region == "wt-wt":
            region = "us-en"

        last_error = None
        for attempt in range(2):
            try:
                ddgs = self._ddgs_class()
                raw_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=settings.SEARCH_SAFE_SEARCH
                ))

                if raw_results:
                    results = []
                    for r in raw_results:
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            source="duckduckgo"
                        ))
                    logger.info(f"DuckDuckGo returned {len(results)} results for: {query[:50]}")
                    return results

                # Empty results on first attempt — retry with fresh instance
                if attempt == 0:
                    logger.info(f"DuckDuckGo returned 0 results (attempt 1), retrying...")
                    import time
                    time.sleep(2)

            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(f"DuckDuckGo search attempt 1 failed: {e}, retrying...")
                    import time
                    time.sleep(2)

        if last_error:
            logger.error(f"DuckDuckGo search failed after 2 attempts: {last_error}")
        else:
            logger.warning(f"DuckDuckGo returned 0 results after 2 attempts for: {query[:50]}")
        return []


class SerperProvider(BaseSearchProvider):
    """Serper.dev search provider (Google results, 2500 free/month).
    Stubbed for future implementation.
    """

    def __init__(self):
        if not settings.SERPER_API_KEY:
            raise ValueError(
                "SERPER_API_KEY not configured. Get key at: https://serper.dev"
            )
        logger.info("Serper search provider initialized")

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Serper.dev API (Google results)."""
        try:
            import requests as req

            headers = {
                "X-API-KEY": settings.SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": max_results
            }

            response = req.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("organic", []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    source="serper"
                ))

            logger.info(f"Serper returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []


class GoogleScraperProvider(BaseSearchProvider):
    """Google search via googlesearch-python (free, no API key).
    Used as automatic fallback when DuckDuckGo returns no results.

    Package: pip install googlesearch-python
    Uses Google's search page directly — better coverage for niche/technical
    queries where DuckDuckGo's unofficial API often returns 0 results.
    """

    def __init__(self):
        try:
            import googlesearch
            # Fix: The library's default user agent mimics Lynx (text browser),
            # which causes Google to serve a "JavaScript required" page instead
            # of actual search results. Override with a real browser UA.
            googlesearch.get_useragent = lambda: (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
            self._search_fn = googlesearch.search
            logger.info("Google scraper fallback provider initialized")
        except ImportError:
            logger.warning(
                "googlesearch-python not installed. "
                "Fallback search unavailable. "
                "Run: pip install googlesearch-python"
            )
            raise

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search Google via scraping and return results.

        Uses advanced=True to get title + url + description per result.
        """
        try:
            raw_results = list(self._search_fn(
                query,
                num_results=max_results,
                advanced=True,
                sleep_interval=1
            ))

            results = []
            for r in raw_results:
                results.append(SearchResult(
                    title=getattr(r, 'title', '') or '',
                    url=getattr(r, 'url', '') or '',
                    snippet=getattr(r, 'description', '') or '',
                    source="google"
                ))

            logger.info(f"Google returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []


def get_search_provider() -> BaseSearchProvider:
    """Get the configured search provider"""
    provider = settings.SEARCH_PROVIDER

    if provider == "duckduckgo":
        return DuckDuckGoProvider()
    elif provider == "serper":
        return SerperProvider()
    else:
        logger.warning(f"Unknown search provider '{provider}', defaulting to DuckDuckGo")
        return DuckDuckGoProvider()


def get_fallback_provider() -> Optional[BaseSearchProvider]:
    """Get fallback search provider (Google scraper) if available and enabled.

    Returns None if:
    - Fallback is disabled in settings
    - googlesearch-python is not installed
    - Primary provider is already Google-based (serper)
    """
    if not getattr(settings, 'SEARCH_FALLBACK_ENABLED', True):
        logger.info("Search fallback disabled in settings")
        return None

    # No point falling back to Google if primary is already Google-based
    if settings.SEARCH_PROVIDER == "serper":
        logger.info("Primary provider is Serper (Google), skipping Google fallback")
        return None

    try:
        return GoogleScraperProvider()
    except (ImportError, Exception) as e:
        logger.info(f"Fallback search provider not available: {e}")
        return None
