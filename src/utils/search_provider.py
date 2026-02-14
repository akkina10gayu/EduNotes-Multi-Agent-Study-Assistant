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
from typing import List
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
    """DuckDuckGo search provider (free, no API key)"""

    def __init__(self):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            logger.info("DuckDuckGo search provider initialized")
        except ImportError:
            logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            raise

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search DuckDuckGo and return results.

        Uses DDGS().text() which returns list of dicts:
        [{"title": "...", "href": "...", "body": "..."}, ...]
        """
        try:
            raw_results = self.ddgs.text(
                query,
                max_results=max_results,
                region=settings.SEARCH_REGION,
                safesearch=settings.SEARCH_SAFE_SEARCH
            )

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

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
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
