"""
Academic paper search via Semantic Scholar and arXiv APIs.
Both APIs are free and require no API keys.
"""
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.utils.cache_utils import cache, get_cache_key

logger = get_logger(__name__)


class AcademicSearch:
    """Search for academic papers using free APIs."""

    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_API = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.logger = logger

    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for related academic papers.

        Tries Semantic Scholar first, falls back to arXiv.
        Results are cached for 24 hours.

        Returns:
            List of paper dicts with: title, authors, year, abstract, url, citations
        """
        cache_key = get_cache_key("academic", query.lower().strip())
        cached = cache.get(cache_key)
        if cached is not None:
            self.logger.info(f"Academic search cache HIT for: {query[:50]}")
            return cached

        papers = []

        # Try Semantic Scholar first
        try:
            papers = self._search_semantic_scholar(query, max_results)
            if papers:
                self.logger.info(f"Semantic Scholar returned {len(papers)} papers")
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search failed: {e}")

        # Fallback to arXiv
        if not papers:
            try:
                papers = self._search_arxiv(query, max_results)
                if papers:
                    self.logger.info(f"arXiv returned {len(papers)} papers")
            except Exception as e:
                self.logger.warning(f"arXiv search also failed: {e}")

        if papers:
            cache.set(cache_key, papers, expire=86400)  # 24h cache

        return papers

    def _search_semantic_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search via Semantic Scholar API (free, no key needed)."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,citationCount,url,externalIds,openAccessPdf"
        }

        resp = requests.get(
            self.SEMANTIC_SCHOLAR_API,
            params=params,
            timeout=10,
            headers={"User-Agent": "EduNotes/2.0 (Academic Research Tool)"}
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            if not item.get("title"):
                continue

            # Build author string
            authors = item.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Get PDF link if available
            pdf_url = ""
            if item.get("openAccessPdf"):
                pdf_url = item["openAccessPdf"].get("url", "")

            # Get arXiv ID if available
            arxiv_id = ""
            ext_ids = item.get("externalIds", {})
            if ext_ids and ext_ids.get("ArXiv"):
                arxiv_id = ext_ids["ArXiv"]

            papers.append({
                "title": item["title"],
                "authors": author_str,
                "year": item.get("year", ""),
                "abstract": (item.get("abstract") or "")[:300],
                "url": item.get("url", ""),
                "pdf_url": pdf_url,
                "citations": item.get("citationCount", 0),
                "arxiv_id": arxiv_id,
                "source": "Semantic Scholar"
            })

        return papers

    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search via arXiv API (free, no key needed)."""
        params = {
            "search_query": f"all:{query}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results
        }

        resp = requests.get(self.ARXIV_API, params=params, timeout=10)
        resp.raise_for_status()

        # Parse Atom XML
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:id", ns)

            # Get authors
            authors = entry.findall("atom:author/atom:name", ns)
            author_str = ", ".join(a.text for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Extract year from published date
            year = ""
            if published is not None and published.text:
                year = published.text[:4]

            # Get PDF link
            pdf_url = ""
            for lnk in entry.findall("atom:link", ns):
                if lnk.get("title") == "pdf":
                    pdf_url = lnk.get("href", "")
                    break

            papers.append({
                "title": title.text.strip() if title is not None else "",
                "authors": author_str,
                "year": year,
                "abstract": (summary.text.strip() if summary is not None else "")[:300],
                "url": link.text.strip() if link is not None else "",
                "pdf_url": pdf_url,
                "citations": 0,  # arXiv doesn't provide citation count
                "arxiv_id": "",
                "source": "arXiv"
            })

        return papers

    def format_papers_markdown(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers list as markdown for appending to notes."""
        if not papers:
            return ""

        lines = ["\n\n---\n\n## Related Research Papers\n"]

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", "Unknown")
            year = paper.get("year", "")
            abstract = paper.get("abstract", "")
            url = paper.get("url", "")
            pdf_url = paper.get("pdf_url", "")
            citations = paper.get("citations", 0)

            # Title line with link
            if url:
                lines.append(f"**{i}. [{title}]({url})**")
            else:
                lines.append(f"**{i}. {title}**")

            # Metadata line
            meta_parts = []
            if year:
                meta_parts.append(str(year))
            if authors:
                meta_parts.append(f"*{authors}*")
            if citations:
                meta_parts.append(f"{citations:,} citations")
            if meta_parts:
                lines.append(" | ".join(meta_parts))

            # Abstract
            if abstract:
                lines.append(f"> {abstract}...")

            # PDF link
            if pdf_url:
                lines.append(f"[PDF]({pdf_url})")

            lines.append("")  # Blank line between papers

        return "\n".join(lines)


# Singleton
_academic_search = None


def get_academic_search() -> AcademicSearch:
    """Get or create AcademicSearch instance."""
    global _academic_search
    if _academic_search is None:
        _academic_search = AcademicSearch()
    return _academic_search
