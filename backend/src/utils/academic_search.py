"""
Academic paper search via OpenAlex, arXiv, and Semantic Scholar APIs.
All APIs are free and require no API keys.
Includes retry logic with exponential backoff to handle 429 rate limits.
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from src.utils.logger import get_logger
from src.utils.cache_utils import cache, get_cache_key

logger = get_logger(__name__)


class AcademicSearch:
    """Search for academic papers using free APIs."""

    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
    OPENALEX_API = "https://api.openalex.org/works"
    ARXIV_API = "https://export.arxiv.org/api/query"

    def __init__(self):
        self.logger = logger
        self._last_request_time = {}  # Per-host request timestamps for rate limiting

    # Minimum seconds between requests to each API host
    _HOST_MIN_INTERVAL = {
        'api.semanticscholar.org': 1.1,   # ~1 req/sec for unauthenticated
        'export.arxiv.org': 3.5,           # arXiv requires 3s between requests
        'api.openalex.org': 0.15,          # Very generous rate limits
    }

    def _enforce_rate_limit(self, url: str):
        """Enforce minimum interval between requests to the same API host."""
        host = url.split('/')[2]
        min_interval = self._HOST_MIN_INTERVAL.get(host, 0.5)
        now = time.time()
        last = self._last_request_time.get(host, 0)
        wait = min_interval - (now - last)
        if wait > 0:
            self.logger.debug(f"Rate limit: waiting {wait:.1f}s before {host} request")
            time.sleep(wait)
        self._last_request_time[host] = time.time()

    def _extract_topic_from_title(self, title: str) -> str:
        """Extract the core topic from a webpage title, stripping brand/nav noise.

        Page titles often contain separators with brand names or navigation:
            "Aman's AI Journal • Primers • Vision Language Models"
            "Understanding Transformers | Towards Data Science"
            "BERT Explained - My ML Blog"

        Returns the most topic-relevant segment.
        """
        separator_pattern = r'\s*[•·|—–»«]\s*|\s+-\s+|\s*::\s*'
        if not re.search(separator_pattern, title):
            return title.strip()

        parts = re.split(separator_pattern, title)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]

        if not parts:
            return title.strip()
        if len(parts) == 1:
            return parts[0]

        # Score each segment: more content words = more likely the actual topic.
        # Penalize possessives which often indicate brand names ("Aman's Blog").
        def _topic_score(s):
            words = s.split()
            content_words = [w for w in words if len(w) > 2]
            score = len(content_words)
            if any("\u2019" in w or "'" in w for w in words):
                score *= 0.3
            return score

        if len(parts) == 2:
            # Two-part titles usually follow "Topic | Brand" convention.
            # Prefer the first segment unless the second scores notably higher.
            s0, s1 = _topic_score(parts[0]), _topic_score(parts[1])
            return parts[1] if s1 > s0 * 1.5 else parts[0]

        return max(parts, key=_topic_score)

    def extract_search_query(self, title: str, url: str = "") -> str:
        """Build an academic search query from a page title and optional URL.

        For URL inputs, the URL path often contains a clean topic slug
        (e.g., /primers/ai/vision-language-models/ -> "vision language models").
        Falls back to extracting the topic from the page title.
        """
        if url:
            try:
                path = urlparse(url).path.strip('/')
                if path:
                    segments = [s for s in path.split('/') if s and len(s) > 2]
                    if segments:
                        topic = segments[-1].replace('-', ' ').replace('_', ' ')
                        topic = re.sub(r'\.\w+$', '', topic).strip()
                        if len(topic) >= 5 and any(c.isalpha() for c in topic):
                            return topic
            except Exception:
                pass

        return self._clean_search_query(title)

    def _clean_search_query(self, query: str) -> str:
        """Clean and shorten query for academic search APIs.

        Handles page titles with brand/navigation separators (bullet, pipe,
        dash, etc.), strips markdown/special chars, and limits to ~60 chars
        ending at a word boundary.
        """
        # Step 1: Extract topic from separator-delimited titles
        query = self._extract_topic_from_title(query)

        # Step 2: Strip markdown, special chars, and unicode punctuation
        for ch in '#*[](){}|`~><\u2022\u00b7\u2014\u2013\u00bb\u00ab\u2018\u2019\n\r\t':
            query = query.replace(ch, ' ')
        query = ' '.join(query.split()).strip()

        # Step 3: Limit length at word boundary
        if len(query) > 60:
            cut = query.rfind(' ', 0, 60)
            query = query[:cut] if cut > 20 else query[:60]
        return query.strip()

    def _request_with_retry(self, url: str, params: dict,
                            headers: dict = None,
                            max_retries: int = 3) -> requests.Response:
        """HTTP GET with per-host rate limiting and retry on 429."""
        for attempt in range(max_retries + 1):
            self._enforce_rate_limit(url)
            resp = requests.get(
                url, params=params, timeout=15, headers=headers or {}
            )
            if resp.status_code == 429 and attempt < max_retries:
                # Respect Retry-After header if present
                retry_after = resp.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait = min(float(retry_after), 30)
                    except (ValueError, TypeError):
                        wait = 3 * (2 ** attempt)  # 3, 6, 12
                else:
                    wait = 3 * (2 ** attempt)  # 3, 6, 12
                host = url.split('/')[2]
                self.logger.info(
                    f"Rate limited (429) by {host}, "
                    f"retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        # Unreachable — loop always returns or raises
        raise requests.exceptions.HTTPError("Max retries exceeded")

    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for related academic papers.

        Tries OpenAlex -> arXiv -> Semantic Scholar, with retry on 429.
        Results are cached for 24 hours.

        Returns:
            List of paper dicts with: title, authors, year, abstract, url, citations
        """
        cache_key = get_cache_key("academic", query.lower().strip())
        cached = cache.get(cache_key)
        if cached is not None:
            self.logger.info(f"Academic search cache HIT for: {query[:50]}")
            return cached

        clean_query = self._clean_search_query(query)
        if len(clean_query) < 5:
            self.logger.warning(f"Query too short for academic search: '{clean_query}'")
            return []

        self.logger.info(f"Academic search query: '{clean_query}'")
        papers = []

        # OpenAlex first (broadest coverage, best reliability, good relevance)
        try:
            papers = self._search_openalex(clean_query, max_results)
            if papers:
                self.logger.info(f"OpenAlex returned {len(papers)} papers")
        except Exception as e:
            self.logger.warning(f"OpenAlex search failed: {e}")

        # Fallback to arXiv (strong for CS/AI/ML preprints, relevance-sorted)
        if not papers:
            try:
                papers = self._search_arxiv(clean_query, max_results)
                if papers:
                    self.logger.info(f"arXiv returned {len(papers)} papers")
            except Exception as e:
                self.logger.warning(f"arXiv search failed: {e}")

        # Fallback to Semantic Scholar (good metadata but strict rate limits)
        if not papers:
            try:
                papers = self._search_semantic_scholar(clean_query, max_results)
                if papers:
                    self.logger.info(f"Semantic Scholar returned {len(papers)} papers")
            except Exception as e:
                self.logger.warning(f"Semantic Scholar search failed: {e}")

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
        headers = {"User-Agent": "EduNotes/2.0 (Academic Research Tool)"}

        resp = self._request_with_retry(
            self.SEMANTIC_SCHOLAR_API, params, headers
        )
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

    def _search_openalex(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search via OpenAlex API (free, generous rate limits)."""
        params = {
            "search": query,
            "per_page": max_results,
            "mailto": "edunotes@research.app",  # Polite pool: faster responses
        }
        headers = {"User-Agent": "EduNotes/2.0 (Academic Research Tool)"}

        resp = self._request_with_retry(self.OPENALEX_API, params, headers)
        data = resp.json()

        papers = []
        for item in data.get("results", []):
            if not item.get("title"):
                continue

            # Build author string
            authorships = item.get("authorships", [])
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in authorships[:3]
            ]
            author_str = ", ".join(a for a in authors if a)
            if len(authorships) > 3:
                author_str += " et al."

            # Get URL (DOI preferred, then landing page)
            url = ""
            if item.get("doi"):
                url = item["doi"]
            elif item.get("primary_location"):
                url = item["primary_location"].get("landing_page_url", "") or ""
            if not url:
                url = item.get("id", "")

            # Get open access PDF URL
            pdf_url = ""
            oa = item.get("open_access") or {}
            if oa.get("oa_url"):
                pdf_url = oa["oa_url"]

            # Reconstruct abstract from inverted index
            abstract = self._reconstruct_abstract(
                item.get("abstract_inverted_index")
            )

            papers.append({
                "title": item["title"],
                "authors": author_str,
                "year": item.get("publication_year", ""),
                "abstract": abstract[:300],
                "url": url,
                "pdf_url": pdf_url,
                "citations": item.get("cited_by_count", 0),
                "arxiv_id": "",
                "source": "OpenAlex"
            })

        return papers

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort()
        return " ".join(w for _, w in words)

    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search via arXiv API (free, no key needed)."""
        params = {
            "search_query": f"all:{query}",
            "sortBy": "relevance",
            "max_results": max_results,
        }
        headers = {"User-Agent": "EduNotes/2.0 (Academic Research Tool)"}

        resp = self._request_with_retry(self.ARXIV_API, params, headers)

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

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape markdown-breaking characters in text from external APIs.

        Academic paper titles and abstracts commonly contain characters like
        < > * _ [ ] (e.g., 'p < 0.05', 'CNN vs ViT', 'method [1]') that
        break Streamlit's markdown parser, causing the entire notes document
        to render as raw text.
        """
        # Escape backslash first to avoid double-escaping
        text = text.replace('\\', '\\\\')
        for ch in ('*', '_', '`', '<', '>', '[', ']', '#', '~'):
            text = text.replace(ch, '\\' + ch)
        return text

    def format_papers_markdown(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers list as markdown for appending to notes."""
        if not papers:
            return ""

        lines = ["\n\n---\n\n## Related Research Papers\n"]

        for i, paper in enumerate(papers, 1):
            title = self._escape_md(paper.get("title", "Untitled"))
            authors = self._escape_md(paper.get("authors", "Unknown"))
            year = paper.get("year", "")
            # Clean abstract: escape markdown chars + collapse newlines
            abstract = paper.get("abstract", "").replace('\n', ' ')
            abstract = self._escape_md(abstract)
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

            # Abstract (blank line before blockquote for reliable parsing)
            if abstract:
                lines.append("")
                lines.append(f"> {abstract}...")

            # PDF link
            if pdf_url:
                lines.append("")
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
