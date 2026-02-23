"""
Web scraping agent for fetching content from URLs
"""
import asyncio
import aiohttp
from newspaper import Article
from bs4 import BeautifulSoup
import requests
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.base import BaseAgent
from src.utils.cache_utils import cached
from config import settings

class ScraperAgent(BaseAgent):
    """Agent for scraping web content"""

    def __init__(self):
        super().__init__("ScraperAgent")
        # Browser-like headers to avoid 403 Forbidden from sites (Medium, etc.)
        # that block bot/custom User-Agents like "EduNotes/1.0"
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/131.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    @cached("scraper", ttl=settings.CACHE_SCRAPE_TTL)
    def scrape_with_newspaper(self, url: str) -> Dict[str, Any]:
        """Scrape article using newspaper3k"""
        try:
            from newspaper import Config
            config = Config()
            config.browser_user_agent = self.headers['User-Agent']
            config.request_timeout = settings.SCRAPER_TIMEOUT
            config.fetch_images = False

            article = Article(url, config=config)
            article.download()
            article.parse()
            article.nlp()
            
            return {
                'success': True,
                'content': article.text,
                'title': article.title,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'keywords': article.keywords,
                'summary': article.summary,
                'top_image': article.top_image or '',
                'url': url,
                'scraper': 'newspaper3k'
            }
        except Exception as e:
            self.logger.warning(f"Newspaper3k failed for {url}: {e}")
            return None
    
    def _parse_html_content(self, html: str, url: str, scraper_name: str) -> Dict[str, Any]:
        """Shared HTML parsing logic for BeautifulSoup-based scrapers."""
        soup = BeautifulSoup(html, 'lxml')

        # Remove scripts, styles, navigation, ads, and other noise
        for element in soup(["script", "style", "nav", "header", "footer", "aside",
                           "button", "form", "input", "select", "option", "menu",
                           "iframe", "embed", "object", "noscript"]):
            element.decompose()

        # Remove elements with navigation/menu classes and IDs
        for element in soup.find_all(class_=lambda x: x and any(nav_word in x.lower()
                                   for nav_word in ['nav', 'menu', 'sidebar', 'header', 'footer', 'ad', 'banner', 'breadcrumb'])):
            element.decompose()

        for element in soup.find_all(id=lambda x: x and any(nav_word in x.lower()
                                   for nav_word in ['nav', 'menu', 'sidebar', 'header', 'footer', 'ad', 'banner', 'breadcrumb'])):
            element.decompose()

        # Try to find main content area first
        main_content = None
        content_selectors = [
            'article', 'main', '.content', '.post-content', '.entry-content',
            '.article-content', '.blog-content', '#content', '#main-content',
            '.post-body', '.entry-body', '.article-body'
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content and len(main_content.get_text().strip()) > 500:
                break

        # Use main content if found, otherwise use body
        content_element = main_content if main_content else soup.find('body')

        if content_element:
            # Extract text from main content area only
            text = content_element.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
        else:
            text = "Unable to extract content"

        # Try to extract title
        title_tag = soup.find('title')
        title = title_tag.string if title_tag else "Untitled"

        # Extract top image from og:image meta tag
        top_image = ''
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            top_image = og_image['content']

        return {
            'success': True,
            'content': text,
            'title': title,
            'top_image': top_image,
            'url': url,
            'scraper': scraper_name
        }

    def scrape_with_beautifulsoup(self, url: str) -> Dict[str, Any]:
        """Fallback scraper using BeautifulSoup"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=settings.SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            return self._parse_html_content(response.content, url, 'beautifulsoup')
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                self.logger.warning(f"BeautifulSoup got 403 for {url}, trying primp...")
                return self._scrape_with_primp(url)
            self.logger.error(f"BeautifulSoup failed for {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"BeautifulSoup failed for {url}: {e}")
            return None

    def _scrape_with_primp(self, url: str) -> Dict[str, Any]:
        """Scrape using primp with browser TLS fingerprinting.

        Sites like Medium block Python's `requests` library via TLS
        fingerprinting (they detect the TLS handshake is not from a real
        browser). primp uses libcurl with browser impersonation to bypass
        this. It is already installed as a dependency of the ddgs package.
        """
        try:
            import primp
            client = primp.Client(
                impersonate="chrome_131",
                follow_redirects=True,
                timeout=settings.SCRAPER_TIMEOUT,
            )
            response = client.get(url)
            if response.status_code != 200:
                self.logger.error(f"primp got {response.status_code} for {url}")
                return None
            self.logger.info(f"primp successfully fetched {url} ({len(response.text)} chars)")
            return self._parse_html_content(response.text, url, 'primp')
        except Exception as e:
            self.logger.error(f"primp failed for {url}: {e}")
            return None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process scraping request"""
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))
            
            url = input_data.get('url')
            if not url:
                return self.handle_error(ValueError("No URL provided"))
            
            # Try newspaper3k first
            result = self.scrape_with_newspaper(url)
            
            # Fallback to BeautifulSoup if needed
            if not result or not result.get('success'):
                self.logger.info(f"Falling back to BeautifulSoup for {url}")
                result = self.scrape_with_beautifulsoup(url)
            
            if result and result.get('success'):
                result['date_scraped'] = datetime.now().isoformat()
                self.logger.info(f"Successfully scraped {url}")
                return result
            else:
                return self.handle_error(Exception(f"Failed to scrape {url}"))
                
        except Exception as e:
            return self.handle_error(e)
    
    async def scrape_multiple(self, urls: list) -> list:
        """Scrape multiple URLs concurrently"""
        tasks = [self.process({'url': url}) for url in urls]
        results = await asyncio.gather(*tasks)
        return results