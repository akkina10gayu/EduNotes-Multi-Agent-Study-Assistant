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
        self.headers = {
            'User-Agent': settings.SCRAPER_USER_AGENT
        }
    
    @cached("scraper", ttl=settings.CACHE_SCRAPE_TTL)
    def scrape_with_newspaper(self, url: str) -> Dict[str, Any]:
        """Scrape article using newspaper3k"""
        try:
            article = Article(url)
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
                'url': url,
                'scraper': 'newspaper3k'
            }
        except Exception as e:
            self.logger.warning(f"Newspaper3k failed for {url}: {e}")
            return None
    
    def scrape_with_beautifulsoup(self, url: str) -> Dict[str, Any]:
        """Fallback scraper using BeautifulSoup"""
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=settings.SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
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
            title = soup.find('title')
            title = title.string if title else "Untitled"
            
            return {
                'success': True,
                'content': text,
                'title': title,
                'url': url,
                'scraper': 'beautifulsoup'
            }
        except Exception as e:
            self.logger.error(f"BeautifulSoup failed for {url}: {e}")
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