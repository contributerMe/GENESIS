"""
Fallback Web Scraper (Requests + BeautifulSoup)
Provides reliable web search & page scraping without third-party API keys.
"""

import re
import time
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class FallbackScraper:
    """
    Fallback Web Scraper using requests + BeautifulSoup with modern user-agent rotation.
    """

    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }

    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo HTML API (more resilient to blocking than Google).
        """
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(search_url, headers=self._get_headers(), timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            results = []
            for result in soup.find_all('a', class_='result__url')[:num_results]:
                link = result.get('href')
                title_elem = result.find_parent('div', class_='result__body')
                title = title_elem.find('a', class_='result__a').get_text() if title_elem else "Search Result"

                if link and link.startswith('http'):
                    results.append({'title': title.strip(), 'url': link.strip()})

            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed for query '{query}': {e}")
            return []

    def scrape_url(self, url: str) -> str:
        """
        Extract clean text content from web page.
        """
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
                tag.decompose()

            selectors = ['article', 'main', '.content', '.article-body', 'p']
            extracted_text = ""

            for sel in selectors:
                elements = soup.select(sel)
                if elements:
                    extracted_text = " ".join([e.get_text().strip() for e in elements])
                    break

            if not extracted_text:
                extracted_text = soup.get_text()

            cleaned = re.sub(r'\s+', ' ', extracted_text).strip()
            return cleaned[:5000]

        except Exception as e:
            logger.error(f"Failed to scrape URL '{url}': {e}")
            return ""

    def comprehensive_search(self, company_name: str, industry: str) -> List[Dict[str, Any]]:
        """
        Execute full market search across company categories.
        """
        logger.info(f"Executing Fallback Search for {company_name} in {industry}...")
        queries = {
            'overview': f"{company_name} company overview business model",
            'financial': f"{company_name} revenue financial performance annual report",
            'competitors': f"{company_name} competitors competitive analysis {industry}",
            'trends': f"{industry} industry trends digital transformation AI",
            'challenges': f"{company_name} challenges technology gaps"
        }

        all_scraped = []
        for category, q in queries.items():
            search_items = self.search_duckduckgo(q, num_results=3)
            for item in search_items:
                content = self.scrape_url(item['url'])
                if content:
                    all_scraped.append({
                        "title": item['title'],
                        "url": item['url'],
                        "content": content,
                        "category": category,
                        "source_type": "web"
                    })
                time.sleep(0.5)

        logger.info(f"Fallback Search collected {len(all_scraped)} web pages.")
        return all_scraped
