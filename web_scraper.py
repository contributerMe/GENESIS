"""
Web Scraper Module for Market Research
Handles Google search and content extraction from web pages
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import quote_plus
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper for market research data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_google(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search Google and return results
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with keys: title, url, snippet
        """
        try:
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='g')[:num_results]:
                title_elem = result.find('h3')
                link_elem = result.find('a')
                snippet_elem = result.find('span', class_=['aCOpRe', 'st'])
                
                if title_elem and link_elem:
                    title = title_elem.get_text()
                    link = link_elem.get('href')
                    snippet = snippet_elem.get_text() if snippet_elem else ""
                    
                    if link.startswith('http'):
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def scrape_content(self, url: str) -> str:
        """
        Scrape content from a URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Cleaned text content from the webpage
        """
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract main content using multiple selectors
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.entry-content', '.article-content', 'p'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            if not content:
                content = soup.get_text()
            
            # Clean and limit content
            content = re.sub(r'\s+', ' ', content).strip()
            print(content[:500])  # Print first 500 characters for debugging
            return content
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return ""
    
    def comprehensive_search(self, company_name: str, industry: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Perform comprehensive search for company and industry data
        
        Args:
            company_name: Name of the company to research
            industry: Industry sector
            
        Returns:
            Dictionary with categorized search results and content
        """
        search_queries = {
            'company_overview': f"{company_name} company overview business model",
            'financial_info': f"{company_name} revenue financial performance annual report",
            'recent_news': f"{company_name} news 2024 2025 recent developments",
            'competitors': f"{company_name} competitors competitive analysis {industry}",
            'industry_trends': f"{industry} industry trends 2024 2025 market analysis",
            'challenges': f"{company_name} challenges problems issues {industry}",
            'technology': f"{company_name} technology digital transformation AI automation"
        }
        
        all_content = {}
        
        for category, query in search_queries.items():
            logger.info(f"Searching for: {category}")
            search_results = self.search_google(query, 5)
            
            category_content = []
            for result in search_results:
                content = self.scrape_content(result['url'])
                if content:
                    category_content.append({
                        'title': result['title'],
                        'url': result['url'],
                        'content': content
                    })
                time.sleep(1)  # Rate limiting
            
            all_content[category] = category_content
        print(all_content) # for debugging
        return all_content
    
if __name__ == "__main__":
    scraper = WebScraper()
    company_name = "OpenAI"
    industry = "Artificial Intelligence"
    
    results = scraper.comprehensive_search(company_name, industry)
    print(results)  # Print the results for debugging    