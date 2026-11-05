"""
Firecrawl & Tavily AI Search Tool Integration
Provides clean Markdown extraction and JS rendering for web scraping.
"""

import logging
from typing import List, Dict, Any, Optional
from ai.settings import get_settings

logger = logging.getLogger(__name__)

class ModernWebScraper:
    """
    Modern Scraper featuring Firecrawl API & Tavily Search integration with fallback capability.
    """

    def __init__(self):
        settings = get_settings()
        self.tavily_key = settings.tavily_api_key
        self.firecrawl_key = settings.firecrawl_api_key

    def search_and_scrape(self, company_name: str, industry: str) -> List[Dict[str, Any]]:
        """
        Execute comprehensive search & scraping using Firecrawl if key is present,
        otherwise falling back to Tavily or FallbackScraper.
        """
        results = []
        
        # 1. Try Firecrawl First
        if self.firecrawl_key:
            try:
                from firecrawl import Firecrawl
                logger.info(f"Executing Firecrawl AI Search & Scrape for {company_name} in {industry}...")
                app = Firecrawl(api_key=self.firecrawl_key)
                
                query = f"{company_name} company overview business model competitive analysis {industry}"
                
                # Use Firecrawl's search functionality which searches and scrapes automatically
                search_res = app.search(
                    query=query,
                    limit=10,
                    scrape_options={"formats": ["markdown"], "only_main_content": True},
                )
                if isinstance(search_res, dict):
                    data = search_res.get("data", search_res)
                    web_results = data.get("web", []) if isinstance(data, dict) else data
                else:
                    web_results = getattr(search_res, "web", [])

                if web_results:
                    for item in web_results:
                        get_value = item.get if isinstance(item, dict) else lambda key, default="": getattr(item, key, default)
                        results.append({
                            "title": get_value("title", f"{company_name} Search Result"),
                            "url": get_value("url", ""),
                            "content": get_value("markdown", "") or get_value("content", ""),
                            "category": "search_result",
                            "source_type": "web"
                        })
                
                if results:
                    logger.info(f"Firecrawl returned {len(results)} high-quality results.")
                    return results

            except Exception as e:
                logger.error(f"Firecrawl search failed: {e}. Falling back to Tavily.")

        # 2. Try Tavily when Firecrawl is unavailable, errors, or returns no data.
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                logger.info(f"Executing Tavily AI Search for {company_name} in {industry}...")
                tavily = TavilyClient(api_key=self.tavily_key)
                
                queries = [
                    f"{company_name} company overview business model",
                    f"{company_name} financial performance revenue annual report",
                    f"{company_name} competitors competitive analysis {industry}",
                    f"{industry} industry trends digital transformation AI",
                    f"{company_name} challenges technology gaps"
                ]

                for query in queries:
                    search_res = tavily.search(query=query, max_results=3)
                    for item in search_res.get("results", []):
                        results.append({
                            "title": item.get("title", f"{company_name} Search Result"),
                            "url": item.get("url", ""),
                            "content": item.get("content", ""),
                            "category": "search_result",
                            "source_type": "web"
                        })
                
                if results:
                    logger.info(f"Tavily Search returned {len(results)} high-quality results.")
                    return results

            except Exception as e:
                logger.error(f"Tavily search failed: {e}. Falling back to standard async crawler.")

        # 3. Fallback to standard scraper
        from ai.tools.fallback_scraper import FallbackScraper
        logger.info("Executing fallback web scraper...")
        fallback_crawler = FallbackScraper()
        return fallback_crawler.comprehensive_search(company_name, industry)
