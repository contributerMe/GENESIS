
import logging
from typing import Dict, Any
from ai.state import ResearchState
from ai.tools.firecrawl_scraper import ModernWebScraper
from ai.tools.user_doc_parser import UserDocumentParser

logger = logging.getLogger(__name__)

def researcher_node(state: ResearchState) -> ResearchState:
    """
    LangGraph Node: Executes web search & content scraping + user document ingestion.
    """
    logger.info("--- [Agent Node: Researcher] Starting data acquisition & document ingestion ---")
    inputs_dict = state.get("inputs", {})
    company_name = inputs_dict.get("company_name", "")
    industry = inputs_dict.get("industry", "")
    uploaded_files = inputs_dict.get("uploaded_files", [])

    uploaded_docs_list = []
    if uploaded_files:
        logger.info(f"Ingesting {len(uploaded_files)} user-supplied documents...")
        parsed_user_docs = UserDocumentParser.parse_multiple(uploaded_files)
        uploaded_docs_list = [d.model_dump() for d in parsed_user_docs]

    try:
        scraper = ModernWebScraper()
        scraped_docs = scraper.search_and_scrape(company_name, industry)
        logger.info(f"Researcher Node completed. Web docs: {len(scraped_docs)}, User uploads: {len(uploaded_docs_list)}")
        return {
            "scraped_documents": scraped_docs,
            "uploaded_documents": uploaded_docs_list,
            "current_step": "researcher_completed"
        }
    except Exception as e:
        logger.error(f"Researcher Node failed: {e}")
        return {
            "scraped_documents": [],
            "uploaded_documents": uploaded_docs_list,
            "errors": state.get("errors", []) + [f"Researcher: {e}"],
            "current_step": "researcher_failed"
        }

