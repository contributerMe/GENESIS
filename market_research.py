"""
Market Research Module
Main orchestrator that combines web scraping and RAG system for comprehensive market research
"""

from langchain_openai import ChatOpenAI
import logging
from typing import Dict
from web_scraper import WebScraper
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedMarketResearch:
    """Optimized market research system using web scraping + RAG + minimal LLM calls"""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize market research system
        
        Args:
            openai_api_key: OpenAI API key for LLM and embeddings
        """
        self.scraper = WebScraper()
        self.rag = RAGSystem(openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=openai_api_key
        )
        self.openai_api_key = openai_api_key
    
    def generate_research_report(self, company_name: str, industry: str) -> str:
        """
        Generate comprehensive research report using RAG + minimal LLM calls
        
        Args:
            company_name: Name of the company to research
            industry: Industry sector
            
        Returns:
            Comprehensive market research report
        """
        try:
            # Step 1: Scrape data
            logger.info(f"Starting research for {company_name} in {industry} industry")
            scraped_data = self.scraper.comprehensive_search(company_name, industry)
            
            if not scraped_data:
                return self._generate_error_report("No data could be scraped from web sources.")
            
            # Step 2: Build RAG system
            logger.info("Building RAG knowledge base...")
            if not self.rag.build_vectorstore(scraped_data):
                return self._generate_error_report("Failed to build knowledge base from scraped data.")
            
            if not self.rag.setup_qa_chain(self.llm):
                return self._generate_error_report("Failed to setup question-answering system.")
            
            # Step 3: Extract structured information using RAG
            logger.info("Extracting structured information...")
            research_findings = self._extract_research_findings(company_name, industry)
            
            # Step 4: Generate AI use cases and final report (2 LLM CALLS TOTAL)
            logger.info("Generating AI use cases...")
            ai_use_cases = self._generate_ai_use_cases(company_name, industry, research_findings)
            
            logger.info("Generating final report...")
            final_report = self._generate_final_report(company_name, industry, research_findings, ai_use_cases)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Research report generation failed: {e}")
            return self._generate_error_report(f"An error occurred during report generation: {str(e)}")
    
    def _extract_research_findings(self, company_name: str, industry: str) -> Dict[str, str]:
        """
        Extract structured information using RAG system
        
        Args:
            company_name: Name of the company
            industry: Industry sector
            
        Returns:
            Dictionary of research findings
        """
        research_questions = {
            'company_overview': f"What is {company_name}? Provide company background, history, and business model.",
            'financial_performance': f"What is {company_name}'s financial performance, revenue, and market position?",
            'products_services': f"What products and services does {company_name} offer?",
            'competitors': f"Who are {company_name}'s main competitors in the {industry} industry?",
            'industry_trends': f"What are the current trends and challenges in the {industry} industry?",
            'recent_developments': f"What are the recent news and developments about {company_name}?",
            'challenges': f"What are the main challenges and pain points facing {company_name}?",
            'technology_gaps': f"What technology gaps or digital transformation needs does {company_name} have?"
        }
        
        research_findings = {}
        for key, question in research_questions.items():
            logger.info(f"Querying: {key}")
            research_findings[key] = self.rag.query(question)
        
        return research_findings
    
    def _generate_ai_use_cases(self, company_name: str, industry: str, research_findings: Dict[str, str]) -> str:
        """
        Generate AI use cases based on research findings (1 LLM CALL)
        
        Args:
            company_name: Name of the company
            industry: Industry sector
            research_findings: Extracted research findings
            
        Returns:
            AI use cases and recommendations
        """
        prompt = f"""
        Based on the following research findings about {company_name} in the {industry} industry, generate 5 specific AI use cases that could address their challenges and opportunities.

        Research Findings:
        - Company Overview: {research_findings.get('company_overview', 'N/A')}
        - Challenges: {research_findings.get('challenges', 'N/A')}
        - Technology Gaps: {research_findings.get('technology_gaps', 'N/A')}
        - Industry Trends: {research_findings.get('industry_trends', 'N/A')}

        For each AI use case, provide:
        1. Problem Statement
        2. AI Solution Description
        3. Expected Benefits
        4. Implementation Complexity (Low/Medium/High)
        5. Estimated ROI Timeline
        6. Required Technologies

        Format as a structured list with clear headings.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"AI use case generation failed: {e}")
            return "Unable to generate AI use cases due to processing error."
    
    def _generate_final_report(self, company_name: str, industry: str, research_findings: Dict[str, str], ai_use_cases: str) -> str:
        """
        Generate final comprehensive report (1 LLM CALL)
        
        Args:
            company_name: Name of the company
            industry: Industry sector
            research_findings: Extracted research findings
            ai_use_cases: Generated AI use cases
            
        Returns:
            Final comprehensive report
        """
        prompt = f"""
        Create a comprehensive, executive-ready market research report for {company_name} in the {industry} industry.

        Structure the report with the following sections:

        # Executive Summary
        Provide a concise overview of key findings and recommendations.

        # Company Overview
        {research_findings.get('company_overview', 'N/A')}

        # Financial Performance & Market Position
        {research_findings.get('financial_performance', 'N/A')}

        # Products & Services
        {research_findings.get('products_services', 'N/A')}

        # Competitive Landscape
        {research_findings.get('competitors', 'N/A')}

        # Industry Analysis
        {research_findings.get('industry_trends', 'N/A')}

        # Recent Developments
        {research_findings.get('recent_developments', 'N/A')}

        # Key Challenges Identified
        {research_findings.get('challenges', 'N/A')}

        # AI Use Cases & Recommendations
        {ai_use_cases}

        # Implementation Roadmap
        Provide a prioritized roadmap for implementing the AI solutions.

        # Expected ROI & Benefits
        Summarize the expected return on investment and key benefits.

        # Next Steps & Recommendations
        Provide specific, actionable next steps.

        Format this as a professional report suitable for C-level executives. Use clear headings and bullet points where appropriate.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
            return f"Report generation encountered an error: {str(e)}"
    
    def _generate_error_report(self, error_message: str) -> str:
        """
        Generate a basic error report when the main process fails
        
        Args:
            error_message: Description of the error
            
        Returns:
            Error report string
        """
        return f"""
        # Market Research Report - Error

        ## Error Summary
        {error_message}

        ## Recommendations
        - Check your internet connection
        - Verify the company name and industry are correct
        - Ensure OpenAI API key is valid and has sufficient credits
        - Try again in a few minutes

        ## Support
        If the problem persists, please check the application logs for more detailed error information.
        """
    
    def get_system_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current research system state
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "rag_initialized": self.rag.vectorstore is not None,
            "qa_chain_ready": self.rag.qa_chain is not None,
        }
        
        if self.rag.vectorstore:
            stats.update(self.rag.get_vectorstore_stats())
        
        return stats