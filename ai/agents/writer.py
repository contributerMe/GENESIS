"""
Executive Writer Agent Node
Assembles all findings into a C-Level Report Object via litellm.completion().
"""

import logging
from typing import Dict, Any
from ai.state import ResearchState, ResearchReportData, AIUseCase, DatasetReference, SourceCitation
from ai.llm import LiteLLMRouter

logger = logging.getLogger(__name__)

def writer_node(state: ResearchState) -> ResearchState:
    """
    LangGraph Node: Synthesizes final C-level Executive Report via litellm.completion().
    """
    logger.info("--- [Agent Node: Writer] Assembling C-Level Executive Report via LiteLLM ---")
    inputs = state.get("inputs", {})
    findings = state.get("research_findings", {})
    use_cases = state.get("ai_use_cases", [])
    datasets = state.get("datasets", [])
    citations_data = state.get("citations", [])
    
    company_name = inputs.get("company_name", "")
    industry = inputs.get("industry", "")
    provider = inputs.get("provider", "openai")
    model_name = inputs.get("model_name")
    api_key = inputs.get("api_key")

    system_prompt = LiteLLMRouter.load_prompt("writer")
    user_data = {
        "company_name": company_name,
        "industry": industry,
        "research_findings": {
            "company_overview": findings.get('company_overview', 'N/A')[:1000],
            "financial_performance": findings.get('financial_performance', 'N/A')[:1000],
            "competitors": findings.get('competitors', 'N/A')[:1000],
            "challenges": findings.get('challenges', 'N/A')[:1000],
            "technology_gaps": findings.get('technology_gaps', 'N/A')[:1000]
        }
    }
    messages = LiteLLMRouter.build_messages(system_prompt, user_data)

    try:
        report_obj: ResearchReportData = LiteLLMRouter.completion(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            messages=messages,
            response_format=ResearchReportData
        )
        report_obj.ai_use_cases = [AIUseCase(**uc) for uc in use_cases]
        report_obj.datasets_found = [DatasetReference(**ds) for ds in datasets]
        report_obj.citations = [SourceCitation(**c) for c in citations_data]
        report_dict = report_obj.model_dump()
        logger.info("Writer Node successfully synthesized structured C-Level Report via LiteLLM.")
        return {
            "final_report": report_dict,
            "current_step": "writer_completed"
        }
    except Exception as e:
        logger.error(f"Writer Node error via LiteLLM: {e}. Building fallback report data.")
        report_dict = ResearchReportData(
            executive_summary=f"Executive Market Analysis for {company_name} in {industry}.",
            company_overview=findings.get("company_overview", f"{company_name} is a leading enterprise in {industry}."),
            financial_performance=findings.get("financial_performance", "Financial performance remains strong."),
            products_services=findings.get("products_services", "Core offerings and specialized solutions."),
            competitive_landscape=findings.get("competitors", "Competes with major industry players."),
            industry_analysis=f"The {industry} sector is undergoing rapid digital transformation.",
            key_challenges=findings.get("challenges", "Key operational and technology adaptation challenges."),
            ai_use_cases=[AIUseCase(**uc) for uc in use_cases],
            implementation_roadmap="Phase 1: Pilot RAG (0-3 mos). Phase 2: Scale AI Workflows (3-6 mos).",
            datasets_found=[DatasetReference(**ds) for ds in datasets]
        ).model_dump()
        return {
            "final_report": report_dict,
            "errors": state.get("errors", []) + [f"Writer: {e}"],
            "current_step": "writer_fallback"
        }
