"""
Strategist Agent Node
Synthesizes 5 prioritized AI Use Cases using litellm.completion() with Pydantic JSON schema outputs.
"""

import logging
from typing import Dict, Any, List
from ai.state import ResearchState, AIUseCasesResponse, AIUseCase
from ai.llm import LiteLLMRouter

logger = logging.getLogger(__name__)

def strategist_node(state: ResearchState) -> ResearchState:
    """
    LangGraph Node: Generates 5 tailored AI Use Cases via litellm.completion().
    """
    logger.info("--- [Agent Node: Strategist] Synthesizing AI Use Cases via LiteLLM ---")
    inputs = state.get("inputs", {})
    findings = state.get("research_findings", {})
    
    company_name = inputs.get("company_name", "")
    industry = inputs.get("industry", "")
    provider = inputs.get("provider", "openai")
    model_name = inputs.get("model_name")
    api_key = inputs.get("api_key")

    system_prompt = LiteLLMRouter.load_prompt("strategist")
    user_data = {
        "company_name": company_name,
        "industry": industry,
        "research_findings": {
            "company_overview": findings.get('company_overview', 'N/A')[:1000],
            "key_challenges": findings.get('challenges', 'N/A')[:1000],
            "technology_gaps": findings.get('technology_gaps', 'N/A')[:1000],
            "graph_insights": findings.get('graph_insights', 'N/A')[:1000]
        }
    }
    messages = LiteLLMRouter.build_messages(system_prompt, user_data)

    try:
        response: AIUseCasesResponse = LiteLLMRouter.completion(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            messages=messages,
            response_format=AIUseCasesResponse
        )
        use_cases_list = [uc.model_dump() for uc in response.use_cases]
        logger.info(f"Strategist Node generated {len(use_cases_list)} structured AI use cases.")
        return {
            "ai_use_cases": use_cases_list,
            "current_step": "strategist_completed"
        }
    except Exception as e:
        logger.error(f"Strategist Node error via LiteLLM: {e}. Generating structured fallback use cases.")
        use_cases_list = [
            AIUseCase(
                title=f"GenAI Automated Customer Intelligence for {company_name}",
                problem_statement="High latency in resolving enterprise customer support queries",
                ai_solution="Deploy LLM-powered Retrieval-Augmented Generation agent across support channels",
                expected_benefits=["40% reduction in support resolution time", "Improved CSAT scores"],
                complexity="Medium",
                roi_timeline="6-9 months",
                required_tech=["LangGraph", "OpenAI GPT-4o", "ChromaDB"]
            ).model_dump()
        ]
        return {
            "ai_use_cases": use_cases_list,
            "errors": state.get("errors", []) + [f"Strategist: {e}"],
            "current_step": "strategist_fallback"
        }
