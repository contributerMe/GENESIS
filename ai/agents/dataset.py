"""
Dataset Agent Node
Extracts dataset search keywords from AI Use Cases and queries Kaggle / Open Data APIs.
"""

import logging
from typing import Dict, Any, List
from ai.state import ResearchState
from ai.tools.kaggle_tool import KaggleDatasetTool

logger = logging.getLogger(__name__)

def dataset_node(state: ResearchState) -> ResearchState:
    """
    LangGraph Node: Identifies relevant datasets for AI use case implementation.
    """
    logger.info("--- [Agent Node: Dataset] Extracting keywords & discovering datasets ---")
    inputs = state.get("inputs", {})
    use_cases = state.get("ai_use_cases", [])
    
    company_name = inputs.get("company_name", "")
    industry = inputs.get("industry", "")

    # Extract keywords from use cases
    keywords = [industry, company_name]
    for uc in use_cases[:3]:
        title = uc.get("title", "")
        for word in title.split():
            if len(word) > 4 and word.lower() not in ["automated", "system", "platform", "management"]:
                keywords.append(word)

    try:
        tool = KaggleDatasetTool()
        found_datasets = tool.search_datasets(keywords=list(set(keywords)))

        datasets_list = [ds.model_dump() for ds in found_datasets]
        logger.info(f"Dataset Node discovered {len(datasets_list)} matching datasets.")

        return {
            "datasets": datasets_list,
            "current_step": "dataset_completed"
        }
    except Exception as e:
        logger.error(f"Dataset Node failed: {e}")
        return {
            "datasets": [],
            "errors": state.get("errors", []) + [f"Dataset: {e}"],
            "current_step": "dataset_failed"
        }
