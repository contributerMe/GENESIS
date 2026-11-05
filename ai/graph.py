"""
Links Researcher, Knowledge, Strategist, Dataset, and Writer Agent Nodes in a resilient graph workflow.
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from ai.state import ResearchState
from ai.agents.researcher import researcher_node
from ai.agents.knowledge import knowledge_node
from ai.agents.strategist import strategist_node
from ai.agents.dataset import dataset_node
from ai.agents.writer import writer_node

logger = logging.getLogger(__name__)

def build_market_research_graph() -> StateGraph:
    """
    Construct and compile the Multi-Agent LangGraph workflow.
    """
    workflow = StateGraph(ResearchState)

    # 1. Add Agent Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("knowledge", knowledge_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("dataset", dataset_node)
    workflow.add_node("writer", writer_node)

    # 2. Add Graph Edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "knowledge")
    workflow.add_edge("knowledge", "strategist")
    workflow.add_edge("strategist", "dataset")
    workflow.add_edge("dataset", "writer")
    workflow.add_edge("writer", END)

    logger.info("Compiled LangGraph Multi-Agent StateGraph successfully.")
    return workflow.compile()

# Lazy singleton — compiled only on first access, not at import time
_compiled_graph = None

def get_research_graph():
    """
    Lazy factory for the compiled research graph.
    Avoids import-time compilation that crashes the entire module chain on any failure.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_market_research_graph()
    return _compiled_graph

# Backwards-compatible alias (will be removed in next major version)
compiled_research_graph = None  # Use get_research_graph() instead
