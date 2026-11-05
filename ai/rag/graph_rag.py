"""
Graph-Enhanced RAG (GraphRAG) Module
Parses scraped documents into interconnected Knowledge Graph Entities & Relationships
using litellm.completion() with Pydantic response formatting.
"""

import logging
from typing import List, Dict, Any, Optional
import networkx as nx
from ai.state import KnowledgeEntity, KnowledgeRelation, GraphRAGData
from ai.llm import LiteLLMRouter

logger = logging.getLogger(__name__)

class GraphRAGEngine:
    """
    Engine for extracting Knowledge Graph entities & edges using litellm.completion().
    """

    def __init__(self, provider: str = "openai", model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.graph = nx.DiGraph()

    def build_knowledge_graph(self, documents: List[Dict[str, Any]]) -> GraphRAGData:
        """
        Extract Entities and Relationships from document texts using litellm.completion().
        """
        logger.info(f"Building Knowledge Graph from {len(documents)} document sources via LiteLLM...")
        
        all_entities: List[KnowledgeEntity] = []
        all_relations: List[KnowledgeRelation] = []

        combined_text = "\n\n".join([doc.get("content", "")[:1500] for doc in documents[:5]])
        if not combined_text.strip():
            return GraphRAGData(entities=[], relations=[])

        system_prompt = LiteLLMRouter.load_prompt("graph_rag")
        user_data = {
            "source_text": combined_text[:4000]
        }
        messages = LiteLLMRouter.build_messages(system_prompt, user_data)

        try:
            extracted_graph: GraphRAGData = LiteLLMRouter.completion(
                provider=self.provider,
                model_name=self.model_name,
                api_key=self.api_key,
                messages=messages,
                response_format=GraphRAGData
            )

            for entity in extracted_graph.entities:
                self.graph.add_node(entity.name, type=entity.type, description=entity.description)
                all_entities.append(entity)

            for rel in extracted_graph.relations:
                self.graph.add_edge(rel.source_entity, rel.target_entity, relation=rel.relationship_type, context=rel.context)
                all_relations.append(rel)

            logger.info(f"Extracted Graph: {len(all_entities)} entities, {len(all_relations)} relations.")
            return extracted_graph

        except Exception as e:
            logger.error(f"GraphRAG extraction error via LiteLLM: {e}")
            return GraphRAGData(entities=[], relations=[])

    def query_graph_insights(self, company_name: str, industry: str) -> str:
        """
        Traverse Knowledge Graph nodes & edges to synthesize multi-hop insights.
        """
        if self.graph.number_of_nodes() == 0:
            return "Knowledge graph is empty."

        summary_lines = []
        summary_lines.append(f"Knowledge Graph Nodes ({self.graph.number_of_nodes()}):")
        for node, data in list(self.graph.nodes(data=True))[:10]:
            summary_lines.append(f" - [{data.get('type', 'Entity')}] {node}: {data.get('description', '')}")

        summary_lines.append(f"\nKnowledge Graph Edges ({self.graph.number_of_edges()}):")
        for u, v, data in list(self.graph.edges(data=True))[:10]:
            summary_lines.append(f" - {u} --[{data.get('relation', 'RELATED')}]--> {v}")

        return "\n".join(summary_lines)
