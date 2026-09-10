import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from ai.state import ResearchState, SourceCitation
from ai.llm import EmbeddingFactory
from ai.embeddings.vector_factory import VectorStoreFactory
from ai.rag.graph_rag import GraphRAGEngine
from ai.rag.hybrid_retriever import HybridReRankRetriever
from ai.settings import get_settings

logger = logging.getLogger(__name__)

def knowledge_node(state: ResearchState) -> ResearchState:
    """
    LangGraph Node: Builds Vector Store & Knowledge Graph via LiteLLM over web & user uploaded docs.
    """
    logger.info("--- [Agent Node: Knowledge] Building unified vector index & GraphRAG ---")
    inputs = state.get("inputs", {})
    scraped = state.get("scraped_documents", [])
    uploaded = state.get("uploaded_documents", [])

    company_name = inputs.get("company_name", "")
    industry = inputs.get("industry", "")
    provider = inputs.get("provider", "openai")
    model_name = inputs.get("model_name")
    api_key = inputs.get("api_key")
    vector_db_choice = inputs.get("vector_db_choice", "chroma")
    retrieval_mode = inputs.get("retrieval_mode", "graph_rag")

    settings = get_settings()
    all_raw_docs = (scraped + uploaded)[:settings.max_sources_per_run]
    if not all_raw_docs:
        logger.warning("No scraped or uploaded documents found in state. Skipping knowledge indexing.")
        return {"current_step": "knowledge_skipped"}

    # Assign citation IDs and create SourceCitation objects
    citations_list: List[Dict[str, Any]] = []
    documents: List[Document] = []

    for idx, item in enumerate(all_raw_docs, 1):
        cit_id = f"[SRC-{idx}]"
        title = item.get("title", f"Source {idx}")
        url_or_path = item.get("url") or item.get("file_path") or ""
        source_type = item.get("source_type", "web")
        content = item.get("content", "")[:settings.max_document_chars]

        citation_obj = SourceCitation(
            citation_id=cit_id,
            title=title,
            source_type=source_type,
            url_or_path=url_or_path,
            snippet=content[:200]
        )
        citations_list.append(citation_obj.model_dump())

        if content:
            documents.append(
                Document(
                    page_content=f"{cit_id} {title}\n{content}",
                    metadata={
                        "citation_id": cit_id,
                        "title": title,
                        "url_or_path": url_or_path,
                        "source_type": source_type,
                        "category": item.get("category", "")
                    }
                )
            )

    research_findings: Dict[str, str] = {}
    graph_rag_data = None
    node_errors: list = []

    if retrieval_mode == "graph_rag":
        try:
            logger.info("Executing GraphRAG Knowledge Graph Extraction via LiteLLM...")
            graph_engine = GraphRAGEngine(provider=provider, model_name=model_name, api_key=api_key)
            graph_rag_data = graph_engine.build_knowledge_graph(all_raw_docs)
            research_findings["graph_insights"] = graph_engine.query_graph_insights(company_name, industry)
        except Exception:
            logger.exception("GraphRAG extraction failed; continuing with vector retrieval.")
            node_errors.append("Knowledge (GraphRAG): graph extraction failed")

    try:
        embeddings = EmbeddingFactory.get_embeddings(provider=provider, api_key=api_key)
        vectorstore = VectorStoreFactory.create_vector_store(
            documents=documents,
            embeddings=embeddings,
            vector_db_choice=vector_db_choice,
            collection_name=f"genesis_{company_name.lower().replace(' ', '_')}"
        )

        hybrid_retriever = HybridReRankRetriever(vectorstore=vectorstore, documents=documents)
        
        queries = {
            "company_overview": f"What is {company_name}? History, background, and business model.",
            "financial_performance": f"What is {company_name}'s financial performance and revenue?",
            "competitors": f"Who are {company_name}'s main competitors in {industry}?",
            "challenges": f"What pain points, risks, and challenges faces {company_name}?",
            "technology_gaps": f"What digital transformation and technology gaps exist for {company_name}?"
        }

        for q_key, q_text in queries.items():
            top_docs = hybrid_retriever.retrieve(q_text, top_k=3)
            # Include citation IDs in findings for downstream report referencing
            research_findings[q_key] = "\n".join([f"{d.metadata.get('citation_id', '')} {d.page_content}" for d in top_docs])

    except Exception as e:
        logger.error(f"Vector store indexing failed: {e}. Populating direct summary fallback.")
        research_findings["company_overview"] = all_raw_docs[0].get("content", "")[:2000] if all_raw_docs else ""
        node_errors.append(f"Knowledge (VectorStore): {e}")

    logger.info(f"Knowledge Node completed. Indexed {len(documents)} documents across {len(citations_list)} citations.")
    result = {
        "graph_rag_data": graph_rag_data.model_dump() if graph_rag_data else None,
        "research_findings": research_findings,
        "citations": citations_list,
        "current_step": "knowledge_completed"
    }
    if node_errors:
        result["errors"] = state.get("errors", []) + node_errors
    return result
