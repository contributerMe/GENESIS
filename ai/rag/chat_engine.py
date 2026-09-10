import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from ai.state import ResearchState, SourceCitation
from ai.llm import LiteLLMRouter, EmbeddingFactory
from ai.embeddings.vector_factory import VectorStoreFactory
from ai.rag.hybrid_retriever import HybridReRankRetriever

logger = logging.getLogger(__name__)


class DynamicRAGChatEngine:
    """
    Dynamic RAG Chatbot interface for answering questions against a ResearchState workflow
    or active document pool, providing citations and clickable file/URL symlinks.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_db_choice: str = "chroma"
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.vector_db_choice = vector_db_choice

    def answer_question(
        self,
        user_query: str,
        state: ResearchState,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Execute RAG search over state documents and generate an answer with citations.
        """
        scraped = state.get("scraped_documents", [])
        uploaded = state.get("uploaded_documents", [])
        citations_data = state.get("citations", [])
        all_raw_docs = scraped + uploaded

        if not all_raw_docs:
            return {
                "answer": "No documents available for RAG search. Please run market research or upload documents first.",
                "citations": [],
                "sources_used": []
            }

        # Reconstruct citation lookup map
        citation_map: Dict[str, Dict[str, Any]] = {}
        for cit in citations_data:
            cit_id = cit.get("citation_id")
            if cit_id:
                citation_map[cit_id] = cit

        # Re-build document corpus for retriever
        documents: List[Document] = []
        for idx, item in enumerate(all_raw_docs, 1):
            cit_id = f"[SRC-{idx}]"
            title = item.get("title", f"Source {idx}")
            url_or_path = item.get("url") or item.get("file_path") or ""
            source_type = item.get("source_type", "web")
            content = item.get("content", "")

            if content:
                documents.append(
                    Document(
                        page_content=f"{cit_id} {title}\n{content}",
                        metadata={
                            "citation_id": cit_id,
                            "title": title,
                            "url_or_path": url_or_path,
                            "source_type": source_type
                        }
                    )
                )

        try:
            company_name = state.get("inputs", {}).get("company_name", "enterprise")
            embeddings = EmbeddingFactory.get_embeddings(provider=self.provider, api_key=self.api_key)
            vectorstore = VectorStoreFactory.create_vector_store(
                documents=documents,
                embeddings=embeddings,
                vector_db_choice=self.vector_db_choice,
                collection_name=f"genesis_{company_name.lower().replace(' ', '_')}"
            )

            retriever = HybridReRankRetriever(vectorstore=vectorstore, documents=documents)
            retrieved_docs = retriever.retrieve(user_query, top_k=top_k)

            # Build context string with explicit citations
            context_blocks = []
            used_citations = []
            for doc in retrieved_docs:
                cit_id = doc.metadata.get("citation_id", "")
                title = doc.metadata.get("title", "")
                url_or_path = doc.metadata.get("url_or_path", "")
                context_blocks.append(f"Source {cit_id} ({title}):\n{doc.page_content[:1500]}")
                if cit_id in citation_map:
                    used_citations.append(citation_map[cit_id])
                else:
                    used_citations.append({
                        "citation_id": cit_id,
                        "title": title,
                        "url_or_path": url_or_path,
                        "source_type": doc.metadata.get("source_type", "web")
                    })

            system_prompt = LiteLLMRouter.load_prompt("chat_engine")
            user_data = {
                "user_question": user_query,
                "context_documents": context_blocks
            }
            messages = LiteLLMRouter.build_messages(system_prompt, user_data)

            raw_answer = LiteLLMRouter.completion(
                provider=self.provider,
                model_name=self.model_name,
                api_key=self.api_key,
                messages=messages
            )

            # Append citation symlinks footer
            footer_lines = ["\n\n### 🔗 Sources & Symlinks:"]
            for cit in used_citations:
                c_id = cit.get("citation_id", "")
                c_title = cit.get("title", "")
                c_url = cit.get("url_or_path", "")
                c_type = cit.get("source_type", "web")
                footer_lines.append(f"- **{c_id}** [{c_title}]({c_url}) *({c_type})*")

            final_answer = raw_answer + "\n" + "\n".join(footer_lines)

            return {
                "answer": final_answer,
                "citations": used_citations,
                "sources_retrieved": len(retrieved_docs)
            }

        except Exception:
            logger.exception("Dynamic RAG chatbot execution failed")
            return {
                "answer": "The research assistant could not complete that request. Please try again.",
                "citations": [],
                "sources_retrieved": 0
            }
