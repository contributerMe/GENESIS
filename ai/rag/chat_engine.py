import logging
import re
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
            result = self.answer_from_retriever(user_query, retriever, top_k=top_k)
            used_citations = [citation_map.get(c["citation_id"], c) for c in result["citations"]]

            # Append citation symlinks footer
            footer_lines = ["\n\n### 🔗 Sources & Symlinks:"]
            for cit in used_citations:
                c_id = cit.get("citation_id", "")
                c_title = cit.get("title", "")
                c_url = cit.get("url_or_path", "")
                c_type = cit.get("source_type", "web")
                footer_lines.append(f"- **{c_id}** [{c_title}]({c_url}) *({c_type})*")

            final_answer = result["answer"] + "\n" + "\n".join(footer_lines)

            return {
                "answer": final_answer,
                "citations": used_citations,
                "sources_retrieved": result["sources_retrieved"]
            }

        except Exception:
            logger.exception("Dynamic RAG chatbot execution failed")
            return {
                "answer": "The research assistant could not complete that request. Please try again.",
                "citations": [],
                "sources_retrieved": 0
            }

    def answer_from_retriever(
        self,
        question: str,
        retriever: HybridReRankRetriever,
        top_k: int = 4,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Answer against an existing index, with citations from the retrieved documents."""
        history = history or []
        search_query = f"{history[-1]['question']}\n{question}" if history else question
        documents = retriever.retrieve(search_query, top_k=top_k)
        if not documents:
            return {"answer": "I couldn't find relevant evidence in this job's sources.",
                    "citations": [], "sources_retrieved": 0}

        passages = []
        citations = {}
        for doc in documents:
            tag = doc.metadata["citation_id"]
            title = doc.metadata["title"]
            url = doc.metadata.get("url_or_path", "")
            passages.append(f"Source {tag} ({title}):\n{doc.page_content[:1500]}")
            citations[tag] = SourceCitation(
                citation_id=tag,
                title=title,
                source_type=doc.metadata.get("source_type", "web"),
                url_or_path=url,
            ).model_dump()

        messages = LiteLLMRouter.build_messages(LiteLLMRouter.load_prompt("chat_engine"), {
            "user_question": question,
            "conversation_history": history,
            "context_documents": passages,
        })
        answer = LiteLLMRouter.completion(
            provider=self.provider, model_name=self.model_name, api_key=self.api_key,
            messages=messages, max_tokens=1000, timeout=25,
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The model returned an empty chat answer.")
        answer = re.sub(r"\[SRC-\d+\]", lambda tag: tag[0] if tag[0] in citations else "", answer)
        return {"answer": answer.strip(), "citations": list(citations.values()),
                "sources_retrieved": len(documents)}
