"""
Hybrid Retrieval & Cross-Encoder Re-ranking Engine
Combines BM25 lexical keyword search + Dense Vector embeddings with RRF ranking and Cross-Encoder Re-ranking.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

class HybridReRankRetriever:
    """
    Advanced Retriever featuring BM25 Lexical Search + Dense Vector Search + Cross-Encoder Re-ranking.
    """

    def __init__(self, vectorstore: VectorStore, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self._init_bm25()
        self._init_reranker()

    def _init_bm25(self):
        """Initialize BM25 Lexical Retriever."""
        try:
            from langchain_community.retrievers import BM25Retriever
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 5
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")
            self.bm25_retriever = None

    def _init_reranker(self):
        """Initialize FlashRank Cross-Encoder once (avoids ~2-3s model reload per query)."""
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")
            logger.info("FlashRank Cross-Encoder re-ranker initialized.")
        except Exception as e:
            logger.warning(f"FlashRank unavailable ({e}). Re-ranking will be skipped.")
            self.ranker = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Execute Hybrid Search (BM25 + Dense Vector) with Reciprocal Rank Fusion (RRF).
        """
        logger.info(f"Executing Hybrid Retrieval for query: '{query}'")

        # 1. Dense Vector Retrieval
        dense_docs = self.vectorstore.similarity_search(query, k=top_k * 2)

        # 2. BM25 Lexical Retrieval
        lexical_docs = []
        if self.bm25_retriever:
            try:
                lexical_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.warning(f"BM25 search error: {e}")

        # 3. Reciprocal Rank Fusion (RRF)
        fused_docs = self._reciprocal_rank_fusion(dense_docs, lexical_docs, top_k=top_k * 2)

        # 4. Cross-Encoder Re-Ranking
        reranked_docs = self._rerank_documents(query, fused_docs, top_k=top_k)

        return reranked_docs

    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        lexical_docs: List[Document],
        k: int = 60,
        top_k: int = 10
    ) -> List[Document]:
        """RRF score calculation."""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content[:100]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        for rank, doc in enumerate(lexical_docs):
            doc_id = doc.page_content[:100]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]

    def _rerank_documents(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank candidate documents using pre-loaded Cross-Encoder (FlashRank)."""
        if not self.ranker:
            return docs[:top_k]

        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
            rerank_req = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_req)

            sorted_docs = [docs[item["id"]] for item in results[:top_k]]
            logger.info(f"Re-ranked {len(docs)} candidates down to top {len(sorted_docs)} using FlashRank.")
            return sorted_docs
        except Exception as e:
            logger.warning(f"FlashRank re-ranking error ({e}). Returning RRF ranked results.")
            return docs[:top_k]

