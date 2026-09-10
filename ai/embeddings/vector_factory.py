"""
Pluggable Vector Store Factory (ChromaDB vs Pinecone)
Provides persistent local ChromaDB storage or cloud Pinecone storage.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from ai.settings import get_settings

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """
    Factory for creating and managing persistent vector stores (ChromaDB or Pinecone).
    """

    @staticmethod
    def create_vector_store(
        documents: List[Document],
        embeddings: Embeddings,
        vector_db_choice: str = "chroma",
        collection_name: str = "genesis_market_research"
    ) -> VectorStore:
        """
        Build and persist a vector store from document objects.
        """
        choice = vector_db_choice.lower().strip()

        if choice == "pinecone":
            settings = get_settings()
            api_key = settings.pinecone_api_key
            if not api_key:
                logger.warning("PINECONE_API_KEY not found. Falling back to local persistent ChromaDB.")
                return VectorStoreFactory._build_chroma(documents, embeddings, collection_name)
            
            try:
                from langchain_pinecone import PineconeVectorStore
                logger.info(f"Indexing {len(documents)} documents into Pinecone Cloud index '{collection_name}'...")
                return PineconeVectorStore.from_documents(
                    documents,
                    embeddings,
                    index_name=collection_name
                )
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {e}. Falling back to ChromaDB.")
                return VectorStoreFactory._build_chroma(documents, embeddings, collection_name)
        else:
            return VectorStoreFactory._build_chroma(documents, embeddings, collection_name)

    @staticmethod
    def _build_chroma(
        documents: List[Document],
        embeddings: Embeddings,
        collection_name: str
    ) -> VectorStore:
        """Build local persistent ChromaDB store."""
        from langchain_community.vectorstores import Chroma
        settings = get_settings()
        persist_dir = os.path.abspath(settings.vector_store_path)
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"Indexing {len(documents)} documents into local persistent ChromaDB at '{persist_dir}'...")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
