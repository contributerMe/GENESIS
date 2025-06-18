"""
RAG System Module for Market Research
Handles document processing, vector storage, and retrieval-augmented generation
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system for processing and querying scraped data"""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize RAG system with OpenAI embeddings
        
        Args:
            openai_api_key: OpenAI API key for embeddings
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None
    
    def build_vectorstore(self, scraped_data: Dict[str, List[Dict[str, str]]]) -> bool:
        """
        Build vector store from scraped data
        
        Args:
            scraped_data: Dictionary containing categorized scraped content
            
        Returns:
            True if vectorstore was built successfully, False otherwise
        """
        try:
            all_texts = []
            metadatas = []
             
            for category, content_list in scraped_data.items():
                for item in content_list:
                    if item.get('content'):
                        chunks = self.text_splitter.split_text(item['content'])
                        for chunk in chunks:
                            all_texts.append(chunk)
                            metadatas.append({
                                'category': category,
                                'title': item.get('title', ''),
                                'url': item.get('url', '')
                            })
            
            if all_texts:
                self.vectorstore = FAISS.from_texts(
                    all_texts, 
                    self.embeddings, 
                    metadatas=metadatas
                )
                logger.info(f"Built vectorstore with {len(all_texts)} text chunks")
                return True
            # else:
            #     logger.warning("No text content found to build vectorstore")
            #     return False
                
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
            return False
    
    def setup_qa_chain(self, llm) -> bool:
        """
        Setup QA chain with the vector store
        
        Args:
            llm: Language model instance
            
        Returns:
            True if QA chain was setup successfully, False otherwise
        """
        try:
            if self.vectorstore:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True
                )
                logger.info("QA chain setup completed")
                return True
            else:
                logger.error("Cannot setup QA chain: vectorstore not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            return False
    
    def query(self, question: str) -> str:
        """
        Query the RAG system
        
        Args:
            question: Question to ask the RAG system
            
        Returns:
            Answer from the RAG system
        """
        if not self.qa_chain:
            return "Knowledge base not initialized."
            
        try:
            result = self.qa_chain({"query": question})
            return result.get('result', 'No answer found.')
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return "Unable to retrieve information from the knowledge base."
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Get relevant documents for a query without generating an answer
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.vectorstore:
            return []
            
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def get_vectorstore_stats(self) -> Dict[str, int]:
        """
        Get statistics about the vectorstore
        
        Returns:
            Dictionary with vectorstore statistics
        """
        if not self.vectorstore:
            return {"total_documents": 0, "total_vectors": 0}
            
        try:
            # Get basic stats (this may vary depending on FAISS version)
            index_size = self.vectorstore.index.ntotal if hasattr(self.vectorstore.index, 'ntotal') else 0
            return {
                "total_vectors": index_size,
                "embedding_dimension": self.vectorstore.index.d if hasattr(self.vectorstore.index, 'd') else 0
            }
        except Exception as e:
            logger.error(f"Failed to get vectorstore stats: {e}")
            return {"total_vectors": 0, "embedding_dimension": 0}