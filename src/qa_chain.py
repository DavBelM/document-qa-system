"""
QA Chain implementation for Retrieval-Augmented Generation.
Combines document retrieval with language model generation for intelligent Q&A.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory

from .document_processor import DocumentProcessor
from .vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentQAChain:
    """
    Complete RAG pipeline for document question-answering.
    Handles document processing, vector storage, and response generation.
    """
    
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_docs_retrieve: int = 5):
        """
        Initialize the QA chain.
        
        Args:
            model_name: OpenAI model to use for generation
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens in response
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_docs_retrieve: Maximum documents to retrieve for context
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_docs_retrieve = max_docs_retrieve
        
        # Initialize components
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            return_messages=True
        )
        
        # Custom prompt template
        self.qa_prompt = PromptTemplate(
            template="""You are an intelligent document assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If the answer isn't in the context, say "I don't have enough information to answer that question"
- Always cite the source documents when possible
- Be concise but comprehensive

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = None
        self.is_initialized = False
        
    def load_documents(self, documents_path: str) -> Dict[str, Any]:
        """
        Load and process documents from a file or directory.
        
        Args:
            documents_path: Path to documents
            
        Returns:
            Dictionary with loading results
        """
        try:
            logger.info(f"Loading documents from: {documents_path}")
            
            # Process documents
            documents = self.document_processor.process_documents(documents_path)
            
            if not documents:
                raise ValueError("No documents were loaded")
                
            # Add to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            # Get statistics
            doc_stats = self.document_processor.get_document_stats(documents)
            vector_stats = self.vector_store.get_collection_stats()
            
            result = {
                "status": "success",
                "documents_loaded": len(documents),
                "document_ids": doc_ids,
                "document_stats": doc_stats,
                "vector_stats": vector_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain with retriever."""
        try:
            retriever = self.vector_store.get_retriever(
                search_type="similarity",
                k=self.max_docs_retrieve
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.qa_prompt,
                    "verbose": False
                },
                return_source_documents=True
            )
            
            self.is_initialized = True
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            raise
    
    def ask(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: Question to ask
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            raise ValueError("QA chain not initialized. Load documents first.")
            
        try:
            logger.info(f"Processing question: '{question[:100]}...'")
            
            # Get response from QA chain
            response = self.qa_chain({"query": question})
            
            # Process source documents
            sources = []
            if include_sources and "source_documents" in response:
                sources = self._process_source_documents(response["source_documents"])
            
            # Add to conversation memory
            self.memory.save_context(
                {"question": question},
                {"answer": response["result"]}
            )
            
            result = {
                "question": question,
                "answer": response["result"],
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name
            }
            
            logger.info("Successfully generated response")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def _process_source_documents(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Process source documents for response.
        
        Args:
            source_docs: List of source documents
            
        Returns:
            List of processed source information
        """
        sources = []
        
        for i, doc in enumerate(source_docs):
            source_info = {
                "source_id": i + 1,
                "filename": doc.metadata.get("filename", "Unknown"),
                "file_type": doc.metadata.get("file_type", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "chunk_size": doc.metadata.get("chunk_size", len(doc.page_content))
            }
            sources.append(source_info)
            
        return sources
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation exchanges
        """
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        "question": messages[i].content,
                        "answer": messages[i + 1].content
                    })
                    
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of matching documents
        """
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            stats = {
                "qa_chain_initialized": self.is_initialized,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_docs_retrieve": self.max_docs_retrieve,
                "conversation_length": len(self.memory.chat_memory.messages) // 2,
                "vector_store_stats": self.vector_store.get_collection_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of responses
        """
        responses = []
        
        for question in questions:
            response = self.ask(question)
            responses.append(response)
            
        return responses
