"""
Vector store management using Chroma for document embeddings and retrieval.
Handles embedding generation, storage, and similarity search operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using Chroma database.
    Handles document embeddings and similarity search operations.
    """
    
    def __init__(self, 
                 persist_directory: str = "./vectorstore",
                 collection_name: str = "document_qa",
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection in the database
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
        
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create a new one."""
        try:
            # Try to load existing vectorstore
            if self._vectorstore_exists():
                logger.info(f"Loading existing vectorstore from {self.persist_directory}")
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            else:
                logger.info("Creating new vectorstore")
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise
    
    def _vectorstore_exists(self) -> bool:
        """Check if vectorstore already exists."""
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        return chroma_db_path.exists()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
            
        logger.info(f"Adding {len(documents)} documents to vectorstore")
        
        try:
            # Generate unique IDs for documents
            doc_ids = [f"doc_{i}_{doc.metadata.get('chunk_id', i)}" 
                      for i, doc in enumerate(documents)]
            
            # Add documents to vectorstore
            self.vectorstore.add_documents(documents, ids=doc_ids)
            
            # Persist the changes
            self.vectorstore.persist()
            
            logger.info(f"Successfully added {len(documents)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
            
        try:
            logger.info(f"Performing similarity search for query: '{query[:50]}...'")
            
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
                
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
            
        try:
            logger.info(f"Performing similarity search with scores for: '{query[:50]}...'")
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            raise
    
    def get_retriever(self, 
                     search_type: str = "similarity",
                     k: int = 5,
                     score_threshold: Optional[float] = None):
        """
        Get a retriever object for use with LangChain.
        
        Args:
            search_type: Type of search ("similarity" or "mmr")
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score threshold
            
        Returns:
            LangChain retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
            
        search_kwargs = {"k": k}
        
        if score_threshold:
            search_kwargs["score_threshold"] = score_threshold
            
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                logger.info(f"Deleted collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.vectorstore:
            return {"error": "Vectorstore not initialized"}
            
        try:
            # Get the underlying collection
            collection = self.vectorstore._collection
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embedding_model
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def update_documents(self, documents: List[Document], doc_ids: List[str]):
        """
        Update existing documents in the vector store.
        
        Args:
            documents: Updated documents
            doc_ids: IDs of documents to update
        """
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of IDs")
            
        try:
            logger.info(f"Updating {len(documents)} documents")
            
            # Chroma doesn't have direct update, so we delete and re-add
            self.vectorstore._collection.delete(ids=doc_ids)
            self.add_documents(documents)
            
            logger.info("Successfully updated documents")
            
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            raise
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: int = 10) -> List[Document]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        try:
            logger.info(f"Searching by metadata: {metadata_filter}")
            
            # Use Chroma's filter capability
            results = self.vectorstore.similarity_search(
                query="", 
                k=limit, 
                filter=metadata_filter
            )
            
            logger.info(f"Found {len(results)} documents matching metadata filter")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            raise
