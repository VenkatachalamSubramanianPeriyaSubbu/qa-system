"""
RAG Retriever module for semantic search over member messages
Uses sentence-transformers for embeddings and FAISS for efficient vector search
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant messages using semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', top_k: int = 10):
        """
        Initialize RAG retriever
        
        Args:
            model_name: Sentence transformer model name
            top_k: Number of top relevant messages to retrieve
        """
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.index = None
        self.messages = []
        self.embeddings = None
        
        logger.info(f"Initializing RAG retriever with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def index_messages(self, messages: List[Dict[str, Any]]):
        """
        Create embeddings and index for all messages
        
        Args:
            messages: List of message dictionaries
        """
        if not messages:
            logger.warning("No messages to index")
            return
        
        logger.info(f"Indexing {len(messages)} messages...")
        
        self.messages = messages
        
        texts = []
        for msg in messages:
            member_name = msg.get("member_name", "Unknown")
            content = msg.get("content", "")
            text = f"{member_name}: {content}"
            texts.append(text)
        
        logger.info("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        logger.info(f"Successfully indexed {len(messages)} messages")
    
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant messages for a question
        
        Args:
            question: The user's question
            top_k: Number of results to return (uses self.top_k if None)
        
        Returns:
            List of relevant message dictionaries
        """
        if self.index is None or not self.messages:
            logger.warning("Index not initialized. Call index_messages() first.")
            return []
        
        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.messages))
        
        question_embedding = self.model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(question_embedding, k)
        
        relevant_messages = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.messages):
                msg = self.messages[idx].copy()
                msg['relevance_score'] = float(distance)
                relevant_messages.append(msg)
        
        logger.info(f"Retrieved {len(relevant_messages)} relevant messages for question")
        
        return relevant_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data"""
        return {
            "total_messages": len(self.messages),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_name": self.model_name,
            "top_k": self.top_k,
            "index_initialized": self.index is not None
        }

