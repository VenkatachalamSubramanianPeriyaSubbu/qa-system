"""
QA Processor module using OpenAI API with RAG
Uses semantic search to find relevant messages before generating answers
"""

from openai import OpenAI
import logging
from typing import List, Dict, Any, Optional

from config import get_settings
from prompts import QA_PROMPT_TEMPLATE
from graph_rag_retriever import GraphRAGRetriever

logger = logging.getLogger(__name__)


class QAProcessor:
    """Processes natural language questions using OpenAI with RAG"""
    
    def __init__(self, use_rag: bool = True, top_k: int = 10, model: str = "gpt-4o-mini"):
        """
        Initialize QA Processor
        
        Args:
            use_rag: Whether to use RAG for retrieval (default: True)
            top_k: Number of relevant messages to retrieve with RAG
            model: OpenAI model to use (default: gpt-4o-mini for speed and cost)
        """
        self.settings = get_settings()
        self.use_rag = use_rag
        self.top_k = top_k
        self.model_name = model
        self.client = self._initialize_client()
        self.rag_retriever = GraphRAGRetriever(top_k=top_k) if use_rag else None
        
    def _initialize_client(self):
        """Initialize the OpenAI client"""
        try:
            client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
            return client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def index_messages(self, member_data: List[Dict[str, Any]]):
        """
        Index messages for RAG retrieval
        
        Args:
            member_data: List of all member messages
        """
        if self.use_rag and self.rag_retriever:
            logger.info("Indexing messages for RAG...")
            self.rag_retriever.index_messages(member_data)
    
    async def process_question(self, question: str, member_data: List[Dict[str, Any]]) -> str:
        """
        Process a natural language question and generate an answer using RAG
        
        Args:
            question: The natural language question
            member_data: List of member messages (full dataset)
        
        Returns:
            Answer string
        """
        try:
            if self.use_rag and self.rag_retriever:
                logger.info(f"Using RAG to retrieve top {self.top_k} relevant messages")
                relevant_messages = self.rag_retriever.retrieve(question, top_k=self.top_k)
                context = self._prepare_context(relevant_messages)
                logger.info(f"Retrieved {len(relevant_messages)} relevant messages")
            else:
                logger.info("RAG disabled, using all messages")
                context = self._prepare_context(member_data)
            
            prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided member messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I encountered an error while processing your question. Please try rephrasing it."
    
    def _prepare_context(self, messages: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from messages for Gemini
        
        Args:
            messages: List of relevant messages
        
        Returns:
            Context string
        """
        if not messages:
            return "No relevant information found in the member messages."
        
        context_parts = []
        for msg in messages:
            member_name = msg.get("member_name", "Unknown")
            content = msg.get("content", "")
            
            context_part = f"Member: {member_name}\nMessage: {content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
