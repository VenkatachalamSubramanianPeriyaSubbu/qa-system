"""
Graph RAG Retriever using Knowledge Graph
Extracts entities and relationships from messages to build a graph
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class GraphRAGRetriever:
    """Retrieves relevant messages using knowledge graph and semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', top_k: int = 10):
        """
        Initialize Graph RAG retriever
        
        Args:
            model_name: Sentence transformer model for embeddings
            top_k: Number of top relevant messages to retrieve
        """
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.graph = nx.DiGraph()
        self.messages = []
        self.message_embeddings = {}
        self.entity_to_messages = defaultdict(set)
        
        logger.info(f"Initializing Graph RAG retriever with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities from text using simple heuristics
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted entities
        """
        entities = set()
        
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                if i > 0 and words[i-1][0].isupper():
                    entities.add(f"{words[i-1]} {word}")
                else:
                    entities.add(word)
        
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b'
        dates = re.findall(date_pattern, text)
        entities.update(dates)
        
        locations = ['London', 'Paris', 'Tokyo', 'New York', 'Milan', 'Monaco', 'Santorini', 'Switzerland']
        for location in locations:
            if location in text:
                entities.add(location)
        
        return entities
    
    def _extract_relationships(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities
        
        Args:
            text: Input text
            entities: Set of entities in the text
            
        Returns:
            List of (entity1, relationship, entity2) tuples
        """
        relationships = []
        text_lower = text.lower()
        
        entity_list = list(entities)
        
        relation_patterns = [
            ('traveling to', 'TRAVELS_TO'),
            ('going to', 'TRAVELS_TO'),
            ('trip to', 'TRAVELS_TO'),
            ('visiting', 'VISITS'),
            ('at', 'LOCATED_AT'),
            ('in', 'LOCATED_IN'),
            ('prefers', 'PREFERS'),
            ('likes', 'LIKES'),
            ('wants', 'WANTS'),
            ('needs', 'NEEDS'),
            ('planning', 'PLANS'),
        ]
        
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                if entity1.lower() in text_lower and entity2.lower() in text_lower:
                    for pattern, rel_type in relation_patterns:
                        if pattern in text_lower:
                            idx1 = text_lower.index(entity1.lower())
                            idx2 = text_lower.index(entity2.lower())
                            pattern_idx = text_lower.index(pattern)
                            
                            if idx1 < pattern_idx < idx2:
                                relationships.append((entity1, rel_type, entity2))
                                break
        
        return relationships
    
    def index_messages(self, messages: List[Dict[str, Any]]):
        """
        Build knowledge graph from messages
        
        Args:
            messages: List of message dictionaries
        """
        if not messages:
            logger.warning("No messages to index")
            return
        
        logger.info(f"Building knowledge graph from {len(messages)} messages...")
        
        self.messages = messages
        self.graph.clear()
        self.entity_to_messages.clear()
        self.message_embeddings.clear()
        
        for idx, msg in enumerate(messages):
            member_name = msg.get("member_name", "Unknown")
            content = msg.get("content", "")
            text = f"{member_name}: {content}"
            
            self.graph.add_node(f"msg_{idx}", type='message', content=text, data=msg)
            
            if member_name and member_name != "Unknown":
                if not self.graph.has_node(member_name):
                    self.graph.add_node(member_name, type='person')
                self.graph.add_edge(member_name, f"msg_{idx}", relation='SENT')
            
            entities = self._extract_entities(content)
            
            for entity in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, type='entity')
                
                self.graph.add_edge(f"msg_{idx}", entity, relation='MENTIONS')
                self.entity_to_messages[entity].add(idx)
            
            relationships = self._extract_relationships(content, entities)
            for entity1, rel_type, entity2 in relationships:
                if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                    self.graph.add_edge(entity1, entity2, relation=rel_type)
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.message_embeddings[idx] = embedding
        
        logger.info(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _find_relevant_entities(self, query: str) -> Set[str]:
        """
        Find entities relevant to the query
        
        Args:
            query: User query
            
        Returns:
            Set of relevant entity names
        """
        relevant_entities = set()
        
        query_entities = self._extract_entities(query)
        for entity in query_entities:
            if self.graph.has_node(entity):
                relevant_entities.add(entity)
                
                neighbors = list(self.graph.neighbors(entity))
                for neighbor in neighbors:
                    if self.graph.nodes[neighbor].get('type') == 'entity':
                        relevant_entities.add(neighbor)
        
        query_lower = query.lower()
        for node in self.graph.nodes():
            if node.lower() in query_lower:
                relevant_entities.add(node)
        
        return relevant_entities
    
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant messages using graph traversal and semantic search
        
        Args:
            question: User's question
            top_k: Number of results to return
            
        Returns:
            List of relevant message dictionaries
        """
        if not self.messages:
            logger.warning("No messages indexed")
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        relevant_entities = self._find_relevant_entities(question)
        
        candidate_message_ids = set()
        for entity in relevant_entities:
            if entity in self.entity_to_messages:
                candidate_message_ids.update(self.entity_to_messages[entity])
            
            if self.graph.has_node(entity):
                for neighbor in self.graph.neighbors(entity):
                    if neighbor.startswith('msg_'):
                        msg_idx = int(neighbor.split('_')[1])
                        candidate_message_ids.add(msg_idx)
        
        if not candidate_message_ids:
            candidate_message_ids = set(range(len(self.messages)))
        
        question_embedding = self.model.encode(question, convert_to_numpy=True)
        
        scored_messages = []
        for msg_idx in candidate_message_ids:
            if msg_idx in self.message_embeddings:
                msg_embedding = self.message_embeddings[msg_idx]
                similarity = np.dot(question_embedding, msg_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(msg_embedding)
                )
                
                graph_boost = 1.0
                msg_node = f"msg_{msg_idx}"
                if self.graph.has_node(msg_node):
                    connected_entities = [n for n in self.graph.neighbors(msg_node) 
                                        if n in relevant_entities]
                    graph_boost = 1.0 + (0.2 * len(connected_entities))
                
                final_score = similarity * graph_boost
                scored_messages.append((msg_idx, final_score))
        
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        top_messages = scored_messages[:k]
        
        relevant_messages = []
        for msg_idx, score in top_messages:
            msg = self.messages[msg_idx].copy()
            msg['relevance_score'] = float(score)
            relevant_messages.append(msg)
        
        logger.info(f"Retrieved {len(relevant_messages)} relevant messages using graph RAG")
        
        return relevant_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        person_nodes = sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'person')
        entity_nodes = sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'entity')
        message_nodes = sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'message')
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "person_nodes": person_nodes,
            "entity_nodes": entity_nodes,
            "message_nodes": message_nodes,
            "total_messages": len(self.messages),
            "model_name": self.model_name,
            "top_k": self.top_k
        }

