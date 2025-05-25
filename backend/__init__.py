from typing import Dict
from vector_db import VectorDB
import json
import os

class ChatEngine:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.loaded_subjects = set()
    
    def _load_subject_data(self, department: str, semester: str, subject: str) -> Dict:
        """Load JSON data for a subject"""
        path = f"data/departments/{department}/{semester}/{subject}.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subject data not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Index subject if not already indexed
        subject_key = f"{department}_{semester}_{subject}"
        if subject_key not in self.loaded_subjects:
            self.vector_db.index_subject(department, semester, subject, data)
            self.loaded_subjects.add(subject_key)
        
        return data
    
    def _build_context(self, search_results: list) -> str:
        """Build context from search results"""
        context = "Relevant course materials:\n\n"
        for result in search_results:
            context += f"{result['content']}\n\n---\n\n"
        return context
    
    def _generate_response(self, context: str, query: str) -> str:
        """Generate response using the context (simplified - in practice you'd use an LLM)"""
        # In a real implementation, you'd use LangChain or direct LLM API calls here
        # This is a simplified version that just returns the most relevant context
        
        if not context:
            return "I couldn't find relevant information in the course materials. Please try asking about a different topic."
        
        return f"Based on the course materials:\n\n{context.split('---')[0]}\n[This is a simulated response. In the full implementation, an LLM would generate a proper answer.]"
    
    def process(self, department: str, semester: str, subject: str, message: str) -> str:
        """Process a chat message"""
        # 1. Load subject data
        subject_data = self._load_subject_data(department, semester, subject)
        
        # 2. Search for relevant content
        search_results = self.vector_db.search(message, department, semester, subject)
        
        # 3. Build context
        context = self._build_context(search_results)
        
        # 4. Generate response
        response = self._generate_response(context, message)
        
        return response