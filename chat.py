import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Any
import numpy as np
from mistral_wrapper import MistralWrapper
from llm_utils import FORMAT_INSTRUCTIONS, EXAMPLE_RESPONSE, retry_llm_call

logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self, kg_builder):
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.history = []
        self.kg_builder = kg_builder 
        self.mistral = MistralWrapper()
        self.embeddings_cache = {}  # Cache for computed embeddings
        self.format_instructions = FORMAT_INSTRUCTIONS
        self.example_response = EXAMPLE_RESPONSE

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def _prepare_context(self, query: Any) -> np.ndarray:
        """Safely prepare context array for processing."""
        try:
            # Handle None or empty inputs
            if not query:
                return np.array([[]])
            
            # Convert to numpy array if needed
            if not isinstance(query, np.ndarray):
                # Convert lists or other iterables to flat list first
                if isinstance(query, (list, tuple)):
                    query = [str(item) for item in query]  # Convert all items to strings
                else:
                    query = [str(query)]  # Single item as string
                query = np.array(query)

            # Ensure 2D array
            if query.ndim == 1:
                query = query.reshape(-1, 1)
            elif query.ndim == 0:
                query = np.array([[str(query)]])

            return query
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return np.array([[]])

    def generate_response(self, message: str) -> str:
        try:
            # Query the knowledge graph with safety checks
            try:
                query_results = self.kg_builder.query_graph(message)
                # Convert query results to list if needed
                if query_results is None:
                    query_results = []
                elif not isinstance(query_results, (list, tuple)):
                    query_results = [query_results]
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {str(e)}")
                query_results = []

            # Format context
            context_items = []
            for result in query_results:
                if isinstance(result, dict):
                    ctx = f"{result.get('subject', '')} {result.get('relation', '')} {result.get('object', '')}"
                    context_items.append(ctx.strip())
                elif isinstance(result, str):
                    context_items.append(result)

            kg_context = "\n".join(f"- {item}" for item in context_items) if context_items else "No direct context found."

            # Prepare conversation history
            recent_history = self.history[-5:] if self.history else []
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant with access to a knowledge graph.
Use the provided context to answer questions accurately.

{self.format_instructions}

Here's an example of a valid response:
{json.dumps(self.example_response, indent=2)}"""
                }
            ]
            messages.extend([{"role": m["role"], "content": m["content"]} for m in recent_history])
            
            # Add current context and query
            prompt = f"""Context from knowledge graph:
{kg_context}

User question: {message}

Remember to format your response as a valid JSON object following the structure above."""

            messages.append({"role": "user", "content": prompt})

            try:
                response = self.mistral.generate_with_context(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                
                if not response:
                    return "I apologize, but I couldn't generate a response. Please try again."
                
                return response.strip()
                
            except ConnectionError as e:
                logger.error(f"Connection error with Mistral API: {str(e)}")
                return "I'm currently having trouble connecting to my reasoning engine. Please try again in a moment."
            except Exception as e:
                logger.error(f"Mistral generation error: {str(e)}")
                return "I apologize, but I encountered an error generating a response."

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"

    def _generate_with_context(self, message: str, context: np.ndarray) -> str:
        # Use context to generate response
        # Add your response generation logic here
        return f"Generated response based on context of shape {context.shape}"

class ChatManager:
    def __init__(self, kg_builder):
        self.sessions = {}
        self.kg_builder = kg_builder
        self.session_timeout = timedelta(hours=1)

    def create_session(self) -> str:
        session = ChatSession(self.kg_builder)
        self.sessions[session.session_id] = session
        return session.session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        # Check if session has expired
        if datetime.now() - session.created_at > self.session_timeout:
            del self.sessions[session_id]
            return None
            
        return session

    def cleanup_expired_sessions(self):
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.created_at > self.session_timeout
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]
