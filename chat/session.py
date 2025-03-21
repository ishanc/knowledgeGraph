from datetime import datetime, timedelta
import logging
from mistral_wrapper import MistralWrapper
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Add logger
logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self, session_id: str, knowledge_graph, ttl_minutes: int = 30):
        self.session_id = session_id
        self.kg = knowledge_graph
        self.conversation_history = []
        self.last_activity = datetime.now()
        self.ttl = timedelta(minutes=ttl_minutes)
        self.mistral = MistralWrapper()
        
    def is_expired(self) -> bool:
        return datetime.now() - self.last_activity > self.ttl
        
    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.last_activity = datetime.now()
        
    def get_relevant_context(self, query: str, max_results: int = 5) -> str:
        # Query knowledge graph for relevant information
        results = []
        
        # Search nodes by similarity
        query_embedding = self.kg.get_embedding(query)
        for node, data in self.kg.nx_graph.nodes(data=True):
            if 'embedding' in data:
                similarity = self.kg.cosine_similarity(
                    query_embedding.reshape(1, -1),
                    data['embedding'].reshape(1, -1)
                )[0][0]
                if similarity > 0.5:  # Threshold for relevance
                    results.append({
                        'content': str(node),
                        'type': data.get('type', 'unknown'),
                        'similarity': float(similarity)
                    })
        
        # Sort by similarity and format context
        results.sort(key=lambda x: x['similarity'], reverse=True)
        context_items = results[:max_results]
        
        if not context_items:
            return ""
        return "\n".join([f"{item['type']}: {item['content']}" for item in context_items])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_response(self, user_message: str) -> str:
        try:
            # Get relevant context from knowledge graph
            context = self.get_relevant_context(user_message)
            
            # Format conversation messages array
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a knowledgeable assistant with access to a knowledge graph.
                    Base your responses on the following context and previous conversation.
                    If you're not confident about something, acknowledge your uncertainty. Do not use any general knowledge, common sense or external information. Only use the provided context and history to generate responses.
                    
                    Relevant Context:
                    {context}"""
                }
            ]
            
            # Add conversation history in chronological order
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Add retry handling
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = self.mistral.generate_with_context(
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    if response:
                        return response
                    retry_count += 1
                    time.sleep(2 ** retry_count)  # Exponential backoff
                except Exception as e:
                    logger.debug(f"Attempt {retry_count + 1} failed: {str(e)}")
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    time.sleep(2 ** retry_count)
                    
            return "I apologize, but I was unable to generate a response after multiple attempts."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
