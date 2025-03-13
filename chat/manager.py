import uuid
from typing import Dict, Optional
from chat.session import ChatSession  # Use absolute import

class ChatManager:
    def __init__(self, knowledge_graph):
        self.sessions: Dict[str, ChatSession] = {}
        self.knowledge_graph = knowledge_graph
        
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id, self.knowledge_graph)
        return session_id
        
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            del self.sessions[session_id]
            return None
        return session
        
    def cleanup_expired_sessions(self):
        expired = [sid for sid, session in self.sessions.items() if session.is_expired()]
        for sid in expired:
            del self.sessions[sid]
