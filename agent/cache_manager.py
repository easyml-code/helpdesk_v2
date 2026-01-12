from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import uuid
from logs.log import logger, set_trace_id
from metrics.prometheus import cached_messages_gauge


@dataclass
class Message:
    """Single message in a chat"""
    message_id: str
    role: str  # user or assistant
    content: str
    tokens: int
    created_at: str


@dataclass
class ChatSession:
    """Complete chat session"""
    chat_id: str
    user_id: str
    session_id: str
    created_at: str
    updated_at: str
    messages: List[Message]


class CacheManager:
    """
    Simple cache manager for chat messages without versioning.
    Handles in-memory caching of chat sessions and messages.
    """
    
    def __init__(self):
        # In-memory cache: {chat_id: ChatSession}
        self.cache: Dict[str, ChatSession] = {}
        self._update_metrics()
    
    def _update_metrics(self):
        """Update Prometheus metrics"""
        total_messages = sum(
            len(chat.messages) 
            for chat in self.cache.values()
        )
        cached_messages_gauge.set(total_messages)
    
    def _generate_id(self, prefix: str = "msg") -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def create_chat_session(self, chat_id: str, user_id: str, session_id: str) -> ChatSession:
        """Create new chat session in cache"""
        trace_id = set_trace_id()
        
        session = ChatSession(
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z",
            messages=[]
        )
        
        self.cache[chat_id] = session
        self._update_metrics()
        
        logger.info(f"chat_session_created - chat_id={chat_id}, trace_id={trace_id}")
        return session
    
    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        tokens: int
    ) -> str:
        """
        Add message to chat cache.
        
        Returns:
            message_id
        """
        if chat_id not in self.cache:
            logger.warning(f"chat_not_in_cache - chat_id={chat_id}")
            return None
        
        session = self.cache[chat_id]
        message_id = self._generate_id("msg")
        
        message = Message(
            message_id=message_id,
            role=role,
            content=content,
            tokens=tokens,
            created_at=datetime.utcnow().isoformat() + "Z"
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        self._update_metrics()
        
        logger.info(
            f"message_added - chat_id={chat_id}, message_id={message_id}, "
            f"role={role}, tokens={tokens}"
        )
        
        return message_id
    
    def get_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat"""
        if chat_id not in self.cache:
            return []
        
        session = self.cache[chat_id]
        return [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "tokens": msg.tokens,
                "created_at": msg.created_at
            }
            for msg in session.messages
        ]
    
    def load_chat(self, chat_id: str, user_id: str, session_id: str, messages: List[Dict[str, Any]]):
        """Load chat from database into cache"""
        session = ChatSession(
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z",
            messages=[]
        )
        
        for msg in messages:
            message = Message(
                message_id=msg.get("message_id", self._generate_id("msg")),
                role=msg["role"],
                content=msg["content"],
                tokens=msg.get("tokens", 0),
                created_at=msg.get("created_at", datetime.utcnow().isoformat() + "Z")
            )
            session.messages.append(message)
        
        self.cache[chat_id] = session
        self._update_metrics()
        
        logger.info(f"chat_loaded - chat_id={chat_id}, messages={len(messages)}")
    
    def get_session(self, chat_id: str) -> Optional[ChatSession]:
        """Get chat session from cache"""
        return self.cache.get(chat_id)
    
    def clear_chat(self, chat_id: str):
        """Remove chat from cache"""
        if chat_id in self.cache:
            del self.cache[chat_id]
            self._update_metrics()
            logger.info(f"chat_cleared_from_cache - chat_id={chat_id}")


# Global instance
cache_manager = CacheManager()