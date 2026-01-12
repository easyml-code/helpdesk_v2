from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
import json
import uuid
import hashlib
from logs.log import logger, set_trace_id
from metrics.prometheus import cached_messages_gauge


@dataclass
class RolePair:
    """User-Assistant message pair"""
    user: Dict[str, Any]
    assistant: Dict[str, Any]


@dataclass
class ModelConfig:
    """LLM model configuration"""
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int


@dataclass
class SystemContext:
    """System prompt context"""
    system_prompt_hash: str
    system_prompt_version: str


@dataclass
class ToolContext:
    """Tool configuration context"""
    tool_config_hash: str
    tools_enabled: List[str]


@dataclass
class InputContext:
    """Input context for LLM execution"""
    message_versions_used: List[Dict[str, str]]
    summaries_used: List[str]
    tool_outputs_injected: List[str]


@dataclass
class TokenMetrics:
    """Token usage metrics"""
    input: int
    output: int
    internal: int


@dataclass
class ExecutionMetrics:
    """Execution metrics"""
    tokens: TokenMetrics
    latency_ms: float


@dataclass
class LLMExecution:
    """Complete LLM execution record"""
    execution_id: str
    trigger_reason: str
    model_config: ModelConfig
    system_context: SystemContext
    tool_context: ToolContext
    input_context: InputContext
    metrics: ExecutionMetrics
    created_at: str


@dataclass
class MessageVersion:
    """Single version of a message"""
    version_id: str
    parent_version_id: Optional[str]
    created_at: str
    role_pair: RolePair
    llm_execution: LLMExecution


@dataclass
class LogicalMessage:
    """Logical message with multiple versions"""
    logical_message_id: str
    created_at: str
    versions: Dict[str, MessageVersion]


@dataclass
class ChatSession:
    """Complete chat session with versioning"""
    chat_id: str
    created_at: str
    updated_at: str
    active_message_map: Dict[str, str]  # msg_id -> version_id
    messages: Dict[str, LogicalMessage]


class CacheManager:
    """
    Advanced cache manager with message versioning support.
    Supports branching conversations and version switching.
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
    
    def _hash_content(self, content: str) -> str:
        """Hash content for versioning"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def create_chat_session(self, chat_id: str) -> ChatSession:
        """Create new chat session"""
        trace_id = set_trace_id()
        
        session = ChatSession(
            chat_id=chat_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z",
            active_message_map={},
            messages={}
        )
        
        self.cache[chat_id] = session
        self._update_metrics()
        
        logger.info(f"chat_session_created - chat_id={chat_id}, trace_id={trace_id}")
        return session
    
    def add_message_version(
        self,
        chat_id: str,
        logical_msg_id: Optional[str],
        user_content: str,
        assistant_content: str,
        model_config: Dict[str, Any],
        execution_metrics: Dict[str, Any],
        trigger_reason: str = "user_message",
        parent_version_id: Optional[str] = None,
        edited: bool = False
    ) -> tuple[str, str]:
        """
        Add new message version to chat.
        
        Returns:
            (logical_message_id, version_id)
        """
        if chat_id not in self.cache:
            self.create_chat_session(chat_id)
        
        session = self.cache[chat_id]
        
        # Generate IDs
        if logical_msg_id is None:
            logical_msg_id = self._generate_id("msg")
        
        version_id = self._generate_id("v")
        execution_id = self._generate_id("exec")
        
        # Create system context hash
        system_prompt_hash = self._hash_content(model_config.get("system_prompt", ""))
        
        # Create tool context hash
        tools = model_config.get("tools", [])
        tool_config_hash = self._hash_content(json.dumps(sorted(tools)))
        
        # Build execution record
        llm_execution = LLMExecution(
            execution_id=execution_id,
            trigger_reason=trigger_reason,
            model_config=ModelConfig(
                model_name=model_config["model_name"],
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 1.0),
                max_tokens=model_config.get("max_tokens", 8000)
            ),
            system_context=SystemContext(
                system_prompt_hash=system_prompt_hash,
                system_prompt_version="v1"
            ),
            tool_context=ToolContext(
                tool_config_hash=tool_config_hash,
                tools_enabled=tools
            ),
            input_context=InputContext(
                message_versions_used=[
                    {"msg_id": msg_id, "version": ver_id}
                    for msg_id, ver_id in session.active_message_map.items()
                ],
                summaries_used=[],
                tool_outputs_injected=[]
            ),
            metrics=ExecutionMetrics(
                tokens=TokenMetrics(
                    input=execution_metrics.get("input_tokens", 0),
                    output=execution_metrics.get("output_tokens", 0),
                    internal=execution_metrics.get("input_tokens", 0) + 
                             execution_metrics.get("output_tokens", 0)
                ),
                latency_ms=execution_metrics.get("latency_ms", 0)
            ),
            created_at=datetime.utcnow().isoformat() + "Z"
        )
        
        # Create version
        version = MessageVersion(
            version_id=version_id,
            parent_version_id=parent_version_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            role_pair=RolePair(
                user={"content": user_content, "edited": edited},
                assistant={"content": assistant_content}
            ),
            llm_execution=llm_execution
        )
        
        # Add to logical message
        if logical_msg_id not in session.messages:
            session.messages[logical_msg_id] = LogicalMessage(
                logical_message_id=logical_msg_id,
                created_at=datetime.utcnow().isoformat() + "Z",
                versions={}
            )
        
        session.messages[logical_msg_id].versions[version_id] = version
        
        # Update active version
        session.active_message_map[logical_msg_id] = version_id
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        self._update_metrics()
        
        logger.info(
            f"message_version_added - chat_id={chat_id}, "
            f"logical_msg_id={logical_msg_id}, version_id={version_id}, "
            f"trigger={trigger_reason}"
        )
        
        return logical_msg_id, version_id
    
    def switch_message_version(
        self,
        chat_id: str,
        logical_msg_id: str,
        target_version_id: str
    ) -> bool:
        """Switch to a different version of a message"""
        if chat_id not in self.cache:
            logger.error(f"Chat {chat_id} not found in cache")
            return False
        
        session = self.cache[chat_id]
        
        if logical_msg_id not in session.messages:
            logger.error(f"Message {logical_msg_id} not found in chat {chat_id}")
            return False
        
        logical_msg = session.messages[logical_msg_id]
        
        if target_version_id not in logical_msg.versions:
            logger.error(f"Version {target_version_id} not found for message {logical_msg_id}")
            return False
        
        # Update active version
        old_version = session.active_message_map.get(logical_msg_id)
        session.active_message_map[logical_msg_id] = target_version_id
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        logger.info(
            f"message_version_switched - chat_id={chat_id}, "
            f"msg_id={logical_msg_id}, from={old_version}, to={target_version_id}"
        )
        
        return True
    
    def get_active_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all currently active message versions"""
        if chat_id not in self.cache:
            return []
        
        session = self.cache[chat_id]
        messages = []
        
        for msg_id, version_id in session.active_message_map.items():
            if msg_id in session.messages:
                logical_msg = session.messages[msg_id]
                if version_id in logical_msg.versions:
                    version = logical_msg.versions[version_id]
                    messages.append({
                        "logical_message_id": msg_id,
                        "version_id": version_id,
                        "user_content": version.role_pair.user["content"],
                        "assistant_content": version.role_pair.assistant["content"],
                        "created_at": version.created_at
                    })
        
        return messages
    
    def get_message_versions(self, chat_id: str, logical_msg_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific message"""
        if chat_id not in self.cache:
            return []
        
        session = self.cache[chat_id]
        
        if logical_msg_id not in session.messages:
            return []
        
        logical_msg = session.messages[logical_msg_id]
        versions = []
        
        for version_id, version in logical_msg.versions.items():
            versions.append({
                "version_id": version_id,
                "parent_version_id": version.parent_version_id,
                "is_active": session.active_message_map.get(logical_msg_id) == version_id,
                "user_content": version.role_pair.user["content"],
                "assistant_content": version.role_pair.assistant["content"],
                "edited": version.role_pair.user.get("edited", False),
                "created_at": version.created_at
            })
        
        return versions
    
    def serialize_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Serialize chat session to dict (for DB storage)"""
        if chat_id not in self.cache:
            return None
        
        session = self.cache[chat_id]
        
        # Convert dataclasses to dict recursively
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            else:
                return obj
        
        return to_dict(session)
    
    def clear_chat(self, chat_id: str):
        """Remove chat from cache"""
        if chat_id in self.cache:
            del self.cache[chat_id]
            self._update_metrics()
            logger.info(f"chat_cleared_from_cache - chat_id={chat_id}")


# Global instance
cache_manager = CacheManager()