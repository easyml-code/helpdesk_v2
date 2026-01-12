from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from typing import Optional, Any, Iterator
from logs.log import logger


class WindowedCheckpointer(BaseCheckpointSaver):
    """
    Windowed checkpointer that keeps only last N messages in memory.
    Uses MemorySaver as backend (NO REDIS).
    
    This prevents context overflow while maintaining conversation continuity.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of recent messages to keep (default: 10)
        """
        self.checkpointer = MemorySaver()
        self.window_size = window_size
        logger.info(f"WindowedCheckpointer initialized with window_size={window_size}")
    
    def get_tuple(self, config):
        """Get checkpoint with windowed messages"""
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            state = checkpoint_tuple.checkpoint.get("channel_values", {})
            
            # Apply windowing to messages
            if "messages" in state and isinstance(state["messages"], list):
                original_count = len(state["messages"])
                
                # Keep only last window_size messages
                state["messages"] = state["messages"][-self.window_size:]
                
                windowed_count = len(state["messages"])
                
                if windowed_count < original_count:
                    logger.info(
                        f"checkpoint_windowed - "
                        f"original={original_count}, windowed={windowed_count}, "
                        f"dropped={original_count - windowed_count}"
                    )
        
        return checkpoint_tuple
    
    def list(self, config, *, filter: Optional[dict] = None, before: Optional[dict] = None, limit: Optional[int] = None) -> Iterator:
        """List checkpoints (delegates to underlying checkpointer)"""
        return self.checkpointer.list(config, filter=filter, before=before, limit=limit)
    
    def put(self, config, checkpoint, metadata, new_versions):
        """Save checkpoint (delegates to underlying checkpointer)"""
        return self.checkpointer.put(config, checkpoint, metadata, new_versions)
    
    def put_writes(self, config, writes, task_id):
        """
        Save intermediate writes (CRITICAL: this was missing in original)
        """
        return self.checkpointer.put_writes(
            config=config,
            writes=writes,
            task_id=task_id
        )
    
    def get_window_info(self, config) -> dict:
        """Get windowing statistics for a config"""
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            return {
                "total_messages": 0,
                "windowed_messages": 0,
                "dropped_messages": 0
            }
        
        state = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = state.get("messages", [])
        total = len(messages)
        windowed = min(total, self.window_size)
        
        return {
            "total_messages": total,
            "windowed_messages": windowed,
            "dropped_messages": max(0, total - windowed),
            "window_size": self.window_size
        }


class AdaptiveWindowedCheckpointer(WindowedCheckpointer):
    """
    Advanced windowed checkpointer with adaptive window sizing.
    Adjusts window size based on message token counts.
    """
    
    def __init__(
        self,
        base_window_size: int = 10,
        max_window_tokens: int = 8000,
        min_window_size: int = 4
    ):
        """
        Args:
            base_window_size: Default window size
            max_window_tokens: Maximum total tokens in window
            min_window_size: Minimum messages to keep
        """
        super().__init__(window_size=base_window_size)
        self.max_window_tokens = max_window_tokens
        self.min_window_size = min_window_size
        logger.info(
            f"AdaptiveWindowedCheckpointer initialized - "
            f"base_window={base_window_size}, max_tokens={max_window_tokens}"
        )
    
    def _estimate_tokens(self, message) -> int:
        """Estimate token count for a message (rough approximation)"""
        if hasattr(message, 'content'):
            content = message.content
        elif isinstance(message, dict):
            content = message.get('content', '')
        else:
            content = str(message)
        
        # Rough estimate: 1 token ≈ 4 characters
        return len(str(content)) // 4
    
    def get_tuple(self, config):
        """Get checkpoint with adaptive windowing"""
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            state = checkpoint_tuple.checkpoint.get("channel_values", {})
            
            if "messages" in state and isinstance(state["messages"], list):
                messages = state["messages"]
                original_count = len(messages)
                
                # Apply adaptive windowing
                windowed_messages = []
                total_tokens = 0
                
                # Iterate from most recent to oldest
                for msg in reversed(messages):
                    msg_tokens = self._estimate_tokens(msg)
                    
                    # Check constraints
                    if len(windowed_messages) >= self.window_size:
                        break
                    
                    if total_tokens + msg_tokens > self.max_window_tokens:
                        # Only break if we have minimum messages
                        if len(windowed_messages) >= self.min_window_size:
                            break
                    
                    windowed_messages.insert(0, msg)
                    total_tokens += msg_tokens
                
                state["messages"] = windowed_messages
                
                logger.info(
                    f"adaptive_checkpoint_windowed - "
                    f"original={original_count}, windowed={len(windowed_messages)}, "
                    f"tokens={total_tokens}"
                )
        
        return checkpoint_tuple