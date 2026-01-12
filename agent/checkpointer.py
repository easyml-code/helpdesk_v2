from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing import Optional, Iterator
from logs.log import logger


class AsyncWindowedCheckpointer(BaseCheckpointSaver):
    """
    Async windowed checkpointer that keeps only last N messages in memory.
    Prevents context overflow while maintaining conversation continuity.
    """
    
    def __init__(self, window_size: int = 10):
        self.checkpointer = MemorySaver()
        self.window_size = window_size
        logger.info(f"AsyncWindowedCheckpointer initialized with window_size={window_size}")
    
    async def aget_tuple(self, config):
        """Async get checkpoint with windowed messages"""
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            state = checkpoint_tuple.checkpoint.get("channel_values", {})
            
            if "messages" in state and isinstance(state["messages"], list):
                original_count = len(state["messages"])
                state["messages"] = state["messages"][-self.window_size:]
                windowed_count = len(state["messages"])
                
                if windowed_count < original_count:
                    logger.info(
                        f"checkpoint_windowed - original={original_count}, "
                        f"windowed={windowed_count}, dropped={original_count - windowed_count}"
                    )
        
        return checkpoint_tuple
    
    def get_tuple(self, config):
        """Sync get (fallback)"""
        return self.checkpointer.get_tuple(config)
    
    async def alist(self, config, *, filter=None, before=None, limit=None):
        """Async list checkpoints"""
        return self.checkpointer.list(config, filter=filter, before=before, limit=limit)
    
    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator:
        """Sync list checkpoints"""
        return self.checkpointer.list(config, filter=filter, before=before, limit=limit)
    
    async def aput(self, config, checkpoint, metadata, new_versions):
        """Async save checkpoint"""
        return self.checkpointer.put(config, checkpoint, metadata, new_versions)
    
    def put(self, config, checkpoint, metadata, new_versions):
        """Sync save checkpoint"""
        return self.checkpointer.put(config, checkpoint, metadata, new_versions)
    
    async def aput_writes(self, config, writes, task_id):
        """Async save intermediate writes"""
        return self.checkpointer.put_writes(config=config, writes=writes, task_id=task_id)
    
    def put_writes(self, config, writes, task_id):
        """Sync save intermediate writes"""
        return self.checkpointer.put_writes(config=config, writes=writes, task_id=task_id)


class AsyncAdaptiveWindowedCheckpointer(AsyncWindowedCheckpointer):
    """
    Adaptive windowing based on token counts.
    """
    
    def __init__(self, base_window_size: int = 10, max_window_tokens: int = 8000, min_window_size: int = 4):
        super().__init__(window_size=base_window_size)
        self.max_window_tokens = max_window_tokens
        self.min_window_size = min_window_size
        logger.info(
            f"AsyncAdaptiveWindowedCheckpointer initialized - "
            f"base_window={base_window_size}, max_tokens={max_window_tokens}"
        )
    
    def _estimate_tokens(self, message) -> int:
        """Estimate token count"""
        if hasattr(message, 'content'):
            content = message.content
        elif isinstance(message, dict):
            content = message.get('content', '')
        else:
            content = str(message)
        return len(str(content)) // 4
    
    async def aget_tuple(self, config):
        """Async get with adaptive windowing"""
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            state = checkpoint_tuple.checkpoint.get("channel_values", {})
            
            if "messages" in state and isinstance(state["messages"], list):
                messages = state["messages"]
                original_count = len(messages)
                
                windowed_messages = []
                total_tokens = 0
                
                for msg in reversed(messages):
                    msg_tokens = self._estimate_tokens(msg)
                    
                    if len(windowed_messages) >= self.window_size:
                        break
                    
                    if total_tokens + msg_tokens > self.max_window_tokens:
                        if len(windowed_messages) >= self.min_window_size:
                            break
                    
                    windowed_messages.insert(0, msg)
                    total_tokens += msg_tokens
                
                state["messages"] = windowed_messages
                
                logger.info(
                    f"adaptive_checkpoint_windowed - original={original_count}, "
                    f"windowed={len(windowed_messages)}, tokens={total_tokens}"
                )
        
        return checkpoint_tuple