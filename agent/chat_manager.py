from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from database.client import run_query
from database.utils import pg_escape
from logs.log import logger
from config import settings
import uuid
import asyncio
from collections import defaultdict


class ChatSessionManager:
    """
    Manages chat sessions with proper token tracking, metrics caching, and async DB persistence.
    """
    
    def __init__(
        self,
        max_context_multiplier: int = 100,
        llm_context_limit: int = 8000,
        session_timeout_minutes: int = 55,
        auto_save_interval_minutes: int = 5
    ):
        self.max_tokens_per_chat = max_context_multiplier * llm_context_limit
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.auto_save_interval = timedelta(minutes=auto_save_interval_minutes)
        
        # Active chats cache
        self.active_chats: Dict[str, Dict[str, Any]] = {}
        
        # Track unsaved messages per chat
        self.unsaved_messages: Dict[str, List[Dict]] = defaultdict(list)
        
        # Track if metrics have been saved for a session
        self.metrics_saved: Dict[str, bool] = {}
        
        logger.info(
            f"ChatSessionManager initialized - max_tokens={self.max_tokens_per_chat}, "
            f"auto_save_interval={auto_save_interval_minutes}min"
        )
    
    def generate_chat_id(self) -> str:
        return f"chat_{uuid.uuid4().hex[:16]}"
    
    def generate_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:16]}"
    
    async def create_new_chat(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create new chat in database"""
        
        chat_id = self.generate_chat_id()
        session_id = self.generate_session_id()
        
        insert_query = f"""
        INSERT INTO chats (chat_id, user_id, topic, total_tokens, is_active, created_at, updated_at)
        VALUES (
            '{chat_id}',
            '{pg_escape(user_id)}',
            {f"'{pg_escape(topic)}'" if topic else 'NULL'},
            0,
            true,
            NOW(),
            NOW()
        )
        RETURNING chat_id;
        """
        
        try:
            await run_query(insert_query, access_token, refresh_token)
            logger.info(f"chat_created_db - chat_id={chat_id}")
            
            self.active_chats[chat_id] = {
                'messages': [],
                'cumulative_tokens': {'total': 0, 'input': 0, 'output': 0, 'by_turn': []},
                'session_id': session_id,
                'session_start_time': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'last_save': datetime.utcnow(),
                'user_id': user_id,
                'is_new': True,
                'message_count': 0,
                'turn_count': 0,
                'tool_execution_count': 0,
                'error_count': 0,
                'total_latency_ms': 0,
                'latency_records': []
            }
            
            # Mark metrics as not saved
            self.metrics_saved[chat_id] = False
            
            return {
                'chat_id': chat_id,
                'session_id': session_id,
                'is_new': True,
                'total_tokens': 0
            }
            
        except Exception as e:
            logger.error(f"chat_create_error - error={e}", exc_info=True)
            raise
    
    async def get_or_load_chat(
        self,
        chat_id: str,
        user_id: str,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Load chat from DB if not in cache"""
        
        if chat_id in self.active_chats:
            logger.info(f"chat_cache_hit - chat_id={chat_id}")
            chat = self.active_chats[chat_id]
            return {
                'chat_id': chat_id,
                'session_id': chat['session_id'],
                'is_new': False,
                'total_tokens': chat['cumulative_tokens']['total'],
                'messages': chat['messages']
            }
        
        logger.info(f"chat_cache_miss_loading - chat_id={chat_id}")
        
        # Load chat metadata
        chat_query = f"""
        SELECT chat_id, topic, total_tokens, created_at
        FROM chats
        WHERE chat_id = '{pg_escape(chat_id)}'
        AND user_id = '{pg_escape(user_id)}';
        """
        
        chat_result = await run_query(chat_query, access_token, refresh_token)
        
        if not chat_result:
            raise ValueError(f"Chat {chat_id} not found")
        
        # Load messages
        messages_query = f"""
        SELECT session_id, role, content, tokens, created_at
        FROM messages
        WHERE chat_id = '{pg_escape(chat_id)}'
        ORDER BY created_at ASC;
        """
        
        messages_result = await run_query(messages_query, access_token, refresh_token)
        
        # Calculate tokens
        cumulative_input = sum(msg['tokens'] for msg in messages_result if msg['role'] == 'user')
        cumulative_output = sum(msg['tokens'] for msg in messages_result if msg['role'] == 'assistant')
        cumulative_total = cumulative_input + cumulative_output
        
        turn_history = [
            {'role': msg['role'], 'tokens': msg['tokens'], 'timestamp': msg['created_at']}
            for msg in messages_result
        ]
        
        # Cache it
        session_id = self.generate_session_id()
        self.active_chats[chat_id] = {
            'messages': [
                {'role': msg['role'], 'content': msg['content'], 'tokens': msg['tokens'], 'session_id': msg['session_id']}
                for msg in messages_result
            ],
            'cumulative_tokens': {
                'total': cumulative_total,
                'input': cumulative_input,
                'output': cumulative_output,
                'by_turn': turn_history
            },
            'session_id': session_id,
            'session_start_time': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'last_save': datetime.utcnow(),
            'user_id': user_id,
            'is_new': False,
            'message_count': len(messages_result),
            'turn_count': len([m for m in messages_result if m['role'] == 'user']),
            'tool_execution_count': 0,
            'error_count': 0,
            'total_latency_ms': 0,
            'latency_records': []
        }
        
        # Mark metrics as not saved for this new session
        self.metrics_saved[chat_id] = False
        
        logger.info(
            f"chat_loaded - chat_id={chat_id}, messages={len(messages_result)}, "
            f"total_tokens={cumulative_total}"
        )
        
        return {
            'chat_id': chat_id,
            'session_id': session_id,
            'is_new': False,
            'total_tokens': cumulative_total,
            'messages': self.active_chats[chat_id]['messages']
        }
    
    def add_message_to_cache(self, chat_id: str, role: str, content: str, tokens: int):
        """Add message to cache and track for DB save"""
        
        if chat_id not in self.active_chats:
            logger.warning(f"chat_not_in_cache - chat_id={chat_id}")
            return
        
        chat_data = self.active_chats[chat_id]
        
        message = {
            'role': role,
            'content': content,
            'tokens': tokens,
            'session_id': chat_data['session_id'],
            'created_at': datetime.utcnow()
        }
        
        chat_data['messages'].append(message)
        self.unsaved_messages[chat_id].append(message)
        
        # Update tokens
        chat_data['cumulative_tokens']['total'] += tokens
        if role == 'user':
            chat_data['cumulative_tokens']['input'] += tokens
        elif role == 'assistant':
            chat_data['cumulative_tokens']['output'] += tokens
        
        chat_data['cumulative_tokens']['by_turn'].append({
            'role': role,
            'tokens': tokens,
            'cumulative_total': chat_data['cumulative_tokens']['total'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update message count
        chat_data['message_count'] += 1
        if role == 'user':
            chat_data['turn_count'] += 1
        
        chat_data['last_activity'] = datetime.utcnow()
        
        logger.info(
            f"message_cached - chat_id={chat_id}, role={role}, tokens={tokens}, "
            f"cumulative_total={chat_data['cumulative_tokens']['total']}, "
            f"unsaved_count={len(self.unsaved_messages[chat_id])}"
        )
    
    def track_llm_call(self, chat_id: str, latency_ms: float, success: bool = True):
        """Track LLM call metrics for a chat"""
        if chat_id not in self.active_chats:
            return
        
        chat_data = self.active_chats[chat_id]
        chat_data['total_latency_ms'] += latency_ms
        chat_data['latency_records'].append(latency_ms)
        
        if not success:
            chat_data['error_count'] += 1
        
        logger.debug(f"llm_call_tracked - chat_id={chat_id}, latency_ms={latency_ms}")
    
    def track_tool_execution(self, chat_id: str):
        """Track tool execution for a chat"""
        if chat_id not in self.active_chats:
            return
        
        chat_data = self.active_chats[chat_id]
        chat_data['tool_execution_count'] += 1
        
        logger.debug(f"tool_execution_tracked - chat_id={chat_id}")
    
    async def check_token_limit(self, chat_id: str) -> bool:
        """Check token limit"""
        if chat_id not in self.active_chats:
            return True
        
        cumulative_total = self.active_chats[chat_id]['cumulative_tokens']['total']
        
        if cumulative_total >= self.max_tokens_per_chat:
            logger.warning(
                f"token_limit_exceeded - chat_id={chat_id}, "
                f"cumulative={cumulative_total}, limit={self.max_tokens_per_chat}"
            )
            return False
        
        return True
    
    def get_token_stats(self, chat_id: str) -> Dict[str, Any]:
        """Get token statistics"""
        if chat_id not in self.active_chats:
            return {}
        return self.active_chats[chat_id]['cumulative_tokens']
    
    async def save_chat_to_db(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str,
        force: bool = False
    ) -> int:
        """Async batch save unsaved messages to database"""
        
        if chat_id not in self.active_chats:
            return 0
        
        unsaved = self.unsaved_messages.get(chat_id, [])
        
        if not unsaved:
            return 0
        
        # Check if auto-save is needed
        chat_data = self.active_chats[chat_id]
        time_since_save = datetime.utcnow() - chat_data['last_save']
        
        if not force and time_since_save < self.auto_save_interval:
            logger.debug(f"skipping_auto_save - chat_id={chat_id}, time_since_save={time_since_save}")
            return 0
        
        try:
            # Batch insert messages
            values = []
            for msg in unsaved:
                values.append(
                    f"('{chat_id}', '{msg['session_id']}', '{pg_escape(msg['role'])}', "
                    f"'{pg_escape(msg['content'])}', {msg['tokens']}, '{msg['created_at'].isoformat()}')"
                )
            
            insert_query = f"""
            INSERT INTO messages (chat_id, session_id, role, content, tokens, created_at)
            VALUES {', '.join(values)};
            """
            
            await run_query(insert_query, access_token, refresh_token)
            
            # Update chat total_tokens
            cumulative_total = chat_data['cumulative_tokens']['total']
            
            update_query = f"""
            UPDATE chats
            SET total_tokens = {cumulative_total}, updated_at = NOW()
            WHERE chat_id = '{chat_id}';
            """
            
            await run_query(update_query, access_token, refresh_token)
            
            # Clear unsaved messages
            saved_count = len(unsaved)
            self.unsaved_messages[chat_id] = []
            chat_data['last_save'] = datetime.utcnow()
            
            logger.info(
                f"chat_saved - chat_id={chat_id}, messages_saved={saved_count}, "
                f"cumulative_tokens={cumulative_total}"
            )
            
            return saved_count
            
        except Exception as e:
            logger.error(f"chat_save_error - chat_id={chat_id}, error={e}", exc_info=True)
            raise
    
    async def save_session_metrics(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str
    ):
        """Save session metrics to database (only once per session)"""
        if chat_id not in self.active_chats:
            logger.warning(f"save_metrics_chat_not_found - chat_id={chat_id}")
            return
        
        # FIX: Check if metrics already saved for this session
        if self.metrics_saved.get(chat_id, False):
            logger.info(f"metrics_already_saved - chat_id={chat_id}")
            return
        
        chat_data = self.active_chats[chat_id]
        
        # Calculate session metrics
        session_start = chat_data['session_start_time']
        session_end = datetime.utcnow()
        session_duration_seconds = (session_end - session_start).total_seconds()
        
        tokens = chat_data['cumulative_tokens']
        message_count = chat_data['message_count']
        turn_count = chat_data['turn_count']
        tool_count = chat_data['tool_execution_count']
        error_count = chat_data['error_count']
        total_latency = chat_data['total_latency_ms']
        
        # Calculate latency stats
        latency_records = chat_data['latency_records']
        avg_latency = total_latency / len(latency_records) if latency_records else 0
        min_latency = min(latency_records) if latency_records else 0
        max_latency = max(latency_records) if latency_records else 0

        # Calculate cost
        estimated_cost_usd = self._calculate_cost(
            tokens['input'], 
            tokens['output']
        )
        cost_per_message = estimated_cost_usd / message_count if message_count > 0 else 0
        
        # FIX: Use INSERT ... ON CONFLICT to prevent duplicate key errors
        metrics_query = f"""
        INSERT INTO chat_metrics (
            chat_id,
            session_id,
            total_input_tokens,
            total_output_tokens,
            total_tokens,
            total_latency_ms,
            avg_latency_ms,
            min_latency_ms,
            max_latency_ms,
            message_count,
            tool_execution_count,
            error_count,
            estimated_cost_usd,
            cost_per_message,
            session_start,
            session_end,
            created_at,
            updated_at
        ) VALUES (
            '{chat_id}',
            '{chat_data['session_id']}',
            {tokens['input']},
            {tokens['output']},
            {tokens['total']},
            {total_latency},
            {avg_latency},
            {min_latency},
            {max_latency},
            {message_count},
            {tool_count},
            {error_count},
            {estimated_cost_usd},
            {cost_per_message},
            '{session_start.isoformat()}',
            '{session_end.isoformat()}',
            NOW(),
            NOW()
        )
        ON CONFLICT (chat_id, session_id) DO UPDATE SET
            total_input_tokens = EXCLUDED.total_input_tokens,
            total_output_tokens = EXCLUDED.total_output_tokens,
            total_tokens = EXCLUDED.total_tokens,
            total_latency_ms = EXCLUDED.total_latency_ms,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            min_latency_ms = EXCLUDED.min_latency_ms,
            max_latency_ms = EXCLUDED.max_latency_ms,
            message_count = EXCLUDED.message_count,
            tool_execution_count = EXCLUDED.tool_execution_count,
            error_count = EXCLUDED.error_count,
            session_end = EXCLUDED.session_end,
            updated_at = NOW();
        """
        
        try:
            await run_query(metrics_query, access_token, refresh_token)
            
            # Mark as saved
            self.metrics_saved[chat_id] = True
            
            logger.info(
                f"session_metrics_saved - chat_id={chat_id}, "
                f"duration={session_duration_seconds:.1f}s, messages={message_count}, "
                f"turns={turn_count}, tokens={tokens['total']}"
            )
        except Exception as e:
            logger.error(f"session_metrics_save_error - chat_id={chat_id}, error={e}", exc_info=True)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on OPENAI pricing"""
        INPUT_RATE = 2 / 1000000
        OUTPUT_RATE = 10 / 1000000
        
        return (input_tokens * INPUT_RATE) + (output_tokens * OUTPUT_RATE)

    async def end_session(self, chat_id: str, access_token: str, refresh_token: str):
        """End session, force save, and record metrics"""
        if chat_id not in self.active_chats:
            logger.warning(f"end_session_chat_not_found - chat_id={chat_id}")
            return
        
        # Force save all messages
        await self.save_chat_to_db(chat_id, access_token, refresh_token, force=True)
        
        # Save session metrics
        await self.save_session_metrics(chat_id, access_token, refresh_token)
        
        logger.info(f"session_ended - chat_id={chat_id}")
    
    async def load_chat_history(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load chat list for user"""
        
        query = f"""
        SELECT 
            c.chat_id,
            c.topic,
            c.total_tokens,
            c.is_active,
            c.created_at,
            c.updated_at,
            COUNT(m.message_id) as message_count
        FROM chats c
        LEFT JOIN messages m ON c.chat_id = m.chat_id
        WHERE c.user_id = '{pg_escape(user_id)}'
        GROUP BY c.chat_id, c.topic, c.total_tokens, c.is_active, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT {limit};
        """
        
        try:
            chats = await run_query(query, access_token, refresh_token)
            logger.info(f"chat_history_loaded - user_id={user_id[:8]}..., count={len(chats)}")
            return chats
        except Exception as e:
            logger.error(f"chat_history_error - error={e}", exc_info=True)
            return []
    
    async def load_chat_messages(
        self,
        chat_id: str,
        user_id: str,
        access_token: str,
        refresh_token: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Load messages for a chat"""
        
        query = f"""
        SELECT role, content, tokens, created_at
        FROM messages
        WHERE chat_id = '{pg_escape(chat_id)}'
        ORDER BY created_at ASC
        LIMIT {limit};
        """
        
        try:
            messages = await run_query(query, access_token, refresh_token)
            return messages
        except Exception as e:
            logger.error(f"load_messages_error - error={e}", exc_info=True)
            return []
    
    def get_cached_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get messages from cache"""
        if chat_id not in self.active_chats:
            return []
        return self.active_chats[chat_id]['messages']
    
    async def get_user_metrics(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Get aggregate metrics for a user"""
        
        query = f"""
        SELECT 
            COUNT(DISTINCT c.chat_id) as total_chats,
            COALESCE(SUM(cm.message_count), 0) as total_messages,
            COALESCE(SUM(cm.total_tokens), 0) as total_tokens,
            COALESCE(SUM(cm.total_input_tokens), 0) as total_input_tokens,
            COALESCE(SUM(cm.total_output_tokens), 0) as total_output_tokens,
            COALESCE(AVG(cm.avg_latency_ms), 0) as avg_latency_ms,
            COALESCE(SUM(cm.tool_execution_count), 0) as total_tool_executions,
            MAX(cm.session_end) as last_activity
        FROM chats c
        LEFT JOIN chat_metrics cm ON c.chat_id = cm.chat_id
        WHERE c.user_id = '{pg_escape(user_id)}';
        """
        
        try:
            result = await run_query(query, access_token, refresh_token)
            if result:
                return result[0]
            return {}
        except Exception as e:
            logger.error(f"user_metrics_error - error={e}", exc_info=True)
            return {}
    
    async def get_system_metrics(
        self,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Get system-wide metrics for developers"""
        
        query = """
        SELECT 
            COUNT(DISTINCT c.chat_id) as total_chats,
            COUNT(DISTINCT c.user_id) as total_users,
            COALESCE(SUM(cm.message_count), 0) as total_messages,
            COALESCE(SUM(cm.total_tokens), 0) as total_tokens,
            COALESCE(AVG(cm.avg_latency_ms), 0) as avg_latency_ms,
            COALESCE(SUM(cm.tool_execution_count), 0) as total_tool_executions,
            COALESCE(SUM(cm.error_count), 0) as total_errors
        FROM chats c
        LEFT JOIN chat_metrics cm ON c.chat_id = cm.chat_id;
        """
        
        try:
            result = await run_query(query, access_token, refresh_token)
            if result:
                return result[0]
            return {}
        except Exception as e:
            logger.error(f"system_metrics_error - error={e}", exc_info=True)
            return {}


# Global instance
chat_manager = ChatSessionManager(
    max_context_multiplier=getattr(settings, 'MAX_CONTEXT_MULTIPLIER', 100),
    llm_context_limit=getattr(settings, 'LLM_MAX_TOKENS', 8000),
    session_timeout_minutes=getattr(settings, 'SESSION_TIMEOUT_MINUTES', 55),
    auto_save_interval_minutes=getattr(settings, 'AUTO_SAVE_INTERVAL_MINUTES', 5)
)