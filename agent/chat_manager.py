from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from database.client import run_query
from database.utils import pg_escape
from logs.log import logger
from config import settings
import uuid


class ChatSessionManager:
    """
    Manages chat sessions with PROPER token tracking.
    Fixed: Accumulates tokens correctly across entire chat session.
    """
    
    def __init__(
        self,
        max_context_multiplier: int = 100,
        llm_context_limit: int = 8000,
        session_timeout_minutes: int = 55
    ):
        self.max_tokens_per_chat = max_context_multiplier * llm_context_limit
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Cache for active chats
        # FIXED: Store cumulative token count properly
        self.active_chats: Dict[str, Dict[str, Any]] = {}
        # Format: {
        #   chat_id: {
        #     'messages': [],
        #     'cumulative_tokens': {
        #       'total': 0,
        #       'input': 0,
        #       'output': 0,
        #       'by_turn': []  # Track each turn
        #     },
        #     'session_id': '',
        #     'last_activity': datetime
        #   }
        # }
    
    def generate_chat_id(self) -> str:
        """Generate unique chat ID"""
        return f"chat_{uuid.uuid4().hex[:16]}"
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:16]}"
    
    async def create_new_chat(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new chat in database FIRST, then cache it"""
        
        chat_id = self.generate_chat_id()
        session_id = self.generate_session_id()
        
        # INSERT chat into database
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
            
            # Initialize cache with PROPER token tracking
            self.active_chats[chat_id] = {
                'messages': [],
                'cumulative_tokens': {
                    'total': 0,
                    'input': 0,
                    'output': 0,
                    'by_turn': []
                },
                'session_id': session_id,
                'last_activity': datetime.utcnow(),
                'user_id': user_id,
                'is_new': True
            }
            
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
        """Load existing chat from DB if not in cache"""
        
        # Check cache first
        if chat_id in self.active_chats:
            logger.info(f"chat_cache_hit - chat_id={chat_id}")
            return {
                'chat_id': chat_id,
                'session_id': self.active_chats[chat_id]['session_id'],
                'is_new': False,
                'total_tokens': self.active_chats[chat_id]['cumulative_tokens']['total'],
                'messages': self.active_chats[chat_id]['messages']
            }
        
        # Load from database
        logger.info(f"chat_cache_miss_loading - chat_id={chat_id}")
        
        # Get chat metadata
        chat_query = f"""
        SELECT chat_id, topic, total_tokens, created_at
        FROM chats
        WHERE chat_id = '{pg_escape(chat_id)}'
        AND user_id = '{pg_escape(user_id)}';
        """
        
        chat_result = await run_query(chat_query, access_token, refresh_token)
        
        if not chat_result:
            raise ValueError(f"Chat {chat_id} not found or access denied")
        
        # Load messages
        messages_query = f"""
        SELECT session_id, role, content, tokens, created_at
        FROM messages
        WHERE chat_id = '{pg_escape(chat_id)}'
        ORDER BY created_at ASC;
        """
        
        messages_result = await run_query(messages_query, access_token, refresh_token)
        
        # Calculate cumulative tokens from loaded messages
        cumulative_input = 0
        cumulative_output = 0
        turn_history = []
        
        for msg in messages_result:
            msg_tokens = msg.get('tokens', 0)
            if msg['role'] == 'user':
                cumulative_input += msg_tokens
            elif msg['role'] == 'assistant':
                cumulative_output += msg_tokens
            
            turn_history.append({
                'role': msg['role'],
                'tokens': msg_tokens,
                'timestamp': msg['created_at']
            })
        
        cumulative_total = cumulative_input + cumulative_output
        
        # Cache the loaded chat
        session_id = self.generate_session_id()
        self.active_chats[chat_id] = {
            'messages': [
                {
                    'role': msg['role'],
                    'content': msg['content'],
                    'tokens': msg['tokens'],
                    'session_id': msg['session_id']
                }
                for msg in messages_result
            ],
            'cumulative_tokens': {
                'total': cumulative_total,
                'input': cumulative_input,
                'output': cumulative_output,
                'by_turn': turn_history
            },
            'session_id': session_id,
            'last_activity': datetime.utcnow(),
            'user_id': user_id,
            'is_new': False
        }
        
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
    
    def add_message_to_cache(
        self,
        chat_id: str,
        role: str,
        content: str,
        tokens: int
    ):
        """
        Add message to cache with PROPER token accumulation.
        FIXED: Now correctly accumulates tokens across the session.
        """
        
        if chat_id not in self.active_chats:
            logger.warning(f"chat_not_in_cache - chat_id={chat_id}")
            return
        
        chat_data = self.active_chats[chat_id]
        
        # Add message
        chat_data['messages'].append({
            'role': role,
            'content': content,
            'tokens': tokens,
            'session_id': chat_data['session_id']
        })
        
        # FIXED: Properly accumulate tokens
        chat_data['cumulative_tokens']['total'] += tokens
        
        if role == 'user':
            chat_data['cumulative_tokens']['input'] += tokens
        elif role == 'assistant':
            chat_data['cumulative_tokens']['output'] += tokens
        
        # Track by turn
        chat_data['cumulative_tokens']['by_turn'].append({
            'role': role,
            'tokens': tokens,
            'cumulative_total': chat_data['cumulative_tokens']['total'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        chat_data['last_activity'] = datetime.utcnow()
        
        logger.info(
            f"message_cached - chat_id={chat_id}, role={role}, "
            f"msg_tokens={tokens}, cumulative_total={chat_data['cumulative_tokens']['total']}, "
            f"cumulative_input={chat_data['cumulative_tokens']['input']}, "
            f"cumulative_output={chat_data['cumulative_tokens']['output']}"
        )
    
    async def check_token_limit(self, chat_id: str) -> bool:
        """Check if chat has exceeded token limit (from cumulative count)"""
        
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
        """Get detailed token statistics for a chat"""
        if chat_id not in self.active_chats:
            return {}
        
        return self.active_chats[chat_id]['cumulative_tokens']
    
    async def save_chat_to_db(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str
    ) -> int:
        """Batch save all cached messages to database"""
        
        if chat_id not in self.active_chats:
            logger.warning(f"chat_not_cached - chat_id={chat_id}")
            return 0
        
        cached = self.active_chats[chat_id]
        messages = cached['messages']
        
        if not messages:
            logger.info(f"no_messages_to_save - chat_id={chat_id}")
            return 0
        
        try:
            # Determine new messages
            if cached.get('is_new', False):
                new_messages = messages
            else:
                # Get existing count from DB
                count_query = f"""
                SELECT COUNT(*) as count
                FROM messages
                WHERE chat_id = '{pg_escape(chat_id)}';
                """
                count_result = await run_query(count_query, access_token, refresh_token)
                existing_count = count_result[0]['count'] if count_result else 0
                new_messages = messages[existing_count:]
            
            if not new_messages:
                logger.info(f"all_messages_saved - chat_id={chat_id}")
                return 0
            
            # Batch insert
            values = []
            for msg in new_messages:
                values.append(
                    f"('{chat_id}', '{msg['session_id']}', '{pg_escape(msg['role'])}', "
                    f"'{pg_escape(msg['content'])}', {msg['tokens']}, NOW())"
                )
            
            insert_query = f"""
            INSERT INTO messages (chat_id, session_id, role, content, tokens, created_at)
            VALUES {', '.join(values)};
            """
            
            await run_query(insert_query, access_token, refresh_token)
            
            # Update chat with CUMULATIVE total
            cumulative_total = cached['cumulative_tokens']['total']
            
            update_query = f"""
            UPDATE chats
            SET total_tokens = {cumulative_total},
                updated_at = NOW()
            WHERE chat_id = '{chat_id}';
            """
            
            await run_query(update_query, access_token, refresh_token)
            
            logger.info(
                f"chat_saved - chat_id={chat_id}, new_messages={len(new_messages)}, "
                f"cumulative_tokens={cumulative_total}"
            )
            
            cached['is_new'] = False
            return len(new_messages)
            
        except Exception as e:
            logger.error(f"chat_save_error - chat_id={chat_id}, error={e}", exc_info=True)
            raise
    
    async def switch_chat(
        self,
        old_chat_id: Optional[str],
        new_chat_id: str,
        user_id: str,
        access_token: str,
        refresh_token: str
    ):
        """Save old chat and load new chat"""
        
        if old_chat_id and old_chat_id in self.active_chats:
            logger.info(f"saving_old_chat - chat_id={old_chat_id}")
            await self.save_chat_to_db(old_chat_id, access_token, refresh_token)
        
        logger.info(f"loading_new_chat - chat_id={new_chat_id}")
        return await self.get_or_load_chat(new_chat_id, user_id, access_token, refresh_token)
    
    async def end_session(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str
    ):
        """End session and save everything"""
        
        if chat_id in self.active_chats:
            await self.save_chat_to_db(chat_id, access_token, refresh_token)
            logger.info(f"session_ended - chat_id={chat_id}")
    
    async def cleanup_inactive_chats(self):
        """Remove inactive chats from cache"""
        
        now = datetime.utcnow()
        inactive = []
        
        for chat_id, data in self.active_chats.items():
            if now - data['last_activity'] > self.session_timeout:
                inactive.append(chat_id)
        
        for chat_id in inactive:
            del self.active_chats[chat_id]
            logger.info(f"chat_cleaned - chat_id={chat_id}")
    
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
    
    def get_cached_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get messages from cache"""
        
        if chat_id not in self.active_chats:
            return []
        
        return self.active_chats[chat_id]['messages']


# Global instance
chat_manager = ChatSessionManager(
    max_context_multiplier=getattr(settings, 'MAX_CONTEXT_MULTIPLIER', 100),
    llm_context_limit=getattr(settings, 'LLM_MAX_TOKENS', 8000),
    session_timeout_minutes=getattr(settings, 'SESSION_TIMEOUT_MINUTES', 55)
)