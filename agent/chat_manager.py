from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from database.client import run_query
from database.utils import pg_escape
from logs.log import logger
from config import settings
import uuid

class ChatSessionManager:
    """Manages chat sessions with proper caching and batch saves"""
    
    def __init__(
        self,
        max_context_multiplier: int = 10,
        llm_context_limit: int = 8000,
        session_timeout_minutes: int = 5
    ):
        self.max_tokens_per_chat = max_context_multiplier * llm_context_limit
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Cache for active chats (in-memory message storage)
        self.active_chats: Dict[str, Dict[str, Any]] = {}
        # Format: {chat_id: {'messages': [], 'total_tokens': 0, 'session_id': '', 'last_activity': datetime}}
        
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
        
        # INSERT chat into database FIRST
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
            logger.info(f"✅ Created new chat in DB: {chat_id}")
            
            # Initialize in-memory cache
            self.active_chats[chat_id] = {
                'messages': [],
                'total_tokens': 0,
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
            logger.error(f"❌ Error creating chat: {e}")
            raise
    
    async def get_or_load_chat(
        self,
        chat_id: str,
        user_id: str,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Load existing chat from DB if not in cache"""
        
        # Check if already cached
        if chat_id in self.active_chats:
            logger.info(f"📦 Using cached chat: {chat_id}")
            return {
                'chat_id': chat_id,
                'session_id': self.active_chats[chat_id]['session_id'],
                'is_new': False,
                'total_tokens': self.active_chats[chat_id]['total_tokens'],
                'messages': self.active_chats[chat_id]['messages']
            }
        
        # Load from database
        logger.info(f"💾 Loading chat from DB: {chat_id}")
        
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
        
        # Cache the loaded chat
        session_id = self.generate_session_id()  # New session for loaded chat
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
            'total_tokens': chat_result[0]['total_tokens'],
            'session_id': session_id,
            'last_activity': datetime.utcnow(),
            'user_id': user_id,
            'is_new': False
        }
        
        return {
            'chat_id': chat_id,
            'session_id': session_id,
            'is_new': False,
            'total_tokens': self.active_chats[chat_id]['total_tokens'],
            'messages': self.active_chats[chat_id]['messages']
        }
    
    def add_message_to_cache(
        self,
        chat_id: str,
        role: str,
        content: str,
        tokens: int
    ):
        """Add message to in-memory cache only"""
        
        if chat_id not in self.active_chats:
            logger.warning(f"⚠️ Chat {chat_id} not in cache, skipping message")
            return
        
        self.active_chats[chat_id]['messages'].append({
            'role': role,
            'content': content,
            'tokens': tokens,
            'session_id': self.active_chats[chat_id]['session_id']
        })
        
        self.active_chats[chat_id]['total_tokens'] += tokens
        self.active_chats[chat_id]['last_activity'] = datetime.utcnow()
        
        logger.info(f"📝 Message cached for {chat_id} (total: {len(self.active_chats[chat_id]['messages'])} msgs)")
    
    async def check_token_limit(self, chat_id: str) -> bool:
        """Check if chat has exceeded token limit (from cache)"""
        
        if chat_id not in self.active_chats:
            return True
        
        total_tokens = self.active_chats[chat_id]['total_tokens']
        
        if total_tokens >= self.max_tokens_per_chat:
            logger.warning(f"⚠️ Chat {chat_id} exceeded limit: {total_tokens}/{self.max_tokens_per_chat}")
            return False
        
        return True
    
    async def save_chat_to_db(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str
    ) -> int:
        """Batch save all cached messages to database"""
        
        if chat_id not in self.active_chats:
            logger.warning(f"⚠️ No cached data for chat {chat_id}")
            return 0
        
        cached = self.active_chats[chat_id]
        messages = cached['messages']
        
        if not messages:
            logger.info(f"ℹ️ No messages to save for {chat_id}")
            return 0
        
        try:
            # Check if chat was newly created (only has cached messages)
            if cached.get('is_new', False):
                # All messages are new, insert all
                new_messages = messages
            else:
                # Load existing message count from DB
                count_query = f"""
                SELECT COUNT(*) as count
                FROM messages
                WHERE chat_id = '{pg_escape(chat_id)}';
                """
                count_result = await run_query(count_query, access_token, refresh_token)
                existing_count = count_result[0]['count'] if count_result else 0
                
                # Only save messages that aren't in DB yet
                new_messages = messages[existing_count:]
            
            if not new_messages:
                logger.info(f"ℹ️ All messages already saved for {chat_id}")
                return 0
            
            # Batch insert new messages
            values = []
            total_new_tokens = 0
            
            for msg in new_messages:
                values.append(
                    f"('{chat_id}', '{msg['session_id']}', '{pg_escape(msg['role'])}', "
                    f"'{pg_escape(msg['content'])}', {msg['tokens']}, NOW())"
                )
                total_new_tokens += msg['tokens']
            
            insert_query = f"""
            INSERT INTO messages (chat_id, session_id, role, content, tokens, created_at)
            VALUES {', '.join(values)};
            """
            
            await run_query(insert_query, access_token, refresh_token)
            
            # Update chat total tokens
            update_query = f"""
            UPDATE chats
            SET total_tokens = {cached['total_tokens']},
                updated_at = NOW()
            WHERE chat_id = '{chat_id}';
            """
            
            await run_query(update_query, access_token, refresh_token)
            
            logger.info(f"💾 Saved {len(new_messages)} messages for {chat_id} ({total_new_tokens} tokens)")
            
            # Mark as no longer new
            cached['is_new'] = False
            
            return len(new_messages)
            
        except Exception as e:
            logger.error(f"❌ Error saving chat {chat_id}: {e}")
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
        
        # Save old chat if exists
        if old_chat_id and old_chat_id in self.active_chats:
            logger.info(f"💾 Saving old chat: {old_chat_id}")
            await self.save_chat_to_db(old_chat_id, access_token, refresh_token)
        
        # Load new chat
        logger.info(f"📂 Loading new chat: {new_chat_id}")
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
            logger.info(f"✅ Session ended and saved for {chat_id}")
    
    async def cleanup_inactive_chats(self):
        """Remove inactive chats from cache (optional optimization)"""
        
        now = datetime.utcnow()
        inactive = []
        
        for chat_id, data in self.active_chats.items():
            if now - data['last_activity'] > self.session_timeout:
                inactive.append(chat_id)
        
        for chat_id in inactive:
            del self.active_chats[chat_id]
            logger.info(f"🗑️ Cleaned up inactive chat: {chat_id}")
    
    async def load_chat_history(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load chat list for user (metadata only)"""
        
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
            logger.info(f"📋 Loaded {len(chats)} chat summaries for user {user_id}")
            return chats
        except Exception as e:
            logger.error(f"❌ Error loading chat history: {e}")
            return []
    
    def get_cached_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get messages from cache (for display)"""
        
        if chat_id not in self.active_chats:
            return []
        
        return self.active_chats[chat_id]['messages']


# Global instance
chat_manager = ChatSessionManager(
    session_timeout_minutes=getattr(settings, 'SESSION_TIMEOUT_MINUTES', 5)
)