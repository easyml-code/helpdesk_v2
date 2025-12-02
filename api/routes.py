from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from agent.chat_manager import chat_manager
from agent.state import AgentState
from database.client import get_access_token
from logs.log import logger
from config import settings
import time
import jwt


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None  # If None -> create new, if provided -> use existing
    topic: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    session_id: str
    is_new_chat: bool


class ChatHistoryResponse(BaseModel):
    chats: List[dict]
    total: int


class MessageHistoryResponse(BaseModel):
    messages: List[dict]
    chat_id: str
    total: int


async def get_user_from_token(access_token: str) -> str:
    """Extract user ID from JWT token"""
    try:
        decoded = jwt.decode(
            access_token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True}
        )
        return decoded.get("sub")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid access token")


@router.post("/auth/login")
async def login(email: str, password: str):
    """Authenticate user and return tokens"""
    try:
        access_token, refresh_token = await get_access_token(email, password)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    access_token: str,
    refresh_token: str
):
    """Send message and get AI response"""
    
    try:
        # Get user ID from token
        user_id = await get_user_from_token(access_token)
        
        # Determine if new chat or existing
        if request.chat_id:
            # Load existing chat (will use cache if available)
            chat_info = await chat_manager.get_or_load_chat(
                chat_id=request.chat_id,
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token
            )
            logger.info(f"📂 Using existing chat: {request.chat_id}")
        else:
            # Create new chat in DB FIRST
            chat_info = await chat_manager.create_new_chat(
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                topic=request.topic
            )
            logger.info(f"✨ Created new chat: {chat_info['chat_id']}")
        
        chat_id = chat_info['chat_id']
        session_id = chat_info['session_id']
        is_new_chat = chat_info['is_new']
        
        # Check token limit
        if not await chat_manager.check_token_limit(chat_id):
            return ChatResponse(
                response="This chat has reached its maximum length. Please start a new chat to continue.",
                chat_id=chat_id,
                session_id=session_id,
                is_new_chat=False
            )
        
        # Prepare initial state with cached messages
        cached_messages = chat_manager.get_cached_messages(chat_id)
        
        # Convert cached messages to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage
        lc_messages = []
        for msg in cached_messages:
            if msg['role'] == 'user':
                lc_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                lc_messages.append(AIMessage(content=msg['content']))
        
        # Add new user message
        lc_messages.append(HumanMessage(content=request.message))
        
        initial_state: AgentState = {
            "messages": lc_messages,
            "chat_id": chat_id,
            "session_id": session_id,
            "user_id": user_id,
            "current_topic": request.topic,
            "total_tokens": chat_info.get('total_tokens', 0),
            "session_start_time": time.time()
        }
        
        # Run agent graph
        config = {
            "configurable": {
                "thread_id": chat_id,
                "access_token": access_token,
                "refresh_token": refresh_token
            }
        }
        
        logger.info(f"🤖 Processing message for chat {chat_id}")
        result = await agent_graph.ainvoke(initial_state, config)
        
        # Extract AI response
        ai_response = result["messages"][-1].content if result["messages"] else "No response generated"
        
        return ChatResponse(
            response=ai_response,
            chat_id=chat_id,
            session_id=session_id,
            is_new_chat=is_new_chat
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/switch")
async def switch_chat(
    old_chat_id: Optional[str],
    new_chat_id: str,
    access_token: str,
    refresh_token: str
):
    """Switch from one chat to another (saves old, loads new)"""
    
    try:
        user_id = await get_user_from_token(access_token)
        
        chat_info = await chat_manager.switch_chat(
            old_chat_id=old_chat_id,
            new_chat_id=new_chat_id,
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token
        )
        
        return {
            "status": "success",
            "chat_id": new_chat_id,
            "message_count": len(chat_info['messages'])
        }
        
    except Exception as e:
        logger.error(f"❌ Error switching chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    access_token: str,
    refresh_token: str,
    limit: int = 50
):
    """Get all chats for the authenticated user (metadata only)"""
    
    try:
        user_id = await get_user_from_token(access_token)
        
        chats = await chat_manager.load_chat_history(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            limit=limit
        )
        
        return ChatHistoryResponse(
            chats=chats,
            total=len(chats)
        )
        
    except Exception as e:
        logger.error(f"❌ Error loading chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{chat_id}/messages", response_model=MessageHistoryResponse)
async def get_chat_messages(
    chat_id: str,
    access_token: str,
    refresh_token: str
):
    """Get messages for a specific chat (from cache or DB)"""
    
    try:
        user_id = await get_user_from_token(access_token)
        
        # Try cache first
        cached = chat_manager.get_cached_messages(chat_id)
        
        if cached:
            logger.info(f"📦 Returning cached messages for {chat_id}")
            return MessageHistoryResponse(
                messages=cached,
                chat_id=chat_id,
                total=len(cached)
            )
        
        # Load from DB
        chat_info = await chat_manager.get_or_load_chat(
            chat_id=chat_id,
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token
        )
        
        return MessageHistoryResponse(
            messages=chat_info['messages'],
            chat_id=chat_id,
            total=len(chat_info['messages'])
        )
        
    except Exception as e:
        logger.error(f"❌ Error loading messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/{chat_id}/end")
async def end_chat_session(
    chat_id: str,
    access_token: str,
    refresh_token: str
):
    """End session and save all cached messages"""
    
    try:
        await chat_manager.end_session(
            chat_id=chat_id,
            access_token=access_token,
            refresh_token=refresh_token
        )
        
        return {"status": "success", "message": "Session ended and messages saved"}
        
    except Exception as e:
        logger.error(f"❌ Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def run_sql_query(
    query: str
):
    """Run arbitrary SQL query against the database"""
    from database.client import run_query
    access_token, refresh_token = await get_access_token(
        email="vikram.dev@pwc.com",
        password="Test@123"
    )

    try:
        results = await run_query(
            query=query,
            access_token=access_token,
            refresh_token=refresh_token
        )
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ SQL query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))