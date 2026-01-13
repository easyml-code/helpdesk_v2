from fastapi import APIRouter, HTTPException, Depends, Header, Cookie, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import agent_graph
from agent.cache_manager import cache_manager
from agent.chat_manager import chat_manager
from agent.state import AgentState
from database.client import get_access_token
from logs.log import logger, set_trace_id, set_request_id, set_user_id, clear_context
from metrics.prometheus import (
    track_http_request, requests_per_user, 
    user_sessions_active, get_metrics,
    active_chats_gauge, chat_messages_total
)
from config import settings
import time
import jwt
import hashlib
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)
router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
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


class UserMetricsResponse(BaseModel):
    total_chats: int
    total_messages: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float
    total_tool_executions: int
    last_activity: Optional[str]  # Now properly typed as Optional[str]


class SystemMetricsResponse(BaseModel):
    total_chats: int
    total_users: int
    total_messages: int
    total_tokens: int
    avg_latency_ms: float
    total_tool_executions: int
    total_errors: int


# ============================================================================
# AUTHENTICATION
# ============================================================================

async def authenticate_user(
    auth: HTTPAuthorizationCredentials = Depends(security),
    x_refresh_token: str | None = Header(None, alias="X-Refresh-Token"),
    refresh_cookie: str | None = Cookie(None, alias="refresh_token"),
) -> dict:
    """Authenticate user and return tokens"""
    if not auth or not auth.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    
    refresh_token = x_refresh_token or refresh_cookie
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    
    return {"access_token": auth.credentials, "refresh_token": refresh_token}


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
        logger.error(f"token_decode_error - error={e}")
        raise HTTPException(status_code=401, detail="Invalid access token")


def hash_user_id(user_id: str) -> str:
    """Hash user ID for privacy in metrics"""
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def background_save_chat(chat_id: str, access_token: str, refresh_token: str):
    """Background task to save chat"""
    try:
        await chat_manager.save_chat_to_db(chat_id, access_token, refresh_token, force=True)
    except Exception as e:
        logger.error(f"background_save_error - chat_id={chat_id}, error={e}")


# ============================================================================
# ROUTES
# ============================================================================

@router.post("/auth/login")
async def login(email: str, password: str):
    """Authenticate user and return tokens"""
    start_time = time.time()
    set_trace_id()
    set_request_id()
    
    try:
        access_token, refresh_token = await get_access_token(email, password)
        
        await track_request("/auth/login", "POST", start_time, 200)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        await track_request("/auth/login", "POST", start_time, e.status_code)
        raise
    except Exception as e:
        logger.error(f"login_error - error={e}", exc_info=True)
        await track_request("/auth/login", "POST", start_time, 500)
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        clear_context()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    tokens: dict = Depends(authenticate_user)
):
    """Process chat message with auto-save"""
    start_time = time.time()
    set_trace_id()
    set_request_id()
    
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    try:
        # Get user ID
        user_id = await get_user_from_token(access_token)
        set_user_id(user_id)
        user_id_hash = hash_user_id(user_id)
        
        # Track user request
        requests_per_user.labels(user_id_hash=user_id_hash).inc()
        chat_messages_total.labels(role="user").inc()
        
        # Get or create chat
        if request.chat_id:
            chat_id = request.chat_id
            chat_info = await chat_manager.get_or_load_chat(
                chat_id, user_id, access_token, refresh_token
            )
            is_new_chat = False
            logger.info(f"chat_continue - chat_id={chat_id}")
        else:
            chat_info = await chat_manager.create_new_chat(
                user_id, access_token, refresh_token, request.topic
            )
            chat_id = chat_info['chat_id']
            is_new_chat = True
            logger.info(f"chat_created - chat_id={chat_id}")
            
            # Create cache session
            cache_manager.create_chat_session(chat_id, user_id, chat_info['session_id'])
        
        # Track active chats
        active_chats_gauge.inc()
        user_sessions_active.inc()
        
        # Get cached messages
        cached_messages = chat_manager.get_cached_messages(chat_id)
        
        # Convert to LangChain format
        lc_messages = []
        for msg in cached_messages:
            if msg['role'] == 'user':
                lc_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                lc_messages.append(AIMessage(content=msg['content']))
        
        # Add new user message
        lc_messages.append(HumanMessage(content=request.message))
        
        # Prepare config
        config = {
            "configurable": {
                "thread_id": chat_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "max_context_tokens": settings.MAX_CONTEXT_MULTIPLIER * settings.LLM_MAX_TOKENS
            }
        }

        # Prepare state
        initial_state: AgentState = {
            "messages": lc_messages,
            "chat_id": chat_id,
            "session_id": chat_info['session_id'],
            "user_id": user_id,
            "current_topic": request.topic,
            "total_tokens": chat_info.get('total_tokens', 0),
            "session_start_time": time.time(),
            "config": config["configurable"],
            "results": []
        }
        
        # Run agent
        logger.info(f"agent_invoke_start - chat_id={chat_id}")
        result = await agent_graph.ainvoke(initial_state, config)
        
        # Extract response
        ai_response = result["messages"][-1].content if result["messages"] else "No response"
        
        # Schedule background save
        background_tasks.add_task(
            background_save_chat,
            chat_id,
            access_token,
            refresh_token
        )
        
        # Track success
        await track_request("/chat", "POST", start_time, 200)
        
        return ChatResponse(
            response=ai_response,
            chat_id=chat_id,
            session_id=chat_info['session_id'],
            is_new_chat=is_new_chat
        )
        
    except Exception as e:
        logger.error(f"chat_error - error={e}", exc_info=True)
        await track_request("/chat", "POST", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_chats_gauge.dec()
        user_sessions_active.dec()
        clear_context()


@router.post("/chat/{chat_id}/end")
async def end_chat(
    chat_id: str,
    tokens: dict = Depends(authenticate_user)
):
    """End chat session, force save messages and metrics"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        # End session (saves messages + metrics)
        await chat_manager.end_session(
            chat_id,
            tokens["access_token"],
            tokens["refresh_token"]
        )
        
        await track_request(f"/chat/{chat_id}/end", "POST", start_time, 200)
        
        return {
            "status": "success",
            "chat_id": chat_id,
            "message": "Chat session ended, messages and metrics saved"
        }
        
    except Exception as e:
        logger.error(f"end_chat_error - error={e}", exc_info=True)
        await track_request(f"/chat/{chat_id}/end", "POST", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    tokens: dict = Depends(authenticate_user)
):
    """Get chat history for user"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        chats = await chat_manager.load_chat_history(
            user_id,
            tokens["access_token"],
            tokens["refresh_token"],
            limit=getattr(settings, 'CHAT_HISTORY_LIMIT', 50)
        )
        
        await track_request("/chat/history", "GET", start_time, 200)
        
        return ChatHistoryResponse(
            chats=chats,
            total=len(chats)
        )
        
    except Exception as e:
        logger.error(f"chat_history_error - error={e}", exc_info=True)
        await track_request("/chat/history", "GET", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/chat/{chat_id}/messages", response_model=MessageHistoryResponse)
async def get_chat_messages(
    chat_id: str,
    tokens: dict = Depends(authenticate_user)
):
    """Get messages for a specific chat"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        messages = await chat_manager.load_chat_messages(
            chat_id,
            user_id,
            tokens["access_token"],
            tokens["refresh_token"],
            limit=getattr(settings, 'MESSAGE_HISTORY_LIMIT', 100)
        )
        
        await track_request(f"/chat/{chat_id}/messages", "GET", start_time, 200)
        
        return MessageHistoryResponse(
            messages=messages,
            chat_id=chat_id,
            total=len(messages)
        )
        
    except Exception as e:
        logger.error(f"get_messages_error - error={e}", exc_info=True)
        await track_request(f"/chat/{chat_id}/messages", "GET", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()



@router.get("/metrics/user", response_model=UserMetricsResponse)
async def get_user_metrics(
    tokens: dict = Depends(authenticate_user)
):
    """Get aggregate metrics for current user"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        metrics = await chat_manager.get_user_metrics(
            user_id,
            tokens["access_token"],
            tokens["refresh_token"]
        )
        
        await track_request("/metrics/user", "GET", start_time, 200)
        
        # FIX: Convert datetime to ISO string
        last_activity = metrics.get('last_activity')
        if last_activity and not isinstance(last_activity, str):
            last_activity = last_activity.isoformat() if hasattr(last_activity, 'isoformat') else str(last_activity)
        
        return UserMetricsResponse(
            total_chats=metrics.get('total_chats', 0),
            total_messages=metrics.get('total_messages', 0),
            total_tokens=metrics.get('total_tokens', 0),
            total_input_tokens=metrics.get('total_input_tokens', 0),
            total_output_tokens=metrics.get('total_output_tokens', 0),
            avg_latency_ms=metrics.get('avg_latency_ms', 0),
            total_tool_executions=metrics.get('total_tool_executions', 0),
            last_activity=last_activity
        )
        
    except Exception as e:
        logger.error(f"user_metrics_error - error={e}", exc_info=True)
        await track_request("/metrics/user", "GET", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    tokens: dict = Depends(authenticate_user)
):
    """Get system-wide metrics (for developers/admins)"""
    start_time = time.time()
    set_trace_id()
    
    try:
        # In production, you might want to add admin role check here
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        metrics = await chat_manager.get_system_metrics(
            tokens["access_token"],
            tokens["refresh_token"]
        )
        
        await track_request("/metrics/system", "GET", start_time, 200)
        
        return SystemMetricsResponse(
            total_chats=metrics.get('total_chats', 0),
            total_users=metrics.get('total_users', 0),
            total_messages=metrics.get('total_messages', 0),
            total_tokens=metrics.get('total_tokens', 0),
            avg_latency_ms=metrics.get('avg_latency_ms', 0),
            total_tool_executions=metrics.get('total_tool_executions', 0),
            total_errors=metrics.get('total_errors', 0)
        )
        
    except Exception as e:
        logger.error(f"system_metrics_error - error={e}", exc_info=True)
        await track_request("/metrics/system", "GET", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_chats": active_chats_gauge._value._value
    }



async def track_request(endpoint: str, method: str, start_time: float, status: int):
    """Track HTTP request metrics"""
    duration = time.time() - start_time
    track_http_request(method, endpoint, status, duration)