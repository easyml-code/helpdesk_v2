from fastapi import APIRouter, HTTPException, Depends, Header, Cookie, Response
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from agent.cache_manager import cache_manager
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
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database.utils import get_new_tokens

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


class VersionSwitchRequest(BaseModel):
    logical_message_id: str
    target_version_id: str


class MessageVersionsResponse(BaseModel):
    logical_message_id: str
    versions: List[dict]
    active_version_id: str


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
# MIDDLEWARE HELPERS
# ============================================================================

async def track_request(endpoint: str, method: str, start_time: float, status: int):
    """Track HTTP request metrics"""
    duration = time.time() - start_time
    track_http_request(method, endpoint, status, duration)


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
    tokens: dict = Depends(authenticate_user)
):
    """Process chat message"""
    start_time = time.time()
    set_trace_id()
    set_request_id()
    
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    try:
        # Get user ID and set context
        user_id = await get_user_from_token(access_token)
        set_user_id(user_id)
        user_id_hash = hash_user_id(user_id)
        
        # Track user request
        requests_per_user.labels(user_id_hash=user_id_hash).inc()
        chat_messages_total.labels(role="user").inc()
        
        # Get or create chat
        if request.chat_id:
            chat_id = request.chat_id
            is_new_chat = False
            
            # Ensure chat exists in cache
            if not cache_manager.get_active_messages(chat_id):
                cache_manager.create_chat_session(chat_id)
            
            logger.info(f"chat_continue - chat_id={chat_id}, user_id_hash={user_id_hash}")
        else:
            # Create new chat
            from agent.cache_manager import cache_manager as cm
            chat_id = cm._generate_id("chat")
            cache_manager.create_chat_session(chat_id)
            is_new_chat = True
            
            logger.info(f"chat_created - chat_id={chat_id}, user_id_hash={user_id_hash}")
        
        # Track active chats
        active_chats_gauge.inc()
        user_sessions_active.inc()
        
        # Get cached messages
        cached_messages = cache_manager.get_active_messages(chat_id)
        
        # Convert to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage
        lc_messages = []
        for msg in cached_messages:
            lc_messages.append(HumanMessage(content=msg['user_content']))
            lc_messages.append(AIMessage(content=msg['assistant_content']))
        
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
            "session_id": f"session_{int(time.time())}",
            "user_id": user_id,
            "current_topic": request.topic,
            "total_tokens": 0,
            "session_start_time": time.time(),
            "config": config["configurable"],
            "results": []
        }
        
        # Run agent
        logger.info(f"agent_invoke_start - chat_id={chat_id}")
        result = await agent_graph.ainvoke(initial_state, config)
        
        # Extract response
        ai_response = result["messages"][-1].content if result["messages"] else "No response generated"
        
        # Track success
        await track_request("/chat", "POST", start_time, 200)
        
        return ChatResponse(
            response=ai_response,
            chat_id=chat_id,
            session_id=initial_state["session_id"],
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


@router.post("/chat/{chat_id}/version/switch")
async def switch_message_version(
    chat_id: str,
    request: VersionSwitchRequest,
    tokens: dict = Depends(authenticate_user)
):
    """Switch to a different version of a message"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        success = cache_manager.switch_message_version(
            chat_id=chat_id,
            logical_msg_id=request.logical_message_id,
            target_version_id=request.target_version_id
        )
        
        if not success:
            await track_request(f"/chat/{chat_id}/version/switch", "POST", start_time, 404)
            raise HTTPException(status_code=404, detail="Version not found")
        
        await track_request(f"/chat/{chat_id}/version/switch", "POST", start_time, 200)
        
        return {
            "status": "success",
            "chat_id": chat_id,
            "logical_message_id": request.logical_message_id,
            "active_version_id": request.target_version_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"version_switch_error - error={e}", exc_info=True)
        await track_request(f"/chat/{chat_id}/version/switch", "POST", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/chat/{chat_id}/message/{message_id}/versions", response_model=MessageVersionsResponse)
async def get_message_versions(
    chat_id: str,
    message_id: str,
    tokens: dict = Depends(authenticate_user)
):
    """Get all versions of a specific message"""
    start_time = time.time()
    set_trace_id()
    
    try:
        user_id = await get_user_from_token(tokens["access_token"])
        set_user_id(user_id)
        
        versions = cache_manager.get_message_versions(chat_id, message_id)
        
        if not versions:
            await track_request(f"/chat/{chat_id}/message/{message_id}/versions", "GET", start_time, 404)
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Find active version
        active_version = next(
            (v["version_id"] for v in versions if v["is_active"]),
            versions[0]["version_id"] if versions else None
        )
        
        await track_request(f"/chat/{chat_id}/message/{message_id}/versions", "GET", start_time, 200)
        
        return MessageVersionsResponse(
            logical_message_id=message_id,
            versions=versions,
            active_version_id=active_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_versions_error - error={e}", exc_info=True)
        await track_request(f"/chat/{chat_id}/message/{message_id}/versions", "GET", start_time, 500)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
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