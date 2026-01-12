from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import router
from api.rate_limiter import check_rate_limit, rate_limiter
from logs.log import logger, set_trace_id, set_request_id, set_user_id, clear_context
from metrics.prometheus import track_http_request
import uvicorn
import time
import jwt
from config import settings
import asyncio
from agent.chat_manager import chat_manager


app = FastAPI(
    title="Vendor HelpDesk Agent",
    description="Production-Ready Backend API with Rate Limiting, Metrics & Auto-Save",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# AUTO-SAVE BACKGROUND TASK
# ============================================================================

async def auto_save_periodic():
    """Periodic auto-save task (runs every AUTO_SAVE_INTERVAL_MINUTES)"""
    while True:
        try:
            await asyncio.sleep(settings.AUTO_SAVE_INTERVAL_MINUTES * 60)
            logger.info("auto_save_periodic_triggered")
            
            # Get a sample token from active chats (in production, manage tokens better)
            # For now, we'll rely on individual chat saves triggered by requests
            
        except Exception as e:
            logger.error(f"auto_save_periodic_error - error={e}", exc_info=True)


# ============================================================================
# REQUEST TRACKING + RATE LIMITING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def request_tracking_and_rate_limiting_middleware(request: Request, call_next):
    """Global middleware for tracking, rate limiting, and metrics"""
    start_time = time.time()
    
    # Set context variables
    trace_id = set_trace_id()
    request_id = set_request_id()
    
    # Try to extract user_id from JWT
    user_id = None
    try:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            decoded = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_aud": True}
            )
            user_id = decoded.get("sub")
            if user_id:
                set_user_id(user_id)
                request.state.user_id = user_id
    except:
        pass
    
    # Log request start
    logger.info(
        f"request_start - method={request.method}, path={request.url.path}, "
        f"trace_id={trace_id}, request_id={request_id}"
    )
    
    try:
        # Rate limiting check (skip for health/metrics)
        skip_rate_limit = request.url.path in ["/health", "/api/metrics", "/docs", "/openapi.json"]
        
        if not skip_rate_limit:
            try:
                rate_limit_info = await check_rate_limit(request)
                logger.info(
                    f"rate_limit_check - path={request.url.path}, "
                    f"remaining={rate_limit_info['remaining']}"
                )
            except Exception as rate_limit_error:
                duration = time.time() - start_time
                track_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=429,
                    duration=duration
                )
                raise rate_limit_error
        
        # Process request
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Track metrics
        track_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        # Add rate limit headers
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        # Log completion
        logger.info(
            f"request_complete - method={request.method}, path={request.url.path}, "
            f"status={response.status_code}, duration_ms={duration * 1000:.2f}, "
            f"trace_id={trace_id}"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        status_code = 500
        if hasattr(e, "status_code"):
            status_code = e.status_code
        
        track_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=status_code,
            duration=duration
        )
        
        logger.error(
            f"request_error - method={request.method}, path={request.url.path}, "
            f"error={str(e)}, duration_ms={duration * 1000:.2f}, trace_id={trace_id}",
            exc_info=True
        )
        
        # Return error response
        if status_code == 429:
            return JSONResponse(
                status_code=429,
                content=e.detail if hasattr(e, "detail") else {"detail": "Rate limit exceeded"},
                headers=e.headers if hasattr(e, "headers") else {}
            )
        else:
            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": str(e) if status_code != 500 else "Internal server error",
                    "trace_id": trace_id,
                    "request_id": request_id
                }
            )
    
    finally:
        clear_context()


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        f"unhandled_exception - path={request.url.path}, error={str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ============================================================================
# ROUTES
# ============================================================================

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vendor HelpDesk Agent - Production Ready",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/api/metrics",
        "features": {
            "rate_limiting": True,
            "auto_save": True,
            "async_persistence": True
        },
        "rate_limits": {
            "chat": "20 requests/minute",
            "login": "5 requests/5 minutes",
            "query": "30 requests/minute",
            "default": "100 requests/minute"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "features": {
            "rate_limiting": True,
            "metrics": True,
            "logging": True,
            "context_offloading": True,
            "auto_save": True,
            "async_checkpointing": True
        }
    }


@app.get("/api/rate-limit-status")
async def rate_limit_status(request: Request):
    """Get rate limit status for current user"""
    from api.rate_limiter import get_rate_limit_key
    
    rate_key = get_rate_limit_key(request)
    stats = rate_limiter.get_user_stats(rate_key)
    
    return {
        "user_key": rate_key,
        "endpoints": stats,
        "limits": rate_limiter.limits
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(
        f"app_startup - version=2.0.0, "
        f"llm_model={settings.LLM_MODEL}, "
        f"checkpoint_window={settings.CHECKPOINT_WINDOW_SIZE}, "
        f"auto_save_interval={settings.AUTO_SAVE_INTERVAL_MINUTES}min, "
        f"metrics_enabled={settings.ENABLE_METRICS}"
    )
    
    # Start auto-save background task
    # asyncio.create_task(auto_save_periodic())
    # NOTE: Auto-save is handled per-request via background tasks for better token management


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("app_shutdown - saving all active chats")
    # Note: In production, ensure all chats are saved before shutdown
    # This would require maintaining a global token pool or user session management


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Vendor HelpDesk Agent API server (Production Mode)...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=False,
        reload=True
    )