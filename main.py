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


app = FastAPI(
    title="Vendor HelpDesk Agent",
    description="Production-Ready Backend API with Rate Limiting, Metrics & Logging",
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
# REQUEST TRACKING + RATE LIMITING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def request_tracking_and_rate_limiting_middleware(request: Request, call_next):
    """
    Global middleware for:
    1. Request tracking (trace_id, request_id, user_id)
    2. Rate limiting
    3. Metrics collection
    4. Error handling
    """
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
                # Store in request state for rate limiter
                request.state.user_id = user_id
    except:
        pass  # User not authenticated
    
    # Log request start
    logger.info(
        f"request_start - method={request.method}, "
        f"path={request.url.path}, trace_id={trace_id}, request_id={request_id}"
    )
    
    try:
        # ============================================================
        # RATE LIMITING CHECK
        # ============================================================
        # Skip rate limiting for health/metrics endpoints
        skip_rate_limit = request.url.path in ["/health", "/api/metrics", "/docs", "/openapi.json"]
        
        if not skip_rate_limit:
            try:
                rate_limit_info = await check_rate_limit(request)
                logger.info(
                    f"rate_limit_check - path={request.url.path}, "
                    f"remaining={rate_limit_info['remaining']}, "
                    f"limit={rate_limit_info['limit']}"
                )
            except Exception as rate_limit_error:
                # Rate limit exceeded - return 429
                duration = time.time() - start_time
                track_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=429,
                    duration=duration
                )
                
                # Re-raise to let FastAPI handle it
                raise rate_limit_error
        
        # ============================================================
        # PROCESS REQUEST
        # ============================================================
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Track metrics
        track_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        # Add rate limit headers to response
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        # Log request completion
        logger.info(
            f"request_complete - method={request.method}, "
            f"path={request.url.path}, status={response.status_code}, "
            f"duration_ms={duration * 1000:.2f}, "
            f"trace_id={trace_id}, request_id={request_id}"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Determine status code
        status_code = 500
        if hasattr(e, "status_code"):
            status_code = e.status_code
        
        # Track error
        track_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=status_code,
            duration=duration
        )
        
        # Log error
        logger.error(
            f"request_error - method={request.method}, "
            f"path={request.url.path}, error={str(e)}, "
            f"duration_ms={duration * 1000:.2f}, "
            f"trace_id={trace_id}, request_id={request_id}",
            exc_info=True
        )
        
        # Return appropriate error response
        if status_code == 429:
            # Rate limit error - pass through with headers
            return JSONResponse(
                status_code=429,
                content=e.detail if hasattr(e, "detail") else {"detail": "Rate limit exceeded"},
                headers=e.headers if hasattr(e, "headers") else {}
            )
        else:
            # Other errors
            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": str(e) if status_code != 500 else "Internal server error",
                    "trace_id": trace_id,
                    "request_id": request_id
                }
            )
    
    finally:
        # Clear context
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
        "message": "Vendor HelpDesk Agent - Production Ready with Rate Limiting",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/api/metrics",
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
            "context_offloading": True
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
        f"llm_model={settings.LLM_MODEl}, "
        f"checkpoint_window={settings.CHECKPOINT_WINDOW_SIZE}, "
        f"metrics_enabled={settings.ENABLE_METRICS}, "
        f"rate_limiting=enabled"
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("app_shutdown")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Vendor HelpDesk Agent API server (Production Mode with Rate Limiting)...")
    
    # Production configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=False,
        reload=True
    )