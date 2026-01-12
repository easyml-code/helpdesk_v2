from fastapi import Request, HTTPException
from typing import Dict, Tuple
from datetime import datetime, timedelta
import time
from collections import defaultdict
from logs.log import logger
from metrics.prometheus import Counter, Histogram
import hashlib
from metrics.prometheus import (
    rate_limit_exceeded_total,
    rate_limit_window_requests
)


class InMemoryRateLimiter:
    """
    In-memory rate limiter with sliding window.
    Tracks requests per user per endpoint.
    """
    
    def __init__(self):
        # Storage: {user_id: {endpoint: [(timestamp, count)]}}
        self.requests: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        
        # Rate limit configurations
        self.limits = {
            "default": (100, 60),        # 100 requests per 60 seconds
            "/api/chat": (20, 60),       # 20 chat requests per minute
            "/api/auth/login": (5, 300), # 5 login attempts per 5 minutes
            "/api/query": (30, 60),      # 30 queries per minute
        }
        
        logger.info("InMemoryRateLimiter initialized")
    
    def _cleanup_old_requests(self, user_id: str, endpoint: str, window_seconds: int):
        """Remove requests older than window"""
        if user_id not in self.requests or endpoint not in self.requests[user_id]:
            return
        
        cutoff_time = time.time() - window_seconds
        self.requests[user_id][endpoint] = [
            (ts, count) for ts, count in self.requests[user_id][endpoint]
            if ts > cutoff_time
        ]
    
    def _get_request_count(self, user_id: str, endpoint: str) -> int:
        """Get total request count in current window"""
        if user_id not in self.requests or endpoint not in self.requests[user_id]:
            return 0
        
        return sum(count for _, count in self.requests[user_id][endpoint])
    
    def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str,
        increment: bool = True
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limit.
        
        Args:
            user_id: User identifier (hashed)
            endpoint: API endpoint
            increment: Whether to increment counter
            
        Returns:
            (allowed, info_dict)
        """
        # Get limit config for endpoint
        limit_count, window_seconds = self.limits.get(
            endpoint, 
            self.limits["default"]
        )
        
        # Cleanup old requests
        self._cleanup_old_requests(user_id, endpoint, window_seconds)
        
        # Get current count
        current_count = self._get_request_count(user_id, endpoint)
        
        # Track in Prometheus
        rate_limit_window_requests.labels(endpoint=endpoint).observe(current_count)
        
        # Check limit
        if current_count >= limit_count:
            user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            rate_limit_exceeded_total.labels(
                endpoint=endpoint, 
                user_id_hash=user_id_hash
            ).inc()
            
            logger.warning(
                f"rate_limit_exceeded - user_id_hash={user_id_hash}, "
                f"endpoint={endpoint}, count={current_count}, limit={limit_count}"
            )
            
            # Calculate retry_after
            if self.requests[user_id][endpoint]:
                oldest_ts = self.requests[user_id][endpoint][0][0]
                retry_after = int(window_seconds - (time.time() - oldest_ts)) + 1
            else:
                retry_after = window_seconds
            
            return False, {
                "allowed": False,
                "limit": limit_count,
                "remaining": 0,
                "reset": int(time.time() + retry_after),
                "retry_after": retry_after
            }
        
        # Increment counter if allowed
        if increment:
            self.requests[user_id][endpoint].append((time.time(), 1))
        
        remaining = limit_count - current_count - (1 if increment else 0)
        
        return True, {
            "allowed": True,
            "limit": limit_count,
            "remaining": remaining,
            "reset": int(time.time() + window_seconds),
            "retry_after": 0
        }
    
    def get_user_stats(self, user_id: str) -> Dict[str, any]:
        """Get rate limit stats for a user"""
        if user_id not in self.requests:
            return {}
        
        stats = {}
        for endpoint, requests in self.requests[user_id].items():
            limit_count, window_seconds = self.limits.get(
                endpoint, 
                self.limits["default"]
            )
            
            self._cleanup_old_requests(user_id, endpoint, window_seconds)
            current_count = self._get_request_count(user_id, endpoint)
            
            stats[endpoint] = {
                "count": current_count,
                "limit": limit_count,
                "window_seconds": window_seconds
            }
        
        return stats


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


def get_rate_limit_key(request: Request) -> str:
    """
    Extract rate limit key from request.
    Uses user_id from JWT if available, otherwise IP address.
    """
    # Try to get user_id from state (set by auth middleware)
    user_id = getattr(request.state, "user_id", None)
    
    if user_id:
        return f"user:{user_id}"
    
    # Fallback to IP address
    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    
    return f"ip:{client_ip}"


async def check_rate_limit(request: Request, endpoint: str = None):
    """
    Middleware function to check rate limit.
    Raises HTTPException if limit exceeded.
    """
    # Determine endpoint
    if endpoint is None:
        endpoint = request.url.path
    
    # Get rate limit key
    rate_key = get_rate_limit_key(request)
    
    # Check limit
    allowed, info = rate_limiter.check_rate_limit(rate_key, endpoint)
    
    # Set headers
    request.state.rate_limit_info = info
    
    if not allowed:
        logger.warning(
            f"rate_limit_blocked - key={rate_key}, endpoint={endpoint}, "
            f"retry_after={info['retry_after']}"
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": info["limit"],
                "retry_after": info["retry_after"],
                "reset": info["reset"]
            },
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info["retry_after"])
            }
        )
    
    return info