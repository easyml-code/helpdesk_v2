import logging
from logging.handlers import RotatingFileHandler
import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import contextvars

# Context variables for request tracking
trace_id_var = contextvars.ContextVar('trace_id', default=None)
request_id_var = contextvars.ContextVar('request_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class ProductionFormatter(logging.Formatter):
    """Production-level structured formatter with trace_id, request_id, user_id"""
    
    def format(self, record):
        # Get context variables
        trace_id = trace_id_var.get()
        request_id = request_id_var.get()
        user_id = user_id_var.get()
        
        # Create structured log entry
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "trace_id": trace_id,
            "request_id": request_id,
            "user_id_hash": self._hash_user_id(user_id) if user_id else None,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)
    
    @staticmethod
    def _hash_user_id(user_id: str) -> str:
        """Hash user_id for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context information"""
    
    def process(self, msg, kwargs):
        # Add context to extra
        extra = kwargs.get('extra', {})
        extra.update({
            'trace_id': trace_id_var.get(),
            'request_id': request_id_var.get(),
            'user_id': user_id_var.get(),
        })
        kwargs['extra'] = extra
        return msg, kwargs


# Setup handlers
json_formatter = ProductionFormatter()

# Main application log (JSON structured)
app_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=10
)
app_handler.setFormatter(json_formatter)
app_handler.setLevel(logging.INFO)

# Error log (separate file for errors only)
error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
error_handler.setFormatter(json_formatter)
error_handler.setLevel(logging.ERROR)

# Console handler for development (human-readable)
console_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.DEBUG)

# Create base logger
base_logger = logging.getLogger("app_logger")
base_logger.setLevel(logging.INFO)
base_logger.addHandler(app_handler)
base_logger.addHandler(error_handler)
base_logger.addHandler(console_handler)

# Create context-aware logger
logger = ContextLogger(base_logger, {})


# Context management functions
def set_trace_id(trace_id: Optional[str] = None):
    """Set trace ID for current context"""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id


def set_request_id(request_id: Optional[str] = None):
    """Set request ID for current context"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def set_user_id(user_id: str):
    """Set user ID for current context"""
    user_id_var.set(user_id)


def clear_context():
    """Clear all context variables"""
    trace_id_var.set(None)
    request_id_var.set(None)
    user_id_var.set(None)


def log_query(query: str, execution_time_ms: float, row_count: int):
    """Log database query with hashed content"""
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    logger.info(
        "db_query_executed",
        extra={
            'extra_data': {
                'query_hash': query_hash,
                'execution_time_ms': execution_time_ms,
                'row_count': row_count,
                'query_preview': query[:100]
            }
        }
    )


def log_llm_call(model: str, input_tokens: int, output_tokens: int, latency_ms: float):
    """Log LLM API call"""
    logger.info(
        "llm_call",
        extra={
            'extra_data': {
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'latency_ms': latency_ms
            }
        }
    )