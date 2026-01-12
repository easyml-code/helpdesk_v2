from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import generate_latest, REGISTRY
from typing import Optional
import time

# ============================================================================
# REQUEST METRICS
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# ============================================================================
# RATE LIMITING METRICS
# ============================================================================

rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['endpoint', 'user_id_hash']
)

rate_limit_window_requests = Histogram(
    'rate_limit_window_requests',
    'Requests per user in current window',
    ['endpoint'],
    buckets=(1, 5, 10, 20, 50, 100, 200)
)

rate_limit_blocks_total = Counter(
    'rate_limit_blocks_total',
    'Total blocked requests due to rate limiting',
    ['endpoint']
)

# ============================================================================
# LLM METRICS
# ============================================================================

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request latency',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'token_type']  # token_type: input, output
)

llm_cost_total = Counter(
    'llm_cost_usd_total',
    'Total estimated LLM cost in USD',
    ['model']
)

# ============================================================================
# DATABASE METRICS
# ============================================================================

db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['query_type', 'status']  # query_type: SELECT, INSERT, UPDATE, DELETE
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

db_rows_affected = Summary(
    'db_rows_affected',
    'Number of rows affected by database operations',
    ['query_type']
)

# ============================================================================
# CHAT METRICS
# ============================================================================

active_chats_gauge = Gauge(
    'active_chats_total',
    'Number of currently active chat sessions'
)

chat_messages_total = Counter(
    'chat_messages_total',
    'Total chat messages',
    ['role']  # user, assistant, tool
)

chat_tokens_total = Counter(
    'chat_tokens_total',
    'Total tokens in chat sessions',
    ['chat_id']
)

chat_duration_seconds = Histogram(
    'chat_session_duration_seconds',
    'Chat session duration',
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

# ============================================================================
# TOOL EXECUTION METRICS
# ============================================================================

tool_executions_total = Counter(
    'tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

tool_execution_duration_seconds = Histogram(
    'tool_execution_duration_seconds',
    'Tool execution latency',
    ['tool_name'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# ============================================================================
# CONTEXT OFFLOADING METRICS
# ============================================================================

context_offload_operations_total = Counter(
    'context_offload_operations_total',
    'Total context offload operations',
    ['operation', 'status']  # operation: store/retrieve, status: success/error
)

context_offload_chunks_total = Counter(
    'context_offload_chunks_total',
    'Total chunks created',
    ['session_type']
)

context_offload_rows_offloaded = Summary(
    'context_offload_rows_offloaded',
    'Number of rows offloaded per operation'
)

context_chunk_retrievals_total = Counter(
    'context_chunk_retrievals_total',
    'Total chunk retrievals',
    ['session_id']
)

# ============================================================================
# ERROR METRICS
# ============================================================================

errors_total = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'component']
)

auth_failures_total = Counter(
    'auth_failures_total',
    'Total authentication failures',
    ['reason']
)

# ============================================================================
# USER METRICS
# ============================================================================

requests_per_user = Counter(
    'requests_per_user_total',
    'Total requests per user',
    ['user_id_hash']  # hashed for privacy
)

user_sessions_active = Gauge(
    'user_sessions_active',
    'Number of active user sessions'
)

# ============================================================================
# CACHE METRICS
# ============================================================================

cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # operation: get/set/delete, result: hit/miss/error
)

cached_messages_gauge = Gauge(
    'cached_messages_total',
    'Number of messages currently in cache'
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class MetricsTimer:
    """Context manager for timing operations"""
    
    def __init__(self, histogram, labels: dict = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.labels:
            self.histogram.labels(**self.labels).observe(duration)
        else:
            self.histogram.observe(duration)


def track_http_request(method: str, endpoint: str, status: int, duration: float):
    """Track HTTP request metrics"""
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    # Track rate limit blocks
    if status == 429:
        rate_limit_blocks_total.labels(endpoint=endpoint).inc()


def track_llm_call(model: str, input_tokens: int, output_tokens: int, duration: float, success: bool = True):
    """Track LLM call metrics"""
    status = "success" if success else "error"
    llm_requests_total.labels(model=model, status=status).inc()
    llm_request_duration_seconds.labels(model=model).observe(duration)
    
    if success:
        llm_tokens_total.labels(model=model, token_type="input").inc(input_tokens)
        llm_tokens_total.labels(model=model, token_type="output").inc(output_tokens)
        
        # Estimate cost
        cost = estimate_llm_cost(model, input_tokens, output_tokens)
        llm_cost_total.labels(model=model).inc(cost)


def estimate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate LLM cost in USD"""
    rates = {
        "llama-3.3-70b-versatile": {"input": 0.00059 / 1000, "output": 0.00079 / 1000},
        "default": {"input": 0.0005 / 1000, "output": 0.0015 / 1000}
    }
    
    rate = rates.get(model, rates["default"])
    return (input_tokens * rate["input"]) + (output_tokens * rate["output"])


def track_db_query(query_type: str, duration: float, rows: int, success: bool = True):
    """Track database query metrics"""
    status = "success" if success else "error"
    db_queries_total.labels(query_type=query_type, status=status).inc()
    
    if success:
        db_query_duration_seconds.labels(query_type=query_type).observe(duration)
        db_rows_affected.labels(query_type=query_type).observe(rows)


def track_tool_execution(tool_name: str, duration: float, success: bool = True):
    """Track tool execution metrics"""
    status = "success" if success else "error"
    tool_executions_total.labels(tool_name=tool_name, status=status).inc()
    
    if success:
        tool_execution_duration_seconds.labels(tool_name=tool_name).observe(duration)


def track_context_offload(operation: str, rows: int, chunks: int, success: bool = True):
    """Track context offloading operations"""
    status = "success" if success else "error"
    context_offload_operations_total.labels(operation=operation, status=status).inc()
    
    if success and operation == "store":
        context_offload_chunks_total.labels(session_type="database_query").inc(chunks)
        context_offload_rows_offloaded.observe(rows)


def track_chunk_retrieval(session_id: str, chunk_count: int):
    """Track chunk retrieval operations"""
    context_chunk_retrievals_total.labels(session_id=session_id).inc(chunk_count)


def track_error(error_type: str, component: str):
    """Track error occurrence"""
    errors_total.labels(error_type=error_type, component=component).inc()


def get_metrics():
    """Get current metrics in Prometheus format"""
    return generate_latest(REGISTRY)