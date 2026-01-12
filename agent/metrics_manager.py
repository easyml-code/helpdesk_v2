from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from logs.log import logger
from database.client import run_query
from database.utils import pg_escape
import asyncio


class MetricsManager:
    """
    Manages metrics with in-memory caching and periodic database push.
    Similar to chat message caching pattern.
    """
    
    def __init__(self, auto_push_interval_minutes: int = 5):
        # In-memory metrics cache: {chat_id: metrics_data}
        self.cache: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'llm_calls': [],
            'tool_executions': [],
            'token_usage': {'total_input': 0, 'total_output': 0, 'by_turn': []},
            'response_times': [],
            'errors': [],
            'last_updated': datetime.utcnow(),
            'last_push': datetime.utcnow()
        })
        
        # Unsaved metrics per chat
        self.unsaved_metrics: Dict[str, List[Dict]] = defaultdict(list)
        
        self.auto_push_interval_minutes = auto_push_interval_minutes
        
        logger.info(f"MetricsManager initialized - auto_push_interval={auto_push_interval_minutes}min")
    
    def track_llm_call(
        self,
        chat_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True
    ):
        """Track LLM API call metrics"""
        metric = {
            'type': 'llm_call',
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'latency_ms': latency_ms,
            'success': success,
            'timestamp': datetime.utcnow()
        }
        
        self.cache[chat_id]['llm_calls'].append(metric)
        self.cache[chat_id]['token_usage']['total_input'] += input_tokens
        self.cache[chat_id]['token_usage']['total_output'] += output_tokens
        self.cache[chat_id]['token_usage']['by_turn'].append({
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.cache[chat_id]['last_updated'] = datetime.utcnow()
        
        self.unsaved_metrics[chat_id].append(metric)
        
        logger.info(
            f"llm_call_tracked - chat_id={chat_id}, tokens={input_tokens + output_tokens}, "
            f"latency={latency_ms:.2f}ms"
        )
    
    def track_tool_execution(
        self,
        chat_id: str,
        tool_name: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Track tool execution metrics"""
        metric = {
            'type': 'tool_execution',
            'tool_name': tool_name,
            'duration_ms': duration_ms,
            'success': success,
            'error': error,
            'timestamp': datetime.utcnow()
        }
        
        self.cache[chat_id]['tool_executions'].append(metric)
        self.cache[chat_id]['last_updated'] = datetime.utcnow()
        
        self.unsaved_metrics[chat_id].append(metric)
        
        logger.info(
            f"tool_execution_tracked - chat_id={chat_id}, tool={tool_name}, "
            f"duration={duration_ms:.2f}ms, success={success}"
        )
    
    def track_error(
        self,
        chat_id: str,
        error_type: str,
        error_message: str,
        component: str
    ):
        """Track error occurrence"""
        metric = {
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'component': component,
            'timestamp': datetime.utcnow()
        }
        
        self.cache[chat_id]['errors'].append(metric)
        self.cache[chat_id]['last_updated'] = datetime.utcnow()
        
        self.unsaved_metrics[chat_id].append(metric)
        
        logger.warning(
            f"error_tracked - chat_id={chat_id}, type={error_type}, component={component}"
        )
    
    def get_chat_metrics(self, chat_id: str) -> Dict[str, Any]:
        """Get current metrics for a chat (from cache)"""
        if chat_id not in self.cache:
            return {}
        
        metrics = self.cache[chat_id]
        
        return {
            'chat_id': chat_id,
            'total_llm_calls': len(metrics['llm_calls']),
            'total_tool_executions': len(metrics['tool_executions']),
            'total_errors': len(metrics['errors']),
            'token_usage': {
                'total_input': metrics['token_usage']['total_input'],
                'total_output': metrics['token_usage']['total_output'],
                'total': metrics['token_usage']['total_input'] + metrics['token_usage']['total_output']
            },
            'avg_response_time_ms': (
                sum(m['latency_ms'] for m in metrics['llm_calls']) / len(metrics['llm_calls'])
                if metrics['llm_calls'] else 0
            ),
            'last_updated': metrics['last_updated'].isoformat(),
            'unsaved_count': len(self.unsaved_metrics.get(chat_id, []))
        }
    
    async def push_metrics_to_db(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str,
        force: bool = False
    ) -> int:
        """Push unsaved metrics to database"""
        if chat_id not in self.cache:
            return 0
        
        unsaved = self.unsaved_metrics.get(chat_id, [])
        if not unsaved:
            return 0
        
        # Check if push is needed
        cache_data = self.cache[chat_id]
        time_since_push = datetime.utcnow() - cache_data['last_push']
        
        if not force and time_since_push.total_seconds() < (self.auto_push_interval_minutes * 60):
            logger.debug(f"skipping_metrics_push - chat_id={chat_id}, time_since_push={time_since_push}")
            return 0
        
        try:
            # Batch insert metrics
            values = []
            for metric in unsaved:
                metric_type = metric['type']
                metric_data = {k: v for k, v in metric.items() if k not in ['type', 'timestamp']}
                
                values.append(
                    f"('{chat_id}', '{metric_type}', '{pg_escape(str(metric_data))}', "
                    f"'{metric['timestamp'].isoformat()}')"
                )
            
            if not values:
                return 0
            
            insert_query = f"""
            INSERT INTO chat_metrics_detailed (chat_id, metric_type, metric_data, created_at)
            VALUES {', '.join(values)};
            """
            
            await run_query(insert_query, access_token, refresh_token)
            
            # Clear unsaved metrics
            pushed_count = len(unsaved)
            self.unsaved_metrics[chat_id] = []
            cache_data['last_push'] = datetime.utcnow()
            
            logger.info(
                f"metrics_pushed - chat_id={chat_id}, count={pushed_count}"
            )
            
            return pushed_count
            
        except Exception as e:
            logger.error(f"metrics_push_error - chat_id={chat_id}, error={e}", exc_info=True)
            raise
    
    async def load_metrics_from_db(
        self,
        chat_id: str,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Load metrics from database"""
        try:
            query = f"""
            SELECT metric_type, metric_data, created_at
            FROM chat_metrics_detailed
            WHERE chat_id = '{pg_escape(chat_id)}'
            ORDER BY created_at ASC;
            """
            
            results = await run_query(query, access_token, refresh_token)
            
            # Reconstruct cache from DB
            metrics_cache = {
                'llm_calls': [],
                'tool_executions': [],
                'token_usage': {'total_input': 0, 'total_output': 0, 'by_turn': []},
                'response_times': [],
                'errors': [],
                'last_updated': datetime.utcnow(),
                'last_push': datetime.utcnow()
            }
            
            for row in results:
                metric_type = row['metric_type']
                metric_data = eval(row['metric_data'])  # Safe since we control the data
                metric_data['timestamp'] = row['created_at']
                metric_data['type'] = metric_type
                
                if metric_type == 'llm_call':
                    metrics_cache['llm_calls'].append(metric_data)
                    metrics_cache['token_usage']['total_input'] += metric_data.get('input_tokens', 0)
                    metrics_cache['token_usage']['total_output'] += metric_data.get('output_tokens', 0)
                elif metric_type == 'tool_execution':
                    metrics_cache['tool_executions'].append(metric_data)
                elif metric_type == 'error':
                    metrics_cache['errors'].append(metric_data)
            
            self.cache[chat_id] = metrics_cache
            
            logger.info(
                f"metrics_loaded - chat_id={chat_id}, llm_calls={len(metrics_cache['llm_calls'])}, "
                f"tool_executions={len(metrics_cache['tool_executions'])}"
            )
            
            return self.get_chat_metrics(chat_id)
            
        except Exception as e:
            logger.error(f"metrics_load_error - chat_id={chat_id}, error={e}", exc_info=True)
            return {}
    
    async def get_user_aggregate_metrics(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Get aggregate metrics for a user across all chats"""
        try:
            query = f"""
            SELECT 
                COUNT(DISTINCT c.chat_id) as total_chats,
                SUM(c.total_tokens) as total_tokens,
                COUNT(m.message_id) as total_messages,
                COUNT(DISTINCT CASE WHEN m.role = 'user' THEN m.message_id END) as total_user_messages,
                AVG(cm.session_duration_seconds) as avg_session_duration,
                MAX(c.updated_at) as last_activity
            FROM chats c
            LEFT JOIN messages m ON c.chat_id = m.chat_id
            LEFT JOIN chat_metrics cm ON c.chat_id = cm.chat_id
            WHERE c.user_id = '{pg_escape(user_id)}'
            GROUP BY c.user_id;
            """
            
            results = await run_query(query, access_token, refresh_token)
            
            if results:
                return results[0]
            return {}
            
        except Exception as e:
            logger.error(f"user_metrics_error - error={e}", exc_info=True)
            return {}
    
    def clear_cache(self, chat_id: str):
        """Clear metrics cache for a chat"""
        if chat_id in self.cache:
            del self.cache[chat_id]
        if chat_id in self.unsaved_metrics:
            del self.unsaved_metrics[chat_id]
        logger.info(f"metrics_cache_cleared - chat_id={chat_id}")


# Global instance
metrics_manager = MetricsManager(auto_push_interval_minutes=5)