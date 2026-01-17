"""
Monitoring and Observability Module

Tracks API performance metrics, query latency, and system health.
"""
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
from collections import defaultdict, deque
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates application metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_count = 0
        self.error_count = 0
        self.request_durations = deque(maxlen=max_history)
        self.embedding_durations = deque(maxlen=max_history)
        self.query_durations = deque(maxlen=max_history)
        self.endpoint_counts = defaultdict(int)
        self.error_types = defaultdict(int)
        self.lock = threading.Lock()
        
    def record_request(self, endpoint: str, duration: float, status: str = "success"):
        """Record a request metric"""
        with self.lock:
            self.request_count += 1
            self.request_durations.append(duration)
            self.endpoint_counts[endpoint] += 1
            
            if status == "error":
                self.error_count += 1
            
            logger.info(f"Request to {endpoint} completed in {duration:.3f}s - {status}")
    
    def record_embedding(self, duration: float, num_texts: int = 1):
        """Record embedding generation metric"""
        with self.lock:
            self.embedding_durations.append(duration)
            logger.info(f"Generated {num_texts} embeddings in {duration:.3f}s")
    
    def record_query(self, duration: float, num_results: int = 0):
        """Record vector query metric"""
        with self.lock:
            self.query_durations.append(duration)
            logger.info(f"Vector query returned {num_results} results in {duration:.3f}s")
    
    def record_error(self, error_type: str, endpoint: str):
        """Record an error"""
        with self.lock:
            self.error_count += 1
            self.error_types[error_type] += 1
            logger.error(f"Error in {endpoint}: {error_type}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        with self.lock:
            def safe_avg(values):
                return sum(values) / len(values) if values else 0
            
            def safe_percentile(values, percentile):
                if not values:
                    return 0
                sorted_vals = sorted(values)
                idx = int(len(sorted_vals) * percentile / 100)
                return sorted_vals[min(idx, len(sorted_vals) - 1)]
            
            return {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "avg_request_duration_ms": safe_avg(self.request_durations) * 1000,
                "p50_request_duration_ms": safe_percentile(self.request_durations, 50) * 1000,
                "p95_request_duration_ms": safe_percentile(self.request_durations, 95) * 1000,
                "p99_request_duration_ms": safe_percentile(self.request_durations, 99) * 1000,
                "avg_embedding_duration_ms": safe_avg(self.embedding_durations) * 1000,
                "avg_query_duration_ms": safe_avg(self.query_durations) * 1000,
                "endpoint_counts": dict(self.endpoint_counts),
                "error_types": dict(self.error_types),
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.request_count = 0
            self.error_count = 0
            self.request_durations.clear()
            self.embedding_durations.clear()
            self.query_durations.clear()
            self.endpoint_counts.clear()
            self.error_types.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_request(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics_collector.record_error(type(e).__name__, endpoint)
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_request(endpoint, duration, status)
        
        return wrapper
    return decorator


def track_embedding(func):
    """Decorator to track embedding generation"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Try to determine number of texts
        num_texts = 1
        if args and isinstance(args[0], list):
            num_texts = len(args[0])
        
        metrics_collector.record_embedding(duration, num_texts)
        return result
    
    return wrapper


def track_query(func):
    """Decorator to track vector queries"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Try to determine number of results
        num_results = 0
        if isinstance(result, dict) and "matches" in result:
            num_results = len(result["matches"])
        
        metrics_collector.record_query(duration, num_results)
        return result
    
    return wrapper


class RequestLogger:
    """Middleware for logging all requests"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            path = scope.get("path", "")
            method = scope.get("method", "")
            
            logger.info(f"→ {method} {path}")
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    status = message.get("status", 0)
                    logger.info(f"← {method} {path} - {status} - {duration:.3f}s")
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
