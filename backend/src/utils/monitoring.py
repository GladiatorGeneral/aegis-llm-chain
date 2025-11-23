"""Monitoring utilities for performance and health tracking."""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_average_latency(self, operation: str) -> Optional[float]:
        """Get average latency for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        latencies = [m["latency_ms"] for m in self.metrics[operation]]
        return sum(latencies) / len(latencies)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for operation, metrics in self.metrics.items():
            if metrics:
                latencies = [m["latency_ms"] for m in metrics]
                summary[operation] = {
                    "count": len(metrics),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies)
                }
        return summary

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.time() - start_time) * 1000
                performance_monitor.record_latency(operation_name, latency_ms)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.time() - start_time) * 1000
                performance_monitor.record_latency(operation_name, latency_ms)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
