"""
FrankenAI v12.0 - Monitoring
Lightweight metrics and health tracking
"""

import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque, defaultdict


class MetricsCollector:
    """Thread-safe metrics collection compatible with v11 request_counter"""
    
    def __init__(self, cloudwatch_enabled: bool = False):
        self._lock = threading.RLock()
        self.cloudwatch_enabled = cloudwatch_enabled
        
        # Metrics compatible with v11 request_counter structure
        self.metrics = {
            'count': 0,
            'cache_hits': 0,
            'pass_cache_hits': 0,
            'groq_pass1_only': 0,
            'groq_pass2_exits': 0,
            'groq_pass3_exits': 0,
            'groq_full_4pass': 0,
            'claude_passes': 0,
            'claude_skipped': 0,
            'claude_finisher_passes': 0,
            'claude_finisher_skipped': 0,
            'grok_passes': 0,
            'grok_reverted': 0,
            'frankenai_synthesis': 0,
            'quality_gate_triggers': 0,
            'word_count_adjustments': 0,
            'code_analysis_queries': 0,
            'code_lines_analyzed': 0,
            'estimated_cost': 0.0
        }
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.errors = deque(maxlen=100)
    
    def __getitem__(self, key: str):
        """Dict-like access for compatibility"""
        with self._lock:
            return self.metrics.get(key, 0)
    
    def __setitem__(self, key: str, value):
        """Dict-like assignment for compatibility"""
        with self._lock:
            self.metrics[key] = value
    
    def get(self, key: str, default=None):
        """Dict-like get for compatibility"""
        with self._lock:
            return self.metrics.get(key, default)
    
    def record_request_time(self, duration: float):
        """Record request timing"""
        with self._lock:
            self.request_times.append({
                'duration': duration,
                'timestamp': time.time()
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            total_groq = (self.metrics['groq_pass1_only'] + 
                         self.metrics['groq_pass2_exits'] + 
                         self.metrics['groq_pass3_exits'] + 
                         self.metrics['groq_full_4pass'])
            
            if self.request_times:
                avg_time = sum(r['duration'] for r in self.request_times) / len(self.request_times)
            else:
                avg_time = 0
            
            return {
                'total_requests': self.metrics['count'],
                'total_groq': total_groq,
                'avg_request_time': f"{avg_time:.2f}s",
                'estimated_cost': f"${self.metrics['estimated_cost']:.2f}",
                **self.metrics
            }


class HealthCheck:
    """Simple health checking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_request = time.time()
    
    def ping(self):
        """Record activity"""
        self.last_request = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get health status"""
        uptime = time.time() - self.start_time
        return {
            'status': 'healthy',
            'uptime_seconds': int(uptime),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instances
metrics: Optional[MetricsCollector] = None
health: Optional[HealthCheck] = None


def init_monitoring(cloudwatch_enabled: bool = False):
    """Initialize monitoring"""
    global metrics, health
    metrics = MetricsCollector(cloudwatch_enabled)
    health = HealthCheck()
    return metrics, health
