"""
FrankenAI v12.0 - Storage Layer
Abstraction for cache and session storage with multiple backends
"""

import json
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class StorageBackend(ABC):
    """Abstract storage backend interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass


class FileStorageBackend(StorageBackend):
    """Thread-safe file-based storage with atomic writes"""
    
    def __init__(self, base_dir: str = "cache"):
        self.base_dir = base_dir
        self._lock = threading.RLock()
        os.makedirs(base_dir, exist_ok=True)
        
    def _get_path(self, key: str) -> str:
        """Get file path for key"""
        return os.path.join(self.base_dir, f"{key}.json")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve from file"""
        path = self._get_path(key)
        
        with self._lock:
            if not os.path.exists(path):
                return None
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check TTL if present
                if 'ttl' in data and 'timestamp' in data:
                    if time.time() - data['timestamp'] > data['ttl']:
                        self.delete(key)
                        return None
                
                return data.get('value')
            except (json.JSONDecodeError, IOError):
                return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store with atomic write"""
        path = self._get_path(key)
        
        data = {
            'value': value,
            'timestamp': time.time()
        }
        if ttl:
            data['ttl'] = ttl
        
        with self._lock:
            try:
                # Atomic write using tempfile
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=self.base_dir,
                    prefix='.tmp_',
                    suffix='.json'
                )
                
                try:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Atomic replace
                    os.replace(temp_path, path)
                    return True
                    
                except Exception:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
                    
            except Exception as e:
                print(f"Storage error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete file"""
        path = self._get_path(key)
        
        with self._lock:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                return True
            except OSError:
                return False
    
    def exists(self, key: str) -> bool:
        """Check if file exists"""
        return os.path.exists(self._get_path(key))
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys"""
        with self._lock:
            try:
                files = os.listdir(self.base_dir)
                keys = [f.replace('.json', '') for f in files if f.endswith('.json')]
                if prefix:
                    keys = [k for k in keys if k.startswith(prefix)]
                return keys
            except OSError:
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._lock:
            keys = self.list_keys()
            total_size = 0
            
            for key in keys:
                path = self._get_path(key)
                if os.path.exists(path):
                    total_size += os.path.getsize(path)
            
            return {
                'backend': 'file',
                'base_dir': self.base_dir,
                'total_keys': len(keys),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / 1024 / 1024, 2)
            }


class S3StorageBackend(StorageBackend):
    """S3-based storage backend (optional, requires boto3)"""
    
    def __init__(self, bucket: str, prefix: str = "frankenai/cache/"):
        try:
            import boto3
            self.s3 = boto3.client('s3')
            self.bucket = bucket
            self.prefix = prefix
            self._lock = threading.RLock()
        except ImportError:
            raise ImportError("boto3 required for S3 backend: pip install boto3")
    
    def _get_key(self, key: str) -> str:
        """Get S3 key with prefix"""
        return f"{self.prefix}{key}.json"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve from S3"""
        s3_key = self._get_key(key)
        
        with self._lock:
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
                data = json.loads(response['Body'].read().decode('utf-8'))
                
                # Check TTL
                if 'ttl' in data and 'timestamp' in data:
                    if time.time() - data['timestamp'] > data['ttl']:
                        self.delete(key)
                        return None
                
                return data.get('value')
            except self.s3.exceptions.NoSuchKey:
                return None
            except Exception as e:
                print(f"S3 get error: {e}")
                return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store in S3"""
        s3_key = self._get_key(key)
        
        data = {
            'value': value,
            'timestamp': time.time()
        }
        if ttl:
            data['ttl'] = ttl
        
        with self._lock:
            try:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=json.dumps(data, ensure_ascii=False),
                    ContentType='application/json'
                )
                return True
            except Exception as e:
                print(f"S3 put error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete from S3"""
        s3_key = self._get_key(key)
        
        with self._lock:
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
                return True
            except Exception as e:
                print(f"S3 delete error: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in S3"""
        s3_key = self._get_key(key)
        
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys from S3"""
        with self._lock:
            try:
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=f"{self.prefix}{prefix}"
                )
                
                keys = []
                for obj in response.get('Contents', []):
                    key = obj['Key'].replace(self.prefix, '').replace('.json', '')
                    keys.append(key)
                
                return keys
            except Exception as e:
                print(f"S3 list error: {e}")
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics"""
        keys = self.list_keys()
        return {
            'backend': 's3',
            'bucket': self.bucket,
            'prefix': self.prefix,
            'total_keys': len(keys)
        }


class ConversationHistory:
    """Thread-safe conversation history with TTL and size limits"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._history: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = min(ttl_seconds // 4, 300)
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def add(self, key: str, value: Any) -> None:
        """Add or update entry"""
        current_time = time.time()
        
        with self._lock:
            # Update existing (move to end for LRU)
            if key in self._history:
                del self._history[key]
            
            # Add new entry
            self._history[key] = {
                'value': value,
                'timestamp': current_time,
                'access_count': 1
            }
            
            # Enforce size limit
            while len(self._history) > self.max_size:
                self._history.popitem(last=False)
    
    def get(self, key: str) -> Optional[Any]:
        """Get entry, None if expired or missing"""
        current_time = time.time()
        
        with self._lock:
            if key not in self._history:
                return None
            
            entry = self._history[key]
            
            # Check expiry
            if current_time - entry['timestamp'] > self.ttl_seconds:
                del self._history[key]
                return None
            
            # Update access info
            entry['access_count'] += 1
            entry['last_access'] = current_time
            self._history.move_to_end(key)
            
            return entry['value']
    
    def _cleanup_expired(self, current_time: Optional[float] = None) -> int:
        """Remove expired entries"""
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.ttl_seconds
        
        expired_keys = [
            key for key, entry in self._history.items()
            if entry['timestamp'] < cutoff_time
        ]
        
        for key in expired_keys:
            del self._history[key]
        
        self._last_cleanup = current_time
        return len(expired_keys)
    
    def _periodic_cleanup(self):
        """Background cleanup thread"""
        while True:
            time.sleep(self._cleanup_interval)
            with self._lock:
                removed = self._cleanup_expired()
                if removed > 0:
                    print(f"  -> Cleaned up {removed} expired conversations")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for entry in self._history.values()
                if current_time - entry['timestamp'] > self.ttl_seconds
            )
            
            return {
                'total_entries': len(self._history),
                'expired_entries': expired_count,
                'active_entries': len(self._history) - expired_count,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }
    
    def clear(self) -> None:
        """Clear all history"""
        with self._lock:
            self._history.clear()


def create_storage_backend(backend_type: str = "file", **kwargs) -> StorageBackend:
    """Factory function to create storage backend"""
    backends = {
        'file': FileStorageBackend,
        's3': S3StorageBackend
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type](**kwargs)
