"""
Caching utilities using diskcache
"""
from diskcache import Cache
from typing import Any, Optional
from functools import wraps
import hashlib
from config import settings

# Initialize cache
cache = Cache(str(settings.CACHE_DIR))

def get_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    key_parts = [prefix]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()

def cached(prefix: str, ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip 'self' argument for instance methods — object memory address
            # changes across restarts, making persistent disk cache keys invalid
            cache_args = args
            if args and hasattr(args[0], '__class__') and not isinstance(args[0], (str, int, float, bool, bytes, list, tuple, dict)):
                cache_args = args[1:]
            cache_key = get_cache_key(prefix, *cache_args, **kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, expire=ttl)
            return result
        return wrapper
    return decorator

def clear_cache(prefix: Optional[str] = None):
    """Clear cache entries"""
    if prefix:
        # Clear specific prefix
        for key in list(cache.iterkeys()):
            if key.startswith(prefix):
                del cache[key]
    else:
        # Clear all
        cache.clear()