import logging

import redis
from fastapi import Header, HTTPException, status

from .settings import settings

# Create a connection pool singleton (shared across requests)
_redis_pool = None

logger = logging.getLogger(__name__)


def get_redis_pool():
    """Get or create the Redis connection pool (singleton pattern)."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            max_connections=50,  # Adjust based on your concurrency needs
        )
    return _redis_pool


# Provide a StrictRedis client using the shared pool
def get_redis() -> redis.StrictRedis:
    """Get a Redis client from the connection pool."""
    pool = get_redis_pool()
    return redis.StrictRedis(connection_pool=pool)


# Simple API key guard for admin endpoints (header: X-API-Key)
def require_api_key(x_api_key: str | None = Header(default=None)):
    # Check if we're in dev mode
    if settings.DEBUG:
        logger.warning('API key guard is disabled in dev mode')
        return
    else:
        logger.info('Checking API key')
    expected = settings.API_KEY
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
