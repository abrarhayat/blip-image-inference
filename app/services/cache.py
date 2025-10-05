import hashlib
import json
import logging

from redis import Redis

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self, client: Redis, ttl: int = 24 * 3600, prefix: str = "v1"):
        self.r = client
        self.ttl = ttl
        self.prefix = prefix

    # Compose a namespaced key for a single-image caption cache
    def img_key(self, sha: str) -> str:
        return f"{self.prefix}:img:{sha}"

    # Compose a namespaced key for a multi-image (collective) caption cache
    def collection_key(self, sha: str) -> str:
        return f"{self.prefix}:collection:{sha}"

    # Safe get that tolerates Redis outages (returns None)
    def get_json(self, key: str):
        try:
            val = self.r.get(key)
            return json.loads(val) if val else None
        except Exception:
            logger.warning("Unable to retrieve cache key: %s", key)
            return None

    # Safe set that tolerates Redis outages (best-effort cache)
    def set_json(self, key: str, value: dict):
        try:
            self.r.set(key, json.dumps(value), ex=self.ttl)
        except Exception:
            logger.warning("Unable to store cache key: %s", key)
            pass

    # Hash raw bytes deterministically (used for cache keys)
    @staticmethod
    def hash_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()
