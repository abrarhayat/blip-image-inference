from fastapi import APIRouter, Depends
from app.deps import get_redis, require_api_key

# All routes here require X-API-Key by default (via dependencies=...)
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_api_key)])

@router.post("/reset-cache")
def reset_redis_cache(rdb = Depends(get_redis)):
    # Flush DB: destructive — keep protected
    rdb.flushdb()
    return {"message": "Redis cache cleared"}