import os, logging

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

# Load .env from current directory or nearest parent automatically
# This makes uvicorn runs work regardless of --env-file usage or CWD
load_dotenv(find_dotenv(), override=False)

logger = logging.getLogger(__name__)

class Settings(BaseModel):
    # Environment flags
    ENV: str = Field(default=os.getenv("ENV", "production"))
    DEBUG: bool = Field(default=os.getenv("DEBUG", "false").lower() == "true")

    # Upload limits: enforce at proxy; keep here for business rules if needed
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))  # 10 MB

    # Redis connection
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # Admin protection
    API_KEY: str | None = os.getenv("API_KEY")

    # Model registry capacity (max models kept in memory)
    MODEL_CAPACITY: int = int(os.getenv("MODEL_CAPACITY", 1))

    # Cache TTL (seconds)
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 24 * 3600))


settings = Settings()

if settings.DEBUG:
    logger.warning("DEBUG mode is enabled")
