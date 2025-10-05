from typing import List, Optional

from pydantic import BaseModel, Field


class CaptionQuery(BaseModel):
    # Model selection per request
    model: str = Field("blip", pattern=r"^(blip|blip2|gemma|intern_vlm)$")
    # Optional per-request prompts (fallback to sensible defaults server-side)
    caption_prompt: Optional[str] = None
    flag_caption_prompt: Optional[str] = None


class CaptionItem(BaseModel):
    filename: str
    caption: str
    tags: List[str] = []
    flagged: bool | None = None
    cache: bool = False


class CaptionResponse(BaseModel):
    results: List[CaptionItem]


class CollectiveResponse(BaseModel):
    collective_caption: str
    count: int
    tags: List[str] = []
    flagged: bool | None = None
