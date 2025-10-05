import logging
from io import BytesIO
from typing import List

from PIL import Image
from transformers import InternVLForConditionalGeneration, Gemma3ForConditionalGeneration

from app.deps import get_redis
from app.schemas import CaptionQuery, CaptionResponse, CollectiveResponse
from app.services.cache import Cache
from app.services.model_registry import registry
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from app.inference.captioning import infer_image_caption, infer_collective_caption
from app.inference.flagging import is_flagged
from app.inference.tagging import generate_spacy_tags

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["caption"])

# Limit to common image MIME types; reject others early with 415
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/caption-images", response_model=CaptionResponse)
async def caption_images(
        images: List[UploadFile] = File(...),
        query: CaptionQuery = Depends(),
        rdb=Depends(get_redis),
):
    # Resolve the requested model (thread-safe, memory-bounded)
    processor, model, device = registry.get(query.model)
    cache = Cache(rdb)

    results = []
    for f in images:
        if f.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {f.content_type}")

        # Read the upload once; use bytes for hashing and decoding
        file_bytes = await f.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        # Cache key is v1:img:{sha256(bytes)}
        file_hash = cache.hash_bytes(file_bytes)
        key = cache.img_key(file_hash)

        # Fast path: return cached caption/tags
        cached = cache.get_json(key)
        if cached:
            results.append({"filename": f.filename, **cached, "cache": True})
            continue

        # Decode image from bytes (convert to RGB for consistency)
        image = Image.open(BytesIO(file_bytes)).convert("RGB")

        # Run captioning (prompt optional); keep inference code minimal in route
        caption = infer_image_caption(processor, model, device, image, query.caption_prompt)
        if isinstance(model, InternVLForConditionalGeneration) or isinstance(model, Gemma3ForConditionalGeneration):
            flagged = is_flagged(processor, model, device, [image], query.flag_caption_prompt)
        else:
            flagged = None
        # Fallback message if model returns nothing
        if not caption:
            caption = "No caption could be generated."
            item = {"caption": caption, "tags": [], "flagged": bool(flagged)}
        else:
            item = {"caption": caption, "tags": generate_spacy_tags(caption), "flagged": bool(flagged)}

        # Best-effort cache write (non-fatal on Redis outage)
        cache.set_json(key, item)
        results.append({"filename": f.filename, **item, "cache": False})

    return {"results": results}


@router.post("/caption-collective-images", response_model=CollectiveResponse)
async def caption_collective_images(
        images: List[UploadFile] = File(...),
        query: CaptionQuery = Depends(),
        rdb=Depends(get_redis),
):
    # Collective captioning is only supported on certain models
    if query.model not in {"gemma", "intern_vlm"}:
        raise HTTPException(status_code=400, detail="Collective captioning only supported for Gemma / InternVLM models")

    processor, model, device = registry.get(query.model)
    cache = Cache(rdb)

    # Prepare images and their hashes
    file_hashes = []
    pil_images = []
    for f in images:
        if f.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {f.content_type}")
        file_bytes = await f.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        file_hashes.append(cache.hash_bytes(file_bytes))
        pil_images.append(Image.open(BytesIO(file_bytes)).convert("RGB"))

    # Combined hash for the collection; order matters (keep client order)
    combined_hash = cache.hash_bytes("".join(file_hashes).encode("utf-8"))
    key = cache.collection_key(combined_hash)

    cached = cache.get_json(key)
    if cached:
        return cached

    # Generate a single caption for the whole set
    collective_caption = infer_collective_caption(
        processor, model, device, pil_images, query.caption_prompt, max_new_tokens=200
    )
    flagged = is_flagged(processor, model, device, pil_images, query.flag_caption_prompt, max_new_tokens=200)

    response = {
        "collective_caption": collective_caption or "No caption could be generated.",
        "count": len(pil_images),
        "tags": generate_spacy_tags(collective_caption) if collective_caption else [],
        "flagged": bool(flagged),
    }

    cache.set_json(key, response)
    return response
