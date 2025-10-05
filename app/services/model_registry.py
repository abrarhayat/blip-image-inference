import gc
import logging
import os
import threading
from collections import OrderedDict
from typing import Tuple, Union, TYPE_CHECKING

import torch

# Only import these for type checking to avoid runtime errors
if TYPE_CHECKING:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        Blip2Processor, Blip2ForConditionalGeneration,
        Gemma3Processor, Gemma3ForConditionalGeneration,
        InternVLProcessor, InternVLForConditionalGeneration
    )

from app.models.blip import initialize_blip_model, initialize_blip2_model
from app.models.gemma import initialize_gemma_model
from app.models.intern_vlm import initialize_intern_vlm_model

# Allowed model keys (validate early for clearer errors)
ALLOWED_MODELS = {"blip", "blip2", "gemma", "intern_vlm"}

# Type aliases for clarity (use Any at runtime since AutoProcessor returns different types)
if TYPE_CHECKING:
    ProcessorType = Union[BlipProcessor, Blip2Processor, Gemma3Processor, InternVLProcessor]
    ModelType = Union[
        BlipForConditionalGeneration,
        Blip2ForConditionalGeneration,
        Gemma3ForConditionalGeneration,
        InternVLForConditionalGeneration
    ]
    ModelTuple = Tuple[ProcessorType, ModelType, str]
else:
    ModelTuple = Tuple

# Instantiate logger for this module
logger = logging.getLogger(__name__)


# Internal factory to load a model by key
def _load_model(key: str) -> ModelTuple:
    if key == "blip2":
        return initialize_blip2_model()
    if key == "gemma":
        return initialize_gemma_model()
    if key == "intern_vlm":
        return initialize_intern_vlm_model()
    return initialize_blip_model()


class ModelRegistry:
    def __init__(self, max_models_loaded: int = 1):
        # Re-entrant lock for thread-safety (API workers may be multi-threaded)
        self._lock = threading.RLock()
        # LRU cache: key -> (processor, model, device)
        self._cache: 'OrderedDict[str, ModelTuple]' = OrderedDict()
        self._max = max_models_loaded

    def get(self, key: str) -> ModelTuple:
        if key not in ALLOWED_MODELS:
            # Fail-fast on invalid user input
            raise ValueError("Invalid model key")
        with self._lock:
            # Fast path: cache hit => refresh LRU ordering
            if key in self._cache:
                processor, model, device = self._cache.pop(key)
                self._cache[key] = (processor, model, device)
                return processor, model, device
            # Cache miss: load model lazily
            processor, model, device = _load_model(key)
            try:
                # Inference mode (disable dropout, etc.)
                model.eval()
            except Exception:
                # Some models may not expose eval(); ignore quietly
                logger.warning("Unable to set model to eval mode for model: %s; skipping", key)
                pass
            # Add to LRU
            self._cache[key] = (processor, model, device)
            # Evict oldest until under capacity (free memory, including GPU)
            while len(self._cache) > self._max:
                k, (p, m, d) = self._cache.popitem(last=False)
                try:
                    # Move evicted model weights off GPU if possible
                    m.cpu()
                except Exception:
                    logger.warning("Unable to move model weights off GPU for model: %s; skipping", k)
                    pass
                # Drop strong refs to help GC
                del p, m
                gc.collect()
                if torch.cuda.is_available():
                    # Return freed memory back to CUDA allocator
                    torch.cuda.empty_cache()
            return processor, model, device


# Singleton registry (capacity from env)
registry = ModelRegistry(max_models_loaded=int(os.getenv("MODEL_CAPACITY", 1)))
