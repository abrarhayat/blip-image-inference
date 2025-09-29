import argparse
import hashlib
import os
from typing import Union

from PIL import Image
from dotenv import load_dotenv
from flask import Flask, json, request, jsonify, render_template
from transformers import Gemma3Processor, BlipForConditionalGeneration, Blip2ForConditionalGeneration, \
    Gemma3ForConditionalGeneration, InternVLForConditionalGeneration

from blip import initialize_blip_model, initialize_blip2_model, DEFAULT_BLIP_PROMPT, DEFAUALT_BLIP2_PROMPT
from gemma import initialize_gemma_model, DEFAULT_GEMMA_PROMPT, DEFAULT_FLAG_GEMMA_PROMPT
from inference import infer_image_caption, infer_collective_caption, is_flagged
from intern_vlm import initialize_intern_vlm_model, DEFAULT_INTERN_VLM_PROMPT, DEFAULT_INTERN_VLM_FLAG_PROMPT
from redis_config import get_redis_client
from spacy_tagging import generate_spacy_tags

load_dotenv()
app = Flask(__name__)

# Global state for selected model
SELECTED_MODEL_KEY = None
processor = None
model = None
device = None
model_name = None
caption_prompt = None
flag_caption_prompt = None

# Store models loaded in a dictionary for easy access
MODEL_CACHE = {}


def get_default_prompt_for_model(model_key: str) -> Union[str, None]:
    if model_key == "blip":
        return DEFAULT_BLIP_PROMPT
    elif model_key == "blip2":
        return DEFAUALT_BLIP2_PROMPT
    elif model_key == "gemma":
        return DEFAULT_GEMMA_PROMPT
    elif model_key == "intern_vlm":
        return DEFAULT_INTERN_VLM_PROMPT
    return None


def get_flag_prompt_for_model(model_key: str) -> Union[str, None]:
    if model_key == "gemma":
        return DEFAULT_FLAG_GEMMA_PROMPT
    elif model_key == "intern_vlm":
        return DEFAULT_INTERN_VLM_FLAG_PROMPT
    return None


# Load BLIP model + processor once (warm start)
def get_model_and_processor(model_key) -> tuple[
    Gemma3Processor, Union[BlipForConditionalGeneration, Blip2ForConditionalGeneration,
    Gemma3ForConditionalGeneration, InternVLForConditionalGeneration], str]:
    model_result = MODEL_CACHE.get(model_key)
    if model_result:
        return model_result

    if model_key == "blip2":
        MODEL_CACHE[model_key] = initialize_blip2_model()
    elif model_key == "gemma":
        MODEL_CACHE[model_key] = initialize_gemma_model()
    elif model_key == "intern_vlm":
        MODEL_CACHE[model_key] = initialize_intern_vlm_model()
    else:
        MODEL_CACHE[model_key] = initialize_blip_model()

    return MODEL_CACHE[model_key]


# Initialize Redis client
REDIS_CACHE_TTL = 60 * 60 * 24  # cache for 24h
rdb = get_redis_client()


@app.route("/", methods=["GET"])
def index():
    global model_name, caption_prompt, flag_caption_prompt
    if caption_prompt is None:
        caption_prompt = get_default_prompt_for_model(SELECTED_MODEL_KEY)
    if flag_caption_prompt is None:
        flag_caption_prompt = get_flag_prompt_for_model(SELECTED_MODEL_KEY)
    return render_template("index.html", model_name=model_name, caption_prompt=caption_prompt,
                           flag_caption_prompt=flag_caption_prompt)


@app.route("/set-model", methods=["POST"])
def set_model():
    global SELECTED_MODEL_KEY, processor, model, device, model_name, flag_caption_prompt
    data = request.get_json()
    model_key = data.get("model", "blip")
    processor, model, device = get_model_and_processor(model_key)
    model_name = type(model).__name__
    SELECTED_MODEL_KEY = model_key
    # Reset the flag prompt to the model's default
    flag_caption_prompt = get_flag_prompt_for_model(model_key)
    return jsonify({"model_name": model_name, "flag_caption_prompt": flag_caption_prompt}), 200


@app.route("/set-caption-prompt", methods=["POST"])
def set_caption_prompt():
    global caption_prompt
    data = request.get_json(silent=True) or {}
    new_prompt = data.get("caption_prompt", "")
    # Normalize empty prompt to None for clearer downstream handling
    caption_prompt = new_prompt.strip() or None
    return jsonify({"caption_prompt": caption_prompt}), 200


@app.route("/set-flag-prompt", methods=["POST"])
def set_flag_prompt():
    global flag_caption_prompt
    data = request.get_json(silent=True) or {}
    new_prompt = data.get("flag_caption_prompt", "")
    # Normalize empty prompt to None for clearer downstream handling
    flag_caption_prompt = new_prompt.strip() or None
    return jsonify({"flag_caption_prompt": flag_caption_prompt}), 200


@app.route("/caption-images", methods=["POST"])
def caption_images():
    global processor, model, device, caption_prompt, flag_caption_prompt
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for f in files:
        # Hash file contents to support caching
        file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Check Redis cache
        cached_result = rdb.get(file_hash)
        if cached_result:
            result = json.loads(cached_result)
            results.append({"filename": f.filename, **result})
            continue

        # Process new image
        image = Image.open(f.stream).convert("RGB")
        caption = infer_image_caption(processor, model, device, image, caption_prompt)
        flagged = is_flagged(processor, model, device, [image], flag_caption_prompt)
        if (caption is None or len(caption) == 0):
            caption = "No caption could be generated."
            results.append({"filename": f.filename, "caption": caption, "tags": []})
            continue

        result = {"caption": caption, "tags": generate_spacy_tags(caption), "flagged": flagged}

        # Cache result
        rdb.set(file_hash, json.dumps(result), ex=REDIS_CACHE_TTL)
        results.append({"filename": f.filename, **result})

    return jsonify({"results": results})


@app.route("/caption-collective-images", methods=["POST"])
def caption_collective_images():
    """Generate a single caption for a collection of images (Gemma / InternVLM only).

    Returns JSON of the form:
    {"collective_caption": str, "count": int}
    """
    global processor, model, device, caption_prompt, flag_caption_prompt
    if not isinstance(model, (Gemma3ForConditionalGeneration, InternVLForConditionalGeneration)):
        return jsonify({"error": "Collective captioning only supported for Gemma / InternVLM models"}), 400

    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    if len(files) == 0:
        return jsonify({"error": "No images uploaded"}), 400

    images = []
    hashes = []
    for f in files:
        file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        hashes.append(file_hash)
        image = Image.open(f.stream).convert("RGB")
        images.append(image)

    # Optionally cache by a combined hash of all images
    combined_hash = hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
    cache_key = f"collection:{combined_hash}"
    cached = rdb.get(cache_key)
    if cached:
        return jsonify(json.loads(cached))

    collective_caption = infer_collective_caption(processor, model, device, images, caption_prompt, max_new_tokens=200)
    flagged = is_flagged(processor, model, device, images, flag_caption_prompt, max_new_tokens=200)
    response = {"collective_caption": collective_caption, "count": len(images),
                "tags": generate_spacy_tags(collective_caption), "flagged": flagged}
    rdb.set(cache_key, json.dumps(response), ex=REDIS_CACHE_TTL)
    return jsonify(response)


@app.route("/reset-redis", methods=["GET"])
def reset_redis():
    rdb.flushdb()
    return jsonify({"message": "Redis cache cleared"}), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP Image Captioning Flask Server")
    parser.add_argument("--blip2", action="store_true", help="Use BLIP2 model (default: BLIP1)")
    parser.add_argument("--gemma", action="store_true", help="Optionally use Gemma for inference")
    parser.add_argument("--intern-vlm", action="store_true", help="Optionally use Intern VLM for inference")
    parser.add_argument("--caption-prompt", type=str,
                        default=None,
                        help="Optional caption prompt for inference")
    parser.add_argument("--flag-caption-prompt", type=str, default=None,
                        help="Optional flag prompt for inference")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the Flask server on")
    args = parser.parse_args()

    print(f"Starting Flask server with options:")
    print(f"  Caption prompt: {args.caption_prompt}")
    print(f"  Port: {args.port}")

    # Determine initial model key
    if args.blip2:
        SELECTED_MODEL_KEY = "blip2"
    elif args.gemma:
        SELECTED_MODEL_KEY = "gemma"
    elif args.intern_vlm:
        SELECTED_MODEL_KEY = "intern_vlm"
    else:
        SELECTED_MODEL_KEY = "blip"

    # Load model and processor based on user choice
    processor, model, device = get_model_and_processor(SELECTED_MODEL_KEY)
    model_name = type(model).__name__
    print(f"Using model: {model_name} on device: {device}")
    # Set optional caption prompt
    caption_prompt = args.caption_prompt
    # Set option flag prompt
    flag_caption_prompt = args.flag_caption_prompt
    port = args.port or int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
