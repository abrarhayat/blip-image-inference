import os
import argparse
from dotenv import load_dotenv
import hashlib
from blip import initialize_blip_model, initialize_blip2_model
from gemma import initialize_gemma_model
from intern_vlm import initialize_intern_vlm_model
from inference import infer_image_caption
from flask import Flask, json, request, jsonify, render_template
from PIL import Image
from spacy_tagging import generate_spacy_tags
from redis_config import get_redis_client
from typing import Union
from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration, Gemma3ForConditionalGeneration, InternVLForConditionalGeneration


load_dotenv()
app = Flask(__name__)

# Global state for selected model
SELECTED_MODEL_KEY = None
processor = None
model = None
device = None
model_name = None
CAPTION_PROMPT = None

# Load BLIP model + processor once (warm start)
def get_model_and_processor(model_key) -> tuple[AutoProcessor, Union[BlipForConditionalGeneration, Blip2ForConditionalGeneration,
                                                                         Gemma3ForConditionalGeneration, InternVLForConditionalGeneration], str]:
    if model_key == "blip2":
        return initialize_blip2_model()
    elif model_key == "gemma":
        return initialize_gemma_model()
    elif model_key == "intern_vlm":
        return initialize_intern_vlm_model()
    else:
        return initialize_blip_model()

# Initialize Redis client
REDIS_CACHE_TTL = 60 * 60 * 24  # cache for 24h
rdb = get_redis_client()

@app.route("/", methods=["GET"])
def index():
    global model_name, CAPTION_PROMPT
    return render_template("index.html", model_name=model_name, caption_prompt=CAPTION_PROMPT or "No caption prompt provided.")
@app.route("/set-model", methods=["POST"])
def set_model():
    global SELECTED_MODEL_KEY, processor, model, device, model_name
    data = request.get_json()
    model_key = data.get("model", "blip")
    processor, model, device = get_model_and_processor(model_key)
    model_name = type(model).__name__
    SELECTED_MODEL_KEY = model_key
    return jsonify({"model_name": model_name}), 200

@app.route("/caption-images", methods=["POST"])
def caption_images():
    global processor, model, device, CAPTION_PROMPT
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
        caption = infer_image_caption(processor, model, device, image, CAPTION_PROMPT)
        if(caption is None or len(caption) == 0):
            caption = "No caption could be generated."
            results.append({"filename": f.filename, "caption": caption, "tags": []})
            continue

        result = {"caption": caption, "tags": generate_spacy_tags(caption)}

        # Cache result
        rdb.set(file_hash, json.dumps(result), ex=REDIS_CACHE_TTL)
        results.append({"filename": f.filename, **result})

    return jsonify({"results": results})

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
    CAPTION_PROMPT = args.caption_prompt
    port = args.port or int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
