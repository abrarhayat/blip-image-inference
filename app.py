import os
import argparse
from dotenv import load_dotenv
import hashlib
from flask import Flask, json, request, jsonify, render_template
from PIL import Image
from blip_inference import initialize_blip_model, initialize_blip2_model, infer_image_caption
from spacy_tagging import generate_spacy_tags
from redis_config import get_redis_client


load_dotenv()
app = Flask(__name__)

# Load BLIP model + processor once (warm start)
def get_model_and_processor(use_blip2):
    if use_blip2:
        return initialize_blip2_model()
    else:
        return initialize_blip_model()

# Initialize Redis client
REDIS_CACHE_TTL = 60 * 60 * 24  # cache for 24h
rdb = get_redis_client()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_name="BLIP2" if args.blip2 else "BLIP1", caption_prompt=CAPTION_PROMPT or "No caption prompt provided.")

@app.route("/caption-images", methods=["POST"])
def caption_images():
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
    parser.add_argument("--caption-prompt", type=str, 
                        default=None, 
                        help="Optional caption prompt for inference")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the Flask server on")
    args = parser.parse_args()

    print(f"Starting Flask server with options:")
    print(f"  BLIP2 model: {args.blip2}")
    print(f"  Caption prompt: {args.caption_prompt}")
    print(f"  Port: {args.port}")

    # Load model and processor based on user choice
    processor, model, device = get_model_and_processor(args.blip2)
    # Set optional caption prompt
    CAPTION_PROMPT = args.caption_prompt
    port = args.port or int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
