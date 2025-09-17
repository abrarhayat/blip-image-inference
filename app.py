import os
from dotenv import load_dotenv
import hashlib
from flask import Flask, json, request, jsonify
from PIL import Image
from blip_inference import initialize_model, infer_image_caption
from redis_config import get_redis_client


load_dotenv()
app = Flask(__name__)

# Load BLIP model + processor once (warm start)
_, model, _ = initialize_model()

# Initialize Redis client
REDIS_CACHE_TTL = 60 * 60 * 24  # cache for 24h
rdb = get_redis_client()

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
        caption = infer_image_caption(image)

        result = {"caption": caption}

        # Cache result
        rdb.set(file_hash, json.dumps(result), ex=REDIS_CACHE_TTL)
        results.append({"filename": f.filename, **result})

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
