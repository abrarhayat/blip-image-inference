# Image Captioning API (FastAPI: BLIP, BLIP2, Gemma 3, InternVLM)

A production‑like FastAPI service for automatic image captioning using multiple vision‑language models, with Redis caching, a minimal browser test page, and production run instructions.

- Framework: FastAPI (auto‑docs at /docs)
- Python: 3.11 (recommended and assumed)
- Models: BLIP, BLIP2, Gemma 3, InternVLM (select per request via ?model=)
- Caching: Redis with TTL and namespaced keys
- Admin: cache reset route protected by API key (disabled in DEBUG)

## Supported models
- [BLIP (Salesforce/blip-image-captioning-base)](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [BLIP2 (Salesforce/blip2-opt-2.7b)](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [Gemma 3 (google/gemma-3-4b-it)](https://huggingface.co/google/gemma-3-4b-it)
- [InternVLM (OpenGVLab/InternVL3-1B-hf)](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Model is selected per request using the query parameter `?model=blip|blip2|gemma|intern_vlm`.

## Requirements
- Python 3.11
- Redis 7.x (local or container)

## Quick start (local dev)
1) Create and activate a Conda environment (Python 3.11)
```bash
conda create -n vlm-api python=3.11 -y
conda activate vlm-api
```

2) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

3) Start Redis (Docker example)
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

4) Configure environment
```bash
cp .env.example .env
# edit .env as needed (API_KEY, Redis host, etc.)
```

5) Run the API (development)
```bash
uvicorn --env-file .env app.main:app --reload --port 8000
```
or simply
```bash
sh scripts/start-local.sh
```

6) Open the test page and docs
- Test page (upload images): http://localhost:8000/
- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/healthz
- Metrics (Prometheus): http://localhost:8000/metrics

## Environment variables
See `.env.example` for all options. Common values:

- ENV=development|production|test
- DEBUG=true|false
- PORT=8000
- API_KEY=change-me                 # required for /api/admin/* when DEBUG=false
- REDIS_HOST=localhost
- REDIS_PORT=6379
- REDIS_DB=0
- MODEL_CAPACITY=1                  # max distinct models kept in memory
- CACHE_TTL_SECONDS=86400           # Redis TTL for cache entries

## Endpoints

### POST /api/caption-images
- Multipart form: `images` (one or more files)
- Query params:
  - `model` = `blip|blip2|gemma|intern_vlm` (default `blip`)
  - `caption_prompt` (optional)
  - `flag_caption_prompt` (optional)
- Accepts: image/jpeg, image/png, image/webp
- Response:
```json
{
  "results": [
    { "filename": "img.jpg", "caption": "...", "tags": ["..."], "flagged": false, "cache": false }
  ]
}
```
- Example:
```bash
curl -X POST \
  -F "images=@path/to/image.jpg" \
  "http://localhost:8000/api/caption-images?model=blip&caption_prompt=a%20photo%20of"
```

### POST /api/caption-collective-images
- Multipart form: `images` (multiple files)
- Only supported for `gemma` or `intern_vlm`
- Query params: same as above
- Response:
```json
{
  "collective_caption": "...",
  "count": 3,
  "tags": ["..."],
  "flagged": false
}
```
- Example:
```bash
curl -X POST \
  -F "images=@img1.jpg" -F "images=@img2.jpg" \
  "http://localhost:8000/api/caption-collective-images?model=gemma"
```

### POST /api/admin/reset-cache
- Header: `X-API-Key: <your-key>` (not required when `DEBUG=true`)
- Effect: Flushes the Redis database used by this service.
- Example:
```bash
curl -X POST -H "X-API-Key: change-me" http://localhost:8000/api/admin/reset-cache
```

## Caching details
- Single image cache key: `v1:img:{sha256(image_bytes)}`
- Collective cache key: `v1:collection:{sha256(concatenated_hashes)}`
- TTL: `CACHE_TTL_SECONDS` (default 86400 seconds)
- Redis failures are tolerated: requests still proceed without cache.

## Test page
- Served at `/` via Jinja2 template `app/templates/index.html` with assets under `app/static/`
- Lets you pick the model, optional prompts, and upload one or more images
- Calls the endpoints above and renders the results

## Running in production (Gunicorn + Uvicorn workers)
Use the provided config in `scripts/gunicorn_conf.py`.
```bash
ENV=production API_KEY=change-me \
  gunicorn -c scripts/gunicorn_conf.py app.main:app
```
Defaults bind to `0.0.0.0:8000`. GPU workloads typically run 1 worker per GPU.

## Docker (CPU baseline)
Build and run:
```bash
docker build -t vlm-api:latest .
docker run --rm -p 8000:8000 --env-file .env vlm-api:latest
```
For GPU, base on an NVIDIA CUDA runtime image and install matching Torch CUDA wheels; run with `--gpus all`.

## Notes and tips
- First request to a given model will download weights; start with `?model=blip` for a quick first run.
- Per‑request model selection avoids global mutable state.
- Model registry capacity (`MODEL_CAPACITY`) limits the number of simultaneously loaded models to avoid OOM.
- Logging uses Python stdlib `logging.basicConfig(...)` initialized in `app/main.py`.

## Development and tests
- Linting/formatting and type checking are optional; focus is on working API.
- Basic tests live under `tests/`.
- Run tests (pytest):
```bash
pytest -q
```
