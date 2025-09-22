# Image Captioning Flask Server (BLIP, BLIP2, Gemma, InternVLM)


This project provides a Flask server for automatic image captioning using multiple state-of-the-art models from Hugging Face Transformers, with Redis caching for efficient repeated inference.

Supported models for image captioning:

- [BLIP (Salesforce/blip-image-captioning-base)](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [BLIP2 (Salesforce/blip2-opt-2.7b)](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [Gemma (google/gemma-3-4b-it)](https://huggingface.co/google/gemma-3-4b-it)
- [InternVLM (OpenGVLab/InternVL3-1B-hf)](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Note: Collective mode (multi-image → one caption) is supported only for Gemma and InternVLM.

You can select which model to use for inference via command-line arguments when starting the server. The API endpoint allows you to upload images and receive captions and tags in response. Redis is used to cache results for faster repeated requests.

## Recommended Environment Setup

It is recommended to use a virtual Python environment to avoid dependency conflicts. You can create one using conda:

```bash
conda create --name profile_name python=3.9
conda activate profile_name
```

## Setup

Install dependencies (including Redis and dotenv):

```bash
pip install -r requirements.txt
```

If you use a custom requirements file, ensure it includes:

- flask
- pillow
- transformers
- torch
- torchvision
- redis
- python-dotenv

## Running Redis with Docker

You can quickly set up a Redis server using Docker:

```bash
docker pull redis:latest
docker run -d --name redis-server -p 6379:6379 redis:latest
```

This will start a Redis server accessible on port 6379. Make sure your `.env` and application configuration match these settings.

## Environment Variables

Configuration values such as the server port are managed using a `.env` file. An example file `.env.example` is provided:

```ini
PORT=5001
```

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env
```

The Flask server will read the port number from `.env` when starting.

## Usage

Start the Flask server:

### Run with default settings (BLIP model, no caption prompt, and default port from .env or fallback to 5001)

```bash
python app.py
```


### Run with options (Language Model, Optional Caption Prompt and server port)


If ***no model argument is provided***, the server will use the ***base BLIP model*** by default.

You can choose which model to use and optionally provide a caption prompt via command-line arguments when starting the Flask server:

```bash
python app.py --blip2 --caption-prompt "Question: Describe the image as someone who is posting this on social media. Answer:"
python app.py --gemma --caption-prompt "What animal is on the candy."
python app.py --intern-vlm --caption-prompt "Please describe the image explicitly."
```

**Arguments:**

- `--blip2`: Use the BLIP2 model (`Salesforce/blip2-opt-2.7b`). 
- `--gemma`: Use the Gemma model (`google/gemma-3-4b-it`).
- `--intern-vlm`: Use the InternVLM model (`OpenGVLab/InternVL3-1B-hf`).
- `--caption-prompt`: Optional prompt to guide caption generation. If omitted, the model will generate an unconditional caption.
- `--port`: Specify the port for the Flask server (default is 5001, or value from `.env`). Example: `--port 8080`


## Sample Caption Prompts

**BLIP1 (Salesforce/blip-image-captioning-base):**

- "a photo of"

**BLIP2 (Salesforce/blip2-opt-2.7b):**

- "Question: Describe the image as someone who is posting this on social media. Answer:"

**Gemma (google/gemma-3-4b-it):**

- "What animal is on the candy."

**InternVLM (OpenGVLab/InternVL3-1B-hf):**

- "Please describe the image explicitly."


<a id="model-docs"></a>
### Model Documentation & Best Practices

- [BLIP1 Documentation](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [BLIP2 Documentation](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [Gemma Documentation](https://huggingface.co/google/gemma-3-4b-it)
- [InternVLM Documentation](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Refer to the Hugging Face model cards above for more prompt examples and best practices for each model.

Redis must be running and accessible at the host/port specified in your `.env` file.

### API Endpoint

`POST /caption-images`


Upload one or more images using multipart form data under the key `images`. The server will return captions and automatically generated tags for each image. Tags are extracted from the caption using spaCy NLP and NLTK bigrams, including noun phrases, nouns, adjectives, and common bigrams.

Example response:

```json
{
	"results": [
		{
			"filename": "your_image.jpg",
			"caption": "a woman sitting on the beach with her dog",
			"tags": [
				"woman", "beach", "dog", "sitting", "woman_sitting", "beach_dog"
			]
		}
	]
}
```

Example using `curl`:

```bash
curl -X POST -F "images=@your_image.jpg" http://localhost:5001/caption-images
```

#### Tagging Feature

The API includes a `tags` field in its response for each image. Tags are generated from the caption using spaCy and NLTK:

- Noun phrases (noun_chunks)
- Nouns, proper nouns, and adjectives
- Simple bigrams (two-word combinations)

This helps with downstream tasks such as search, filtering, and categorization.

#### Collective Captioning Endpoint

`POST /caption-collective-images`

Upload multiple images using multipart form data under the key `images`. The server returns a single caption for the entire collection along with `tags` extracted from that caption using spaCy.

- Supported models: Gemma, InternVLM
- Response:
```json
{
  "collective_caption": "a cozy living room with a gray sofa and plant",
  "count": 3,
  "tags": ["living room", "sofa", "plant", "cozy", "gray sofa"]
}
```

Example using `curl`:
```bash
curl -X POST \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" \
  http://localhost:5001/caption-collective-images
```

Notes:
- Requests are cached in Redis using a combined SHA-256 hash of the uploaded images.
- If a non-supported model is active, the endpoint returns `400` with an explanatory error.

---

### Model Selection Endpoint

`POST /set-model`

Change the currently active model for image captioning while the server is running. Send a JSON payload with the desired model key:

Example request:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"model": "blip2"}' \
     http://localhost:5001/set-model
```

Valid model keys:
- `blip` (BLIP1)
- `blip2` (BLIP2)
- `gemma` (Gemma)
- `intern_vlm` (InternVLM)

Example response:
```json
{
  "model_name": "Blip2ForConditionalGeneration"
}
```

This endpoint allows you to switch models dynamically from the web UI or via API calls. All subsequent caption requests will use the selected model until changed again.

---

### Caption Prompt Endpoint

`POST /set-caption-prompt`

Set or update the caption prompt used to guide caption generation. The prompt is optional; if not set (or cleared) the models will produce an unconditional caption. See the model-specific guidance in the [Model Documentation & Best Practices](#model-docs) section.

Send a JSON body with the key `caption_prompt`:

Example request:
```bash
curl -X POST -H "Content-Type: application/json" \
		 -d '{"caption_prompt": "Describe the scene focusing on emotions."}' \
		 http://localhost:5001/set-caption-prompt
```

Example response:
```json
{
	"caption_prompt": "Describe the scene focusing on emotions."
}
```

Clearing / removing the prompt:
```bash
curl -X POST -H "Content-Type: application/json" \
		 -d '{"caption_prompt": ""}' \
		 http://localhost:5001/set-caption-prompt
```
Response when cleared (normalized to null/None internally):
```json
{
	"caption_prompt": null
}
```

Notes:
- The current prompt is displayed on the test page and updates immediately after saving.
- An empty or whitespace-only string is treated as `null` (no prompt) for clearer downstream logic.
- The prompt applies to all subsequent `/caption-images` requests until changed again.

---

### Redis Reset Endpoint

`GET /reset-redis`

Clears all cached results from Redis. Useful for testing or development when you want to reset the cache.

Example using `curl`:

```bash
curl -X GET http://localhost:5001/reset-redis
```

Example response:

```json
{
	"message": "Redis cache cleared"
}
```

### Redis Configuration

Redis connection details are read from your `.env` file. Example variables:

```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

See `.env.example` for a template.


---

### Optional Browser Test Page (index.html)

An optional browser-based test page is included to help you try the API without writing a client. It is served by the Flask app at the root path `/` and lives at `templates/index.html`.

How to use:
1. Ensure Redis is running and the app is started: `python app.py`.
2. Open your browser to `http://localhost:5001/` (or the port set in your `.env`).
3. Use the file picker to select one or more images and click "Caption Images".
4. For each image, you will see:
   - A preview thumbnail.
   - The generated caption when ready.
   - Auto-generated tags shown as pills.

Notes:
- The page sends `multipart/form-data` to `POST /caption-images` with files under the `images` field.
- Results are cached in Redis for 24 hours based on a SHA-256 hash of the image bytes. Re-uploading the same file will return cached results instantly.
- You can clear the cache via `GET /reset-redis` (see section above).
- This page is intended for local development and demos. If deploying publicly, harden or remove the route and template as needed.

## Unit Tests

Unit tests for inference are provided in `tests/` directory. These tests verify that both models generate expected captions for a sample image, using both conditional and unconditional prompts.

To run the tests:

```bash
python -m unittest discover tests
```

This will execute all tests in the `tests/` directory and report any failures or errors. Make sure you have all dependencies installed and a working internet connection to download the sample image.
