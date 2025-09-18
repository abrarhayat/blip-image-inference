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

# BLIP Image Captioning Flask Server


This project provides a Flask server for automatic image captioning using the BLIP (Bootstrapped Language Image Pretraining) model from Hugging Face Transformers, with Redis caching for efficient repeated inference.

BLIP is a state-of-the-art vision-language model that generates captions for images, both conditionally (with a prompt) and unconditionally. This server exposes an API endpoint to upload images and receive captions in response. Redis is used to cache results for faster repeated requests.

[Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

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
- redis
- python-dotenv


## Usage

Start the Flask server:

```bash
python app.py
```

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

You can still run `blip_inference.py` directly for standalone captioning demos.

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
