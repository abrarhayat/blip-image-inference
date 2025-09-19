# BLIP/BLIP2 Image Captioning Flask Server

This project supports both BLIP (Bootstrapped Language Image Pretraining) and BLIP2 models for image captioning. BLIP2 is an advanced vision-language model that builds on BLIP, offering improved performance and support for larger language models. You can select either BLIP or BLIP2 when running the Flask server. These models generate captions for images, both conditionally (with a prompt) and unconditionally.

This project provides a Flask server for automatic image captioning using either BLIP or BLIP2 models from Hugging Face Transformers, with Redis caching for efficient repeated inference.

The server in this project exposes an API endpoint to upload images and receive captions and tags in response. Redis is used to cache results for faster repeated requests.

- [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)

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

### Run with default settings (Base BLIP and no caption prompt)

```bash
python app.py
```

### Run with specific arguments

You can choose which BLIP model to use and optionally provide a caption prompt via command-line arguments when starting the Flask server:

```bash
python app.py --blip2 --caption-prompt "Question: Describe the image as someone who is posting this on social media. Answer:"
```

**Arguments:**

- `--blip2`: Use the BLIP2 model (`Salesforce/blip2-opt-2.7b`). Omit for BLIP1 (`Salesforce/blip-image-captioning-base`).
- `--caption-prompt`: Optional prompt to guide caption generation. If omitted, the model will generate an unconditional caption.
- `--port`: Specify the port for the Flask server (default is 5001, or value from `.env`). Example: `--port 8080`

## Sample Caption Prompts

**BLIP1 (Salesforce/blip-image-captioning-base):**

- "a photo of"

**BLIP2 (Salesforce/blip2-opt-2.7b):**

- "Question: Describe the image as someone who is posting this on social media. Answer:"

### Model Documentation & Best Practices

- [BLIP1 Documentation](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [BLIP2 Documentation](https://huggingface.co/Salesforce/blip2-opt-2.7b)

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

## Unit Tests

Unit tests for BLIP and BLIP2 inference are provided in `tests/test_blip_inference.py`. These tests verify that both models generate expected captions for a sample image, using both conditional and unconditional prompts.

To run the tests:

```bash
python -m unittest discover tests
```

This will execute all tests in the `tests/` directory and report any failures or errors. Make sure you have all dependencies installed and a working internet connection to download the sample image.
