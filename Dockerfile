# Dockerfile
FROM python:3.11

# System libs for Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first to leverage Docker layer cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy src
COPY app ./app
COPY scripts/gunicorn_conf.py ./scripts/gunicorn_conf.py

ENV PYTHONUNBUFFERED=1 ENV=production
EXPOSE 8000
CMD ["gunicorn", "-c", "scripts/gunicorn_conf.py", "app.main:app"]