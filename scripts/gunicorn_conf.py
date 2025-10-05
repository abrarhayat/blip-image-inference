import os

wsgi_app = "app.main:app"
worker_class = "uvicorn.workers.UvicornWorker"
workers = int(os.getenv("WEB_CONCURRENCY", 1))  # GPU-bound: 1 worker per GPU
threads = int(os.getenv("WEB_THREADS", 2))
bind = os.getenv("BIND", "0.0.0.0:8000")
timeout = int(os.getenv("TIMEOUT", 300))
keepalive = 5
accesslog = "-"  # consider raising to WARNING in prod via logger config
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
