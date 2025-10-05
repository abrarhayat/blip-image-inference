import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

from app.prompts import DEFAULT_PROMPTS
from app.routers import caption, admin
from app.settings import settings

# -- Minimal, centralized logging setup (stdlib only) --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",  # common, readable format
)

# Quieter access logs in production
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Create FastAPI app after logging setup so any startup errors are logged with our format
app = FastAPI(title="VLM Caption API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" if settings.DEBUG else "https://your-frontend.example"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (API endpoints)
app.include_router(caption.router)
app.include_router(admin.router)

# Serve /static/* files (CSS/JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates directory
templates = Jinja2Templates(directory="app/templates")


# Simple test page at root
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_prompts": DEFAULT_PROMPTS})


# Lightweight metrics at /metrics (Prometheus)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# Health endpoints — used by load balancers and orchestration
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
