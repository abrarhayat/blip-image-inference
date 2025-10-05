import io
from PIL import Image
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def make_png_bytes():
    # Create a tiny in-memory image for testing
    im = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def test_caption_images_ok():
    files = [("images", ("tiny.png", make_png_bytes(), "image/png"))]
    resp = client.post("/api/caption-images?model=blip", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data

def test_collective_rejects_for_blip():
    files = [("images", ("tiny.png", make_png_bytes(), "image/png"))]
    resp = client.post("/api/caption-collective-images?model=blip", files=files)
    assert resp.status_code == 400