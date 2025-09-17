import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def initialize_model():
    """Initialize the BLIP model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

processor, model, device = initialize_model()


def infer_image_caption(image: Image.Image, optional_caption: str = None):
    """Run inference on a single image and return the caption."""
    if optional_caption:
        inputs = processor(image, optional_caption, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def run_demo_inference():
    """Test the BLIP model with a sample image."""
    IMAGE_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=5).raw).convert('RGB')
    # conditional image captioning
    CAPTIONING_TEXT = "a photo of"
    inputs = processor(raw_image, CAPTIONING_TEXT, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photo of a woman sitting on the beach with her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a woman sitting on the beach with her dog

if __name__ == "__main__":
    run_demo_inference()
