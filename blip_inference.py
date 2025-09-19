import requests
from typing import Union
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_blip_model():
    """Initialize the BLIP model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    return processor, model, DEVICE

def initialize_blip2_model():
    """Initialize the BLIP model and processor."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(DEVICE)
    return processor, model, DEVICE


def infer_image_caption(processor: Union[BlipProcessor, Blip2Processor], 
                        model: Union[BlipForConditionalGeneration, Blip2ForConditionalGeneration], 
                        device: str, image: Image.Image, optional_caption_prompt: str = None):
    """Run inference on a single image and return the caption."""
    if optional_caption_prompt:
        inputs = processor(images=image, text=optional_caption_prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Remove the optional caption prefix if it was used
        if optional_caption_prompt and caption.lower().startswith(optional_caption_prompt.lower()):
            caption = caption.replace(optional_caption_prompt, '').strip()
    return caption

def run_demo_inference():
    print(f"Running demo inference on device: {DEVICE} with BLIP model")
    processor, model, _ = initialize_blip_model()
    """Test the BLIP model with a sample image."""
    IMAGE_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=5).raw).convert('RGB')
    # conditional image captioning
    CAPTIONING_TEXT = "a photo of"
    print(infer_image_caption(processor, model, DEVICE, raw_image, CAPTIONING_TEXT))
    # >>> a photo of a woman sitting on the beach with her dog

    # unconditional image captioning
    print(infer_image_caption(processor, model, DEVICE, raw_image))
    # >>> a woman sitting on the beach with her dog
    print("\n\n")

def run_demo_inference_blip2():
    print(f"Running demo inference on device: {DEVICE} with BLIP 2 model")
    processor, model, _ = initialize_blip2_model()
    """Test the BLIP model with a sample image."""
    IMAGE_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=5).raw).convert('RGB')
    # conditional image captioning
    CAPTIONING_TEXT = "Question: Describe the image as someone who is posting this on social media. Answer:"
    print(infer_image_caption(processor, model, DEVICE, raw_image, CAPTIONING_TEXT))
    # >>> A woman is sitting on the beach with her dog and is holding a cell phone in her hand.

    # unconditional image captioning
    print(infer_image_caption(processor, model, DEVICE, raw_image))
    # >>> a woman sitting on the beach with her dog
    print("\n\n")  

if __name__ == "__main__":
    run_demo_inference()
    run_demo_inference_blip2()
