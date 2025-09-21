import time
from typing import Union
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, Gemma3ForConditionalGeneration, InternVLForConditionalGeneration
import torch

def infer_image_caption(processor: Union[BlipProcessor, Blip2Processor], 
                        model: Union[BlipForConditionalGeneration, Blip2ForConditionalGeneration, 
                                     Gemma3ForConditionalGeneration, InternVLForConditionalGeneration], 
                        device: str, image: Image.Image, optional_caption_prompt: str = None):
    start_time = time.time()
    if isinstance(model, Gemma3ForConditionalGeneration) or isinstance(model, InternVLForConditionalGeneration):
        """Run Gemma or Intern VLM inference on a single image and return the caption."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant who generates captions for images." if not optional_caption_prompt else optional_caption_prompt}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Generate a caption for this image. Keep the caption concise within 10 words. Return a single caption without any additional text."},
                ]
            }
        ]
        inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
        caption = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        end_time = time.time()
    elif isinstance(model, (BlipForConditionalGeneration, Blip2ForConditionalGeneration)):
        """Run BLIP inference on a single image and return the caption."""
        
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
            end_time = time.time()
    else:
        raise ValueError("Unsupported model type for inference.")
    print(f"Generated Caption: {caption}")
    print(f"Time taken for inference: {end_time - start_time} s")
    return caption