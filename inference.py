import time
from typing import Union

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, \
    Gemma3Processor, Gemma3ForConditionalGeneration, InternVLProcessor, InternVLForConditionalGeneration


def infer_image_caption(processor: Union[BlipProcessor, Blip2Processor, Gemma3Processor, InternVLProcessor],
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
                    {"type": "text",
                     "text": "You are a helpful assistant who generates captions for images." if not optional_caption_prompt else optional_caption_prompt}
                ]
            },
            {
                "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": "Generate a caption for this image. Keep the caption concise within 10 words. Return a single caption without any additional text."},
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


def infer_collective_caption(processor: Gemma3Processor,
                             model: Union[Gemma3ForConditionalGeneration, InternVLForConditionalGeneration],
                             device: str,
                             images: list[Image.Image],
                             optional_caption_prompt: str = None,
                             max_new_tokens: int = 80) -> str:
    """Generate a single caption that describes a collection of images (Gemma / InternVLM only).

    Parameters
    ----------
    processor : model processor supporting apply_chat_template
    model : Gemma3ForConditionalGeneration | InternVLForConditionalGeneration
    device : str (unused directly but kept for interface symmetry)
    images : list[PIL.Image]
    optional_caption_prompt : Optional str prompt instruction
    max_new_tokens : generation length cap

    Returns
    -------
    str : Collective caption
    """
    print(f"Running collective captioning on {len(images)} images using device: {device}")
    if not isinstance(model, (Gemma3ForConditionalGeneration, InternVLForConditionalGeneration)):
        raise ValueError("Collection captioning is only supported for Gemma / InternVLM models.")

    start_time = time.time()
    system_text = optional_caption_prompt or (
        "You are a helpful assistant generating ONE concise caption summarizing ALL provided images."
    )
    user_instruction = (
        "Provide a single holistic caption (<= 25 words) that best summarizes the full set of images."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": ([{"type": "image", "image": img} for img in images] +
                                     [{"type": "text", "text": user_instruction}])}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, cache_implementation="static")
    caption = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    end_time = time.time()
    print(f"Generated Collective Caption: {caption}")
    print(f"Time taken for collective inference: {end_time - start_time} s (images={len(images)})")
    return caption
