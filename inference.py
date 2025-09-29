import json
import re
import time
from typing import Union

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, \
    Gemma3Processor, Gemma3ForConditionalGeneration, InternVLProcessor, InternVLForConditionalGeneration

from gemma import DEFAULT_GEMMA_PROMPT, DEFAULT_FLAG_GEMMA_PROMPT


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
                     "text": DEFAULT_GEMMA_PROMPT if not optional_caption_prompt else optional_caption_prompt}
                ]
            },
            {
                "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": "Generate a caption for this image. Keep the caption concise within 25 words. Return a single caption without any additional text."},
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


def infer_collective_caption(processor: Union[Gemma3Processor, InternVLProcessor],
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


def is_flagged(processor: Union[Gemma3Processor, InternVLProcessor],
               model: Union[Gemma3ForConditionalGeneration, InternVLForConditionalGeneration],
               device: str,
               images: list[Image.Image],
               optional_flag_prompt: str = None,
               max_new_tokens: int = 80) -> bool:
    """Verify if any of the provided images were downloaded from the internet. Only supported for Gemma / InternVLM.

    Parameters
    ----------
    processor : model processor supporting apply_chat_template
    model : Gemma3ForConditionalGeneration | InternVLForConditionalGeneration
    device : str (unused directly but kept for interface symmetry)
    images : list[PIL.Image]
    optional_flag_prompt : Optional str prompt instruction
    max_new_tokens : generation length cap

    Returns
    -------
    str : Collective caption
    """
    print(f"Running flag images using device: {device} with prompt: {optional_flag_prompt}")
    if not isinstance(model, (Gemma3ForConditionalGeneration, InternVLForConditionalGeneration)):
        raise ValueError("Collection captioning is only supported for Gemma / InternVLM models.")

    start_time = time.time()
    system_text = optional_flag_prompt or (
        DEFAULT_FLAG_GEMMA_PROMPT
    )
    user_instruction = (
        "Respond with a single JSON object with a single key 'flag' and value 'true' or 'false' based on your system instruction. DO NOT include any formatting, additional text, or explanation. Only respond with the JSON object."
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
    json_response = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    end_time = time.time()
    print(f"\nGenerated JSON Response: {json_response}")
    print(f"Extracted Flag: {extract_flag_json(json_response)}")
    print(f"Time taken for flag json inference: {end_time - start_time} s (images={len(images)})")
    return extract_flag_json(json_response)


def extract_flag_json(text: str):
    """
    Extracts and validates a JSON object with a 'flag' key from model output.
    Handles extra formatting (e.g., ```json ... ```).
    Returns the parsed dict if valid, else None.
    """
    # Remove code block formatting if present
    text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE)
    # Find the first {...} block
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group())
        print(f"Extracted JSON: {obj}")
        # Validate structure
        if (isinstance(obj, dict) and "flag" in obj
                and
                (isinstance(obj["flag"], bool)
                 or obj["flag"].lower() in ["true", "false"])):
            return bool(obj["flag"])
    except Exception:
        return False
    return False
