import torch
from transformers import AutoProcessor, InternVLProcessor, InternVLForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_INTERN_VLM_PROMPT = "You are a helpful assistant who generates captions for images. You keep the caption concise within 25 words and return a single caption without any additional text."
DEFAULT_INTERN_VLM_FLAG_PROMPT = "You are a helpful assistant who generates json objects from images. You respond with a single JSON object with a single key 'flag' and value 'true' or 'false' based on any of the images have any animals on them. DO NOT include any formatting, additional text, or explanation. Only respond with the JSON object."


def initialize_intern_vlm_model() -> tuple[InternVLProcessor, InternVLForConditionalGeneration, str]:
    """Initialize the InternVLM model and processor."""
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    model = InternVLForConditionalGeneration.from_pretrained("OpenGVLab/InternVL3-1B-hf").to(DEVICE)
    return processor, model, DEVICE


def run_demo_intern_vlm_inference():
    processor, model, _ = initialize_intern_vlm_model()
    """Test the InternVLM model with a sample image."""
    print(f"Running demo inference on device: {DEVICE} with Intern VLM model")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image",
                 "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                {"type": "text",
                 "text": "Generate a caption for this image. Keep the caption concise within 25 words. Return a single caption without any additional text."}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
    # >>> A ginger and gray cat walking in the snow.


if __name__ == "__main__":
    run_demo_intern_vlm_inference()
