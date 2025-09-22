import time
import torch
from transformers import AutoProcessor, Gemma3Processor, Gemma3ForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def initialize_gemma_model() -> tuple[Gemma3Processor, Gemma3ForConditionalGeneration, str]:
    """Initialize the Gemma model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        "google/gemma-3-4b-it",
        padding_side="left"
    )
    return processor, model, DEVICE

def run_demo_gemma_inference():
    print(f"Running demo inference on device: {DEVICE} with Gemma model")
    processor, model, _ = initialize_gemma_model()
    """Test the Gemma model with a sample image."""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant who generates captions for images."}
            ]
        },
        {
            "role": "user", "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
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
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
    print(processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
    #>>> A fluffy, snow-covered Pallas' cat exploring the winter.
    print(f"Time taken for inference: {time.time() - start_time} s")

if __name__ == "__main__":
    run_demo_gemma_inference()