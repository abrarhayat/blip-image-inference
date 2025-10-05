DEFAULT_PROMPTS: dict = {
    "blip": {
        "caption_prompt": "a photo of",
    },
    "blip2": {
        "caption_prompt": "Describe the image as if posting on social media, concise and vivid.",
    },
    "gemma": {
        "caption_prompt": (
            "You are a helpful assistant who generates captions for images. "
            "You keep the caption concise within 25 words and return a single caption without any additional text."
        ),
        "flag_caption_prompt": (
            "You are a helpful assistant who generates json objects from images. You respond with a single JSON object "
            "with a single key 'flag' and value 'true' or 'false' based on any of the images have any animals on them. "
            "DO NOT include any formatting, additional text, or explanation. Only respond with the JSON object."
        ),
    },
    "intern_vlm": {
        "caption_prompt": "You are a helpful assistant who generates captions for images. You keep the caption concise within 25 words and return a single caption without any additional text.",
        "flag_caption_prompt": (
            "You are a helpful assistant who generates json objects from images. You respond with a single JSON object "
            "with a single key 'flag' and value 'true' or 'false' based on any of the images have any animals on them. "
            "DO NOT include any formatting, additional text, or explanation. Only respond with the JSON object."),
    },
}
