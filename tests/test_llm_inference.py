import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from PIL import Image
import requests
from gemma import initialize_gemma_model
from intern_vlm import initialize_intern_vlm_model
from inference import infer_image_caption


class TestLLMInference(unittest.TestCase):
    def setUp(self):
        self.image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
        self.raw_image = Image.open(requests.get(self.image_url, stream=True, timeout=5).raw).convert('RGB')

    def test_gemma_conditional(self):
        processor, model, device = initialize_gemma_model()
        prompt = "What animal is in this picture?"
        caption = infer_image_caption(processor, model, device, self.raw_image, prompt)
        self.assertTrue(caption is not None and len(caption) > 0)

    def test_gemma_unconditional(self):
        processor, model, device = initialize_gemma_model()
        caption = infer_image_caption(processor, model, device, self.raw_image)
        self.assertTrue(caption is not None and len(caption) > 0)

    def test_intern_vlm_conditional(self):
        processor, model, device = initialize_intern_vlm_model()
        prompt = "Please describe the image explicitly."
        caption = infer_image_caption(processor, model, device, self.raw_image, prompt)
        print("Caption from Intern VLM (conditional):", caption)
        self.assertTrue(caption is not None and len(caption) > 0)

    def test_intern_vlm_unconditional(self):
        processor, model, device = initialize_intern_vlm_model()
        caption = infer_image_caption(processor, model, device, self.raw_image)
        self.assertTrue(caption is not None and len(caption) > 0)


if __name__ == "__main__":
    unittest.main()
