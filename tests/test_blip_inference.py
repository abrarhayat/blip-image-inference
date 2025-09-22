import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from PIL import Image
import requests
from blip import initialize_blip_model, initialize_blip2_model
from inference import infer_image_caption


class TestBlipInference(unittest.TestCase):
    def setUp(self):
        self.image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
        self.raw_image = Image.open(requests.get(self.image_url, stream=True, timeout=5).raw).convert('RGB')

    def test_blip_conditional(self):
        processor, model, device = initialize_blip_model()
        caption = infer_image_caption(processor, model, device, self.raw_image, "a photo of")
        self.assertIn("a woman and her dog on the beach", caption)

    def test_blip_unconditional(self):
        processor, model, device = initialize_blip_model()
        caption = infer_image_caption(processor, model, device, self.raw_image)
        self.assertIn("a woman sitting on the beach with her dog", caption)

    def test_blip2_conditional(self):
        processor, model, device = initialize_blip2_model()
        prompt = "Question: Describe the image as someone who is posting this on social media. Answer:"
        caption = infer_image_caption(processor, model, device, self.raw_image, prompt)
        self.assertIn("A woman is sitting on the beach with her dog and is holding a cell phone in her hand.", caption)

    def test_blip2_unconditional(self):
        processor, model, device = initialize_blip2_model()
        caption = infer_image_caption(processor, model, device, self.raw_image)
        self.assertIn("a woman sitting on the beach with a dog", caption)


if __name__ == "__main__":
    unittest.main()
