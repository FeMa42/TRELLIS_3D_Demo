"""
Gemini 2.5 Flash Image pipeline adapter.

This module provides a wrapper around Google's Gemini 2.5 Flash Image API
that matches the interface of diffusers pipelines (like FLUX).

Usage:
    from modules.gemini_pipeline import GeminiImagePipeline

    pipe = GeminiImagePipeline(api_key="your-api-key")
    result = pipe(["A sunset over mountains"], height=512, width=512)
    images = result.images
"""

import os
from typing import List, Optional, Union
from collections import namedtuple
from io import BytesIO
import time

from google import genai
from google.genai import types
from PIL import Image


class GeminiImagePipeline:
    """
    Adapter to make Gemini 2.5 Flash Image API look like a diffusers pipeline.

    This allows Gemini to be used as a drop-in replacement for FLUX
    in the existing codebase with minimal changes.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini Image Pipeline.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set the environment variable or pass api_key parameter.\n"
                "Get your API key from: https://aistudio.google.com"
            )

        self.client = genai.Client(api_key=self.api_key)

        # Track generation statistics
        self.total_images_generated = 0
        self.total_api_calls = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float = 0.0,  # Ignored for Gemini
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,  # Ignored for Gemini
        max_sequence_length: int = 256,  # Ignored for Gemini
        generator=None  # Seed handled differently in Gemini
    ):
        """
        Generate images matching FLUX pipeline interface.

        Args:
            prompt: Text prompt(s) for image generation
            guidance_scale: Ignored (Gemini doesn't use CFG)
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Ignored (Gemini handles internally)
            max_sequence_length: Ignored (Gemini handles internally)
            generator: Ignored (Gemini doesn't support seed)

        Returns:
            namedtuple with .images attribute containing list of PIL Images
        """
        # Normalize prompt to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Convert dimensions to Gemini aspect ratio
        aspect_ratio = self._get_aspect_ratio(width, height)

        images = []

        for single_prompt in prompts:
            try:
                image = self._generate_single_image(single_prompt, aspect_ratio)
                if image is not None:
                    images.append(image)
                    self.total_images_generated += 1
                else:
                    print(f"⚠️ Failed to generate image for prompt: {single_prompt[:50]}...")

            except Exception as e:
                print(f"⚠️ Gemini API error for prompt '{single_prompt[:50]}...': {e}")
                # Continue with next prompt instead of failing completely

        # Return result matching FLUX format
        Result = namedtuple('GenerationResult', ['images'])
        return Result(images=images)

    def _generate_single_image(
        self,
        prompt: str,
        aspect_ratio: str,
        max_retries: int = 3
    ) -> Optional[Image.Image]:
        """
        Generate a single image with retry logic.

        Args:
            prompt: Text description
            aspect_ratio: Gemini aspect ratio string (e.g., "1:1", "16:9")
            max_retries: Maximum number of retry attempts

        Returns:
            PIL Image or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                self.total_api_calls += 1

                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
                    )
                )

                # Extract image from response
                for part in response.parts:
                    if part.inline_data:
                        return Image.open(BytesIO(part.inline_data.data))

                # No image in response
                print(f"⚠️ No image in Gemini response (attempt {attempt + 1}/{max_retries})")

            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed
                    print(f"❌ All retries exhausted: {e}")
                    return None

                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"⚠️ API error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

        return None

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """
        Convert pixel dimensions to nearest Gemini aspect ratio.

        Gemini supports: 1:1, 3:4, 4:3, 9:16, 16:9, and others.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Aspect ratio string (e.g., "16:9")
        """
        ratio = width / height

        # Map to closest supported aspect ratio
        if abs(ratio - 1.0) < 0.1:
            return "1:1"  # Square
        elif ratio > 1.7:
            return "16:9"  # Wide landscape
        elif ratio > 1.2:
            return "4:3"  # Landscape
        elif ratio < 0.6:
            return "9:16"  # Tall portrait
        elif ratio < 0.8:
            return "3:4"  # Portrait
        else:
            return "1:1"  # Default to square

    def to(self, device):
        """
        Compatibility method for .to(device) calls.
        Gemini doesn't use devices, so this is a no-op.

        Args:
            device: Ignored

        Returns:
            self for method chaining
        """
        return self

    def enable_model_cpu_offload(self):
        """
        Compatibility method for CPU offloading.
        Gemini is API-based, so this is a no-op.
        """
        pass

    def get_stats(self) -> dict:
        """
        Get generation statistics.

        Returns:
            Dictionary with usage statistics
        """
        estimated_cost = self.total_images_generated * 0.039

        return {
            "total_images_generated": self.total_images_generated,
            "total_api_calls": self.total_api_calls,
            "estimated_cost_usd": round(estimated_cost, 2),
            "model": "gemini-2.5-flash-image"
        }


# Factory function for consistency with other modules
def create_gemini_pipeline(api_key: Optional[str] = None) -> GeminiImagePipeline:
    """
    Create a Gemini image generation pipeline.

    Args:
        api_key: Optional API key (defaults to GEMINI_API_KEY env var)

    Returns:
        GeminiImagePipeline instance
    """
    return GeminiImagePipeline(api_key=api_key)
