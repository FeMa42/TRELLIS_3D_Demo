"""
Content moderation module for TRELLIS Streamlit application.

This module handles text and image content moderation using:
- OpenAI Moderation API for text content
- Stable Diffusion Safety Checker for image content

Usage:
    from modules.content_moderation import ContentModerator
    
    moderator = ContentModerator()
    is_safe, details = moderator.check_text_safety("user prompt")
    filtered_images = moderator.check_image_safety(image_list)
"""

import torch
from openai import OpenAI
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from typing import List, Tuple, Dict, Any
from PIL import Image


class ContentModerator:
    """
    Handles content moderation for both text and image content.
    """
    
    def __init__(self):
        """Initialize the content moderation system."""
        self.openai_client = None
        self.text_moderation_enabled = False
        self.image_moderation_enabled = False
        self.safety_checker = None
        self.safety_feature_extractor = None
        
        self._init_text_moderation()
        self._init_image_moderation()
    
    def _init_text_moderation(self) -> None:
        """Initialize OpenAI text moderation."""
        try:
            self.openai_client = OpenAI()
            self.text_moderation_enabled = True
            print("✅ OpenAI text moderation initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: OpenAI client initialization failed: {e}")
            print("Content moderation will be disabled. Set OPENAI_API_KEY to enable.")
            self.openai_client = None
            self.text_moderation_enabled = False
    
    def _init_image_moderation(self) -> None:
        """Initialize image safety checker."""
        try:
            self.safety_feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
            
            if torch.cuda.is_available():
                self.safety_checker = self.safety_checker.to("cuda")
            
            self.image_moderation_enabled = True
            print("✅ Image safety checker initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Image safety checker initialization failed: {e}")
            print("Image content moderation will be disabled.")
            self.safety_feature_extractor = None
            self.safety_checker = None
            self.image_moderation_enabled = False
    
    def check_text_safety(self, prompt: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a text prompt is safe using the OpenAI Moderation API.
        
        Args:
            prompt: The user-provided text prompt to check.
            
        Returns:
            A tuple containing:
            - bool: True if the prompt is safe, False otherwise.
            - dict: A dictionary with detailed category scores from the API.
        """
        if not self.text_moderation_enabled or not self.openai_client:
            print("⚠️ OpenAI client not initialized. Cannot perform text moderation.")
            # Fail safe: assume the prompt is safe if the service is unavailable
            return True, {"warning": "Text moderation unavailable"}
        
        if not prompt:
            return True, {}  # An empty prompt is safe
        
        try:
            response = self.openai_client.moderations.create(input=prompt)
            result = response.results[0]
            
            # The 'flagged' attribute gives a simple True/False answer
            is_flagged = result.flagged
            
            # Get individual category scores for granular control  
            category_scores = result.category_scores.model_dump()
            
            # For debugging and logging
            print(f"📊 Text moderation: Flagged={is_flagged}, Scores={category_scores}")
            
            return not is_flagged, category_scores
            
        except Exception as e:
            print(f"❌ Error calling OpenAI Moderation API: {e}")
            # In case of an API error, it's safer to block the content
            return False, {"error": str(e)}
    
    def _perform_safety_check(self, images: List[Image.Image]) -> Tuple[List[Image.Image], List[bool]]:
        """
        Internal method to perform safety check on images.
        
        Args:
            images: List of PIL images to check.
            
        Returns:
            Tuple of (filtered_images, has_nsfw_concepts)
        """
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt")
        
        if torch.cuda.is_available():
            safety_checker_input = safety_checker_input.to("cuda")
        
        image_tensor_list = [pil_to_tensor(img).unsqueeze(0) for img in images]
        
        filtered_images, has_nsfw_concept = self.safety_checker(
            images=torch.cat(image_tensor_list), 
            clip_input=safety_checker_input.pixel_values
        )
        
        # Convert back to PIL images
        filtered_images = [to_pil_image(filtered_images[i]) for i in range(filtered_images.shape[0])]
        
        return filtered_images, has_nsfw_concept
    
    def check_image_safety(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Check images for inappropriate content using StableDiffusionSafetyChecker.
        
        Args:
            images: List of PIL images to check.
            
        Returns:
            List of filtered images (NSFW content may be blurred/replaced).
        """
        if not self.image_moderation_enabled or self.safety_checker is None:
            print("⚠️ Image safety checker not available, returning images as-is")
            return images
        
        try:
            filtered_images, has_nsfw_concepts = self._perform_safety_check(images)
            
            # Log any flagged content
            flagged_count = sum(has_nsfw_concepts)
            if flagged_count > 0:
                print(f"🛡️ Image safety: {flagged_count}/{len(images)} images flagged")
            
            return filtered_images
            
        except Exception as e:
            print(f"❌ Error in image safety check: {e}")
            # On error, return original images
            return images
    
    def get_status(self) -> Dict[str, bool]:
        """
        Get the current status of moderation systems.
        
        Returns:
            Dictionary with moderation system statuses.
        """
        return {
            "text_moderation_enabled": self.text_moderation_enabled,
            "image_moderation_enabled": self.image_moderation_enabled,
            "openai_available": self.openai_client is not None,
            "safety_checker_available": self.safety_checker is not None
        }
    
    def get_safety_summary(self) -> List[str]:
        """
        Get a summary of enabled safety features for UI display.
        
        Returns:
            List of enabled safety feature names.
        """
        enabled_features = []
        
        if self.text_moderation_enabled:
            enabled_features.append("Text")
        
        if self.image_moderation_enabled:
            enabled_features.append("Image")
        
        return enabled_features


# Global instance for backward compatibility
_global_moderator = None


def get_content_moderator() -> ContentModerator:
    """
    Get the global ContentModerator instance.
    
    Returns:
        ContentModerator: The global moderator instance.
    """
    global _global_moderator
    if _global_moderator is None:
        _global_moderator = ContentModerator()
    return _global_moderator


# Convenience functions for backward compatibility
def is_prompt_safe_openai(prompt: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Legacy function for checking text safety.
    
    Args:
        prompt: Text prompt to check.
        
    Returns:
        Tuple of (is_safe, category_scores).
    """
    moderator = get_content_moderator()
    return moderator.check_text_safety(prompt)


def check_image_safety(images: List[Image.Image]) -> List[Image.Image]:
    """
    Legacy function for checking image safety.
    
    Args:
        images: List of PIL images to check.
        
    Returns:
        List of filtered images.
    """
    moderator = get_content_moderator()
    return moderator.check_image_safety(images)