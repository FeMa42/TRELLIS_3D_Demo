"""
Generation pipeline module for TRELLIS Streamlit application.

This module handles image and 3D model generation workflows including:
- Image generation with FLUX pipeline
- 3D model generation with TRELLIS pipeline
- Content filtering and quality scoring
- File management and exports

Usage:
    from modules.generation_pipeline import GenerationPipeline
    
    generator = GenerationPipeline()
    images = generator.generate_images("A dragon", num_images=4)
    glb_path, print_ready_glb_path, stl_path = generator.generate_3d_model(selected_image)
"""

import os
import gc
import tempfile
import random
import numpy as np
import torch
import imageio
from PIL import Image
from typing import List, Optional, Tuple
from trellis.utils import render_utils, postprocessing_utils
from trellis.utils.postprocessing_utils import to_glb_simple


class GenerationPipeline:
    """
    Handles image and 3D model generation workflows.
    """
    
    def __init__(self, flux_pipeline=None, trellis_pipeline=None, reward_model=None, content_moderator=None, max_batch_size: int = 4):
        """
        Initialize the GenerationPipeline.

        Args:
            flux_pipeline: Image generation pipeline (FLUX or Gemini) - kept for backward compatibility
            trellis_pipeline: TRELLIS 3D generation pipeline
            reward_model: Model for scoring image quality
            content_moderator: Content moderation system
            max_batch_size: Maximum number of images to generate in one batch (default: 4, optimized for L40S)
        """
        self.image_pipeline = flux_pipeline  # Rename internally
        self.trellis_pipeline = trellis_pipeline
        self.reward_model = reward_model
        self.content_moderator = content_moderator

        # Default generation parameters
        self.default_guidance_scale = 0.0
        self.default_num_inference_steps = 4
        self.default_height = 512
        self.default_width = 512
        self.default_max_sequence_length = 256
        self.max_batch_size = max_batch_size  # Configurable batch size limit
    
    def set_models(self, flux_pipeline=None, trellis_pipeline=None, reward_model=None, content_moderator=None):
        """
        Set the pipeline models.

        Args:
            flux_pipeline: Image generation pipeline (FLUX or Gemini) - kept for backward compatibility
            trellis_pipeline: TRELLIS 3D generation pipeline
            reward_model: Model for scoring image quality
            content_moderator: Content moderation system
        """
        if flux_pipeline is not None:
            self.image_pipeline = flux_pipeline  # Assign to image_pipeline
        if trellis_pipeline is not None:
            self.trellis_pipeline = trellis_pipeline
        if reward_model is not None:
            self.reward_model = reward_model
        if content_moderator is not None:
            self.content_moderator = content_moderator
    
    def _create_enhanced_prompt(self, base_prompt: str, custom_suffix: Optional[str] = None) -> str:
        """
        Enhance the user prompt with 3D printing optimizations.

        Args:
            base_prompt: Original user prompt
            custom_suffix: Optional custom suffix to append (uses default if None)

        Returns:
            str: Enhanced prompt optimized for 3D printing
        """
        # Default suffix for 3D printing optimization
        if custom_suffix is None:
            # custom_suffix = " Render of high quality 3D model on neutral background. Solid, contiguous mesh, optimized for 3D printing."
            custom_suffix = " Render of high quality 3D model on neutral background, optimized for 3D printing."

        return base_prompt + custom_suffix
    
    def _generate_seed(self, base_seed: Optional[int] = None) -> int:
        """
        Generate a random seed for reproducible results.

        Args:
            base_seed: Optional base seed for reproducibility

        Returns:
            int: Generated seed (returns base_seed directly if provided)
        """
        if base_seed is not None and base_seed > 0:
            return base_seed  # Use seed directly for reproducibility
        return random.randint(0, 999999)
    
    def generate_images(self,
                       prompt: str,
                       num_images: int = 4,
                       base_seed: Optional[int] = None,
                       guidance_scale: Optional[float] = None,
                       num_inference_steps: Optional[int] = None,
                       height: int = 512,
                       width: int = 512,
                       max_sequence_length: int = 256,
                       negative_prompt: Optional[str] = None,
                       prompt_suffix: Optional[str] = None) -> List[Image.Image]:
        """
        Generate images using the image generation pipeline (FLUX, Gemini, or Qwen).

        Args:
            prompt: Text description of desired object
            num_images: Number of images to generate
            base_seed: Optional seed for reproducibility
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            height: Image height
            width: Image width
            max_sequence_length: Maximum sequence length
            negative_prompt: Negative prompt (Qwen-specific, ignored by FLUX/Gemini)
            prompt_suffix: Custom prompt suffix for 3D printing optimization (uses default if None)

        Returns:
            List of generated PIL Images
        """
        if self.image_pipeline is None:
            raise ValueError("Image pipeline not loaded")

        # Generate seed
        actual_seed = self._generate_seed(base_seed)
        print(f"🎨 Generating images with seed: {actual_seed}")

        # Use provided parameters or defaults
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        num_inference_steps = num_inference_steps if num_inference_steps is not None else self.default_num_inference_steps

        # Limit batch size to prevent memory issues (configurable, default: 4)
        batch_size = min(num_images, self.max_batch_size)
        if num_images > batch_size:
            print(f"⚠️ Reducing number of images to {batch_size} due to batch size limit.")
            num_images = batch_size

        # Enhance prompt for 3D printing with custom or default suffix
        enhanced_prompt = self._create_enhanced_prompt(prompt, custom_suffix=prompt_suffix)

        # Generate images
        images = self.image_pipeline(
            [enhanced_prompt] * batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt,  # Qwen will use it, others will ignore
            generator=torch.Generator("cpu").manual_seed(actual_seed)
        ).images
        
        # Apply content filtering if available
        if self.content_moderator is not None:
            filtered_images = self.content_moderator.check_image_safety(images)
        else:
            filtered_images = images
        
        # Score and rank images if reward model is available
        if len(filtered_images) > 0 and self.reward_model is not None:
            try:
                rewards = self.reward_model.score(enhanced_prompt, filtered_images)
                top_indices = np.argsort(rewards)[-min(num_images, len(filtered_images)):]
                filtered_images = [filtered_images[i] for i in top_indices]
            except Exception as e:
                print(f"⚠️ Warning: Could not score images: {e}")
        
        print(f"✅ Generated {len(filtered_images)} filtered images")
        return filtered_images

    def generate_images_return_raw(self,
                       prompt: str,
                       num_images: int = 4,
                       base_seed: Optional[int] = None,
                       guidance_scale: Optional[float] = None,
                       num_inference_steps: Optional[int] = None,
                       height: int = 512,
                       width: int = 512,
                       max_sequence_length: int = 256,
                       negative_prompt: Optional[str] = None) -> List[Image.Image]:
        """
        Generate images using the image generation pipeline (returns both filtered and raw).

        Args:
            prompt: Text description of desired object
            num_images: Number of images to generate
            base_seed: Optional seed for reproducibility
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            height: Image height
            width: Image width
            max_sequence_length: Maximum sequence length
            negative_prompt: Negative prompt (Qwen-specific, ignored by FLUX/Gemini)

        Returns:
            Tuple of (filtered_images, raw_images)
        """
        if self.image_pipeline is None:
            raise ValueError("Image pipeline not loaded")

        # Generate seed
        actual_seed = self._generate_seed(base_seed)

        # Use provided parameters or defaults
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        num_inference_steps = num_inference_steps if num_inference_steps is not None else self.default_num_inference_steps

        # Limit batch size to prevent memory issues (configurable, default: 4)
        batch_size = min(num_images, self.max_batch_size)
        if num_images > batch_size:
            print(f"⚠️ Reducing number of images to {batch_size} due to batch size limit.")
            num_images = batch_size

        # Enhance prompt for 3D printing
        # enhanced_prompt = self._create_enhanced_prompt(prompt)
        enhanced_prompt = prompt 
        
        # Generate images
        images = self.image_pipeline(
            [enhanced_prompt] * batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt,  # Qwen will use it, others will ignore
            generator=torch.Generator("cpu").manual_seed(actual_seed)
        ).images
        
        # Apply content filtering if available
        if self.content_moderator is not None:
            filtered_images = self.content_moderator.check_image_safety(images)
        else:
            filtered_images = images
        
        # Score and rank images if reward model is available
        if len(filtered_images) > 0 and self.reward_model is not None:
            try:
                rewards = self.reward_model.score(enhanced_prompt, filtered_images)
                top_indices = np.argsort(rewards)[-min(num_images, len(filtered_images)):]
                filtered_images = [filtered_images[i] for i in top_indices]
            except Exception as e:
                print(f"⚠️ Warning: Could not score images: {e}")
        
        return filtered_images, images
    
    def generate_3d_model(self,
                         image: Image.Image,
                         base_seed: Optional[int] = None,
                         sparse_structure_steps: int = 24,
                         sparse_structure_cfg: float = 7.5,
                         slat_steps: int = 24,
                         slat_cfg: float = 3.0,
                         texture_size=1024) -> Tuple[str, str, str]:
        """
        Generate a 3D model from an image using the TRELLIS pipeline.

        Args:
            image: Input PIL Image
            base_seed: Optional seed for reproducibility
            sparse_structure_steps: Steps for sparse structure sampling
            sparse_structure_cfg: CFG strength for sparse structure
            slat_steps: Steps for slat sampling
            slat_cfg: CFG strength for slat sampling
            texture_size: Size for texture rendering (unused, kept for API compatibility)

        Returns:
            Tuple of (glb_path, print_ready_glb_path, stl_path)
        """
        if self.trellis_pipeline is None:
            raise ValueError("TRELLIS pipeline not loaded")
        
        # Generate seed
        actual_seed = self._generate_seed(base_seed)
        print(f"🔮 Generating 3D model with seed: {actual_seed}")
        
        # Run TRELLIS pipeline with device context
        # Set default device to match TRELLIS device for hardcoded .cuda() calls
        original_device = torch.cuda.current_device()
        
        # Get correct device for TRELLIS operations
        if (hasattr(self.trellis_pipeline, '_cpu_offload_enabled') and 
            self.trellis_pipeline._cpu_offload_enabled and 
            self.trellis_pipeline._offload_manager is not None):
            # Use execution device from offload manager when CPU offloading is active
            trellis_device = self.trellis_pipeline._offload_manager.execution_device
        else:
            # Use device of actual model parameters when no CPU offloading
            trellis_device = next(self.trellis_pipeline.models['sparse_structure_flow_model'].parameters()).device
        
        target_device_idx = trellis_device.index if hasattr(trellis_device, 'index') else 0
        
        try:
            torch.cuda.set_device(target_device_idx)
            outputs = self.trellis_pipeline.run(
                image,
                seed=actual_seed,
                formats=['mesh'],
                sparse_structure_sampler_params={
                    "steps": sparse_structure_steps,
                    "cfg_strength": sparse_structure_cfg,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_cfg,
                },
            )
        finally:
            # Restore original device
            torch.cuda.set_device(original_device)
        
        # Create temporary directory for outputs
        temp_dir = tempfile.mkdtemp()

        gc.collect()
        torch.cuda.empty_cache()

        # Export GLB using memory-efficient mesh-only path
        glb_path = os.path.join(temp_dir, "3d_model.glb")
        glb = to_glb_simple(
            outputs['mesh'][0],
            simplify=0.95,
            color=(180, 180, 220),  # Light blue color
            fill_holes=True,
            remove_floating=True,
            verbose=False
        )
        glb.export(glb_path)

        # Clean up outputs to free memory
        del outputs

        # Print-ready watertight remesh (env-gated) -> GLB (normals viewer) + STL (download).
        from modules.simple_stl_converter import export_print_ready
        print_ready_glb_path, stl_path = export_print_ready(glb_path, temp_dir)

        print(f"✅ 3D model generated: {glb_path}")
        return glb_path, print_ready_glb_path, stl_path
    
    def get_generation_info(self, seed: int) -> dict:
        """
        Get information about the generation parameters.
        
        Args:
            seed: Seed used for generation
            
        Returns:
            Dictionary with generation information
        """
        return {
            "seed": seed,
            "guidance_scale": self.default_guidance_scale,
            "num_inference_steps": self.default_num_inference_steps,
            "image_dimensions": f"{self.default_width}x{self.default_height}",
            "content_filtering_enabled": self.content_moderator is not None,
            "quality_scoring_enabled": self.reward_model is not None
        }

    @property
    def flux_pipeline(self):
        """Backward compatibility property."""
        return self.image_pipeline
    
    def validate_models(self) -> dict:
        """
        Validate that required models are available.
        
        Returns:
            Dictionary with model availability status
        """
        return {
            "image_pipeline_available": self.image_pipeline is not None,
            "flux_available": self.image_pipeline is not None,  # Backward compat
            "trellis_available": self.trellis_pipeline is not None,
            "reward_model_available": self.reward_model is not None,
            "content_moderator_available": self.content_moderator is not None,
            "image_generation_ready": self.image_pipeline is not None,
            "3d_generation_ready": self.trellis_pipeline is not None
        }


# Global instance for easy access
_global_generation_pipeline = None


def get_generation_pipeline() -> GenerationPipeline:
    """
    Get the global GenerationPipeline instance.

    Returns:
        GenerationPipeline: The global pipeline instance

    Note:
        Reads MAX_BATCH_SIZE environment variable if set (default: 4)
    """
    global _global_generation_pipeline
    if _global_generation_pipeline is None:
        # Read max batch size from environment variable (default: 4)
        max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "4"))
        _global_generation_pipeline = GenerationPipeline(max_batch_size=max_batch_size)
    return _global_generation_pipeline


# Convenience functions for backward compatibility
def generate_images(prompt: str, num_images: int = 4, base_seed: Optional[int] = None) -> List[Image.Image]:
    """
    Legacy function for image generation.
    
    Args:
        prompt: Text description
        num_images: Number of images to generate
        base_seed: Optional seed
        
    Returns:
        List of generated images
    """
    pipeline = get_generation_pipeline()
    return pipeline.generate_images(prompt, num_images, base_seed)


def generate_3d_model(image: Image.Image, base_seed: Optional[int] = None) -> Tuple[str, str, str]:
    """
    Legacy function for 3D model generation.

    Args:
        image: Input image
        base_seed: Optional seed

    Returns:
        Tuple of (glb_path, print_ready_glb_path, stl_path)
    """
    pipeline = get_generation_pipeline()
    return pipeline.generate_3d_model(image, base_seed)