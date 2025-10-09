"""
Qwen/Qwen-Image pipeline adapter.

This module provides a wrapper around Qwen/Qwen-Image diffusers pipeline
that standardizes the interface to match FLUX and Gemini pipelines.

Usage:
    from modules.qwen_pipeline import QwenImagePipeline

    pipe = QwenImagePipeline(device="cuda:0", enable_cpu_offload=True)
    result = pipe("A sunset over mountains", guidance_scale=5.5, height=512, width=512)
    images = result.images
"""

import os
import torch
from typing import Union, List, Optional
from collections import namedtuple
from diffusers import DiffusionPipeline


class QwenImagePipeline:
    """
    Adapter to make Qwen/Qwen-Image pipeline consistent with FLUX/Gemini interface.

    This allows Qwen to be used as a drop-in replacement in the existing codebase
    with parameter translation (guidance_scale → true_cfg_scale).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Image",
        device: str = "cuda:0",
        enable_cpu_offload: bool = True,
        load_in_8bit: bool = True,
        torch_dtype = torch.bfloat16,
        use_fast_sampling: bool = False,
        batch_size: int = 4
    ):
        """
        Initialize Qwen Image Pipeline.

        Args:
            model_name: Qwen model identifier (default: "Qwen/Qwen-Image")
            device: Target device (e.g., "cuda:0")
            enable_cpu_offload: Enable model CPU offloading for memory efficiency
            load_in_8bit: Enable 8-bit quantization to reduce VRAM usage
            torch_dtype: PyTorch data type (default: bfloat16)
            use_fast_sampling: Enable fast sampling with Lightning LoRA (4 steps)
            batch_size: Batch size for parallel image generation (default: 4)

        Note:
            - Requires trellis_gemma environment (newer diffusers/transformers)
            - Peak VRAM usage: ~40GB (without 8-bit), ~25-30GB (with 8-bit)
            - CPU offloading significantly reduces VRAM requirements
            - Fast sampling: Uses Lightning LoRA for 10x speedup (40 → 4 steps)
            - Batch generation: Processes multiple prompts in parallel for ~4x speedup
        """
        self.model_name = model_name
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.torch_dtype = torch_dtype
        self.use_fast_sampling = use_fast_sampling
        self.batch_size = batch_size

        print(f"🔧 Loading Qwen Image pipeline from {model_name}...")
        print(f"   Device: {device}")
        print(f"   8-bit quantization: {load_in_8bit}")
        print(f"   CPU offloading: {enable_cpu_offload}")
        print(f"   Fast sampling: {use_fast_sampling}")

        # Load pipeline with configuration
        # Note: load_in_8bit keyword is passed but may be ignored by some pipeline versions
        # The pipeline will use the most memory-efficient configuration available
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                load_in_8bit=load_in_8bit  # May be ignored depending on pipeline version
            )
        except TypeError as e:
            # If load_in_8bit is not supported, load without it
            print(f"⚠️ load_in_8bit not supported, loading normally: {e}")
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )

        # Apply CPU offloading if requested
        if enable_cpu_offload:
            print("   Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload()
            # Enable attention slicing for additional VRAM savings
            print("   Enabling attention slicing for VRAM optimization...")
            self.pipe.enable_attention_slicing()
        else:
            # Move to device directly
            self.pipe = self.pipe.to(device)

        # Load Lightning LoRA for fast sampling
        if use_fast_sampling:
            print("   Loading Lightning LoRA for fast sampling...")
            try:
                self.pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                    weight_name="Qwen-Image-Lightning-4steps-V1.0.safetensors"
                )
                print("   ✅ Lightning LoRA loaded successfully!")
            except Exception as e:
                print(f"   ⚠️ Failed to load Lightning LoRA: {e}")
                print("   ⚠️ Continuing with normal sampling mode")
                self.use_fast_sampling = False

        # Set generation configs based on sampling mode
        if self.use_fast_sampling:
            self.default_cfg_scale = 2.0
            self.default_num_steps = 4
        else:
            self.default_cfg_scale = 5.5
            self.default_num_steps = 40

        print("✅ Qwen pipeline loaded successfully!")

        # Track generation statistics
        self.total_images_generated = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: Optional[float] = None,  # Maps to true_cfg_scale in Qwen
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        max_sequence_length: int = 256,  # Ignored for Qwen
        generator = None,  # Seed control
        negative_prompt: Optional[str] = None
    ):
        """
        Generate images matching FLUX/Gemini pipeline interface.

        Args:
            prompt: Text prompt(s) for image generation
            guidance_scale: Guidance scale (maps to true_cfg_scale for Qwen).
                          Defaults to 2.0 (fast) or 5.5 (normal) based on sampling mode.
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps.
                               Defaults to 4 (fast) or 40 (normal) based on sampling mode.
            max_sequence_length: Ignored (for FLUX compatibility)
            generator: Torch generator for seed control
            negative_prompt: Negative prompt (Qwen-specific, defaults to safety prompt)

        Returns:
            namedtuple with .images attribute containing list of PIL Images
        """
        # Use defaults based on sampling mode if not provided
        if guidance_scale is None:
            guidance_scale = self.default_cfg_scale
        if num_inference_steps is None:
            num_inference_steps = self.default_num_steps

        # Normalize prompt to list for batch processing
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Default negative prompt for content safety
        if negative_prompt is None:
            negative_prompt = "adult content, nsfw, inappropriate"

        all_images = []

        # Process prompts in batches for better GPU utilization
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]

            # Pad batch to fixed size if needed
            original_batch_size = len(batch_prompts)
            while len(batch_prompts) < self.batch_size:
                batch_prompts.append(batch_prompts[-1])  # Duplicate last prompt to fill batch

            # Clear CUDA cache before batch
            torch.cuda.empty_cache()

            try:
                # Generate batch with torch.no_grad() for memory efficiency
                with torch.no_grad():
                    result = self.pipe(
                        prompt=batch_prompts,  # ← Pass list of prompts for batch generation!
                        negative_prompt=[negative_prompt] * self.batch_size,  # ← List of negative prompts
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        true_cfg_scale=guidance_scale,  # ← Key parameter mapping
                        generator=generator
                    )

                # Extract images (only keep original batch size, drop padding)
                if hasattr(result, 'images') and len(result.images) > 0:
                    batch_images = result.images[:original_batch_size]
                    all_images.extend(batch_images)
                    self.total_images_generated += len(batch_images)
                else:
                    print(f"⚠️ No images generated for batch {i//self.batch_size + 1}")

            except Exception as e:
                print(f"⚠️ Qwen batch generation error (batch {i//self.batch_size + 1}): {e}")
                # Continue with next batch instead of failing completely

        # Return result matching FLUX/Gemini format
        Result = namedtuple('GenerationResult', ['images'])
        return Result(images=all_images)

    def get_stats(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "total_images_generated": self.total_images_generated,
            "model_name": self.model_name,
            "device": self.device,
            "cpu_offload_enabled": self.enable_cpu_offload,
            "fast_sampling_enabled": self.use_fast_sampling,
            "default_cfg_scale": self.default_cfg_scale,
            "default_num_steps": self.default_num_steps,
            "batch_size": self.batch_size
        }

    def get_generation_config(self) -> dict:
        """
        Get default generation configuration for Qwen.

        Returns:
            Dictionary with generation parameters
        """
        return {
            "guidance_scale": self.default_cfg_scale,
            "num_inference_steps": self.default_num_steps,
            "model_name": self.model_name,
            "fast_sampling": self.use_fast_sampling,
            "cpu_offload": self.enable_cpu_offload,
            "batch_size": self.batch_size
        }

    def warmup(self) -> None:
        """
        Warm up the Qwen pipeline with a test generation.

        This pre-compiles CUDA kernels and initializes caches for faster
        subsequent generations.
        """
        print("🔥 Warming up Qwen pipeline...")

        # Get generation config
        config = self.get_generation_config()

        prompt = "A Goblin riding a Roomba vacuum cleaner into battle. High Quality Render of 3/4 front view of the 3D object, studio lighting, clean background."

        # Qwen warmup generation
        # Uses dynamic parameters from get_generation_config() (respects FAST_IMAGE_SAMPLING)
        _ = self(
            prompt,
            guidance_scale=config["guidance_scale"],  # Dynamic: 2.0 (fast) or 5.5 (normal)
            height=512,
            width=512,
            num_inference_steps=config["num_inference_steps"],  # Dynamic: 4 (fast) or 40 (normal)
            generator=torch.Generator("cpu").manual_seed(42)
        ).images

        print("✅ Qwen warmup complete")

    def to(self, device):
        """
        Compatibility method for .to(device) calls.

        Args:
            device: Target device

        Returns:
            self for method chaining
        """
        if not self.enable_cpu_offload:
            self.pipe = self.pipe.to(device)
            self.device = device
        return self

    def enable_model_cpu_offload(self):
        """
        Compatibility method for CPU offloading.

        Note: This is handled during initialization. This method is a no-op
        provided for API compatibility.
        """
        if not self.enable_cpu_offload:
            print("⚠️ CPU offload not enabled during initialization. Restart pipeline to enable.")


def load_qwen_pipeline(
    device: str = "cuda:0",
    enable_cpu_offload: bool = True,
    load_in_8bit: bool = True,
    use_fast_sampling: bool = False,
    batch_size: int = 4
) -> QwenImagePipeline:
    """
    Convenience function to load Qwen pipeline with common settings.

    Args:
        device: Target device
        enable_cpu_offload: Enable CPU offloading
        load_in_8bit: Enable 8-bit quantization
        use_fast_sampling: Enable fast sampling with Lightning LoRA
        batch_size: Batch size for parallel image generation (default: 4)

    Returns:
        Initialized QwenImagePipeline
    """
    model_name = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image")

    return QwenImagePipeline(
        model_name=model_name,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
        load_in_8bit=load_in_8bit,
        use_fast_sampling=use_fast_sampling,
        batch_size=batch_size
    )
