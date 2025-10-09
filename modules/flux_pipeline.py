"""
FLUX.1 image generation pipeline adapter.

This module provides a wrapper around FLUX.1-Schnell and FLUX.1-dev diffusers pipelines
that standardizes the interface to match other image generation backends (Gemini, Qwen).

Usage:
    from modules.flux_pipeline import FLUXImagePipeline

    pipe = FLUXImagePipeline(device="cuda:0", enable_cpu_offload=True)
    result = pipe(["A sunset over mountains"], guidance_scale=0.0, height=512, width=512)
    images = result.images
"""

import os
import torch
from typing import Union, List, Optional
from collections import namedtuple
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig


class FLUXImagePipeline:
    """
    Adapter to make FLUX.1 pipelines consistent with other image generation backends.

    This self-contained class handles all FLUX-specific configuration including:
    - Model variant selection (Schnell vs Dev)
    - Quantization (4-bit/8-bit)
    - CPU offloading for memory efficiency
    - Optional torch.compile() optimization with persistent caching
    - Fast sampling mode via FAST_IMAGE_SAMPLING env var
    """

    def __init__(
        self,
        device: str = "cuda:0",
        enable_cpu_offload: bool = True,
        use_fast_sampling: bool = False,
        enable_compilation: bool = False
    ):
        """
        Initialize FLUX Image Pipeline.

        Args:
            device: Target device (e.g., "cuda:0")
            enable_cpu_offload: Enable model CPU offloading for memory efficiency
            use_fast_sampling: Enable fast sampling mode (uses Schnell, fewer steps)
            enable_compilation: Enable torch.compile() optimization with caching

        Environment Variables:
            USE_FLUX_DEV: Use FLUX.1-dev instead of Schnell (default: false)
            FAST_IMAGE_SAMPLING: Override use_fast_sampling (default: false)
            ENABLE_IMAGE_CPU_OFFLOAD: Override enable_cpu_offload (default: from param)
            COMPILE_FLUX_OPTIMIZATION: Override enable_compilation (default: false)

        Note:
            - Fast sampling always uses Schnell (overrides USE_FLUX_DEV)
            - Peak VRAM with quantization: ~16-24GB
            - CPU offloading reduces VRAM by 50-70%
        """
        self.device = device

        # Fast sampling mode (env var overrides parameter)
        self.use_fast_sampling = os.environ.get("FAST_IMAGE_SAMPLING", "false").lower() == "true" or use_fast_sampling

        # Model selection: fast sampling always uses Schnell
        if self.use_fast_sampling:
            self.model_name = "black-forest-labs/FLUX.1-Schnell"
            print("🚀 Fast sampling enabled: Using FLUX.1-Schnell")
        elif os.environ.get("USE_FLUX_DEV", "false").lower() == "true":
            self.model_name = "black-forest-labs/FLUX.1-dev"
            print("🔧 Using FLUX.1-dev model")
        else:
            self.model_name = "black-forest-labs/FLUX.1-Schnell"

        # CPU offloading (env var can override)
        enable_offload_env = os.environ.get("ENABLE_IMAGE_CPU_OFFLOAD") or os.environ.get("ENABLE_FLUX_CPU_OFFLOAD")
        if enable_offload_env:
            self.enable_cpu_offload = enable_offload_env.lower() == "true"
        else:
            self.enable_cpu_offload = enable_cpu_offload

        # Only enable CPU offload on cuda:0 or cuda (not on multi-GPU secondary device)
        self.use_cpu_offload = (device == "cuda:0" or device == "cuda") and self.enable_cpu_offload

        # Compilation optimization (env var can override)
        compile_env = os.environ.get("COMPILE_FLUX_OPTIMIZATION", "false").lower() == "true"
        self.enable_compilation = compile_env or enable_compilation

        # Setup compilation cache if enabled
        if self.enable_compilation:
            self._setup_compilation_cache()

        # Setup torch configuration for optimal performance
        self._setup_torch_config()

        # Load the pipeline
        print(f"🔧 Loading FLUX pipeline on {device}...")
        self.pipe = self._load_pipeline()

        # Track generation statistics
        self.total_images_generated = 0
        self.total_batches = 0

        print("✅ FLUX pipeline loaded successfully!")

    def _setup_torch_config(self) -> None:
        """Setup torch configuration for optimal performance."""
        torch._dynamo.config.cache_size_limit = 1000
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

    def _setup_compilation_cache(self):
        """Configure persistent compilation caching using environment variables."""
        self.current_dir = os.getcwd()
        self.compilation_cache_dir = os.path.join(self.current_dir, ".torch_compile_cache")

        # Create cache directory
        os.makedirs(self.compilation_cache_dir, exist_ok=True)

        # Set caching environment variables
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = self.compilation_cache_dir
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        os.environ["TORCH_COMPILE_CACHE_DIR"] = self.compilation_cache_dir

        # Triton cache directory (used by inductor)
        triton_cache_dir = os.path.join(self.compilation_cache_dir, "triton")
        os.makedirs(triton_cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

        # Additional optimizations
        os.environ["TORCH_COMPILE_DEBUG"] = "0"

        print(f"   Compilation cache enabled: {self.compilation_cache_dir}")

    def clear_compilation_cache(self):
        """Clear the compilation cache (useful after PyTorch updates)."""
        if not hasattr(self, 'compilation_cache_dir'):
            print("⚠️ Compilation cache not enabled")
            return

        import shutil
        if os.path.exists(self.compilation_cache_dir):
            try:
                shutil.rmtree(self.compilation_cache_dir)
                # Recreate empty directory
                os.makedirs(self.compilation_cache_dir, exist_ok=True)
                print("✅ Compilation cache cleared")
            except Exception as e:
                print(f"⚠️ Could not clear cache: {e}")

    def get_cache_info(self) -> dict:
        """Get information about the compilation cache."""
        if not hasattr(self, 'compilation_cache_dir'):
            return {"enabled": False, "exists": False, "size_mb": 0, "num_files": 0}

        if not os.path.exists(self.compilation_cache_dir):
            return {"enabled": True, "exists": False, "size_mb": 0, "num_files": 0}

        total_size = 0
        num_files = 0

        for root, _, files in os.walk(self.compilation_cache_dir):
            num_files += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except Exception:
                    pass

        return {
            "enabled": True,
            "exists": True,
            "size_mb": total_size / (1024 * 1024),
            "num_files": num_files,
            "path": self.compilation_cache_dir
        }

    def _create_quantization_config(self, use_4bit: bool = True) -> PipelineQuantizationConfig:
        """Create quantization configuration for FLUX pipeline."""
        if use_4bit:
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                components_to_quantize=["transformer", "text_encoder_2"],
            )
        else:
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={
                    "load_in_8bit": True
                },
                components_to_quantize=["transformer", "text_encoder_2"],
            )

    def _load_pipeline(self) -> DiffusionPipeline:
        """
        Load and optimize the FLUX pipeline.

        Returns:
            DiffusionPipeline: Configured FLUX pipeline
        """
        # Create quantization config
        pipeline_quant_config = self._create_quantization_config()

        # Load base pipeline
        flux_pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=pipeline_quant_config,
            torch_dtype=torch.bfloat16,
        )

        # Apply optimization strategy
        if self.use_cpu_offload:
            # CPU offloading for maximum memory efficiency (no compilation)
            flux_pipe.enable_model_cpu_offload()
            # Enable attention slicing for additional VRAM savings
            print("   Enabling CPU offload and attention slicing for VRAM optimization...")
            flux_pipe.enable_attention_slicing()
        else:
            # Move to device directly
            flux_pipe = flux_pipe.to(self.device)

        # Apply compilation if enabled
        if self.enable_compilation:
            print("   Compiling transformer for maximum speed...")
            flux_pipe.transformer.compile()

        return flux_pipe

    def warmup(self) -> None:
        """
        Warm up the FLUX pipeline with a test generation.

        This pre-compiles CUDA kernels and initializes caches for faster
        subsequent generations.
        """
        print("🔥 Warming up FLUX pipeline...")

        # Get generation config
        config = self.get_generation_config()

        batch_size = 4
        prompt = "A Goblin riding a Roomba vacuum cleaner into battle. High Quality Render of 3/4 front view of the 3D object, studio lighting, clean background."
        prompts = [prompt] * batch_size

        # Determine device for generator (CPU is diffusers best practice)
        generator_device = "cpu"

        # Warm-up generation
        _ = self.pipe(
            prompts,
            guidance_scale=config["guidance_scale"],
            height=512,
            width=512,
            num_inference_steps=config["num_inference_steps"],
            max_sequence_length=256,
            generator=torch.Generator(generator_device).manual_seed(42)
        ).images

        print("✅ FLUX warmup complete")

    def __call__(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: Optional[float] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        max_sequence_length: int = 256,
        negative_prompt: Optional[str] = None,
        generator=None
    ):
        """
        Generate images matching standard pipeline interface.

        Args:
            prompt: Text prompt(s) for image generation
            guidance_scale: Guidance scale (CFG). Defaults based on model:
                          - Schnell: 0.0 (distilled, no CFG)
                          - Dev: 4.5 (needs guidance)
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps. Defaults based on model:
                               - Schnell: 4 steps (fast)
                               - Dev: 28 steps (quality)
            max_sequence_length: Maximum prompt token length
            negative_prompt: Negative prompt (ignored by FLUX, for compatibility with Qwen)
            generator: Torch generator for seed control

        Returns:
            namedtuple with .images attribute containing list of PIL Images
        """
        # Get defaults if not provided
        config = self.get_generation_config()
        if guidance_scale is None:
            guidance_scale = config["guidance_scale"]
        if num_inference_steps is None:
            num_inference_steps = config["num_inference_steps"]

        # Normalize prompt to list for consistent handling
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Generate images
        result = self.pipe(
            prompts,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        )

        # Update statistics
        self.total_images_generated += len(result.images)
        self.total_batches += 1

        # Return result (already in correct format)
        return result

    def get_generation_config(self) -> dict:
        """
        Get default generation configuration for the active FLUX model.

        Returns:
            Dictionary with generation parameters (guidance_scale, num_inference_steps, etc.)
        """
        # Compute parameters based on model type
        if self.model_name == "black-forest-labs/FLUX.1-Schnell":
            guidance_scale = 0.0  # Distilled model, no CFG
            num_inference_steps = 4  # Fast generation
        else:  # FLUX.1-dev
            guidance_scale = 4.5  # Needs guidance
            num_inference_steps = 28  # Higher quality

        return {
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "model_name": self.model_name,
            "fast_sampling": self.use_fast_sampling,
            "cpu_offload": self.use_cpu_offload,
            "compilation_enabled": self.enable_compilation
        }

    def get_stats(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "total_images_generated": self.total_images_generated,
            "total_batches": self.total_batches,
            "model_name": self.model_name,
            "device": self.device,
            "cpu_offload_enabled": self.use_cpu_offload,
            "fast_sampling_enabled": self.use_fast_sampling,
            "compilation_enabled": self.enable_compilation,
            **self.get_generation_config()
        }

    def to(self, device):
        """
        Compatibility method for .to(device) calls.

        Args:
            device: Target device

        Returns:
            self for method chaining
        """
        if not self.use_cpu_offload:
            self.pipe = self.pipe.to(device)
            self.device = device
        return self

    def enable_model_cpu_offload(self):
        """
        Compatibility method for CPU offloading.

        Note: This is handled during initialization. This method is a no-op
        provided for API compatibility.
        """
        if not self.use_cpu_offload:
            print("⚠️ CPU offload not enabled during initialization. Restart pipeline to enable.")


def load_flux_pipeline(
    device: str = "cuda:0",
    enable_cpu_offload: bool = True,
    use_fast_sampling: bool = False,
    enable_compilation: bool = False
) -> FLUXImagePipeline:
    """
    Convenience function to load FLUX pipeline with common settings.

    Args:
        device: Target device
        enable_cpu_offload: Enable CPU offloading
        use_fast_sampling: Enable fast sampling mode
        enable_compilation: Enable torch.compile() optimization

    Returns:
        Initialized FLUXImagePipeline
    """
    return FLUXImagePipeline(
        device=device,
        enable_cpu_offload=enable_cpu_offload,
        use_fast_sampling=use_fast_sampling,
        enable_compilation=enable_compilation
    )
