"""
Model management module for TRELLIS Streamlit application.

This module handles loading and caching of AI models including:
- FLUX.1-Schnell image generation pipeline
- TRELLIS image-to-3D pipeline  
- Image-based prompt evaluator reward model
- Model optimization (quantization, compilation, warming)

Usage:
    from modules.model_manager import ModelManager
    
    manager = ModelManager()
    manager.load_all_models()
    flux_pipe = manager.get_flux_pipeline()
    trellis_pipe = manager.get_trellis_pipeline()
"""
import os
import torch
import streamlit as st
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.eval_utils import ImageBasedPromptEvaluator
from typing import Optional, Tuple, Dict, Any
import gc

# TRELLIS_MODEL_ID = "JeffreyXiang/TRELLIS-image-large"
# load TRELLIS_MODEL_ID from environment variable if set, otherwise use default
TRELLIS_MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "JeffreyXiang/TRELLIS-image-large")

class ModelManager:
    """
    Manages AI model loading, caching, and optimization.
    """
    
    def __init__(self, device_config=None, enable_trellis_cpu_offload=True, image_model="flux"):
        """Initialize the ModelManager.

        Args:
            device_config: Dict with device placement e.g.
                          {"flux": "cuda:0", "trellis": "cuda:1"} for dual GPU
                          or None for single GPU (default)
            enable_trellis_cpu_offload: Enable CPU offloading for TRELLIS models
            image_model: Image generation model ('flux' or 'gemini')
        """
        self.image_model = image_model.lower()
        self.image_pipeline = None  # Will hold FLUX or Gemini pipeline
        self.flux_pipeline = None  # Deprecated, kept for backward compatibility
        self.trellis_pipeline = None
        self.reward_model = None
        self.guidance_scale = 0.0
        self.num_inference_steps = 4
        
        # Device configuration with safety checks
        self.device_config = self._validate_device_config(device_config)
        self.flux_device = self.device_config.get("flux", "cuda:0")
        self.trellis_device = self.device_config.get("trellis", "cuda:0")
        
        # Multi-GPU and TRELLIS CPU offloading are mutually exclusive
        if self.flux_device != self.trellis_device and enable_trellis_cpu_offload:
            print("⚠️ Multi-GPU mode detected: Disabling TRELLIS CPU offloading (incompatible)")
            print("   Multi-GPU placement and CPU offloading cannot be used together for TRELLIS")
            print(f"   TRELLIS will use dedicated GPU: {self.trellis_device}")
            self.enable_trellis_cpu_offload = False
        else:
            self.enable_trellis_cpu_offload = enable_trellis_cpu_offload
        
        # Model configuration (only for FLUX)
        if self.image_model == "flux":
            if os.environ.get("USE_FLUX_DEV", "false").lower() == "true":
                self.flux_model_name = "black-forest-labs/FLUX.1-dev"
            else:
                self.flux_model_name = "black-forest-labs/FLUX.1-Schnell"
            self._setup_torch_config()
        elif self.image_model == "gemini":
            # Gemini doesn't need model configuration
            self.flux_model_name = None
        else:
            raise ValueError(f"Unknown image_model: {self.image_model}. Must be 'flux' or 'gemini'")

        # CPU offloading configuration with mutual exclusivity logic (FLUX only)
        if self.image_model == "flux":
            self.enable_flux_cpu_offload = os.environ.get("ENABLE_FLUX_CPU_OFFLOAD", "false").lower() == "true"
            self.use_flux_cpu_offload = (self.flux_device == "cuda:0" or self.flux_device == "cuda") and self.enable_flux_cpu_offload
        else:
            self.enable_flux_cpu_offload = False
            self.use_flux_cpu_offload = False

        # Reward model configuration (optional, disabled by default to save VRAM)
        self.enable_reward_model = os.environ.get("ENABLE_REWARD_MODEL", "false").lower() == "true"

        if os.environ.get("COMPILE_FLUX_OPTIMIZATION", "false").lower() == "true":
            # Compilation cache settings
            self.current_dir = os.getcwd()
            self.compilation_cache_dir = os.path.join(self.current_dir, ".torch_compile_cache")
            self.enable_compilation_cache = True
            self._setup_compilation_cache()
    
    def _setup_compilation_cache(self):
        """Configure persistent compilation caching using environment variables."""
        if not self.enable_compilation_cache:
            return
        
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

    def clear_compilation_cache(self):
        """Clear the compilation cache (useful after PyTorch updates)."""
        import shutil
        if os.path.exists(self.compilation_cache_dir):
            try:
                shutil.rmtree(self.compilation_cache_dir)
                # Recreate empty directory
                os.makedirs(self.compilation_cache_dir, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Could not clear cache: {e}")

    def get_cache_info(self) -> dict:
        """Get information about the compilation cache."""
        if not os.path.exists(self.compilation_cache_dir):
            return {"exists": False, "size_mb": 0, "num_files": 0}
        
        total_size = 0
        num_files = 0
        
        for root, dirs, files in os.walk(self.compilation_cache_dir):
            num_files += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except:
                    pass
        
        return {
            "exists": True,
            "size_mb": total_size / (1024 * 1024),
            "num_files": num_files,
            "path": self.compilation_cache_dir
        }

    def _validate_device_config(self, device_config) -> Dict[str, str]:
        """
        Validate device configuration based on available hardware.
        
        Args:
            device_config: Requested device configuration
            
        Returns:
            Dict with validated device configuration
        """
        # Default to single GPU
        default_config = {"flux": "cuda:0", "trellis": "cuda:0"}
        
        if device_config is None:
            return default_config
        
        # Check if we have enough GPUs for multi-GPU setup
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"⚠️ Only {num_gpus} GPU(s) available, using single GPU mode")
            return default_config
        
        flux_device = device_config.get("flux", "cuda:0")
        trellis_device = device_config.get("trellis", "cuda:0")
        
        # Parse device indices
        try:
            flux_idx = int(flux_device.split(":")[1]) if ":" in flux_device else 0
            trellis_idx = int(trellis_device.split(":")[1]) if ":" in trellis_device else 0
        except (ValueError, IndexError):
            print("⚠️ Invalid device specification, using single GPU mode")
            return default_config
        
        # Check if requested devices exist
        if flux_idx >= num_gpus or trellis_idx >= num_gpus:
            print(f"⚠️ Requested GPU index exceeds available GPUs ({num_gpus}), using single GPU mode")
            return default_config
        
        return device_config
    
    def _setup_torch_config(self) -> None:
        """Setup torch configuration for optimal performance."""
        torch._dynamo.config.cache_size_limit = 1000
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
    
    def _setup_flux_parameters(self) -> None:
        """Configure FLUX pipeline parameters based on model type."""
        if self.flux_model_name == "black-forest-labs/FLUX.1-Schnell":
            self.guidance_scale = 0.0
            self.num_inference_steps = 4
        else:
            self.guidance_scale = 4.5
            self.num_inference_steps = 28
    
    def _create_quantization_config(self, use_4bit=True) -> PipelineQuantizationConfig:
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

    def _load_gemini_pipeline(self):
        """
        Load Gemini 2.5 Flash Image pipeline.

        Returns:
            GeminiImagePipeline: Configured Gemini pipeline
        """
        print("🔧 Loading Gemini 2.5 Flash Image pipeline...")

        from modules.gemini_pipeline import GeminiImagePipeline

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set.\n"
                "Get your API key from: https://aistudio.google.com\n"
                "Then set it with: export GEMINI_API_KEY='your-key'"
            )

        pipeline = GeminiImagePipeline(api_key=api_key)
        print("✅ Gemini pipeline loaded successfully!")
        return pipeline

    def _load_image_pipeline(self):
        """
        Load image generation pipeline based on selected model.

        Returns:
            Image pipeline (FLUX or Gemini)
        """
        if self.image_model == "flux":
            return self._load_flux_pipeline()
        elif self.image_model == "gemini":
            return self._load_gemini_pipeline()
        else:
            raise ValueError(f"Unknown image_model: {self.image_model}")

    def _load_flux_pipeline(self) -> DiffusionPipeline:
        """
        Load and optimize the FLUX image generation pipeline.
        
        Returns:
            DiffusionPipeline: Configured FLUX pipeline
        """
        print(f"🔧 Loading FLUX pipeline on {self.flux_device}...")

        # Setup parameters
        self._setup_flux_parameters()
        
        # Create quantization config
        pipeline_quant_config = self._create_quantization_config()

        # Load base pipeline
        flux_pipe = DiffusionPipeline.from_pretrained(
            self.flux_model_name,
            quantization_config=pipeline_quant_config,
            torch_dtype=torch.bfloat16,
        )
        
        # Apply optimization strategy
        if self.use_flux_cpu_offload:
            # CPU offloading for maximum memory efficiency (no compilation)
            flux_pipe.enable_model_cpu_offload()
        else:
            # Compilation for maximum speed (no CPU offloading)
            flux_pipe = flux_pipe.to(self.flux_device)
        
        if os.environ.get("COMPILE_FLUX_OPTIMIZATION", "false").lower() == "true":
            # flux_pipe = torch.jit.optimize(flux_pipe)
            flux_pipe.transformer.compile()
        return flux_pipe
    
    def _warm_up_flux_pipeline(self, flux_pipe: DiffusionPipeline) -> None:
        """
        Warm up the FLUX pipeline with a test generation.
        
        Args:
            flux_pipe: The FLUX pipeline to warm up
        """
        print("🔥 Warming up FLUX pipeline...")
        
        batch_size = 4
        prompt = "A Goblin riding a Roomba vacuum cleaner into battle. High Quality Render of 3/4 front view of the 3D object, studio lighting, clean background."
        prompts = [prompt] * batch_size
        
        # Determine device for generator
        if self.use_flux_cpu_offload:
            # CPU offload always uses cuda:0, but generator should be on CPU for diffusers
            generator_device = "cpu"
        else:
            # For manual placement, use CPU generator (diffusers best practice)
            generator_device = "cpu"
        
        # Warm-up generation
        _ = flux_pipe(
            prompts,
            guidance_scale=self.guidance_scale, 
            height=512, 
            width=512,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=256,
            generator=torch.Generator(generator_device).manual_seed(42)
        ).images
    
    def _load_reward_model(self) -> Optional[Any]:
        """
        Load the image-based prompt evaluator reward model.

        Returns:
            Reward model for image quality evaluation, or None if disabled
        """
        if not self.enable_reward_model:
            print("ℹ️ Reward model disabled (saves ~1GB VRAM). Set ENABLE_REWARD_MODEL=true to enable image quality ranking.")
            return None

        print("🔧 Loading reward model (ImageReward + CLIP)...")
        image_based_prompt_evaluator = ImageBasedPromptEvaluator()
        reward_model = image_based_prompt_evaluator.reward_model
        return reward_model
    
    def _load_trellis_pipeline(self) -> TrellisImageTo3DPipeline:
        """
        Load the TRELLIS image-to-3D pipeline.
        
        Returns:
            TrellisImageTo3DPipeline: Configured TRELLIS pipeline
        """
        print(f"🔧 Loading TRELLIS pipeline on {self.trellis_device}...")
        
        # Set default CUDA device to ensure hardcoded .cuda() calls go to correct device
        original_device = torch.cuda.current_device()
        target_device_idx = int(self.trellis_device.split(':')[1]) if ':' in self.trellis_device else 0
        
        try:
            torch.cuda.set_device(target_device_idx)
            pipeline = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_ID)
            pipeline.to(self.trellis_device)
            
            # Enable CPU offloading if requested
            if self.enable_trellis_cpu_offload:
                pipeline.enable_model_cpu_offload(execution_device=self.trellis_device)
                
        finally:
            # Restore original device
            torch.cuda.set_device(original_device)
        return pipeline
    
    def load_all_models(self) -> Tuple[Any, TrellisImageTo3DPipeline, Optional[Any]]:
        """
        Load all AI models with progress indication.

        Returns:
            Tuple of (image_pipeline, trellis_pipeline, reward_model)
            Note: reward_model may be None if disabled via ENABLE_REWARD_MODEL env var
        """
        model_name = self.image_model.upper()

        with st.spinner(f"🚀 Loading {model_name} Image Model... {'(1-2 minutes)' if self.image_model == 'flux' else '(instant)'}"):
            # Load image generation pipeline
            self.image_pipeline = self._load_image_pipeline()

            # Warm up only for FLUX
            if self.image_model == "flux":
                self._warm_up_flux_pipeline(self.image_pipeline)

            # Load reward model (optional)
            self.reward_model = self._load_reward_model()

        with st.spinner("🔧 Loading 3D Model Generator..."):
            # Load TRELLIS pipeline
            self.trellis_pipeline = self._load_trellis_pipeline()

        # Set flux_pipeline for backward compatibility
        self.flux_pipeline = self.image_pipeline

        return self.image_pipeline, self.trellis_pipeline, self.reward_model
    
    def get_image_pipeline(self):
        """
        Get the image generation pipeline instance.

        Returns:
            Image pipeline (FLUX or Gemini) or None if not loaded
        """
        return self.image_pipeline

    def get_flux_pipeline(self) -> Optional[DiffusionPipeline]:
        """
        Get the FLUX pipeline instance (backward compatibility).
        Actually returns image_pipeline regardless of backend.

        Returns:
            Image pipeline or None if not loaded
        """
        return self.image_pipeline

    def get_trellis_pipeline(self) -> Optional[TrellisImageTo3DPipeline]:
        """
        Get the TRELLIS pipeline instance.
        
        Returns:
            TrellisImageTo3DPipeline or None if not loaded
        """
        return self.trellis_pipeline
    
    def get_reward_model(self) -> Optional[Any]:
        """
        Get the reward model instance.
        
        Returns:
            Reward model or None if not loaded
        """
        return self.reward_model
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get the current generation configuration.
        
        Returns:
            Dictionary with generation parameters
        """
        return {
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "flux_model_name": self.flux_model_name
        }
    
    def are_models_loaded(self) -> bool:
        """
        Check if all required models are loaded.
        Note: reward_model is optional and not required.

        Returns:
            bool: True if all required models are loaded
        """
        return (
            self.image_pipeline is not None and
            self.trellis_pipeline is not None
        )
    
    def clear_memory(self) -> None:
        """Clear GPU memory and run garbage collection."""
        torch.cuda.empty_cache()
        gc.collect()
    
    def get_model_status(self) -> Dict[str, bool]:
        """
        Get the loading status of all models.

        Returns:
            Dictionary with model loading status
        """
        return {
            "image_model_type": self.image_model,
            "image_pipeline_loaded": self.image_pipeline is not None,
            "flux_loaded": self.image_pipeline is not None,  # Backward compat
            "trellis_loaded": self.trellis_pipeline is not None,
            "reward_loaded": self.reward_model is not None,
            "reward_enabled": self.enable_reward_model,
            "all_loaded": self.are_models_loaded()
        }


# Global instance for caching across Streamlit runs
_global_model_manager = None


def get_model_manager(device_config=None, enable_trellis_cpu_offload=True, image_model="flux") -> ModelManager:
    """
    Get the global ModelManager instance.

    Args:
        device_config: Optional device configuration dict
        enable_trellis_cpu_offload: Enable CPU offloading for TRELLIS models
        image_model: Image generation model ('flux' or 'gemini')

    Returns:
        ModelManager: The global model manager instance
    """
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager(
            device_config=device_config,
            enable_trellis_cpu_offload=enable_trellis_cpu_offload,
            image_model=image_model
        )
    return _global_model_manager


# Convenience functions for backward compatibility
def load_models():
    """
    Legacy function for loading models.
    
    Returns:
        Tuple of (flux_pipeline, trellis_pipeline, reward_model)
    """
    manager = get_model_manager()
    return manager.load_all_models()


def get_flux_pipeline():
    """Legacy function for getting FLUX pipeline."""
    manager = get_model_manager()
    return manager.get_flux_pipeline()


def get_trellis_pipeline():
    """Legacy function for getting TRELLIS pipeline."""
    manager = get_model_manager()
    return manager.get_trellis_pipeline()


def get_reward_model():
    """Legacy function for getting reward model."""
    manager = get_model_manager()
    return manager.get_reward_model()