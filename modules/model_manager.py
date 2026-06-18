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
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None  # Placeholder
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
            image_model: Image generation model ('flux', 'gemini', or 'qwen')
        """
        self.image_model = image_model.lower()
        self.image_pipeline = None  # Will hold FLUX, Gemini, or Qwen pipeline
        self.flux_pipeline = None  # Deprecated, kept for backward compatibility
        self.trellis_pipeline = None
        self.reward_model = None

        # Fast sampling mode configuration (unified across all backends)
        self.use_fast_sampling = os.environ.get("FAST_IMAGE_SAMPLING", "false").lower() == "true"

        # Device configuration with safety checks
        self.device_config = self._validate_device_config(device_config)
        self.image_device = self.device_config.get("image", "cuda:0")  # Renamed from flux_device
        self.flux_device = self.image_device  # Backward compatibility
        self.trellis_device = self.device_config.get("trellis", "cuda:0")

        # Multi-GPU and TRELLIS CPU offloading are mutually exclusive
        if self.image_device != self.trellis_device and enable_trellis_cpu_offload:
            print("⚠️ Multi-GPU mode detected: Disabling TRELLIS CPU offloading (incompatible)")
            print("   Multi-GPU placement and CPU offloading cannot be used together for TRELLIS")
            print(f"   TRELLIS will use dedicated GPU: {self.trellis_device}")
            self.enable_trellis_cpu_offload = False
        else:
            self.enable_trellis_cpu_offload = enable_trellis_cpu_offload

        # Model-specific configuration
        if self.image_model == "qwen":
            self.qwen_model_name = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image")
            if self.use_fast_sampling:
                print("🚀 Fast sampling enabled: Will load Qwen Lightning LoRA")
        elif self.image_model not in ["flux", "gemini", "qwen"]:
            raise ValueError(f"Unknown image_model: {self.image_model}. Must be 'flux', 'gemini', or 'qwen'")

        # CPU offloading configuration (Qwen-specific, FLUX and Gemini handle their own)
        # Support both old (ENABLE_FLUX_CPU_OFFLOAD) and new (ENABLE_IMAGE_CPU_OFFLOAD) env vars
        enable_image_offload_env = os.environ.get("ENABLE_IMAGE_CPU_OFFLOAD") or os.environ.get("ENABLE_FLUX_CPU_OFFLOAD")
        self.enable_image_cpu_offload = enable_image_offload_env and enable_image_offload_env.lower() == "true"

        if self.image_model == "qwen":
            self.enable_qwen_cpu_offload = self.enable_image_cpu_offload
            self.use_qwen_cpu_offload = (self.image_device == "cuda:0" or self.image_device == "cuda") and self.enable_qwen_cpu_offload

        # Reward model configuration (optional, disabled by default to save VRAM)
        self.enable_reward_model = os.environ.get("ENABLE_REWARD_MODEL", "false").lower() == "true"

    def _validate_device_config(self, device_config) -> Dict[str, str]:
        """
        Validate device configuration based on available hardware.

        Args:
            device_config: Requested device configuration
                          Use "image" key for image model device (replaces deprecated "flux")

        Returns:
            Dict with validated device configuration (with "image" and "trellis" keys)
        """
        # Default to single GPU
        default_config = {"image": "cuda:0", "trellis": "cuda:0"}

        if device_config is None:
            return default_config

        # Check if we have enough GPUs for multi-GPU setup
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"⚠️ Only {num_gpus} GPU(s) available, using single GPU mode")
            return default_config

        # Support both "image" (new) and "flux" (deprecated) keys
        image_device = device_config.get("image") or device_config.get("flux", "cuda:0")
        trellis_device = device_config.get("trellis", "cuda:0")

        # Parse device indices
        try:
            image_idx = int(image_device.split(":")[1]) if ":" in image_device else 0
            trellis_idx = int(trellis_device.split(":")[1]) if ":" in trellis_device else 0
        except (ValueError, IndexError):
            print("⚠️ Invalid device specification, using single GPU mode")
            return default_config

        # Check if requested devices exist
        if image_idx >= num_gpus or trellis_idx >= num_gpus:
            print(f"⚠️ Requested GPU index exceeds available GPUs ({num_gpus}), using single GPU mode")
            return default_config

        # Return normalized config with "image" key
        return {"image": image_device, "trellis": trellis_device}

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

    def _load_qwen_pipeline(self):
        """
        Load Qwen/Qwen-Image pipeline.

        Returns:
            QwenImagePipeline: Configured Qwen pipeline
        """
        from modules.qwen_pipeline import QwenImagePipeline

        # Get 8-bit loading preference (default: True)
        load_in_8bit = os.environ.get("QWEN_LOAD_IN_8BIT", "true").lower() == "true"

        pipeline = QwenImagePipeline(
            model_name=self.qwen_model_name,
            device=self.image_device,  # Use same device as other image models
            enable_cpu_offload=self.enable_qwen_cpu_offload,
            load_in_8bit=load_in_8bit,
            use_fast_sampling=self.use_fast_sampling
        )
        return pipeline

    def _load_image_pipeline(self):
        """
        Load image generation pipeline based on selected model.

        Returns:
            Image pipeline (FLUX, Gemini, or Qwen)
        """
        if self.image_model == "flux":
            return self._load_flux_pipeline()
        elif self.image_model == "gemini":
            return self._load_gemini_pipeline()
        elif self.image_model == "qwen":
            return self._load_qwen_pipeline()
        else:
            raise ValueError(f"Unknown image_model: {self.image_model}")

    def _load_flux_pipeline(self):
        """
        Load FLUX image generation pipeline using the FLUX adapter.

        Returns:
            FLUXImagePipeline: Configured FLUX pipeline
        """
        from modules.flux_pipeline import FLUXImagePipeline

        # FLUX adapter handles all configuration internally via env vars
        pipeline = FLUXImagePipeline(
            device=self.image_device,
            use_fast_sampling=self.use_fast_sampling
        )
        return pipeline

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

            # Optional printability DPO LoRA on the Stage-1 sparse-structure flow model.
            # See printability_optimization_3d.md. Stacks with TRELLIS_STAGE1_FILL_HOLES.
            lora_dir = os.environ.get('TRELLIS_STAGE1_LORA', '').strip()
            if lora_dir:
                from peft import PeftModel
                flow = pipeline.models['sparse_structure_flow_model']
                pipeline.models['sparse_structure_flow_model'] = PeftModel.from_pretrained(
                    flow, lora_dir, is_trainable=False)
                print(f"[printability] Loaded Stage-1 LoRA from {lora_dir}")

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

        # Determine loading time estimate
        loading_time = {
            "flux": "(1-2 minutes)",
            "qwen": "(1-2 minutes)",
            "gemini": "(instant)"
        }.get(self.image_model, "")

        if HAS_STREAMLIT:
            with st.spinner(f"🚀 Loading {model_name} Image Model... {loading_time}"):
                # Load image generation pipeline
                self.image_pipeline = self._load_image_pipeline()

                # Warm up pipelines (skip for Gemini) - let adapters handle their own warmup
                if hasattr(self.image_pipeline, 'warmup'):
                    self.image_pipeline.warmup()

                # Load reward model (optional)
                self.reward_model = self._load_reward_model()
        else:
            # Load image generation pipeline
            self.image_pipeline = self._load_image_pipeline()

            # Warm up pipelines (skip for Gemini) - let adapters handle their own warmup
            if hasattr(self.image_pipeline, 'warmup'):
                self.image_pipeline.warmup()

            # Load reward model (optional)
            self.reward_model = self._load_reward_model()

        if HAS_STREAMLIT:
            with st.spinner("🔧 Loading 3D Model Generator..."):
                # Load TRELLIS pipeline
                self.trellis_pipeline = self._load_trellis_pipeline()
        else:
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

    def get_flux_pipeline(self):
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
        Get the current generation configuration for the active image model.

        This method works both before and after the pipeline is loaded:
        - If pipeline is loaded: delegates to its get_generation_config()
        - If pipeline not loaded: computes defaults based on image_model and configuration

        Returns:
            Dictionary with generation parameters optimized for the active backend
        """
        # Delegate to adapter if available (pipeline already loaded)
        if self.image_pipeline and hasattr(self.image_pipeline, 'get_generation_config'):
            return self.image_pipeline.get_generation_config()

        # Compute config based on image_model and configuration (before pipeline is loaded)
        if self.image_model == "qwen":
            # Compute Qwen defaults based on fast sampling mode
            if self.use_fast_sampling:
                return {
                    "guidance_scale": 1.8,  # Lightning LoRA cfg_scale
                    "num_inference_steps": 4,  # Lightning LoRA steps
                    "model_name": self.qwen_model_name if hasattr(self, 'qwen_model_name') else "Qwen/Qwen-Image",
                    "fast_sampling": True
                }
            else:
                return {
                    "guidance_scale": 5.5,  # Normal cfg_scale
                    "num_inference_steps": 40,  # Normal steps
                    "model_name": self.qwen_model_name if hasattr(self, 'qwen_model_name') else "Qwen/Qwen-Image",
                    "fast_sampling": False
                }

        elif self.image_model == "flux":
            # Compute FLUX defaults based on model variant
            # Determine model name from env vars
            if self.use_fast_sampling:
                model_name = "black-forest-labs/FLUX.1-Schnell"
            elif os.environ.get("USE_FLUX_DEV", "false").lower() == "true":
                model_name = "black-forest-labs/FLUX.1-dev"
            else:
                model_name = "black-forest-labs/FLUX.1-Schnell"

            # Set parameters based on model
            if model_name == "black-forest-labs/FLUX.1-Schnell":
                return {
                    "guidance_scale": 0.0,  # Distilled, no CFG
                    "num_inference_steps": 4,  # Fast
                    "model_name": model_name,
                    "fast_sampling": self.use_fast_sampling
                }
            else:  # FLUX.1-dev
                return {
                    "guidance_scale": 4.5,  # Needs guidance
                    "num_inference_steps": 28,  # Quality
                    "model_name": model_name,
                    "fast_sampling": False
                }

        elif self.image_model == "gemini":
            # Gemini defaults (API-based, parameters mostly ignored)
            return {
                "guidance_scale": 0.0,  # Ignored by API
                "num_inference_steps": 4,  # Ignored by API
                "model_name": "gemini-2.5-flash-image",
                "fast_sampling": False
            }

        else:
            # Unknown model fallback
            return {
                "guidance_scale": 0.0,
                "num_inference_steps": 4,
                "model_name": "unknown",
                "fast_sampling": False
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