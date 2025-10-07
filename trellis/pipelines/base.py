from typing import *
import torch
import torch.nn as nn
from .. import models
from ..utils.offload_hooks import OffloadManager


class Pipeline:
    """
    A base class for pipelines with CPU offloading support.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
            
        # Initialize CPU offloading manager
        self._offload_manager = None
        self._cpu_offload_enabled = False

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        """Move all models to specified device."""
        if self._cpu_offload_enabled:
            # Update offload manager's execution device
            if self._offload_manager is not None:
                self._offload_manager.execution_device = torch.device(device)
        else:
            # Standard device movement
            for model in self.models.values():
                model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
    
    def enable_model_cpu_offload(self, execution_device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Enable CPU offloading for all pipeline models.
        
        This will move models to CPU when not in use and automatically move them to GPU
        during forward passes, significantly reducing VRAM usage.
        
        Args:
            execution_device: Device to use for model execution (default: current device or cuda:0)
        """
        if self._cpu_offload_enabled:
            print("⚠️ CPU offloading already enabled")
            return
        
        # Determine execution device
        if execution_device is None:
            try:
                execution_device = self.device
            except RuntimeError:
                execution_device = "cuda:0"
        
        # Initialize offload manager
        self._offload_manager = OffloadManager(execution_device)
        
        # Register all models for offloading
        for name, model in self.models.items():
            self._offload_manager.register_module(name, model)
        
        # Enable offloading
        self._offload_manager.enable_offload()
        self._cpu_offload_enabled = True
    
    def disable_model_cpu_offload(self) -> None:
        """Disable CPU offloading and restore models to execution device."""
        if not self._cpu_offload_enabled:
            return
            
        if self._offload_manager is not None:
            self._offload_manager.disable_offload()
            
        self._cpu_offload_enabled = False
    
    @property
    def is_cpu_offload_enabled(self) -> bool:
        """Check if CPU offloading is currently enabled."""
        return self._cpu_offload_enabled
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            device = self.device if not self._cpu_offload_enabled else self._offload_manager.execution_device
            if isinstance(device, torch.device) and device.type == "cuda":
                device_idx = device.index or 0
                stats.update({
                    "gpu_allocated_gb": torch.cuda.memory_allocated(device_idx) / 1024**3,
                    "gpu_reserved_gb": torch.cuda.memory_reserved(device_idx) / 1024**3,
                    "gpu_max_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / 1024**3,
                })
        
        stats.update({
            "cpu_offload_enabled": self._cpu_offload_enabled,
            "models_count": len(self.models),
        })
        
        return stats
