"""
CPU offloading hooks for TRELLIS pipelines.

This module implements CPU offloading functionality similar to diffusers/accelerate,
allowing TRELLIS models to be automatically moved between CPU and GPU during inference
to reduce VRAM usage.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import gc
from contextlib import contextmanager
import weakref


class TrellisOffloadHook:
    """
    A hook that manages automatic CPU/GPU offloading for TRELLIS model components.
    
    This hook is installed on PyTorch modules and automatically:
    1. Moves module to GPU before forward pass
    2. Moves module back to CPU after forward pass  
    3. Manages memory cleanup between operations
    
    Based on accelerate library's AlignDevicesHook pattern.
    """
    
    def __init__(
        self,
        execution_device: Union[str, torch.device] = "cuda:0",
        offload_device: Union[str, torch.device] = "cpu", 
        offload_buffers: bool = True,
        place_submodules: bool = False,
        module_name: Optional[str] = None
    ):
        """
        Initialize the offload hook.
        
        Args:
            execution_device: Device to move module to during forward pass
            offload_device: Device to store module when not in use (typically CPU)
            offload_buffers: Whether to also offload module buffers
            place_submodules: Whether to also manage submodules 
            module_name: Optional name for debugging/logging
        """
        self.execution_device = torch.device(execution_device)
        self.offload_device = torch.device(offload_device)
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.module_name = module_name or "unknown"
        
        # Track hook state
        self.hook_handles = []
        self.original_devices = {}
        self.is_active = False
        
    def __call__(self, module: nn.Module, *args, **kwargs):
        """
        Pre-forward hook: Move module to execution device.
        """
        if not self.is_active:
            return
            
        # Store original device for restoration
        if not hasattr(self, '_original_device_stored'):
            self._store_original_devices(module)
            self._original_device_stored = True
            
        # Move to execution device
        self._move_to_device(module, self.execution_device)
        
    def post_forward_hook(self, module: nn.Module, input: Any, output: Any):
        """
        Post-forward hook: Move module back to offload device.
        """
        if not self.is_active:
            return output
            
        # Move back to offload device
        self._move_to_device(module, self.offload_device)
        
        # Clean up GPU memory
        if self.execution_device.type == "cuda":
            torch.cuda.empty_cache()
            
        return output
    
    def _store_original_devices(self, module: nn.Module):
        """Store original device locations for restoration."""
        for name, param in module.named_parameters(recurse=True):
            self.original_devices[f"param_{name}"] = param.device
            
        if self.offload_buffers:
            for name, buffer in module.named_buffers(recurse=True):
                self.original_devices[f"buffer_{name}"] = buffer.device
    
    def _move_to_device(self, module: nn.Module, target_device: torch.device):
        """Move module parameters and buffers to target device."""
        try:
            # FIXED: Always recurse into submodules to move all parameters
            # This is essential for neural networks with multiple layers
            for param in module.parameters(recurse=True):
                if param.device != target_device:
                    param.data = param.data.to(target_device, non_blocking=True)
                    
            # Move buffers if requested
            if self.offload_buffers:
                for buffer in module.buffers(recurse=True):
                    if buffer.device != target_device:
                        buffer.data = buffer.data.to(target_device, non_blocking=True)
                        
        except Exception as e:
            print(f"⚠️ Warning: Failed to move {self.module_name} to {target_device}: {e}")
    
    def install(self, module: nn.Module) -> None:
        """Install the offload hooks on a module."""
        if self.is_active:
            return
            
        # Store original device locations before moving
        self._store_original_devices(module)
        self._original_device_stored = True
        
        # Install pre-forward hook
        pre_handle = module.register_forward_pre_hook(self.__call__)
        self.hook_handles.append(pre_handle)
        
        # Install post-forward hook  
        post_handle = module.register_forward_hook(self.post_forward_hook)
        self.hook_handles.append(post_handle)
        
        # CRITICAL FIX: Actually move module to offload device immediately
        self._move_to_device(module, self.offload_device)
        
        self.is_active = True
        
    def remove(self, module: Optional[nn.Module] = None) -> None:
        """Remove all hooks from the module."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # Move module back to execution device when hooks are removed
        if module is not None:
            self._move_to_device(module, self.execution_device)
        
        self.is_active = False


class OffloadManager:
    """
    Manages multiple offload hooks for a pipeline with stage-aware operations.
    """
    
    def __init__(self, execution_device: Union[str, torch.device] = "cuda:0"):
        self.execution_device = torch.device(execution_device)
        self.hooks: Dict[str, TrellisOffloadHook] = {}
        self.modules: Dict[str, nn.Module] = {}
        self.is_enabled = False
        
    def register_module(
        self, 
        name: str, 
        module: nn.Module,
        offload_buffers: bool = True,
        place_submodules: bool = False
    ) -> None:
        """Register a module for CPU offloading."""
        hook = TrellisOffloadHook(
            execution_device=self.execution_device,
            offload_device="cpu",
            offload_buffers=offload_buffers,
            place_submodules=place_submodules,
            module_name=name
        )
        
        self.hooks[name] = hook
        self.modules[name] = weakref.ref(module)
        
        if self.is_enabled:
            hook.install(module)
    
    def enable_offload(self) -> None:
        """Enable CPU offloading for all registered modules."""
        if self.is_enabled:
            return
            
        for name, hook in self.hooks.items():
            module_ref = self.modules.get(name)
            if module_ref is not None:
                module = module_ref()
                if module is not None:
                    hook.install(module)
                    
        self.is_enabled = True
    
    def disable_offload(self) -> None:
        """Disable CPU offloading and restore original devices."""
        if not self.is_enabled:
            return
            
        for name, hook in self.hooks.items():
            module_ref = self.modules.get(name)
            if module_ref is not None:
                module = module_ref()
                if module is not None:
                    hook.remove(module)
                else:
                    hook.remove()
            else:
                hook.remove()
                    
        self.is_enabled = False
    
    @contextmanager
    def stage_context(self, stage_name: str, active_modules: list = None):
        """
        Context manager for pipeline stages that only activates specific modules.
        
        Args:
            stage_name: Name of the pipeline stage
            active_modules: List of module names to keep on GPU during this stage
        """
        if not self.is_enabled or not active_modules:
            yield
            return
        
        # Temporarily disable hooks for non-active modules
        inactive_hooks = []
        for name, hook in self.hooks.items():
            if name not in active_modules and hook.is_active:
                hook.remove()
                inactive_hooks.append(name)
        
        # Ensure active modules are on GPU
        for module_name in active_modules:
            module_ref = self.modules.get(module_name)
            if module_ref is not None:
                module = module_ref()
                if module is not None:
                    module.to(self.execution_device)
        
        try:
            yield
        finally:
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Move active modules back to CPU
            for module_name in active_modules:
                module_ref = self.modules.get(module_name)
                if module_ref is not None:
                    module = module_ref()
                    if module is not None:
                        module.to("cpu")
            
            # Re-enable hooks for inactive modules
            for name in inactive_hooks:
                hook = self.hooks[name]
                module_ref = self.modules.get(name)
                if module_ref is not None:
                    module = module_ref()
                    if module is not None:
                        hook.install(module)