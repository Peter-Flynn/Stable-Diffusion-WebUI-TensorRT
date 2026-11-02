"""Where the PyTorch UNet's weights live while a TensorRT engine does the denoising.

A1111 lets us move `diffusion_model` between devices directly.

reForge/Forge track weight placement in their model manager (ldm_patched model_management),
so moving tensors behind its back desyncs it: it keeps believing the weights are on the GPU
and never moves them back, which ends in "Expected all tensors to be on the same device"
on the next PyTorch pass. There we unload through the manager instead, and while TensorRT is
active we make reForge's `sampling_prepare` load the UNet in NO_VRAM mode: weights stay in
system RAM but are streamed per layer if PyTorch needs them (e.g. a TensorRT fallback).
"""

import importlib.util
import sys

import torch

from modules import devices, shared

IS_FORGE_BACKEND = importlib.util.find_spec("ldm_patched") is not None

_sampling_hook_installed = False


def _model_management():
    import ldm_patched.modules.model_management as model_management

    return model_management


def _unet_patcher():
    forge_objects = getattr(getattr(shared, "sd_model", None), "forge_objects", None)
    return getattr(forge_objects, "unet", None)


def _diffusion_model():
    return shared.sd_model.model.diffusion_model


def unload_unet_from_manager(min_loaded_bytes: int = 0) -> bool:
    """reForge/Forge: unload the UNet (and its clones) through the model manager."""
    patcher = _unet_patcher()
    if patcher is None:
        return False
    mm = _model_management()
    unloaded = False
    for i in reversed(range(len(mm.current_loaded_models))):
        loaded = mm.current_loaded_models[i]
        model = loaded.model
        if model is None or not (model is patcher or patcher.is_clone(model)):
            continue
        if loaded.model_loaded_memory() <= min_loaded_bytes:
            continue
        loaded.model_unload()
        mm.current_loaded_models.pop(i)
        unloaded = True
    if unloaded:
        mm.soft_empty_cache(force=True)
    return unloaded


def offload_torch_unet():
    """Free the PyTorch UNet's VRAM while a TensorRT engine is active."""
    try:
        if IS_FORGE_BACKEND:
            unload_unet_from_manager()
        else:
            _diffusion_model().to(devices.cpu)
        devices.torch_gc()
    except Exception as e:
        print(f"[TensorRT] Could not offload the PyTorch UNet: {e}", flush=True)


def restore_torch_unet():
    """Make the PyTorch UNet usable again (TensorRT deactivated or falling back)."""
    try:
        if IS_FORGE_BACKEND:
            return  # the model manager loads it on the next sampling_prepare / forward
        _diffusion_model().to(devices.device)
    except Exception as e:
        print(f"[TensorRT] Could not restore the PyTorch UNet: {e}", flush=True)


def ensure_module_on(module: torch.nn.Module, device):
    """PyTorch fallback inside a TensorRT generation. On reForge the manager streams weights
    (NO_VRAM load), so only move modules ourselves on A1111."""
    if IS_FORGE_BACKEND:
        return
    param = next(module.parameters(), None)
    if param is not None and param.device != device:
        module.to(device)


def free_all_models():
    """Unload everything the reForge/Forge model manager holds (reloaded on demand)."""
    if not IS_FORGE_BACKEND:
        return False
    try:
        mm = _model_management()
        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
        return True
    except Exception as e:
        print(f"[TensorRT] Could not unload models: {e}", flush=True)
        return False


def install_sampling_hook(is_trt_active):
    """reForge: while TensorRT is active, keep sampling_prepare from loading the UNet weights
    into VRAM (load them in NO_VRAM mode instead), so the UNet isn't held twice."""
    global _sampling_hook_installed
    if _sampling_hook_installed or not IS_FORGE_BACKEND:
        return
    try:
        import modules_forge.forge_sampler as forge_sampler
    except Exception:
        return

    original = forge_sampler.sampling_prepare
    if getattr(original, "_trt_wrapped", False):
        _sampling_hook_installed = True
        return

    def sampling_prepare(unet, x, *args, **kwargs):
        if not is_trt_active():
            return original(unet, x, *args, **kwargs)
        mm = _model_management()
        # A fully loaded UNet (from an earlier PyTorch generation) is unloaded first;
        # an already-streamed one (<= 256 MB on the GPU) is left as is.
        unload_unet_from_manager(min_loaded_bytes=256 * 1024 * 1024)
        previous = mm.vram_state
        mm.vram_state = mm.VRAMState.NO_VRAM
        try:
            return original(unet, x, *args, **kwargs)
        finally:
            mm.vram_state = previous

    sampling_prepare._trt_wrapped = True
    forge_sampler.sampling_prepare = sampling_prepare
    # Modules that did `from modules_forge.forge_sampler import sampling_prepare` hold their own reference.
    for name, module in list(sys.modules.items()):
        if module is not None and getattr(module, "sampling_prepare", None) is original:
            setattr(module, "sampling_prepare", sampling_prepare)
    _sampling_hook_installed = True
    print("[TensorRT] Hooked reForge sampling_prepare (UNet weights stay off the GPU while TensorRT is active).", flush=True)
