"""Where the PyTorch UNet's weights live while a TensorRT engine does the denoising.

A1111 lets us move `diffusion_model` between devices directly.

reForge/Forge track weight placement in their model manager (ldm_patched model_management),
so moving tensors behind its back desyncs it: it keeps believing the weights are on the GPU
and never moves them back, which ends in "Expected all tensors to be on the same device"
on the next PyTorch pass. There we unload through the manager instead, and while TensorRT is
active we make reForge's `sampling_prepare` load the UNet in NO_VRAM mode: weights stay in
system RAM but are streamed per layer if PyTorch needs them (e.g. a TensorRT fallback).

Whatever we unload must be put back the way reForge had it. reForge loads the UNet with
force_full_load at checkpoint load; if it's instead reloaded by `sampling_prepare`, that uses a
conservative budget (free VRAM minus estimated inference memory), loads the UNet only
partially and streams the rest from RAM every step - generation slows to a crawl.
"""

import gc
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
            # Fully, like reForge's checkpoint load. Leaving it to sampling_prepare would load it
            # only partially (conservative budget) and stream the rest every step.
            patcher = _unet_patcher()
            if patcher is not None:
                _model_management().load_models_gpu([patcher], force_full_load=True)
            return
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
    """Unload everything the reForge/Forge model manager holds. Pair with snapshot/restore."""
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


def snapshot_loaded_models():
    """What reForge/Forge has on the GPU, and whether each model is fully loaded, so it can be
    put back exactly after an export. Models with almost nothing on the GPU (e.g. the UNet
    streamed while TensorRT is active) are left to their owners."""
    if not IS_FORGE_BACKEND:
        return []
    snapshot = []
    try:
        for loaded in list(_model_management().current_loaded_models):
            patcher = loaded.model
            if patcher is None or loaded.model_loaded_memory() <= 256 * 1024 * 1024:
                continue
            snapshot.append((patcher, loaded.model_offloaded_memory() == 0))
    except Exception as e:
        print(f"[TensorRT] Could not record loaded models: {e}", flush=True)
    return snapshot


def restore_loaded_models(snapshot):
    """Reload the models recorded by snapshot_loaded_models(), fully loaded ones fully."""
    if not snapshot:
        return
    mm = _model_management()
    full = [patcher for patcher, fully_loaded in snapshot if fully_loaded]
    partial = [patcher for patcher, fully_loaded in snapshot if not fully_loaded]
    try:
        if full:
            mm.load_models_gpu(full, force_full_load=True)
        if partial:
            mm.load_models_gpu(partial)
        report_free_vram("after restoring models")
    except Exception as e:
        print(f"[TensorRT] Could not restore models after export: {e}", flush=True)


def _rebind(owner, name, original, replacement):
    """Replace a function, including in modules that did `from ... import name`."""
    setattr(owner, name, replacement)
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, name, None) is original:
            setattr(module, name, replacement)


def install_sampling_hook(is_trt_active, on_sampling_done=None):
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
    _rebind(forge_sampler, "sampling_prepare", original, sampling_prepare)

    # After every sampler run (base pass, Hires. fix pass, img2img), before the VAE decode.
    original_cleanup = getattr(forge_sampler, "sampling_cleanup", None)
    if on_sampling_done is not None and original_cleanup is not None:

        def sampling_cleanup(unet, *args, **kwargs):
            try:
                on_sampling_done()
            except Exception as e:
                print(f"[TensorRT] Could not release activation memory: {e}", flush=True)
            return original_cleanup(unet, *args, **kwargs)

        _rebind(forge_sampler, "sampling_cleanup", original_cleanup, sampling_cleanup)
    _sampling_hook_installed = True
    print("[TensorRT] Hooked reForge sampling_prepare (UNet weights stay off the GPU while TensorRT is active).", flush=True)


def report_free_vram(label: str):
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"[TensorRT] Free GPU memory {label}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
    except Exception:
        pass


def free_vram_for_export():
    """Clear VRAM for ONNX tracing / an engine build: release an active TensorRT engine (it holds
    VRAM outside PyTorch) and unload every model reForge/Forge's manager holds. Take a
    snapshot_loaded_models() first and restore_loaded_models() afterwards."""
    from modules import sd_unet

    if sd_unet.current_unet is not None:
        try:
            release = getattr(sd_unet.current_unet, "release_engine", None)
            (release or sd_unet.current_unet.deactivate)()
        except Exception as e:
            print(f"[TensorRT] Could not release the current UNet: {e}", flush=True)
        sd_unet.current_unet = None
        sd_unet.current_unet_option = None  # the WebUI re-activates it on the next generation
    free_all_models()
    gc.collect()
    torch.cuda.empty_cache()
