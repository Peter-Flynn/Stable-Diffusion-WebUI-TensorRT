"""Where the PyTorch UNet's weights live while TensorRT does the denoising, and the hooks that
route the UNet forward to TensorRT.

reForge/Forge track weight placement in their model manager, so models are only ever moved
through it: tensors moved behind its back leave it believing they're still on the GPU. And
whatever is unloaded must be reloaded the way reForge had it (the UNet fully, as at checkpoint
load): `sampling_prepare` alone reloads it with a conservative budget, partially, and streams
the rest from RAM every step. A1111 lets us move `diffusion_model` directly.

This module isn't re-imported by "Reload UI" (scripts are), so the hooks live here and call the
current script's callbacks through `callbacks`.
"""

import gc
import importlib.util
import sys

import torch

from modules import devices, sd_unet, shared

IS_FORGE_BACKEND = importlib.util.find_spec("ldm_patched") is not None

_STREAMED_BYTES = 256 * 1024 * 1024  # a model with less than this on the GPU counts as streamed

_UNET_MODULES = (
    "ldm_patched.ldm.modules.diffusionmodules.openaimodel",  # reForge/Forge
    "ldm.modules.diffusionmodules.openaimodel",  # A1111 SD1/SD2
    "sgm.modules.diffusionmodules.openaimodel",  # A1111 SDXL
)

callbacks = {"trt_active": lambda: False, "sampling_done": lambda: None}


class TrtFallback(Exception):
    """Raised by the TensorRT UNet's forward to run PyTorch instead."""


def forge_model_management():
    import ldm_patched.modules.model_management as model_management

    return model_management


def _unet_patcher():
    return getattr(getattr(getattr(shared, "sd_model", None), "forge_objects", None), "unet", None)


def report_free_vram(label: str):
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"[TensorRT] Free GPU memory {label}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
    except Exception:
        pass


def unload_unet_from_manager(min_loaded_bytes: int = 0):
    patcher = _unet_patcher()
    if patcher is None:
        return
    mm = forge_model_management()
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


def offload_torch_unet():
    try:
        if IS_FORGE_BACKEND:
            unload_unet_from_manager()
        else:
            shared.sd_model.model.diffusion_model.to(devices.cpu)
        devices.torch_gc()
    except Exception as e:
        print(f"[TensorRT] Could not offload the PyTorch UNet: {e}", flush=True)


def restore_torch_unet():
    """Give the PyTorch UNet back its normal, fully loaded state."""
    try:
        if IS_FORGE_BACKEND:
            patcher = _unet_patcher()
            if patcher is not None:
                forge_model_management().load_models_gpu([patcher], force_full_load=True)
        elif not getattr(shared.sd_model, "lowvram", False):  # A1111's lowvram mode places it itself
            shared.sd_model.model.diffusion_model.to(devices.device)
    except Exception as e:
        print(f"[TensorRT] Could not restore the PyTorch UNet: {e}", flush=True)


def free_all_models():
    if not IS_FORGE_BACKEND:
        return
    try:
        mm = forge_model_management()
        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception as e:
        print(f"[TensorRT] Could not unload models: {e}", flush=True)


def snapshot_loaded_models():
    """(patcher, fully loaded) for each model reForge has on the GPU. Streamed models are skipped,
    except the UNet an SD Unet replaces: exports drop the SD Unet, so the UNet must come back
    fully loaded, as with SD Unet = None."""
    if not IS_FORGE_BACKEND:
        return []
    snapshot = []
    try:
        for loaded in list(forge_model_management().current_loaded_models):
            if loaded.model is not None and loaded.model_loaded_memory() > _STREAMED_BYTES:
                snapshot.append((loaded.model, loaded.model_offloaded_memory() == 0))
        patcher = _unet_patcher()
        if sd_unet.current_unet is not None and patcher is not None and not any(p is patcher or patcher.is_clone(p) for p, _ in snapshot):
            snapshot.append((patcher, True))
    except Exception as e:
        print(f"[TensorRT] Could not record loaded models: {e}", flush=True)
    return snapshot


def restore_loaded_models(snapshot):
    if not snapshot:
        return
    mm = forge_model_management()
    try:
        full = [patcher for patcher, fully in snapshot if fully]
        partial = [patcher for patcher, fully in snapshot if not fully]
        if full:
            mm.load_models_gpu(full, force_full_load=True)
        if partial:
            mm.load_models_gpu(partial)
        report_free_vram("after restoring models")
    except Exception as e:
        print(f"[TensorRT] Could not restore models: {e}", flush=True)


def free_vram_for_export():
    """Release an active TensorRT engine (it holds VRAM outside PyTorch) and unload everything.
    Pair with snapshot_loaded_models() / restore_loaded_models()."""
    unet = sd_unet.current_unet
    if unet is not None:
        try:
            getattr(unet, "release_engine", unet.deactivate)()
        except Exception as e:
            print(f"[TensorRT] Could not release the current UNet: {e}", flush=True)
        sd_unet.current_unet = None
        sd_unet.current_unet_option = None  # the WebUI re-activates it on the next generation
    free_all_models()
    gc.collect()
    torch.cuda.empty_cache()


def _wrap_unet_forward(original):
    def forward(self, x, timesteps=None, context=None, *args, **kwargs):
        unet = sd_unet.current_unet
        if unet is None:
            return original(self, x, timesteps, context, *args, **kwargs)
        try:
            return unet.forward(x, timesteps, context, *args, **kwargs)
        except TrtFallback:
            pass
        if not IS_FORGE_BACKEND:  # reForge streams or restores the weights itself
            param = next(self.parameters(), None)
            if param is not None and param.device != x.device:
                self.to(x.device)
        sd_unet.current_unet = None  # so wrapped forwards (A1111's own hook) run PyTorch too
        try:
            return original(self, x, timesteps, context, *args, **kwargs)
        finally:
            sd_unet.current_unet = unet

    return forward


def _rebind(owner, name, original, replacement):
    """Replace a function, including in modules that did `from ... import name`."""
    setattr(owner, name, replacement)
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)  # not getattr(module, name): lazy modules import on access
        if isinstance(namespace, dict) and namespace.get(name) is original:
            setattr(module, name, replacement)


def _hook_forge_sampler():
    import modules_forge.forge_sampler as forge_sampler

    prepare, cleanup = forge_sampler.sampling_prepare, forge_sampler.sampling_cleanup
    if getattr(prepare, "_trt_hooked", False):
        return

    def sampling_prepare(unet, x, *args, **kwargs):
        # While TensorRT is active, load the UNet in NO_VRAM mode: weights stay in RAM (streamed
        # if PyTorch needs them) instead of sitting in VRAM next to the engine.
        if not callbacks["trt_active"]():
            return prepare(unet, x, *args, **kwargs)
        mm = forge_model_management()
        unload_unet_from_manager(min_loaded_bytes=_STREAMED_BYTES)
        previous, mm.vram_state = mm.vram_state, mm.VRAMState.NO_VRAM
        try:
            return prepare(unet, x, *args, **kwargs)
        finally:
            mm.vram_state = previous

    def sampling_cleanup(unet, *args, **kwargs):
        try:
            callbacks["sampling_done"]()  # every sampler run ends here, before its VAE decode
        except Exception as e:
            print(f"[TensorRT] Could not release activation memory: {e}", flush=True)
        return cleanup(unet, *args, **kwargs)

    sampling_prepare._trt_hooked = True
    _rebind(forge_sampler, "sampling_prepare", prepare, sampling_prepare)
    _rebind(forge_sampler, "sampling_cleanup", cleanup, sampling_cleanup)


def install_hooks():
    """Idempotent; run before each generation so modules loaded since are covered."""
    for name in _UNET_MODULES:
        cls = getattr(sys.modules.get(name), "UNetModel", None)
        if cls is not None and not getattr(cls, "_trt_hooked", False):
            cls.forward = _wrap_unet_forward(cls.forward)
            cls._trt_hooked = True
    if IS_FORGE_BACKEND:
        try:
            _hook_forge_sampler()
        except Exception as e:
            print(f"[TensorRT] Could not hook reForge's sampler: {e}", flush=True)
