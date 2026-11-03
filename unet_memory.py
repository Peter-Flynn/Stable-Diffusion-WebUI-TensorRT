"""Where the PyTorch UNet's weights live while TensorRT does the denoising, how TensorRT and
reForge's model manager share VRAM, and the hooks that route the UNet forward to TensorRT.

reForge/Forge track weight placement in their model manager, so models are only ever moved
through it: tensors moved behind its back leave it believing they're still on the GPU. A1111 lets
us move `diffusion_model` directly.

The manager can't see TensorRT's memory, so it can neither make room for an engine nor reclaim
one. Both directions are handled here: before an engine loads (and before sampling, for its
activations) the manager unloads models to make room, and when the manager is about to load
weights that won't fit next to an engine, the engine is released (it's reloaded at the next UNet
call). Everything the manager unloads it reloads on demand.

This module isn't re-imported by "Reload UI" (scripts are), so the hooks live here and call the
current script's callbacks through `callbacks`.
"""

import gc
import importlib.util
import sys

import torch

from modules import devices, sd_unet, shared

IS_FORGE_BACKEND = importlib.util.find_spec("ldm_patched") is not None

_UNET_MODULES = (
    "ldm_patched.ldm.modules.diffusionmodules.openaimodel",  # reForge/Forge
    "ldm.modules.diffusionmodules.openaimodel",  # A1111 SD1/SD2
    "sgm.modules.diffusionmodules.openaimodel",  # A1111 SDXL
)

callbacks = {
    "trt_active": lambda: False,  # TensorRT runs the UNet in this generation
    "sampling_done": lambda: None,  # a sampler run ended (before its VAE decode)
    "sampling_memory": lambda: 0,  # VRAM the engine needs free while sampling
    "before_model_load": lambda models: None,  # the manager is about to load `models`
}

# Between sampling_prepare and sampling_cleanup of a TensorRT generation: the engine is in use.
in_trt_sampling = False


class TrtFallback(Exception):
    """Raised by the TensorRT UNet's forward to run PyTorch instead."""


def forge_model_management():
    import ldm_patched.modules.model_management as model_management

    return model_management


def _unet_patcher():
    return getattr(getattr(getattr(shared, "sd_model", None), "forge_objects", None), "unet", None)


def is_unet(patcher) -> bool:
    """Whether a ModelPatcher wraps the checkpoint's UNet (any clone of it)."""
    unet = _unet_patcher()
    return patcher is not None and unet is not None and (patcher is unet or unet.is_clone(patcher))


def report_free_vram(label: str):
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"[TensorRT] Free GPU memory {label}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
    except Exception:
        pass


def unload_unet_from_manager():
    """Unload every clone of the UNet that has weights on the GPU, through the manager."""
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
        if loaded.model_loaded_memory() <= 0:  # already off the GPU, e.g. a NO_VRAM load
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


def restore_torch_unet(input_shape=None):
    """Hand the UNet back to PyTorch. reForge loads it itself when it samples; only a fallback
    in the middle of sampling (sampling_prepare already ran, without it) passes the UNet input's
    shape and loads it here, the way sampling_prepare would. A1111: moved back to the GPU."""
    try:
        if IS_FORGE_BACKEND:
            patcher = _unet_patcher()
            if input_shape is None or patcher is None:
                return
            estimate = patcher.model_options.get("memory_peak_estimation_modifier", patcher.memory_required)
            required = estimate(list(input_shape)) + getattr(patcher, "extra_preserved_memory_during_sampling", 0)
            forge_model_management().load_models_gpu([patcher], memory_required=required)
        elif not getattr(shared.sd_model, "lowvram", False):  # A1111's lowvram mode places it itself
            shared.sd_model.model.diffusion_model.to(devices.device)
    except Exception as e:
        print(f"[TensorRT] Could not restore the PyTorch UNet: {e}", flush=True)


def make_room(size: int):
    """Have the manager unload models until `size` bytes are free, for allocations it can't see
    (TensorRT engines and their activations). A1111: nothing to unload."""
    if not IS_FORGE_BACKEND or size <= 0:
        return
    try:
        mm = forge_model_management()
        mm.free_memory(size, mm.get_torch_device())
    except Exception as e:
        print(f"[TensorRT] Could not free VRAM: {e}", flush=True)


def minimum_inference_memory() -> int:
    return int(forge_model_management().minimum_inference_memory())


def weights_to_load(models) -> int:
    """VRAM the weights of `models` still need (the UNet's excepted: TensorRT replaces it)."""
    total = 0
    for patcher in models:
        if patcher is None or is_unet(patcher):
            continue
        try:
            total += max(0, int(patcher.model_size()) - int(patcher.loaded_size()))
        except Exception:
            pass
    return total


def vram_available_for(models) -> int:
    """Free VRAM plus what the manager could unload to load `models`."""
    mm = forge_model_management()
    device = mm.get_torch_device()
    available = mm.get_free_memory(device)
    for loaded in list(mm.current_loaded_models):
        patcher = loaded.model
        if patcher is None or loaded.device != device:
            continue
        if any(patcher is m or (m is not None and m.is_clone(patcher)) for m in models):
            continue  # its weights are the ones being loaded
        try:
            available += int(loaded.model_loaded_memory())
        except Exception:
            pass
    return available


def free_all_models():
    if not IS_FORGE_BACKEND:
        return
    try:
        mm = forge_model_management()
        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception as e:
        print(f"[TensorRT] Could not unload models: {e}", flush=True)


def free_vram_for_export():
    """Release an active TensorRT engine (it holds VRAM outside PyTorch) and unload everything;
    the WebUI reloads what it needs on the next generation."""
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


def _keep_loaded(memory_required, device, keep_loaded=None):
    """free_memory() while reForge prepares a UNet TensorRT replaces: unload nothing for it."""
    return []


def _hook_forge_sampler():
    import modules_forge.forge_sampler as forge_sampler

    prepare, cleanup = forge_sampler.sampling_prepare, forge_sampler.sampling_cleanup
    if getattr(prepare, "_trt_hooked", False):
        return

    def sampling_prepare(unet, x, *args, **kwargs):
        global in_trt_sampling
        if not callbacks["trt_active"]():
            return prepare(unet, x, *args, **kwargs)
        in_trt_sampling = True
        mm = forge_model_management()
        unload_unet_from_manager()
        make_room(callbacks["sampling_memory"]())  # the engine's activations, allocated on its first step
        # reForge's own prepare still runs (it applies the UNet's patches, e.g. model_sampling
        # overrides), but with the UNet loaded in NO_VRAM mode, weights left in RAM, and nothing
        # unloaded for it: its budget is for a PyTorch UNet that isn't going to run.
        previous = mm.vram_state, mm.free_memory
        mm.vram_state, mm.free_memory = mm.VRAMState.NO_VRAM, _keep_loaded
        try:
            return prepare(unet, x, *args, **kwargs)
        finally:
            mm.vram_state, mm.free_memory = previous

    def sampling_cleanup(unet, *args, **kwargs):
        global in_trt_sampling
        in_trt_sampling = False
        try:
            callbacks["sampling_done"]()  # every sampler run ends here, before its VAE decode
        except Exception as e:
            print(f"[TensorRT] Could not release activation memory: {e}", flush=True)
        return cleanup(unet, *args, **kwargs)

    sampling_prepare._trt_hooked = True
    _rebind(forge_sampler, "sampling_prepare", prepare, sampling_prepare)
    _rebind(forge_sampler, "sampling_cleanup", cleanup, sampling_cleanup)


def _hook_forge_loader():
    mm = forge_model_management()
    original = mm.load_models_gpu
    if getattr(original, "_trt_hooked", False):
        return

    def load_models_gpu(models, *args, **kwargs):
        models = list(models)
        try:
            callbacks["before_model_load"](models)
        except Exception as e:
            print(f"[TensorRT] Could not make room for {len(models)} models: {e}", flush=True)
        return original(models, *args, **kwargs)

    load_models_gpu._trt_hooked = True
    _rebind(mm, "load_models_gpu", original, load_models_gpu)


def install_hooks():
    """Idempotent; run before each generation so modules loaded since are covered."""
    for name in _UNET_MODULES:
        cls = getattr(sys.modules.get(name), "UNetModel", None)
        if cls is not None and not getattr(cls, "_trt_hooked", False):
            cls.forward = _wrap_unet_forward(cls.forward)
            cls._trt_hooked = True
    if IS_FORGE_BACKEND:
        for hook in (_hook_forge_sampler, _hook_forge_loader):
            try:
                hook()
            except Exception as e:
                print(f"[TensorRT] Could not hook reForge's {hook.__name__.removeprefix('_hook_forge_')}: {e}", flush=True)
