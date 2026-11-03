"""VRAM shared between the PyTorch UNet, reForge's model manager and TensorRT, and the hooks that
route the UNet to TensorRT. reForge models only move through its manager (behind its back, it
loses track of them); the manager can't see TensorRT's memory, so room is made for engines here,
and engines make way for its models. "Reload UI" doesn't re-import this module: the hooks call
the current script through `callbacks`."""

import functools
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
    "trt_active": lambda: False,
    "sampling_done": lambda: None,
    "before_model_load": lambda models: None,
}

in_trt_sampling = False  # between sampling_prepare and sampling_cleanup of a TensorRT generation


class TrtFallback(Exception):
    pass


def forge_model_management():
    import ldm_patched.modules.model_management as model_management

    return model_management


def in_inference_mode(fn):
    """For calls that move reForge's weights. Its generations run in inference mode, so a LoRA's
    unpatch leaves inference-tensor weights; moved outside it, they break ("Inference tensors do not
    track version counter")."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper


def _unet_patcher():
    return getattr(getattr(getattr(shared, "sd_model", None), "forge_objects", None), "unet", None)


def is_unet(patcher) -> bool:
    unet = _unet_patcher()
    return patcher is not None and unet is not None and (patcher is unet or unet.is_clone(patcher))


def report_free_vram(label: str):
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"[TensorRT] Free GPU memory {label}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
    except Exception:
        pass


@in_inference_mode
def unload_unet_from_manager():
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
        if loaded.model_loaded_memory() <= 0:
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
    """reForge loads the UNet itself when sampling starts; with `input_shape` (mid-sampling) it's
    loaded here, as sampling_prepare would."""
    try:
        if IS_FORGE_BACKEND:
            patcher = _unet_patcher()
            if input_shape is None or patcher is None:
                return
            estimate = patcher.model_options.get("memory_peak_estimation_modifier", patcher.memory_required)
            required = estimate(list(input_shape)) + getattr(patcher, "extra_preserved_memory_during_sampling", 0)
            in_inference_mode(forge_model_management().load_models_gpu)([patcher], memory_required=required)
        elif not getattr(shared.sd_model, "lowvram", False):
            shared.sd_model.model.diffusion_model.to(devices.device)
    except Exception as e:
        print(f"[TensorRT] Could not restore the PyTorch UNet: {e}", flush=True)


@in_inference_mode
def make_room(size: int):
    """Has the manager unload models until `size` bytes are free (reloaded on demand). Never the
    UNet: it holds no VRAM while TensorRT runs, and unloading it drops its patches mid-sampling."""
    if not IS_FORGE_BACKEND or size <= 0:
        return
    try:
        mm = forge_model_management()
        keep = [loaded for loaded in mm.current_loaded_models if is_unet(loaded.model)]
        mm.free_memory(size, mm.get_torch_device(), keep_loaded=keep)
    except Exception as e:
        print(f"[TensorRT] Could not free VRAM: {e}", flush=True)


def minimum_inference_memory() -> int:
    return int(forge_model_management().minimum_inference_memory())


def weights_to_load(models) -> int:
    """VRAM the weights of `models`, except the UNet's, still need."""
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
    """Free VRAM plus what the manager could unload for `models`."""
    mm = forge_model_management()
    device = mm.get_torch_device()
    available = mm.get_free_memory(device)
    for loaded in list(mm.current_loaded_models):
        patcher = loaded.model
        if patcher is None or loaded.device != device:
            continue
        if any(patcher is m or (m is not None and m.is_clone(patcher)) for m in models):
            continue
        try:
            available += int(loaded.model_loaded_memory())
        except Exception:
            pass
    return available


@in_inference_mode
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
    unet = sd_unet.current_unet
    if unet is not None:
        try:
            getattr(unet, "release_engine", unet.deactivate)()
        except Exception as e:
            print(f"[TensorRT] Could not release the current UNet: {e}", flush=True)
        sd_unet.current_unet = None
        sd_unet.current_unet_option = None
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
        if not IS_FORGE_BACKEND:
            param = next(self.parameters(), None)
            if param is not None and param.device != x.device:
                self.to(x.device)
        sd_unet.current_unet = None  # A1111's own UNet hook would call it again
        try:
            return original(self, x, timesteps, context, *args, **kwargs)
        finally:
            sd_unet.current_unet = unet

    return forward


def _rebind(owner, name, original, replacement):
    """Replace a function, including in modules that did `from ... import name`."""
    setattr(owner, name, replacement)
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)  # getattr(module, name) imports lazy modules
        if isinstance(namespace, dict) and namespace.get(name) is original:
            setattr(module, name, replacement)


def _keep_loaded(memory_required, device, keep_loaded=None):
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
        # reForge's prepare still applies the UNet's patches (e.g. model_sampling), but loads it in
        # NO_VRAM mode and unloads nothing for it: its budget is for a UNet that won't run.
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
            callbacks["sampling_done"]()
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
    """Idempotent; runs before each generation to cover modules loaded since."""
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
