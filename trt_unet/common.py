import functools
import json
import logging
import os
import sys

import torch

logger = logging.getLogger("TensorRT")

_notified = set()


def notify(message: str, error: BaseException = None):
    if message in _notified:
        return
    _notified.add(message)
    if error is not None:
        logger.error(message, exc_info=error)
    else:
        logger.warning(message)
    try:
        import gradio as gr

        gr.Warning(f"TensorRT: {message}")
    except Exception:
        pass


generation = 0


def new_generation():
    global generation
    generation += 1
    _notified.clear()


def mm():
    import ldm_patched.modules.model_management as model_management

    return model_management


def inference_mode(fn):
    # reForge creates and moves its weights under torch.inference_mode(). Loading or unloading a model outside it
    # breaks weights restored after a LoRA ("Inference tensors do not track version counter").
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper


def gpu_tag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


MIN_TRT = 11


def trt_version() -> str:
    import tensorrt

    if int(tensorrt.__version__.split(".")[0]) < MIN_TRT:
        raise RuntimeError(f"TensorRT {tensorrt.__version__} is installed; this extension needs {MIN_TRT} or later")
    return tensorrt.__version__


def trt_tag(version: str = None) -> str:
    return "trt" + ".".join((version or trt_version()).split(".")[:2])


_json = {}


def read_json(path: str):
    # Engine metadata is read on every sampling pass and decode, so parsed once per version of the file. Every write
    # is a new file renamed into place, so (inode, mtime, size) identifies a version. The returned dict is shared:
    # never mutate it.
    stat = os.stat(path)
    version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _json.get(path)
    if cached is None or cached[0] != version:
        with open(path, encoding="utf-8") as f:
            cached = _json[path] = (version, json.load(f))
    return cached[1]


def gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


BUFFER = 0.10


def padded(nbytes: int) -> int:
    return int(nbytes * (1 + BUFFER))


def usable(room: int) -> int:
    return int(room / (1 + BUFFER))


def pooled() -> bool:
    # True when memory PyTorch has freed can only be relied on after empty_cache() returns it to the driver. With
    # --cuda-malloc it stays in the driver's pool in the block sizes it was freed in (seen: 1.7 GiB unused, a 0.5 GiB
    # request failing). With TensorRT on its own allocator, TensorRT can't use PyTorch's cache at all.
    from . import gpu

    return not gpu.attached() or gpu.backend() != "native"


def trim():
    if pooled():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def reserve() -> int:
    try:
        return int(mm().extra_reserved_memory())
    except Exception:
        return 0


def _free_now(device) -> int:
    try:
        if torch.device(device).type == "cuda":
            cached = 0
            if pooled():
                trim()
            else:
                stats = torch.cuda.memory_stats(device)
                cached = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
            return torch.cuda.mem_get_info(device)[0] + max(0, cached)
    except Exception:
        pass
    return mm().get_free_memory(device)


def free(device) -> int:
    return _free_now(device) - reserve()


def obtainable(device, keep=(), engines: bool = True) -> int:
    m = mm()
    held = sum(
        lm.model_loaded_memory()
        for lm in m.current_loaded_models
        if lm.device == device and lm.model is not None and not any(lm.model is k for k in keep)
        and (engines or not getattr(lm.model, "evicted_last", False))
    )
    return _free_now(device) + held


def register(model):
    # What load_models_gpu does, minus its free_memory call, which would evict by reForge's own estimate (1.3x the
    # model's size, at least 0.8 GiB). Engine models make exactly the room they need inside partially_load.
    m = mm()
    entry = next((lm for lm in m.current_loaded_models if lm.model is model), None)
    if entry is not None:
        m.current_loaded_models.remove(entry)
        if entry.model_finalizer is not None:
            entry.model_finalizer.detach()  # model_load() creates a new one
            entry.model_finalizer = None
    else:
        entry = m.LoadedModel(model)
    entry.model_load()
    entry.currently_used = True
    m.current_loaded_models.insert(0, entry)


def make_room(nbytes: int, device, keep_loaded=()) -> int:
    m = mm()
    had = free(device)
    for _ in range(16):
        if had >= nbytes:
            break
        # Loops because a model reForge unloads partially frees a little less than asked. The request is in reForge's
        # own count of free memory, which can be lower than free().
        with torch.inference_mode():
            m.free_memory(m.get_free_memory(device) + nbytes - had, device, keep_loaded=list(keep_loaded))
        now = free(device)
        if now <= had:
            break
        had = now
    return had


def _loaded_models() -> str:
    out = []
    for lm in mm().current_loaded_models:
        model = lm.model
        if model is not None:
            name = getattr(model, "owner", None) or type(getattr(model, "model", model)).__name__
            out.append(f"{name} {gib(lm.model_loaded_memory())}")
    return ", ".join(out) or "nothing"


def _tensorrt() -> str:
    from . import gpu
    from .engine import Arena

    parts = [f"{owner} {gib(n)}" for owner, n in gpu.by_owner()]
    activations = sum(a.memory.numel() for a in list(Arena.instances) if a.memory is not None)
    if activations:
        parts.append(f"activations {gib(activations)}")
    return ", ".join(parts)


def _ram() -> str:
    from . import refits, strips

    supply, _, total = refits.memory()
    parts = [f"{gib(supply)} free of {gib(total)}"]
    count, nbytes = refits.held()
    if count:
        parts.append(f"{count} engine{'s' if count > 1 else ''} kept ({gib(nbytes)})")
    _, pinned, _ = strips.host_memory()
    if pinned:
        parts.append(f"{gib(pinned)} for VAE strips")
    if sys.platform.startswith("linux"):
        m = refits._meminfo()
        swapped = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
        if swapped > 0:
            parts.append(f"{gib(swapped)} in swap")
    return "; ".join(parts)


def memory(when: str):
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        allocated, reserved = (int(stats.get(f"{k}_bytes.all.current", 0)) for k in ("allocated", "reserved"))
        lines = [
            f"VRAM      {gib(total - free)} used of {gib(total)}; PyTorch has {gib(allocated)} allocated, {gib(max(0, reserved - allocated))} cached",
            f"Models    {_loaded_models()}",
        ]
        tensorrt = _tensorrt()
        if tensorrt:
            lines.append(f"TensorRT  {tensorrt}")
        lines.append(f"RAM       {_ram()}")
        logger.debug(f"Memory {when}:\n" + "\n".join("    " + line for line in lines))
    except Exception as e:
        logger.debug(f"Memory {when}: couldn't be read ({e})")
