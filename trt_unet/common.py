import functools

import torch


def log(message: str):
    print(f"[TensorRT] {message}", flush=True)


_notified = set()


def notify(message: str, once: bool = True):
    """Console and UI toast; `once` per generation."""
    if once and message in _notified:
        return
    _notified.add(message)
    log(message)
    try:
        import gradio as gr

        gr.Warning(f"TensorRT: {message}")
    except Exception:
        pass


generation = 0  # counts generations, for what's computed once per generation


def new_generation():
    global generation
    generation += 1
    _notified.clear()


def mm():
    import ldm_patched.modules.model_management as model_management

    return model_management


def inference_mode(fn):
    """reForge creates and moves its weights in inference mode; moving them outside it leaves weights
    restored after a LoRA unusable ("Inference tensors do not track version counter")."""

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
    """The installed TensorRT's version; raises if it's missing or older than this extension needs."""
    import tensorrt

    if int(tensorrt.__version__.split(".")[0]) < MIN_TRT:
        raise RuntimeError(f"TensorRT {tensorrt.__version__} is installed; this extension needs {MIN_TRT} or later")
    return tensorrt.__version__


def trt_tag(version: str = None) -> str:
    return "trt" + ".".join((version or trt_version()).split(".")[:2])


def gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


# --- Memory diagnostics: these only read counters, and change nothing.

_nvml = None  # NVML's handle for this GPU; False once NVML turned out not to be usable


def _process_vram():
    """VRAM held by this process by the driver's own count (NVML), or None where it can't tell (no pynvml,
    or Windows, whose driver doesn't count it per process)."""
    global _nvml
    if _nvml is False:
        return None
    try:
        import os

        import pynvml

        if _nvml is None:
            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            _nvml = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml):
            if p.pid == os.getpid() and p.usedGpuMemory:
                return int(p.usedGpuMemory)
        return None
    except Exception:
        _nvml = False
        return None


def _loaded_models() -> str:
    out = []
    for lm in mm().current_loaded_models:
        model = lm.model
        if model is not None:
            out.append(f"{type(getattr(model, 'model', model)).__name__} {gib(lm.model_loaded_memory())}")
    return ", ".join(out) or "nothing"


def _ours() -> str:
    """What this extension holds outside the models' sizes: TensorRT activation arenas (PyTorch's memory),
    the strips' pinned RAM, engines kept in RAM."""
    parts = []
    try:
        from .engine import Arena

        arenas = [a.memory.numel() for a in list(Arena.instances) if a.memory is not None]
        parts.append(f"activation arenas {gib(sum(arenas))} ({len(arenas)})")
    except Exception:
        pass
    try:
        from . import strips

        kind, nbytes, _ = strips.host_memory()
        if nbytes:
            parts.append(f"strips' {kind} RAM {gib(nbytes)}")
    except Exception:
        pass
    try:
        from . import refits

        count, nbytes = refits.held()
        if count:
            parts.append(f"{count} engines kept in RAM {gib(nbytes)}")
    except Exception:
        pass
    return ", ".join(parts)


def _ram() -> str:
    try:
        import psutil

        ram = psutil.virtual_memory()
        return f"RAM {gib(ram.available)} available of {gib(ram.total)}"
    except Exception as e:
        return f"RAM unreadable ({type(e).__name__}: {e})"


def memory(where: str, extra: str = ""):
    """Logs who holds VRAM and RAM at this point (diagnostics: it reads counters, and changes nothing).
    "Outside PyTorch" is VRAM that PyTorch's allocator doesn't hold: TensorRT's own allocations (engine
    weights, refit memory), CUDA's context and kernels, and, where the driver can't count this process
    alone, other processes."""
    try:
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        allocated, active, reserved = (int(stats.get(f"{k}_bytes.all.current", 0)) for k in ("allocated", "active", "reserved"))
        used, process = total - free, _process_vram()
        if process is not None:
            outside = f"outside PyTorch {gib(process - reserved)} (this process holds {gib(process)}; other processes {gib(used - process)})"
        else:
            outside = f"outside PyTorch {gib(used - reserved)} (other processes included)"
        parts = [
            f"VRAM {gib(used)} used of {gib(total)}, {gib(free)} free (reForge counts {gib(free + reserved - active)} free)",
            f"PyTorch {gib(allocated)} allocated + {gib(reserved - allocated)} cached",
            outside,
            f"loaded: {_loaded_models()}",
            _ours(),
            _ram(),
            extra,
        ]
        log(f"mem | {where} | " + " | ".join(p for p in parts if p))
    except Exception as e:
        log(f"mem | {where} | unreadable ({type(e).__name__}: {e})")
