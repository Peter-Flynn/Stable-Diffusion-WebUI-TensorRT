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


# --- Making room in VRAM: exact sizes, through reForge's model manager


def reserve() -> int:
    """The VRAM reForge keeps free for what no allocator counts (other applications, CUDA's own code): its
    --reserve-vram, or its default for the OS. The only margin anything here keeps."""
    try:
        return int(mm().extra_reserved_memory())
    except Exception:
        return 0


def _free_now(device) -> int:
    """VRAM an allocation could have now: what the driver has free, and what PyTorch holds unused in its
    cache. reForge counts the same (the driver's free memory, plus PyTorch's reserved less its active),
    but with --cuda-malloc PyTorch can report less reserved than active, and reForge's count then falls
    short of what's free by the difference (seen: 1.7 GiB, which had it evict a UNet engine for a decode
    that fit). reForge's count is used where the device isn't CUDA's."""
    try:
        if torch.device(device).type == "cuda":
            stats = torch.cuda.memory_stats(device)
            cached = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
            return torch.cuda.mem_get_info(device)[0] + max(0, cached)
    except Exception:
        pass
    return mm().get_free_memory(device)


def free(device) -> int:
    """VRAM free beside the reserve (_free_now)."""
    return _free_now(device) - reserve()


def obtainable(device, keep=(), engines: bool = True) -> int:
    """The most VRAM making room could give: what's free now (PyTorch's unused cache included, which
    TensorRT allocates from too), and what every loaded model but those in `keep` holds; UNet engines' too
    (models `evicted_last`) unless not `engines`."""
    m = mm()
    held = sum(
        lm.model_loaded_memory()
        for lm in m.current_loaded_models
        if lm.device == device and lm.model is not None and not any(lm.model is k for k in keep)
        and (engines or not getattr(lm.model, "evicted_last", False))
    )
    return _free_now(device) + held


def register(model):
    """Has reForge's manager hold `model` as loaded and the most recently used, as load_models_gpu does but
    without the room that makes first by its own estimate (1.3 times the model's size, and at least its
    minimum for inference, 0.8 GiB beside the reserve): an engine model makes exactly the room it needs as it
    loads, and as it runs. Raises what loading it raises, holding nothing then."""
    m = mm()
    if not hasattr(m, "LoadedModel"):  # a reForge without it: its own way
        m.load_models_gpu([model])
        return
    entry = next((lm for lm in m.current_loaded_models if lm.model is model), None)
    if entry is not None:
        m.current_loaded_models.remove(entry)
        if getattr(entry, "model_finalizer", None) is not None:
            entry.model_finalizer.detach()  # model_load makes a new one
            entry.model_finalizer = None
    else:
        entry = m.LoadedModel(model)
    entry.model_load()
    entry.currently_used = True
    m.current_loaded_models.insert(0, entry)


def make_room(nbytes: int, device, keep_loaded=()) -> int:
    """Has reForge's manager evict models (least recently used first; engines last, see runtime.py) until
    `nbytes` are free beside the reserve: nothing, if they are already. Returns what's free then, beside
    the reserve."""
    from . import gpu

    m = mm()
    short = nbytes - free(device)
    if short > 0:
        with torch.inference_mode():  # models unloaded outside it are left unusable (see inference_mode)
            # By reForge's own count, which can be short of what's free (_free_now): what it counts free
            # now, and the shortfall more
            m.free_memory(m.get_free_memory(device) + short, device, keep_loaded=list(keep_loaded))
    if not gpu.attached():  # TensorRT allocates from the driver: what PyTorch has cached isn't free to it
        torch.cuda.empty_cache()
    return free(device)


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
        from . import gpu

        parts.append(gpu.summary())
    except Exception:
        pass
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
    return ", ".join(p for p in parts if p)


def _ram() -> str:
    """The RAM the system could hand over without paging anything out, as the engines kept in RAM count it
    (refits._supply), of all there is."""
    try:
        import psutil

        from . import refits

        return f"RAM {gib(refits._supply())} to spare of {gib(psutil.virtual_memory().total)}"
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
        try:
            backend = torch.cuda.get_allocator_backend()
        except Exception:
            backend = "?"
        parts = [
            f"VRAM {gib(used)} used of {gib(total)}, {gib(free)} free (reForge counts {gib(free + reserved - active)} free)",
            f"PyTorch ({backend}) {gib(allocated)} allocated + {gib(reserved - allocated)} cached",
            outside,
            f"loaded: {_loaded_models()}",
            _ours(),
            _ram(),
            extra,
        ]
        log(f"mem | {where} | " + " | ".join(p for p in parts if p))
    except Exception as e:
        log(f"mem | {where} | unreadable ({type(e).__name__}: {e})")
