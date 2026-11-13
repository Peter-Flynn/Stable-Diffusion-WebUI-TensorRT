import functools
import json
import logging
import os
import sys

import torch

# Everything this extension says goes through here, at reForge's --loglevel and in its format.
logger = logging.getLogger("TensorRT")

_notified = set()


def notify(message: str, error: BaseException = None):
    """A warning in the console and in the UI, once per generation. With an unexpected `error`, an error
    in the console, with its traceback."""
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


_json = {}  # path -> (what identifies the file's version, its contents)


def read_json(path: str):
    """A JSON file's contents, parsed once for each version of the file: engines' files are read on every
    sampling pass and decode. Each version is a new file (they're written beside and moved into place), so
    its inode, modification time and size tell it apart. Shared between callers, so never changed."""
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
    """For people: "0.4s", "12s", "3m 05s"."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


# --- Making room in VRAM: exact sizes, through reForge's model manager


BUFFER = 0.10  # added to every estimate of VRAM something is about to take, before it's weighed or room is made


def padded(nbytes: int) -> int:
    """An estimate with the buffer on top."""
    return int(nbytes * (1 + BUFFER))


def usable(room: int) -> int:
    """What an estimate may be for it to fit `room` with the buffer on top."""
    return int(room / (1 + BUFFER))


def pooled() -> bool:
    """Whether what PyTorch has freed is only reliably usable once it's back with the driver: with
    --cuda-malloc it stays in the driver's pool in the blocks it was freed in, which a later request of
    another size often can't have (seen: 1.7 GiB unused and a 0.5 GiB request failing); and TensorRT on
    its own allocator can't have PyTorch's cache at all."""
    from . import gpu

    return not gpu.attached() or gpu.backend() != "native"


def trim():
    """Gives what PyTorch holds unused back to the driver, where it has to be to be used (pooled). (It waits
    for the GPU to be done with it first.)"""
    if pooled():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def reserve() -> int:
    """The VRAM reForge keeps free for what no allocator counts (other applications, CUDA's own code): its
    --reserve-vram, or its default for the OS. The only margin anything here keeps."""
    try:
        return int(mm().extra_reserved_memory())
    except Exception:
        return 0


def _free_now(device) -> int:
    """VRAM an allocation could have now: what the driver has free, and, with PyTorch's native allocator,
    what it holds unused in its cache. Otherwise (pooled) that goes back to the driver first, and what's
    still unused then isn't counted: it's pieces of blocks still in use, which nothing can have (seen: 130
    to 340 MiB). reForge counts the driver's free memory plus PyTorch's reserved less its active whatever
    the allocator. reForge's count is used where the device isn't CUDA's."""
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
    entry = next((lm for lm in m.current_loaded_models if lm.model is model), None)
    if entry is not None:
        m.current_loaded_models.remove(entry)
        if entry.model_finalizer is not None:
            entry.model_finalizer.detach()  # model_load makes a new one
            entry.model_finalizer = None
    else:
        entry = m.LoadedModel(model)
    entry.model_load()
    entry.currently_used = True
    m.current_loaded_models.insert(0, entry)


def make_room(nbytes: int, device, keep_loaded=()) -> int:
    """Has reForge's manager evict models (least recently used first; engines last, see runtime.py) until
    `nbytes` are free beside the reserve: nothing, if they are already; again while it's short and the
    asking freed something (a model reForge unloads in part gives back a little less than it was asked
    for). Returns what's free then, beside the reserve."""
    m = mm()
    had = free(device)
    for _ in range(16):
        if had >= nbytes:
            break
        with torch.inference_mode():  # models unloaded outside it are left unusable (see inference_mode)
            # By reForge's own count, which can be short of what's free (_free_now): what it counts free
            # now, and the shortfall more
            m.free_memory(m.get_free_memory(device) + nbytes - had, device, keep_loaded=list(keep_loaded))
        now = free(device)
        if now <= had:  # nothing more goes
            break
        had = now
    return had


# --- Memory diagnostics, at DEBUG: they only read counters, and change nothing.


def _loaded_models() -> str:
    out = []
    for lm in mm().current_loaded_models:
        model = lm.model
        if model is not None:
            name = getattr(model, "owner", None) or type(getattr(model, "model", model)).__name__
            out.append(f"{name} {gib(lm.model_loaded_memory())}")
    return ", ".join(out) or "nothing"


def _tensorrt() -> str:
    """What TensorRT holds in VRAM through PyTorch's allocator, by owner, and its activation memory."""
    from . import gpu
    from .engine import Arena

    parts = [f"{owner} {gib(n)}" for owner, n in gpu.by_owner()]
    activations = sum(a.memory.numel() for a in list(Arena.instances) if a.memory is not None)
    if activations:
        parts.append(f"activations {gib(activations)}")
    return ", ".join(parts)


def _ram() -> str:
    """The RAM free (as refits.memory counts it), what this extension holds there, and what's in swap."""
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
    """Logs, at DEBUG, what holds VRAM and RAM `when` ("after loading ...")."""
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
