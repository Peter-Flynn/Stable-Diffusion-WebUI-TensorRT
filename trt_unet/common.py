import collections
import contextlib
import functools
import json
import logging
import os
import sys

import torch

from . import compat

logger = logging.getLogger("TensorRT")
compat.setup_logger(logger)

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


class Readback:
    # GPU values read on the CPU without a sync: send() copies them to pinned memory and records an event, and ready()
    # yields (tag, rows) for each send whose event has completed, in the order sent. Read buffers and events are reused.
    def __init__(self):
        self._pending = collections.deque()  # (tag, pinned rows, event)
        self._spare = {}  # rows' shape -> [(pinned rows, event)]

    def send(self, tag, *values):
        shape = (len(values), values[0].numel())
        dev = values[0].device if values[0].is_cuda else device()
        spare = self._spare.get(shape)
        if spare:
            host, event = spare.pop()
        else:
            with on_device(dev):
                host, event = torch.empty(shape, dtype=torch.float32, pin_memory=True), torch.cuda.Event()
        for row, value in zip(host, values):
            row.copy_(value.reshape(-1), non_blocking=True)
        event.record(torch.cuda.current_stream(dev))
        self._pending.append((tag, host, event))

    def ready(self, skip=None):
        while self._pending and (skip is None or self._pending[0][0] != skip) and self._pending[0][2].query():
            tag, host, event = self._pending.popleft()
            rows = host.tolist()
            self._spare.setdefault(tuple(host.shape), []).append((host, event))
            yield tag, rows

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def wait(self):  # until every send so far is ready()
        if self._pending:
            self._pending[-1][2].synchronize()

    def clear(self):
        self._pending.clear()


generation = 0


def new_generation():
    global generation
    generation += 1
    _notified.clear()


def mm():
    return compat.mm()


def device() -> torch.device:
    return compat.device(mm())


def on_device(dev=None):
    # Binds the calling thread's CUDA context for libraries (TensorRT, cudart) that read the active thread device.
    dev = torch.device(dev) if dev is not None else device()
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


def inference_mode(fn):
    # reForge creates and moves its weights under torch.inference_mode(). Loading or unloading a model outside it
    # breaks weights restored after a LoRA ("Inference tensors do not track version counter").
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper


def gpu_tag() -> str:
    major, minor = torch.cuda.get_device_capability(device())
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


def _free_now(device, trim_cache: bool = True, at_least: int = None) -> int:
    try:
        if torch.device(device).type == "cuda":
            if pooled():
                if at_least is not None or not trim_cache:
                    driver = int(torch.cuda.mem_get_info(device)[0])
                    if not trim_cache or driver >= at_least:
                        return driver
                trim()
                return int(torch.cuda.mem_get_info(device)[0])
            driver = int(torch.cuda.mem_get_info(device)[0])
            if at_least is not None and driver >= at_least:
                return driver
            stats = torch.cuda.memory_stats(device)
            cached = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
            return driver + max(0, cached)
    except Exception:
        pass
    return mm().get_free_memory(device)


def free(device, trim: bool = True, at_least: int = None) -> int:
    res = reserve()
    want = None if at_least is None else at_least + res
    return _free_now(device, trim_cache=trim, at_least=want) - res


def obtainable(device, keep=(), engines: bool = True, trim: bool = True, at_least: int = None) -> int:
    m = mm()
    held = sum(
        lm.model_loaded_memory()
        for lm in m.current_loaded_models
        if lm.device == device and lm.model is not None and not any(lm.model is k for k in keep)
        and (engines or not getattr(lm.model, "evicted_last", False))
    )
    want = None if at_least is None else max(0, at_least - held)
    return _free_now(device, trim_cache=trim, at_least=want) + held


def register(model):
    # What load_models_gpu does, minus its free_memory call, which would evict by reForge's own estimate (1.3x the
    # model's size, at least 0.8 GiB). Engine models make exactly the room they need inside partially_load.
    m = mm()
    entry = next((lm for lm in m.current_loaded_models if lm.model is model), None)
    if entry is not None:
        m.current_loaded_models.remove(entry)
        if getattr(model, "loaded", False) and getattr(entry, "model_finalizer", None) is not None and getattr(entry, "real_model", None) is not None and entry.real_model() is getattr(model, "model", None):
            entry.currently_used = True
            m.current_loaded_models.insert(0, entry)
            return
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
    had = free(device, at_least=nbytes)
    for _ in range(16):
        if had >= nbytes:
            break
        # Loops because a model reForge unloads partially frees a little less than asked. The request is in reForge's
        # own count of free memory, which can be lower than free().
        with torch.inference_mode():
            m.free_memory(m.get_free_memory(device) + nbytes - had, device, keep_loaded=list(keep_loaded))
        now = free(device, at_least=nbytes)
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
        left = refits.resident()
        still = f", {gib(left)} of it still there" if left < nbytes else ""
        parts.append(f"{count} engine{'s' if count > 1 else ''} kept ({gib(nbytes)}{still}), counted as free")
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
        dev = device()
        free, total = torch.cuda.mem_get_info(dev)
        stats = torch.cuda.memory_stats(dev)
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
