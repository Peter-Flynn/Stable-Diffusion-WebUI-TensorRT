"""TensorRT's GPU memory, from PyTorch's allocator.

By default TensorRT takes its memory (engine weights, execution contexts, refit memory) straight from the
driver, beside PyTorch's caching allocator: neither can use what the other holds, and reForge's count of
free VRAM (the driver's free memory plus PyTorch's unused cache) can't tell what TensorRT holds. Here the
TensorRT runtime allocates through PyTorch's allocator instead (TorchAllocator), so there's one pool and
one count: what TensorRT holds is PyTorch memory in use, what it frees is PyTorch's cache, for anything.

Every allocation is also entered in a ledger under its owner (an engine's model: `owner()` says whose
allocations are being made), so a model's size is what it holds, to the byte, and a load's peak is what
the load took at most. Nothing about memory is estimated from it: it's counted.

TRT_EXTENSION_TORCH_ALLOCATOR=0 leaves TensorRT on its own allocator (the ledger then stays empty, and
sizes come from the driver's free memory before and after)."""

import contextlib
import ctypes
import os
import threading

import torch

from .common import gib, log

_lock = threading.RLock()
_live = {}  # address given to TensorRT -> (address allocated, bytes, owner)
_held = {}  # owner -> bytes it holds
_peaks = {}  # owner -> the most it held since its peak was reset
_owners = []  # stack: whose allocations TensorRT is making now (None: nobody's in particular)
_failures = 0  # allocations PyTorch couldn't satisfy, counted
_allocator = None
_retired = None  # the allocator of a runtime set aside (retire), kept alive with it


def _pointer(memory) -> int:
    """An address TensorRT's bindings hand over: a capsule holding it (or an int, or None)."""
    if memory is None:
        return 0
    if isinstance(memory, int):
        return memory
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return int(get(memory, None) or 0)


def _capsule(address: int):
    """An address as TensorRT's bindings take it back: a capsule, or None for none."""
    if not address:
        return None
    new = ctypes.pythonapi.PyCapsule_New
    new.restype, new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(address, None, None)


def _allocate(size: int, alignment: int, stream) -> int:
    global _failures
    if size <= 0:
        return 0
    alignment = max(int(alignment or 1), 1)
    extra = alignment - 1 if alignment > 512 else 0  # PyTorch's blocks start on 512 bytes
    try:
        base = torch.cuda.caching_allocator_alloc(size + extra, torch.cuda.current_device(), int(stream) if stream else None)
    except torch.cuda.OutOfMemoryError:
        with _lock:
            _failures += 1
        return 0
    except Exception as e:
        log(f"PyTorch's allocator failed TensorRT's request for {gib(size)}: {e}")
        with _lock:
            _failures += 1
        return 0
    address = (base + alignment - 1) // alignment * alignment
    with _lock:
        owner = _owners[-1] if _owners else None
        _live[address] = (base, size, owner)
        _held[owner] = _held.get(owner, 0) + size
        for o in set(_owners):
            if o in _peaks:
                _peaks[o] = max(_peaks[o], _held.get(o, 0))
    return address


def _free(memory, stream) -> bool:
    address = _pointer(memory)
    if not address:
        return True
    with _lock:
        entry = _live.pop(address, None)
    if entry is None:
        return False
    base, size, owner = entry
    # TensorRT may have used it on another stream than the one PyTorch files it under: done there first
    if stream:
        torch.cuda.ExternalStream(int(stream)).synchronize()
    else:
        torch.cuda.synchronize()
    torch.cuda.caching_allocator_delete(base)
    with _lock:
        _held[owner] = _held.get(owner, 0) - size
        if not _held[owner]:
            del _held[owner]
    return True


def _make_allocator():
    import tensorrt as trt

    base = trt.IGpuAllocator  # with the asynchronous methods too, which TensorRT calls

    class TorchAllocator(base):
        """TensorRT's device allocations, from PyTorch's caching allocator (see the module). Addresses go
        back and forth as capsules, as TensorRT's bindings pass them. Nothing may raise back into TensorRT:
        a failure is None (a null address) or False, which TensorRT reports."""

        def __init__(self):
            base.__init__(self)

        def allocate_async(self, size, alignment, flags, stream):
            try:
                return _capsule(_allocate(size, alignment, stream))
            except Exception:
                return None

        def deallocate_async(self, memory, stream):
            try:
                return _free(memory, stream)
            except Exception:
                return False

        def allocate(self, size, alignment, flags):
            return self.allocate_async(size, alignment, flags, None)

        def deallocate(self, memory):
            return self.deallocate_async(memory, None)

        def reallocate(self, address, alignment, new_size):
            return None  # not resized in place: TensorRT allocates anew

    return TorchAllocator()


def attach(runtime) -> bool:
    """Gives the TensorRT runtime PyTorch's allocator, before it deserializes anything. Returns whether it
    did; if not, TensorRT keeps its own."""
    global _allocator
    if os.environ.get("TRT_EXTENSION_TORCH_ALLOCATOR", "").lower() in ("0", "false", "no"):
        log("TensorRT allocates its own GPU memory (TRT_EXTENSION_TORCH_ALLOCATOR is off).")
        return False
    try:
        allocator = _make_allocator()
        runtime.gpu_allocator = allocator
        _allocator = allocator  # TensorRT doesn't keep the Python object alive
        return True
    except Exception as e:
        log(f"Could not give TensorRT PyTorch's allocator, so it allocates its own GPU memory: {e}")
        return False


def attached() -> bool:
    return _allocator is not None


_proven = False  # an engine loaded with PyTorch's allocator: it works with these bindings


def proven():
    global _proven
    _proven = True


def failures() -> int:
    """Allocations PyTorch couldn't satisfy, so far."""
    with _lock:
        return _failures


def suspect(failures_before: int) -> bool:
    """Whether an engine that just failed to load could have failed for PyTorch's allocator: no engine has
    loaded with it yet (TensorRT's bindings could be taking addresses otherwise than their documentation
    says), and none of its allocations failed since `failures_before` (running out of VRAM isn't its doing,
    and loading again with more room is what fixes that)."""
    return _allocator is not None and not _proven and failures() == failures_before


def retire():
    """An engine that failed to load with PyTorch's allocator loaded on a runtime with TensorRT's own, which
    is used from now on (engine.py): what the failed load was given is freed, and the ledger counts from the
    driver from now on (see loading)."""
    global _allocator, _retired
    with _lock:
        entries = list(_live.values())
        _live.clear()
        _held.clear()
    torch.cuda.synchronize()
    for base, _, _ in entries:
        torch.cuda.caching_allocator_delete(base)
    _retired, _allocator = _allocator, None
    log("An engine failed to load with PyTorch's allocator and loaded with TensorRT's own, so TensorRT allocates its own GPU memory from now on.")


@contextlib.contextmanager
def owner(name):
    """TensorRT's allocations in here are `name`'s. (Not per thread: TensorRT may allocate from threads of
    its own, and loads run one at a time.)"""
    with _lock:
        _owners.append(name)
    try:
        yield
    finally:
        with _lock:
            _owners.pop()


def held(name) -> int:
    """Bytes `name` holds of TensorRT's allocations now."""
    with _lock:
        return _held.get(name, 0)


@contextlib.contextmanager
def measuring(name):
    """Within: `name`'s allocations, and the most it holds at once (yields a dict that gets "peak" and
    "failures", allocations that couldn't be satisfied, when the block ends)."""
    result = {}
    with _lock:
        _peaks[name] = _held.get(name, 0)
        failures = _failures
    try:
        with owner(name):
            yield result
    finally:
        with _lock:
            result["peak"] = _peaks.pop(name, 0)
            result["failures"] = _failures - failures


def _outside_pytorch() -> int:
    """VRAM in use that PyTorch's allocator doesn't hold (TensorRT's own, without PyTorch's allocator)."""
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


@contextlib.contextmanager
def loading(name):
    """Measures loading (or adding to) `name`'s engines: yields a dict that gets, when the block ends,
    "held" (what `name` holds then), "peak" (the most it held during the load) and "failures" (allocations
    PyTorch couldn't satisfy). Without PyTorch's allocator, from what the driver showed in use beyond
    PyTorch before and after, which is then all `name` is known to hold."""
    if attached():
        result, m = {}, {}
        try:
            with measuring(name) as m:
                yield result
        finally:  # also when the load failed: the caller asks whether it ran out
            result.update(held=held(name), peak=max(m.get("peak", 0), held(name)), failures=m.get("failures", 0))
        return
    before, result = _outside_pytorch(), {}
    try:
        yield result
    finally:
        grew = max(0, _outside_pytorch() - before)
        with _lock:
            _held[name] = _held.get(name, 0) + grew
        result.update(held=held(name), peak=held(name), failures=0)


def unloaded(name, nbytes: int):
    """`name` unloaded an engine that held `nbytes`: without PyTorch's allocator, that's how the ledger learns
    it (with it, every free is counted)."""
    if not attached():
        with _lock:
            left = _held.get(name, 0) - nbytes
            if left > 0:
                _held[name] = left
            else:
                _held.pop(name, None)


def forget(name):
    """`name` unloaded everything: without PyTorch's allocator, that's how the ledger learns it (with it,
    every free is counted, and anything left is what TensorRT didn't give back)."""
    if not attached():
        with _lock:
            _held.pop(name, None)


def summary() -> str:
    """For the memory diagnostics."""
    if not attached():
        return ""
    with _lock:
        parts = [f"{o or 'unattributed'} {gib(n)}" for o, n in sorted(_held.items(), key=lambda kv: -kv[1]) if n]
    return "TensorRT holds " + (", ".join(parts) if parts else "nothing")
