"""TensorRT's GPU memory: taken through PyTorch's allocator, and counted.

Left to itself, TensorRT takes its memory (engine weights, execution contexts, refit memory) beside PyTorch:
from the driver, or from the driver's pool that reForge's --cuda-malloc has PyTorch use too. Neither can use
what the other holds unused, reForge's count of free VRAM (the driver's free memory plus PyTorch's unused
cache) can't tell what TensorRT holds, and with --cuda-malloc TensorRT's own frees into that pool leave the
pool's count of memory in use above what it holds (seen: by 1.3 GiB after a few dozen generations; PyTorch
reads that count as its own, and reForge then counts that much less VRAM free than there is). Here the
TensorRT runtime allocates through PyTorch's allocator instead (TorchAllocator), whichever backend PyTorch
is on: what TensorRT holds is PyTorch memory in use, what it frees is freed the way PyTorch frees its
tensors, and is PyTorch's cache, for anything.

What an owner (an engine's model: `owner()` says whose allocations are being made) holds is counted as its
engines load: what loading adds to the VRAM in use (in_use), which is what it takes whoever TensorRT got it
from, and unloading takes that off again. Through the allocator every allocation is also entered in a
ledger under its owner, which gives the most a load held on the way and says when one ran out. Nothing
about memory is estimated from either: it's counted.

TensorRT stays on its own allocator with TRT_EXTENSION_TORCH_ALLOCATOR=0, or once an engine that fails
with PyTorch's loads with its own (engine.py); an owner's size is counted the same way then."""

import contextlib
import ctypes
import os
import threading

import torch

from .common import gib, logger

_lock = threading.RLock()
_live = {}  # address given to TensorRT -> (address allocated, bytes, owner)
_ledger = {}  # owner -> bytes TensorRT holds for it through PyTorch's allocator
_peaks = {}  # owner -> the most the ledger had for it since its peak was reset
_held = {}  # owner -> what loading its engines added to the VRAM in use
_owners = []  # stack: whose allocations TensorRT is making now (None: nobody's in particular)
_failures = 0  # allocations PyTorch couldn't satisfy, counted
_allocator = None
_retired = None  # the allocator of a runtime set aside (retire), kept alive with it


def _pointer(memory) -> int:
    """An address TensorRT's bindings hand over: an int (or a capsule holding it, or None)."""
    if memory is None:
        return 0
    if isinstance(memory, int):
        return memory
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return int(get(memory, None) or 0)


def _allocate(size: int, alignment: int) -> int:
    """`size` bytes from PyTorch's allocator, on PyTorch's current stream (as its tensors are: TensorRT may
    use them on streams of its own, and _free waits for those)."""
    global _failures
    if size <= 0:
        return 0
    alignment = max(int(alignment or 1), 1)
    extra = alignment - 1  # wherever PyTorch's block starts, an aligned address is in it with `size` after
    base = None
    for again in (False, True):
        try:
            base = torch.cuda.caching_allocator_alloc(size + extra, torch.cuda.current_device())
            break
        except torch.cuda.OutOfMemoryError:
            break
        except Exception as e:
            # With --cuda-malloc, PyTorch's allocator first raises whatever CUDA error an earlier call left
            # behind on this thread (TensorRT's own, say), which clears it: that's not this request failing
            if again:
                logger.debug(f"PyTorch's allocator refused TensorRT {gib(size)}: {e}")
    if base is None:
        with _lock:
            _failures += 1
        return 0
    address = (base + alignment - 1) // alignment * alignment
    with _lock:
        owner = _owners[-1] if _owners else None
        _live[address] = (base, size, owner)
        _ledger[owner] = _ledger.get(owner, 0) + size
        if owner in _peaks:
            _peaks[owner] = max(_peaks[owner], _ledger[owner])
    return address


def _free(memory) -> bool:
    address = _pointer(memory)
    if not address:
        return True
    with _lock:
        entry = _live.pop(address, None)
    if entry is None:
        return False
    base, size, owner = entry
    for again in (False, True):
        try:
            torch.cuda.synchronize()  # TensorRT may have used it on streams of its own: done there first
            torch.cuda.caching_allocator_delete(base)
            break
        except Exception as e:  # an earlier call's CUDA error, raised here and cleared by that (see _allocate)
            if again:
                logger.warning(f"Couldn't free {gib(size)} of TensorRT's VRAM: {e}")
                with _lock:
                    _live[address] = entry  # still held, and counted so
                return False
    with _lock:
        _ledger[owner] = _ledger.get(owner, 0) - size
        if not _ledger[owner]:
            del _ledger[owner]
    return True


def _make_allocator():
    import tensorrt as trt

    class TorchAllocator(trt.IGpuAllocator):
        """TensorRT's device allocations, from PyTorch's allocator (see the module). TensorRT's bindings call
        allocate and deallocate (its asynchronous calls come down to them), and read the address as an int
        (whatever their documentation's signatures say: a capsule is "not an int" to them, and the
        allocation failed). Nothing may raise back into TensorRT: a failure is 0 (a null address) or False,
        which TensorRT reports."""

        def __init__(self):
            trt.IGpuAllocator.__init__(self)

        def allocate(self, size, alignment, flags):
            try:
                return _allocate(size, alignment)
            except Exception:
                return 0

        def deallocate(self, memory):
            try:
                return _free(memory)
            except Exception:
                return False

        def allocate_async(self, size, alignment, flags, stream):
            return self.allocate(size, alignment, flags)

        def deallocate_async(self, memory, stream):
            return self.deallocate(memory)

        def reallocate(self, address, alignment, new_size):
            return 0  # not resized in place: TensorRT allocates anew

    return TorchAllocator()


def backend() -> str:
    """PyTorch's CUDA allocator: "native", or "cudaMallocAsync" (reForge's --cuda-malloc)."""
    try:
        return torch.cuda.get_allocator_backend()
    except Exception:
        return "native"


def attach(runtime) -> bool:
    """Gives the TensorRT runtime PyTorch's allocator, before it deserializes anything. Returns whether it
    did; if not, TensorRT keeps its own."""
    global _allocator
    if os.environ.get("TRT_EXTENSION_TORCH_ALLOCATOR", "").lower() in ("0", "false", "no"):
        logger.info("TensorRT manages its own VRAM (TRT_EXTENSION_TORCH_ALLOCATOR is off).")
        return False
    try:
        allocator = _make_allocator()
        runtime.gpu_allocator = allocator
        _allocator = allocator  # TensorRT doesn't keep the Python object alive
        return True
    except Exception as e:
        logger.warning(f"Couldn't give TensorRT PyTorch's allocator, so TensorRT manages its own VRAM: {e}")
        return False


def attached() -> bool:
    return _allocator is not None


_proven = False  # an engine ran with PyTorch's allocator: it works with these bindings


def proven():
    global _proven
    _proven = True


def failures() -> int:
    """Allocations PyTorch couldn't satisfy, so far."""
    with _lock:
        return _failures


def suspect(failures_before: int, name=None) -> bool:
    """Whether `name`'s engine that just failed to load could have failed for PyTorch's allocator: no engine
    has run with it yet (TensorRT's bindings could be taking addresses otherwise than their documentation
    says), none of its allocations failed since `failures_before` (running out of VRAM isn't its doing, and
    loading again with more room is what fixes that), and no other owner's engines hold memory through it
    (retiring it frees what it gave)."""
    with _lock:
        others = any(n for o, n in _ledger.items() if o is not None and o != name)
        return _allocator is not None and not _proven and _failures == failures_before and not others


def retire():
    """An engine that failed to load with PyTorch's allocator is loaded on a runtime with TensorRT's own,
    which is used from now on (engine.py): what the failed load was given is freed."""
    global _allocator, _retired
    with _lock:
        entries = list(_live.values())
        _live.clear()
        _ledger.clear()
    for base, _, _ in entries:
        for again in (False, True):
            try:
                torch.cuda.synchronize()
                torch.cuda.caching_allocator_delete(base)
                break
            except Exception:  # as in _free
                pass
    _retired, _allocator = _allocator, None


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
    """Bytes of VRAM `name`'s engines hold: what loading them added to the VRAM in use."""
    with _lock:
        return _held.get(name, 0)


def ledger(name) -> int:
    """Bytes TensorRT holds for `name` through PyTorch's allocator now."""
    with _lock:
        return _ledger.get(name, 0)


def in_use() -> int:
    """VRAM in use on the device, by anyone: what the driver has handed out, less what PyTorch holds unused
    (its reserved less its allocated). What loading an engine adds to it is what the engine holds, wherever
    TensorRT took it from: PyTorch's allocator, the driver, or (--cuda-malloc) the driver's pool whose
    counters PyTorch reports as its own."""
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return total - free - int(stats.get("reserved_bytes.all.current", 0)) + int(stats.get("allocated_bytes.all.current", 0))


@contextlib.contextmanager
def loading(name):
    """Counts loading (or adding to) `name`'s engines: yields a dict that gets, when the block ends, "grew"
    (what it added to the VRAM in use: at least what the ledger entered for `name`), "held" (what `name`
    holds then), "peak" (the most it held on the way, by the ledger; what it holds then, without one) and
    "failures" (allocations PyTorch couldn't satisfy). Also when the load failed: what it left is still
    there until it's unloaded, and the caller asks whether it ran out."""
    result = {}
    try:
        before = in_use()
    except Exception:
        before = None
    with _lock:
        entered = _ledger.get(name, 0)
        _peaks[name] = entered
        failed = _failures
    try:
        with owner(name):
            yield result
    finally:
        try:
            grew = max(0, in_use() - before) if before is not None else 0
        except Exception:
            grew = 0
        with _lock:
            through = _ledger.get(name, 0) - entered
            most = _peaks.pop(name, entered) - entered
            grew = max(grew, through)
            base = _held.get(name, 0)
            _held[name] = base + grew
            # The most on the way: the ledger's most, and whatever came from elsewhere than the ledger
            result.update(grew=grew, held=base + grew, peak=base + max(grew, most + max(0, grew - max(through, 0))), failures=_failures - failed)


def counting(name) -> bool:
    """Whether a load of `name`'s is being counted now (loading)."""
    with _lock:
        return name in _peaks


def grew(name, nbytes: int):
    """`name`'s engines took `nbytes` more after they loaded (as one first ran)."""
    if nbytes > 0:
        with _lock:
            _held[name] = _held.get(name, 0) + nbytes


def unloaded(name, nbytes: int):
    """`name` unloaded an engine that held `nbytes`."""
    with _lock:
        left = _held.get(name, 0) - nbytes
        if left > 0:
            _held[name] = left
        else:
            _held.pop(name, None)


def forget(name):
    """`name` unloaded everything."""
    with _lock:
        _held.pop(name, None)


def by_owner() -> list:
    """For the memory diagnostics: [(owner, bytes)] that TensorRT holds through PyTorch's allocator, the
    most first."""
    with _lock:
        return [(o or "unattributed", n) for o, n in sorted(_ledger.items(), key=lambda kv: -kv[1]) if n]
