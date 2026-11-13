# TensorRT allocates its VRAM through PyTorch's caching allocator (TorchAllocator) instead of its own. Otherwise
# neither can reuse what the other holds unused, reForge's count of free VRAM can't see TensorRT's memory, and with
# --cuda-malloc TensorRT's frees into the shared driver pool leave the pool's in-use count too high (seen: 1.3 GiB
# after a few dozen generations), so reForge thinks less VRAM is free than is. TRT_EXTENSION_TORCH_ALLOCATOR=0, or an
# engine that only loads on TensorRT's own allocator (engine._deserialize, engine.own_allocator), turns this off.
import contextlib
import ctypes
import os
import threading

import torch

from .common import gib, logger

_lock = threading.RLock()
# An "owner" is the model an engine belongs to (TrtModel.owner, VaeModel.owner).
_live = {}  # address given to TensorRT -> (address PyTorch allocated, bytes, owner)
_ledger = {}  # owner -> bytes TensorRT currently holds through PyTorch's allocator
_peaks = {}
_held = {}  # owner -> VRAM its engines hold: the growth in in_use() across their loads (counts any allocator)
_owners = []
_failures = 0
_allocator = None
_retired = None  # the allocator of a runtime replaced by retire(): TensorRT may still call it


def _pointer(memory) -> int:
    if memory is None:
        return 0
    if isinstance(memory, int):
        return memory
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return int(get(memory, None) or 0)


def _allocate(size: int, alignment: int) -> int:
    global _failures
    if size <= 0:
        return 0
    alignment = max(int(alignment or 1), 1)
    extra = alignment - 1  # wherever PyTorch's block starts, it contains an aligned address with `size` bytes after it
    base = None
    for again in (False, True):
        try:
            base = torch.cuda.caching_allocator_alloc(size + extra, torch.cuda.current_device())
            break
        except torch.cuda.OutOfMemoryError:
            break
        except Exception as e:
            # With --cuda-malloc, PyTorch's allocator first raises any CUDA error an earlier call (e.g. TensorRT's) left on
            # this thread, which clears it. So one failure doesn't mean this request failed: try again once.
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
            torch.cuda.synchronize()  # TensorRT may still be using it on a stream of its own
            torch.cuda.caching_allocator_delete(base)
            break
        except Exception as e:  # a stale CUDA error, as in _allocate
            if again:
                logger.warning(f"Couldn't free {gib(size)} of TensorRT's VRAM: {e}")
                with _lock:
                    _live[address] = entry
                return False
    with _lock:
        _ledger[owner] = _ledger.get(owner, 0) - size
        if not _ledger[owner]:
            del _ledger[owner]
    return True


def _make_allocator():
    import tensorrt as trt

    class TorchAllocator(trt.IGpuAllocator):
        # TensorRT's bindings only call allocate and deallocate (the async variants funnel into them) and read the result
        # as a plain int: a capsule, whatever the docs say, is "not an int" to them and counts as a failed allocation.
        # Nothing may raise back into TensorRT: failures return 0 or False.
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
            return 0

    return TorchAllocator()


def backend() -> str:
    try:
        return torch.cuda.get_allocator_backend()
    except Exception:
        return "native"


def attach(runtime) -> bool:
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


_proven = False


def proven():
    global _proven
    _proven = True


def failures() -> int:
    with _lock:
        return _failures


def suspect(failures_before: int, name=None) -> bool:
    # Whether PyTorch's allocator may be why `name`'s engine failed to load: no engine has run with it yet (bindings
    # that take addresses differently than documented would fail every load), no allocation failed since
    # `failures_before` (that's running out of VRAM), and no other owner holds memory through it (retire() frees it all).
    with _lock:
        others = any(n for o, n in _ledger.items() if o is not None and o != name)
        return _allocator is not None and not _proven and _failures == failures_before and not others


def retire():
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
            except Exception:
                pass
    _retired, _allocator = _allocator, None


@contextlib.contextmanager
def owner(name):  # a global stack, not per thread: TensorRT allocates from its own threads, and loads run one at a time
    with _lock:
        _owners.append(name)
    try:
        yield
    finally:
        with _lock:
            _owners.pop()


def held(name) -> int:
    with _lock:
        return _held.get(name, 0)


def ledger(name) -> int:
    with _lock:
        return _ledger.get(name, 0)


def in_use() -> int:
    # VRAM in use by anyone: the driver's used memory less PyTorch's unused cache. An engine load's growth in this is its
    # true size wherever TensorRT got the memory.
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return total - free - int(stats.get("reserved_bytes.all.current", 0)) + int(stats.get("allocated_bytes.all.current", 0))


@contextlib.contextmanager
def loading(name):
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
            # peak: the ledger's maximum, plus anything taken outside the ledger (TensorRT's own allocator)
            result.update(grew=grew, held=base + grew, peak=base + max(grew, most + max(0, grew - max(through, 0))), failures=_failures - failed)


def counting(name) -> bool:
    with _lock:
        return name in _peaks


def grew(name, nbytes: int):
    if nbytes > 0:
        with _lock:
            _held[name] = _held.get(name, 0) + nbytes


def unloaded(name, nbytes: int):
    with _lock:
        left = _held.get(name, 0) - nbytes
        if left > 0:
            _held[name] = left
        else:
            _held.pop(name, None)


def forget(name):
    with _lock:
        _held.pop(name, None)


def by_owner() -> list:
    with _lock:
        return [(o or "unattributed", n) for o, n in sorted(_ledger.items(), key=lambda kv: -kv[1]) if n]
