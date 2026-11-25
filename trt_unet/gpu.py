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

from . import common
from .common import gib, logger

_lock = threading.RLock()
_pool_sync_lock = threading.Lock()  # lock order is strictly _pool_sync_lock -> _lock (never inverted)
# An "owner" is the model an engine belongs to (TrtModel.owner, VaeModel.owner).
_live = {}  # address given to TensorRT -> (address PyTorch allocated, bytes, owner, device)
_ledger = {}  # owner -> bytes TensorRT currently holds through PyTorch's allocator
_peaks = {}
_held = {}  # owner -> VRAM its engines hold: the growth in in_use() across their loads (counts any allocator)
_owners = []
_failures = 0
_allocator = None
_retired = None  # the allocator of a runtime replaced by retire(): TensorRT may still call it

# Bounded two-stage scratch pool active during Engine.refit(): TensorRT allocates and frees thousands of transient
# buffers while refitting (and during ~Refitter()), and under --cuda-malloc each per-free cudaDeviceSynchronize stalls
# for milliseconds. Freed blocks queue in _pool_pending until a batched synchronize promotes them into _pool_ready for
# zero-sync reuse; OOM or pool exit flushes everything immediately so VRAM stays bounded and exact.
POOL_BUDGET = 128 * 2**20
POOL_MIN_BUDGET = 32 * 2**20
_pool_depth = 0
_pool_budget = POOL_BUDGET
_pool_promote = POOL_BUDGET // 2
_pool_ready = []  # [(base, cap, owner, dev), ...] verified idle on all streams
_pool_pending = []  # [(base, cap, owner, dev), ...] freed by TensorRT, awaiting synchronize
_pool_ready_bytes = 0
_pool_pending_bytes = 0


def _pointer(memory) -> int:
    if memory is None:
        return 0
    if isinstance(memory, int):
        return memory
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return int(get(memory, None) or 0)


def _ledger_sub_locked(owner, size: int):
    left = _ledger.get(owner, 0) - size
    if left > 0:
        _ledger[owner] = left
    else:
        _ledger.pop(owner, None)


def _sync_device(dev) -> bool:
    for again in (False, True):
        try:
            torch.cuda.synchronize(dev)
            return True
        except Exception as e:
            if again:
                logger.warning(f"Couldn't synchronize {dev} before freeing TensorRT's VRAM: {e}")
                return False
    return False


def _delete_blocks(blocks):
    for base, size, owner, _ in blocks:
        deleted = False
        for again in (False, True):
            try:
                torch.cuda.caching_allocator_delete(base)
                deleted = True
                break
            except Exception as e:
                if again:
                    logger.warning(f"Couldn't free {gib(size)} of TensorRT's VRAM: {e}")
        if deleted:
            with _lock:
                _ledger_sub_locked(owner, size)


def _promote_pool(trim_to: int = None):
    global _pool_ready_bytes, _pool_pending_bytes
    with _pool_sync_lock:
        with _lock:
            pbatch = list(_pool_pending)
            _pool_pending.clear()
            _pool_pending_bytes = 0
        synced, failed = [], []
        if pbatch:
            ok_by_dev = {}
            for entry in pbatch:
                d = entry[3]
                if d not in ok_by_dev:
                    ok_by_dev[d] = _sync_device(d)
                (synced if ok_by_dev[d] else failed).append(entry)
        evict = []
        with _lock:
            if failed:
                _pool_pending.extend(failed)
                _pool_pending_bytes += sum(e[1] for e in failed)
            if _pool_depth == 0:
                evict.extend(synced)
                evict.extend(_pool_ready)
                _pool_ready.clear()
                _pool_ready_bytes = 0
            else:
                if synced:
                    _pool_ready.extend(synced)
                    _pool_ready_bytes += sum(e[1] for e in synced)
                cap_limit = _pool_budget if trim_to is None else trim_to
                while _pool_ready_bytes > cap_limit and _pool_ready:
                    item = _pool_ready.pop(0)
                    _pool_ready_bytes -= item[1]
                    evict.append(item)
        if evict:
            _delete_blocks(evict)


def promote_pool():
    _promote_pool()


def _flush_pool():
    global _pool_ready_bytes, _pool_pending_bytes
    with _pool_sync_lock:
        with _lock:
            rbatch = list(_pool_ready)
            pbatch = list(_pool_pending)
            _pool_ready.clear()
            _pool_pending.clear()
            _pool_ready_bytes = 0
            _pool_pending_bytes = 0
        if not rbatch and not pbatch:
            return
        deletable = list(rbatch)
        if pbatch:
            ok_by_dev = {}
            unsynced = []
            for entry in pbatch:
                d = entry[3]
                if d not in ok_by_dev:
                    ok_by_dev[d] = _sync_device(d)
                (deletable if ok_by_dev[d] else unsynced).append(entry)
            if unsynced:
                with _lock:
                    _pool_pending.extend(unsynced)
                    _pool_pending_bytes += sum(e[1] for e in unsynced)
        if deletable:
            _delete_blocks(deletable)


@contextlib.contextmanager
def refit_pool(budget: int = None):
    global _pool_depth, _pool_budget, _pool_promote
    with _lock:
        outer = _pool_depth == 0
        if outer:
            if budget is None:
                try:
                    free_b, _ = torch.cuda.mem_get_info(common.device())
                    budget = min(POOL_BUDGET, max(POOL_MIN_BUDGET, free_b // 16))
                except Exception:
                    budget = POOL_BUDGET
            _pool_budget = max(POOL_MIN_BUDGET, int(budget))
            _pool_promote = _pool_budget // 2
        _pool_depth += 1
    try:
        yield
    finally:
        with _lock:
            _pool_depth -= 1
            done = _pool_depth == 0
        if done:
            _flush_pool()


def _take_ready_locked(size: int, alignment: int, dev):
    global _pool_ready_bytes
    best_i, best_cap = -1, None
    max_cap = max(size * 2, size + 256 * 1024)
    for i, (base, cap, _, bd) in enumerate(_pool_ready):
        if bd != dev or cap > max_cap:
            continue
        addr = (base + alignment - 1) // alignment * alignment
        if base + cap >= addr + size and (best_cap is None or cap < best_cap):
            best_i, best_cap = i, cap
            if cap == size:
                break
    if best_i >= 0:
        item = _pool_ready.pop(best_i)
        _pool_ready_bytes -= item[1]
        return item
    return None


def _allocate(size: int, alignment: int) -> int:
    global _failures
    if size <= 0:
        return 0
    alignment = max(int(alignment or 1), 1)
    dev = common.device()
    with _lock:
        if _pool_depth > 0 and _pool_ready:
            reused = _take_ready_locked(size, alignment, dev)
            if reused is not None:
                base, cap, old_owner, _ = reused
                addr = (base + alignment - 1) // alignment * alignment
                owner = _owners[-1] if _owners else None
                _live[addr] = (base, cap, owner, dev)
                if owner != old_owner:
                    _ledger_sub_locked(old_owner, cap)
                    _ledger[owner] = _ledger.get(owner, 0) + cap
                    if owner in _peaks:
                        _peaks[owner] = max(_peaks[owner], _ledger[owner])
                return addr
    # PyTorch's native allocator aligns to 512 bytes and cudaMallocAsync to 256; requesting exact `size` for
    # <= 256-byte alignments avoids pushing power-of-two buffers across allocator block boundaries.
    extra = 0 if alignment <= 256 else alignment - 1
    req = size + extra
    base = None
    for flushed in (False, True):
        oom = False
        for again in (False, True):
            try:
                base = torch.cuda.caching_allocator_alloc(req, dev.index)
                if not extra and base % alignment:
                    torch.cuda.caching_allocator_delete(base)
                    base, extra, req = None, alignment - 1, size + alignment - 1
                    base = torch.cuda.caching_allocator_alloc(req, dev.index)
                break
            except torch.cuda.OutOfMemoryError:
                oom = True
                break
            except Exception as e:
                # With --cuda-malloc, PyTorch's allocator first raises any CUDA error an earlier call (e.g. TensorRT's) left on
                # this thread, which clears it. So one failure doesn't mean this request failed: try again once.
                if again:
                    logger.debug(f"PyTorch's allocator refused TensorRT {gib(size)}: {e}")
        if base is not None or flushed or not oom:
            break
        with _lock:
            has_pool = _pool_depth > 0 and bool(_pool_ready or _pool_pending)
        if not has_pool:
            break
        _flush_pool()
    if base is None:
        with _lock:
            _failures += 1
        return 0
    address = (base + alignment - 1) // alignment * alignment
    held_bytes = req if extra else size
    with _lock:
        owner = _owners[-1] if _owners else None
        _live[address] = (base, held_bytes, owner, dev)
        _ledger[owner] = _ledger.get(owner, 0) + held_bytes
        if owner in _peaks:
            _peaks[owner] = max(_peaks[owner], _ledger[owner])
    return address


def _free(memory) -> bool:
    global _pool_pending_bytes
    address = _pointer(memory)
    if not address:
        return True
    with _lock:
        entry = _live.pop(address, None)
        if entry is None:
            return False
        if _pool_depth > 0:
            _pool_pending.append(entry)
            _pool_pending_bytes += entry[1]
            tot = _pool_ready_bytes + _pool_pending_bytes
            over_budget = tot > _pool_budget
            need_promote = _pool_pending_bytes >= _pool_promote or over_budget
            trim_to = _pool_budget // 2 if over_budget else _pool_budget
        else:
            need_promote = None
    if need_promote is not None:
        if need_promote:
            _promote_pool(trim_to)
        return True
    base, size, owner, dev = entry
    for again in (False, True):
        try:
            torch.cuda.synchronize(dev)  # TensorRT may still be using it on a stream of its own
            torch.cuda.caching_allocator_delete(base)
            break
        except Exception as e:  # a stale CUDA error, as in _allocate
            if again:
                logger.warning(f"Couldn't free {gib(size)} of TensorRT's VRAM: {e}")
                with _lock:
                    _live[address] = entry
                return False
    with _lock:
        _ledger_sub_locked(owner, size)
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
        others = any(n for o, n in _ledger.items() if o != name)
        return _allocator is not None and not _proven and _failures == failures_before and not others


def retire():
    global _allocator, _retired
    _flush_pool()
    with _lock:
        entries = list(_live.values())
        _live.clear()
        _ledger.clear()
    for base, _, _, dev in entries:
        for again in (False, True):
            try:
                torch.cuda.synchronize(dev)
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
    dev = common.device()
    free, total = torch.cuda.mem_get_info(dev)
    stats = torch.cuda.memory_stats(dev)
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
