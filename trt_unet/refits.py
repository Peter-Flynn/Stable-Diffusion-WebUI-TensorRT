# Engines kept in RAM with their weights. Unloading an engine loses its weights, and loading it again means refitting
# every one. So an engine is serialized with its weights before it unloads, and a later load for the same weights
# (same checkpoint, same LoRAs) deserializes that instead. A kept engine is about the size of its weights (GBs).
#
# One is kept whenever the OS can give that much without paging anything out, and stays until the OS needs the memory
# back. A watcher thread checks every INTERVAL seconds while anything is kept, and drops the least recently used
# engine each time the OS is under pressure:
#   Linux: free + droppable cache has stayed under FLOOR of RAM for LOW seconds, or every non-idle task was stalled
#          on memory for STALLED% of the last WINDOW seconds (/proc/pressure/memory, "full").
#   Windows: its low-memory notification is signaled. macOS: memory pressure is critical.
import collections
import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

from .common import gib, logger

INTERVAL = 2.0
WINDOW, STALLED = 60.0, 50.0
FLOOR, LOW = 0.05, 30.0


@dataclass
class Kept:
    signatures: dict
    blobs: dict
    label: str
    source: object = None

    @property
    def nbytes(self) -> int:
        return sum(memoryview(blob).nbytes for blob in self.blobs.values())

    @property
    def orphaned(self) -> bool:
        return self.source is not None and self.source() is None


_kept = {}
_uses = {}  # engine key -> (load count at its previous load, load count at its latest load); see _may_go
_count = 0
_lock = threading.RLock()
_watcher = None


_last = {}


def _meminfo() -> dict:
    with open("/proc/meminfo") as f:
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f if line.split()[1:]}


READS = 5  # tries at reading memory counts before keeping an engine anyway: empty counts mean the reading failed


def _read() -> tuple:
    if sys.platform.startswith("linux"):
        # Counted as htop does, not from MemAvailable, which reads 0 on some systems with most of their RAM free: free
        # memory, buffers and droppable caches, less shared memory (which can't be dropped).
        m = _meminfo()
        supply = m["MemFree"] + m.get("Buffers", 0) + m.get("Cached", 0) + m.get("SReclaimable", 0) - m.get("Shmem", 0)
        if m["MemTotal"] <= 0 or supply <= 0:
            raise RuntimeError("the kernel's memory counts came back empty")
        return supply, m.get("SwapFree", 0), m["MemTotal"]
    import warnings

    import psutil

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # psutil warns "available memory ... set to 0" instead of failing
        ram = psutil.virtual_memory()
    if ram.total <= 0 or ram.available <= 0:
        raise RuntimeError("the system's memory counts came back empty")
    try:
        swap = int(psutil.swap_memory().free)
    except Exception:
        swap = 0
    return int(ram.available), swap, int(ram.total)


def memory(reads: int = 1) -> tuple:
    for attempt in range(reads):
        try:
            return _read()
        except Exception:
            if attempt == reads - 1:
                raise
            time.sleep(0.2)


_windows_signal = None
_stalls = collections.deque()  # (time.monotonic(), cumulative "full" stall microseconds) per check; [0] is the window's start
_settled = 0.0  # when an engine was last dropped: stall history from before then is discarded
_stalls_lock = threading.Lock()


_low_since = None


def _settle():
    global _settled, _low_since
    with _stalls_lock:
        _stalls.clear()
        _settled = time.monotonic()
        _low_since = None


def _low(supply: int, total: int):
    global _low_since
    now = time.monotonic()
    with _stalls_lock:
        if supply >= FLOOR * total:
            _low_since = None
            return None
        if _low_since is None:
            _low_since = now
        return now - _low_since


def _stalled():
    with open("/proc/pressure/memory") as f:
        full = next(dict(kv.split("=") for kv in line.split()[1:]) for line in f if line.startswith("full"))
    now, total = time.monotonic(), float(full["total"])
    with _stalls_lock:
        _stalls.append((now, total))
        while len(_stalls) > 1 and now - _stalls[1][0] >= WINDOW:
            _stalls.popleft()
        when, base = _stalls[0]
        clean = now - _settled
    if now - when >= WINDOW:
        return (total - base) / ((now - when) * 1e4), now - when
    return (float(full["avg60"]), 60.0) if clean >= WINDOW else None


def _pressure() -> str:
    global _windows_signal
    try:
        if sys.platform.startswith("linux"):
            try:
                supply, _, total = memory()
            except Exception:
                supply = total = None
            low = _low(supply, total) if total else None
            if low is not None and low >= LOW:
                return f"under {FLOOR:.0%} of its RAM has been free for {low:.0f}s ({gib(supply)} now)"
            try:
                verdict = _stalled()
            except (OSError, StopIteration, KeyError, ValueError):
                return ""
            if verdict is not None and verdict[0] >= STALLED:
                return f"everything waited on memory for {verdict[0]:.0f}% of the last {verdict[1]:.0f}s"
            return ""
        if os.name == "nt":
            k32 = ctypes.windll.kernel32
            if _windows_signal is None:
                k32.CreateMemoryResourceNotification.restype = ctypes.c_void_p
                _windows_signal = k32.CreateMemoryResourceNotification(0)  # 0: LowMemoryResourceNotification
            low = ctypes.c_int(0)
            if _windows_signal and k32.QueryMemoryResourceNotification(ctypes.c_void_p(_windows_signal), ctypes.byref(low)) and low.value:
                return "Windows signals low memory"
            return ""
        if sys.platform == "darwin":
            libc, level, size = ctypes.CDLL("libc.dylib"), ctypes.c_int(0), ctypes.c_size_t(4)
            if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(level), ctypes.byref(size), None, 0) == 0 and level.value >= 4:
                return "macOS' memory pressure is critical"
            return ""
    except Exception:
        pass
    return ""


def explain() -> str:
    if "error" in _last:
        return f"RAM couldn't be read ({_last['error']})"
    if "pressure" in _last:
        return f"the system needs the memory, as {_last['pressure']}"
    if not _last:
        return "nothing weighed"
    floor = f", and {gib(_last['floor'])} must stay free" if _last["floor"] else ""
    others = f"; engines kept that could make way hold {gib(_last['others'])}" if _last["others"] else ""
    return f"it needs {gib(_last['nbytes'])}, {gib(_last['supply'])} is free{floor}{others}"


def _give_back():
    # Makes the allocator return freed memory to the OS now (tcmalloc, preloaded by webui.sh, holds it for a while),
    # so the OS counts it as available.
    if os.name == "nt":
        return
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "MallocExtension_ReleaseFreeMemory"):
            libc.MallocExtension_ReleaseFreeMemory()
        elif hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def get(key: str):
    with _lock:
        entry = _kept.pop(key, None)
        if entry is not None:
            _kept[key] = entry
        return entry


def drop(key: str, role: str = None):
    with _lock:
        entry = _kept.get(key)
        if entry is None:
            return
        if role in (None, "unet"):
            del _kept[key]
        else:
            entry.blobs.pop(role, None)
            entry.signatures.pop(role, None)
    del entry
    _give_back()


def clear():
    with _lock:
        _kept.clear()
        _uses.clear()
    _give_back()


def held() -> tuple:
    with _lock:
        return len(_kept), sum(entry.nbytes for entry in _kept.values())


def used(key: str):
    global _count
    with _lock:
        _count += 1
        _uses[key] = (_uses.get(key, (None, None))[1], _count)


def _may_go(k: str, key: str) -> bool:
    # `k` may be dropped to make room for `key` only if `k` hasn't been loaded since `key`'s previous load. Otherwise two
    # engines alternating with room for only one (a Hires. fix's two profiles) would evict each other on every switch.
    before = _uses.get(key, (None, None))[0]
    return before is not None and (_uses.get(k, (None, None))[1] or 0) < before


def room(nbytes: int, key: str = None) -> bool:
    _last.clear()
    why = _pressure()
    if why:
        _last["pressure"] = why
        return False
    try:
        supply, _, total = memory(READS)
    except Exception as e:
        _last["error"] = f"{type(e).__name__}: {e}"
        return True
    floor = int(FLOOR * total)

    def fits(have: int) -> bool:
        return have - nbytes >= floor

    dropped = 0
    with _lock:
        others = [k for k in _kept if k != key and _may_go(k, key)]
        _last.update(nbytes=nbytes, supply=supply, floor=floor, others=sum(_kept[k].nbytes for k in others))
        if not fits(supply + _last["others"]):
            return False
        while not fits(supply) and others:
            entry = _kept.pop(others.pop(0))
            supply += entry.nbytes
            dropped += 1
            logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}) to make room for another.")
            del entry
    if dropped:
        _give_back()
        _settle()
    return fits(supply)


def put(key: str, entry: Kept):
    global _watcher
    with _lock:
        _kept.pop(key, None)
        _kept[key] = entry
        if _watcher is None:
            _watcher = threading.Thread(target=_watch, name="trt-refits", daemon=True)
            _watcher.start()


def trim() -> int:
    dropped = 0
    with _lock:
        for key in [k for k, entry in _kept.items() if entry.orphaned]:
            entry = _kept.pop(key)
            logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): its checkpoint isn't loaded anymore.")
            dropped += 1
            del entry
        why = _pressure() if _kept else ""
        if why:  # one per check: the next check must see the memory this one freed
            entry = _kept.pop(next(iter(_kept)))
            logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): the system needs the memory, as {why}.")
            dropped += 1
            del entry
    if dropped:
        _give_back()
        _settle()
    return dropped


def _watch():
    global _watcher
    failed = False
    while True:
        time.sleep(INTERVAL)
        with _lock:
            if not _kept:
                _watcher = None
                return
        try:
            trim()
        except Exception as e:
            if not failed:
                failed = True
                logger.warning(f"Couldn't check whether the system needs the RAM that engines are kept in: {e}")
