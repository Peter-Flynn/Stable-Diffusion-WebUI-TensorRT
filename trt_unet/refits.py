"""Refitted engines kept in RAM.

An engine loses its weights when it's unloaded (another profile's engine takes its place for a Hires. fix,
or reForge needs its VRAM), and loading it again would mean refitting every one of them. So an engine that's
unloaded with its weights in is serialized with them first, and loading it again for the same weights (the
same checkpoint with the same LoRAs) deserializes that instead: nothing is gathered, patched or refitted.

A kept engine is about as large as its weights (GBs), and RAM is there to be used: generating is what the
machine is for while reForge runs, so one is kept whenever the system can hand over that much without
paging anything out (its free memory and the file cache it can drop, as the OS counts them: memory) and
stays kept until the system needs the memory: other programs' pages going to swap as the system makes
room, and the stalls that brings for a while, are not that. While any is kept, the OS is asked every few
seconds whether it does (_pressure), and the least recently used goes each time it says so:

- Linux: its free memory and droppable caches have been under FLOOR of the RAM for LOW seconds on end (it
  had that long to swap something out by itself, and didn't get back above it; swap isn't counted as
  memory to give), or everything that runs, generating included, has been stalled on memory at once for
  STALLED % of the last WINDOW seconds (the kernel's "full" stall time, /proc/pressure/memory): it does
  nothing but page.
- Windows: its low-memory notification. macOS: its memory pressure is critical.

One is also dropped to make room for another, but not one that's been used since the other was used the
time before last: two engines taking turns (a Hires. fix's) with room for one would push each other out
at every turn, each time at the cost of keeping one. One whose checkpoint is gone is dropped too, and one
kept for other weights than those wanted now when that's found out (a LoRA changed: the next unload keeps
the engine as it is then)."""

import collections
import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

from .common import gib, logger

INTERVAL = 2.0  # seconds between looks at whether the system needs the memory, while engines are kept
# Linux: it does when every task that isn't idle has been stalled on memory at once for STALLED % of the
# last WINDOW seconds, or when what it could hand over without swapping has been under FLOOR of its RAM
# for LOW seconds on end
WINDOW, STALLED = 60.0, 50.0
FLOOR, LOW = 0.05, 30.0


@dataclass
class Kept:
    signatures: dict  # role ("unet", "shallow") -> {weight name: the signature of what the engine holds}
    blobs: dict  # role -> the engine serialized with its weights (a buffer)
    label: str  # for the log
    source: object = None  # () -> the UNet module the weights are from, None once it's gone (a weak reference)

    @property
    def nbytes(self) -> int:
        return sum(memoryview(blob).nbytes for blob in self.blobs.values())

    @property
    def orphaned(self) -> bool:
        """Its checkpoint is gone: these weights won't be asked for again."""
        return self.source is not None and self.source() is None


_kept = {}  # engine key -> Kept, the least recently used first
_uses = {}  # engine key -> (when its engine was loaded the time before last, when it last was), as counts
_count = 0  # engine loads so far (used)
_lock = threading.RLock()
_watcher = None  # the thread asking whether the system needs the memory, while there's anything kept


_last = {}  # what room() last weighed, for the log (explain)


def _meminfo() -> dict:
    with open("/proc/meminfo") as f:
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f if line.split()[1:]}


READS = 5  # times the OS is asked for its memory counts before keeping an engine goes ahead without them (room); the watcher just looks again


def _read() -> tuple:
    if sys.platform.startswith("linux"):
        # As htop counts it (which reads true where the kernel's own MemAvailable doesn't: that reads 0 on
        # some systems with most of their RAM free, and is never looked at here): what's free, and the
        # buffers and caches the kernel drops on demand, less shared memory, which it can't drop
        m = _meminfo()
        supply = m["MemFree"] + m.get("Buffers", 0) + m.get("Cached", 0) + m.get("SReclaimable", 0) - m.get("Shmem", 0)
        if m["MemTotal"] <= 0 or supply <= 0:
            raise RuntimeError("the kernel's memory counts came back empty")
        return supply, m.get("SwapFree", 0), m["MemTotal"]
    import warnings

    import psutil

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # "available memory stats couldn't be determined and was set to 0"
        ram = psutil.virtual_memory()
    if ram.total <= 0 or ram.available <= 0:
        raise RuntimeError("the system's memory counts came back empty")
    try:
        swap = int(psutil.swap_memory().free)
    except Exception:
        swap = 0
    return int(ram.available), swap, int(ram.total)


def memory(reads: int = 1) -> tuple:
    """(The RAM the OS can hand over without paging anything out, as it counts it: on Linux, the free
    memory and the buffers and caches it drops on demand, as htop counts them; on Windows and macOS, what they
    call available (free, standby or inactive). The swap space it has free. All its RAM.) Counts that come
    back empty or with a warning are the tool failing, not the system out of memory: it's asked up to
    `reads` times, and raises if it still can't tell, which nothing is decided on."""
    for attempt in range(reads):
        try:
            return _read()
        except Exception:
            if attempt == reads - 1:
                raise
            time.sleep(0.2)


_windows_signal = None  # Windows: the low-memory notification's handle
_stalls = collections.deque()  # Linux: (when, microseconds every task was stalled on memory at once, in all) at each look, the first the base of the window
_settled = 0.0  # when (time.monotonic) an engine was last dropped: the window starts over
_stalls_lock = threading.Lock()


_low_since = None  # when what the system could hand over was first seen under FLOOR of its RAM, if it still is


def _settle():
    """An engine was dropped: the stalls and the low memory up to now say nothing of the system as it is
    without it."""
    global _settled, _low_since
    with _stalls_lock:
        _stalls.clear()
        _settled = time.monotonic()
        _low_since = None


def _low(supply: int, total: int):
    """The seconds what the system could hand over has been under FLOOR of its RAM, as of this look (0.0
    at the first that sees it so), or None if it isn't: it starts over whenever it's back above."""
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
    """Linux: (the % of the last WINDOW seconds in which every task that wasn't idle was stalled on memory
    at once, its seconds), or None if there's no such window since an engine was last dropped. The window
    is this module's own looks; before there's a WINDOW of them, the kernel's own one-minute average.
    Raises on a kernel without pressure stall information."""
    with open("/proc/pressure/memory") as f:
        full = next(dict(kv.split("=") for kv in line.split()[1:]) for line in f if line.startswith("full"))
    now, total = time.monotonic(), float(full["total"])
    with _stalls_lock:
        _stalls.append((now, total))
        while len(_stalls) > 1 and now - _stalls[1][0] >= WINDOW:  # the newest look that's a window old is the base
            _stalls.popleft()
        when, base = _stalls[0]
        clean = now - _settled
    if now - when >= WINDOW:
        return (total - base) / ((now - when) * 1e4), now - when
    return (float(full["avg60"]), 60.0) if clean >= WINDOW else None


def _pressure() -> str:
    """Why the system needs the kept engines' memory, or "" if it doesn't (see the module)."""
    global _windows_signal
    try:
        if sys.platform.startswith("linux"):
            try:
                supply, _, total = memory()
            except Exception:  # the tool failed: nothing is decided on that, and the next look asks again
                supply = total = None
            low = _low(supply, total) if total else None
            if low is not None and low >= LOW:
                return f"under {FLOOR:.0%} of its RAM has been free for {low:.0f}s ({gib(supply)} now)"
            try:
                verdict = _stalled()
            except (OSError, StopIteration, KeyError, ValueError):  # a kernel without it: only the above tells
                return ""
            if verdict is not None and verdict[0] >= STALLED:
                return f"everything waited on memory for {verdict[0]:.0f}% of the last {verdict[1]:.0f}s"
            return ""
        if os.name == "nt":
            k32 = ctypes.windll.kernel32
            if _windows_signal is None:
                k32.CreateMemoryResourceNotification.restype = ctypes.c_void_p
                _windows_signal = k32.CreateMemoryResourceNotification(0)  # LowMemoryResourceNotification
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
    """For the log: what room() last weighed, and why it said no."""
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
    """Hands freed memory back to the system now: allocators can hold on to it (the tcmalloc that webui.sh
    preloads does, for a while), and then it wouldn't show as available."""
    if os.name == "nt":
        return
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "MallocExtension_ReleaseFreeMemory"):  # tcmalloc
            libc.MallocExtension_ReleaseFreeMemory()
        elif hasattr(libc, "malloc_trim"):  # glibc
            libc.malloc_trim(0)
    except Exception:
        pass


def get(key: str):
    """The engine kept under this key (now the most recently used), or None."""
    with _lock:
        entry = _kept.pop(key, None)
        if entry is not None:
            _kept[key] = entry
        return entry


def drop(key: str, role: str = None):
    """Forgets the engine kept under this key, or only its `role` part (the "unet" part takes the rest
    along: a shallow engine is no use without its UNet's)."""
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
    """(How many engines are kept, their bytes)."""
    with _lock:
        return len(_kept), sum(entry.nbytes for entry in _kept.values())


def used(key: str):
    """The engine under this key was loaded (from RAM or not): what room() goes by."""
    global _count
    with _lock:
        _count += 1
        _uses[key] = (_uses.get(key, (None, None))[1], _count)


def _may_go(k: str, key: str) -> bool:
    """Whether the engine kept under `k` may go to make room for `key`'s: it hasn't been used since `key`'s
    was used the time before last."""
    before = _uses.get(key, (None, None))[0]
    return before is not None and (_uses.get(k, (None, None))[1] or 0) < before


def room(nbytes: int, key: str = None) -> bool:
    """Whether to keep `nbytes` more for `key`'s engine: the system doesn't need the memory, can hand that
    much over without paging anything out, and still has FLOOR of its RAM to give then, swap not counted
    (so keeping it isn't what makes the system need it back: _pressure), once the least recently used
    engines that may go for it (_may_go) are dropped if that's what it takes. If dropping them all wouldn't
    make room, none is dropped."""
    _last.clear()
    why = _pressure()
    if why:
        _last["pressure"] = why
        return False
    try:
        supply, _, total = memory(READS)
    except Exception as e:  # the tool failed: nothing is decided on that, and the engine is kept
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
    """Keeps an engine (in place of what was kept under its key); starts asking whether the system needs
    the memory."""
    global _watcher
    with _lock:
        _kept.pop(key, None)
        _kept[key] = entry
        if _watcher is None:
            _watcher = threading.Thread(target=_watch, name="trt-refits", daemon=True)
            _watcher.start()


def trim() -> int:
    """Drops the engines whose checkpoint is gone, and the least recently used one if the system needs the
    memory (one, then a new window: what it says has to follow what was freed). Returns how many it
    dropped."""
    dropped = 0
    with _lock:
        for key in [k for k, entry in _kept.items() if entry.orphaned]:
            entry = _kept.pop(key)
            logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): its checkpoint isn't loaded anymore.")
            dropped += 1
            del entry
        why = _pressure() if _kept else ""
        if why:
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
    failed = False  # said already
    while True:
        time.sleep(INTERVAL)
        with _lock:
            if not _kept:
                _watcher = None
                return
        try:
            trim()
        except Exception as e:  # keep looking: nothing else would
            if not failed:
                failed = True
                logger.warning(f"Couldn't check whether the system needs the RAM that engines are kept in: {e}")
