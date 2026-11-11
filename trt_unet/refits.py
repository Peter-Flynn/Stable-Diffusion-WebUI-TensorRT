"""Refitted engines kept in RAM.

An engine loses its weights when it's unloaded (another profile's engine takes its place for a Hires. fix,
or reForge needs its VRAM), and loading it again would mean refitting every one of them. So an engine that's
unloaded with its weights in is serialized with them first, and loading it again for the same weights (the
same checkpoint with the same LoRAs) deserializes that instead: nothing is gathered, patched or refitted.

A kept engine is about as large as its weights (GBs), and RAM is there to be used: one is kept whenever the
system can hand over that much without paging anything out (its free memory and the file cache it can drop,
as the OS counts them: _supply) and isn't short of memory already, dropping the least recently used ones if
that's what it takes. While any is kept, the OS's own signal that it's short of memory (_pressure: Linux's
pressure stall information, Windows' low-memory notification, macOS' memory pressure level) is looked at
every few seconds, and the least recently used is dropped each time it says so. On Linux that's tasks
having waited on memory for a tenth of a 10-second window with none of this module's own doing in it:
keeping an engine has the kernel reclaim file cache for it, which stalls tasks briefly and is no shortage
(the cache was counted as there to hand over). One whose checkpoint is gone is dropped then too, and one
kept for other weights than those wanted now when that's found out (a LoRA changed: the next unload keeps
the engine as it is then)."""

import collections
import contextlib
import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

from .common import gib, log

INTERVAL = 2.0  # seconds between looks at the OS's memory pressure, while engines are kept
# Linux: the system is short of memory when some task waited on it for PRESSURE_SOME % of a WINDOW of seconds
# (the kernel's stall total, /proc/pressure/memory) in which this module neither kept nor dropped an engine
WINDOW, PRESSURE_SOME = 10.0, 10.0


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
_lock = threading.RLock()
_watcher = None  # the thread looking at the RAM, while there's anything kept


_last = {}  # what room() last weighed, for the log (explain)


def _meminfo() -> dict:
    with open("/proc/meminfo") as f:
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f if line.split()[1:]}


def _supply() -> int:
    """The RAM the OS can hand over without paging anything out, as it counts it: on Linux, the free memory
    and the clean file cache and kernel caches it drops on demand; on Windows and macOS, what they call
    available (free, standby or inactive). Raises if it can't tell."""
    if sys.platform.startswith("linux"):
        m = _meminfo()
        return m["MemFree"] + m.get("Active(file)", 0) + m.get("Inactive(file)", 0) + m.get("SReclaimable", 0)
    import psutil

    return int(psutil.virtual_memory().available)


_swapped = None  # Linux without pressure stall information: pages swapped out, at the last look
_windows_signal = None  # Windows: the low-memory notification's handle
_stalls = collections.deque()  # Linux: (when, microseconds some task waited on memory in all) at each look, since _settled, the first the base of the window
_settled = 0.0  # when (time.monotonic) this module last kept or dropped an engine
_keeping = 0  # engines being kept now
_stalls_lock = threading.Lock()


def _settle():
    """This module kept or dropped an engine: the stalls up to now aren't the system's own."""
    global _settled
    with _stalls_lock:
        _stalls.clear()
        _settled = time.monotonic()


@contextlib.contextmanager
def keeping():
    """While an engine is serialized into RAM: the stalls that causes (the kernel reclaiming file cache for
    it) aren't weighed, during it or in any window it's in."""
    global _keeping
    with _stalls_lock:
        _keeping += 1
    try:
        yield
    finally:
        with _stalls_lock:
            _keeping -= 1
        _settle()


def _stalled():
    """Linux: (the % of a window in which some task waited on memory, its seconds), or None if there's no
    window to judge by yet. The window is the last WINDOW seconds of looks since this module last kept or
    dropped an engine; before there are that many, the kernel's own 10-second average if that has none of
    it either, else nothing. Raises OSError on a kernel without pressure stall information."""
    with open("/proc/pressure/memory") as f:
        some = next(dict(kv.split("=") for kv in line.split()[1:]) for line in f if line.startswith("some"))
    now, total = time.monotonic(), float(some["total"])
    with _stalls_lock:
        if _keeping:
            return None
        _stalls.append((now, total))
        while len(_stalls) > 1 and now - _stalls[1][0] >= WINDOW:  # the newest look that's a window old is the base
            _stalls.popleft()
        when, base = _stalls[0]
        clean = now - _settled
    if now - when >= WINDOW:
        return (total - base) / ((now - when) * 1e4), now - when
    return (float(some["avg10"]), 10.0) if clean >= WINDOW else None


def _pressure() -> str:
    """Why the OS says it's short of memory, or "" if it doesn't."""
    global _swapped, _windows_signal
    try:
        if sys.platform.startswith("linux"):
            try:
                verdict = _stalled()
                if verdict is not None and verdict[0] >= PRESSURE_SOME:
                    return f"tasks waited on memory {verdict[0]:.0f}% of the last {verdict[1]:.0f} s"
                return ""
            except OSError:  # a kernel without it: swapping out at all says so
                with open("/proc/vmstat") as f:
                    out = next(int(line.split()[1]) for line in f if line.startswith("pswpout "))
                before, _swapped = _swapped, out
                return "it's swapping out" if before is not None and out > before else ""
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
            if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(level), ctypes.byref(size), None, 0) == 0 and level.value >= 2:
                return "macOS' memory pressure is " + ("critical" if level.value >= 4 else "elevated")
            return ""
    except Exception:
        pass
    return ""


def explain() -> str:
    """For the log: what room() last weighed, and why it said no."""
    if "error" in _last:
        return f"RAM couldn't be read: {_last['error']}"
    if "pressure" in _last:
        return f"the system is short of memory: {_last['pressure']}"
    if not _last:
        return "nothing weighed"
    return f"needs {gib(_last['nbytes'])}, the system could hand over {gib(_last['supply'])}, other kept engines hold {gib(_last['others'])}"


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
    _give_back()


def held() -> tuple:
    """(How many engines are kept, their bytes)."""
    with _lock:
        return len(_kept), sum(entry.nbytes for entry in _kept.values())


def room(nbytes: int, key: str = None) -> bool:
    """Whether to keep `nbytes` more: the system isn't short of memory, and can hand that much over without
    paging anything out, once the least recently used engines (other than `key`'s) are dropped if that's
    what it takes. If dropping them all wouldn't make room, none is dropped."""
    _last.clear()
    why = _pressure()
    if why:
        _last["pressure"] = why
        return False
    try:
        supply = _supply()
    except Exception as e:  # no telling what it would push out
        _last["error"] = f"{type(e).__name__}: {e}"
        return False
    dropped = 0
    with _lock:
        others = [k for k in _kept if k != key]
        _last.update(nbytes=nbytes, supply=supply, others=sum(_kept[k].nbytes for k in others))
        if supply + _last["others"] < nbytes:
            return False
        while supply < nbytes and others:
            entry = _kept.pop(others.pop(0))
            supply += entry.nbytes
            dropped += 1
            log(f"Dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}) to make room for another.")
            del entry
    if dropped:
        _give_back()
        _settle()
    return supply >= nbytes


def put(key: str, entry: Kept):
    """Keeps an engine (in place of what was kept under its key); starts looking at the RAM."""
    global _watcher
    with _lock:
        _kept.pop(key, None)
        _kept[key] = entry
        if _watcher is None:
            _watcher = threading.Thread(target=_watch, name="trt-refits", daemon=True)
            _watcher.start()
    _settle()


def trim() -> int:
    """Drops the engines whose checkpoint is gone, and the least recently used one if the system says it's
    short of memory (one, then a new window: the OS's signal has to follow what was freed). Returns how
    many it dropped."""
    dropped = 0
    with _lock:
        for key in [k for k, entry in _kept.items() if entry.orphaned]:
            entry = _kept.pop(key)
            log(f"Dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}): its checkpoint is no longer loaded.")
            dropped += 1
            del entry
        why = _pressure() if _kept else ""
        if why:
            entry = _kept.pop(next(iter(_kept)))
            log(f"The system is short of memory ({why}): dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}).")
            dropped += 1
            del entry
    if dropped:
        _give_back()
        _settle()
    return dropped


def _watch():
    global _watcher
    while True:
        time.sleep(INTERVAL)
        with _lock:
            if not _kept:
                _watcher = None
                return
        try:
            trim()
        except Exception as e:  # keep looking: nothing else would
            log(f"Could not check the system's memory for the engines kept in RAM: {e}")
