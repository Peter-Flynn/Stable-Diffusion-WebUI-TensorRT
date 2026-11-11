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
every few seconds, and the least recently used is dropped each time it says so. One whose checkpoint is gone
is dropped then too, and one kept for other weights than those wanted now when that's found out (a LoRA
changed: the next unload keeps the engine as it is then)."""

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

from .common import gib, log

INTERVAL = 2.0  # seconds between looks at the OS's memory pressure, while engines are kept
# Linux: the share (%) of the time since the last look in which some task waited on memory (or all did) that
# counts as the system being short of it, from the kernel's stall totals (/proc/pressure/memory)
PRESSURE_SOME, PRESSURE_FULL = 10.0, 1.0
RECENT = 10.0  # seconds: a last look older than this is too long ago to compare with; the kernel's 10 s average stands in


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
_stalls = None  # Linux: (when, {"some": microseconds tasks waited on memory in all, "full": ...}) at the last look
_stalls_lock = threading.Lock()


def _stall_shares() -> tuple:
    """Linux: ({"some": %, "full": %} of the time some task waited on memory, and all did, since the last
    look, from the kernel's totals; the span that is, for the log). Without a look in the last RECENT
    seconds (or within half a second), the kernel's own average over the last 10 s. Raises OSError on a
    kernel without pressure stall information. Unlike that average, which takes 10 s to forget a spike,
    this follows what was freed by the next look."""
    global _stalls
    with open("/proc/pressure/memory") as f:
        lines = {parts[0]: dict(kv.split("=") for kv in parts[1:]) for parts in (line.split() for line in f) if parts}
    now, totals = time.monotonic(), {k: float(v["total"]) for k, v in lines.items()}
    with _stalls_lock:
        last = _stalls
        if last is not None and now - last[0] < 0.5:
            return {k: float(v["avg10"]) for k, v in lines.items()}, "10 s"
        _stalls = (now, totals)
    if last is None or now - last[0] > RECENT:
        return {k: float(v["avg10"]) for k, v in lines.items()}, "10 s"
    span = now - last[0]
    return {k: max(0.0, totals[k] - last[1].get(k, totals[k])) / (span * 1e4) for k in totals}, f"{span:.0f} s"


def _pressure() -> str:
    """Why the OS says it's short of memory, or "" if it doesn't."""
    global _swapped, _windows_signal
    try:
        if sys.platform.startswith("linux"):
            try:
                shares, span = _stall_shares()
                if shares.get("full", 0.0) >= PRESSURE_FULL or shares.get("some", 0.0) >= PRESSURE_SOME:
                    return f"tasks waited on memory {shares.get('some', 0.0):.0f}% of the last {span}"
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


def trim() -> int:
    """Drops the engines whose checkpoint is gone, and the least recently used one if the system says it's
    short of memory (once a look: the OS's signal follows what's freed). Returns how many it dropped."""
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
