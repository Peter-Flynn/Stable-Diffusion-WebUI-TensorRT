"""Refitted engines kept in RAM.

An engine loses its weights when it's unloaded (another profile's engine takes its place for a Hires. fix,
or reForge needs its VRAM), and loading it again would mean refitting every one of them. So an engine that's
unloaded with its weights in is serialized with them first, and loading it again for the same weights (the
same checkpoint with the same LoRAs) deserializes that instead: nothing is gathered, patched or refitted.

A kept engine is about as large as its weights (GBs). As many are kept as RAM allows: one goes in only if
the reserve (RESERVE_SHARE of the RAM, RESERVE_MIN at least) stays available with it there, and the least
recently used ones are dropped whenever less than the reserve is available, which is looked at every few
seconds while any is kept. One whose checkpoint is gone is dropped then too, and one kept for other weights
than those wanted now when that's found out (a LoRA changed: the next unload keeps the engine as it is then)."""

import ctypes
import os
import threading
import time
from dataclasses import dataclass

from .common import gib, log

RESERVE_SHARE = 0.15  # of the system's RAM, left available to everything else
RESERVE_MIN = 4 * 2**30
INTERVAL = 2.0  # seconds between looks at the RAM available, while engines are kept


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


def _memory():
    """(RAM available, the reserve to leave available), or None if it can't tell."""
    try:
        import psutil

        ram = psutil.virtual_memory()
        return int(ram.available), max(RESERVE_MIN, int(ram.total * RESERVE_SHARE))
    except Exception:
        return None


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
    """Whether RAM allows keeping `nbytes` more, once the least recently used engines (other than `key`'s)
    are dropped if that's what it takes."""
    memory = _memory()
    if memory is None:
        return False  # no telling when it would run low
    available, reserve = memory
    with _lock:
        others = [k for k in _kept if k != key]
        while available - nbytes < reserve and others:
            entry = _kept.pop(others.pop(0))
            available += entry.nbytes
            log(f"Dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}) to make room for another.")
            del entry
    _give_back()
    return available - nbytes >= reserve


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
    """Drops the engines whose checkpoint is gone, and the least recently used ones while less than the
    reserve of RAM is available. Returns how many it dropped."""
    dropped = 0
    with _lock:
        for key in [k for k, entry in _kept.items() if entry.orphaned]:
            entry = _kept.pop(key)
            log(f"Dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}): its checkpoint is no longer loaded.")
            dropped += 1
            del entry
    memory = _memory()
    if memory is None:
        if dropped:
            _give_back()
        return dropped
    available, reserve = memory
    with _lock:
        while _kept and available < reserve:
            entry = _kept.pop(next(iter(_kept)))
            log(f"RAM is running low ({gib(available)} available): dropped the {entry.label} kept in RAM ({gib(entry.nbytes)}).")
            available += entry.nbytes
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
            log(f"Could not check the RAM for the engines kept there: {e}")
