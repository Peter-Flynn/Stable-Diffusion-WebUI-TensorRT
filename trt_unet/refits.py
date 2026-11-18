# Engines kept in RAM with their weights. Unloading an engine loses its weights, and loading it again means refitting
# every one. So an engine is serialized with its weights before it unloads, and a later load for the same weights
# (same checkpoint, same LoRAs) deserializes that instead. A kept engine is about the size of its weights (GBs).
#
# The OS decides how long a kept engine stays. Each is held in memory the OS may take back whenever it needs RAM,
# without writing it anywhere, and is checked to be whole before each use: an engine missing any page is dropped, and
# the load refits instead. Nothing here reads memory pressure: the OS takes back what it needs.
#   Linux 4.12 on: a private anonymous mapping released with MADV_FREE. The kernel reclaims such pages like file pages,
#     without swapping them, and only as many as it needs: on Linux 6.18 (without MGLRU) under a memory cgroup's limit,
#     it took them from the oldest mapping first. (Before 4.12, without swap, it frees them at once.) The serialized
#     copy TensorRT returns goes back to the kernel as it's copied, so keeping an engine takes about its size in RAM at
#     any moment. A page the kernel took back reads as zeros, so each page's first two 8-byte words are replaced by
#     markers, the originals kept in _saved. Before use, the first words are checked and restored: writing a page takes
#     it back from the kernel. The second words are then checked and restored, which catches a page taken between the
#     first check and that write (the write lands on a fresh zero page). After use, the markers go back and the mapping
#     is released again.
#   Windows: VirtualAlloc'd memory released with OfferVirtualMemory and taken back with ReclaimVirtualMemory, which
#     reports whether Windows discarded any of it (Windows 8.1 Update on).
#   Elsewhere nothing is kept (TensorRT doesn't run on macOS).
# Where allocations count against a commit limit (Windows, and Linux with vm.overcommit_memory = 2), kept engines still
# do, and the OS doesn't take them back to make commit available. So there a watcher thread drops the least recently
# used engine whenever available commit is under FLOOR of the limit.
#
# Keeping one needs RAM for it while it's copied (on Windows, twice its size: TensorRT's copy and ours), with FLOOR of
# RAM left free. Engines kept earlier make way, least recently used first, except any loaded since this engine's
# previous load (_may_go).
import contextlib
import ctypes
import mmap
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .common import gib, logger

FLOOR = 0.05
READS = 5  # tries at reading memory counts before keeping an engine anyway: empty counts mean the reading failed
INTERVAL = 2.0  # seconds between checks of the commit limit while engines are kept (where there is one)
PAGE = mmap.PAGESIZE
# Linux: the copy releases each CHUNK as soon as it's written, so at most this much of it is held that the kernel can't
# take back: small next to an engine (GBs).
CHUNK = 64 << 20
# Linux's values on every architecture. Python's mmap module defines the names only if its build's headers had them.
_MADV_DONTNEED, _MADV_FREE, _MADV_DONTFORK, _MADV_HUGEPAGE = 4, 8, 10, 14
_MARKS = (0x5452545245464954, 0x4B45505452414D53)  # any nonzero words: a page the kernel took back reads as zeros
_OFFER_PRIORITY = 2  # VmOfferPriorityLow: Windows discards lower priorities first, and a kept engine can be refitted


class _Memory:
    # Memory the OS may take back while it isn't open. open() yields the data as a memoryview, or None if the OS took
    # any of it, which frees the rest.
    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.size = -(-nbytes // PAGE) * PAGE
        self._state = threading.Lock()
        self._open = self._doomed = self._freed = False

    @contextlib.contextmanager
    def open(self):
        with self._state:
            try:
                whole = not self._freed and not self._doomed and self._take()
            except Exception as e:
                logger.debug(f"Couldn't check a kept engine's memory: {e}")
                whole = False
            if not whole:
                self._release()
            self._open = whole
        if not whole:
            yield None
            return
        full = self._view()
        data = full[: self.nbytes]
        try:
            yield data
        finally:
            data.release()
            full.release()
            with self._state:
                self._open = False
                if self._doomed or not self._give():
                    self._release()

    def free(self):
        # Memory that's open is freed when open() exits
        with self._state:
            if self._open:
                self._doomed = True
            else:
                self._release()

    @property
    def freed(self) -> bool:
        return self._freed or self._doomed

    def resident(self) -> int:
        return 0 if self.freed else self.size

    def intact(self) -> bool:
        return not self.freed


class _LinuxMemory(_Memory):
    pagemap = True  # whether /proc/self/pagemap shows which pages are in RAM; _probe() checks

    def __init__(self, nbytes: int):
        super().__init__(nbytes)
        self._map = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # MADV_DONTFORK: a forked child (a subprocess before its exec) would share the pages, and MADV_FREE skips shared
        # pages. MADV_HUGEPAGE: with 2 MB pages, keeping 2 GB took 0.63s instead of 1.44s and the check before use 61 ms
        # instead of 463 ms (Linux 6.18 VM, transparent huge pages on request).
        for advice in (_MADV_DONTFORK, _MADV_HUGEPAGE):
            try:
                self._map.madvise(advice)
            except OSError:
                pass
        view = np.frombuffer(self._map, dtype=np.uint8)
        self._address = view.ctypes.data
        del view
        self._saved = np.empty((self.size // PAGE, 2), dtype=np.uint64)

    def _words(self):
        return np.frombuffer(self._map, dtype=np.uint64).reshape(-1, PAGE // 8)

    def fill(self, source, consume: bool = False):
        # consume: return source's whole pages to the kernel as they're copied (MADV_DONTNEED; they read as zeros
        # after). For a buffer about to be freed that nothing else reads: malloc'd memory, with glibc or tcmalloc.
        data = np.frombuffer(source, dtype=np.uint8)
        target = np.frombuffer(self._map, dtype=np.uint8)
        words = self._words()
        address = data.ctypes.data
        returned = -(-address // PAGE) * PAGE  # the source's first whole page
        try:
            for start in range(0, self.size, CHUNK):
                end = min(start + CHUNK, self.size)
                if start < self.nbytes:
                    target[start : min(end, self.nbytes)] = data[start : min(end, self.nbytes)]
                pages = slice(start // PAGE, end // PAGE)
                self._saved[pages] = words[pages, :2]
                words[pages, 0], words[pages, 1] = _MARKS
                self._map.madvise(_MADV_FREE, start, end - start)
                copied = (address + min(end, self.nbytes)) // PAGE * PAGE
                if consume and copied > returned:
                    _libc().madvise(returned, copied - returned, _MADV_DONTNEED)
                    returned = copied
        finally:
            del words, target, data

    def _take(self) -> bool:
        words = self._words()
        try:
            if not (words[:, 0] == _MARKS[0]).all():
                return False
            words[:, 0] = self._saved[:, 0]
            if not (words[:, 1] == _MARKS[1]).all():
                return False
            words[:, 1] = self._saved[:, 1]
            return True
        finally:
            del words

    def _give(self) -> bool:
        words = self._words()
        try:
            words[:, 0], words[:, 1] = _MARKS
            self._map.madvise(_MADV_FREE)
            return True
        except OSError:
            return False
        finally:
            del words

    def _view(self):
        return memoryview(self._map)

    def _release(self):
        if self._freed:
            return
        self._freed = True
        self._saved = None
        try:
            self._map.close()
        except BufferError:  # a view of it is still referenced somewhere: return its pages, unmapped once that's gone
            self._map.madvise(_MADV_DONTNEED)

    def resident(self) -> int:
        if self.freed or self._open or not self.pagemap:
            return super().resident()
        want, raw = self.size // PAGE * 8, b""
        try:
            with open("/proc/self/pagemap", "rb") as f:
                while len(raw) < want:
                    more = os.pread(f.fileno(), want - len(raw), self._address // PAGE * 8 + len(raw))
                    if not more:
                        return self.size
                    raw += more
        except OSError:
            return self.size
        return int(np.count_nonzero(np.frombuffer(raw, dtype=np.uint64) >> np.uint64(63))) * PAGE  # bit 63: present

    def intact(self) -> bool:
        return self.resident() == self.size


_k32 = None


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in ("phys", "phys_free", "commit_limit", "commit_free", "virtual", "virtual_free", "extended")
    ]


def _kernel32():
    global _k32
    if _k32 is None:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong)
        k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong)
        k32.OfferVirtualMemory.restype = ctypes.c_ulong
        k32.OfferVirtualMemory.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        k32.ReclaimVirtualMemory.restype = ctypes.c_ulong
        k32.ReclaimVirtualMemory.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        k32.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(_MemoryStatus),)
        _k32 = k32
    return _k32


def _commit():
    # (commit available, commit limit) in bytes, or None where allocations don't count against a limit
    if os.name == "nt":
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(f"GlobalMemoryStatusEx failed (error {ctypes.get_last_error()})")
        return int(status.commit_free), int(status.commit_limit)
    try:
        with open("/proc/sys/vm/overcommit_memory") as f:
            if f.read().strip() != "2":
                return None
    except OSError:
        return None
    m = _meminfo()
    return m["CommitLimit"] - m["Committed_AS"], m["CommitLimit"]


class _WindowsMemory(_Memory):
    def __init__(self, nbytes: int):
        super().__init__(nbytes)
        self._address = _kernel32().VirtualAlloc(None, self.size, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        if not self._address:
            raise MemoryError(f"Windows couldn't commit {gib(self.size)} (error {ctypes.get_last_error()})")

    def _view(self):
        return memoryview((ctypes.c_ubyte * self.size).from_address(self._address))

    def fill(self, source, consume: bool = False):
        data = np.frombuffer(source, dtype=np.uint8)
        target = np.frombuffer((ctypes.c_ubyte * self.size).from_address(self._address), dtype=np.uint8)
        try:
            target[: self.nbytes] = data
        finally:
            del target, data
        if not self._give():
            raise OSError(f"OfferVirtualMemory failed (error {ctypes.get_last_error()})")

    def _take(self) -> bool:
        return _kernel32().ReclaimVirtualMemory(self._address, self.size) == 0  # ERROR_BUSY (170): some was discarded

    def _give(self) -> bool:
        return _kernel32().OfferVirtualMemory(self._address, self.size, _OFFER_PRIORITY) == 0

    def _release(self):
        if not self._freed:
            self._freed = True
            _kernel32().VirtualFree(self._address, 0, 0x8000)  # MEM_RELEASE


_libc_handle = None


def _libc():
    global _libc_handle
    if _libc_handle is None:
        libc = ctypes.CDLL(None)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        _libc_handle = libc
    return _libc_handle


_support = None


def _probe() -> tuple:
    # (memory class or None, why nothing can be kept)
    if sys.platform.startswith("linux"):
        release = os.uname().release
        version = re.match(r"(\d+)\.(\d+)", release)
        if not version or tuple(map(int, version.groups())) < (4, 12):
            return None, f"Linux {release} can't take memory back only when it needs it (Linux 4.12 on can)"
        try:
            memory = _LinuxMemory(2 * PAGE)
            try:
                memory.fill(np.ones(2 * PAGE, dtype=np.uint8))
                _LinuxMemory.pagemap = memory.resident() == memory.size
            finally:
                memory.free()
        except (OSError, ValueError) as e:
            return None, f"this system won't let memory be taken back when it's needed (MADV_FREE: {e})"
        return _LinuxMemory, ""
    if os.name == "nt":
        try:
            _kernel32()
        except (OSError, AttributeError) as e:
            return None, f"this Windows can't take memory back when it needs it (OfferVirtualMemory, Windows 8.1 Update on: {e})"
        return _WindowsMemory, ""
    return None, "this system can't take memory back when it needs it"


def _supported() -> tuple:
    global _support
    if _support is None:
        _support = _probe()
    return _support


@dataclass
class Kept:
    signatures: dict
    blobs: dict  # role ("unet", "shallow") -> _Memory holding the serialized engine
    label: str
    source: object = None

    @property
    def nbytes(self) -> int:
        return sum(memory.nbytes for memory in self.blobs.values())

    def resident(self) -> int:
        return sum(memory.resident() for memory in self.blobs.values())

    def free(self):
        for memory in self.blobs.values():
            memory.free()

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


def _read() -> tuple:
    if sys.platform.startswith("linux"):
        # Counted as htop does, not from MemAvailable, which reads 0 on some systems with most of their RAM free: free
        # memory, buffers and droppable caches, less shared memory (which can't be dropped). MADV_FREE pages, kept
        # engines included, aren't in it.
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


def _measure(reads: int = 1) -> tuple:
    for attempt in range(reads):
        try:
            return _read()
        except Exception:
            if attempt == reads - 1:
                raise
            time.sleep(0.2)


def memory(reads: int = 1) -> tuple:
    # (RAM the system can give, kept engines included; swap free; RAM in all)
    supply, swap, total = _measure(reads)
    if sys.platform.startswith("linux"):
        with _lock:
            entries = list(_kept.values())
        supply += sum(entry.resident() for entry in entries)
    return supply, swap, total


def explain() -> str:
    if "unsupported" in _last:
        return _last["unsupported"]
    if "error" in _last:
        return f"RAM couldn't be read ({_last['error']})"
    if not _last:
        return "nothing weighed"
    floor = f", and {gib(_last['floor'])} must stay free" if _last["floor"] else ""
    others = f"; engines kept that could make way hold {gib(_last['others'])}" if _last["others"] else ""
    commit = ""
    if "commit" in _last:
        available, limit = _last["commit"]
        commit = f"; {gib(available)} of the commit limit ({gib(limit)}) is left, and {gib(FLOOR * limit)} must stay so"
    return f"keeping it takes {gib(_last['need'])} for a while, {gib(_last['supply'])} is free{floor}{others}{commit}"


def _give_back():
    # Makes the allocator return freed memory to the OS now (tcmalloc, preloaded by webui.sh, holds it for a while),
    # so the OS counts it as available.
    if os.name == "nt":
        return
    try:
        libc = _libc()
        if hasattr(libc, "MallocExtension_ReleaseFreeMemory"):
            libc.MallocExtension_ReleaseFreeMemory()
        elif hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def store(serialize):
    # Runs serialize() (returning a buffer nothing else uses) and copies the result into memory the OS can take back;
    # the serialized copy is freed by the time this returns.
    kind, why = _supported()
    if kind is None:
        raise RuntimeError(why)
    source = serialize()
    try:
        memory = kind(memoryview(source).nbytes)
        try:
            memory.fill(source, consume=True)
        except BaseException:
            memory.free()
            raise
        return memory
    finally:
        del source
        _give_back()


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
            gone = list(entry.blobs.values())
        else:
            gone = [entry.blobs.pop(role)] if role in entry.blobs else []
            entry.signatures.pop(role, None)
    for memory in gone:
        memory.free()


def clear():
    with _lock:
        entries = list(_kept.values())
        _kept.clear()
        _uses.clear()
    for entry in entries:
        entry.free()


def held() -> tuple:
    with _lock:
        return len(_kept), sum(entry.nbytes for entry in _kept.values())


def resident() -> int:
    with _lock:
        entries = list(_kept.values())
    return sum(entry.resident() for entry in entries)


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
    kind, why = _supported()
    if kind is None:
        _last["unsupported"] = why
        return False
    try:
        supply, _, total = _measure(READS)
        commit = _commit()
    except Exception as e:
        _last["error"] = f"{type(e).__name__}: {e}"
        return True
    floor = int(FLOOR * total)
    # Linux: TensorRT's serialized copy shrinks in RAM as ours grows, which holds CHUNK the kernel can't take back yet.
    # Commit: both count until TensorRT's is freed.
    need = nbytes + (CHUNK if kind is _LinuxMemory else nbytes)

    def fits(ram: int, commit) -> bool:
        return ram - need >= floor and (commit is None or commit[0] - 2 * nbytes >= FLOOR * commit[1])

    dropped = []
    with _lock:
        if kind is _WindowsMemory:
            supply -= sum(entry.nbytes for entry in _kept.values())  # Windows can count offered memory as available
        others = [k for k in _kept if k != key and _may_go(k, key)]
        # What dropping each would return: on Linux only its pages still in RAM (pagemap)
        gains = {k: (_kept[k].resident() if kind is _LinuxMemory else _kept[k].nbytes, _kept[k].nbytes) for k in others}
        _last.update(need=need, supply=supply, floor=floor, others=sum(ram for ram, _ in gains.values()))
        if commit is not None:
            _last["commit"] = commit
        everything = (commit[0] + sum(c for _, c in gains.values()), commit[1]) if commit is not None else None
        if not fits(supply + _last["others"], everything):
            return False
        while not fits(supply, commit) and others:
            k = others.pop(0)
            ram, committed = gains[k]
            supply += ram
            if commit is not None:
                commit = (commit[0] + committed, commit[1])
            dropped.append(_kept.pop(k))
    for entry in dropped:
        logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}) to make room for another.")
        entry.free()
    return fits(supply, commit)


def put(key: str, entry: Kept):
    global _watcher
    with _lock:
        old = _kept.pop(key, None)
        _kept[key] = entry
        if _watcher is None and _commit_limited():
            _watcher = threading.Thread(target=_watch, name="trt-refits", daemon=True)
            _watcher.start()
    if old is not None:
        for memory in old.blobs.values():
            if not any(memory is kept for kept in entry.blobs.values()):
                memory.free()


def _commit_limited() -> bool:
    try:
        return _commit() is not None
    except Exception:
        return False


def _trim_commit() -> int:
    commit = _commit()
    if commit is None:
        return 0
    available, limit = commit
    dropped = 0
    while available < FLOOR * limit:
        with _lock:
            if not _kept:
                break
            entry = _kept.pop(next(iter(_kept)))
        logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): {gib(available)} of the commit limit ({gib(limit)}) was left.")
        entry.free()
        available += entry.nbytes
        dropped += 1
    return dropped


def trim() -> int:
    with _lock:
        gone = [_kept.pop(key) for key, entry in list(_kept.items()) if entry.orphaned]
    for entry in gone:
        logger.debug(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): its checkpoint isn't loaded anymore.")
        entry.free()
    dropped = len(gone)
    try:
        dropped += _trim_commit()
    except Exception:
        pass
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
            _trim_commit()
        except Exception as e:
            if not failed:
                failed = True
                logger.warning(f"Couldn't check whether the system needs the commit that engines kept in RAM hold: {e}")
