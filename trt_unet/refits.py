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
import concurrent.futures
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
_FILL_SUB = 8 << 20  # L3-friendly sub-chunk for parallel memcpy + page-marking inside each CHUNK
_TAKE_STEP = 512  # 512 pages = 32 KiB of first cachelines: fits in L1d so each page is touched once
# Linux's values on every architecture. Python's mmap module defines the names only if its build's headers had them.
_MADV_DONTNEED, _MADV_FREE, _MADV_DONTFORK, _MADV_HUGEPAGE = 4, 8, 10, 14
_MARKS = (0x5452545245464954, 0x4B45505452414D53)  # any nonzero words: a page the kernel took back reads as zeros
_OFFER_PRIORITY = 2  # VmOfferPriorityLow: Windows discards lower priorities first, and a kept engine can be refitted


def _fill_workers() -> int:
    return min(4, max(1, (os.cpu_count() or 2) // 2))


class _Memory:
    # Memory the OS may take back while it isn't open. open() yields the data as a memoryview, or None if the OS took
    # any of it, which frees the rest.
    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.size = -(-nbytes // PAGE) * PAGE
        self._state = threading.Lock()
        self._open = self._doomed = self._freed = False
        self._fill_done = threading.Event()
        self._fill_done.set()
        self._fill_error = None

    def start_fill(self):
        with self._state:
            self._fill_error = None
            self._fill_done.clear()

    def _end_fill(self, error=None) -> bool:
        # Records fill result and releases failed/doomed memory while keeping _fill_done unset so waiters wait until
        # on_error and _pending_fills cleanup finish. Returns True iff on_error should be notified.
        with self._state:
            self._fill_error = error
            report = error is not None and not self._doomed
            if error is not None or self._doomed:
                self._release()
            return report

    def _complete_fill(self):
        # Holds _state across the final _doomed check and _fill_done.set() so free() cannot miss _release().
        with self._state:
            if self._fill_error is not None or self._doomed:
                self._release()
            self._fill_done.set()

    def finish_fill(self, error=None):
        with self._state:
            self._fill_error = error
            if error is not None or self._doomed:
                self._release()
            self._fill_done.set()

    def wait_fill(self):
        self._fill_done.wait()

    @contextlib.contextmanager
    def open(self):
        self._fill_done.wait()
        with self._state:
            try:
                whole = not self._freed and not self._doomed and self._fill_error is None and self._take()
            except Exception as e:
                logger.warning(f"Couldn't check a kept engine's memory: {e}")
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
        # Memory that's open or filling is freed when open()/fill exits
        with self._state:
            if self._open or not self._fill_done.is_set():
                self._doomed = True
            else:
                self._release()

    @property
    def freed(self) -> bool:
        return self._freed or self._doomed

    def resident(self) -> int:
        return 0 if self.freed or not self._fill_done.is_set() else self.size

    def intact(self) -> bool:
        self._fill_done.wait()
        return not self.freed and self._fill_error is None


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
        saved = self._saved
        address = data.ctypes.data
        returned = -(-address // PAGE) * PAGE  # the source's first whole page
        workers = _fill_workers() if self.size >= (16 << 20) else 1
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

        def copy_mark(span):
            s, e = span
            if s < self.nbytes:
                target[s : min(e, self.nbytes)] = data[s : min(e, self.nbytes)]
            ps = slice(s // PAGE, e // PAGE)
            saved[ps] = words[ps, :2]
            words[ps, 0] = _MARKS[0]
            words[ps, 1] = _MARKS[1]

        try:
            for start in range(0, self.size, CHUNK):
                if self._doomed:
                    return
                end = min(start + CHUNK, self.size)
                if pool is None:
                    copy_mark((start, end))
                else:
                    spans = [(s, min(s + _FILL_SUB, end)) for s in range(start, end, _FILL_SUB)]
                    list(pool.map(copy_mark, spans))
                self._map.madvise(_MADV_FREE, start, end - start)
                copied = (address + min(end, self.nbytes)) // PAGE * PAGE
                if consume and copied > returned:
                    _libc().madvise(returned, copied - returned, _MADV_DONTNEED)
                    returned = copied
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            del words, target, data, saved

    def _take(self) -> bool:
        words = self._words()
        saved = self._saved
        n_pages = self.size // PAGE
        w = sv = None
        try:
            for s in range(0, n_pages, _TAKE_STEP):
                e = min(s + _TAKE_STEP, n_pages)
                w = words[s:e]
                sv = saved[s:e]
                if not (w[:, 0] == _MARKS[0]).all():
                    return False
                w[:, 0] = sv[:, 0]
                if not (w[:, 1] == _MARKS[1]).all():
                    return False
                w[:, 1] = sv[:, 1]
            return True
        finally:
            del w, sv, words, saved

    def _give(self) -> bool:
        words = self._words()
        n_pages = self.size // PAGE
        try:
            if n_pages >= 4096 and (os.cpu_count() or 1) >= 2:
                mid = n_pages // 2

                def mark(span):
                    a, b = span
                    for s in range(a, b, _TAKE_STEP):
                        w = words[s : min(s + _TAKE_STEP, b)]
                        w[:, 0] = _MARKS[0]
                        w[:, 1] = _MARKS[1]

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(mark, ((0, mid), (mid, n_pages))))
            else:
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

    def _pagemap_present(self):
        want = self.size // PAGE * 8
        buf = bytearray(want)
        mv = memoryview(buf)
        try:
            with open("/proc/self/pagemap", "rb", buffering=0) as f:
                f.seek(self._address // PAGE * 8)
                got = 0
                while got < want:
                    n = f.readinto(mv[got:])
                    if not n:
                        return None
                    got += n
        except OSError:
            return None
        finally:
            mv.release()
        return int(np.count_nonzero(np.frombuffer(buf, dtype=np.uint64) >> np.uint64(63))) * PAGE  # bit 63: present

    def resident(self) -> int:
        if self.freed or self._open or not self._fill_done.is_set() or not self.pagemap:
            return super().resident()
        present = self._pagemap_present()
        if present is None:
            _LinuxMemory.pagemap = False
            return self.size
        return present

    def intact(self) -> bool:
        self._fill_done.wait()
        return self._fill_error is None and self.resident() == self.size


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
        workers = _fill_workers() if self.nbytes >= (16 << 20) else 1
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

        def copy_span(span):
            s, e = span
            target[s:e] = data[s:e]

        try:
            for start in range(0, self.nbytes, CHUNK):
                if self._doomed:
                    return
                end = min(start + CHUNK, self.nbytes)
                if pool is None:
                    copy_span((start, end))
                else:
                    spans = [(s, min(s + _FILL_SUB, end)) for s in range(start, end, _FILL_SUB)]
                    list(pool.map(copy_span, spans))
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            del target, data
        if self._doomed:
            return
        rc = _kernel32().OfferVirtualMemory(self._address, self.size, _OFFER_PRIORITY)
        if rc != 0:
            raise OSError(f"OfferVirtualMemory failed (error {rc})")

    def _take(self) -> bool:
        return _kernel32().ReclaimVirtualMemory(self._address, self.size) == 0  # ERROR_BUSY (170): some was discarded

    def _give(self) -> bool:
        return _kernel32().OfferVirtualMemory(self._address, self.size, _OFFER_PRIORITY) == 0

    def _release(self):
        if not self._freed:
            self._freed = True
            if getattr(self, "_address", None):
                _kernel32().VirtualFree(self._address, 0, 0x8000)  # MEM_RELEASE

    def __del__(self):
        with contextlib.suppress(Exception):
            self._release()


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
                _LinuxMemory.pagemap = memory._pagemap_present() == memory.size
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

    def free(self, wait: bool = False):
        for memory in self.blobs.values():
            memory.free()
        if wait:
            for memory in self.blobs.values():
                if hasattr(memory, "wait_fill"):
                    memory.wait_fill()

    @property
    def orphaned(self) -> bool:
        return self.source is not None and self.source() is None


_kept = {}
_uses = {}  # engine key -> (load count at its previous load, load count at its latest load); see _may_go
_count = 0
_lock = threading.RLock()
_watcher = None
_fill_exec = None
_pending_fills = set()


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
            blobs = [m for entry in _kept.values() for m in entry.blobs.values()]
        supply += sum(m.resident() for m in blobs)
    return supply, swap, total


def explain() -> str:
    if "unsupported" in _last:
        return _last["unsupported"]
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


def _wait_pending():
    with _lock:
        pending = list(_pending_fills)
    for m in pending:
        m.wait_fill()


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


def store_async(source, on_error=None):
    # Hands a serialized buffer to a single-worker background thread so engine.unload() and the next engine's load can
    # overlap with the RAM copy. open(), room() and clear() synchronize automatically with any in-flight fill.
    global _fill_exec
    kind, why = _supported()
    if kind is None:
        del source
        _give_back()
        raise RuntimeError(why)
    try:
        memory = kind(memoryview(source).nbytes)
    except BaseException:
        del source
        _give_back()
        raise
    memory.start_fill()
    with _lock:
        if _fill_exec is None:
            _fill_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt-refit-fill")
        _pending_fills.add(memory)
        executor = _fill_exec

    def _run():
        nonlocal source
        err = None
        try:
            if not memory._doomed:
                memory.fill(source, consume=True)
        except BaseException as e:
            err = e
        finally:
            source = None
            _give_back()
            report = memory._end_fill(err)
            try:
                if report and on_error is not None:
                    with contextlib.suppress(Exception):
                        on_error(err, memory)
            finally:
                with _lock:
                    _pending_fills.discard(memory)
                memory._complete_fill()

    try:
        executor.submit(_run)
    except BaseException:
        with _lock:
            _pending_fills.discard(memory)
        source = None
        _give_back()
        memory.finish_fill(RuntimeError("couldn't schedule background RAM fill"))
        raise
    return memory


def get(key: str):
    with _lock:
        entry = _kept.pop(key, None)
        if entry is not None:
            _kept[key] = entry
        return entry


def drop(key: str, role: str = None, blob=None):
    with _lock:
        entry = _kept.get(key)
        if entry is None:
            return
        if blob is not None and entry.blobs.get(role) is not blob:
            return
        if role in (None, "unet"):
            del _kept[key]
            gone = list(entry.blobs.values())
        elif role in entry.blobs or role in entry.signatures:
            gone = [entry.blobs[role]] if role in entry.blobs else []
            blobs = {k: v for k, v in entry.blobs.items() if k != role}
            signatures = {k: v for k, v in entry.signatures.items() if k != role}
            _kept[key] = Kept(signatures, blobs, entry.label, entry.source)
        else:
            gone = []
    for memory in gone:
        memory.free()


def clear():
    with _lock:
        entries = list(_kept.values())
        _kept.clear()
        _uses.clear()
    for entry in entries:
        entry.free(wait=True)
    _wait_pending()


def held() -> tuple:
    with _lock:
        return len(_kept), sum(entry.nbytes for entry in _kept.values())


def resident() -> int:
    with _lock:
        blobs = [m for entry in _kept.values() for m in entry.blobs.values()]
    return sum(m.resident() for m in blobs)


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
    _wait_pending()
    _last.clear()
    kind, why = _supported()
    if kind is None:
        _last["unsupported"] = why
        return False
    try:
        supply, _, total = _measure(READS)
        commit = _commit()
    except Exception:
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
        logger.info(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}) to make room for another.")
        entry.free(wait=True)
    return fits(supply, commit)


def put(key: str, entry: Kept, expected: Kept = None) -> bool:
    global _watcher
    with _lock:
        if expected is not None and _kept.get(key) is not expected:
            return False  # _trim_commit() or another caller dropped expected while new roles were serializing
        bad = [r for r, m in entry.blobs.items() if getattr(m, "_fill_error", None) is not None or m.freed]
        if bad:
            if "unet" in bad or len(bad) == len(entry.blobs):
                return False
            blobs = {r: m for r, m in entry.blobs.items() if r not in bad}
            signatures = {r: s for r, s in entry.signatures.items() if r not in bad}
            entry = Kept(signatures, blobs, entry.label, entry.source)
        old = _kept.pop(key, None)
        _kept[key] = entry
        if _watcher is None and _commit_limited():
            _watcher = threading.Thread(target=_watch, name="trt-refits", daemon=True)
            _watcher.start()
    if old is not None:
        for memory in old.blobs.values():
            if not any(memory is kept for kept in entry.blobs.values()):
                memory.free()
    return True


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
        logger.info(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): {gib(available)} of the commit limit ({gib(limit)}) was left.")
        entry.free()
        available += entry.nbytes
        dropped += 1
    return dropped


def trim() -> int:
    with _lock:
        gone = [_kept.pop(key) for key, entry in list(_kept.items()) if entry.orphaned]
    for entry in gone:
        logger.info(f"Dropped the {entry.label} from RAM ({gib(entry.nbytes)}): its checkpoint isn't loaded anymore.")
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
