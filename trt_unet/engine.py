import atexit
import contextlib
import ctypes
import math
import os
import time
import weakref
from collections import deque

import tensorrt as trt
import torch

from . import common, gpu
from .common import gib, logger

# The refitter prepares host weights on one CPU thread by default, and scans every weight for NaN/Inf. The weights are
# PyTorch's own FP16 tensors and each pass checks the engine's first output for NaN/Inf, so the scan is skipped.
REFITTER_SETTINGS = {"max_threads": os.cpu_count() or 1, "weights_validation": False}
_unsettable = set()
# TRT_EXTENSION_PROFILE=1: per-layer GPU times from TensorRT's profiler, logged after each sampling pass, decode and
# encode. Every engine run then waits for the GPU, so everything is slower.
PROFILE = os.environ.get("TRT_EXTENSION_PROFILE", "").lower() in ("1", "true", "yes")
_layer_times = {}


_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


def torch_dtype(dtype):
    if dtype not in _DTYPES:
        raise TypeError(f"Unsupported TensorRT dtype {dtype}")
    return _DTYPES[dtype]


class Logger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.errors = []
        self.messages = deque(maxlen=50)

    def clear(self):
        self.errors.clear()
        self.messages.clear()

    def log(self, severity, msg):
        if severity == trt.ILogger.Severity.VERBOSE:
            return
        self.messages.append(msg)
        if int(severity) <= int(trt.ILogger.Severity.ERROR):
            self.errors.append(msg)
        logger.debug(f"Runtime {severity.name.lower()}: {msg}")

    @property
    def out_of_memory(self) -> bool:
        return any("OutOfMemory" in m for m in self.messages)


LOGGER = Logger()
_runtime = None
_set_aside = []  # runtimes replaced after an allocator failure, kept alive: engines may still belong to them


class _LayerTimes(trt.IProfiler):
    def __init__(self, plan_path: str):
        trt.IProfiler.__init__(self)
        self.times = _layer_times.setdefault(plan_path, {})

    def report_layer_time(self, layer_name, ms):
        self.times[layer_name] = self.times.get(layer_name, 0.0) + ms


def profile_report(where: str, top: int = 12):
    if not PROFILE:
        return
    for plan, times in sorted(_layer_times.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(times.values())
        if not total:
            continue
        longest = sorted(times.items(), key=lambda kv: -kv[1])[:top]
        logger.info(
            f"Layer times of {os.path.basename(plan)} for the {where}: {total:.1f} ms in {len(times)} layers. The slowest:\n"
            + "\n".join(f"    {ms:8.1f} ms  {name}" for name, ms in longest)
        )
        times.clear()


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = trt.Runtime(LOGGER)
        gpu.attach(_runtime)  # must happen before the first deserialize
    return _runtime


def _deserialize(plan, owner):
    global _runtime
    before = gpu.failures()
    had_live = gpu.ledger(owner) > 0 or any(e.loaded for e in Engine.instances)
    with gpu.owner(owner):
        engine = runtime().deserialize_cuda_engine(plan)
    if engine is not None or LOGGER.out_of_memory or had_live or not gpu.suspect(before, owner):
        return engine
    own = trt.Runtime(LOGGER)
    engine = own.deserialize_cuda_engine(plan)
    if engine is not None:
        _set_aside.append(_runtime)
        _runtime = own
        gpu.retire()
        logger.warning("An engine wouldn't load with PyTorch's allocator, so TensorRT manages its own VRAM from now on.")
    return engine


def own_allocator(failures_before: int, owner, error) -> bool:
    global _runtime
    if any(e.loaded for e in Engine.instances) or not gpu.suspect(failures_before, owner):
        return False
    _set_aside.append(_runtime)
    with common.on_device():
        _runtime = trt.Runtime(LOGGER)
    gpu.retire()
    logger.warning(f"Loading the {owner} failed with PyTorch's allocator ({error}), so TensorRT manages its own VRAM from now on. Trying again.")
    return True


@atexit.register
def _unload_all():  # engines free through PyTorch's allocator, which must happen before interpreter shutdown tears it down
    for engine in list(Engine.instances):
        try:
            engine.unload()
        except Exception:
            pass


def other_tensorrt(maps="/proc/self/maps") -> str:
    if os.name == "nt":  # Windows binds each DLL to the libraries it names, so this can't happen
        return ""
    major = int(trt.__version__.split(".")[0])
    try:
        version = ctypes.CDLL(None).getInferLibVersion()
    except (AttributeError, OSError):
        return ""
    if version // 10000 == major:
        return ""
    try:
        with open(maps) as f:
            paths = sorted({line.split()[-1] for line in f if "/libnvinfer.so." in line and not line.rstrip().endswith(f"libnvinfer.so.{major}")})
    except OSError:
        paths = []
    where = f" ({', '.join(paths)})" if paths else ""
    return (
        f"TensorRT {version // 10000}.{version // 100 % 100}{where} is loaded process-wide, likely by another extension, "
        f"and may be answering in place of TensorRT {trt.__version__}. This extension's preload.py loads its own "
        "TensorRT first to prevent that, but something loaded the other one even earlier."
    )


class Arena:
    # Activation memory shared by engines that never run at once (an engine and its Speed Hacks partner). It only
    # grows until released. `version` changes on every reallocation; an engine whose bound_version differs rebinds.
    instances = weakref.WeakSet()

    def __init__(self):
        self.memory, self.version = None, 0
        Arena.instances.add(self)

    def get(self, size: int):
        if self.memory is None or self.memory.numel() < size:
            dev = common.device()
            self.release()  # bumps the version even if the allocation below fails, so nothing stays bound to freed memory
            try:
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=dev)
            except torch.cuda.OutOfMemoryError:
                if not common.pooled():
                    raise
                common.trim()  # PyTorch's freed blocks may be unusable for this size until returned to the driver
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=dev)
            self.version += 1
        return self.memory

    def release(self):
        if self.memory is not None:
            self.memory = None
            self.version += 1


_weight_names_cache = {}  # plan_path -> ((ino, mtime_ns, size), tuple[str, ...])


def _plan_stat(path: str):
    try:
        st = os.stat(path)
        return st.st_ino, st.st_mtime_ns, st.st_size
    except OSError:
        return None


class Engine:
    instances = weakref.WeakSet()

    def __init__(self, plan_path: str, arena: Arena = None, owner=None):
        self.plan_path = plan_path
        self.arena = arena or Arena()
        self.owner = owner
        self.engine = self.context = None
        self.extra = 0  # VRAM TensorRT took after load (context creation, first run), added to the owner's gpu._held
        self._inputs = ()
        self._outputs = ()
        self._reset_context_state()
        Engine.instances.add(self)

    def _reset_context_state(self):
        self.profile = None
        self.shapes = {}
        self._out_shapes = {}
        self.memory_bound = False
        self.bound_version = None
        self.size = None
        self.size_is_most = False  # TensorRT gave no figure for these shapes: `size` is the profile's maximum
        self.ran = False

    def twin(self):  # another execution context on the same engine and weights, so callers alternating shapes don't thrash one
        other = Engine(self.plan_path, self.arena, self.owner)
        other.engine, other.io, other.weight_names = self.engine, self.io, self.weight_names
        other._inputs, other._outputs = self._inputs, self._outputs
        return other

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load(self, serialized=None) -> float:
        start = time.perf_counter()
        LOGGER.clear()
        plan = serialized
        if plan is None:
            with open(self.plan_path, "rb") as f:
                plan = f.read()
        with common.on_device():
            self.engine = _deserialize(plan, self.owner)
            if self.engine is None:
                said = LOGGER.errors[-1] if LOGGER.errors else "no reason given"
                other = "" if LOGGER.out_of_memory else other_tensorrt()
                what = os.path.basename(self.plan_path) + ("" if serialized is None else " as kept in RAM")
                raise RuntimeError(f"TensorRT {trt.__version__} couldn't load {what}: {said}" + (f" {other}" if other else ""))
            self.io = []
            inputs, outputs = [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                dtype = torch_dtype(self.engine.get_tensor_dtype(name))
                self.io.append((name, is_input, dtype))
                (inputs if is_input else outputs).append((name, dtype))
            self._inputs, self._outputs = tuple(inputs), tuple(outputs)
            version = _plan_stat(self.plan_path)
            cached = _weight_names_cache.get(self.plan_path) if version is not None else None
            if cached is not None and cached[0] == version:
                self.weight_names = list(cached[1])
            else:
                with gpu.owner(self.owner):
                    self.weight_names = list(trt.Refitter(self.engine, LOGGER).get_all_weights())
                if version is not None:
                    _weight_names_cache[self.plan_path] = (version, tuple(self.weight_names))
        self.context = None  # created on first use, after the weights are refitted in
        self._reset_context_state()
        return time.perf_counter() - start

    @contextlib.contextmanager
    def _taking(self, what: str = None):
        # Inside gpu.loading for this owner, the load's own accounting already counts what's allocated here.
        if gpu.counting(self.owner):
            with gpu.owner(self.owner):
                yield
            return
        before = gpu.ledger(self.owner)
        try:
            with gpu.owner(self.owner):
                yield
        finally:
            more = gpu.ledger(self.owner) - before
            if more > 0:
                self.extra += more
                gpu.grew(self.owner, more)
                if what and more >= 16 * 2**20:
                    logger.debug(f"{os.path.basename(self.plan_path)} took {gib(more)} more VRAM {what}.")

    def _context(self):
        if self.context is None:
            LOGGER.clear()
            try:
                with common.on_device(), self._taking():
                    self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            except Exception:
                self.context = None
            if self.context is None:
                said = f": {LOGGER.errors[-1]}" if LOGGER.errors else ""
                raise RuntimeError(f"Could not create a TensorRT execution context{said}")
            if PROFILE:
                try:
                    self._profiler = _LayerTimes(self.plan_path)  # TensorRT doesn't keep the Python object alive
                    self.context.profiler = self._profiler
                    self.context.enqueue_emits_profile = False  # report_to_profiler() after the GPU finishes instead
                except Exception as e:
                    logger.warning(f"Couldn't profile {os.path.basename(self.plan_path)}: {e}")
            self._reset_context_state()
        return self.context

    def _drop_context(self):
        # TensorRT 11 keeps stale dynamic-resize state in live IExecutionContexts across refit_cuda_engine(), causing
        # execute_async_v3() to allocate max-profile scratch on every call (+11.9 GiB on SDXL VAE post, +1.1 GiB on UNet).
        if self.context is not None:
            with common.on_device(), gpu.owner(self.owner):
                before = gpu.ledger(self.owner)
                self.context = None
                freed = min(self.extra, max(0, before - gpu.ledger(self.owner)))
                if freed:
                    self.extra -= freed
                    gpu.unloaded(self.owner, freed)
            self._reset_context_state()

    def unload(self):
        if self.context is not None or self.engine is not None:
            dev = common.device()
            with common.on_device(dev), gpu.owner(self.owner):
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(dev)
                self.context = None  # before the engine it belongs to
                self.engine = None
        self._reset_context_state()
        self.arena.release()
        if self.extra:
            gpu.unloaded(self.owner, self.extra)
            self.extra = 0

    def serialize(self):
        with common.on_device():
            config = self.engine.create_serialization_config()
            config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)  # inherited from the weight-stripped plan
            config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)  # without it, the reloaded engine can never be refitted again
            LOGGER.clear()
            with gpu.owner(self.owner):
                serialized = self.engine.serialize_with_config(config)
        if serialized is None:
            raise RuntimeError("TensorRT could not serialize the engine" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return serialized

    # Under --cuda-malloc, TensorRT's ~Refitter() teardown scales super-linearly with the number of weights refitted in
    # a single Refitter instance (~4.0s at 1,680 SDXL weights vs ~1.2s across 2 balanced chunks). Chunking large refits
    # (with get_missing_weights() companion closure per chunk) + gpu.refit_pool() cuts --cuda-malloc refits >3x.
    ASYNC_REFIT_CHUNK = 850

    def refit(self, names, tensor_for) -> float:
        dev = common.device()
        names = list(names)
        if not names:
            return 0.0
        with common.on_device(dev):
            torch.cuda.synchronize(dev)
            for other in list(Engine.instances):
                if other.engine is self.engine:
                    other._drop_context()
            start = time.perf_counter()
            batch_limit = 32 * 2**20  # bounds extra VRAM when spilling GPU weights to HOST on tight VRAM
            if gpu.backend() == "cudaMallocAsync" and len(names) > self.ASYNC_REFIT_CHUNK:
                n_chunks = math.ceil(len(names) / self.ASYNC_REFIT_CHUNK)
                max_weights = math.ceil(len(names) / n_chunks)
            else:
                max_weights = len(names)
            source = getattr(tensor_for, "__self__", None)
            can_pipe = (
                source is not None
                and callable(getattr(source, "needs_gpu", None))
                and callable(getattr(source, "base", None))
                and callable(getattr(source, "apply_on_gpu", None))
            )
            try:
                free_b = max(0, common._free_now(dev, trim_cache=False) - common.reserve())
                if can_pipe and common.pooled() and free_b < (768 << 20):
                    common.trim()
                    free_b = max(0, common._free_now(dev, trim_cache=False) - common.reserve())
            except Exception:
                free_b = 0
            device_budget = max(0, (free_b - (512 << 20)) // 2)
            use_device = device_budget >= (128 << 20)

            # Double-buffered pinned-H2D + preallocated GPU staging avoids per-weight pageable stalls and allocator
            # churn when computing LoRA/FP8 weights on GPU. Standard LoRA (up, down) tensors are packed alongside base
            # into the same slot so one async DMA transfers all of them for in-place FP32 addmm_. Weights already
            # pinned in host RAM (e.g. Forge Neo --pin-shared-memory) DMA straight to GPU without a CPU memcpy.
            pins = gpu_bufs = copy_stream = compute_stream = ev_copy = ev_use = None
            gpu_buf_size = pin_size = 0
            has_fast_lora = (
                can_pipe
                and callable(getattr(source, "fast_lora", None))
                and callable(getattr(source, "apply_fast_lora", None))
            )
            fast_map = {}
            align16 = lambda n: (n + 15) & ~15
            if can_pipe:
                gpu_stage_sizes = []
                pin_stage_sizes = []
                for n in names:
                    if source.needs_gpu(n):
                        items = source.fast_lora(n) if has_fast_lora else None
                        if items is not None:
                            fast_map[n] = items
                        b = source.base(n)
                        if type(b) is torch.Tensor:
                            gpu_nb = pin_nb = 0
                            if not b.is_cuda and b.is_contiguous():
                                b_nb = align16(b.numel() * b.element_size())
                                gpu_nb += b_nb
                                if not b.is_pinned():
                                    pin_nb += b_nb
                            if items is not None:
                                for up, down, _ in items:
                                    if not up.is_cuda:
                                        u_nb = align16(up.numel() * up.element_size())
                                        gpu_nb += u_nb
                                        if not (up.is_contiguous() and up.is_pinned()):
                                            pin_nb += u_nb
                                    if not down.is_cuda:
                                        d_nb = align16(down.numel() * down.element_size())
                                        gpu_nb += d_nb
                                        if not (down.is_contiguous() and down.is_pinned()):
                                            pin_nb += d_nb
                            if gpu_nb > 0:
                                gpu_stage_sizes.append(gpu_nb)
                            if pin_nb > 0:
                                pin_stage_sizes.append(pin_nb)
                if len(gpu_stage_sizes) >= 2:
                    gpu_buf_size = min(32 << 20, max(gpu_stage_sizes))
                    pin_size = min(32 << 20, max(pin_stage_sizes)) if pin_stage_sizes else 0
                    if 2 * gpu_buf_size <= max(0, free_b - (256 << 20)) // 4:
                        try:
                            if pin_size > 0:
                                pins = [torch.empty(pin_size, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
                            gpu_bufs = [torch.empty(gpu_buf_size, dtype=torch.uint8, device=dev) for _ in range(2)]
                            copy_stream = torch.cuda.Stream(device=dev)
                            compute_stream = torch.cuda.current_stream(dev)
                            ev_copy = [torch.cuda.Event() for _ in range(2)]
                            ev_use = [torch.cuda.Event() for _ in range(2)]
                            # Order stream-ordered gpu_bufs allocations on compute_stream ahead of copy_stream writes.
                            ev_use[0].record(compute_stream)
                            copy_stream.wait_event(ev_use[0])
                        except RuntimeError:
                            pins = gpu_bufs = copy_stream = compute_stream = ev_copy = ev_use = None

            done, pos = set(), 0
            slot_used = [False, False]
            keep_pinned = []
            try:
                with self._taking(), gpu.refit_pool():
                    while pos < len(names):
                        while pos < len(names) and names[pos] in done:
                            pos += 1
                        if pos >= len(names):
                            break
                        keep, gpu_batch, gpu_bytes = [], [], 0
                        device_bytes, has_device = 0, False
                        refitter = trt.Refitter(self.engine, LOGGER)
                        try:
                            for setting, value in REFITTER_SETTINGS.items():
                                try:
                                    setattr(refitter, setting, value)
                                except (AttributeError, TypeError) as e:
                                    if setting not in _unsettable:
                                        _unsettable.add(setting)
                                        logger.debug(f"Couldn't set the refitter's {setting}: {e}")

                            def bind(name, proto, t, loc):
                                nonlocal device_bytes, has_device
                                if t.numel() != proto.size:
                                    raise RuntimeError(f"Weight {name} has {t.numel()} values, the engine expects {proto.size}")
                                keep.append(t)
                                if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), loc):
                                    raise RuntimeError(f"TensorRT rejected weight {name}")
                                if loc == trt.TensorLocation.DEVICE:
                                    device_bytes += t.numel() * t.element_size()
                                    has_device = True

                            def flush():
                                nonlocal gpu_bytes
                                while gpu_batch:
                                    name, proto, t = gpu_batch.pop(0)
                                    bind(name, proto, t.to("cpu"), trt.TensorLocation.HOST)
                                gpu_bytes = 0

                            def bind_tensor(name, proto, t):
                                nonlocal gpu_bytes
                                if t.is_cuda:
                                    nb = t.numel() * t.element_size()
                                    if use_device and device_bytes + nb <= device_budget:
                                        bind(name, proto, t, trt.TensorLocation.DEVICE)
                                    else:
                                        gpu_batch.append((name, proto, t))
                                        gpu_bytes += nb
                                        if gpu_bytes >= batch_limit:
                                            flush()
                                else:
                                    bind(name, proto, t, trt.TensorLocation.HOST)

                            def put(name, proto=None, dt=None):
                                if proto is None:
                                    proto = refitter.get_weights_prototype(name)
                                    dt = torch_dtype(proto.dtype)
                                raw = source.tensor(name, None, dt) if can_pipe else tensor_for(name, None)
                                t = raw.to(dtype=dt).contiguous()
                                bind_tensor(name, proto, t)

                            LOGGER.clear()
                            seen_in_chunk = set()
                            host_items, gpu_items = [], []
                            est_device = 0
                            while pos < len(names) and len(seen_in_chunk) < max_weights:
                                name = names[pos]
                                if name in done or name in seen_in_chunk:
                                    pos += 1
                                    continue
                                proto = refitter.get_weights_prototype(name)
                                dt = torch_dtype(proto.dtype)
                                if can_pipe and source.needs_gpu(name):
                                    nb = proto.size * dt.itemsize
                                    if use_device and est_device > 0 and est_device + nb > device_budget:
                                        break
                                    if use_device:
                                        est_device += nb
                                    gpu_items.append((name, proto, dt))
                                else:
                                    host_items.append((name, proto, dt))
                                seen_in_chunk.add(name)
                                pos += 1

                            for name, proto, dt in host_items:
                                put(name, proto, dt)
                            if gpu_items:

                                def stage(idx, name):
                                    base = source.base(name)
                                    if type(base) is not torch.Tensor:
                                        return None, None, None
                                    items = fast_map.get(name)
                                    base_on_dev = base.is_cuda and base.device == dev
                                    if items is None:
                                        if base_on_dev:
                                            return None, base, None
                                        nb = base.numel() * base.element_size()
                                        base_pinned = not base.is_cuda and base.is_contiguous() and base.is_pinned()
                                        if (
                                            gpu_bufs is None
                                            or base.is_cuda
                                            or not base.is_contiguous()
                                            or nb == 0
                                            or nb > gpu_buf_size
                                            or (not base_pinned and (pins is None or nb > pin_size))
                                        ):
                                            return None, base.to(dev), None
                                        slot = idx & 1
                                        gv = gpu_bufs[slot][:nb].view(base.dtype).view(base.shape)
                                        if base_pinned:
                                            keep_pinned.append(base)
                                            with torch.cuda.stream(copy_stream):
                                                if slot_used[slot]:
                                                    copy_stream.wait_event(ev_use[slot])
                                                gv.copy_(base, non_blocking=True)
                                                ev_copy[slot].record(copy_stream)
                                        else:
                                            if slot_used[slot]:
                                                ev_copy[slot].synchronize()
                                            pv = pins[slot][:nb].view(base.dtype).view(base.shape)
                                            pv.copy_(base)
                                            with torch.cuda.stream(copy_stream):
                                                if slot_used[slot]:
                                                    copy_stream.wait_event(ev_use[slot])
                                                gv.copy_(pv, non_blocking=True)
                                                ev_copy[slot].record(copy_stream)
                                        slot_used[slot] = True
                                        return slot, gv, None

                                    tensors = [base]
                                    scales = []
                                    for up, down, sc in items:
                                        tensors.append(up)
                                        tensors.append(down)
                                        scales.append(sc)
                                    if all(t.is_cuda and t.device == dev for t in tensors):
                                        return None, base, items
                                    can_pack = gpu_bufs is not None and (base_on_dev or (not base.is_cuda and base.is_contiguous()))
                                    base_pinned = can_pack and not base_on_dev and base.is_pinned()
                                    offsets = []
                                    total = 0
                                    pin_total = 0
                                    if can_pack:
                                        for ti, t in enumerate(tensors):
                                            if t.is_cuda:
                                                if t.device != dev:
                                                    can_pack = False
                                                    break
                                                offsets.append(None)
                                            else:
                                                if ti > 0 and not t.is_contiguous():
                                                    t = tensors[ti] = t.contiguous()
                                                nb = t.numel() * t.element_size()
                                                if nb == 0:
                                                    can_pack = False
                                                    break
                                                direct = (ti == 0 and base_pinned) or (ti > 0 and t.is_pinned())
                                                if direct:
                                                    offsets.append((total, nb, t.dtype, t.shape, None))
                                                else:
                                                    offsets.append((total, nb, t.dtype, t.shape, pin_total))
                                                    pin_total = align16(pin_total + nb)
                                                total = align16(total + nb)
                                    if not can_pack or total == 0 or total > gpu_buf_size or (pin_total > 0 and (pins is None or pin_total > pin_size)):
                                        g_base = base if base_on_dev else base.to(dev)
                                        g_items = [
                                            (
                                                u if (u.is_cuda and u.device == dev) else u.to(dev),
                                                d if (d.is_cuda and d.device == dev) else d.to(dev),
                                                sc,
                                            )
                                            for u, d, sc in items
                                        ]
                                        return None, g_base, g_items
                                    slot = idx & 1
                                    gbuf = gpu_bufs[slot]
                                    all_packed = pin_total == total
                                    if pin_total > 0:
                                        if slot_used[slot]:
                                            ev_copy[slot].synchronize()
                                        pbuf = pins[slot]
                                        for t, spec in zip(tensors, offsets):
                                            if spec is not None and spec[4] is not None:
                                                _, nb, tdt, sh, poff = spec
                                                pbuf[poff : poff + nb].view(tdt).view(sh).copy_(t)
                                    if not all_packed:
                                        keep_pinned.extend(t for t, spec in zip(tensors, offsets) if spec is not None and spec[4] is None)
                                    with torch.cuda.stream(copy_stream):
                                        if slot_used[slot]:
                                            copy_stream.wait_event(ev_use[slot])
                                        if all_packed:
                                            gbuf[:total].copy_(pins[slot][:total], non_blocking=True)
                                        else:
                                            for t, spec in zip(tensors, offsets):
                                                if spec is not None:
                                                    goff, nb, tdt, sh, poff = spec
                                                    if poff is None:
                                                        gbuf[goff : goff + nb].view(tdt).view(sh).copy_(t, non_blocking=True)
                                            if pin_total > 0:
                                                # Packed unpinned tensors share contiguous trailing/interleaved offsets in pbuf;
                                                # copy each into its GPU slot (or a single bulk DMA when base is the only pinned tensor).
                                                pbuf = pins[slot]
                                                contiguous_pin = all(
                                                    spec[0] - (offsets[0][0] + align16(offsets[0][1]) if offsets[0] is not None and offsets[0][4] is None else 0) == spec[4]
                                                    for spec in offsets[1:]
                                                    if spec is not None and spec[4] is not None
                                                ) and all(spec is None or spec[4] is not None for spec in offsets[1:])
                                                if contiguous_pin and offsets[0] is not None and offsets[0][4] is None:
                                                    base_end = align16(offsets[0][1])
                                                    gbuf[base_end : base_end + pin_total].copy_(pbuf[:pin_total], non_blocking=True)
                                                else:
                                                    for spec in offsets:
                                                        if spec is not None and spec[4] is not None:
                                                            goff, nb, _, _, poff = spec
                                                            gbuf[goff : goff + nb].copy_(pbuf[poff : poff + nb], non_blocking=True)
                                        ev_copy[slot].record(copy_stream)
                                    slot_used[slot] = True
                                    gviews = [
                                        t if spec is None else gbuf[spec[0] : spec[0] + spec[1]].view(spec[2]).view(spec[3])
                                        for t, spec in zip(tensors, offsets)
                                    ]
                                    g_items = [(gviews[1 + 2 * i], gviews[2 + 2 * i], scales[i]) for i in range(len(scales))]
                                    return slot, gviews[0], g_items

                                staged = stage(0, gpu_items[0][0])
                                for idx, (name, proto, dt) in enumerate(gpu_items):
                                    slot, gpu_base, gpu_lora = staged
                                    if idx + 1 < len(gpu_items):
                                        staged = stage(idx + 1, gpu_items[idx + 1][0])
                                    if gpu_base is None:
                                        put(name, proto, dt)
                                        continue
                                    if slot is not None:
                                        compute_stream.wait_event(ev_copy[slot])
                                    if gpu_lora is not None:
                                        w = source.apply_fast_lora(gpu_base, gpu_lora, dt).contiguous()
                                    else:
                                        w = source.apply_on_gpu(name, gpu_base, dt).contiguous()
                                    if slot is not None:
                                        ev_use[slot].record(compute_stream)
                                    bind_tensor(name, proto, w)
                                if copy_stream is not None and not has_device:
                                    copy_stream.synchronize()
                                    keep_pinned.clear()
                            flush()
                            while True:
                                missing = [n for n in refitter.get_missing_weights() if n not in seen_in_chunk]
                                if not missing:
                                    break
                                for name in missing:
                                    put(name)
                                    seen_in_chunk.add(name)
                                flush()
                            done.update(seen_in_chunk)
                            if has_device:
                                torch.cuda.synchronize(dev)  # TensorRT reads DEVICE weights on its own stream
                                keep_pinned.clear()
                            try:
                                ok = refitter.refit_cuda_engine()
                            finally:
                                refitter.release_refit_resources()  # otherwise the refit's scratch VRAM stays allocated for the engine's lifetime
                        finally:
                            refitter = None  # py::keep_alive pins self.engine if left in an exception traceback
                            gpu_batch.clear()
                            keep.clear()  # an exception's traceback holds this frame, and with it GBs of weights
                        if not ok:
                            raise RuntimeError("Refitting the engine failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
                        if pos < len(names):
                            gpu.promote_pool()
            finally:
                if copy_stream is not None:
                    with contextlib.suppress(Exception):
                        copy_stream.synchronize()
                if compute_stream is not None:
                    with contextlib.suppress(Exception):
                        compute_stream.synchronize()
                keep_pinned.clear()
                gpu_bufs = pins = None
        return time.perf_counter() - start

    def _set_shapes(self, profile: int, shapes: dict, stream: int):
        self._context()
        if profile != self.profile:
            self.profile, self.shapes, self._out_shapes, self.memory_bound, self.size = None, {}, {}, False, None
            if not self.context.set_optimization_profile_async(profile, stream):
                raise RuntimeError(f"Could not select optimization profile {profile}")
            self.profile = profile
        for name, shape in shapes.items():
            if self.shapes.get(name) != shape:
                self.shapes.pop(name, None)
                self._out_shapes.clear()
                self.memory_bound, self.size = False, None
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f"Input {name} {shape} is outside profile {profile}")
                self.shapes[name] = shape

    def _prepare(self, profile: int, shapes: dict, stream: int):
        self._set_shapes(profile, shapes, stream)
        if not self.memory_bound or self.bound_version != self.arena.version or self.arena.memory is None:
            memory = self.arena.get(self.activation_bytes())
            self.context.set_device_memory(memory.data_ptr(), memory.numel())
            self.memory_bound, self.bound_version = True, self.arena.version

    def activation_bytes(self) -> int:
        if self.size is None:
            size = int(self.context.update_device_memory_size_for_shapes())
            self.size_is_most = size <= 0
            self.size = size if size > 0 else int(self.engine.get_device_memory_size_for_profile_v2(self.profile))
        return self.size

    def estimate_activation_bytes(self, profile: int, shapes: dict, stream: int) -> int:
        dev = common.device()
        with common.on_device(dev), _non_default_stream(dev, stream) as run_stream:
            self._set_shapes(profile, shapes, run_stream)
            return self.activation_bytes()

    def io_bytes(self) -> dict:
        return {name: math.prod(self.context.get_tensor_shape(name)) * dtype.itemsize for name, _, dtype in self.io}

    def infer(self, feed: dict, profile: int, stream: int, into: dict = None) -> dict:
        dev = common.device()
        with common.on_device(dev):
            side = None if stream else _exec_entry(dev)[0]
            shapes = {}
            for name, dtype in self._inputs:
                t = feed[name]
                if t.dtype != dtype or t.device != dev or not t.is_contiguous():
                    t = feed[name] = t.to(device=dev, dtype=dtype).contiguous()
                    if side is not None:
                        t.record_stream(side)
                shapes[name] = tuple(t.shape)
            # Context creation and optimization-profile switches run on the execution stream; arena/output allocations
            # then run on PyTorch's current stream before the ready event so cudaMallocAsync orders them ahead of side.
            if self.context is None or profile != self.profile:
                with _non_default_stream(dev, stream) as run_stream:
                    self._set_shapes(profile, shapes, run_stream)
            else:
                self._set_shapes(profile, shapes, stream)
            if not self.memory_bound or self.bound_version != self.arena.version or self.arena.memory is None:
                memory = self.arena.get(self.activation_bytes())
                self.context.set_device_memory(memory.data_ptr(), memory.numel())
                self.memory_bound, self.bound_version = True, self.arena.version
            ctx = self.context
            for name, _ in self._inputs:
                ctx.set_tensor_address(name, feed[name].data_ptr())
            outputs = {}
            out_shapes = self._out_shapes
            for name, dtype in self._outputs:
                shape = out_shapes.get(name)
                if shape is None:
                    shape = out_shapes[name] = tuple(ctx.get_tensor_shape(name))
                given = into.get(name) if into is not None else None
                if given is not None and tuple(given.shape) == shape and given.dtype == dtype and given.device == dev and given.is_contiguous():
                    t = outputs[name] = given
                else:
                    t = outputs[name] = torch.empty(shape, dtype=dtype, device=dev)
                ctx.set_tensor_address(name, t.data_ptr())
            LOGGER.clear()
            if stream:
                if self.ran:
                    ran = ctx.execute_async_v3(stream)
                else:
                    with self._taking("as it first ran"):  # a context allocates more on its first run
                        ran = ctx.execute_async_v3(stream)
            else:
                with _non_default_stream(dev, 0) as run_stream:
                    if self.ran:
                        ran = ctx.execute_async_v3(run_stream)
                    else:
                        with self._taking("as it first ran"):
                            ran = ctx.execute_async_v3(run_stream)
            if not ran:
                raise RuntimeError("TensorRT inference failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
            if not self.ran:
                self.ran = True
                gpu.proven()
            if PROFILE:
                torch.cuda.current_stream(dev).synchronize()
                try:
                    ctx.report_to_profiler()
                except Exception:
                    pass
            return outputs

    def release_activations(self):
        self.arena.release()
        self.memory_bound = False


_exec_streams = {}  # device index -> (Stream, ready Event, done Event)


def _device_index(dev) -> int:
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _exec_entry(dev):
    index = _device_index(dev)
    entry = _exec_streams.get(index)
    if entry is None:
        entry = _exec_streams[index] = (torch.cuda.Stream(device=dev), torch.cuda.Event(), torch.cuda.Event())
    return entry


@contextlib.contextmanager
def _non_default_stream(dev, stream: int):
    # TensorRT 11 inserts host cudaStreamSynchronize() calls on multi-stream engines given the default stream (0).
    # When PyTorch is on stream 0, run TensorRT (and any TorchAllocator allocations it triggers) on a cached
    # non-blocking stream ordered against stream 0 by GPU events.
    if stream:
        yield stream
        return
    side, ready, done = _exec_entry(dev)
    current = torch.cuda.current_stream(dev)
    ready.record(current)
    side.wait_event(ready)
    try:
        with torch.cuda.stream(side):
            yield side.cuda_stream
    finally:
        done.record(side)
        current.wait_event(done)
