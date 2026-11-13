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

from . import gpu
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
DEVICE = "cuda"


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
    with gpu.owner(owner):
        engine = runtime().deserialize_cuda_engine(plan)
    if engine is not None or not gpu.suspect(before, owner):
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
    if not gpu.suspect(failures_before, owner):
        return False
    _set_aside.append(_runtime)
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
            from . import common

            self.release()  # bumps the version even if the allocation below fails, so nothing stays bound to freed memory
            try:
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=DEVICE)
            except torch.cuda.OutOfMemoryError:
                if not common.pooled():
                    raise
                common.trim()  # PyTorch's freed blocks may be unusable for this size until returned to the driver
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=DEVICE)
            self.version += 1
        return self.memory

    def release(self):
        if self.memory is not None:
            self.memory = None
            self.version += 1


class Engine:
    instances = weakref.WeakSet()

    def __init__(self, plan_path: str, arena: Arena = None, owner=None):
        self.plan_path = plan_path
        self.arena = arena or Arena()
        self.owner = owner
        self.engine = self.context = None
        self.extra = 0  # VRAM TensorRT took after load (context creation, first run), added to the owner's gpu._held
        self._reset_context_state()
        Engine.instances.add(self)

    def _reset_context_state(self):
        self.profile = None
        self.shapes = {}
        self.memory_bound = False
        self.bound_version = None
        self.size = None
        self.size_is_most = False  # TensorRT gave no figure for these shapes: `size` is the profile's maximum
        self.ran = False

    def twin(self):  # another execution context on the same engine and weights, so callers alternating shapes don't thrash one
        other = Engine(self.plan_path, self.arena, self.owner)
        other.engine, other.io, other.weight_names = self.engine, self.io, self.weight_names
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
        self.engine = _deserialize(plan, self.owner)
        if self.engine is None:
            said = LOGGER.errors[-1] if LOGGER.errors else "no reason given"
            other = "" if any("OutOfMemory" in m for m in LOGGER.messages) else other_tensorrt()
            what = os.path.basename(self.plan_path) + ("" if serialized is None else " as kept in RAM")
            raise RuntimeError(f"TensorRT {trt.__version__} couldn't load {what}: {said}" + (f" {other}" if other else ""))
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io.append((name, is_input, torch_dtype(self.engine.get_tensor_dtype(name))))
        with gpu.owner(self.owner):
            self.weight_names = list(trt.Refitter(self.engine, LOGGER).get_all_weights())
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
                with self._taking():
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

    def unload(self):
        with gpu.owner(self.owner):
            self.context = None  # before the engine it belongs to
            self.engine = None
        self._reset_context_state()
        self.arena.release()
        if self.extra:
            gpu.unloaded(self.owner, self.extra)
            self.extra = 0

    def serialize(self):
        config = self.engine.create_serialization_config()
        config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)  # inherited from the weight-stripped plan
        config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)  # without it, the reloaded engine can never be refitted again
        LOGGER.clear()
        with gpu.owner(self.owner):
            serialized = self.engine.serialize_with_config(config)
        if serialized is None:
            raise RuntimeError("TensorRT could not serialize the engine" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return serialized

    def refit(self, names, tensor_for) -> float:
        refitter = trt.Refitter(self.engine, LOGGER)
        for setting, value in REFITTER_SETTINGS.items():
            try:
                setattr(refitter, setting, value)
            except (AttributeError, TypeError) as e:
                if setting not in _unsettable:
                    _unsettable.add(setting)
                    logger.debug(f"Couldn't set the refitter's {setting}: {e}")
        keep, start = [], time.perf_counter()

        def put(name):
            proto = refitter.get_weights_prototype(name)
            t = tensor_for(name, "cpu").to(dtype=torch_dtype(proto.dtype)).contiguous()
            if t.numel() != proto.size:
                raise RuntimeError(f"Weight {name} has {t.numel()} values, the engine expects {proto.size}")
            keep.append(t)
            if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), trt.TensorLocation.HOST):
                raise RuntimeError(f"TensorRT rejected weight {name}")

        LOGGER.clear()
        try:
            for name in names:
                put(name)
            for name in refitter.get_missing_weights():
                put(name)
            with self._taking():
                try:
                    ok = refitter.refit_cuda_engine()
                finally:
                    refitter.release_refit_resources()  # otherwise the refit's scratch VRAM stays allocated for the engine's lifetime
        finally:
            keep.clear()  # an exception's traceback holds this frame, and with it GBs of weights
        if not ok:
            raise RuntimeError("Refitting the engine failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return time.perf_counter() - start

    def _set_shapes(self, profile: int, shapes: dict, stream: int):
        self._context()
        if profile != self.profile:
            self.profile, self.shapes, self.memory_bound, self.size = None, {}, False, None
            if not self.context.set_optimization_profile_async(profile, stream):
                raise RuntimeError(f"Could not select optimization profile {profile}")
            self.profile = profile
        for name, shape in shapes.items():
            if self.shapes.get(name) != shape:
                self.shapes.pop(name, None)
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
        self._set_shapes(profile, shapes, stream)
        return self.activation_bytes()

    def io_bytes(self) -> dict:
        return {name: math.prod(self.context.get_tensor_shape(name)) * torch.tensor([], dtype=dtype).element_size() for name, _, dtype in self.io}

    def infer(self, feed: dict, profile: int, stream: int, into: dict = None) -> dict:
        self._prepare(profile, {name: tuple(feed[name].shape) for name, is_input, _ in self.io if is_input}, stream)
        outputs = {}
        for name, is_input, dtype in self.io:
            if is_input:
                t = feed[name]
                if t.dtype != dtype or t.device.type != torch.device(DEVICE).type or not t.is_contiguous():
                    t = feed[name] = t.to(device=DEVICE, dtype=dtype).contiguous()
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                given = (into or {}).get(name)
                if given is not None and tuple(given.shape) == shape and given.dtype == dtype and given.is_contiguous():
                    t = outputs[name] = given
                else:
                    t = outputs[name] = torch.empty(shape, dtype=dtype, device=DEVICE)
            self.context.set_tensor_address(name, t.data_ptr())
        LOGGER.clear()
        if self.ran:
            ran = self.context.execute_async_v3(stream)
        else:
            with self._taking("as it first ran"):  # a context allocates more on its first run
                ran = self.context.execute_async_v3(stream)
        if not ran:
            raise RuntimeError("TensorRT inference failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        if not self.ran:
            self.ran = True
            gpu.proven()
        if PROFILE:
            torch.cuda.current_stream().synchronize()
            try:
                self.context.report_to_profiler()
            except Exception:
                pass
        return outputs

    def release_activations(self):
        self.arena.release()
        self.memory_bound = False
