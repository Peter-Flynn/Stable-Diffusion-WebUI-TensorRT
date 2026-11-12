"""A deserialized TensorRT engine: weights refitted from PyTorch tensors in RAM (or the engine brought back
as it was serialized with them), several optimization profiles in one execution context, and activation
memory from PyTorch's allocator sized for the shapes in use.

The engine's weights are the only copy of them in VRAM: a refit gives TensorRT the weights where PyTorch
keeps them, in RAM, and TensorRT copies them into its own memory. All of TensorRT's memory comes from
PyTorch's allocator (gpu.py), under the engine's owner; if an engine can't load or refit that way before
any has run that way, TensorRT gets its own allocator back (_deserialize, own_allocator)."""

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
from .common import gib, log, memory

# Refitting prepares weights given in RAM for the GPU on the CPU, by default on one thread. Weights come
# straight from PyTorch's FP16 tensors, and the engine's first output is checked for NaN/Inf, so
# TensorRT's own pass over every weight is skipped.
REFITTER_SETTINGS = {"max_threads": os.cpu_count() or 1, "weights_validation": False}
_unsettable = set()
# TRT_EXTENSION_PROFILE=1: each engine's layers timed by TensorRT's own profiler (every run waits for the GPU,
# so everything is slower), logged after each sampling pass, decode and encode (profile_report)
PROFILE = os.environ.get("TRT_EXTENSION_PROFILE", "").lower() in ("1", "true", "yes")
_layer_times = {}  # plan file -> {layer: milliseconds, added up since the last report}


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
    """Prints TensorRT's errors; keeps everything but its verbose messages to explain a failure."""

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
        self.messages.append(f"{severity.name}: {msg}")
        if int(severity) <= int(trt.ILogger.Severity.ERROR):
            self.errors.append(msg)
            print(f"[TensorRT] ERROR: {msg}", flush=True)


LOGGER = Logger()
_runtime = None
_set_aside = []  # a runtime with PyTorch's allocator that an engine failed to load on (_deserialize), kept alive


class _LayerTimes(trt.IProfiler):
    """A context's layers' times on the GPU, added up under its plan file (TRT_EXTENSION_PROFILE)."""

    def __init__(self, plan_path: str):
        trt.IProfiler.__init__(self)
        self.times = _layer_times.setdefault(plan_path, {})

    def report_layer_time(self, layer_name, ms):
        self.times[layer_name] = self.times.get(layer_name, 0.0) + ms


def profile_report(where: str, top: int = 12):
    """Logs, per plan, the layers' time since the last report, the longest first, and forgets it."""
    if not PROFILE:
        return
    for plan, times in sorted(_layer_times.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(times.values())
        if not total:
            continue
        longest = sorted(times.items(), key=lambda kv: -kv[1])[:top]
        log(f"profile | {where} | {os.path.basename(plan)}: {total:.1f} ms in {len(times)} layers; the longest: "
            + "; ".join(f"{name} {ms:.1f} ms" for name, ms in longest))
        times.clear()


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = trt.Runtime(LOGGER)
        gpu.attach(_runtime)  # before anything is deserialized
    return _runtime


def _deserialize(plan, owner):
    """The engine in `plan`, or None if TensorRT can't load it. One that fails with PyTorch's allocator before
    any engine ran with it, and not for want of VRAM, is tried on a runtime with TensorRT's own allocator:
    if it loads there, that runtime is used from now on (gpu.retire); if not, the allocator wasn't why."""
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
    return engine


def own_allocator(failures_before: int, owner) -> bool:
    """After `owner`'s engine loaded but failed past that (its refit, its execution context) and was
    unloaded: if PyTorch's allocator could be why (gpu.suspect: none has run with it yet, it didn't run out,
    nothing else is loaded through it), TensorRT allocates its own memory from now on. Returns whether it
    changed, for the caller to load the engine again."""
    global _runtime
    if not gpu.suspect(failures_before, owner):
        return False
    _set_aside.append(_runtime)
    _runtime = trt.Runtime(LOGGER)
    gpu.retire()
    return True


@atexit.register
def _unload_all():
    """Engines free their memory through PyTorch's allocator (gpu.py): before the interpreter takes PyTorch
    and the allocator apart."""
    for engine in list(Engine.instances):
        try:
            engine.unload()
        except Exception:
            pass


def other_tensorrt(maps="/proc/self/maps") -> str:
    """A TensorRT of another version that's visible to the whole process (Linux), or "". TensorRT's bindings
    use it instead of their own if it was there before they were imported (preload.py imports them first)."""
    if os.name == "nt":  # Windows binds each DLL to the libraries it names
        return ""
    major = int(trt.__version__.split(".")[0])
    try:
        version = ctypes.CDLL(None).getInferLibVersion()
    except (AttributeError, OSError):
        return ""  # none visible to the whole process
    if version // 10000 == major:
        return ""
    try:
        with open(maps) as f:
            paths = sorted({line.split()[-1] for line in f if "/libnvinfer.so." in line and not line.rstrip().endswith(f"libnvinfer.so.{major}")})
    except OSError:
        paths = []
    where = f" ({', '.join(paths)})" if paths else ""
    return (
        f"TensorRT {version // 10000}.{version // 100 % 100}{where} is loaded for the whole process, likely by another "
        f"extension, and TensorRT's bindings may be calling it instead of TensorRT {trt.__version__}. "
        "This extension's preload.py imports TensorRT before other extensions load to prevent that; "
        "something loaded the other TensorRT even earlier."
    )


class Arena:
    """Activation memory from PyTorch's allocator, shared by engines that never run at the same time (an
    engine and its shallow partner): it only grows until it's released, and each engine binds it again
    when it was reallocated."""

    instances = weakref.WeakSet()  # every arena, for the memory diagnostics (common.memory)

    def __init__(self):
        self.memory, self.version = None, 0
        Arena.instances.add(self)

    def get(self, size: int):
        if self.memory is None or self.memory.numel() < size:
            self.release()  # a new version even if the allocation fails: nothing may stay bound to the old one
            self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=DEVICE)
            self.version += 1
        return self.memory

    def release(self):
        if self.memory is not None:
            self.memory = None
            self.version += 1


class Engine:
    """An engine and its execution context. Its TensorRT memory is `owner`'s (gpu.py); twins share both."""

    instances = weakref.WeakSet()  # every engine, to unload before the interpreter exits

    def __init__(self, plan_path: str, arena: Arena = None, owner=None):
        self.plan_path = plan_path
        self.arena = arena or Arena()
        self.owner = owner
        self.engine = self.context = None
        self.extra = 0  # what TensorRT took for it after it loaded (its context, its first run), by the ledger
        self._reset_context_state()
        Engine.instances.add(self)

    def _reset_context_state(self):
        self.profile = None
        self.shapes = {}
        self.memory_bound = False
        self.bound_version = None
        self.size = None  # the activation memory the shapes set need
        self.ran = False  # whether the context has run: its first run is watched (infer)

    def twin(self):
        """Another execution context on the same engine (and weights), with shapes of its own."""
        other = Engine(self.plan_path, self.arena, self.owner)
        other.engine, other.io, other.weight_names = self.engine, self.io, self.weight_names
        return other

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load(self, serialized=None) -> float:
        """Loads the plan, which holds no weights (refit puts them in), or `serialized`: the engine as
        serialize() gave it, with the weights it had then. Returns the seconds it took."""
        start = time.perf_counter()
        LOGGER.clear()
        plan = serialized
        if plan is None:
            with open(self.plan_path, "rb") as f:
                plan = f.read()
        self.engine = _deserialize(plan, self.owner)
        if self.engine is None:
            said = " | ".join(LOGGER.messages) or "nothing"
            other = "" if "OutOfMemory" in said else other_tensorrt()  # (not why it ran out)
            what = self.plan_path if serialized is None else f"{self.plan_path} as kept in RAM with its weights"
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not load {what} ({memoryview(plan).nbytes} bytes). TensorRT said: {said}"
                + (f". {other}" if other else "")
            )
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io.append((name, is_input, torch_dtype(self.engine.get_tensor_dtype(name))))
        with gpu.owner(self.owner):
            self.weight_names = list(trt.Refitter(self.engine, LOGGER).get_all_weights())
        self.context = None  # created once the weights are in
        self._reset_context_state()
        return time.perf_counter() - start

    @contextlib.contextmanager
    def _taking(self, what: str = None):
        """What TensorRT allocates in here, after the engine loaded, is its owner's too (gpu.py's ledger
        says how much: nothing, without PyTorch's allocator). Said if it's `what` and a lot."""
        if gpu.counting(self.owner):  # a load under way counts it
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
                    log(f"{os.path.basename(self.plan_path)} took {gib(more)} more of VRAM {what}.")

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
                    self.context.enqueue_emits_profile = False  # reported when asked, after the GPU is done
                except Exception as e:
                    log(f"Could not profile {os.path.basename(self.plan_path)}: {e}")
            self._reset_context_state()
        return self.context

    def unload(self):
        with gpu.owner(self.owner):
            self.context = None  # before its engine
            self.engine = None
        self._reset_context_state()
        self.arena.release()
        if self.extra:
            gpu.unloaded(self.owner, self.extra)
            self.extra = 0

    def stripped_bytes(self):
        """The bytes of weights the plan was built without, which a refit puts in (TensorRT's own figure), or
        None if it doesn't say."""
        try:
            n = int(self.engine.get_engine_stat(trt.EngineStat.STRIPPED_WEIGHTS_SIZE))
            return n if n > 0 else None
        except Exception:
            return None

    def serialize(self):
        """The engine with the weights it holds now, still refittable: what load() takes in place of the plan
        to bring it back as it is, with nothing to refit. TensorRT's own host memory (a buffer), about as
        large as the weights."""
        config = self.engine.create_serialization_config()
        config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)  # set for a plan built without its weights
        config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)  # else the weights could never change again
        LOGGER.clear()
        with gpu.owner(self.owner):
            serialized = self.engine.serialize_with_config(config)
        if serialized is None:
            raise RuntimeError("TensorRT could not serialize the engine" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return serialized

    def refit(self, names, tensor_for, label: str = None) -> dict:
        """Sets the weights `names` (and any TensorRT needs with them) from `tensor_for(name, "cpu")`: the
        weights in RAM, which TensorRT copies into its own memory, the only copy of them in VRAM. Returns the
        seconds spent getting the weights ("weights") and in TensorRT ("tensorrt"). With a `label`, the
        memory diagnostics are logged around it."""
        if label:
            memory(f"refit {label}: before", f"{len(names)} weights")
        refitter = trt.Refitter(self.engine, LOGGER)
        for setting, value in REFITTER_SETTINGS.items():
            try:
                setattr(refitter, setting, value)
            except (AttributeError, TypeError) as e:  # the Python API reference doesn't say they're settable
                if setting not in _unsettable:
                    _unsettable.add(setting)
                    log(f"Could not set the refitter's {setting}: {e}")
        keep, seconds = [], {"weights": 0.0, "tensorrt": 0.0}

        def put(name):
            start = time.perf_counter()
            proto = refitter.get_weights_prototype(name)
            t = tensor_for(name, "cpu").to(dtype=torch_dtype(proto.dtype)).contiguous()
            if t.numel() != proto.size:
                raise RuntimeError(f"Weight {name} has {t.numel()} values, the engine expects {proto.size}")
            keep.append(t)
            given = time.perf_counter()
            seconds["weights"] += given - start
            if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), trt.TensorLocation.HOST):
                raise RuntimeError(f"TensorRT rejected weight {name}")
            seconds["tensorrt"] += time.perf_counter() - given

        LOGGER.clear()
        try:
            for name in names:
                put(name)
            for name in refitter.get_missing_weights():
                put(name)
            start = time.perf_counter()
            with self._taking():
                try:
                    ok = refitter.refit_cuda_engine()
                finally:
                    refitter.release_refit_resources()  # else the refit's scratch memory stays allocated as long as the engine
            seconds["tensorrt"] += time.perf_counter() - start
        finally:
            keep.clear()  # an error's traceback keeps this frame, and would keep the weights (GBs)
        if not ok:
            raise RuntimeError("Refitting the engine failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        if label:
            memory(f"refit {label}: done")
        return seconds

    def _set_shapes(self, profile: int, shapes: dict, stream: int):
        """Selects the profile and input shapes on the context. What's recorded is only what TensorRT
        accepted, so a failure here makes the next call set everything again."""
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
        """For the profile and input shapes set on the context (asked of TensorRT once per shapes set)."""
        if self.size is None:
            size = int(self.context.update_device_memory_size_for_shapes())  # 0 if it can't tell
            self.size = size if size > 0 else int(self.engine.get_device_memory_size_for_profile_v2(self.profile))
        return self.size

    def estimate_activation_bytes(self, profile: int, shapes: dict, stream: int) -> int:
        self._set_shapes(profile, shapes, stream)
        return self.activation_bytes()

    def io_bytes(self) -> dict:
        """{input or output: its bytes} at the shapes set on the context (TensorRT's output shapes)."""
        return {name: math.prod(self.context.get_tensor_shape(name)) * torch.tensor([], dtype=dtype).element_size() for name, _, dtype in self.io}

    def infer(self, feed: dict, profile: int, stream: int, into: dict = None) -> dict:
        """Runs on the inputs in `feed` that the engine takes (e.g. SDXL's shallow UNet takes no prompt);
        outputs go into the tensors in `into` that have their shape and type, else new ones."""
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
        if self.ran:
            ran = self.context.execute_async_v3(stream)
        else:  # what TensorRT takes only as a context first runs is counted too
            with self._taking("as it first ran"):
                ran = self.context.execute_async_v3(stream)
        if not ran:
            raise RuntimeError("TensorRT inference failed")
        if not self.ran:
            self.ran = True
            gpu.proven()  # PyTorch's allocator works with these bindings
        if PROFILE:
            torch.cuda.current_stream().synchronize()
            try:
                self.context.report_to_profiler()
            except Exception:
                pass
        return outputs

    def release_activations(self):
        """Releases the arena, for every engine sharing it."""
        self.arena.release()
        self.memory_bound = False
