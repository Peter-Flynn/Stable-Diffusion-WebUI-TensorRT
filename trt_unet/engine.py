"""A deserialized TensorRT engine: weights refitted from PyTorch tensors (or the engine brought back as it
was serialized with them), several optimization profiles in one execution context, and activation memory
from PyTorch's allocator sized for the shapes in use."""

import ctypes
import os
import time
import weakref
from collections import deque

import tensorrt as trt
import torch

from .common import gib, log, memory

# Refitting prepares weights given in RAM for the GPU on the CPU, by default on one thread. Weights come
# straight from PyTorch's FP16 tensors, and the engine's first output is checked for NaN/Inf, so
# TensorRT's own pass over every weight is skipped.
REFITTER_SETTINGS = {"max_threads": os.cpu_count() or 1, "weights_validation": False}
_unsettable = set()
# Weights given in VRAM are prepared on the GPU instead, so a refit copies as many there as fit, leaving
# this much free for TensorRT's own refit memory. A stripped engine's first refit needs every weight at
# once, so it can't go in parts; the weights that don't fit go from RAM.
STAGING_RESERVE = 2**30


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


def staging_budget() -> int:
    """Bytes of weights a refit can copy to VRAM: what's free (PyTorch's unused cache included) but the reserve."""
    try:
        free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    except Exception:
        return 0
    return max(0, free - STAGING_RESERVE)


class _StagedRefitFailed(RuntimeError):
    pass


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


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = trt.Runtime(LOGGER)
    return _runtime


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
    def __init__(self, plan_path: str, arena: Arena = None):
        self.plan_path = plan_path
        self.arena = arena or Arena()
        self.engine = self.context = None
        self._reset_context_state()

    def _reset_context_state(self):
        self.profile = None
        self.shapes = {}
        self.memory_bound = False
        self.bound_version = None
        self.size = None  # the activation memory the shapes set need

    def twin(self):
        """Another execution context on the same engine (and weights), with shapes of its own."""
        other = Engine(self.plan_path, self.arena)
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
        self.engine = runtime().deserialize_cuda_engine(plan)
        if self.engine is None:
            said = " | ".join(LOGGER.messages) or "nothing"
            try:
                free, total = torch.cuda.mem_get_info()
                vram = f"{gib(free)} of {gib(total)} VRAM free"
            except Exception as e:
                vram = f"free VRAM unknown ({e})"
            other = other_tensorrt()
            what = self.plan_path if serialized is None else f"{self.plan_path} as kept in RAM with its weights"
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not load {what} ({memoryview(plan).nbytes} bytes; {vram}). TensorRT said: {said}"
                + (f". {other}" if other else "")
            )
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io.append((name, is_input, torch_dtype(self.engine.get_tensor_dtype(name))))
        self.weight_names = list(trt.Refitter(self.engine, LOGGER).get_all_weights())
        self.context = None  # created once the weights are in
        self._reset_context_state()
        return time.perf_counter() - start

    def _context(self):
        if self.context is None:
            try:
                self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            except Exception:
                self.context = None
            if self.context is None:
                raise RuntimeError("Could not create a TensorRT execution context (out of VRAM?)")
            self._reset_context_state()
        return self.context

    def unload(self):
        self.context = None  # before its engine
        self.engine = None
        self._reset_context_state()
        self.arena.release()

    def serialize(self):
        """The engine with the weights it holds now, still refittable: what load() takes in place of the plan
        to bring it back as it is, with nothing to refit. TensorRT's own host memory (a buffer), about as
        large as the weights."""
        config = self.engine.create_serialization_config()
        config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)  # set for a plan built without its weights
        config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)  # else the weights could never change again
        LOGGER.clear()
        serialized = self.engine.serialize_with_config(config)
        if serialized is None:
            raise RuntimeError("TensorRT could not serialize the engine" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return serialized

    def refit(self, names, tensor_for, label: str = None) -> dict:
        """Sets the weights `names` (and any TensorRT needs with them) from `tensor_for(name, device)`, from
        VRAM while there's room (see STAGING_RESERVE). Returns the seconds spent getting the weights
        ("weights") and in TensorRT ("tensorrt"), and the bytes of them given in VRAM ("staged"). With a
        `label`, the memory diagnostics are logged around it."""
        budget = staging_budget()
        if label:
            memory(f"refit {label}: before", f"{len(names)} weights, staging budget {gib(budget)}")
        try:
            seconds = self._refit(names, tensor_for, budget)
            if label:
                memory(f"refit {label}: done", f"{gib(seconds['staged'])} staged in VRAM")
            return seconds
        except _StagedRefitFailed as e:
            failure = str(e)
        # Out of the except block, so its traceback no longer holds the failed refit's copies in VRAM
        log(f"{failure}; refitting from RAM instead.")
        if label:
            memory(f"refit {label}: after the failed refit from VRAM, before emptying PyTorch's cache")
        torch.cuda.empty_cache()
        if label:
            memory(f"refit {label}: after the failed refit from VRAM")
        seconds = self._refit(names, tensor_for, 0)
        if label:
            memory(f"refit {label}: done from RAM")
        return seconds

    def _refit(self, names, tensor_for, budget: int) -> dict:
        refitter = trt.Refitter(self.engine, LOGGER)
        for setting, value in REFITTER_SETTINGS.items():
            try:
                setattr(refitter, setting, value)
            except (AttributeError, TypeError) as e:  # the Python API reference doesn't say they're settable
                if setting not in _unsettable:
                    _unsettable.add(setting)
                    log(f"Could not set the refitter's {setting}: {e}")
        keep, seconds = [], {"weights": 0.0, "tensorrt": 0.0, "staged": 0}

        def get(name, proto):
            nonlocal budget
            dtype = torch_dtype(proto.dtype)
            nbytes = proto.size * dtype.itemsize
            if 0 < nbytes <= budget:
                try:
                    t = tensor_for(name, DEVICE).to(dtype=dtype).contiguous()
                    budget -= nbytes
                    seconds["staged"] += nbytes
                    return t
                except torch.cuda.OutOfMemoryError:
                    budget = 0  # the rest from RAM
            return tensor_for(name, "cpu").to(dtype=dtype).contiguous()

        def put(name):
            start = time.perf_counter()
            proto = refitter.get_weights_prototype(name)
            t = get(name, proto)
            if t.numel() != proto.size:
                raise RuntimeError(f"Weight {name} has {t.numel()} values, the engine expects {proto.size}")
            keep.append(t)
            location = trt.TensorLocation.DEVICE if t.is_cuda else trt.TensorLocation.HOST
            given = time.perf_counter()
            seconds["weights"] += given - start
            if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), location):
                raise RuntimeError(f"TensorRT rejected weight {name}")
            seconds["tensorrt"] += time.perf_counter() - given

        LOGGER.clear()
        try:
            for name in names:
                put(name)
            for name in refitter.get_missing_weights():
                put(name)
            start = time.perf_counter()
            if seconds["staged"]:
                torch.cuda.current_stream().synchronize()  # the copies to VRAM are done before TensorRT reads them
                seconds["weights"] += time.perf_counter() - start
                start = time.perf_counter()
            # Computing LoRA weights on the GPU leaves their FP32 temporaries in PyTorch's cache, taking what
            # STAGING_RESERVE left free from TensorRT, which allocates its refit memory from the driver
            # (without this, the refit can fail in myelin's mapNewAllocation). The staged copies stay.
            torch.cuda.empty_cache()
            try:
                ok = refitter.refit_cuda_engine()
            finally:
                refitter.release_refit_resources()  # else the refit's scratch memory stays allocated as long as the engine
            seconds["tensorrt"] += time.perf_counter() - start
        except torch.cuda.OutOfMemoryError:
            # e.g. a LoRA weight computed on the GPU with the copies already there: try again with none
            if not seconds["staged"]:
                raise
            raise _StagedRefitFailed("Ran out of VRAM refitting from it") from None
        finally:
            keep.clear()  # an error's traceback keeps this frame, and would keep the weights (GBs, maybe in VRAM)
        if not ok:
            said = f": {LOGGER.errors[-1]}" if LOGGER.errors else ""
            if seconds["staged"]:
                raise _StagedRefitFailed(f"Refitting the engine from VRAM failed{said}")
            raise RuntimeError(f"Refitting the engine failed{said}")
        if seconds["staged"]:
            torch.cuda.empty_cache()  # the copies' VRAM back to the driver, for TensorRT and the other models
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
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        return outputs

    def release_activations(self):
        """Releases the arena, for every engine sharing it."""
        self.arena.release()
        self.memory_bound = False
