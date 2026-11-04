"""A deserialized TensorRT engine: weights refitted from PyTorch tensors, several optimization profiles in
one execution context, and activation memory from PyTorch's allocator sized for the shapes in use."""

import ctypes
import os
import time
from collections import deque

import tensorrt as trt
import torch

from .common import gib, log

# Refitting prepares the weights for the GPU on the CPU, by default on one thread. Weights come straight
# from PyTorch's FP16 tensors, and the engine's first output is checked for NaN/Inf, so TensorRT's own
# pass over every weight is skipped.
REFITTER_SETTINGS = {"max_threads": os.cpu_count() or 1, "weights_validation": False}
_unsettable = set()


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


class Engine:
    def __init__(self, plan_path: str):
        self.plan_path = plan_path
        self.engine = self.context = None
        self._reset_context_state()

    def _reset_context_state(self):
        self.profile = None
        self.shapes = {}
        self.memory = None
        self.memory_bound = False

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load(self) -> float:
        """Returns the seconds it took."""
        start = time.perf_counter()
        LOGGER.clear()
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
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not load {self.plan_path} ({len(plan)} bytes; {vram}). TensorRT said: {said}"
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

    def refit(self, names, tensor_for) -> dict:
        """Sets the weights `names` (and any TensorRT needs with them) from `tensor_for(name)`. Returns the
        seconds spent getting the weights ("weights") and in TensorRT ("tensorrt")."""
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
            t = tensor_for(name).to(dtype=torch_dtype(proto.dtype)).contiguous()
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
        for name in names:
            put(name)
        for name in refitter.get_missing_weights():
            put(name)
        start = time.perf_counter()
        try:
            ok = refitter.refit_cuda_engine()
        finally:
            refitter.release_refit_resources()  # else the refit's scratch memory stays allocated as long as the engine
        seconds["tensorrt"] += time.perf_counter() - start
        if not ok:
            raise RuntimeError("Refitting the engine failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))
        return seconds

    def _set_shapes(self, profile: int, shapes: dict, stream: int):
        """Selects the profile and input shapes on the context. What's recorded is only what TensorRT
        accepted, so a failure here makes the next call set everything again."""
        self._context()
        if profile != self.profile:
            self.profile, self.shapes, self.memory_bound = None, {}, False
            if not self.context.set_optimization_profile_async(profile, stream):
                raise RuntimeError(f"Could not select optimization profile {profile}")
            self.profile = profile
        for name, shape in shapes.items():
            if self.shapes.get(name) != shape:
                self.shapes.pop(name, None)
                self.memory_bound = False
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f"Input {name} {shape} is outside profile {profile}")
                self.shapes[name] = shape

    def _prepare(self, profile: int, shapes: dict, stream: int):
        self._set_shapes(profile, shapes, stream)
        if not self.memory_bound:
            size = self.activation_bytes()
            if self.memory is None or self.memory.numel() < size or self.memory.numel() > 2 * size:
                self.memory = None
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=DEVICE)
            self.context.set_device_memory(self.memory.data_ptr(), self.memory.numel())
            self.memory_bound = True

    def activation_bytes(self) -> int:
        """For the profile and input shapes set on the context."""
        size = int(self.context.update_device_memory_size_for_shapes())  # 0 if it can't tell
        return size if size > 0 else int(self.engine.get_device_memory_size_for_profile_v2(self.profile))

    def estimate_activation_bytes(self, profile: int, shapes: dict, stream: int) -> int:
        self._set_shapes(profile, shapes, stream)
        return self.activation_bytes()

    def infer(self, feed: dict, profile: int, stream: int) -> dict:
        self._prepare(profile, {name: tuple(t.shape) for name, t in feed.items()}, stream)
        outputs = {}
        for name, is_input, dtype in self.io:
            if is_input:
                t = feed[name]
                if t.dtype != dtype or t.device.type != torch.device(DEVICE).type or not t.is_contiguous():
                    t = feed[name] = t.to(device=DEVICE, dtype=dtype).contiguous()
            else:
                t = outputs[name] = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=dtype, device=DEVICE)
            self.context.set_tensor_address(name, t.data_ptr())
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        return outputs

    def release_activations(self):
        self.memory = None
        self.memory_bound = False
