"""A deserialized TensorRT engine: weights refitted from PyTorch tensors, several optimization profiles in
one execution context, and activation memory from PyTorch's allocator sized for the shapes in use."""

import tensorrt as trt
import torch


_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}
_MEMORY_HINTS = ("memory", "alloc")
DEVICE = "cuda"


def torch_dtype(dtype):
    if dtype not in _DTYPES:
        raise TypeError(f"Unsupported TensorRT dtype {dtype}")
    return _DTYPES[dtype]


class Logger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.errors = []

    def log(self, severity, msg):
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

    def load(self):
        LOGGER.errors.clear()
        with open(self.plan_path, "rb") as f:
            plan = f.read()
        self.engine = runtime().deserialize_cuda_engine(plan)
        if self.engine is None:
            details = "; ".join(LOGGER.errors[-3:])
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not load {self.plan_path}"
                + (f": {details}" if details else "")
                + (". There may not be enough free VRAM." if any(h in details.lower() for h in _MEMORY_HINTS) else ". Rebuild the engine.")
            )
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io.append((name, is_input, torch_dtype(self.engine.get_tensor_dtype(name))))
        self.weight_names = list(trt.Refitter(self.engine, LOGGER).get_all_weights())
        self.context = None  # created once the weights are in
        self._reset_context_state()

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

    def refit(self, names, tensor_for):
        """Sets the weights `names` (and any TensorRT needs with them) from `tensor_for(name)`."""
        refitter = trt.Refitter(self.engine, LOGGER)
        keep = []

        def put(name):
            proto = refitter.get_weights_prototype(name)
            t = tensor_for(name).to(dtype=torch_dtype(proto.dtype)).contiguous()
            if t.numel() != proto.size:
                raise RuntimeError(f"Weight {name} has {t.numel()} values, the engine expects {proto.size}")
            keep.append(t)
            location = trt.TensorLocation.DEVICE if t.is_cuda else trt.TensorLocation.HOST
            if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), location):
                raise RuntimeError(f"TensorRT rejected weight {name}")

        LOGGER.errors.clear()
        for name in names:
            put(name)
        for name in refitter.get_missing_weights():
            put(name)
        try:
            ok = refitter.refit_cuda_engine()
        finally:
            refitter.release_refit_resources()  # else the refit's scratch memory stays allocated as long as the engine
        if not ok:
            raise RuntimeError("Refitting the engine failed" + (f": {LOGGER.errors[-1]}" if LOGGER.errors else ""))

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
