#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import mmap
import os

import tensorrt as trt
import torch
from tqdm import tqdm

# TensorRT 11 removed the weakly typed precision flags: networks are strongly typed and take their
# precision from the ONNX graph (exported under autocast).
STRONGLY_TYPED = not hasattr(trt.BuilderFlag, "FP16")

_TRT_TO_TORCH = {
    getattr(trt.DataType, t): getattr(torch, d)
    for t, d in (
        ("FLOAT", "float32"),
        ("HALF", "float16"),
        ("BF16", "bfloat16"),
        ("INT8", "int8"),
        ("UINT8", "uint8"),
        ("INT32", "int32"),
        ("INT64", "int64"),
        ("BOOL", "bool"),
        ("FP8", "float8_e4m3fn"),
    )
    if hasattr(trt.DataType, t) and hasattr(torch, d)
}


def _log(message):
    print(f"[TensorRT] {message}", flush=True)


_MEMORY_HINTS = ("insufficient memory", "memory is insufficient", "could not be allocated", "out of memory")


class TRTLogger(trt.ILogger):
    """Prints errors (and build warnings once each), keeps the current operation's errors for
    exception messages, and counts out-of-memory messages, which TensorRT only reports quietly."""

    def __init__(self):
        trt.ILogger.__init__(self)
        self.begin()

    def begin(self, print_warnings=False):
        self.print_warnings = print_warnings
        self.errors = []
        self.memory_messages = 0
        self.first_memory_message = None
        self._printed = set()

    def log(self, severity, msg):
        level = int(severity)
        if level > int(trt.ILogger.Severity.INFO):
            return
        if any(hint in msg.lower() for hint in _MEMORY_HINTS):
            self.memory_messages += 1
            self.first_memory_message = self.first_memory_message or msg
            if level > int(trt.ILogger.Severity.ERROR):
                return
        if level <= int(trt.ILogger.Severity.ERROR):
            self.errors.append(msg)
            print(f"[TensorRT] ERROR: {msg}", flush=True)
        elif level == int(trt.ILogger.Severity.WARNING) and self.print_warnings and msg not in self._printed:
            self._printed.add(msg)
            print(f"[TensorRT] Warning: {msg}", flush=True)


TRT_LOGGER = TRTLogger()


def _torch_dtype(trt_dtype):
    if trt_dtype not in _TRT_TO_TORCH:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return _TRT_TO_TORCH[trt_dtype]


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(memoryview(data))  # no copy: SDXL engines are ~5 GB
    os.replace(tmp_path, path)


class ProgressMonitor(trt.IProgressMonitor):
    """Console: one tqdm bar per build phase, nested up to `max_depth`. UI: `callback(fraction,
    description)` driven by the top-level phase and its active sub-phase."""

    def __init__(self, callback=None, max_depth=3):
        trt.IProgressMonitor.__init__(self)
        self.callback = callback
        self.max_depth = max_depth
        self._phases = {}
        self._top = None
        self._fraction = 0.0
        self._cancel = False

    def _close(self, name):
        phase = self._phases.pop(name, None)
        if phase and phase["tq"] is not None:
            phase["tq"].close()

    def _set(self, name, n):
        phase = self._phases.get(name)
        if phase is None:
            return
        n = max(0, min(int(n), phase["total"]))  # TensorRT may overshoot
        if phase["tq"] is not None:
            phase["tq"].update(n - phase["tq"].n)
        phase["n"] = n
        self._report()

    def _report(self):
        if self.callback is None or self._top not in self._phases:
            return
        top = self._phases[self._top]
        fraction, desc = top["n"], self._top
        child = next((p for p in self._phases.values() if p["parent"] == self._top), None)
        if child is not None:
            fraction += child["n"] / child["total"]
            desc = f"{self._top}: {max(self._phases.items(), key=lambda kv: kv[1]['depth'])[0]}"
        # The estimate dips between sub-phases; never move the bar backwards.
        self._fraction = max(self._fraction, min(fraction / top["total"], 1.0))
        try:
            self.callback(self._fraction, desc)
        except Exception:
            pass

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            self._close(phase_name)  # phase names repeat
            if parent_phase is None:
                depth, self._top, self._fraction = 0, phase_name, 0.0
            else:
                depth = self._phases.get(parent_phase, {}).get("depth", self.max_depth) + 1
            tq = None
            if depth < self.max_depth:
                tq = tqdm(total=num_steps, desc=phase_name, leave=depth == 0, position=depth, dynamic_ncols=True)
            self._phases[phase_name] = {"tq": tq, "depth": depth, "parent": parent_phase, "total": max(int(num_steps), 1), "n": 0}
            self._report()
        except KeyboardInterrupt:
            self._cancel = True  # only step_complete can cancel the build

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._phases:
                self._set(phase_name, self._phases[phase_name]["total"])
            self._close(phase_name)
        except KeyboardInterrupt:
            self._cancel = True

    def step_complete(self, phase_name, step):
        try:
            self._set(phase_name, step)
            return not self._cancel
        except KeyboardInterrupt:
            return False


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.build_memory_messages = 0
        self._clear_state()

    def _clear_state(self):
        self.io_tensors = []  # (name, is_input, torch dtype)
        self._staging = {}
        self._input_shapes = {}
        self._device_memory = None
        self._user_managed_memory = False

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass

    def reset(self, engine_path=None):
        self.context = None  # must go before its engine
        self.engine = None
        self._clear_state()
        if engine_path is not None:
            self.engine_path = engine_path

    def build(self, onnx_path, fp16, input_profiles, refit=True, timing_cache=None, optimization_level=None, progress_callback=None):
        print(f"[TensorRT] Building {self.engine_path}", flush=True)
        TRT_LOGGER.begin(print_warnings=True)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if STRONGLY_TYPED else 0)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        if hasattr(trt.OnnxParserFlag, "NATIVE_INSTANCENORM"):
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            _log(f"TensorRT could not parse {onnx_path}:\n{errors}")
            return False

        config = builder.create_builder_config()
        config.progress_monitor = ProgressMonitor(progress_callback)
        if STRONGLY_TYPED:
            print(f"[TensorRT] TensorRT {trt.__version__}: strongly typed build.", flush=True)
        elif fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if optimization_level is not None:
            config.builder_optimization_level = int(optimization_level)
        if refit:
            if hasattr(trt.BuilderFlag, "REFIT"):
                config.set_flag(trt.BuilderFlag.REFIT)
            else:
                _log("This TensorRT version can't build refittable engines; LoRAs won't work with this engine.")
        if timing_cache:
            try:
                data = b""
                if os.path.isfile(timing_cache):
                    with open(timing_cache, "rb") as f:
                        data = f.read()
                cache = config.create_timing_cache(data)
            except Exception as e:
                _log(f"Ignoring unreadable timing cache {timing_cache}: {e}")
                cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch=True)
        for shapes in input_profiles:
            profile = builder.create_optimization_profile()
            for name, (lo, opt, hi) in shapes.items():
                profile.set_shape(name, min=lo, opt=opt, max=hi)
            config.add_optimization_profile(profile)

        try:
            serialized = builder.build_serialized_network(network, config)
        except Exception as e:
            _log(f"Engine build failed: {e}")
            serialized = None
        finally:
            TRT_LOGGER.print_warnings = False
            self.build_memory_messages = TRT_LOGGER.memory_messages

        if self.build_memory_messages:
            _log(
                f"TensorRT reported {self.build_memory_messages} memory allocation failures while building "
                f'(e.g. "{TRT_LOGGER.first_memory_message}"). Kernels that didn\'t fit were skipped, so the engine '
                "may be slower than it could be. Close other GPU programs or narrow the profile and rebuild."
            )
        if serialized is None:
            _log("Engine build failed: TensorRT returned no engine (see errors above).")
            return False
        try:
            _atomic_write(self.engine_path, serialized)
        except Exception as e:
            _log(f"Could not save the engine: {e}")
            return False
        if timing_cache:
            try:
                _atomic_write(timing_cache, config.get_timing_cache().serialize())
            except Exception as e:
                _log(f"Could not save the timing cache {timing_cache}: {e}")
        return True

    def load(self):
        if self.runtime is None:
            self.runtime = trt.Runtime(TRT_LOGGER)
        TRT_LOGGER.begin()
        try:
            # mmap instead of reading into bytes: saves an engine-sized chunk of RAM per load.
            with open(self.engine_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
                self.engine = self.runtime.deserialize_cuda_engine(data)
        except Exception as e:
            TRT_LOGGER.errors.append(str(e))
            self.engine = None
        if self.engine is None:
            raise RuntimeError(self._load_failure_message())
        self.io_tensors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io_tensors.append((name, is_input, _torch_dtype(self.engine.get_tensor_dtype(name))))

    def _load_failure_message(self):
        details = "\n".join(TRT_LOGGER.errors[-5:])
        text = details.lower()
        if TRT_LOGGER.memory_messages or "memory" in text or "alloc" in text:
            reason = "There isn't enough free GPU memory. Free VRAM (or use a smaller profile) and try again."
        elif any(w in text for w in ("version", "serializ", "magic", "compatib", "platform", "plan")):
            reason = (
                f"It was built with a different TensorRT version or GPU (installed: TensorRT {trt.__version__}). "
                "Rebuild it from the TensorRT tab (Advanced Settings -> Force Rebuild)."
            )
        else:
            reason = "It was built with a different TensorRT version / GPU (rebuild it), or GPU memory ran out."
        message = f"TensorRT could not load {os.path.basename(self.engine_path)}. {reason}"
        return f"{message}\nTensorRT said: {details}" if details else message

    def activate(self):
        """Create the execution context. Its activation memory comes from PyTorch's allocator, sized
        for the shapes in use rather than the profile maximum, so it's shared with the WebUI."""
        self.context = None
        self._user_managed_memory = False
        self._device_memory = None
        self._input_shapes = {}
        strategy = getattr(trt, "ExecutionContextAllocationStrategy", None)
        if strategy is not None and hasattr(trt.IExecutionContext, "set_device_memory"):
            try:
                self.context = self.engine.create_execution_context(strategy.USER_MANAGED)
                self._user_managed_memory = self.context is not None
            except Exception:
                self.context = None
        if self.context is None:
            self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create a TensorRT execution context (out of GPU memory?)")

    def _set_input_shapes(self, feed_dict):
        changed = False
        for name, is_input, _ in self.io_tensors:
            if not is_input or name not in feed_dict:
                continue
            shape = tuple(feed_dict[name].shape)
            if self._input_shapes.get(name) == shape:
                continue
            if not self.context.set_input_shape(name, shape):
                raise ValueError(f"Input '{name}' with shape {shape} is outside the engine's profile.\n{self}")
            self._input_shapes[name] = shape
            changed = True
        if changed or (self._user_managed_memory and self._device_memory is None):
            self._update_device_memory()

    def _update_device_memory(self):
        if not self._user_managed_memory:
            return
        size = 0
        if hasattr(self.context, "update_device_memory_size_for_shapes"):
            try:
                size = int(self.context.update_device_memory_size_for_shapes())
            except Exception:
                size = 0
        if size <= 0:
            size = int(getattr(self.engine, "device_memory_size_v2", 0) or self.engine.device_memory_size)
        size = max(size, 1)
        current = 0 if self._device_memory is None else self._device_memory.numel()
        # Shrink too, so one large pass (e.g. Hires. fix) doesn't pin its peak.
        if current < size or current > 2 * size:
            self._device_memory = None
            self._device_memory = torch.empty(size, dtype=torch.uint8, device="cuda")
        self.context.set_device_memory(self._device_memory.data_ptr(), self._device_memory.numel())

    def release_activation_memory(self):
        """Return activation memory and staging buffers to PyTorch (e.g. for the VAE decode);
        they're reallocated on the next inference."""
        if self._user_managed_memory:
            self._device_memory = None
            self._input_shapes = {}  # forces set_input_shape + set_device_memory next time
        self._staging = {}

    def infer(self, feed_dict, stream):
        """Run on `stream`; returns freshly allocated outputs. Inputs already on the GPU in the
        engine's dtype are bound directly, others are copied into staging buffers."""
        self._set_input_shapes(feed_dict)
        outputs = {}
        for name, is_input, dtype in self.io_tensors:
            if is_input:
                t = feed_dict[name]
                if t.dtype != dtype or not t.is_cuda or not t.is_contiguous():
                    buf = self._staging.get(name)
                    if buf is None or buf.shape != t.shape:
                        buf = self._staging[name] = torch.empty(t.shape, dtype=dtype, device="cuda")
                    buf.copy_(t)
                    t = buf
            else:
                t = outputs[name] = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed.")
        return outputs

    def refit(self, weights: dict):
        """Refit weights, each converted to the dtype the engine expects for it."""
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        refittable = set(refitter.get_all_weights())
        keep_alive = []  # buffers must stay valid until refit_cuda_engine()
        skipped = 0
        for name, wt in weights.items():
            if name not in refittable:
                skipped += 1
                continue
            proto = refitter.get_weights_prototype(name)
            t = wt.detach().to(dtype=_torch_dtype(proto.dtype)).contiguous()
            if t.numel() != proto.size:
                raise RuntimeError(f"Refit weight '{name}' has {t.numel()} elements, the engine expects {proto.size}.")
            keep_alive.append(t)
            location = trt.TensorLocation.DEVICE if t.is_cuda else trt.TensorLocation.HOST
            if not refitter.set_named_weights(name, trt.Weights(proto.dtype, t.data_ptr(), t.numel()), location):
                raise RuntimeError(f"TensorRT rejected refit weight '{name}'.")
        if skipped:
            _log(f"{skipped} LoRA weights aren't refittable in this engine and were skipped.")
        if not refitter.refit_cuda_engine():
            raise RuntimeError("Refitting the TensorRT engine failed.")
        print(f"[TensorRT] Refitted {len(keep_alive)} weights.", flush=True)

    def __str__(self):
        if self.engine is None:
            return f"{self.engine_path} (not loaded)"
        lines = []
        for p in range(self.engine.num_optimization_profiles):
            lines.append(f"Profile {p}:")
            for name, is_input, _ in self.io_tensors:
                if is_input:
                    shape = " / ".join(str(tuple(s)) for s in self.engine.get_tensor_profile_shape(name, p))
                else:
                    shape = tuple(self.engine.get_tensor_shape(name))
                lines.append(f"\t{name} = {shape}")
        return "\n".join(lines)
