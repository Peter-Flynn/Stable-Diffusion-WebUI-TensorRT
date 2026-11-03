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

from engine_builder import TRT_LOGGER

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


def _torch_dtype(trt_dtype):
    if trt_dtype not in _TRT_TO_TORCH:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return _TRT_TO_TORCH[trt_dtype]


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
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

    def max_activation_bytes(self) -> int:
        """Upper bound of the activation memory an inference allocates (the profile maximum)."""
        if self.engine is None:
            return 0
        for attr in ("device_memory_size_v2", "device_memory_size"):
            try:
                size = int(getattr(self.engine, attr, 0) or 0)
            except Exception:
                size = 0
            if size > 0:
                return size
        return 0

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
        try:
            ok = refitter.refit_cuda_engine()
        finally:
            # A refit caches resources for the engine that stay allocated (outside PyTorch, where the
            # WebUI can't reclaim them) for as long as the engine lives, unless released.
            release = getattr(refitter, "release_refit_resources", None)
            if release is not None:
                try:
                    release()
                except Exception as e:
                    _log(f"Could not release refit resources: {e}")
        if not ok:
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
