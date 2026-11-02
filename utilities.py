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
# TensorRT engine wrapper built directly on the TensorRT Python API (no Polygraphy).
# Supports TensorRT 10.x (weakly typed FP16 builds) and TensorRT 11+ (strongly typed
# builds, where BuilderFlag.FP16 no longer exists and precision comes from the ONNX graph).

import os
from collections import OrderedDict
from logging import error, warning

import torch
from torch.cuda import nvtx
import tensorrt as trt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.n = 0

        def update(self, n=1):
            self.n += n

        def refresh(self):
            pass


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

TRT_VERSION = tuple(int(x) for x in trt.__version__.split(".")[:2] if x.isdigit())

# TensorRT 11 removed the weakly typed precision flags; networks must be strongly typed.
HAS_WEAK_TYPING = hasattr(trt.BuilderFlag, "FP16")
HAS_STRONG_TYPING = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")

_TRT_TO_TORCH = {}
for _trt_name, _torch_name in (
    ("FLOAT", "float32"),
    ("HALF", "float16"),
    ("BF16", "bfloat16"),
    ("INT8", "int8"),
    ("UINT8", "uint8"),
    ("INT32", "int32"),
    ("INT64", "int64"),
    ("BOOL", "bool"),
    ("FP8", "float8_e4m3fn"),
):
    if hasattr(trt.DataType, _trt_name) and hasattr(torch, _torch_name):
        _TRT_TO_TORCH[getattr(trt.DataType, _trt_name)] = getattr(torch, _torch_name)


def trt_to_torch_dtype(dtype):
    try:
        return _TRT_TO_TORCH[dtype]
    except KeyError:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _atomic_write(path, data: bytes):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


class TQDMProgressMonitor(trt.IProgressMonitor):
    """Build progress.

    Console: one tqdm bar per phase, nested up to `max_depth`, always closed when the
    phase finishes. UI: an optional `callback(fraction, description)` driven by the
    top-level phase (plus the progress of its active sub-phase), so the WebUI shows a
    single bar instead of every nested tqdm bar."""

    def __init__(self, callback=None, max_depth=3):
        trt.IProgressMonitor.__init__(self)
        self._phases = {}
        self._step_result = True
        self.max_depth = max_depth
        self.callback = callback
        self._top = None
        self._last_fraction = 0.0

    def _close(self, phase_name):
        phase = self._phases.pop(phase_name, None)
        if phase is not None and phase["tq"] is not None:
            phase["tq"].close()

    def _set(self, phase_name, n):
        phase = self._phases.get(phase_name)
        if phase is None:
            return
        n = max(0, min(int(n), phase["total"]))  # TensorRT may overshoot; keep bars <= 100%
        if phase["tq"] is not None and n != phase["tq"].n:
            phase["tq"].update(n - phase["tq"].n)
        phase["n"] = n
        self._report()

    def _report(self):
        if self.callback is None or self._top not in self._phases:
            return
        top = self._phases[self._top]
        fraction = top["n"]
        desc = self._top
        # Add the fractional progress and name of the deepest active phase under the top one.
        child = next((k for k, v in self._phases.items() if v["parent"] == self._top), None)
        if child is not None:
            c = self._phases[child]
            fraction += c["n"] / c["total"]
            deepest = max(self._phases.items(), key=lambda kv: kv[1]["depth"])[0]
            desc = f"{self._top}: {deepest}"
        # Between sub-phases the estimate can dip; never move the UI bar backwards.
        self._last_fraction = max(self._last_fraction, min(fraction / top["total"], 1.0))
        try:
            self.callback(self._last_fraction, desc)
        except Exception:
            pass

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            self._close(phase_name)  # phase names can repeat: never leak the previous bar
            if parent_phase is None:
                depth = 0
                self._top = phase_name
                self._last_fraction = 0.0
            else:
                depth = self._phases.get(parent_phase, {}).get("depth", self.max_depth) + 1
            tq = None
            if depth < self.max_depth:
                tq = tqdm(total=num_steps, desc=phase_name, leave=depth == 0, position=depth, dynamic_ncols=True)
            self._phases[phase_name] = {
                "tq": tq,
                "depth": depth,
                "parent": parent_phase,
                "total": max(int(num_steps), 1),
                "n": 0,
            }
            self._report()
        except KeyboardInterrupt:
            # phase_start cannot cancel the build directly; request it from step_complete.
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._phases:
                self._set(phase_name, self._phases[phase_name]["total"])
            self._close(phase_name)
            self._report()
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            self._set(phase_name, step)
            return self._step_result
        except KeyboardInterrupt:
            # No need to propagate this to TensorRT; just cancel the build.
            return False


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self._clear_state()

    def _clear_state(self):
        # Staging buffers for inputs that can't be bound directly (wrong dtype / non-contiguous).
        self.tensors = OrderedDict()
        # (name, is_input, torch dtype) for every I/O tensor of the engine.
        self.io_tensors = []
        self._input_shapes = {}
        self._device_memory = None
        self._user_managed_memory = False

    def __del__(self):
        try:
            self.reset(None)
        except Exception:
            pass

    def reset(self, engine_path=None):
        # Order matters: the context must be destroyed before its engine.
        self.context = None
        self.engine = None
        self._clear_state()
        self.engine_path = engine_path

    # ------------------------------------------------------------------ build

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,  # kept for API compatibility (Polygraphy-only option)
        enable_all_tactics=False,  # kept for API compatibility (Polygraphy-only option)
        timing_cache=None,
        update_output_names=None,
        optimization_level=None,
        progress_callback=None,
    ):
        print(f"[TensorRT] Building engine for {onnx_path}: {self.engine_path}", flush=True)
        builder = trt.Builder(TRT_LOGGER)

        strongly_typed = HAS_STRONG_TYPING and not HAS_WEAK_TYPING
        network_flags = 0
        if strongly_typed:
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, TRT_LOGGER)
        if hasattr(trt.OnnxParserFlag, "NATIVE_INSTANCENORM"):
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            error(f"TensorRT could not parse {onnx_path}:\n{errors}")
            return 1

        if update_output_names:
            self._set_network_outputs(network, update_output_names)

        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor(callback=progress_callback)

        if strongly_typed:
            print(
                f"[TensorRT] TensorRT {trt.__version__}: strongly typed build "
                "(precision taken from the ONNX graph).",
                flush=True,
            )
        elif fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if optimization_level is not None and hasattr(config, "builder_optimization_level"):
            # 0-5, TensorRT default 3. Higher searches more tactics: slower builds, sometimes faster engines.
            config.builder_optimization_level = int(optimization_level)

        if enable_refit:
            refit_flag = getattr(trt.BuilderFlag, "REFIT", None)
            if refit_flag is not None:
                config.set_flag(refit_flag)
            else:
                warning("This TensorRT version has no BuilderFlag.REFIT; LoRA refitting will be unavailable.")

        if timing_cache:
            try:
                cache_data = b""
                if os.path.isfile(timing_cache):
                    with open(timing_cache, "rb") as f:
                        cache_data = f.read()
                cache = config.create_timing_cache(cache_data)
                config.set_timing_cache(cache, ignore_mismatch=True)
            except Exception as e:
                warning(f"Could not load timing cache {timing_cache} ({e}); starting with an empty one.")
                config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=True)

        for profile_shapes in input_profile or []:
            profile = builder.create_optimization_profile()
            for name, dims in profile_shapes.items():
                if len(dims) != 3:
                    raise ValueError(f"Profile for '{name}' needs (min, opt, max) shapes, got {dims}")
                profile.set_shape(name, min=dims[0], opt=dims[1], max=dims[2])
            config.add_optimization_profile(profile)

        print("[TensorRT] Compiling engine (this can take several minutes)...", flush=True)
        try:
            serialized = builder.build_serialized_network(network, config)
        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1
        if serialized is None:
            error("Failed to build engine: TensorRT returned no engine (see errors above).")
            return 1

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.engine_path)), exist_ok=True)
            _atomic_write(self.engine_path, bytes(serialized))
        except Exception as e:
            error(f"Failed to save engine: {e}")
            return 1

        if timing_cache:
            try:
                cache = config.get_timing_cache()
                if cache is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(timing_cache)), exist_ok=True)
                    _atomic_write(timing_cache, bytes(cache.serialize()))
            except Exception as e:
                warning(f"Could not save timing cache {timing_cache}: {e}")

        print(f"[TensorRT] Engine saved: {self.engine_path}", flush=True)
        return 0

    @staticmethod
    def _set_network_outputs(network, names):
        tensors = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                t = layer.get_output(j)
                tensors[t.name] = t
        for i in reversed(range(network.num_outputs)):
            network.unmark_output(network.get_output(i))
        for name in names:
            if name not in tensors:
                raise ValueError(f"Tensor '{name}' not found in network")
            network.mark_output(tensors[name])

    # -------------------------------------------------------------- runtime

    def load(self):
        print(f"[TensorRT] Loading engine: {self.engine_path}", flush=True)
        with open(self.engine_path, "rb") as f:
            data = f.read()
        if self.runtime is None:
            self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT could not load {os.path.basename(self.engine_path)}. Engines only work with "
                f"the TensorRT version and GPU they were built with (installed: TensorRT {trt.__version__}). "
                "Rebuild it from the TensorRT tab (enable 'Force Rebuild')."
            )

        self.io_tensors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            self.io_tensors.append((name, is_input, dtype))

    def activate(self, reuse_device_memory=False):
        """Create the execution context.

        When available, activation memory is user managed and allocated through
        PyTorch's caching allocator, sized for the shapes actually in use rather
        than the profile maximum. That memory is shared with the rest of the WebUI
        instead of being held privately by TensorRT."""
        self.context = None
        self._user_managed_memory = False
        self._device_memory = None
        self._input_shapes = {}

        strategy = getattr(trt, "ExecutionContextAllocationStrategy", None)
        if strategy is not None and hasattr(strategy, "USER_MANAGED") and hasattr(
            trt.IExecutionContext, "set_device_memory"
        ):
            try:
                self.context = self.engine.create_execution_context(strategy.USER_MANAGED)
                self._user_managed_memory = self.context is not None
            except Exception:
                self.context = None

        if self.context is None:
            self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create a TensorRT execution context (out of GPU memory?)")

    def _set_input_shapes(self, feed_dict):
        changed = False
        for name, is_input, _ in self.io_tensors:
            if not is_input or name not in feed_dict:
                continue
            shape = tuple(feed_dict[name].shape)
            if self._input_shapes.get(name) == shape:
                continue
            if not self.context.set_input_shape(name, shape):
                raise ValueError(
                    f"Input '{name}' with shape {shape} is outside the engine's optimization profile.\n{self}"
                )
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

        if self._device_memory is None or self._device_memory.numel() < size:
            self._device_memory = None  # release the old block first
            self._device_memory = torch.empty(size, dtype=torch.uint8, device="cuda")
        self.context.set_device_memory(self._device_memory.data_ptr(), self._device_memory.numel())

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        """Kept for API compatibility: sets input shapes. Buffers are handled in infer()."""
        if shape_dict:
            self._set_input_shapes(shape_dict)

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        """Run the engine on `stream` and return a dict of freshly allocated output tensors.

        Inputs already on the GPU with the engine's dtype are bound directly (no copy);
        anything else is copied into a reusable staging buffer."""
        nvtx.range_push("set_tensors")
        self._set_input_shapes(feed_dict)

        outputs = {}
        bound_inputs = []  # keep references alive until the call returns
        for name, is_input, dtype in self.io_tensors:
            if is_input:
                t = feed_dict[name]
                if t.dtype != dtype or not t.is_cuda or not t.is_contiguous():
                    buf = self.tensors.get(name)
                    if buf is None or buf.shape != t.shape:
                        buf = torch.empty(t.shape, dtype=dtype, device="cuda")
                        self.tensors[name] = buf
                    buf.copy_(t)
                    t = buf
                bound_inputs.append(t)
                self.context.set_tensor_address(name, t.data_ptr())
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(shape, dtype=dtype, device="cuda")
                outputs[name] = out
                self.context.set_tensor_address(name, out.data_ptr())
        nvtx.range_pop()

        nvtx.range_push("execute")
        ok = self.context.execute_async_v3(stream)
        nvtx.range_pop()
        if not ok:
            raise RuntimeError("TensorRT inference failed.")
        return outputs

    def refit_from_dict(self, refit_weights, is_fp16):
        """Refit engine weights. Each weight is converted to the dtype the engine expects
        (required for strongly typed engines, where weights may be FP32 or FP16)."""
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        refittable = set(refitter.get_all_weights())
        default_dtype = trt.DataType.HALF if is_fp16 else trt.DataType.FLOAT

        keep_alive = []
        missing = []
        for name, wt in refit_weights.items():
            if name not in refittable:
                missing.append(name)
                continue

            trt_dtype = default_dtype
            expected_size = None
            if hasattr(refitter, "get_weights_prototype"):
                try:
                    proto = refitter.get_weights_prototype(name)
                    trt_dtype = proto.dtype
                    expected_size = proto.size
                except Exception:
                    pass

            t = wt.detach().to(dtype=trt_to_torch_dtype(trt_dtype)).contiguous()
            if expected_size is not None and t.numel() != expected_size:
                raise RuntimeError(
                    f"Refit weight '{name}' has {t.numel()} elements, engine expects {expected_size}."
                )
            keep_alive.append(t)

            weights = trt.Weights(trt_dtype, t.data_ptr(), t.numel())
            location = trt.TensorLocation.DEVICE if t.is_cuda else trt.TensorLocation.HOST
            try:
                ok = refitter.set_named_weights(name, weights, location)
            except TypeError:  # older bindings without the location argument
                t_host = t.cpu() if t.is_cuda else t
                keep_alive.append(t_host)
                ok = refitter.set_named_weights(
                    name, trt.Weights(trt_dtype, t_host.data_ptr(), t_host.numel())
                )
            if ok is False:
                raise RuntimeError(f"TensorRT rejected refit weight '{name}'.")

        if missing:
            warning(f"{len(missing)} LoRA weights are not refittable in this engine and were skipped.")

        if not refitter.refit_cuda_engine():
            raise RuntimeError("Failed to refit new weights into the TensorRT engine.")
        print(f"[TensorRT] Refitted {len(keep_alive)} weights.", flush=True)

    def __str__(self):
        if self.engine is None:
            return f"TensorRT engine ({self.engine_path}, not loaded)"
        try:
            out = ""
            for p in range(self.engine.num_optimization_profiles):
                out += f"Profile {p}:\n"
                for name, is_input, _ in self.io_tensors:
                    if is_input:
                        shape = self.engine.get_tensor_profile_shape(name, p)
                        shape = " / ".join(str(tuple(s)) for s in shape)
                    else:
                        shape = tuple(self.engine.get_tensor_shape(name))
                    out += f"\t{name} = {shape}\n"
            return out
        except Exception:
            return f"TensorRT engine ({self.engine_path})"
