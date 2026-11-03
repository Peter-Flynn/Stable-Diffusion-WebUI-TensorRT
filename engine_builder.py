#
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
"""Builds TensorRT engines. Run as a script by exporter.export_trt, so the VRAM the builder keeps
goes back to the driver when the build ends; progress and the result are reported on stdout.
Doesn't import torch or the WebUI."""

import json
import os
import sys
import threading
import time

import tensorrt as trt
from tqdm import tqdm

# TensorRT 11 only builds strongly typed networks: precision comes from the ONNX graph.
STRONGLY_TYPED = not hasattr(trt.BuilderFlag, "FP16")

_MEMORY_HINTS = ("insufficient memory", "memory is insufficient", "could not be allocated", "out of memory")

PROGRESS_PREFIX = "@@trt-progress "
RESULT_PREFIX = "@@trt-result "


def _log(message):
    print(f"[TensorRT] {message}", flush=True)


class TRTLogger(trt.ILogger):
    """Prints errors and (once each) build warnings, keeps errors for exception messages, and
    counts out-of-memory messages, which TensorRT only reports as warnings."""

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


def atomic_write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(memoryview(data))
    os.replace(tmp_path, path)


class ProgressMonitor(trt.IProgressMonitor):
    """tqdm bars per build phase (up to `max_depth` deep), and `callback(fraction, description)`
    for the top-level phase and its active sub-phase."""

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
        n = max(0, min(int(n), phase["total"]))
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
        self._fraction = max(self._fraction, min(fraction / top["total"], 1.0))
        try:
            self.callback(self._fraction, desc)
        except Exception:
            pass

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            self._close(phase_name)
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
            self._cancel = True

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


def _is_dynamic(input_profiles) -> bool:
    return any(tuple(lo) != tuple(hi) for shapes in input_profiles for lo, _, hi in shapes.values())


def build_engine(engine_path, onnx_path, fp16, input_profiles, refit=True, timing_cache=None, optimization_level=None, progress_callback=None):
    """Builds and saves an engine. Returns (success, number of memory allocation failures)."""
    print(f"[TensorRT] Building {engine_path}", flush=True)
    TRT_LOGGER.begin(print_warnings=True)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if STRONGLY_TYPED else 0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if hasattr(trt.OnnxParserFlag, "NATIVE_INSTANCENORM"):
        parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        _log(f"TensorRT could not parse {onnx_path}:\n{errors}")
        return False, 0

    config = builder.create_builder_config()
    config.progress_monitor = ProgressMonitor(progress_callback)
    if STRONGLY_TYPED:
        print(f"[TensorRT] TensorRT {trt.__version__}: strongly typed build.", flush=True)
    elif fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if optimization_level is not None:
        config.builder_optimization_level = int(optimization_level)
    resize = getattr(getattr(trt, "PreviewFeature", None), "RUNTIME_ACTIVATION_RESIZE_10_10", None)
    if resize is not None and _is_dynamic(input_profiles):
        config.set_preview_feature(resize, True)  # activation memory sized for the shapes in use
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
    memory_messages = TRT_LOGGER.memory_messages

    if memory_messages:
        _log(
            f"TensorRT reported {memory_messages} memory allocation failures while building "
            f'(e.g. "{TRT_LOGGER.first_memory_message}"). Kernels that didn\'t fit were skipped, so the engine '
            "may be slower than it could be. Close other GPU programs or narrow the profile and rebuild."
        )
    if serialized is None:
        _log("Engine build failed: TensorRT returned no engine (see errors above).")
        return False, memory_messages
    try:
        atomic_write(engine_path, serialized)
    except Exception as e:
        _log(f"Could not save the engine: {e}")
        return False, memory_messages
    if timing_cache:
        try:
            atomic_write(timing_cache, config.get_timing_cache().serialize())
        except Exception as e:
            _log(f"Could not save the timing cache {timing_cache}: {e}")
    return True, memory_messages


def parse_message(line: str):
    """("progress" | "result", payload) for protocol lines, ("output", line) for the rest."""
    for kind, prefix in (("progress", PROGRESS_PREFIX), ("result", RESULT_PREFIX)):
        if line.startswith(prefix):
            try:
                return kind, json.loads(line[len(prefix) :])
            except ValueError:
                break
    return "output", line


def _send(prefix, payload):
    sys.stdout.write(prefix + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _progress_sender(min_interval=0.5):
    last = {"time": 0.0, "desc": None}

    def send(fraction, desc):
        now = time.monotonic()
        if desc == last["desc"] and fraction < 1.0 and now - last["time"] < min_interval:
            return
        last.update(time=now, desc=desc)
        _send(PROGRESS_PREFIX, {"fraction": float(fraction), "desc": str(desc)})

    return send


def _exit_with_parent():
    """The parent holds our stdin open until we exit: EOF means it's gone."""

    def watch():
        try:
            while sys.stdin.read(4096):
                pass
        except Exception:
            pass
        os._exit(3)

    threading.Thread(target=watch, name="trt-parent-watch", daemon=True).start()


def main(argv):
    if len(argv) != 1:
        print("usage: engine_builder.py '<job as JSON>'", file=sys.stderr)
        return 2
    job = json.loads(argv[0])
    _exit_with_parent()
    try:
        ok, memory_messages = build_engine(
            job["engine_path"],
            job["onnx_path"],
            job.get("fp16", True),
            job["profiles"],
            refit=job.get("refit", True),
            timing_cache=job.get("timing_cache"),
            optimization_level=job.get("optimization_level"),
            progress_callback=_progress_sender(),
        )
    except Exception as e:
        _log(f"Engine build failed: {e}")
        ok, memory_messages = False, TRT_LOGGER.memory_messages
    _send(RESULT_PREFIX, {"ok": bool(ok), "memory_messages": int(memory_messages)})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
