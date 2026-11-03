"""Builds a weightless engine, in a process of its own so all the VRAM the builder holds goes back to the
driver when it ends. Run by build.py; reports progress and its result on stdout. Doesn't import torch."""

import json
import os
import sys
import threading
import time

import tensorrt as trt
from tqdm import tqdm

PROGRESS = "@@trt-progress "
RESULT = "@@trt-result "
_MEMORY_HINTS = ("insufficient memory", "memory is insufficient", "could not be allocated", "out of memory")


def log(message):
    print(f"[TensorRT] {message}", flush=True)


class Logger(trt.ILogger):
    """Errors, each warning once, and a count of the out-of-memory warnings TensorRT gives when it has
    to skip kernels."""

    def __init__(self):
        trt.ILogger.__init__(self)
        self.errors, self.memory, self._seen = [], [], set()

    def log(self, severity, msg):
        level = int(severity)
        if level > int(trt.ILogger.Severity.WARNING):
            return
        if any(h in msg.lower() for h in _MEMORY_HINTS) and level == int(trt.ILogger.Severity.WARNING):
            self.memory.append(msg)
        elif level <= int(trt.ILogger.Severity.ERROR):
            self.errors.append(msg)
            print(f"[TensorRT] ERROR: {msg}", flush=True)
        elif msg not in self._seen:
            self._seen.add(msg)
            print(f"[TensorRT] Warning: {msg}", flush=True)


class Progress(trt.IProgressMonitor):
    """tqdm bars per build phase, and `callback(fraction, text)` for the top phase and its active sub-phase."""

    def __init__(self, callback, max_depth=3):
        trt.IProgressMonitor.__init__(self)
        self.callback, self.max_depth = callback, max_depth
        self.phases, self.top, self.fraction, self.cancel = {}, None, 0.0, False

    def _close(self, name):
        phase = self.phases.pop(name, None)
        if phase and phase["bar"] is not None:
            phase["bar"].close()

    def _report(self):
        top = self.phases.get(self.top)
        if top is None:
            return
        fraction, text = top["n"], self.top
        child = next((p for p in self.phases.values() if p["parent"] == self.top), None)
        if child is not None:
            fraction += child["n"] / child["total"]
            text = f"{self.top}: {max(self.phases.items(), key=lambda kv: kv[1]['depth'])[0]}"
        self.fraction = max(self.fraction, min(fraction / top["total"], 1.0))
        self.callback(self.fraction, text)

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            self._close(phase_name)
            if parent_phase is None:
                depth, self.top, self.fraction = 0, phase_name, 0.0
            else:
                depth = self.phases.get(parent_phase, {}).get("depth", self.max_depth) + 1
            bar = tqdm(total=num_steps, desc=phase_name, leave=depth == 0, position=depth, dynamic_ncols=True) if depth < self.max_depth else None
            self.phases[phase_name] = {"bar": bar, "depth": depth, "parent": parent_phase, "total": max(int(num_steps), 1), "n": 0}
            self._report()
        except KeyboardInterrupt:
            self.cancel = True

    def step_complete(self, phase_name, step):
        try:
            phase = self.phases.get(phase_name)
            if phase is not None:
                n = max(0, min(int(step), phase["total"]))
                if phase["bar"] is not None:
                    phase["bar"].update(n - phase["bar"].n)
                phase["n"] = n
                self._report()
            return not self.cancel
        except KeyboardInterrupt:
            return False

    def phase_finish(self, phase_name):
        try:
            phase = self.phases.get(phase_name)
            if phase is not None:
                self.step_complete(phase_name, phase["total"])
            self._close(phase_name)
        except KeyboardInterrupt:
            self.cancel = True


def _send(prefix, payload):
    sys.stdout.write(prefix + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _progress_sender(min_interval=0.5):
    last = {"time": 0.0, "text": None}

    def send(fraction, text):
        now = time.monotonic()
        if text == last["text"] and fraction < 1.0 and now - last["time"] < min_interval:
            return
        last.update(time=now, text=text)
        _send(PROGRESS, {"fraction": float(fraction), "text": str(text)})

    return send


class NoEngine(RuntimeError):
    pass


def build(job, logger, progress, strip=True):
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # strongly typed: precision comes from the FP16 graph
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(job["onnx"]):
        raise RuntimeError("TensorRT could not parse the ONNX model: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    unmarked = [name for name in job["weights"] if not network.mark_weights_refittable(name)]
    if unmarked:
        raise RuntimeError(f"{len(unmarked)} UNet weights aren't in the TensorRT network (e.g. {unmarked[0]})")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)  # only the UNet's weights: they come from the checkpoint
    if strip:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    if job.get("optimization_level") is not None:
        config.builder_optimization_level = int(job["optimization_level"])
    if any(lo != hi for shapes in job["profiles"] for lo, _, hi in shapes.values()):
        config.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True)  # activation memory sized for the shapes in use
    config.progress_monitor = progress

    cache_path = job.get("timing_cache")
    data = b""
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = f.read()
    try:
        cache = config.create_timing_cache(data)
    except Exception as e:
        log(f"Ignoring the unreadable timing cache: {e}")
        cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=True)

    for shapes in job["profiles"]:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            profile.set_shape(name, min=lo, opt=opt, max=hi)
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise NoEngine("TensorRT returned no engine" + (f": {logger.errors[-1]}" if logger.errors else ""))
    with open(job["plan"], "wb") as f:
        f.write(memoryview(plan))
    if cache_path:
        with open(cache_path + ".tmp", "wb") as f:
            f.write(memoryview(config.get_timing_cache().serialize()))
        os.replace(cache_path + ".tmp", cache_path)


def _exit_with_parent():
    """The parent holds our stdin open: EOF means it's gone, and nobody would use the engine."""

    def watch():
        try:
            while sys.stdin.read(4096):
                pass
        except Exception:
            pass
        os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    _exit_with_parent()
    logger = Logger()
    result = {"ok": False, "error": None}
    try:
        log(f"Building with TensorRT {trt.__version__}")
        try:
            build(job, logger, Progress(_progress_sender()))
            result["stripped"] = True
        except NoEngine:
            if not any(w in " ".join(logger.errors).lower() for w in ("refit", "strip", "weight")):
                raise
            # Without stripping, the plan holds placeholder weights (GBs) but runs the same way.
            log("Building a weight-stripped engine failed; building one that stores placeholder weights instead.")
            logger.errors.clear()
            build(job, logger, Progress(_progress_sender()), strip=False)
            result["stripped"] = False
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
        log(f"Engine build failed: {e}")
    result["memory_warnings"] = len(logger.memory)
    result["memory_example"] = logger.memory[0] if logger.memory else None
    _send(RESULT, result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
