"""Builds a weightless engine, in a process of its own so all the VRAM the builder holds goes back to the
driver when it ends. Run by export.py; reports progress and its result on stdout. Doesn't import torch."""

import json
import os
import sys
import threading
import time

import numpy as np
import tensorrt as trt
from tqdm import tqdm

PLACEHOLDER_MARK = 1717.0  # export.py starts each FP16 placeholder with this and its index
PLACEHOLDER_PREFIX = "placeholder/"  # names of the null placeholders' weights: "placeholder/<UNet weight>"
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


def _checked(logger, what, call):
    """TensorRT logs a rejected API call instead of raising: one that logged an error failed."""
    before = len(logger.errors)
    result = call()
    if len(logger.errors) > before:
        raise RuntimeError(f"{what}: {logger.errors[-1]}")
    return result


def _null_placeholders(network, logger, names, sizes) -> set:
    """Gives the UNet weights in constant layers TensorRT's null placeholders (a type and a size, no data),
    which the builder doesn't hold or convert. As NVIDIA documents them, a placeholder has to come with
    the layer, so each weight gets a new constant layer created with one, named "placeholder/<weight>" (the
    parser already gave the old one the weight's name), and the layers that read the old one read the new
    one; the old one is left unused. Returns the UNet weights given placeholders; raises if TensorRT
    rejects anything, before the network is built."""
    found = {}  # UNet weight index -> its constant layers (export.py's placeholder, told apart by marker and size)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONSTANT:
            continue
        layer.__class__ = trt.IConstantLayer
        values = layer.weights  # a NumPy array: TensorRT's layers hand out their weights' values
        values = values if isinstance(values, np.ndarray) else values.numpy()
        if values.dtype != np.float16 or values.size < 3:
            continue
        head = values.reshape(-1)[:3]
        if head[0] != PLACEHOLDER_MARK or not np.isfinite(head).all():
            continue  # not one of ours, or the parser changed it (e.g. transposed): stays as it is
        index = int(head[1]) * 2048 + int(head[2])
        if index >= len(names) or values.size != sizes[index]:
            continue
        found.setdefault(index, []).append(layer)

    replacement = {}  # old layer's output tensor name -> the new layer's
    for index, layers in found.items():
        name = names[index]
        placeholder = trt.Weights(trt.DataType.HALF, 0, sizes[index])
        new = _checked(logger, f"adding a null placeholder for {name}", lambda: network.add_constant(layers[0].shape, placeholder))
        if new is None:
            raise RuntimeError(f"TensorRT didn't add a null placeholder for {name}")
        # TensorRT's own descriptor of the placeholder carries the name
        if not _checked(logger, f"naming the null placeholder for {name}", lambda: network.set_weights_name(new.weights, PLACEHOLDER_PREFIX + name)):
            raise RuntimeError(f"TensorRT didn't take the name of the null placeholder for {name}")
        for layer in layers:
            replacement[layer.get_output(0).name] = new.get_output(0)

    rewired = set()
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if tensor is not None and tensor.name in replacement:
                _checked(logger, f"connecting {layer.name} to a null placeholder", lambda: layer.set_input(j, replacement[tensor.name]))
                if layer.get_input(j).name != replacement[tensor.name].name:
                    raise RuntimeError(f"TensorRT didn't connect {layer.name} to its null placeholder")
                rewired.add(tensor.name)
    unused = set(replacement) - rewired
    if unused:
        raise RuntimeError(f"{len(unused)} weights' constant layers had no readers to connect")
    log(f"Null placeholders for {len(found)} of {len(names)} UNet weights")
    return {names[index] for index in found}


def build(job, logger, progress, strip=True, null=False):
    """null: TensorRT's null weight placeholders (strip only) for the UNet weights in constant layers,
    which keeps the builder from holding them; else the export's placeholder data, which it holds like
    real weights."""
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # strongly typed: precision comes from the FP16 graph
    parser = trt.OnnxParser(network, logger)
    # The parser opens files through narrow paths, which on Windows can't hold every character (e.g. a
    # user folder named in Japanese): parse from the model's folder with an ASCII relative path.
    cwd, (folder, name) = os.getcwd(), os.path.split(os.path.abspath(job["onnx"]))
    os.chdir(folder)
    try:
        parsed = parser.parse_from_file(name)
    finally:
        os.chdir(cwd)
    if not parsed:
        raise RuntimeError("TensorRT could not parse the ONNX model: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    nulls = _null_placeholders(network, logger, job["placeholders"], job["placeholder_sizes"]) if null and strip else set()
    # Every UNet weight has to be refittable, or the engine bakes in the placeholder values.
    unmarked = [
        name for name in job["weights"]
        if not (
            network.are_weights_marked_refittable(PLACEHOLDER_PREFIX + name)
            or network.mark_weights_refittable(PLACEHOLDER_PREFIX + name)
            if name in nulls
            else network.mark_weights_refittable(name)
        )
    ]
    if unmarked:
        raise RuntimeError(f"{len(unmarked)} UNet weights aren't refittable in the TensorRT network (e.g. {unmarked[0]})")

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
            build(job, logger, Progress(_progress_sender()), null=bool(job.get("null")))
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
