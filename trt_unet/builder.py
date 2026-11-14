# Builds one engine in a subprocess, so all the VRAM the builder used returns to the driver when it exits. Talks to
# export.run_builder over stdout (PROGRESS, LOG and RESULT lines) and draws a progress bar on stderr. Must not import
# torch.
import contextlib
import json
import os
import signal
import sys
import threading
import time
import traceback

import tensorrt as trt
from tqdm import tqdm

PROGRESS = "@@trt-progress "
LOG = "@@trt-log "
RESULT = "@@trt-result "
_MEMORY_HINTS = ("insufficient memory", "memory is insufficient", "could not be allocated", "out of memory")
# TensorRT 11 reports kernels it skips for lack of VRAM as errors ("Error Code 2: OutOfMemory (Requested size was
# ...)"), though the build continues. Counted, not logged: a build can produce thousands.
_ALLOCATION_FAILED = ("outofmemory (requested size",)


def log(level: str, message: str):
    _send(LOG, {"level": level, "message": message})


class Logger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.errors, self._seen = [], set()
        self.memory_warnings, self.memory_example = 0, None

    def log(self, severity, msg):
        level = int(severity)
        if level > int(trt.ILogger.Severity.WARNING):
            return
        lower = msg.lower()
        skipped = any(h in lower for h in _MEMORY_HINTS) and level == int(trt.ILogger.Severity.WARNING)
        if skipped or any(h in lower for h in _ALLOCATION_FAILED):
            self.memory_warnings += 1
            self.memory_example = self.memory_example or msg
        elif level <= int(trt.ILogger.Severity.ERROR):
            self.errors.append(msg)
            log("DEBUG", f"Builder error: {msg}")
        elif msg not in self._seen:
            self._seen.add(msg)
            log("DEBUG", f"Builder warning: {msg}")


class Progress(trt.IProgressMonitor):
    def __init__(self, callback, max_depth=1):
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
            bar = tqdm(total=num_steps, desc=phase_name, leave=False, position=depth, dynamic_ncols=True) if depth < self.max_depth else None
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
    network = builder.create_network(0)  # strongly typed: layer precisions come from the FP16 ONNX graph
    parser = trt.OnnxParser(network, logger)
    # The ONNX parser opens files through narrow (ANSI) paths on Windows, which fail on e.g. a Japanese user folder name:
    # parse from the file's folder by a relative name.
    cwd, (folder, name) = os.getcwd(), os.path.split(os.path.abspath(job["onnx"]))
    os.chdir(folder)
    try:
        parsed = parser.parse_from_file(name)
    finally:
        os.chdir(cwd)
    if not parsed:
        raise RuntimeError("TensorRT could not parse the ONNX model: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    unmarked = [name for name in job["weights"] if not network.mark_weights_refittable(name)]
    if unmarked:
        raise RuntimeError(f"{len(unmarked)} of the model's weights aren't in the TensorRT network (e.g. {unmarked[0]})")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
    if strip:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    if job.get("exact_fp32"):
        # fresh.py's FP32 layers: TF32's 10-bit mantissa breaks its Gram-Schmidt fit. FP16 layers aren't affected.
        config.clear_flag(trt.BuilderFlag.TF32)
    if job.get("optimization_level") is not None:
        config.builder_optimization_level = int(job["optimization_level"])
    if any(lo != hi for shapes in job["profiles"] for lo, _, hi in shapes.values()):
        config.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True)
    config.progress_monitor = progress

    cache_path = job.get("timing_cache")
    data = b""
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = f.read()
    try:
        cache = config.create_timing_cache(data)
    except Exception as e:
        log("WARNING", f"The timing cache couldn't be read, so this build starts without it: {e}")
        cache = config.create_timing_cache(b"")
    del data
    config.set_timing_cache(cache, ignore_mismatch=True)

    present = {network.get_input(i).name for i in range(network.num_inputs)}  # SDXL's shallow UNet has no prompt input
    for shapes in job["profiles"]:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            if name in present:
                profile.set_shape(name, min=lo, opt=opt, max=hi)
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        said = logger.errors[-1] if logger.errors else logger.memory_example
        raise NoEngine("TensorRT returned no engine" + (f": {said}" if said else ""))
    with open(job["plan"], "wb") as f:
        f.write(memoryview(plan))
    del plan
    if cache_path:
        with open(cache_path + ".tmp", "wb") as f:
            f.write(memoryview(config.get_timing_cache().serialize()))
        os.replace(cache_path + ".tmp", cache_path)


def _exit_with_parent():
    # The parent holds this process's stdin open and never writes to it, so EOF means the parent died. Kills the whole
    # process group too (export.py starts this process as a group leader on POSIX).
    def watch():
        try:
            while sys.stdin.read(4096):
                pass
        except Exception:
            pass
        if os.name != "nt" and os.getpgid(0) == os.getpid():
            with contextlib.suppress(OSError):
                os.killpg(0, signal.SIGKILL)
        os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def main(job_path):
    _exit_with_parent()
    logger, strip = Logger(), True
    result = {"ok": False, "error": None}
    try:
        with open(job_path, encoding="utf-8") as f:
            job = json.load(f)
        strip = job.get("strip", True)
        log("DEBUG", f"Building {os.path.basename(job['plan'])} with TensorRT {trt.__version__}.")
        build(job, logger, Progress(_progress_sender()), strip=strip)
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e) or type(e).__name__
        if not isinstance(e, RuntimeError):
            log("DEBUG", f"The build failed:\n{traceback.format_exc().rstrip()}")
        result["retry_without_strip"] = strip and isinstance(e, NoEngine) and any(w in " ".join(logger.errors).lower() for w in ("refit", "strip", "weight"))
    result["memory_warnings"] = logger.memory_warnings
    _send(RESULT, result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
