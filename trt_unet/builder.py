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


# TensorRT 11 spends ~95% of build time inside subphases of "Building engine from subgraph" (total=1) without advancing
# its parent step counter between sibling phases, causing top-level step fractions to latch and stall during timing.
_TRT11_RANGES = {
    "Cloning network graph": (0.00, 0.02),
    "Preparing graph for optimization": (0.02, 0.05),
    "Optimizing graph": (0.05, 0.12),
    "Timing graph nodes": (0.12, 0.72),
    "Timing tensor reformats": (0.72, 0.92),
    "Determining refit engines": (0.92, 0.98),
}


class Progress(trt.IProgressMonitor):
    def __init__(self, callback, max_depth=1):
        trt.IProgressMonitor.__init__(self)
        self.callback, self.max_depth = callback, max_depth
        self.phases, self.top, self.fraction, self.cancel, self._trt11 = {}, None, 0.0, False, False

    def _close(self, name):
        phase = self.phases.pop(name, None)
        if phase and phase["bar"] is not None:
            phase["bar"].close()

    def _progress_of(self, name) -> float:
        phase = self.phases[name]
        done = phase["n"]
        child = next((k for k, p in self.phases.items() if p["parent"] == name), None)
        if child is not None and done < phase["total"]:
            done += self._progress_of(child)
        return min(done / phase["total"], 1.0)

    def _report(self):
        if self.top not in self.phases:
            return
        deepest = max(self.phases.items(), key=lambda kv: kv[1]["depth"])[0]
        text = f"{self.top}: {deepest}" if deepest != self.top else self.top
        if self._trt11:
            top = self.phases[self.top]
            if top["n"] >= top["total"]:
                frac = 1.0
            else:
                frac = self.fraction
                for name, (lo, hi) in _TRT11_RANGES.items():
                    if name in self.phases:
                        frac = lo + (hi - lo) * self._progress_of(name)
                        break
        else:
            frac = self._progress_of(self.top)
        self.fraction = max(self.fraction, frac)
        self.callback(self.fraction, text)

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            if parent_phase is None:
                for name in list(self.phases):
                    self._close(name)
                depth, self.top, self._trt11 = 0, phase_name, False
            else:
                self._close(phase_name)
                depth = self.phases.get(parent_phase, {}).get("depth", self.max_depth) + 1
                if phase_name in _TRT11_RANGES:
                    self._trt11 = True
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
            self.cancel = True
            return False

    def phase_finish(self, phase_name):
        try:
            phase = self.phases.get(phase_name)
            if phase is not None:
                if phase["bar"] is not None:
                    phase["bar"].update(phase["total"] - phase["bar"].n)
                phase["n"] = phase["total"]
                if phase_name == self.top or (self._trt11 and phase_name in _TRT11_RANGES):
                    self._report()
            self._close(phase_name)
        except KeyboardInterrupt:
            self.cancel = True


    def close(self):
        for name in list(self.phases):
            self._close(name)


def _send(prefix, payload):
    sys.stdout.write(prefix + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _progress_sender(min_interval=0.5):
    last = {"time": 0.0, "text": None, "fraction": -1.0}

    def send(fraction, text):
        now = time.monotonic()
        done_now = fraction >= 1.0 and last["fraction"] < 1.0
        if text == last["text"] and not done_now and now - last["time"] < min_interval:
            return
        last.update(time=now, text=text, fraction=float(fraction))
        _send(PROGRESS, {"fraction": float(fraction), "text": str(text)})

    return send


class NoEngine(RuntimeError):
    pass


def _load_timing_cache(builder, cache_path, logger):
    config = builder.create_builder_config()
    try:
        try:
            data = b""
            if cache_path and os.path.isfile(cache_path):
                with open(cache_path, "rb") as f:
                    data = f.read()
            cache = config.create_timing_cache(data)
            if cache is None:
                raise ValueError("TensorRT rejected the timing cache")
        except Exception as e:
            log("WARNING", f"The timing cache couldn't be read, so this build starts without it: {e}")
            logger.errors.clear()
            cache = config.create_timing_cache(b"")
        finally:
            data = b""
        return cache
    finally:
        del config


def _save_timing_cache(cache, cache_path):
    if not cache_path or cache is None:
        return
    tmp = cache_path + ".tmp"
    try:
        serialized = cache.serialize()
        if serialized is not None:
            with open(tmp, "wb") as f:
                f.write(memoryview(serialized))
            os.replace(tmp, cache_path)
    except Exception as e:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        log("DEBUG", f"Couldn't save the timing cache: {e}")


def _build_one(builder, cache, job, logger, progress, strip=True):
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
    if cache is not None:
        config.set_timing_cache(cache, ignore_mismatch=True)

    present = {network.get_input(i).name for i in range(network.num_inputs)}  # SDXL's shallow UNet has no prompt input
    for shapes in job["profiles"]:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            if name in present:
                profile.set_shape(name, min=lo, opt=opt, max=hi)
        config.add_optimization_profile(profile)

    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        progress.close()
    if plan is None:
        said = logger.errors[-1] if logger.errors else logger.memory_example
        raise NoEngine("TensorRT returned no engine" + (f": {said}" if said else ""))
    with open(job["plan"], "wb") as f:
        f.write(memoryview(plan))
    del plan, config, parser, network


def build(job, logger, progress, strip=True):
    builder = trt.Builder(logger)
    cache_path = job.get("timing_cache")
    cache = _load_timing_cache(builder, cache_path, logger)
    _build_one(builder, cache, job, logger, progress, strip=strip)
    _save_timing_cache(cache, cache_path)


def build_batch(batch_job, logger, send_progress):
    # Builds a group of small engines (VAE strips) in one process sharing one trt.Builder and one in-memory
    # ITimingCache, avoiding per-engine CUDA/TRT startup and repeated cache serialization.
    jobs = batch_job["jobs"]
    builder = trt.Builder(logger)
    cache_path = batch_job.get("timing_cache")
    cache = _load_timing_cache(builder, cache_path, logger)
    default_strip = batch_job.get("strip", True)
    default_opt = batch_job.get("optimization_level")
    results, n = [], max(1, len(jobs))
    any_built = False
    try:
        for i, sub in enumerate(jobs):
            sub_strip = sub.get("strip", default_strip)
            sub_job = {
                **sub,
                "optimization_level": sub.get("optimization_level", default_opt),
            }
            label = sub.get("label") or os.path.basename(sub["plan"])
            log("DEBUG", f"Building {os.path.basename(sub['plan'])} with TensorRT {trt.__version__}.")
            logger.errors.clear()
            logger.memory_example = None
            sub_send = lambda f, t, i=i, label=label: send_progress((i + f) / n, f"{label}: {t}")
            try:
                _build_one(builder, cache, sub_job, logger, Progress(sub_send), strip=sub_strip)
                any_built = True
                results.append({"key": sub["key"], "ok": True, "stripped": sub_strip})
            except Exception as e:
                error = str(e) or type(e).__name__
                if not isinstance(e, RuntimeError):
                    log("DEBUG", f"The build failed:\n{traceback.format_exc().rstrip()}")
                retry = sub_strip and isinstance(e, NoEngine) and any(w in " ".join(logger.errors).lower() for w in ("refit", "strip", "weight"))
                return {"ok": False, "error": error, "failed_key": sub["key"], "retry_without_strip": retry, "results": results}
        return {"ok": True, "error": None, "results": results}
    finally:
        if any_built:
            _save_timing_cache(cache, cache_path)


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
        if "jobs" in job:
            result = build_batch(job, logger, _progress_sender())
        else:
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
