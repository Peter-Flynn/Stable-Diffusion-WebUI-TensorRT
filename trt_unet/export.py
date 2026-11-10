"""Exporting an architecture to a weightless engine: trace a fresh FP16 UNet to an ONNX graph whose
weights are placeholders, then build it in builder.py's process."""

import contextlib
import copy
import gc
import inspect
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
import warnings

import torch
import torch.nn.functional as F

from . import common, speed, store
from .arch import Arch, unet_config
from .common import gib, log

INPUTS = ("sample", "timesteps", "encoder_hidden_states", "y")
PLACEHOLDER_STRIDE = 256  # bytes between the placeholder weights' starts
CANCEL = threading.Event()
MEMORY_RESERVE = 2 * 2**30  # a build stops when less RAM and swap than this (or 5% of the RAM, if more) is free
DYNAMIC_AXES = {
    "sample": {0: "batch", 2: "height", 3: "width"},
    "timesteps": {0: "batch"},
    "encoder_hidden_states": {0: "batch", 1: "context"},
    "y": {0: "batch"},
    "latent": {0: "batch", 2: "height", 3: "width"},
}


def _attention(q, k, v, mask=None, upcast=False, scale=None):
    """Matmul + softmax, which ONNX exports and TensorRT fuses. q and k are each scaled by the square
    root of the scale, so q.k can't overflow in FP16."""
    root = math.sqrt(scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
    if upcast:
        q, k = q.float(), k.float()
    scores = torch.matmul(q * root, (k * root).transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
    return torch.matmul(scores.softmax(-1).to(v.dtype), v)


def _optimized_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    if mask is not None:
        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    out = _attention(q, k, v, mask, upcast=attn_precision == torch.float32)
    return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if is_causal:
        attn_mask = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device).tril()
    return _attention(query, key, value, attn_mask, scale=scale)


@contextlib.contextmanager
def _traceable_attention():
    import ldm_patched.ldm.modules.attention as attention

    swaps = [(F, "scaled_dot_product_attention", _sdpa)]
    swaps += [(attention, name, _optimized_attention) for name in ("optimized_attention", "optimized_attention_masked") if hasattr(attention, name)]
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
    for owner, name, fn in swaps:
        setattr(owner, name, fn)
    try:
        yield
    finally:
        for owner, name, fn in originals:
            setattr(owner, name, fn)


def _fresh_unet(config: dict, device, dtype=torch.float16):
    """The architecture with random weights: tracing doesn't depend on them, and distinct values keep
    the exporter from merging identical tensors."""
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

    unet = UNetModel(**config, dtype=dtype, device=device, operations=ops.disable_weight_init).eval()
    for t in unet.state_dict(keep_vars=True).values():
        if t.is_floating_point():
            t.normal_(0, 0.02)
    return unet


def _weightless_graph(onnx_path: str, input_names, agree_with, quiet: bool = False) -> dict:
    """Checks the exported weights' shapes against each {name: shape} in `agree_with`, tidies the graph
    (graph.py), and turns the weight inputs into initializers that read a small file of zeros, so the graph
    stays small: TensorRT only needs their shapes, the checkpoint provides the values. TensorRT tells
    weights apart by their memory (the parser maps the file once), so each starts at its own offset;
    their ranges overlap, so the file is about as large as the largest weight. Returns {weight name:
    shape in the checkpoint}."""
    import onnx

    from . import graph as graph_mod

    model = onnx.load(onnx_path)
    graph = model.graph
    used = {name for node in graph.node for name in node.input}
    shapes = {}
    for value in list(graph.input):
        if value.name in input_names:
            continue
        if value.name in used:
            shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
        else:
            graph.input.remove(value)
    wrong = [k for k, s in shapes.items() if any(other.get(k) != s for other in agree_with)]
    if wrong:
        raise RuntimeError(f"{len(wrong)} exported weights don't match the loaded UNet (e.g. {wrong[0]})")
    start = time.time()
    try:
        tidied = copy.deepcopy(model)  # small: the weights aren't in it yet
        before, after = graph_mod.tidy(tidied, shapes)
        model, graph = tidied, tidied.graph
        if not quiet:
            log(f"Tidied the graph from {before} to {after} nodes in {time.time() - start:.0f}s")
    except Exception as e:  # the graph as exported runs the same, with more layers
        traceback.print_exc()
        log(f"Could not tidy the graph ({e}); building from it as exported.")
    used = {name for node in graph.node for name in node.input}
    for value in [i for i in graph.input if i.name in shapes and i.name not in used]:  # TensorRT couldn't mark it refittable
        graph.input.remove(value)
        del shapes[value.name]

    sizes = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.FLOAT: 4, onnx.TensorProto.INT64: 8}
    data_name = os.path.basename(onnx_path) + ".zeros"
    end = 0
    for index, value in enumerate([i for i in graph.input if i.name in shapes]):
        graph.input.remove(value)
        t = value.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]  # as the graph takes it: GroupNorm's unsqueezed
        nbytes = math.prod(dims) * sizes[t.elem_type]
        offset = index * PLACEHOLDER_STRIDE
        end = max(end, offset + nbytes)
        init = graph.initializer.add()
        init.name, init.data_type, init.data_location = value.name, t.elem_type, onnx.TensorProto.EXTERNAL
        init.dims.extend(dims)
        for key, val in (("location", data_name), ("offset", str(offset)), ("length", str(nbytes))):
            entry = init.external_data.add()
            entry.key, entry.value = key, val
    with open(os.path.join(os.path.dirname(onnx_path), data_name), "wb") as f:
        f.truncate(end)
    onnx.save(model, onnx_path)
    return shapes


def _as_full(unet, x, timesteps, context, y=None):
    return speed.full_forward(unet, x, timesteps, context, y)


def _as_shallow(unet, x, timesteps, context, *rest):
    y, deep = rest if len(rest) == 2 else (None, rest[0])
    return speed.shallow_forward(unet, x, timesteps, context, y, deep)


@common.inference_mode
def export_onnx(arch: Arch, patcher, folder: str, device=None, dtype=torch.float16) -> dict:
    """Writes folder/unet.onnx, the UNet, which also outputs its deep feature, and folder/shallow.onnx, its
    outermost blocks, which take that feature (speed.py). Returns {"unet": {weight name: shape in the
    checkpoint}, "shallow": the same for the shallow graph, or None if it couldn't be exported, "deep":
    {"channels", "scale": the latent's size over the feature's, "branch"}}."""
    device = device or common.mm().get_torch_device()
    unet = _fresh_unet(unet_config(patcher), device, dtype)
    names = [n for n in INPUTS if n != "y" or arch.adm_channels]
    inputs = [
        torch.randn(2, arch.in_channels, 64, 64, dtype=dtype, device=device),
        torch.rand(2, device=device) * 999,  # timesteps stay FP32: FP16 can't hold fractional ones near 1000
        torch.randn(2, 77, arch.context_dim, dtype=dtype, device=device),
    ]
    if arch.adm_channels:
        inputs.append(torch.randn(2, arch.adm_channels, dtype=dtype, device=device))
    paths = {"unet": os.path.join(folder, "unet.onnx"), "shallow": os.path.join(folder, "shallow.onnx")}
    common_kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        common_kwargs["dynamo"] = False
    axes = {k: v for k, v in DYNAMIC_AXES.items() if k in names or k == "latent"}
    log(f"Exporting the {arch.name} UNet graphs...")
    shallow = True
    with _traceable_attention(), warnings.catch_warnings():
        # The values it warns about (shape checks, the head size) are fixed for an architecture.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        try:
            deep = speed.full_forward(unet, *inputs)[1]
            scale = inputs[0].shape[2] // deep.shape[2]
            axes["deep"] = {0: "batch", 2: "height", 3: "width"} if scale == 1 else {0: "batch", 2: "deep_height", 3: "deep_width"}
            info = {"channels": int(deep.shape[1]), "scale": int(scale), "branch": speed.BRANCH}
            unet.forward = types.MethodType(_as_full, unet)  # the parameters keep their names
            torch.onnx.export(unet, tuple(inputs), paths["unet"], input_names=names, output_names=["latent", "deep"], dynamic_axes=axes, **common_kwargs)
        except Exception as e:  # the UNet as reForge runs it, without Speed Hacks
            traceback.print_exc()
            log(f"Could not export the UNet with its deep feature ({e}); exporting it without Speed Hacks.")
            shallow, deep, info = False, None, None
            with contextlib.suppress(AttributeError):
                del unet.forward
            axes.pop("deep", None)
            torch.onnx.export(unet, tuple(inputs), paths["unet"], input_names=names, output_names=["latent"], dynamic_axes=axes, **common_kwargs)
        try:
            if shallow:
                unet.forward = types.MethodType(_as_shallow, unet)
                torch.onnx.export(
                    unet, (*inputs, deep), paths["shallow"], input_names=names + ["deep"], output_names=["latent"], dynamic_axes=axes, **common_kwargs
                )
        except Exception as e:
            traceback.print_exc()
            log(f"Could not export the shallow UNet graph ({e}); the engines will be built without Speed Hacks.")
            shallow = False
        finally:
            with contextlib.suppress(AttributeError):
                del unet.forward
    exported = {k: list(v.shape) for k, v in unet.state_dict(keep_vars=True).items()}
    del unet, inputs, deep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loaded = {k: list(v.shape) for k, v in patcher.model.diffusion_model.state_dict(keep_vars=True).items()}
    result = {"unet": _weightless_graph(paths["unet"], names, (exported, loaded)), "shallow": None, "deep": info}
    if shallow:
        try:
            result["shallow"] = _weightless_graph(paths["shallow"], names + ["deep"], (exported, loaded))
        except Exception as e:
            traceback.print_exc()
            log(f"Could not prepare the shallow UNet graph ({e}); the engines will be built without Speed Hacks.")
    return result


def _builder_env() -> dict:
    """Unbuffered UTF-8 output, the WebUI's GPU as the only visible device, and the system's memory allocator:
    the tcmalloc webui.sh preloads is slow to give memory the build frees back to the system."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    preload = [p for p in re.split(r"[\s:]+", env.get("LD_PRELOAD", "")) if p]
    if any("tcmalloc" in os.path.basename(p) for p in preload):
        preload = [p for p in preload if "tcmalloc" not in os.path.basename(p)]
        if preload:
            env["LD_PRELOAD"] = ":".join(preload)
        else:
            del env["LD_PRELOAD"]
    try:
        index = common.mm().get_torch_device().index
        index = torch.cuda.current_device() if index is None else index
    except Exception:
        return env
    visible = [d.strip() for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    elif index < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[index]
    return env


def _memory_short():
    """Why the build has to stop, if the system is about to run out of memory, else None."""
    try:
        import psutil

        ram, swap = psutil.virtual_memory(), psutil.swap_memory()
    except Exception:
        return None
    free = ram.available + swap.free
    if free >= max(MEMORY_RESERVE, ram.total // 20):
        return None
    return f"Stopped before the system ran out of memory ({gib(free)} of RAM and swap left). Build fewer or narrower profiles, or close other programs."


def _exited(proc, block=False) -> bool:
    """Whether the builder has exited. On POSIX it isn't reaped: until it is, no other process can get its ID,
    which is also its process group's, so the group can still be killed."""
    if os.name == "nt":
        return (proc.wait() if block else proc.poll()) is not None
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)) is not None
    except ChildProcessError:  # reaped already
        return True


def _kill_tree(proc):
    """The builder and whatever it started."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with contextlib.suppress(OSError):
            proc.kill()
    else:
        with contextlib.suppress(OSError):  # nothing left
            os.killpg(proc.pid, signal.SIGKILL)


def run_builder(job_path: str, progress=None) -> dict:
    """Runs builder.py on the job, as the leader of a process group of its own (on POSIX). When the builder
    exits, is cancelled, or the system is about to run out of memory, everything in the group is killed."""
    from . import builder

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(builder.__file__), job_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_builder_env(), text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=os.name != "nt",
    )
    result, state, lock = {}, {}, threading.Lock()

    def stop(reason=None):
        with lock:
            if "code" not in state:  # not reaped yet
                if reason:
                    state.setdefault("reason", reason)
                _kill_tree(proc)

    def watch():
        while "code" not in state:
            if CANCEL.wait(0.5):
                return stop("cancelled")
            if _exited(proc):
                return stop()  # anything it left running
            short = _memory_short()
            if short:
                log(short)
                return stop(short)

    threading.Thread(target=watch, daemon=True).start()
    ended = False
    try:
        for line in proc.stdout:
            if line.startswith(builder.PROGRESS):
                if progress is not None:
                    with contextlib.suppress(Exception):
                        message = json.loads(line[len(builder.PROGRESS):])
                        progress(message["fraction"], message["text"])
            elif line.startswith(builder.RESULT):
                with contextlib.suppress(ValueError):
                    result = json.loads(line[len(builder.RESULT):])
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        ended = True
    finally:
        if not ended:
            stop()
        _exited(proc, block=True)
        with lock:
            _kill_tree(proc)
            state["code"] = proc.wait()
        for stream in (proc.stdin, proc.stdout):
            with contextlib.suppress(Exception):
                stream.close()
    code, reason = state["code"], state.get("reason")
    if CANCEL.is_set() or reason == "cancelled":
        result = {"ok": False, "error": "cancelled"}
    elif reason:
        result = {"ok": False, "error": reason}
    elif not result:
        result = {"ok": False, "error": f"the builder exited unexpectedly (code {code})"}
    result["ok"] = bool(result.get("ok")) and code == 0
    return result


def _build_plan(job: dict, folder: str, step) -> dict:
    """Runs the builder on a job, again without stripping the weights if TensorRT can't strip them."""
    for strip in (True, False):
        with open(os.path.join(folder, "job.json"), "w", encoding="utf-8") as f:
            json.dump({**job, "strip": strip}, f)
        result = run_builder(os.path.join(folder, "job.json"), step)
        result["stripped"] = strip
        if not result.get("retry_without_strip"):
            return result
        # Without stripping, the plan holds placeholder weights (GBs) but runs the same way.
        log("Building a weight-stripped engine failed; building one that stores placeholder weights instead.")
    return result


SHALLOW_SHARE = 0.15  # of each engine's build progress, for its shallow engine


def build(arch: Arch, patcher, todo: list, optimization_level=None, progress=None, mode=store.PER_PROFILE, profiles=None) -> dict:
    """Exports the UNet graphs once and builds the engines for the (index, Profile)s in `todo`, each in a
    builder process of its own: per-profile, an engine for each, one at a time, so a build only ever holds
    one profile; merged, one engine holding them all. Each engine gets a shallow engine for Speed Hacks
    (speed.py) beside it; if that one can't be built, the engine is kept without it. Then, if the VAE's
    engines (vae.py) don't cover `profiles` (all of the architecture's), it builds those. Returns the last
    builder result, with "records": the engines saved, those before a failure or cancel included,
    "hack_errors": why shallow engines couldn't be built, and "vae": the VAE engines' Record, or
    "vae_error": why they couldn't be built."""
    from . import runtime
    from . import vae as vae_mod

    common.trt_version()  # installed and new enough
    runtime.release_all()
    common.inference_mode(common.mm().unload_all_models)()
    common.mm().soft_empty_cache(force=True)
    folder = store.work_dir(arch)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    records, result, warnings, hack_errors = [], {"ok": True}, 0, []
    before = {r.engine_key: r for r in store.engines(arch.id)}  # what these builds replace
    vae_part = vae_mod.needs_build(arch, profiles) if profiles else ""
    merged = mode == store.MERGED
    groups = ([todo] if todo else []) if merged else [[item] for item in todo]  # the profiles of each engine
    total = len(groups) + (1 if vae_part else 0)  # parts of the build's progress
    vae_record = vae_error = None
    try:
        graphs = shapes = None
        if groups:
            if progress:
                progress(0.0, "Exporting the UNet graphs")
            graphs = export_onnx(arch, patcher, folder)
            shapes = graphs["unet"]
        for n, group in enumerate(groups):
            if CANCEL.is_set():
                result = {"ok": False, "error": "cancelled"}
                break
            start = time.time()
            if merged:
                what, detail = f"merged engine ({len(group)} profiles)", "; ".join(p.describe() for _, p in group)
            else:
                what, detail = f"engine for profile {group[0][0] + 1}", f"{group[0][1].describe()} ({n + 1} of {len(groups)})"
            step = shallow_step = None
            if progress:
                title = (f"Merged engine, {len(group)} profiles" if merged else f"Profile {group[0][0] + 1} ({n + 1} of {len(groups)} to build)")
                share = 1 - SHALLOW_SHARE if graphs["shallow"] is not None else 1
                step = lambda f, t, n=n, title=title, share=share: progress((n + share * f) / total, f"{title}: {t}")
                shallow_step = lambda f, t, n=n, title=title: progress((n + 1 - SHALLOW_SHARE + SHALLOW_SHARE * f) / total, f"{title}, Speed Hacks part: {t}")
            unet_shapes = [p.unet_shapes(arch.in_channels, arch.context_dim, arch.adm_channels) for _, p in group]
            job = {
                "onnx": os.path.join(folder, "unet.onnx"),
                "plan": os.path.join(folder, "engine.plan"),
                "weights": sorted(shapes),
                "profiles": unet_shapes,
                "timing_cache": store.timing_cache_path(),
                "optimization_level": optimization_level,
            }
            log(f"Building the {arch.label} {what}, {detail}; the first build takes several minutes.")
            result = _build_plan(job, folder, step)
            warnings += result.get("memory_warnings") or 0
            if not result["ok"]:
                break
            hacks = shallow_plan = hack_error = None
            if graphs["shallow"] is not None:
                shallow_job = {
                    **job,
                    "onnx": os.path.join(folder, "shallow.onnx"),
                    "plan": os.path.join(folder, "shallow.plan"),
                    "weights": sorted(graphs["shallow"]),
                    "profiles": [speed.shallow_shapes(s, graphs["deep"]) for s in unet_shapes],
                }
                log(f"Building its Speed Hacks part, the shallow UNet ({len(shallow_job['weights'])} of {len(job['weights'])} weights).")
                shallow = _build_plan(shallow_job, folder, shallow_step)
                warnings += shallow.get("memory_warnings") or 0
                if shallow.get("error") == "cancelled" or CANCEL.is_set():
                    result = {"ok": False, "error": "cancelled"}
                    break
                if shallow["ok"]:
                    shallow_plan = shallow_job["plan"]
                    hacks = {
                        "deep": graphs["deep"],
                        "shapes": shallow_job["profiles"],
                        "weights": shallow_job["weights"],
                        "weight_bytes": sum(math.prod(graphs["shallow"][w]) * 2 for w in shallow_job["weights"]),
                        "stripped": shallow["stripped"],
                    }
                else:
                    hack_error = str(shallow.get("error"))
                    log(f"Building the Speed Hacks part failed ({hack_error}); keeping the engine without it.")
            else:
                hack_error = "the shallow UNet graph couldn't be exported"
            if hack_error:
                hack_errors.append(hack_error)
            if merged:
                meta = {"mode": store.MERGED, "profiles": [p.to_dict() for _, p in group], "indexes": [i for i, _ in group], "shapes": unet_shapes}
            else:
                meta = {"mode": store.PER_PROFILE, "profile": group[0][1].to_dict(), "index": group[0][0], "shapes": unet_shapes[0]}
            meta.update({
                "weights": job["weights"],
                "weight_bytes": sum(math.prod(s) * 2 for s in shapes.values()),
                "stripped": result["stripped"],
                "trt": common.trt_version(),
                "gpu": common.gpu_tag(),
                "gpu_name": torch.cuda.get_device_name(),
                "created": time.strftime("%Y-%m-%d %H:%M"),
            })
            if hacks:
                meta["speed_hacks"] = hacks
            else:
                meta["speed_hacks_error"] = hack_error  # built since Speed Hacks, without them: not outdated
                replaced = before.get(store.merged_key([p for _, p in group]) if merged else store.profile_key(group[0][1]))
                meta["speed_hacks_retried"] = replaced is not None and "speed_hacks_error" in replaced.meta
            records.append(store.save(arch, job["plan"], meta, shallow_plan))
            log(f"Built {records[-1].describe()} in {time.time() - start:.0f}s")
        if vae_part and result["ok"] and not CANCEL.is_set():
            start = time.time()
            log(f"Building the {arch.label} engines for {vae_part}.")
            step = (lambda f, t: progress((len(groups) + f) / total, f"VAE: {t}")) if progress else None
            vae_record, vae_error, skipped = vae_mod.build(arch, profiles, folder, optimization_level, step)
            warnings += skipped
            if vae_record is not None:
                log(f"Built the VAE's engines in {time.time() - start:.0f}s")
            elif vae_error == "cancelled" or CANCEL.is_set():
                result = {"ok": False, "error": "cancelled"}
            else:
                log(f"Building the VAE's engines failed ({vae_error}); the VAE runs on PyTorch.")
        return {**result, "records": records, "memory_warnings": warnings, "hack_errors": hack_errors, "vae": vae_record, "vae_error": vae_error}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
