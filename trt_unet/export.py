"""Exporting an architecture to a weightless engine: trace a fresh FP16 UNet to an ONNX graph whose
weights are placeholders, then build it in builder.py's process."""

import contextlib
import gc
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

from . import common, store
from .arch import Arch, unet_config
from .common import log

INPUTS = ("sample", "timesteps", "encoder_hidden_states", "y")
CANCEL = threading.Event()
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


def _placeholder_weights(onnx_path: str, input_names) -> dict:
    """Turns the exported weight inputs into initializers that all read one shared block of zeros, so
    the graph stays small: TensorRT only needs their shapes, the checkpoint provides the values."""
    import onnx

    model = onnx.load(onnx_path)
    graph = model.graph
    used = {name for node in graph.node for name in node.input}
    weights = [i for i in graph.input if i.name not in input_names]
    sizes = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.FLOAT: 4, onnx.TensorProto.INT64: 8}
    data_name = os.path.basename(onnx_path) + ".zeros"
    shapes, largest = {}, 0
    for value in weights:
        graph.input.remove(value)
        if value.name not in used:
            continue
        t = value.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        nbytes = math.prod(dims) * sizes[t.elem_type]
        largest = max(largest, nbytes)
        init = graph.initializer.add()
        init.name, init.data_type, init.data_location = value.name, t.elem_type, onnx.TensorProto.EXTERNAL
        init.dims.extend(dims)
        for key, val in (("location", data_name), ("offset", "0"), ("length", str(nbytes))):
            entry = init.external_data.add()
            entry.key, entry.value = key, val
        shapes[value.name] = dims
    with open(os.path.join(os.path.dirname(onnx_path), data_name), "wb") as f:
        f.truncate(largest)
    onnx.save(model, onnx_path)
    return shapes


@common.inference_mode
def export_onnx(arch: Arch, patcher, folder: str, device=None, dtype=torch.float16) -> dict:
    """Writes folder/unet.onnx; returns {weight name: shape}."""
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
    path = os.path.join(folder, "unet.onnx")
    kwargs = dict(
        export_params=False,
        opset_version=17,
        do_constant_folding=False,
        input_names=names,
        output_names=["latent"],
        dynamic_axes={k: v for k, v in DYNAMIC_AXES.items() if k in names or k == "latent"},
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    log(f"Exporting the {arch.family} UNet graph...")
    with _traceable_attention():
        torch.onnx.export(unet, tuple(inputs), path, **kwargs)
    expected = {k: list(v.shape) for k, v in unet.state_dict(keep_vars=True).items()}
    del unet, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shapes = _placeholder_weights(path, names)
    loaded = {k: list(v.shape) for k, v in patcher.model.diffusion_model.state_dict(keep_vars=True).items()}
    wrong = [k for k, s in shapes.items() if expected.get(k) != s or loaded.get(k) != s]
    if wrong:
        raise RuntimeError(f"{len(wrong)} exported weights don't match the loaded UNet (e.g. {wrong[0]})")
    return shapes


def _builder_env() -> dict:
    """Unbuffered UTF-8 output, and the WebUI's GPU as the only visible device."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
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


def run_builder(job_path: str, progress=None) -> dict:
    from . import builder

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(builder.__file__), job_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_builder_env(), text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    result = {}

    def cancel_watch():
        while proc.poll() is None:
            if CANCEL.wait(0.5):
                proc.kill()
                return

    threading.Thread(target=cancel_watch, daemon=True).start()
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
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout):
            with contextlib.suppress(Exception):
                stream.close()
    if CANCEL.is_set():
        result = {"ok": False, "error": "cancelled"}
    elif not result:
        result = {"ok": False, "error": f"the builder exited unexpectedly (code {code})"}
    result["ok"] = bool(result.get("ok")) and code == 0
    return result


def build(arch: Arch, patcher, profile_list: list, optimization_level=None, progress=None) -> dict:
    """Exports and builds an engine for `arch` with these profiles. Returns the builder's result plus the
    saved Record as "record"."""
    from . import runtime

    runtime.release_all()
    common.inference_mode(common.mm().unload_all_models)()
    common.mm().soft_empty_cache(force=True)
    folder = store.work_dir(arch)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    try:
        if progress:
            progress(0.0, "Exporting the UNet graph")
        start = time.time()
        shapes = export_onnx(arch, patcher, folder)
        if CANCEL.is_set():
            return {"ok": False, "error": "cancelled"}
        profile_shapes = [p.unet_shapes(arch.in_channels, arch.context_dim, arch.adm_channels) for p in profile_list]
        job = {
            "onnx": os.path.join(folder, "unet.onnx"),
            "plan": os.path.join(folder, "engine.plan"),
            "weights": sorted(shapes),
            "profiles": profile_shapes,
            "timing_cache": store.timing_cache_path(),
            "optimization_level": optimization_level,
        }
        with open(os.path.join(folder, "job.json"), "w", encoding="utf-8") as f:
            json.dump(job, f)
        log(f"Building the {arch.label} engine ({len(profile_list)} profiles); the first build takes several minutes.")
        result = run_builder(os.path.join(folder, "job.json"), progress)
        if result["ok"]:
            meta = {
                "profiles": [p.to_dict() for p in profile_list],
                "shapes": profile_shapes,
                "weights": job["weights"],
                "weight_bytes": sum(math.prod(s) * 2 for s in shapes.values()),
                "trt": common.trt_version(),
                "gpu": common.gpu_tag(),
                "gpu_name": torch.cuda.get_device_name(),
                "created": time.strftime("%Y-%m-%d %H:%M"),
            }
            result["record"] = store.save(arch, job["plan"], meta)
            log(f"Built {result['record'].describe()} in {time.time() - start:.0f}s")
        return result
    finally:
        shutil.rmtree(folder, ignore_errors=True)
