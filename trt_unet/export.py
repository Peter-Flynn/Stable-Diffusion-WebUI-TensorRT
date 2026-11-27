import contextlib
import copy
import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
import warnings

import torch
import torch.nn.functional as F

from . import common, dit, dit_speed, fresh, quant, speed, store
from .arch import Arch, unet_config
from .common import duration, gib, logger

INPUTS = ("sample", "timesteps", "encoder_hidden_states", "y")
PLACEHOLDER_STRIDE = 256
CANCEL = threading.Event()
MEMORY_RESERVE = 2 * 2**30
DYNAMIC_AXES = {
    "sample": {0: "batch", 2: "height", 3: "width"},
    "timesteps": {0: "batch"},
    "encoder_hidden_states": {0: "batch", 1: "context"},
    "y": {0: "batch"},
    "latent": {0: "batch", 2: "height", 3: "width"},
}
# Constant folding would rename the weights. dynamo=False keeps the TorchScript exporter, no longer the default in
# newer PyTorch.
ONNX_KWARGS = dict(export_params=False, opset_version=17, do_constant_folding=False)
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    ONNX_KWARGS["dynamo"] = False


@contextlib.contextmanager
def quiet_warnings(seen: set = None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for message in dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in caught):
                if seen is not None:
                    if message in seen:
                        continue
                    seen.add(message)
                logger.debug(f"While exporting: {message}")


@contextlib.contextmanager
def fast_weightless_export():
    # With export_params=False and randomized weights, TorchScript ONNX export spends tens of seconds hashing all
    # weight tensors to deduplicate identical ones and copying them into shape-type inference even though weight inputs
    # already carry their shapes and dtypes. Strip floating-point weights from those passes; keep integer buffers.
    c = getattr(torch, "_C", None)
    orig_dedup = getattr(c, "_jit_pass_onnx_deduplicate_initializers", None)
    orig_sti = getattr(c, "_jit_pass_onnx_graph_shape_type_inference", None)
    patched = []
    try:
        if orig_dedup is not None:
            c._jit_pass_onnx_deduplicate_initializers = lambda *args, **kwargs: None
            patched.append(("_jit_pass_onnx_deduplicate_initializers", orig_dedup))
        if orig_sti is not None:
            def _shape_type_inference(graph, params_dict, *args, **kwargs):
                if isinstance(params_dict, dict):
                    params_dict = {k: v for k, v in params_dict.items() if not (isinstance(v, torch.Tensor) and v.is_floating_point())}
                return orig_sti(graph, params_dict, *args, **kwargs)
            c._jit_pass_onnx_graph_shape_type_inference = _shape_type_inference
            patched.append(("_jit_pass_onnx_graph_shape_type_inference", orig_sti))
    except Exception:
        for name, orig in patched:
            with contextlib.suppress(Exception):
                setattr(c, name, orig)
        patched.clear()
    try:
        yield
    finally:
        for name, orig in patched:
            with contextlib.suppress(Exception):
                setattr(c, name, orig)


def _attention(q, k, v, mask=None, upcast=False, scale=None):
    # Exported as plain matmul + softmax, which TensorRT fuses. q and k are each multiplied by sqrt(scale) rather than
    # the product by scale, so q.k can't overflow FP16.
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


_ORIG_GROUP_NORM = F.group_norm


def _fp32_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    # TensorRT 11's Myelin __myl_Gnor / __myl_GnorSilu kernels corrupt dynamic GroupNorm (in FP16, BF16, and FP32) when
    # C <= 640 and max spatial size is small (e.g. SDXL input_blocks.7.0 in *-1024 profiles, SD 1.5 in 256-768),
    # reading uninitialized activation memory (NaN or large artifacts). Decomposing into explicit FP32 reductions avoids
    # __myl_Gnor completely and fuses Cast + GroupNorm + SiLU into a faster Myelin kernel.
    if input.ndim < 3:
        return _ORIG_GROUP_NORM(input, num_groups, weight, bias, eps)
    y = input.float().reshape(input.shape[0], num_groups, -1)
    mean = y.mean(dim=-1, keepdim=True)
    diff = y - mean
    var = (diff * diff).mean(dim=-1, keepdim=True)
    y = (diff * torch.rsqrt(var + eps)).reshape_as(input)
    axes = (0,) + (-1,) * (input.ndim - 2)
    if weight is not None:
        w = weight
        for axis in axes:
            w = w.unsqueeze(axis)
        y = y * w.float()
    if bias is not None:
        b = bias
        for axis in axes:
            b = b.unsqueeze(axis)
        y = y + b.float()
    return y.to(input.dtype)


@contextlib.contextmanager
def _traceable_attention():
    from . import compat

    swaps = [(F, "scaled_dot_product_attention", _sdpa), (F, "group_norm", _fp32_group_norm)]
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
    for owner, name, fn in swaps:
        setattr(owner, name, fn)
    try:
        with compat.patch_attention(_optimized_attention):
            yield
    finally:
        for owner, name, fn in originals:
            setattr(owner, name, fn)


def _fresh_unet(config: dict, device, dtype=torch.float16):
    # Random weights: tracing doesn't depend on values, and identical tensors (e.g. all-zero) get merged by the exporter,
    # losing weight names.
    from . import compat

    unet = compat.fresh_unet(config, device, dtype)
    for t in unet.state_dict(keep_vars=True).values():
        if t.is_floating_point():
            t.normal_(0, 0.02)
    return unet


def _fresh_dit(kind: str, config: dict, device, dtype=torch.bfloat16):
    from . import compat

    model = compat.fresh_dit(kind, config, device, dtype)
    is_cuda = torch.device(device).type == "cuda"
    for idx, t in enumerate(model.state_dict(keep_vars=True).values()):
        if t.is_floating_point():
            if is_cuda:
                t.normal_(0, 0.02)
            else:
                # Distinct finite non-zero fill avoids multi-second CPU PRNG on 5B-parameter models while keeping
                # weight identities distinct during TorchScript tracing.
                t.fill_(0.01 + (idx % 997) * 1e-4)
    return model


def _weightless_graph(
    onnx_path: str,
    input_names,
    agree_with,
    quiet: bool = False,
    *,
    quantize_precision: str = None,
    act_dtype: torch.dtype = None,
    weight_bytes_out: list = None,
) -> dict:
    # Turns the weight inputs into initializers backed by one small sparse file of zeros: TensorRT needs only their
    # shapes, and refitting supplies the values. The parser maps that file once and TensorRT identifies weights by
    # their address, so each starts PLACEHOLDER_STRIDE bytes after the previous; their ranges overlap, which keeps the
    # file about as large as the largest weight.
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
        raise RuntimeError(f"{len(wrong)} exported weights don't match the loaded model's (e.g. {wrong[0]})")
    start = time.time()
    try:
        tidied = copy.deepcopy(model)
        before, after = graph_mod.tidy(tidied, shapes)
        model, graph = tidied, tidied.graph
        if not quiet:
            logger.debug(f"Tidied {os.path.basename(onnx_path)} from {before:,} to {after:,} nodes in {duration(time.time() - start)}.")
    except Exception as e:
        logger.warning(f"Couldn't tidy {os.path.basename(onnx_path)}, so it's built as exported: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
    used = {name for node in graph.node for name in node.input}
    for value in [i for i in graph.input if i.name in shapes and i.name not in used]:  # unused weights can't be marked refittable
        graph.input.remove(value)
        del shapes[value.name]

    if quantize_precision and quant.is_quantized_precision(quantize_precision):
        quant.quantize_onnx_graph(model, quantize_precision, act_dtype or torch.bfloat16, shapes=shapes)

    data_name = os.path.basename(onnx_path) + ".zeros"
    end = 0
    total_bytes = 0
    weight_inputs = [i for i in graph.input if i.name in shapes]
    regular_inputs = [v for v in weight_inputs if not quant.is_any_scale_weight(v.name)]
    scale_inputs = [v for v in weight_inputs if quant.is_any_scale_weight(v.name)]

    def _add_external_init(value, offset: int, nbytes: int, dims: list):
        graph.input.remove(value)
        init = graph.initializer.add()
        init.name, init.data_type, init.data_location = value.name, value.type.tensor_type.elem_type, onnx.TensorProto.EXTERNAL
        init.dims.extend(dims)
        for key, val in (("location", data_name), ("offset", str(offset)), ("length", str(nbytes))):
            entry = init.external_data.add()
            entry.key, entry.value = key, val

    for index, value in enumerate(regular_inputs):
        t = value.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]  # the graph's shape, which can differ from the checkpoint's (see graph._unsqueezed_weights)
        nbytes = quant.onnx_tensor_nbytes(t.elem_type, dims)
        total_bytes += nbytes
        offset = index * PLACEHOLDER_STRIDE
        end = max(end, offset + nbytes)
        _add_external_init(value, offset, nbytes, dims)

    # TensorRT's ONNX parser validates that DequantizeLinear scale initializers are strictly positive at parse time;
    # place each scale in a non-overlapping tail slice filled with distinct positive dummy values so TRT parses
    # cleanly and never merges/deduplicates refittable scale tensors by value.
    _elem_to_torch = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.BFLOAT16: torch.bfloat16,
    }
    scale_writes = []
    for s_idx, value in enumerate(scale_inputs):
        t = value.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        nbytes = quant.onnx_tensor_nbytes(t.elem_type, dims)
        total_bytes += nbytes
        offset = (end + 15) & ~15
        end = offset + nbytes
        _add_external_init(value, offset, nbytes, dims)
        dt = _elem_to_torch[t.elem_type]
        if dt == torch.bfloat16:
            v = torch.tensor(0x3F80 + (s_idx % 1024) + 1, dtype=torch.uint16).view(torch.bfloat16)
        elif dt == torch.float16:
            v = torch.tensor(0x3C00 + (s_idx % 1024) + 1, dtype=torch.uint16).view(torch.float16)
        else:
            v = torch.tensor(1.0 + (s_idx + 1) * 1e-4, dtype=torch.float32)
        buf = torch.ones(dims, dtype=dt).reshape(-1)
        if buf.numel() >= 1:
            buf[0] = v
        if buf.numel() >= 2:
            buf[1] = 1.0 + ((s_idx // 1024) + 1) * 0.015625
        scale_writes.append((offset, buf.contiguous().view(torch.uint8).numpy().tobytes()))

    with open(os.path.join(os.path.dirname(onnx_path), data_name), "wb") as f:
        f.truncate(end)
        for offset, raw in scale_writes:
            f.seek(offset)
            f.write(raw)
    onnx.save(model, onnx_path)
    if weight_bytes_out is not None:
        weight_bytes_out.append(total_bytes)
    return shapes


def _as_full(unet, x, timesteps, context, y=None):
    return speed.full_forward(unet, x, timesteps, context, y)


def _as_shallow(unet, x, timesteps, context, *rest):
    rest, tracked = rest[: -len(fresh.INPUTS)], dict(zip(fresh.INPUTS, rest[-len(fresh.INPUTS):]))
    y, deep = rest if len(rest) == 2 else (None, rest[0])
    return fresh.shallow(lambda sample, t: speed.shallow_forward(unet, sample, t, context, y, deep), x, timesteps, tracked)


def _dit_trace_device(patcher, requested_device):
    dev = torch.device(requested_device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return dev
    dm = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    est_w16 = quant.param_count(dm) * 2
    try:
        free_vram = int(torch.cuda.mem_get_info(dev)[0])
    except Exception:
        free_vram = 0
    # 5B BF16 models (~10 GiB) exceed 8-12 GB GPUs during tracing; tracing on CPU uses RAM and takes ~2-4s on tiny shapes.
    if est_w16 and free_vram < est_w16 + (3 << 30):
        logger.debug(f"Tracing DiT ONNX on CPU ({gib(est_w16)} weights vs {gib(free_vram)} free VRAM).")
        return torch.device("cpu")
    return dev


def _export_dit_onnx(arch: Arch, patcher, folder: str, device, precision: str) -> dict:
    act_dtype = quant.compute_dtype(precision)
    fp32_res = quant.uses_fp32_residual(precision)
    trace_dev = _dit_trace_device(patcher, device)
    model = _fresh_dit(arch.kind, unet_config(patcher), trace_dev, act_dtype)
    paths = {"unet": os.path.join(folder, "unet.onnx"), "shallow": os.path.join(folder, "shallow.onnx")}
    use_shallow = bool(dit_speed.USES_SHALLOW_ENGINE)

    if arch.kind == "anima":
        head_dim = int(model.blocks[0].self_attn.head_dim)
        h_p, w_p = 4, 4
        names = ["sample", "timesteps", "encoder_hidden_states", "rope"]
        inputs = (
            torch.randn(2, arch.in_channels, 1, h_p * 2, w_p * 2, dtype=act_dtype, device=trace_dev),
            torch.rand(2, device=trace_dev, dtype=torch.float32),
            torch.randn(2, 8, arch.context_dim, dtype=act_dtype, device=trace_dev),
            torch.randn(1, h_p, w_p, 1, head_dim // 2, 2, dtype=torch.float32, device=trace_dev),
        )
        axes = {
            "sample": {0: "batch", 3: "height", 4: "width"},
            "timesteps": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "context"},
            "rope": {1: "patch_h", 2: "patch_w"},
            "latent": {0: "batch", 3: "height", 4: "width"},
        }
        full_fn = lambda m, x, t, c, r: dit.anima_traceable_forward(
            m, x, t, c, r, fp32_residual=fp32_res, return_trunk_delta=use_shallow, attn_fn=_optimized_attention
        )
        shallow_fn = lambda m, x, t, c, r, d: dit.anima_shallow_forward(
            m, x, t, c, r, d, fp32_residual=fp32_res, attn_fn=_optimized_attention
        )
        delta_dim = int(model.final_layer.hidden_size)
    elif arch.kind == "moss_image":
        head_dim = int(model.layers[0].q_norm.weight.shape[0])
        h_lat, w_lat, s_txt = 4, 4, 6
        names = ["sample", "timesteps", "encoder_hidden_states", "rope_img", "rope_txt"]
        inputs = (
            torch.randn(2, arch.in_channels, h_lat, w_lat, dtype=act_dtype, device=trace_dev),
            torch.rand(2, device=trace_dev, dtype=torch.float32),
            torch.randn(2, s_txt, arch.context_dim, dtype=act_dtype, device=trace_dev),
            torch.randn(1, h_lat, w_lat, 1, head_dim // 2, 2, dtype=torch.float32, device=trace_dev),
            torch.randn(1, s_txt, 1, head_dim // 2, 2, dtype=torch.float32, device=trace_dev),
        )
        axes = {
            "sample": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "context"},
            "rope_img": {1: "height", 2: "width"},
            "rope_txt": {1: "context"},
            "latent": {0: "batch", 2: "height", 3: "width"},
        }
        full_fn = lambda m, x, t, c, ri, rt: dit.moss_traceable_forward(
            m, x, t, c, ri, rt, fp32_residual=fp32_res, return_trunk_delta=use_shallow, attn_fn=_optimized_attention
        )
        shallow_fn = lambda m, x, t, c, ri, rt, d: dit.moss_shallow_forward(
            m, x, t, c, ri, rt, d, fp32_residual=fp32_res, attn_fn=_optimized_attention
        )
        delta_dim = int(model.final_norm_video.weight.shape[0])
    else:
        raise ValueError(f"Unsupported DiT kind {arch.kind!r}")

    logger.info(f"Exporting the {arch.name} DiT to ONNX ({quant.LABELS.get(precision, precision)}).")
    shallow_exported = False
    delta_example = None
    out_names = ["latent", "trunk_delta"] if use_shallow else ["latent"]
    if use_shallow:
        axes["trunk_delta"] = {0: "batch", 1: "img_tokens"}
    with _traceable_attention(), fast_weightless_export(), quiet_warnings():
        try:
            model.forward = types.MethodType(full_fn, model)
            if use_shallow:
                delta_example = full_fn(model, *inputs)[1]
            torch.onnx.export(model, inputs, paths["unet"], input_names=names, output_names=out_names, dynamic_axes=axes, **ONNX_KWARGS)
            if use_shallow and delta_example is not None:
                model.forward = types.MethodType(shallow_fn, model)
                torch.onnx.export(
                    model,
                    (*inputs, delta_example),
                    paths["shallow"],
                    input_names=names + ["trunk_delta"],
                    output_names=["latent"],
                    dynamic_axes={k: v for k, v in axes.items()},
                    **ONNX_KWARGS,
                )
                shallow_exported = True
        finally:
            with contextlib.suppress(AttributeError):
                del model.forward

    exported = {k: list(v.shape) for k, v in model.state_dict(keep_vars=True).items()}
    del model, inputs, delta_example
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dm = patcher.model.diffusion_model
    loaded = {k: list(v.shape) for k, v in {**dm.state_dict(keep_vars=True), **dict(dm.named_parameters())}.items()}
    wb_full = []
    full_shapes = _weightless_graph(
        paths["unet"], names, (exported, loaded), quantize_precision=precision, act_dtype=act_dtype, weight_bytes_out=wb_full
    )
    shallow_shapes = None
    wb_shallow = []
    if shallow_exported:
        shallow_shapes = _weightless_graph(
            paths["shallow"], names + ["trunk_delta"], (exported, loaded), quantize_precision=precision, act_dtype=act_dtype, weight_bytes_out=wb_shallow
        )
    return {
        "unet": full_shapes,
        "shallow": shallow_shapes,
        "deep": {"delta_dim": delta_dim} if shallow_exported else None,
        "weight_bytes": wb_full[0] if wb_full else sum(math.prod(s) * 2 for s in full_shapes.values()),
        "shallow_weight_bytes": wb_shallow[0] if wb_shallow else 0,
        "precision": precision,
    }


@common.inference_mode
def export_onnx(arch: Arch, patcher, folder: str, device=None, dtype=torch.float16, precision: str = None) -> dict:
    device = device or common.device()
    if arch.is_dit:
        prec = precision or store.active_precision(arch, patcher)
        return _export_dit_onnx(arch, patcher, folder, device, prec)
    unet = _fresh_unet(unet_config(patcher), device, dtype)
    names = [n for n in INPUTS if n != "y" or arch.adm_channels]
    inputs = [
        torch.randn(2, arch.in_channels, 64, 64, dtype=dtype, device=device),
        torch.rand(2, device=device) * 999,  # FP32: FP16 can't represent fractional timesteps near 1000
        torch.randn(2, 77, arch.context_dim, dtype=dtype, device=device),
    ]
    if arch.adm_channels:
        inputs.append(torch.randn(2, arch.adm_channels, dtype=dtype, device=device))
    paths = {"unet": os.path.join(folder, "unet.onnx"), "shallow": os.path.join(folder, "shallow.onnx")}
    axes = {k: v for k, v in DYNAMIC_AXES.items() if k in names or k == "latent"}
    logger.info(f"Exporting the {arch.name} UNet to ONNX.")
    shallow = True
    with _traceable_attention(), fast_weightless_export(), quiet_warnings():
        try:
            deep = speed.full_forward(unet, *inputs)[1]
            scale = inputs[0].shape[2] // deep.shape[2]
            axes["deep"] = {0: "batch", 2: "height", 3: "width"} if scale == 1 else {0: "batch", 2: "deep_height", 3: "deep_width"}
            info = {"channels": int(deep.shape[1]), "scale": int(scale), "branch": speed.BRANCH, "latent_channels": int(unet.out_channels)}
            unet.forward = types.MethodType(_as_full, unet)  # on the UNet itself so the weights keep their names
            torch.onnx.export(unet, tuple(inputs), paths["unet"], input_names=names, output_names=["latent", "deep"], dynamic_axes=axes, **ONNX_KWARGS)
        except Exception as e:
            logger.warning(f"Couldn't export the UNet for Speed Hacks, so these engines are built without them: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
            shallow, deep, info = False, None, None
            with contextlib.suppress(AttributeError):
                del unet.forward
            axes.pop("deep", None)
            torch.onnx.export(unet, tuple(inputs), paths["unet"], input_names=names, output_names=["latent"], dynamic_axes=axes, **ONNX_KWARGS)
        try:
            if shallow:
                unet.forward = types.MethodType(_as_shallow, unet)
                torch.onnx.export(
                    unet, (*inputs, deep, *fresh.example_inputs(inputs[0], info["latent_channels"])), paths["shallow"], input_names=names + ["deep", *fresh.INPUTS],
                    output_names=fresh.OUTPUTS, dynamic_axes={**axes, **fresh.dynamic_axes()}, **ONNX_KWARGS,
                )
        except Exception as e:
            logger.warning(f"Couldn't export the UNet's Speed Hacks part, so these engines are built without it: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
            shallow = False
        finally:
            with contextlib.suppress(AttributeError):
                del unet.forward
    exported = {k: list(v.shape) for k, v in unet.state_dict(keep_vars=True).items()}
    del unet, inputs, deep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dm = patcher.model.diffusion_model
    loaded = {k: list(v.shape) for k, v in {**dm.state_dict(keep_vars=True), **dict(dm.named_parameters())}.items()}
    result = {"unet": _weightless_graph(paths["unet"], names, (exported, loaded)), "shallow": None, "deep": info}
    if shallow:
        try:
            result["shallow"] = _weightless_graph(paths["shallow"], names + ["deep", *fresh.INPUTS], (exported, loaded))
        except Exception as e:
            logger.warning(f"Couldn't prepare the UNet's Speed Hacks part, so these engines are built without it: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
    return result


def _builder_env() -> dict:
    # Drops tcmalloc from LD_PRELOAD (webui.sh preloads it): it's slow to return the memory a build frees to the system.
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    preload = [p for p in re.split(r"[\s:]+", env.get("LD_PRELOAD", "")) if p]
    if any("tcmalloc" in os.path.basename(p) for p in preload):
        preload = [p for p in preload if "tcmalloc" not in os.path.basename(p)]
        if preload:
            env["LD_PRELOAD"] = ":".join(preload)
        else:
            del env["LD_PRELOAD"]
    try:
        index = common.device().index
        if index is None:
            return env
    except Exception:
        return env
    visible = [d.strip() for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    elif index < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[index]
    return env


def _memory_short():
    try:
        from . import refits

        available, swap, total = refits.memory()
    except Exception:
        return None
    free = available + swap
    if free >= max(MEMORY_RESERVE, total // 20):
        return None
    return f"Stopped before the system ran out of memory ({gib(free)} of RAM and swap left). Build fewer or narrower profiles, or close other programs."


def _exited(proc, block=False) -> bool:
    # WNOWAIT: doesn't reap the builder, so its PID, which is also its process group's ID, can't be reused yet, and
    # os.killpg(proc.pid) can still safely kill whatever it started.
    if os.name == "nt":
        return (proc.wait() if block else proc.poll()) is not None
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)) is not None
    except ChildProcessError:
        return True


def _kill_tree(proc):
    if os.name == "nt":
        with contextlib.suppress(OSError):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with contextlib.suppress(OSError):
            proc.kill()
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


def run_builder(job_path: str, progress=None) -> dict:
    from . import builder

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(builder.__file__), job_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_builder_env(), text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=os.name != "nt",
    )
    result, state, lock = {}, {}, threading.Lock()

    def stop(reason=None):
        with lock:
            if "code" not in state:
                if reason:
                    state.setdefault("reason", reason)
                _kill_tree(proc)

    def watch():
        while "code" not in state:
            if CANCEL.wait(0.5):
                return stop("cancelled")
            if _exited(proc):
                if os.name != "nt":
                    stop()
                return
            short = _memory_short()
            if short:
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
            elif line.startswith(builder.LOG):
                with contextlib.suppress(Exception):
                    message = json.loads(line[len(builder.LOG):])
                    logger.log(logging.getLevelName(message["level"]), message["message"])
            elif line.startswith(builder.RESULT):
                with contextlib.suppress(ValueError):
                    result = json.loads(line[len(builder.RESULT):])
            elif line.strip():
                logger.debug(f"Builder: {line.rstrip()}")
        ended = True
    finally:
        if not ended:
            stop()
        _exited(proc, block=True)
        with lock:
            if os.name != "nt":
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


def _build_plan(job: dict, folder: str, step, strip: bool = True) -> dict:
    for s in ((True, False) if strip else (False,)):
        with open(os.path.join(folder, "job.json"), "w", encoding="utf-8") as f:
            json.dump({**job, "strip": s}, f)
        result = run_builder(os.path.join(folder, "job.json"), step)
        result["stripped"] = s
        if not result.get("retry_without_strip"):
            return result
        logger.info("TensorRT couldn't leave the weights out of the engine, so it's building one with placeholder weights: a larger file, as fast.")
    return result


def _build_batch(batch_job: dict, folder: str, step) -> dict:
    job_path = os.path.join(folder, "batch.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(batch_job, f)
    return run_builder(job_path, step)


SHALLOW_SHARE = 0.15


def _dit_shallow_shapes(shapes: dict, delta_dim: int, kind: str) -> dict:
    if kind == "anima":
        delta = [[b, (h // 2) * (w // 2), delta_dim] for b, _, _, h, w in shapes["sample"]]
    else:
        delta = [[b, h * w, delta_dim] for b, _, h, w in shapes["sample"]]
    return {**shapes, "trunk_delta": delta}


def build(arch: Arch, patcher, todo: list, optimization_level=None, progress=None, mode=store.PER_PROFILE, profiles=None, precision: str = None) -> dict:
    from . import runtime
    from . import vae as vae_mod
    from . import vae_build

    if CANCEL.is_set():
        return {"ok": False, "error": "cancelled", "records": [], "memory_warnings": 0, "hack_errors": [], "vae": None, "vae_error": None}
    common.trt_version()
    runtime.release_all()
    common.inference_mode(common.mm().unload_all_models)()
    common.mm().soft_empty_cache(force=True)
    folder = store.work_dir(arch)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    prec = (precision or store.active_precision(arch, patcher)) if arch.is_dit else None
    records, result, memory_warnings, hack_errors = [], {"ok": True}, 0, []
    before = {r.engine_key: r for r in store.engines(arch.id, precision=prec)}
    vae_part = vae_mod.needs_build(arch, profiles) if profiles else ""
    merged = mode == store.MERGED
    groups = ([todo] if todo else []) if merged else [[item] for item in todo]
    total = len(groups) + (1 if vae_part else 0)
    vae_record = vae_error = None
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    expect_shallow = dit_speed.USES_SHALLOW_ENGINE if arch.is_dit else True
    try:
        graphs = shapes = None
        if groups:
            if progress:
                progress(0.0, f"Exporting the {arch.name} graphs" if arch.is_dit else "Exporting the UNet graphs")
            graphs = export_onnx(arch, patcher, folder, precision=prec)
            shapes = graphs["unet"]
        for n, group in enumerate(groups):
            if CANCEL.is_set():
                result = {"ok": False, "error": "cancelled"}
                break
            start = time.time()
            prec_note = f" [{quant.LABELS.get(prec, prec)}]" if arch.is_dit else ""
            what = f"merged {arch.name} engine{prec_note}" if merged else f"{arch.name} engine for profile {group[0][0] + 1}{prec_note}"
            step = shallow_step = None
            if progress:
                title = (f"Merged engine, {len(group)} profiles" if merged else f"Profile {group[0][0] + 1} ({n + 1} of {len(groups)} to build)")
                share = 1 - SHALLOW_SHARE if graphs["shallow"] is not None else 1
                step = lambda f, t, n=n, title=title, share=share: progress((n + share * f) / total, f"{title}: {t}")
                shallow_step = lambda f, t, n=n, title=title: progress((n + 1 - SHALLOW_SHARE + SHALLOW_SHARE * f) / total, f"{title}, Speed Hacks part: {t}")
            unet_shapes = [p.engine_shapes(arch, module) for _, p in group]
            job = {
                "onnx": os.path.join(folder, "unet.onnx"),
                "plan": os.path.join(folder, "engine.plan"),
                "weights": sorted(shapes),
                "profiles": unet_shapes,
                "timing_cache": store.timing_cache_path(),
                "optimization_level": optimization_level,
            }
            if merged:
                logger.info(f"Building the {what} for {len(group)} profiles: " + "; ".join(p.describe() for _, p in group))
            else:
                logger.info(f"Building the {what} ({n + 1} of {len(groups)}): {group[0][1].describe()}")
            result = _build_plan(job, folder, step)
            memory_warnings += result.get("memory_warnings") or 0
            if not result["ok"]:
                if result.get("error") != "cancelled":
                    logger.error(f"Couldn't build the {what}: {result.get('error')}")
                break
            hacks = shallow_plan = hack_error = None
            if expect_shallow:
                if graphs["shallow"] is not None:
                    if arch.is_dit:
                        shallow_profiles = [_dit_shallow_shapes(s, graphs["deep"]["delta_dim"], arch.kind) for s in unet_shapes]
                    else:
                        shallow_profiles = [{**speed.shallow_shapes(s, graphs["deep"]), **fresh.input_shapes(s, graphs["deep"]["latent_channels"])} for s in unet_shapes]
                    shallow_job = {
                        **job,
                        "onnx": os.path.join(folder, "shallow.onnx"),
                        "plan": os.path.join(folder, "shallow.plan"),
                        "weights": sorted(graphs["shallow"]),
                        "profiles": shallow_profiles,
                        "exact_fp32": not arch.is_dit,
                    }
                    logger.info(f"Building the {what}'s Speed Hacks part.")
                    shallow = _build_plan(shallow_job, folder, shallow_step)
                    memory_warnings += shallow.get("memory_warnings") or 0
                    if shallow.get("error") == "cancelled" or CANCEL.is_set():
                        result = {"ok": False, "error": "cancelled"}
                        break
                    if shallow["ok"]:
                        shallow_plan = shallow_job["plan"]
                        if arch.is_dit:
                            hacks = {
                                "deep": graphs["deep"],
                                "shapes": shallow_job["profiles"],
                                "weights": shallow_job["weights"],
                                "weight_bytes": int(graphs.get("shallow_weight_bytes") or 0),
                                "stripped": shallow["stripped"],
                                "dit_speed": dit_speed.SHALLOW_SIGNATURE,
                            }
                        else:
                            hacks = {
                                "deep": graphs["deep"],
                                "shapes": shallow_job["profiles"],
                                "weights": shallow_job["weights"],
                                "weight_bytes": sum(math.prod(graphs["shallow"][w]) * 2 for w in shallow_job["weights"]),
                                "stripped": shallow["stripped"],
                                "fresh": fresh.SIGNATURE,
                            }
                    else:
                        hack_error = str(shallow.get("error"))
                        logger.warning(f"Couldn't build the {what}'s Speed Hacks part, so it runs without Speed Hacks: {hack_error}")
                else:
                    hack_error = "the shallow graph couldn't be exported"
                if hack_error:
                    hack_errors.append(hack_error)
            if merged:
                meta = {"mode": store.MERGED, "profiles": [p.to_dict() for _, p in group], "indexes": [i for i, _ in group], "shapes": unet_shapes}
            else:
                meta = {"mode": store.PER_PROFILE, "profile": group[0][1].to_dict(), "index": group[0][0], "shapes": unet_shapes[0]}
            weight_bytes = int(graphs["weight_bytes"]) if arch.is_dit else sum(math.prod(s) * 2 for s in shapes.values())
            meta.update({
                "weights": job["weights"],
                "weight_bytes": weight_bytes,
                "stripped": result["stripped"],
                "graph": store.DIT_GRAPH_VERSION if arch.is_dit else store.GRAPH_VERSION,
                "trt": common.trt_version(),
                "gpu": common.gpu_tag(),
                "gpu_name": torch.cuda.get_device_name(common.device()),
                "created": time.strftime("%Y-%m-%d %H:%M"),
            })
            if arch.is_dit:
                meta["precision"] = prec
            if hacks:
                meta["speed_hacks"] = hacks
            elif expect_shallow:
                meta["speed_hacks_error"] = hack_error
                base_k = store.merged_key([p for _, p in group]) if merged else store.profile_key(group[0][1])
                replaced = before.get(f"{prec}-{base_k}" if arch.is_dit else base_k)
                meta["speed_hacks_retried"] = replaced is not None and "speed_hacks_error" in replaced.meta
            records.append(store.save(arch, job["plan"], meta, shallow_plan))
            logger.info(f"Built the {what} in {duration(time.time() - start)} ({gib(records[-1].disk_bytes)}).")
        if vae_part and result["ok"]:
            if CANCEL.is_set():
                result = {"ok": False, "error": "cancelled"}
            else:
                start = time.time()
                logger.info(f"Building the engines for {vae_part}.")
                step = (lambda f, t: progress((len(groups) + f) / total, f"VAE: {t}")) if progress else None
                vae_record, vae_error, skipped = vae_build.build(arch, profiles, folder, optimization_level, step)
                memory_warnings += skipped
                if vae_record is not None:
                    logger.info(f"Built the VAE engines in {duration(time.time() - start)}.")
                elif vae_error == "cancelled" or CANCEL.is_set():
                    result = {"ok": False, "error": "cancelled"}
        if memory_warnings:
            logger.info(f"TensorRT skipped {memory_warnings} kernels that needed more VRAM than was free, so the largest sizes may run a little slower.")
        return {**result, "records": records, "memory_warnings": memory_warnings, "hack_errors": hack_errors, "vae": vae_record, "vae_error": vae_error}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
