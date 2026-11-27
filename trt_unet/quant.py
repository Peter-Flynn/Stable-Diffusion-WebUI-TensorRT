# GPU precision capabilities, VRAM-aware auto-selection, ONNX weight/activation quantization, and on-the-fly GPU refit
# quantization for DiT architectures (Anima, MOSS-Image). Keeps checkpoint quantization formats (BF16, FP8, MXFP8,
# NVFP4, INT8-ConvRot, W4A4, W4A8, W6A8, GGUF + LoRA) completely orthogonal to engine execution precision.
import math

import torch
import torch.nn.functional as F

from . import common

AUTO = "auto"
BF16 = "bf16"
FP16 = "fp16"
FP8 = "fp8"
INT8 = "int8"
INT4 = "int4"

EXPLICIT_PRECISIONS = (BF16, FP8, INT8, INT4, FP16)
ALL_PRECISIONS = (AUTO, *EXPLICIT_PRECISIONS)
LABELS = {
    AUTO: "Auto",
    BF16: "BF16",
    FP8: "FP8",
    INT8: "INT8",
    INT4: "INT4",
    FP16: "FP16",
}
SCALE_SUFFIX = ".__trt_scale__"
ACT_SCALE_SUFFIX = ".__trt_act_scale__"
TRANSPOSE_SUFFIX = ".__trt_t__"
INT4_BLOCK_SIZE = 64
# Headroom left beyond engine weights for activations, CUDA context, and sampler state when auto-selecting precision.
AUTO_HEADROOM = int(3.2 * 2**30)
# Minimum weight matrix dimensions eligible for 8-bit/4-bit quantization (leaves embedders, norms, modulations exact).
MIN_QUANT_DIM = 512

# Role-based fallback activation scales calibrated on real Anima 2B & MOSS-Image 5B trajectories:
# - FP8 E4M3 is floating-point (range [-448, 448] at scale=1.0), so 2-tier constants cover 16 orders of magnitude
#   with zero calibration cost (0.25 -> [-112, 112] for small cross-attn/text outputs; 2.0 -> [-896, 896] for main blocks).
# - INT8 uses refittable per-site activation scales auto-calibrated on the loaded checkpoint; these role defaults are
#   used only as fallback if calibration is unavailable (e.g. unit tests without a full DiT module).
_FP8_SMALL_PATTERNS = ("cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj", "text_embedder.")
_INT8_ROLE_FALLBACKS = (
    ("cross_attn.output_proj", 0.01),
    ("cross_attn.k_proj", 0.02),
    ("cross_attn.v_proj", 0.02),
    ("text_embedder.blocks.", 0.15),
    ("cross_attn.q_proj", 2.0),
    ("self_attn.output_proj", 0.6),
    ("self_attn.", 2.0),
    (".proj.o", 1.0),
    (".proj.q", 1.8),
    (".proj.k", 1.8),
    (".proj.v", 1.8),
    ("ffn_up_gate", 0.3),
    ("ffn_down", 0.8),
    ("mlp.layer1", 0.5),
    ("mlp.layer2", 1.0),
)
_act_scale_cache = {}  # (arch_id, base_version_key, kind) -> dict[base_weight_name, float]


def is_scale_weight(name: str) -> bool:
    return name.endswith(SCALE_SUFFIX)


def is_act_scale_weight(name: str) -> bool:
    return name.endswith(ACT_SCALE_SUFFIX)


def is_any_scale_weight(name: str) -> bool:
    return name.endswith(SCALE_SUFFIX) or name.endswith(ACT_SCALE_SUFFIX)


def weight_name_of_scale(name: str) -> str:
    """Strips only SCALE_SUFFIX so an engine scale name maps back to its engine weight name (preserving TRANSPOSE_SUFFIX)."""
    return name[: -len(SCALE_SUFFIX)] if name.endswith(SCALE_SUFFIX) else name


def is_transposed_weight(name: str) -> bool:
    core = name[: -len(SCALE_SUFFIX)] if name.endswith(SCALE_SUFFIX) else name
    if core.endswith(ACT_SCALE_SUFFIX):
        core = core[: -len(ACT_SCALE_SUFFIX)]
    return core.endswith(TRANSPOSE_SUFFIX)


def base_weight_name(name: str) -> str:
    """Maps any quantized engine weight, weight-scale, or act-scale name back to the checkpoint's parameter name."""
    if name.endswith(SCALE_SUFFIX):
        name = name[: -len(SCALE_SUFFIX)]
    elif name.endswith(ACT_SCALE_SUFFIX):
        name = name[: -len(ACT_SCALE_SUFFIX)]
    if name.endswith(TRANSPOSE_SUFFIX):
        name = name[: -len(TRANSPOSE_SUFFIX)]
    return name


def scale_weight_name(name: str) -> str:
    return name if name.endswith(SCALE_SUFFIX) else name + SCALE_SUFFIX


def act_scale_weight_name(name: str) -> str:
    return name if name.endswith(ACT_SCALE_SUFFIX) else name + ACT_SCALE_SUFFIX


def transposed_weight_name(name: str) -> str:
    return name if name.endswith(TRANSPOSE_SUFFIX) else name + TRANSPOSE_SUFFIX


def gpu_sm(dev=None) -> tuple:
    try:
        d = dev if dev is not None else common.device()
        major, minor = torch.cuda.get_device_capability(d)
        return int(major), int(minor)
    except Exception:
        return (8, 0)


def gpu_vram(dev=None) -> int:
    try:
        d = dev if dev is not None else common.device()
        return int(torch.cuda.get_device_properties(d).total_memory)
    except Exception:
        return 0


def supported_precisions(sm: tuple = None) -> tuple:
    sm = sm if sm is not None else gpu_sm()
    if sm >= (8, 9):  # Ada, Hopper, Blackwell: native BF16 + FP8 W8A8 + INT8 W8A8 + INT4 WoQ + FP16
        return (BF16, FP8, INT8, INT4, FP16)
    if sm >= (8, 0):  # Ampere: native BF16 + INT8 W8A8 + INT4 WoQ + FP16
        return (BF16, INT8, INT4, FP16)
    if sm >= (7, 5):  # Turing: FP16 (with FP32 residual) + INT8 W8A8 + INT4 WoQ
        return (FP16, INT8, INT4)
    return (FP16,)


def param_count(module) -> int:
    if module is None:
        return 0
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += int(math.prod(p.shape))
    return total


def estimated_weight_bytes(params: int, precision: str) -> int:
    if params <= 0:
        return 0
    if precision in (BF16, FP16):
        return params * 2
    if precision == FP8:
        return int(params * 1.05)  # 1 B/weight + ~5% unquantized embedders/norms/modulations
    if precision == INT8:
        return int(params * 1.25)  # 1 B/weight + ~20% native FFN down-projections + embedders/norms
    if precision == INT4:
        return int(params * 0.59)  # 0.5 B/weight + block-64 FP32 scales + ~5% unquantized layers
    return params * 2


def auto_precision(arch, params: int, sm: tuple = None, total_vram: int = None) -> str:
    if not getattr(arch, "is_dit", False):
        return FP16
    sm = sm if sm is not None else gpu_sm()
    allowed = supported_precisions(sm)
    vram = gpu_vram() if total_vram is None else int(total_vram)
    full_16 = BF16 if BF16 in allowed else FP16
    if vram <= 0 or params <= 0:
        return full_16
    if estimated_weight_bytes(params, full_16) + AUTO_HEADROOM <= vram:
        return full_16
    quant_8 = FP8 if FP8 in allowed else (INT8 if INT8 in allowed else None)
    if quant_8 is not None and estimated_weight_bytes(params, quant_8) + AUTO_HEADROOM <= vram:
        return quant_8
    if INT4 in allowed:
        return INT4
    return quant_8 or full_16


def resolve_precision(arch, patcher=None, requested: str = None, sm: tuple = None, total_vram: int = None) -> str:
    if not getattr(arch, "is_dit", False):
        return FP16
    sm = sm if sm is not None else gpu_sm()
    allowed = supported_precisions(sm)
    if requested and requested != AUTO:
        if requested in allowed:
            return requested
        common.logger.warning(f"Precision {requested!r} isn't supported on sm{sm[0]}{sm[1]}; selecting automatically.")
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None) if patcher is not None else None
    params = param_count(module)
    return auto_precision(arch, params, sm=sm, total_vram=total_vram)


def compute_dtype(precision: str, sm: tuple = None) -> torch.dtype:
    sm = sm if sm is not None else gpu_sm()
    if precision in (BF16, FP8):
        return torch.bfloat16
    if precision == FP16:
        return torch.float16
    return torch.bfloat16 if sm >= (8, 0) else torch.float16


def uses_fp32_residual(precision: str, sm: tuple = None) -> bool:
    return compute_dtype(precision, sm) == torch.float16


def is_quantized_precision(precision: str) -> bool:
    return precision in (FP8, INT8, INT4)


def _eligible_weight(name: str, dims: list, precision: str) -> bool:
    if len(dims) != 2 or not name.endswith(".weight"):
        return False
    if dims[0] < MIN_QUANT_DIM or dims[1] < MIN_QUANT_DIM:
        return False
    if not name.startswith(("blocks.", "layers.", "text_embedder.blocks.")):
        return False
    if "adaln" in name or "norm" in name:
        return False
    # Uniform INT8 activations clip severely after unbounded / extreme-dynamic-range FFN activations
    # (MOSS-Image SiTU in [-100, 100] -> ffn_down, Anima ReLU^2 -> mlp.layer2); FP8's exponent and
    # INT4's unquantized W4A16 activations handle them cleanly.
    if precision == INT8 and ("ffn_down" in name or "mlp.layer2" in name):
        return False
    if precision == INT4 and dims[1] % INT4_BLOCK_SIZE != 0:
        return False
    return True


def _onnx_dtype_for(dtype: torch.dtype):
    import onnx

    return {
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
        torch.float32: onnx.TensorProto.FLOAT,
    }[dtype]


def _make_scalar_initializer(name: str, value: float, act_dtype: torch.dtype):
    import numpy as np
    import onnx
    from onnx import helper

    if act_dtype == torch.bfloat16:
        raw = torch.tensor(float(value), dtype=torch.bfloat16).view(torch.uint16).numpy().tobytes()
        return helper.make_tensor(name, onnx.TensorProto.BFLOAT16, [], raw, raw=True)
    if act_dtype == torch.float16:
        return helper.make_tensor(name, onnx.TensorProto.FLOAT16, [], [np.float16(value)])
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, [], [float(value)])


def _fp8_act_scale_for(weight_name: str) -> float:
    return 0.25 if any(p in weight_name for p in _FP8_SMALL_PATTERNS) else 2.0


def fallback_int8_act_scale(weight_name: str) -> float:
    base = base_weight_name(weight_name)
    for pat, val in _INT8_ROLE_FALLBACKS:
        if pat in base:
            return val
    return 0.5


_REDUCE_OPS_18 = frozenset({
    "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
})


def _upgrade_opset17_nodes(graph):
    """Upgrades opset-17 nodes that changed signature in opset 18+ (Reduce* axes attr -> input, Split num_outputs)."""
    import onnx
    from onnx import helper

    axes_cache = {}
    for node in graph.node:
        if node.op_type in _REDUCE_OPS_18:
            axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
            if axes_attr is not None:
                axes = tuple(axes_attr.ints)
                node.attribute.remove(axes_attr)
                if len(node.input) < 2 or not node.input[1]:
                    init_name = axes_cache.get(axes)
                    if init_name is None:
                        tag = "_".join(f"m{abs(a)}" if a < 0 else str(a) for a in axes)
                        init_name = f"__trt_reduce_axes_{tag}"
                        graph.initializer.append(
                            helper.make_tensor(init_name, onnx.TensorProto.INT64, [len(axes)], list(axes))
                        )
                        axes_cache[axes] = init_name
                    if len(node.input) < 2:
                        node.input.append(init_name)
                    else:
                        node.input[1] = init_name
        elif node.op_type == "Split":
            if (len(node.input) < 2 or not node.input[1]) and not any(a.name == "num_outputs" for a in node.attribute):
                node.attribute.append(helper.make_attribute("num_outputs", len(node.output)))


def _is_2d_transpose(node) -> bool:
    if node.op_type != "Transpose":
        return False
    perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
    return perm is None or perm == [1, 0]


def quantize_onnx_graph(
    model,
    precision: str,
    act_dtype: torch.dtype,
    *,
    fp32_scale_for_int4: bool = True,
    shapes: dict = None,
) -> dict:
    """Rewrites eligible 2D DiT Linear weights in an ONNX model for true low-VRAM TensorRT execution:

    - FP8 (sm89+): explicit W8A8 (`QuantizeLinear -> DequantizeLinear` on activations with floating-point E4M3 scale,
      per-channel FP8 weights) -> true 1-byte weight storage in VRAM + native SM89 FP8 Tensor Core GEMMs (~1.5-1.65x).
    - INT8 (sm75+): explicit W8A8 with refittable per-site activation scales (`<w>.__trt_act_scale__`, auto-calibrated
      per checkpoint) + per-channel INT8 weights -> true 1-byte weight storage + native INT8 Tensor Core GEMMs (~1.4-1.55x).
    - INT4 (sm75+): block-64 W4A16 with `Transpose(1, 0)` folded into `<w>.__trt_t__` (`(K, N)` layout, FP32 block scale
      + Cast) -> true 0.5-byte weight storage in VRAM, 1.43x faster & -300 MB less activation workspace than untransposed.

    If `shapes` is passed, updates it in-place (removing any transposed-weight old name and adding new weight/scale names).
    Returns a dict of newly added/updated `{name: [dims]}`.
    """
    if not is_quantized_precision(precision):
        return {}

    import onnx
    from onnx import helper

    graph = model.graph
    act_elem = _onnx_dtype_for(act_dtype)
    q_elem = {
        FP8: onnx.TensorProto.FLOAT8E4M3FN,
        INT8: onnx.TensorProto.INT8,
        INT4: onnx.TensorProto.INT4,
    }[precision]
    min_opset = 21 if precision == INT4 else 19
    min_ir = 10 if precision == INT4 else 9
    if model.ir_version < min_ir:
        model.ir_version = min_ir
    upgraded_from_17 = False
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version < min_opset:
            if imp.version < 18:
                upgraded_from_17 = True
            imp.version = min_opset
    if upgraded_from_17:
        _upgrade_opset17_nodes(graph)

    # TRT 11.5's refittable INT4 WoQ requires FP32 scale + Cast (BF16 scale triggers CASK non-refittable assertion);
    # FP8/INT8 W8A8 use native act_dtype scales so DQ outputs directly in compute dtype.
    scale_elem = onnx.TensorProto.FLOAT if (precision == INT4 and fp32_scale_for_int4) else act_elem
    needs_cast = scale_elem != act_elem

    consumers = {}
    for node in graph.node:
        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    added_shapes = {}
    removed_names = set()
    renames = {}  # old tensor name -> new tensor name
    remove_node_ids = set()
    dq_nodes = []
    new_inputs = []
    dq_weight_origin = {}  # tensor output name -> original checkpoint weight name

    for inp in list(graph.input):
        t = inp.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if not _eligible_weight(inp.name, dims, precision):
            continue
        orig_name = inp.name
        c_nodes = consumers.get(orig_name, [])
        fold_transpose = precision == INT4 and len(c_nodes) == 1 and _is_2d_transpose(c_nodes[0])

        t.elem_type = q_elem
        if fold_transpose:
            tr_node = c_nodes[0]
            w_name = transposed_weight_name(orig_name)
            n_out, k_in = dims[0], dims[1]
            inp.name = w_name
            del t.shape.dim[:]
            d0 = t.shape.dim.add()
            d0.dim_value = k_in
            d1 = t.shape.dim.add()
            d1.dim_value = n_out
            scale_dims = [k_in // INT4_BLOCK_SIZE, n_out]
            dq_attrs = {"axis": 0, "block_size": INT4_BLOCK_SIZE}
            s_name = scale_weight_name(w_name)
            dq_out = w_name + ".__trt_dq__"
            final_out = w_name + ".__trt_dq_cast__" if needs_cast else dq_out
            new_inputs.append(helper.make_tensor_value_info(s_name, scale_elem, scale_dims))
            added_shapes[w_name] = [k_in, n_out]
            added_shapes[s_name] = scale_dims
            removed_names.add(orig_name)
            dq_nodes.append(helper.make_node("DequantizeLinear", [w_name, s_name], [dq_out], name=w_name + ".dq", **dq_attrs))
            if needs_cast:
                dq_nodes.append(helper.make_node("Cast", [dq_out], [final_out], name=w_name + ".dq_cast", to=act_elem))
            renames[tr_node.output[0]] = final_out
            remove_node_ids.add(id(tr_node))
            dq_weight_origin[final_out] = orig_name
        else:
            w_name = orig_name
            s_name = scale_weight_name(w_name)
            dq_out = w_name + ".__trt_dq__"
            final_out = w_name + ".__trt_dq_cast__" if needs_cast else dq_out
            if precision == INT4:
                scale_dims = [dims[0], dims[1] // INT4_BLOCK_SIZE]
                dq_attrs = {"axis": 1, "block_size": INT4_BLOCK_SIZE}
            else:
                scale_dims = [dims[0]]
                dq_attrs = {"axis": 0}
            new_inputs.append(helper.make_tensor_value_info(s_name, scale_elem, scale_dims))
            added_shapes[s_name] = scale_dims
            dq_nodes.append(helper.make_node("DequantizeLinear", [w_name, s_name], [dq_out], name=w_name + ".dq", **dq_attrs))
            if needs_cast:
                dq_nodes.append(helper.make_node("Cast", [dq_out], [final_out], name=w_name + ".dq_cast", to=act_elem))
            renames[w_name] = final_out
            dq_weight_origin[final_out] = orig_name

    if not renames:
        return {}

    kept_nodes = []
    for node in graph.node:
        if id(node) in remove_node_ids:
            continue
        for i, inp_name in enumerate(node.input):
            if inp_name in renames:
                node.input[i] = renames[inp_name]
        kept_nodes.append(node)

    # For FP8 and INT8, insert explicit static activation QuantizeLinear -> DequantizeLinear before MatMul/Gemm
    # nodes consuming quantized weights so TensorRT keeps weights at 1 byte in VRAM and selects FP8/INT8 Tensor Cores.
    if precision in (FP8, INT8):
        changed = True
        while changed:
            changed = False
            for node in kept_nodes:
                if node.op_type in ("Cast", "Transpose") and node.input and node.input[0] in dq_weight_origin:
                    out_n = node.output[0]
                    if out_n not in dq_weight_origin:
                        dq_weight_origin[out_n] = dq_weight_origin[node.input[0]]
                        changed = True

        zp_name = f"__trt_act_zp_{precision}"
        zp_raw = b"\x00"
        graph.initializer.append(helper.make_tensor(zp_name, q_elem, [], zp_raw, raw=True))

        fp8_scale_inits = {}
        qdq_cache = {}  # (act_in, scale_key) -> dq_out
        final_nodes = []
        for node in kept_nodes:
            if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in dq_weight_origin:
                orig_w = dq_weight_origin[node.input[1]]
                act_in = node.input[0]
                if precision == FP8:
                    s_val = _fp8_act_scale_for(orig_w)
                    s_name = fp8_scale_inits.get(s_val)
                    if s_name is None:
                        tag = str(s_val).replace(".", "p")
                        s_name = f"__trt_fp8_act_scale_{tag}"
                        graph.initializer.append(_make_scalar_initializer(s_name, s_val, act_dtype))
                        fp8_scale_inits[s_val] = s_name
                    cache_key = (act_in, s_name)
                else:
                    # INT8: deduplicate by activation tensor (shared Q/K/V share one refittable scalar scale).
                    cache_key = (act_in, None)
                    s_name = None

                dq_out = qdq_cache.get(cache_key)
                if dq_out is None:
                    idx = len(qdq_cache)
                    if precision == INT8:
                        s_name = act_scale_weight_name(orig_w)
                        if s_name not in added_shapes:
                            new_inputs.append(helper.make_tensor_value_info(s_name, act_elem, []))
                            added_shapes[s_name] = []
                    q_out = f"__trt_act_q_{idx}"
                    dq_out = f"__trt_act_dq_{idx}"
                    final_nodes.append(
                        helper.make_node("QuantizeLinear", [act_in, s_name, zp_name], [q_out], name=f"__trt_act_ql_{idx}")
                    )
                    final_nodes.append(
                        helper.make_node("DequantizeLinear", [q_out, s_name, zp_name], [dq_out], name=f"__trt_act_dql_{idx}")
                    )
                    qdq_cache[cache_key] = dq_out
                node.input[0] = dq_out
            final_nodes.append(node)
        kept_nodes = final_nodes

    graph.input.extend(new_inputs)
    del graph.node[:]
    graph.node.extend(dq_nodes + kept_nodes)

    if shapes is not None:
        for old_n in removed_names:
            shapes.pop(old_n, None)
        shapes.update(added_shapes)
    return added_shapes


def pack_int4(q_int8: torch.Tensor) -> torch.Tensor:
    """Packs signed INT4 values in [-8, 7] (held in int8) two per byte: even -> low nibble, odd -> high nibble.

    Matches TensorRT 11.5's kINT4 layout specification (if element count is odd, pads trailing high nibble with 0).
    """
    flat = q_int8.reshape(-1).contiguous()
    n = flat.numel()
    if n & 1:
        flat = torch.cat([flat, flat.new_zeros(1)])
    pairs = flat.view(-1, 2).to(torch.uint8)
    packed = (pairs[:, 0] & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
    return packed.contiguous()


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of pack_int4(): unpacks packed uint8 bytes back into signed int8 values in [-8, 7]."""
    u = packed.reshape(-1).to(torch.uint8)
    lo = (u & 0x0F).to(torch.int8)
    hi = ((u >> 4) & 0x0F).to(torch.int8)
    # Sign-extend 4-bit two's complement (values >= 8 become negative: v - 16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=1).reshape(-1)[:numel]


def quantize_for_refit(weight_fp32: torch.Tensor, kind: str, scale_dtype: torch.dtype, *, transposed: bool = False) -> tuple:
    """Quantizes a 2D FP32 weight on its current device into `(quantized_weight, scale)` for TRT refit.

    `kind` is one of `FP8`, `INT8`, `INT4`. If `transposed=True`, transposes `(N, K) -> (K, N)` before quantizing
    along the K (dim=0) / N (dim=1) axes to match `Transpose`-folded ONNX graphs. Scales are strictly positive.
    """
    if weight_fp32.ndim != 2:
        raise ValueError(f"Expected 2D weight for {kind} quantization, got shape {tuple(weight_fp32.shape)}")
    w = weight_fp32 if weight_fp32.dtype == torch.float32 else weight_fp32.float()
    min_scale = 1e-4 if scale_dtype == torch.float16 else 1e-12
    if transposed:
        wt = w.t().contiguous()
        k_in, n_out = wt.shape
        if kind == FP8:
            amax = wt.abs().amax(dim=0)
            scale = (amax / 448.0).clamp_min(min_scale).to(scale_dtype).contiguous()
            q = (wt / scale.float().unsqueeze(0)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            return q, scale
        if kind == INT8:
            amax = wt.abs().amax(dim=0)
            scale = (amax / 127.0).clamp_min(min_scale).to(scale_dtype).contiguous()
            q = (wt / scale.float().unsqueeze(0)).round().clamp(-127.0, 127.0).to(torch.int8).contiguous()
            return q, scale
        if kind == INT4:
            if k_in % INT4_BLOCK_SIZE != 0:
                raise ValueError(f"Weight width {k_in} is not divisible by INT4 block size {INT4_BLOCK_SIZE}")
            blocks = wt.reshape(k_in // INT4_BLOCK_SIZE, INT4_BLOCK_SIZE, n_out)
            amax = blocks.abs().amax(dim=1)  # (K // B, N)
            scale = (amax / 7.0).clamp_min(min_scale).to(scale_dtype).contiguous()
            q_i8 = (blocks / scale.float().unsqueeze(1)).round().clamp(-8.0, 7.0).to(torch.int8).reshape(k_in, n_out)
            return pack_int4(q_i8), scale
        raise ValueError(f"Unsupported quantization kind {kind!r}")

    if kind == FP8:
        amax = w.abs().amax(dim=1)
        scale = (amax / 448.0).clamp_min(min_scale).to(scale_dtype).contiguous()
        q = (w / scale.float().unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
        return q, scale
    if kind == INT8:
        amax = w.abs().amax(dim=1)
        scale = (amax / 127.0).clamp_min(min_scale).to(scale_dtype).contiguous()
        q = (w / scale.float().unsqueeze(1)).round().clamp(-127.0, 127.0).to(torch.int8).contiguous()
        return q, scale
    if kind == INT4:
        n, k = w.shape
        if k % INT4_BLOCK_SIZE != 0:
            raise ValueError(f"Weight width {k} is not divisible by INT4 block size {INT4_BLOCK_SIZE}")
        blocks = w.reshape(n, k // INT4_BLOCK_SIZE, INT4_BLOCK_SIZE)
        amax = blocks.abs().amax(dim=-1)
        scale = (amax / 7.0).clamp_min(min_scale).to(scale_dtype).contiguous()
        q_i8 = (blocks / scale.float().unsqueeze(-1)).round().clamp(-8.0, 7.0).to(torch.int8)
        return pack_int4(q_i8), scale
    raise ValueError(f"Unsupported quantization kind {kind!r}")


class _CalibProxy:
    """Non-mutating layer-streaming proxy around a DiT module for 1-shot INT8 activation calibration.

    Materializes one Linear weight at a time on `device` via `EffectiveWeights.apply_on_gpu()` (supporting BF16, FP8,
    INT8-ConvRot, NVFP4, GGUF, and LoRA checkpoints with <150 MB peak VRAM) and records max(|x|) per eligible Linear.
    """

    def __init__(self, mod, prefix: str, source, device, dtype: torch.dtype, stats: dict):
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_stats", stats)

    def _child_prefix(self, key) -> str:
        p = object.__getattribute__(self, "_prefix")
        return f"{p}.{key}" if p else str(key)

    def __getattr__(self, attr: str):
        mod = object.__getattribute__(self, "_mod")
        val = getattr(mod, attr)
        if isinstance(val, torch.nn.Module):
            return _CalibProxy(
                val,
                self._child_prefix(attr),
                object.__getattribute__(self, "_source"),
                object.__getattribute__(self, "_device"),
                object.__getattribute__(self, "_dtype"),
                object.__getattribute__(self, "_stats"),
            )
        if isinstance(val, torch.Tensor):
            dev = object.__getattribute__(self, "_device")
            if attr in ("weight", "bias"):
                p_name = self._child_prefix(attr)
                source = object.__getattribute__(self, "_source")
                if source.has(p_name):
                    base_t = source.base(p_name)
                    t_dev = base_t if (isinstance(base_t, torch.Tensor) and base_t.device == dev) else base_t.to(dev)
                    return source.apply_on_gpu(p_name, t_dev, torch.float32) if source.needs_gpu(p_name) else t_dev
            return val.to(dev) if val.device != dev else val
        return val

    def __getitem__(self, key):
        mod = object.__getattribute__(self, "_mod")
        val = mod[key]
        if isinstance(val, torch.nn.Module):
            idx_key = key if isinstance(key, str) else str(key)
            return _CalibProxy(
                val,
                self._child_prefix(idx_key),
                object.__getattribute__(self, "_source"),
                object.__getattribute__(self, "_device"),
                object.__getattribute__(self, "_dtype"),
                object.__getattribute__(self, "_stats"),
            )
        return val

    def __iter__(self):
        mod = object.__getattribute__(self, "_mod")
        for i, child in enumerate(mod):
            if isinstance(child, torch.nn.Module):
                yield _CalibProxy(
                    child,
                    self._child_prefix(i),
                    object.__getattribute__(self, "_source"),
                    object.__getattribute__(self, "_device"),
                    object.__getattribute__(self, "_dtype"),
                    object.__getattribute__(self, "_stats"),
                )
            else:
                yield child

    def __len__(self):
        return len(object.__getattribute__(self, "_mod"))

    def __call__(self, *args, **kwargs):
        mod = object.__getattribute__(self, "_mod")
        is_linear = isinstance(mod, torch.nn.Linear) or (
            type(mod).__name__ == "Linear" and hasattr(mod, "weight") and hasattr(mod, "bias")
        )
        if is_linear:
            x = args[0]
            prefix = object.__getattribute__(self, "_prefix")
            source = object.__getattribute__(self, "_source")
            dev = object.__getattribute__(self, "_device")
            dtype = object.__getattribute__(self, "_dtype")
            stats = object.__getattribute__(self, "_stats")
            w_name = f"{prefix}.weight"
            if x.is_floating_point():
                mx = float(x.detach().float().abs().max().item())
                if math.isfinite(mx) and mx > stats.get(w_name, 0.0):
                    stats[w_name] = mx
            base_w = source.base(w_name)
            w_dev = base_w if (isinstance(base_w, torch.Tensor) and base_w.device == dev) else base_w.to(dev)
            w = source.apply_on_gpu(w_name, w_dev, dtype) if source.needs_gpu(w_name) else w_dev.to(dtype)
            b_name = f"{prefix}.bias"
            b = None
            if mod.bias is not None and source.has(b_name):
                base_b = source.base(b_name)
                b_dev = base_b if (isinstance(base_b, torch.Tensor) and base_b.device == dev) else base_b.to(dev)
                b = source.apply_on_gpu(b_name, b_dev, dtype) if source.needs_gpu(b_name) else b_dev.to(dtype)
            return F.linear(x.to(w.dtype), w, b)
        # Sequential / container modules (e.g., t_embedder[1], mlp, time_embedding, adaln_modulation):
        # run child modules sequentially if Sequential, else delegate to mod's forward bound to this proxy.
        if isinstance(mod, torch.nn.Sequential):
            out = args[0]
            for child in self:
                out = child(out)
            return out
        return type(mod).forward(self, *args, **kwargs)


def calibrate_int8_scales(arch, source, device=None) -> dict:
    """Computes per-Linear INT8 activation scales via 3 tiny 128x128 streaming forward passes (~0.4s, <150 MB VRAM).

    Cached per `(arch.id, source.base_version, dtype)` so refitting LoRAs or switching profiles reuses cached scales instantly.
    """
    from . import dit

    if arch is None or not getattr(arch, "is_dit", False) or source is None:
        return {}
    dtype = compute_dtype(INT8)
    cache_key = (arch.id, source.base_version[1], str(dtype))
    cached = _act_scale_cache.get(cache_key)
    if cached is not None:
        return cached

    dev = device if device is not None else common.device()
    fp32_res = uses_fp32_residual(INT8)
    dm = source.module
    stats = {}
    proxy = _CalibProxy(dm, "", source, dev, dtype, stats)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)

    try:
        with torch.inference_mode():
            for sig, x_std in ((0.95, 1.0), (0.50, 0.85), (0.08, 0.70)):
                t = torch.tensor([sig], device=dev, dtype=torch.float32)
                if arch.kind == "anima":
                    x = torch.randn((1, arch.in_channels, 1, 16, 16), generator=gen, device=dev, dtype=dtype) * x_std
                    ctx = torch.randn((1, 16, arch.context_dim), generator=gen, device=dev, dtype=dtype) * 0.25
                    rope = dit.anima_rope(dm, 16, 16, dev)
                    dit.anima_traceable_forward(proxy, x, t, ctx, rope, fp32_residual=fp32_res)
                elif arch.kind == "moss_image":
                    x = torch.randn((1, arch.in_channels, 8, 8), generator=gen, device=dev, dtype=dtype) * x_std
                    ctx = torch.randn((1, 32, arch.context_dim), generator=gen, device=dev, dtype=dtype) * 2.0
                    ri, rt = dit.moss_rope(dm, 8, 8, 32, dev)
                    dit.moss_traceable_forward(proxy, x, t, ctx, ri, rt, fp32_residual=fp32_res)
    except Exception as e:
        common.logger.debug(f"INT8 activation calibration fell back to role defaults: {e}")
        return {}

    # Apply 1.25x headroom over measured max(|x|) so real prompts/LoRAs don't clip while keeping high INT8 resolution.
    scales = {w_name: max(1e-4, mx * 1.25 / 127.0) for w_name, mx in stats.items() if mx > 0.0}
    if scales:
        if len(_act_scale_cache) >= 4:
            _act_scale_cache.pop(next(iter(_act_scale_cache)))
        _act_scale_cache[cache_key] = scales
    return scales


def onnx_tensor_nbytes(elem_type: int, dims: list) -> int:
    import onnx

    numel = math.prod(dims)
    if elem_type == onnx.TensorProto.INT4:
        return (numel + 1) // 2
    sizes = {
        onnx.TensorProto.FLOAT8E4M3FN: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.BFLOAT16: 2,
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
    }
    return numel * sizes[elem_type]
