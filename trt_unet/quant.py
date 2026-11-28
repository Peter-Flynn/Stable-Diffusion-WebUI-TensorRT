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
FP4 = "fp4"
INT8 = "int8"
INT4 = "int4"

EXPLICIT_PRECISIONS = (BF16, FP8, FP4, INT8, INT4, FP16)
ALL_PRECISIONS = (AUTO, *EXPLICIT_PRECISIONS)
LABELS = {
    AUTO: "Auto",
    BF16: "BF16",
    FP8: "FP8",
    FP4: "NVFP4",
    INT8: "INT8",
    INT4: "INT4",
    FP16: "FP16",
}
SCALE_SUFFIX = ".__trt_scale__"
GSCALE_SUFFIX = ".__trt_gscale__"
ACT_SCALE_SUFFIX = ".__trt_act_scale__"
TRANSPOSE_SUFFIX = ".__trt_t__"
INT4_BLOCK_SIZE = 64
FP4_BLOCK_SIZE = 16
# Smallest strictly positive float8_e4m3fn subnormal (2^-9) so NVFP4 block scales never underflow to 0.
_FP8_MIN_POS = 2.0**-9
# Headroom left beyond engine weights for activations, CUDA context, and sampler state when auto-selecting precision.
AUTO_HEADROOM = int(3.2 * 2**30)
# Minimum weight matrix dimensions eligible for 8-bit/4-bit quantization (leaves embedders, norms, modulations exact).
MIN_QUANT_DIM = 512

# Role-based fallback activation scales calibrated on real Anima 2B & MOSS-Image 5B trajectories.
# Both FP8, INT8, and NVFP4 use refittable per-site activation scales auto-calibrated on the loaded checkpoint
# (+ active LoRAs); these role defaults are used only when calibration is unavailable (e.g. unit tests without a DiT).
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
# Bounded LRU list of (cache_key, amax_dict) entries using exact == equality (supports weakref _Same patch keys).
_act_amax_cache = []


def is_scale_weight(name: str) -> bool:
    return name.endswith(SCALE_SUFFIX)


def is_gscale_weight(name: str) -> bool:
    return name.endswith(GSCALE_SUFFIX)


def is_weight_scale_weight(name: str) -> bool:
    return name.endswith((SCALE_SUFFIX, GSCALE_SUFFIX))


def is_act_scale_weight(name: str) -> bool:
    return name.endswith(ACT_SCALE_SUFFIX)


def is_any_scale_weight(name: str) -> bool:
    return name.endswith((SCALE_SUFFIX, GSCALE_SUFFIX, ACT_SCALE_SUFFIX))


def weight_name_of_scale(name: str) -> str:
    """Strips SCALE_SUFFIX or GSCALE_SUFFIX so an engine scale maps to its engine weight (preserving TRANSPOSE_SUFFIX)."""
    if name.endswith(SCALE_SUFFIX):
        return name[: -len(SCALE_SUFFIX)]
    if name.endswith(GSCALE_SUFFIX):
        return name[: -len(GSCALE_SUFFIX)]
    return name


def is_transposed_weight(name: str) -> bool:
    core = weight_name_of_scale(name)
    if core.endswith(ACT_SCALE_SUFFIX):
        core = core[: -len(ACT_SCALE_SUFFIX)]
    return core.endswith(TRANSPOSE_SUFFIX)


def base_weight_name(name: str) -> str:
    """Maps any quantized engine weight, weight-scale, global-scale, or act-scale back to the checkpoint parameter."""
    if name.endswith(SCALE_SUFFIX):
        name = name[: -len(SCALE_SUFFIX)]
    elif name.endswith(GSCALE_SUFFIX):
        name = name[: -len(GSCALE_SUFFIX)]
    elif name.endswith(ACT_SCALE_SUFFIX):
        name = name[: -len(ACT_SCALE_SUFFIX)]
    if name.endswith(TRANSPOSE_SUFFIX):
        name = name[: -len(TRANSPOSE_SUFFIX)]
    return name


def scale_weight_name(name: str) -> str:
    return name if name.endswith(SCALE_SUFFIX) else name + SCALE_SUFFIX


def gscale_weight_name(name: str) -> str:
    return name if name.endswith(GSCALE_SUFFIX) else name + GSCALE_SUFFIX


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


_fp4_avail_cache = None


def _fp4_available() -> bool:
    """Checks runtime support for NVFP4 (torch FP8 block scales, TensorRT DataType.FP4, ONNX >= 1.18 FLOAT4E2M1)."""
    global _fp4_avail_cache
    if _fp4_avail_cache is not None:
        return _fp4_avail_cache
    if not hasattr(torch, "float8_e4m3fn"):
        _fp4_avail_cache = False
        return False
    try:
        import tensorrt as trt

        if getattr(trt.DataType, "FP4", None) is None:
            _fp4_avail_cache = False
            return False
    except ImportError:
        pass
    try:
        import onnx

        if getattr(onnx.TensorProto, "FLOAT4E2M1", None) is None:
            _fp4_avail_cache = False
            return False
    except ImportError:
        pass
    _fp4_avail_cache = True
    return True


def supported_precisions(sm: tuple = None) -> tuple:
    sm = sm if sm is not None else gpu_sm()
    if sm >= (10, 0) and _fp4_available():  # Blackwell+: native BF16 + FP8 W8A8 + NVFP4 W4A4 + INT8 W8A8 + INT4 WoQ + FP16
        return (BF16, FP8, FP4, INT8, INT4, FP16)
    if sm >= (8, 9):  # Ada, Hopper (or Blackwell on onnx < 1.18): native BF16 + FP8 W8A8 + INT8 W8A8 + INT4 WoQ + FP16
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
    if precision == FP4:
        return int(params * 0.60)  # 0.5 B/weight + FP8 block-16 scales + FP8 V-proj safety + ~5% unquantized layers
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
    if FP4 in allowed:
        return FP4
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
    if precision in (BF16, FP8, FP4):
        return torch.bfloat16
    if precision == FP16:
        return torch.float16
    return torch.bfloat16 if sm >= (8, 0) else torch.float16


def uses_fp32_residual(precision: str, sm: tuple = None) -> bool:
    return compute_dtype(precision, sm) == torch.float16


def is_quantized_precision(precision: str) -> bool:
    return precision in (FP8, FP4, INT8, INT4)


def _fp4_uses_fp8_fallback(name: str) -> bool:
    # Keep separate attention Value projections in FP8 inside NVFP4 engines:
    # 1) Avoids NVIDIA/TensorRT#4879 (batched V-GEMM FP4 activation/scale transpose mismatch on SM100/SM120).
    # 2) Protects the highest-sensitivity attention projection at minimal (~4%) VRAM overhead.
    return name.endswith((".v_proj.weight", ".proj.v.weight"))


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
    # (MOSS-Image SiTU in [-100, 100] -> ffn_down, Anima GELU up to ~170 -> mlp.layer2); FP8's exponent,
    # NVFP4's dynamic block-16 FP8 scales, and INT4's unquantized W4A16 activations handle them cleanly.
    if precision == INT8 and ("ffn_down" in name or "mlp.layer2" in name):
        return False
    if precision == INT4 and dims[1] % INT4_BLOCK_SIZE != 0:
        return False
    if precision == FP4 and not _fp4_uses_fp8_fallback(name) and dims[1] % FP4_BLOCK_SIZE != 0:
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


def fallback_act_scale(weight_name: str, kind: str = INT8) -> float:
    base = base_weight_name(weight_name)
    if kind == FP8:
        return _fp8_act_scale_for(base)
    if kind == FP4:
        return max(1e-6, _fp8_act_scale_for(base) / 6.0)
    for pat, val in _INT8_ROLE_FALLBACKS:
        if pat in base:
            return val
    return 0.5


def fallback_int8_act_scale(weight_name: str) -> float:
    return fallback_act_scale(weight_name, INT8)


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

    - FP8 (sm89+): explicit W8A8 with refittable per-site activation scales (`<w>.__trt_act_scale__`, auto-calibrated
      per checkpoint + active LoRAs) + per-channel FP8 weights -> 1 B/weight + native SM89+ FP8 Tensor Core GEMMs.
    - FP4 / NVFP4 (sm100+): native Blackwell block-16 W4A4 (`FLOAT4E2M1` weights + `FLOAT8E4M3FN` block scales +
      scalar global weight/activation scales + `TRT_FP4DynamicQuantize`), keeping separate attention V projections
      in FP8 to avoid TensorRT#4879 batched V-GEMM corruption and preserve attention fidelity.
    - INT8 (sm75+): explicit W8A8 with refittable per-site activation scales + per-channel INT8 weights.
    - INT4 (sm75+): block-64 W4A16 with `Transpose(1, 0)` folded into `<w>.__trt_t__` (`(K, N)` layout, FP32 block scale
      + Cast) -> 0.5 B/weight in VRAM.

    If `shapes` is passed, updates it in-place (removing any transposed-weight old name and adding new weight/scale names).
    Returns a dict of newly added/updated `{name: [dims]}`.
    """
    if not is_quantized_precision(precision):
        return {}

    import onnx
    from onnx import helper

    elem_fp4 = getattr(onnx.TensorProto, "FLOAT4E2M1", None)
    if precision == FP4 and elem_fp4 is None:
        raise RuntimeError("NVFP4 quantization requires onnx >= 1.18 with TensorProto.FLOAT4E2M1")

    graph = model.graph
    act_elem = _onnx_dtype_for(act_dtype)
    q_elem_map = {
        FP8: onnx.TensorProto.FLOAT8E4M3FN,
        INT8: onnx.TensorProto.INT8,
        INT4: onnx.TensorProto.INT4,
    }
    if elem_fp4 is not None:
        q_elem_map[FP4] = elem_fp4

    min_opset = 23 if precision == FP4 else (21 if precision == INT4 else 19)
    min_ir = 11 if precision == FP4 else (10 if precision == INT4 else 9)
    if model.ir_version < min_ir:
        model.ir_version = min_ir
    upgraded_from_17 = False
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version < min_opset:
            if imp.version < 18:
                upgraded_from_17 = True
            imp.version = min_opset
    if precision == FP4 and not any(imp.domain == "trt" for imp in model.opset_import):
        model.opset_import.append(helper.make_opsetid("trt", 1))
    if upgraded_from_17:
        _upgrade_opset17_nodes(graph)

    # TRT 11.5's refittable INT4 WoQ requires FP32 scale + Cast (BF16 scale triggers CASK non-refittable assertion);
    # FP8/INT8/NVFP4 use native act_dtype (global) scales so DQ outputs directly in compute dtype.
    int4_scale_elem = onnx.TensorProto.FLOAT if fp32_scale_for_int4 else act_elem
    int4_needs_cast = int4_scale_elem != act_elem

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
    dq_weight_origin = {}  # tensor output name -> (original checkpoint weight name, w_kind)
    used_w_kinds = set()

    for inp in list(graph.input):
        t = inp.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if not _eligible_weight(inp.name, dims, precision):
            continue
        orig_name = inp.name
        w_kind = FP8 if (precision == FP4 and _fp4_uses_fp8_fallback(orig_name)) else precision
        used_w_kinds.add(w_kind)
        c_nodes = consumers.get(orig_name, [])
        fold_transpose = w_kind == INT4 and len(c_nodes) == 1 and _is_2d_transpose(c_nodes[0])

        t.elem_type = q_elem_map[w_kind]
        if w_kind == FP4:
            # Keep [N, K] unfolded with axis=-1, block_size=16 -> Transpose(1, 0) -> MatMul (native Blackwell FP4 pattern).
            w_name = orig_name
            s_name = scale_weight_name(w_name)
            gs_name = gscale_weight_name(w_name)
            sf_dims = [dims[0], dims[1] // FP4_BLOCK_SIZE]
            scales_out = w_name + ".__trt_scales__"
            dq_out = w_name + ".__trt_dq__"
            new_inputs.append(helper.make_tensor_value_info(s_name, onnx.TensorProto.FLOAT8E4M3FN, sf_dims))
            new_inputs.append(helper.make_tensor_value_info(gs_name, act_elem, []))
            added_shapes[s_name] = sf_dims
            added_shapes[gs_name] = []
            dq_nodes.append(helper.make_node("DequantizeLinear", [s_name, gs_name], [scales_out], name=w_name + ".dq_s"))
            dq_nodes.append(
                helper.make_node("DequantizeLinear", [w_name, scales_out], [dq_out], name=w_name + ".dq", axis=-1, block_size=FP4_BLOCK_SIZE)
            )
            renames[w_name] = dq_out
            dq_weight_origin[dq_out] = (orig_name, FP4)
        elif fold_transpose:
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
            final_out = w_name + ".__trt_dq_cast__" if int4_needs_cast else dq_out
            new_inputs.append(helper.make_tensor_value_info(s_name, int4_scale_elem, scale_dims))
            added_shapes[w_name] = [k_in, n_out]
            added_shapes[s_name] = scale_dims
            removed_names.add(orig_name)
            dq_nodes.append(helper.make_node("DequantizeLinear", [w_name, s_name], [dq_out], name=w_name + ".dq", **dq_attrs))
            if int4_needs_cast:
                dq_nodes.append(helper.make_node("Cast", [dq_out], [final_out], name=w_name + ".dq_cast", to=act_elem))
            renames[tr_node.output[0]] = final_out
            remove_node_ids.add(id(tr_node))
            dq_weight_origin[final_out] = (orig_name, INT4)
        else:
            w_name = orig_name
            s_name = scale_weight_name(w_name)
            dq_out = w_name + ".__trt_dq__"
            scale_elem = int4_scale_elem if w_kind == INT4 else act_elem
            needs_cast = w_kind == INT4 and int4_needs_cast
            final_out = w_name + ".__trt_dq_cast__" if needs_cast else dq_out
            if w_kind == INT4:
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
            dq_weight_origin[final_out] = (orig_name, w_kind)

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

    # For FP8, INT8, and NVFP4, insert explicit activation quantization before MatMul/Gemm nodes consuming quantized
    # weights so TensorRT keeps weights compressed in VRAM and selects FP8/INT8/FP4 Tensor Core kernels.
    if precision in (FP8, INT8, FP4):
        changed = True
        while changed:
            changed = False
            for node in kept_nodes:
                if node.op_type in ("Cast", "Transpose") and node.input and node.input[0] in dq_weight_origin:
                    out_n = node.output[0]
                    if out_n not in dq_weight_origin:
                        dq_weight_origin[out_n] = dq_weight_origin[node.input[0]]
                        changed = True

        zp_names = {}
        for zp_kind in (FP8, INT8):
            if zp_kind in used_w_kinds:
                zp_name = f"__trt_act_zp_{zp_kind}"
                graph.initializer.append(helper.make_tensor(zp_name, q_elem_map[zp_kind], [], b"\x00", raw=True))
                zp_names[zp_kind] = zp_name

        qdq_cache = {}  # (act_in, w_kind) -> dq_out
        final_nodes = []
        for node in kept_nodes:
            if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in dq_weight_origin:
                orig_w, w_kind = dq_weight_origin[node.input[1]]
                act_in = node.input[0]
                cache_key = (act_in, w_kind)
                dq_out = qdq_cache.get(cache_key)
                if dq_out is None:
                    idx = len(qdq_cache)
                    s_name = act_scale_weight_name(orig_w)
                    if s_name not in added_shapes:
                        new_inputs.append(helper.make_tensor_value_info(s_name, act_elem, []))
                        added_shapes[s_name] = []
                    if w_kind == FP4:
                        f4_out = f"__trt_act_f4_{idx}"
                        sf_out = f"__trt_act_sf_{idx}"
                        sc_out = f"__trt_act_sc_{idx}"
                        dq_out = f"__trt_act_dq_{idx}"
                        final_nodes.append(
                            helper.make_node(
                                "TRT_FP4DynamicQuantize",
                                [act_in, s_name],
                                [f4_out, sf_out],
                                name=f"__trt_act_fp4q_{idx}",
                                domain="trt",
                                axis=-1,
                                block_size=FP4_BLOCK_SIZE,
                                scale_type=onnx.TensorProto.FLOAT8E4M3FN,
                            )
                        )
                        final_nodes.append(
                            helper.make_node("DequantizeLinear", [sf_out, s_name], [sc_out], name=f"__trt_act_dqs_{idx}")
                        )
                        final_nodes.append(
                            helper.make_node(
                                "DequantizeLinear",
                                [f4_out, sc_out],
                                [dq_out],
                                name=f"__trt_act_dql_{idx}",
                                axis=-1,
                                block_size=FP4_BLOCK_SIZE,
                            )
                        )
                    else:
                        zp_name = zp_names[w_kind]
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
    u = flat.view(torch.uint8) if flat.dtype == torch.int8 else flat.to(torch.uint8)
    pairs = u.view(-1, 2)
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


def pack_fp4(x_fp32: torch.Tensor) -> torch.Tensor:
    """Quantizes values in [-6, 6] to NVIDIA FP4 E2M1 ({0, 0.5, 1, 1.5, 2, 3, 4, 6}, round-to-nearest-even),
    packed two nibbles per uint8 (low nibble = even index, high nibble = odd index; bit 3 = sign).
    """
    flat = x_fp32.float().reshape(-1).contiguous()
    if flat.numel() & 1:
        flat = torch.cat([flat, flat.new_zeros(1)])
    sign = (flat < 0).to(torch.uint8) << 3
    a = flat.abs().clamp(max=6.0)
    # Exact round-to-nearest-even midpoints for {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}:
    # even-side ties (0.25->0, 1.25->2, 2.5->4, 5.0->6) use '>'; odd-side ties (0.75->2, 1.75->4, 3.5->6) use '>='.
    code = torch.zeros_like(sign)
    code = torch.where(a > 0.25, torch.full_like(code, 1), code)
    code = torch.where(a >= 0.75, torch.full_like(code, 2), code)
    code = torch.where(a > 1.25, torch.full_like(code, 3), code)
    code = torch.where(a >= 1.75, torch.full_like(code, 4), code)
    code = torch.where(a > 2.50, torch.full_like(code, 5), code)
    code = torch.where(a >= 3.50, torch.full_like(code, 6), code)
    code = torch.where(a > 5.00, torch.full_like(code, 7), code)
    nib = sign | code
    pairs = nib.view(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).contiguous()


def unpack_fp4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of pack_fp4(): unpacks packed uint8 bytes back into FP32 E2M1 values in [-6.0, 6.0]."""
    u = packed.reshape(-1).to(torch.uint8)
    lo = u & 0x0F
    hi = (u >> 4) & 0x0F
    nibs = torch.stack([lo, hi], dim=1).reshape(-1)[:numel]
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=u.device, dtype=torch.float32)
    vals = lut[(nibs & 0x07).long()]
    return torch.where((nibs & 0x08) != 0, -vals, vals)


def quantize_for_refit(
    weight_fp32: torch.Tensor,
    kind: str,
    scale_dtype: torch.dtype,
    *,
    gscale_dtype: torch.dtype = None,
    transposed: bool = False,
    in_place: bool = False,
) -> tuple:
    """Quantizes a 2D FP32 weight on its current device for TRT refit.

    - `FP8`, `INT8`, `INT4`: returns `(quantized_weight, scale)`.
    - `FP4` (NVFP4): returns `(packed_fp4, fp8_block_scales, scalar_global_scale)`.
    If `transposed=True`, transposes `(N, K) -> (K, N)` before quantizing along K/N to match `Transpose`-folded ONNX
    graphs. All scales are strictly positive and computed asynchronously on the tensor's device with zero host syncs.
    """
    if weight_fp32.ndim != 2:
        raise ValueError(f"Expected 2D weight for {kind} quantization, got shape {tuple(weight_fp32.shape)}")
    w = weight_fp32 if weight_fp32.dtype == torch.float32 else weight_fp32.float()
    if transposed:
        w = w.t().contiguous()
    # Guard against views aliasing caller storage (e.g. an already-transposed view whose .t().contiguous() is zero-copy).
    fresh_storage = w.untyped_storage().data_ptr() != weight_fp32.untyped_storage().data_ptr()
    owned = (in_place and w.is_contiguous()) or fresh_storage
    min_scale = 1e-4 if scale_dtype == torch.float16 else 1e-12
    reduce_dim = 0 if transposed else 1
    if kind == FP8:
        amax = torch.maximum(w.amax(dim=reduce_dim), -w.amin(dim=reduce_dim))
        scale = (amax / 448.0).clamp_min(min_scale).to(scale_dtype).contiguous()
        inv = scale.float().unsqueeze(reduce_dim)
        q_f = w.div_(inv).clamp_(-448.0, 448.0) if owned else (w / inv).clamp(-448.0, 448.0)
        return q_f.to(torch.float8_e4m3fn).contiguous(), scale
    if kind == INT8:
        amax = torch.maximum(w.amax(dim=reduce_dim), -w.amin(dim=reduce_dim))
        scale = (amax / 127.0).clamp_min(min_scale).to(scale_dtype).contiguous()
        inv = scale.float().unsqueeze(reduce_dim)
        q_f = w.div_(inv).round_().clamp_(-127.0, 127.0) if owned else (w / inv).round().clamp(-127.0, 127.0)
        return q_f.to(torch.int8).contiguous(), scale
    if kind == INT4:
        k_dim = w.shape[0] if transposed else w.shape[1]
        if k_dim % INT4_BLOCK_SIZE != 0:
            raise ValueError(f"Weight width {k_dim} is not divisible by INT4 block size {INT4_BLOCK_SIZE}")
        if transposed:
            k_in, n_out = w.shape
            blocks = w.reshape(k_in // INT4_BLOCK_SIZE, INT4_BLOCK_SIZE, n_out)
            amax = torch.maximum(blocks.amax(dim=1), -blocks.amin(dim=1))
            scale = (amax / 7.0).clamp_min(min_scale).to(scale_dtype).contiguous()
            inv = scale.float().unsqueeze(1)
            q_f = blocks.div_(inv).round_().clamp_(-8.0, 7.0) if owned else (blocks / inv).round().clamp(-8.0, 7.0)
            q_i8 = q_f.to(torch.int8).reshape(k_in, n_out)
        else:
            n_out, k_in = w.shape
            blocks = w.reshape(n_out, k_in // INT4_BLOCK_SIZE, INT4_BLOCK_SIZE)
            amax = torch.maximum(blocks.amax(dim=-1), -blocks.amin(dim=-1))
            scale = (amax / 7.0).clamp_min(min_scale).to(scale_dtype).contiguous()
            inv = scale.float().unsqueeze(-1)
            q_f = blocks.div_(inv).round_().clamp_(-8.0, 7.0) if owned else (blocks / inv).round().clamp(-8.0, 7.0)
            q_i8 = q_f.to(torch.int8)
        return pack_int4(q_i8), scale
    if kind == FP4:
        k_dim = w.shape[0] if transposed else w.shape[1]
        if k_dim % FP4_BLOCK_SIZE != 0:
            raise ValueError(f"Weight width {k_dim} is not divisible by FP4 block size {FP4_BLOCK_SIZE}")
        gs_dt = gscale_dtype or torch.bfloat16
        min_gs = 1e-4 if gs_dt == torch.float16 else 1e-12
        w_amax = torch.maximum(w.amax(), -w.amin())
        w_gs = (w_amax / 2688.0).clamp_min(min_gs).to(gs_dt).reshape(())
        gs_f = w_gs.float()
        if transposed:
            k_in, n_out = w.shape
            blocks = w.reshape(k_in // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE, n_out)
            blk_dim = 1
        else:
            n_out, k_in = w.shape
            blocks = w.reshape(n_out, k_in // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE)
            blk_dim = -1
        blk_amax = torch.maximum(blocks.amax(dim=blk_dim), -blocks.amin(dim=blk_dim))
        w_sf = (blk_amax / (6.0 * gs_f)).clamp(_FP8_MIN_POS, 448.0).to(torch.float8_e4m3fn).contiguous()
        eff = (w_sf.float() * gs_f).unsqueeze(blk_dim)
        q_f = blocks.div_(eff).clamp_(-6.0, 6.0) if owned else (blocks / eff).clamp(-6.0, 6.0)
        return pack_fp4(q_f), w_sf, w_gs
    raise ValueError(f"Unsupported quantization kind {kind!r}")


class _CalibStaging:
    """Double-buffered pinned-host + single reusable GPU buffer for streaming CPU weights during activation calibration."""

    def __init__(self, source, device):
        self.dev = torch.device(device)
        self.buf_nb = 0
        self.pins = None
        self.gpu_buf = None
        self.ev_dma = None
        self.slot_used = [False, False]
        self.slot = 0
        if self.dev.type != "cuda":
            return
        max_cpu = max_unpinned = 0
        for name in source.params:
            raw = source.base(name)
            if type(raw) is torch.Tensor and not raw.is_cuda and raw.is_contiguous():
                nb = raw.numel() * raw.element_size()
                if nb > max_cpu:
                    max_cpu = nb
                if not raw.is_pinned() and nb > max_unpinned:
                    max_unpinned = nb
        if max_cpu < (1 << 20):
            return
        try:
            free_b = int(torch.cuda.mem_get_info(self.dev)[0])
        except Exception:
            free_b = 0
        cap = min(max_cpu, 160 << 20, max(0, free_b // 4)) if free_b else min(max_cpu, 160 << 20)
        if cap < (1 << 20):
            return
        try:
            with common.on_device(self.dev):
                self.gpu_buf = torch.empty(cap, dtype=torch.uint8, device=self.dev)
                if max_unpinned > 0:
                    pin_cap = min(cap, max_unpinned)
                    self.pins = [torch.empty(pin_cap, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
                    self.ev_dma = [torch.cuda.Event() for _ in range(2)]
            self.buf_nb = cap
        except RuntimeError:
            self.pins = self.gpu_buf = self.ev_dma = None
            self.buf_nb = 0

    def stage(self, t):
        if type(t) is not torch.Tensor:
            return t.to(self.dev)
        if t.is_cuda and t.device == self.dev:
            return t
        if t.is_cuda or not t.is_contiguous() or self.gpu_buf is None:
            return t.to(self.dev)
        nb = t.numel() * t.element_size()
        if nb == 0 or nb > self.buf_nb:
            return t.to(self.dev)
        gv = self.gpu_buf[:nb].view(t.dtype).view(t.shape)
        if t.is_pinned():
            gv.copy_(t, non_blocking=True)
            return gv
        if self.pins is None or nb > self.pins[0].numel():
            return t.to(self.dev)
        s = self.slot
        self.slot = s ^ 1
        if self.slot_used[s]:
            self.ev_dma[s].synchronize()
        pv = self.pins[s][:nb].view(t.dtype).view(t.shape)
        pv.copy_(t)
        gv.copy_(pv, non_blocking=True)
        self.ev_dma[s].record(torch.cuda.current_stream(self.dev))
        self.slot_used[s] = True
        return gv

    def release(self):
        if self.dev.type == "cuda" and (self.gpu_buf is not None or self.pins is not None):
            try:
                torch.cuda.current_stream(self.dev).synchronize()
            except Exception:
                pass
        self.pins = self.gpu_buf = self.ev_dma = None
        self.buf_nb = 0


class _CalibProxy:
    """Non-mutating layer-streaming proxy around a DiT module for 1-shot FP8/INT8/NVFP4 activation calibration.

    Materializes one Linear weight at a time on `device` via `EffectiveWeights.apply_on_gpu()` (supporting BF16, FP8,
    INT8-ConvRot, NVFP4, GGUF, and LoRA checkpoints with <150 MB peak VRAM) and records max(|x|) asynchronously per
    eligible Linear without per-layer CUDA stream synchronization stalls.
    """

    def __init__(self, mod, prefix: str, source, device, dtype: torch.dtype, stats: dict, staging: _CalibStaging = None):
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_stats", stats)
        object.__setattr__(self, "_staging", staging)

    def _child_prefix(self, key) -> str:
        p = object.__getattribute__(self, "_prefix")
        return f"{p}.{key}" if p else str(key)

    def _wrap(self, child, key):
        return type(self)(
            child,
            self._child_prefix(key),
            object.__getattribute__(self, "_source"),
            object.__getattribute__(self, "_device"),
            object.__getattribute__(self, "_dtype"),
            object.__getattribute__(self, "_stats"),
            object.__getattribute__(self, "_staging"),
        )

    def __getattr__(self, attr: str):
        mod = object.__getattribute__(self, "_mod")
        val = getattr(mod, attr)
        if isinstance(val, torch.nn.Module):
            return self._wrap(val, attr)
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
        val = object.__getattribute__(self, "_mod")[key]
        if isinstance(val, torch.nn.Module):
            return self._wrap(val, key if isinstance(key, str) else str(key))
        return val

    def __iter__(self):
        for i, child in enumerate(object.__getattribute__(self, "_mod")):
            yield self._wrap(child, i) if isinstance(child, torch.nn.Module) else child

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
            staging = object.__getattribute__(self, "_staging")
            w_name = f"{prefix}.weight"
            if x.is_floating_point():
                xf = x.detach()
                mx = torch.maximum(xf.amax(), -xf.amin()).float()
                mx = torch.where(torch.isfinite(mx), mx, mx.new_zeros(()))
                prev = stats.get(w_name)
                stats[w_name] = mx if prev is None else torch.maximum(prev, mx)
            base_w = source.base(w_name)
            if staging is not None:
                w_dev = staging.stage(base_w)
            else:
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


def _calibrate_act_amax(arch, source, kind: str = INT8, device=None) -> dict:
    """Measures per-Linear activation max(|x|) via one batched B=3 128x128 streaming pass (<150 MB VRAM).

    Cached by `(arch.id, source.base_version, active_patches_sig, dtype, fp32_res)` so FP8, INT8, and NVFP4
    engines share one calibration pass per checkpoint + LoRA combination, while LoRA changes automatically
    recalibrate activation ranges.
    """
    from . import dit

    if arch is None or not getattr(arch, "is_dit", False) or source is None:
        return {}
    dtype = compute_dtype(kind)
    fp32_res = uses_fp32_residual(kind)
    patch_sig = source.all_patches_signature() if callable(getattr(source, "all_patches_signature", None)) else ()
    cache_key = (arch.id, source.base_version, patch_sig, str(dtype), fp32_res)
    for i, (k, val) in enumerate(_act_amax_cache):
        if k == cache_key:
            _act_amax_cache.append(_act_amax_cache.pop(i))
            return val

    dev = torch.device(device) if device is not None else common.device()
    dm = source.module
    stats = {}
    sigmas = (0.95, 0.50, 0.08)
    x_stds = (1.0, 0.85, 0.70)
    with common.on_device(dev):
        staging = _CalibStaging(source, dev)
        proxy = _CalibProxy(dm, "", source, dev, dtype, stats, staging=staging)
        gen = torch.Generator(device=dev)
        gen.manual_seed(0)
        try:
            with torch.inference_mode():
                t = torch.tensor(sigmas, device=dev, dtype=torch.float32)
                if arch.kind == "anima":
                    x = torch.cat(
                        [torch.randn((1, arch.in_channels, 1, 16, 16), generator=gen, device=dev, dtype=dtype) * s for s in x_stds],
                        dim=0,
                    )
                    ctx = torch.randn((len(sigmas), 16, arch.context_dim), generator=gen, device=dev, dtype=dtype) * 0.25
                    rope = dit.anima_rope(dm, 16, 16, dev)
                    dit.anima_traceable_forward(proxy, x, t, ctx, rope, fp32_residual=fp32_res)
                elif arch.kind == "moss_image":
                    x = torch.cat(
                        [torch.randn((1, arch.in_channels, 8, 8), generator=gen, device=dev, dtype=dtype) * s for s in x_stds],
                        dim=0,
                    )
                    ctx = torch.randn((len(sigmas), 32, arch.context_dim), generator=gen, device=dev, dtype=dtype) * 2.0
                    ri, rt = dit.moss_rope(dm, 8, 8, 32, dev)
                    dit.moss_traceable_forward(proxy, x, t, ctx, ri, rt, fp32_residual=fp32_res)
                keys = list(stats.keys())
                vals = torch.stack([stats[k] for k in keys]).cpu().tolist() if keys else []
        except Exception as e:
            common.logger.debug(f"{kind.upper()} activation calibration fell back to role defaults: {e}")
            return {}
        finally:
            staging.release()

    amaxes = {w_name: float(mx) for w_name, mx in zip(keys, vals) if math.isfinite(mx) and mx > 0.0}
    if amaxes:
        if len(_act_amax_cache) >= 4:
            _act_amax_cache.pop(0)
        _act_amax_cache.append((cache_key, amaxes))
    return amaxes


# Floating-point activation headroom for FP8 E4M3FN and NVFP4 (which uses FP8 E4M3FN dynamic block-16 scales):
# E4M3FN has a 28,672x normal dynamic range [2^-6, 448], so 48x headroom above synthetic 128x128 calibration amax
# prevents hard-clipping 10-45x coherent-image outlier spikes (e.g. Anima blocks.12.mlp.layer2) while keeping
# 597x full-mantissa normal precision below cal_amax (no subnormal underflow on tiny cross-attn projections).
_FP_ACT_HEADROOM = 48.0


def calibrate_act_scales(arch, source, kind: str = INT8, device=None) -> dict:
    """Returns per-Linear activation scales for `INT8` (1.25x/127), `FP8` (48x/448), or `FP4` (48x/2688)."""
    amaxes = _calibrate_act_amax(arch, source, kind=kind, device=device)
    if not amaxes:
        return {}
    if kind == FP8:
        return {w_name: max(1e-6, mx * _FP_ACT_HEADROOM / 448.0) for w_name, mx in amaxes.items()}
    if kind == FP4:
        return {w_name: max(1e-6, mx * _FP_ACT_HEADROOM / 2688.0) for w_name, mx in amaxes.items()}
    return {w_name: max(1e-4, mx * 1.25 / 127.0) for w_name, mx in amaxes.items()}


def calibrate_int8_scales(arch, source, device=None) -> dict:
    return calibrate_act_scales(arch, source, kind=INT8, device=device)


def onnx_tensor_nbytes(elem_type: int, dims: list) -> int:
    import onnx

    numel = math.prod(dims)
    elem_fp4 = getattr(onnx.TensorProto, "FLOAT4E2M1", None)
    if elem_type == onnx.TensorProto.INT4 or (elem_fp4 is not None and elem_type == elem_fp4):
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
