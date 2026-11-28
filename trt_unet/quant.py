# GPU precision capabilities, VRAM-aware auto-selection, ONNX weight/activation quantization, and on-the-fly GPU refit
# quantization for DiT architectures (Anima, MOSS-Image). Keeps checkpoint quantization formats (BF16, FP8, MXFP8,
# NVFP4, INT8-ConvRot, W4A4, W4A8, W6A8, GGUF + LoRA) completely orthogonal to engine execution precision.
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from . import common, compat

AUTO = "auto"
BF16 = "bf16"
FP16 = "fp16"
MXFP8 = "mxfp8"
FP8 = "fp8"
FP4 = "fp4"
INT8 = "int8"
INT4 = "int4"

EXPLICIT_PRECISIONS = (BF16, MXFP8, FP8, FP4, INT8, INT4, FP16)
ALL_PRECISIONS = (AUTO, *EXPLICIT_PRECISIONS)
LABELS = {
    AUTO: "Auto",
    BF16: "BF16",
    MXFP8: "MXFP8",
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
MXFP8_BLOCK_SIZE = 32
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
_mxfp8_avail_cache = None


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


def _mxfp8_available() -> bool:
    """Checks runtime support for MXFP8 (torch FP8, TensorRT DataType.E8M0 & DataType.FP8, ONNX FLOAT8E4M3FN)."""
    global _mxfp8_avail_cache
    if _mxfp8_avail_cache is not None:
        return _mxfp8_avail_cache
    if not hasattr(torch, "float8_e4m3fn"):
        _mxfp8_avail_cache = False
        return False
    try:
        import tensorrt as trt

        if getattr(trt.DataType, "E8M0", None) is None or getattr(trt.DataType, "FP8", None) is None:
            _mxfp8_avail_cache = False
            return False
    except ImportError:
        pass
    try:
        import onnx

        if getattr(onnx.TensorProto, "FLOAT8E4M3FN", None) is None:
            _mxfp8_avail_cache = False
            return False
    except ImportError:
        pass
    _mxfp8_avail_cache = True
    return True


def supported_precisions(sm: tuple = None) -> tuple:
    sm = sm if sm is not None else gpu_sm()
    if sm >= (10, 0):  # Blackwell+: native BF16 + MXFP8 W8A8 + FP8 W8A8 + NVFP4 W4A4 + INT8 W8A8 + INT4 WoQ + FP16
        precs = [BF16]
        if _mxfp8_available():
            precs.append(MXFP8)
        precs.append(FP8)
        if _fp4_available():
            precs.append(FP4)
        precs.extend((INT8, INT4, FP16))
        return tuple(precs)
    if sm >= (8, 9):  # Ada, Hopper: native BF16 + FP8 W8A8 + INT8 W8A8 + INT4 WoQ + FP16 (no hardware E8M0/FP4)
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
    if precision == MXFP8:
        return int(params * 1.08)  # 1 B/weight + 1/32 B E8M0 block scale + ~5% unquantized embedders/norms/modulations
    if precision == FP8:
        return int(params * 1.05)  # 1 B/weight + ~5% unquantized embedders/norms/modulations
    if precision == FP4:
        return int(params * 0.60)  # 0.5 B/weight + FP8 block-16 scales + FP8 V-proj safety + ~5% unquantized layers
    if precision == INT8:
        return int(params * 1.25)  # 1 B/weight + ~20% native FFN down-projections + embedders/norms
    if precision == INT4:
        return int(params * 0.59)  # 0.5 B/weight + block-64 FP32 scales + ~5% unquantized layers
    return params * 2


@dataclass
class CheckpointQuantInfo:
    """Model-agnostic summary of quantized Linear layers in a loaded checkpoint."""

    total: int = 0
    counts: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    fp4_native: int = 0
    fp4_total: int = 0
    mxfp8_native: int = 0
    mxfp8_total: int = 0
    int8_native: int = 0
    int8_total: int = 0

    @property
    def quantized_total(self) -> int:
        return sum(c for fmt, c in self.counts.items() if fmt is not None)

    @property
    def dominant(self):
        if self.total <= 0 or self.quantized_total * 2 < self.total:
            return None
        non_float = [(c, fmt) for fmt, c in self.counts.items() if fmt is not None and c > 0]
        if not non_float:
            return None
        non_float.sort(reverse=True)
        return non_float[0][1]

    @property
    def dominant_label(self):
        dom = self.dominant
        if dom is None:
            return None
        # Mixed GGUF checkpoints (e.g. Q4_K_M mixes Q4_K + Q6_K) display all major GGUF types cleanly.
        gguf_types = sorted(
            ((c, fmt.split(":", 1)[1]) for fmt, c in self.counts.items() if isinstance(fmt, str) and fmt.startswith("gguf:") and c > 0),
            reverse=True,
        )
        if dom.startswith("gguf:") and len(gguf_types) > 1:
            top = "/".join(name for _, name in gguf_types[:3])
            return f"GGUF ({top})"
        return self.labels.get(dom, dom)

    def is_native_passthrough(self, precision: str) -> bool:
        if precision == FP4:
            return self.fp4_total > 0 and self.fp4_native * 5 >= self.fp4_total * 4  # >= 80%
        if precision == MXFP8:
            return self.mxfp8_total > 0 and self.mxfp8_native * 5 >= self.mxfp8_total * 4
        if precision == FP8:
            return self.total > 0 and self.counts.get(FP8, 0) * 5 >= self.total * 4
        if precision == INT8:
            return self.int8_total > 0 and self.int8_native * 5 >= self.int8_total * 4
        return False

    @property
    def native_precision(self):
        for p in (FP4, MXFP8, FP8, INT8):
            if self.is_native_passthrough(p):
                return p
        return None

    def is_non_native_quantized_for(self, precision: str) -> bool:
        return self.dominant is not None and self.dominant != precision and not self.is_native_passthrough(precision)


_DEFAULT_QUANT_PREFIXES = ("blocks.", "layers.", "text_embedder.blocks.")
_DEFAULT_INT8_SKIP = ("ffn_down", "mlp.layer2")
_DEFAULT_FP4_FP8_FALLBACK_SUFFIXES = (".v_proj.weight", ".proj.v.weight")


def _quant_prefixes(arch=None) -> tuple:
    val = getattr(arch, "quant_prefixes", None) if arch is not None else None
    return _DEFAULT_QUANT_PREFIXES if val is None else val


def _int8_skip_substrings(arch=None) -> tuple:
    val = getattr(arch, "int8_skip_substrings", None) if arch is not None else None
    return _DEFAULT_INT8_SKIP if val is None else val


def _fp4_fp8_fallback_suffixes(arch=None) -> tuple:
    val = getattr(arch, "fp4_fp8_fallback_suffixes", None) if arch is not None else None
    return _DEFAULT_FP4_FP8_FALLBACK_SUFFIXES if val is None else val


def shared_input_siblings(weight_name: str) -> tuple:
    """Returns checkpoint weight names guaranteed to share the exact same input activation tensor as `weight_name`.

    Used to resolve calibrated checkpoint `input_scale` values across shared activation-quantization sites in
    hybrid checkpoints (e.g. if `self_attn.q_proj` is kept in BF16 while `k_proj`/`v_proj` carry calibrated scales).
    Crucially, `cross_attn.q_proj` is NEVER grouped with `cross_attn.{k_proj,v_proj}` because Q consumes image tokens
    while K/V consume text context.
    """
    base = base_weight_name(weight_name)
    if not base.endswith(".weight"):
        return (base,)
    stem = base[: -len(".weight")]
    if "." not in stem:
        return (base,)
    parent, leaf = stem.rsplit(".", 1)
    parent_leaf = parent.rsplit(".", 1)[-1]
    if leaf in ("q_proj", "k_proj", "v_proj"):
        if parent_leaf in ("self_attn", "attn1"):
            return tuple(f"{parent}.{s}.weight" for s in ("q_proj", "k_proj", "v_proj"))
        if leaf in ("k_proj", "v_proj"):
            return (f"{parent}.k_proj.weight", f"{parent}.v_proj.weight")
        return (base,)
    if leaf in ("q", "k", "v") and parent_leaf == "proj":
        return tuple(f"{parent}.{s}.weight" for s in ("q", "k", "v"))
    if leaf in ("to_q", "to_k", "to_v"):
        if parent_leaf in ("self_attn", "attn1"):
            return tuple(f"{parent}.{s}.weight" for s in ("to_q", "to_k", "to_v"))
        if leaf in ("to_k", "to_v"):
            return (f"{parent}.to_k.weight", f"{parent}.to_v.weight")
        return (base,)
    for pair in (("ffn_up", "ffn_gate"), ("gate_proj", "up_proj"), ("w1", "w3")):
        if leaf in pair:
            return tuple(f"{parent}.{s}.weight" for s in pair)
    return (base,)


def checkpoint_quant_info(patcher, arch=None) -> CheckpointQuantInfo:
    """Inspects eligible 2D Linear weights in `patcher.model.diffusion_model` to detect checkpoint quantization."""
    dm = getattr(getattr(patcher, "model", None), "diffusion_model", None) if patcher is not None else None
    if dm is None:
        return CheckpointQuantInfo()
    backup = getattr(patcher, "backup", None) or {}
    preferred = []
    fallback_all = []
    for mod_name, mod in dm.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            continue
        w_param = getattr(mod, "weight", None)
        if w_param is None or not isinstance(w_param, torch.Tensor):
            continue
        w_name = f"{mod_name}.weight" if mod_name else "weight"
        b_entry = backup.get("diffusion_model." + w_name)
        w = getattr(b_entry, "weight", b_entry) if b_entry is not None else w_param
        shape = tuple(getattr(w, "shape", ()))
        if len(shape) != 2 or shape[0] < MIN_QUANT_DIM or shape[1] < MIN_QUANT_DIM:
            continue
        entry = (w_name, w, mod, shape)
        fallback_all.append(entry)
        if _eligible_weight(w_name, list(shape), FP8, arch=arch):
            preferred.append(entry)
    entries = preferred if preferred else fallback_all
    if not entries:
        return CheckpointQuantInfo()
    counts = {}
    labels = {}
    fp4_native = fp4_total = mxfp8_native = mxfp8_total = int8_native = int8_total = 0
    int8_skips = _int8_skip_substrings(arch)
    for w_name, w, mod, shape in entries:
        fmt, lbl = compat.layer_quant_format(w, mod)
        counts[fmt] = counts.get(fmt, 0) + 1
        if fmt is not None and lbl and fmt not in labels:
            labels[fmt] = lbl
        if not _fp4_uses_fp8_fallback(w_name, arch=arch) and shape[1] % FP4_BLOCK_SIZE == 0:
            fp4_total += 1
            if fmt == FP4:
                fp4_native += 1
        if shape[1] % MXFP8_BLOCK_SIZE == 0:
            mxfp8_total += 1
            if fmt == MXFP8:
                mxfp8_native += 1
        if not int8_skips or not any(s in w_name for s in int8_skips):
            int8_total += 1
            if fmt == INT8:
                int8_native += 1
    return CheckpointQuantInfo(
        total=len(entries),
        counts=counts,
        labels=labels,
        fp4_native=fp4_native,
        fp4_total=fp4_total,
        mxfp8_native=mxfp8_native,
        mxfp8_total=mxfp8_total,
        int8_native=int8_native,
        int8_total=int8_total,
    )


def auto_precision(arch, params: int, sm: tuple = None, total_vram: int = None, ckpt_info: CheckpointQuantInfo = None) -> str:
    if not getattr(arch, "is_dit", False):
        return FP16
    sm = sm if sm is not None else gpu_sm()
    allowed = supported_precisions(sm)
    vram = gpu_vram() if total_vram is None else int(total_vram)
    full_16 = BF16 if BF16 in allowed else FP16
    # Match native NVFP4 / MXFP8 / FP8 checkpoints directly when supported by this GPU and within VRAM for zero-loss
    # pass-through and peak Tensor Core speed; GPUs without support (e.g. NVFP4/MXFP8 on Ada/Ampere, FP8 on
    # Ampere/Turing) seamlessly fall through and upcast to the highest supported precision that fits VRAM.
    native = ckpt_info.native_precision if ckpt_info is not None else None
    if native in (FP4, MXFP8, FP8) and native in allowed:
        if vram <= 0 or params <= 0 or estimated_weight_bytes(params, native) + AUTO_HEADROOM <= vram:
            return native
    if vram <= 0 or params <= 0:
        return full_16
    if estimated_weight_bytes(params, full_16) + AUTO_HEADROOM <= vram:
        return full_16
    for quant_8 in (MXFP8, FP8, INT8):
        if quant_8 in allowed and estimated_weight_bytes(params, quant_8) + AUTO_HEADROOM <= vram:
            return quant_8
    if FP4 in allowed:
        return FP4
    if INT4 in allowed:
        return INT4
    return next((q for q in (MXFP8, FP8, INT8) if q in allowed), full_16)


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
    ckpt_info = checkpoint_quant_info(patcher, arch=arch) if patcher is not None else None
    return auto_precision(arch, params, sm=sm, total_vram=total_vram, ckpt_info=ckpt_info)


def compute_dtype(precision: str, sm: tuple = None) -> torch.dtype:
    sm = sm if sm is not None else gpu_sm()
    if precision in (BF16, MXFP8, FP8, FP4):
        return torch.bfloat16
    if precision == FP16:
        return torch.float16
    return torch.bfloat16 if sm >= (8, 0) else torch.float16


def uses_fp32_residual(precision: str, sm: tuple = None) -> bool:
    return compute_dtype(precision, sm) == torch.float16


def is_quantized_precision(precision: str) -> bool:
    return precision in (MXFP8, FP8, FP4, INT8, INT4)


def _fp4_uses_fp8_fallback(name: str, arch=None) -> bool:
    # Keep separate attention Value projections in FP8 inside NVFP4 engines:
    # 1) Avoids NVIDIA/TensorRT#4879 (batched V-GEMM FP4 activation/scale transpose mismatch on SM100/SM120).
    # 2) Protects the highest-sensitivity attention projection at minimal (~4%) VRAM overhead.
    suffixes = _fp4_fp8_fallback_suffixes(arch)
    return bool(suffixes) and name.endswith(suffixes)


def _eligible_weight(name: str, dims: list, precision: str, arch=None) -> bool:
    if len(dims) != 2 or not name.endswith(".weight"):
        return False
    if dims[0] < MIN_QUANT_DIM or dims[1] < MIN_QUANT_DIM:
        return False
    prefixes = _quant_prefixes(arch)
    if not prefixes or not name.startswith(prefixes):
        return False
    if "adaln" in name or "norm" in name:
        return False
    # Uniform INT8 activations clip severely after unbounded / extreme-dynamic-range FFN activations
    # (MOSS-Image SiTU in [-100, 100] -> ffn_down, Anima GELU up to ~170 -> mlp.layer2); FP8's exponent,
    # MXFP8's dynamic block-32 E8M0 scales, NVFP4's block-16 scales, and INT4's W4A16 handle them cleanly.
    int8_skips = _int8_skip_substrings(arch)
    if precision == INT8 and int8_skips and any(s in name for s in int8_skips):
        return False
    if precision == INT4 and dims[1] % INT4_BLOCK_SIZE != 0:
        return False
    if precision == MXFP8 and dims[1] % MXFP8_BLOCK_SIZE != 0:
        return False
    if precision == FP4 and not _fp4_uses_fp8_fallback(name, arch=arch) and dims[1] % FP4_BLOCK_SIZE != 0:
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
    arch=None,
) -> dict:
    """Rewrites eligible 2D DiT Linear weights in an ONNX model for true low-VRAM TensorRT execution:

    - MXFP8 (sm100+): native Blackwell block-32 dynamic W8A8 (`FLOAT8E4M3FN` weights + `UINT8` E8M0 block-32 scales +
      ModelOpt `trt` domain `TRT_MXFP8DynamicQuantize` / `TRT_MXFP8DequantizeLinear`), requiring zero activation
      calibration and preserving full per-block dynamic range on both weights and activations.
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
        MXFP8: onnx.TensorProto.FLOAT8E4M3FN,
        FP8: onnx.TensorProto.FLOAT8E4M3FN,
        INT8: onnx.TensorProto.INT8,
        INT4: onnx.TensorProto.INT4,
    }
    if elem_fp4 is not None:
        q_elem_map[FP4] = elem_fp4

    min_opset = 23 if precision == FP4 else (21 if precision == INT4 else (20 if precision == MXFP8 else 19))
    min_ir = 11 if precision == FP4 else (10 if precision in (INT4, MXFP8) else 9)
    if model.ir_version < min_ir:
        model.ir_version = min_ir
    upgraded_from_17 = False
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version < min_opset:
            if imp.version < 18:
                upgraded_from_17 = True
            imp.version = min_opset
    if precision in (FP4, MXFP8) and not any(imp.domain == "trt" for imp in model.opset_import):
        model.opset_import.append(helper.make_opsetid("trt", 1))
    if upgraded_from_17:
        _upgrade_opset17_nodes(graph)

    # TRT 11.5's refittable INT4 WoQ requires FP32 scale + Cast (BF16 scale triggers CASK non-refittable assertion);
    # MXFP8/FP8/INT8/NVFP4 output directly in compute dtype.
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
        if not _eligible_weight(inp.name, dims, precision, arch=arch):
            continue
        orig_name = inp.name
        w_kind = FP8 if (precision == FP4 and _fp4_uses_fp8_fallback(orig_name, arch=arch)) else precision
        used_w_kinds.add(w_kind)
        c_nodes = consumers.get(orig_name, [])
        fold_transpose = w_kind == INT4 and len(c_nodes) == 1 and _is_2d_transpose(c_nodes[0])

        t.elem_type = q_elem_map[w_kind]
        if w_kind == MXFP8:
            # Keep [N, K] unfolded with ModelOpt trt-domain block-32 E8M0 (UINT8 in ONNX) -> Transpose(1, 0) -> MatMul.
            w_name = orig_name
            s_name = scale_weight_name(w_name)
            sf_dims = [dims[0], dims[1] // MXFP8_BLOCK_SIZE]
            dq_out = w_name + ".__trt_dq__"
            new_inputs.append(helper.make_tensor_value_info(s_name, onnx.TensorProto.UINT8, sf_dims))
            added_shapes[s_name] = sf_dims
            dq_nodes.append(
                helper.make_node(
                    "TRT_MXFP8DequantizeLinear",
                    [w_name, s_name],
                    [dq_out],
                    name=w_name + ".dq",
                    domain="trt",
                    axis=-1,
                    block_size=MXFP8_BLOCK_SIZE,
                    output_dtype=act_elem,
                )
            )
            renames[w_name] = dq_out
            dq_weight_origin[dq_out] = (orig_name, MXFP8)
        elif w_kind == FP4:
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

    # For MXFP8, FP8, INT8, and NVFP4, insert explicit activation quantization before MatMul/Gemm nodes consuming
    # quantized weights so TensorRT keeps weights compressed in VRAM and selects quantized Tensor Core kernels.
    if precision in (MXFP8, FP8, INT8, FP4):
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
                    if w_kind == MXFP8:
                        f8_out = f"__trt_act_mxf8_{idx}"
                        sf_out = f"__trt_act_mxsf_{idx}"
                        dq_out = f"__trt_act_dq_{idx}"
                        final_nodes.append(
                            helper.make_node(
                                "TRT_MXFP8DynamicQuantize",
                                [act_in],
                                [f8_out, sf_out],
                                name=f"__trt_act_mxfp8q_{idx}",
                                domain="trt",
                                axis=-1,
                                block_size=MXFP8_BLOCK_SIZE,
                                output_dtype=onnx.TensorProto.FLOAT8E4M3FN,
                            )
                        )
                        final_nodes.append(
                            helper.make_node(
                                "TRT_MXFP8DequantizeLinear",
                                [f8_out, sf_out],
                                [dq_out],
                                name=f"__trt_act_mxfp8dq_{idx}",
                                domain="trt",
                                axis=-1,
                                block_size=MXFP8_BLOCK_SIZE,
                                output_dtype=act_elem,
                            )
                        )
                    else:
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


def swap_fp4_nibbles(qdata: torch.Tensor) -> torch.Tensor:
    """Converts comfy_kitchen's `(even << 4) | odd` FP4 byte packing to TensorRT's `even | (odd << 4)` layout."""
    u = qdata.contiguous()
    if u.dtype != torch.uint8:
        u = u.view(torch.uint8)
    return (((u & 0x0F) << 4) | ((u >> 4) & 0x0F)).contiguous()


def unswizzle_nvfp4_scales(blocked: torch.Tensor, rows: int, cols: int):
    """Exact inverse of comfy_kitchen / Forge Neo's 128x4 `to_blocked()` NVFP4 scale swizzle."""
    if rows <= 0 or cols <= 0:
        return None
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    if blocked.numel() != padded_rows * padded_cols:
        return None
    x = blocked.contiguous().reshape(n_row_blocks, n_col_blocks, 32, 4, 4)
    x = x.transpose(2, 3).reshape(n_row_blocks, n_col_blocks, 128, 4)
    x = x.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return x[:rows, :cols].contiguous()


def sanitize_fp4_block_scales(
    q_trt: torch.Tensor,
    w_sf: torch.Tensor,
    rows: int,
    cols: int,
    gscale: torch.Tensor = None,
    *,
    gscale_dtype: torch.dtype = None,
) -> tuple:
    """Ensures strictly positive valid FP8 E4M3FN block scales and global scale asynchronously on GPU.

    Zero/NaN block scales are replaced by `_FP8_MIN_POS` (0x01) with their 16 nibbles zeroed (bit-exact 0.0);
    negative finite scales clear their sign bit and flip all 16 nibble sign bits (`^ 0x88`) so values stay exact.
    """
    dev = q_trt.device
    sf_u8 = w_sf.to(device=dev).contiguous().view(torch.uint8).reshape(rows, cols // FP4_BLOCK_SIZE)
    zero_blk = (sf_u8 == 0x00) | (sf_u8 == 0x80) | (sf_u8 == 0x7F) | (sf_u8 == 0xFF)
    safe_sf_u8 = torch.where(zero_blk, sf_u8.new_full((), 0x01), sf_u8)
    neg_blk = (safe_sf_u8 & 0x80) != 0
    safe_sf = (safe_sf_u8 & 0x7F).view(torch.float8_e4m3fn).contiguous()

    q_blk = q_trt.contiguous().reshape(rows, cols // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE // 2)
    q_blk = torch.where(zero_blk.unsqueeze(-1), q_blk.new_zeros(()), q_blk)
    q_blk = torch.where(neg_blk.unsqueeze(-1), torch.bitwise_xor(q_blk, 0x88), q_blk)

    gs_dt = gscale_dtype or torch.bfloat16
    if gscale is None:
        safe_gs = torch.ones((), device=dev, dtype=gs_dt)
    else:
        min_gs = 1e-4 if gs_dt == torch.float16 else 1e-12
        gs_f = gscale.to(device=dev, dtype=torch.float32).reshape(())
        zero_gs = ~torch.isfinite(gs_f) | (gs_f == 0)
        neg_gs = (~zero_gs) & (gs_f < 0)
        safe_gs = torch.where(zero_gs, gs_f.new_full((), min_gs), gs_f.abs().clamp_min(min_gs)).to(gs_dt).reshape(())
        q_blk = torch.where(zero_gs, q_blk.new_zeros(()), q_blk)
        q_blk = torch.where(neg_gs, torch.bitwise_xor(q_blk, 0x88), q_blk)
    return q_blk.reshape(-1).contiguous(), safe_sf, safe_gs


def sanitize_mxfp8_block_scales(q_fp8: torch.Tensor, w_sf_u8: torch.Tensor, rows: int, cols: int) -> tuple:
    """Ensures valid finite E8M0 block-32 scales asynchronously on GPU without aliasing input staging buffers.

    NaN E8M0 scale bytes (`0xFF`) are replaced with `127` (`2^0 = 1.0`) and their 32 FP8 weights zeroed (exact 0.0).
    Returns newly owned contiguous `(q_fp8, e8m0_uint8)` tensors.
    """
    dev = q_fp8.device
    sf_u8 = w_sf_u8.to(device=dev).contiguous().view(torch.uint8).reshape(rows, cols // MXFP8_BLOCK_SIZE)
    bad = sf_u8 == 0xFF
    safe_sf = torch.where(bad, sf_u8.new_full((), 127), sf_u8).contiguous()
    q_u8 = q_fp8.contiguous().view(torch.uint8).reshape(rows, cols // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    q_u8 = torch.where(bad.unsqueeze(-1), q_u8.new_zeros(()), q_u8)
    return q_u8.view(torch.float8_e4m3fn).reshape(rows, cols).contiguous(), safe_sf


def gguf_nvfp4_to_trt(raw_data: torch.Tensor, rows: int, cols: int, *, gscale_dtype: torch.dtype = None):
    """Converts raw GGUF NVFP4 blocks (36 B / 64 elems: 4 UE4M3 scales + 32 nibble bytes) directly to TRT NVFP4."""
    if rows <= 0 or cols <= 0 or cols % 64 != 0 or not hasattr(torch, "float8_e4m3fn"):
        return None
    raw = raw_data.contiguous().view(torch.uint8).reshape(-1)
    expected = rows * (cols // 64) * 36
    if raw.numel() != expected:
        return None
    blocks = raw.reshape(rows, cols // 64, 36)
    scale_codes = blocks[..., :4].reshape(rows, cols // FP4_BLOCK_SIZE)
    sc7 = scale_codes & 0x7F
    zero_blk = (sc7 == 0x00) | (sc7 == 0x7F)
    w_sf = torch.where(zero_blk, sc7.new_full((), 0x01), sc7).view(torch.float8_e4m3fn).contiguous()

    packed = blocks[..., 4:].reshape(rows, cols // 64, 4, 8)
    nibbles = torch.cat((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1)
    q_blk = (nibbles[..., ::2] | (nibbles[..., 1::2] << 4)).reshape(rows, cols // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE // 2)
    q_blk = torch.where(zero_blk.unsqueeze(-1), q_blk.new_zeros(()), q_blk)
    w_gs = torch.ones((), device=raw.device, dtype=gscale_dtype or torch.bfloat16)
    return q_blk.reshape(-1).contiguous(), w_sf, w_gs


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

    - `MXFP8`: returns `(fp8_e4m3fn_weight, uint8_e8m0_block32_scales)`.
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
    if kind == MXFP8:
        k_dim = w.shape[0] if transposed else w.shape[1]
        if k_dim % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(f"Weight width {k_dim} is not divisible by MXFP8 block size {MXFP8_BLOCK_SIZE}")
        if transposed:
            k_in, n_out = w.shape
            blocks = w.reshape(k_in // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE, n_out)
            blk_dim = 1
        else:
            n_out, k_in = w.shape
            blocks = w.reshape(n_out, k_in // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
            blk_dim = -1
        blk_amax = torch.maximum(blocks.amax(dim=blk_dim), -blocks.amin(dim=blk_dim))
        # Smallest power-of-2 E8M0 scale >= blk_amax / 448 so no element clips; clamp exponent to [1, 254] (all normal
        # float32 powers of two, avoiding 0xFF NaN), using 127 (2^0 = 1.0) for zero/non-finite blocks.
        bad = ~torch.isfinite(blk_amax) | (blk_amax == 0)
        safe_ratio = torch.where(bad, blk_amax.new_ones(()), (blk_amax / 448.0).clamp(min=2.0**-126, max=2.0**127))
        exp = (torch.ceil(torch.log2(safe_ratio)).to(torch.int32) + 127).clamp(1, 254)
        exp = torch.where(bad, exp.new_full((), 127), exp)
        scale_f32 = (exp << 23).view(torch.float32).unsqueeze(blk_dim)
        q_f = blocks.div_(scale_f32).clamp_(-448.0, 448.0) if owned else (blocks / scale_f32).clamp(-448.0, 448.0)
        q_fp8 = q_f.to(torch.float8_e4m3fn).reshape(w.shape).contiguous()
        return q_fp8, exp.to(torch.uint8).contiguous()
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
    """Double-buffered pinned-host + single reusable GPU buffer for streaming CPU weights during activation calibration.

    Supports plain `torch.Tensor`, `ParameterGGUF`, and `QuantizedTensor` (packing `_qdata` + CPU `_params` tensors).
    """

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
        align16 = lambda n: (n + 15) & ~15
        max_cpu = max_unpinned = 0
        for name in source.params:
            plan = compat.staging_plan(source.base(name))
            if plan is None:
                continue
            parts, _ = plan
            gpu_nb = pin_nb = 0
            ok = True
            for pi, p in enumerate(parts):
                if p.is_cuda:
                    if p.device != self.dev:
                        ok = False
                        break
                    continue
                if pi == 0 and not p.is_contiguous():
                    ok = False
                    break
                nb = p.numel() * p.element_size()
                if nb == 0:
                    ok = False
                    break
                p_nb = align16(nb)
                gpu_nb += p_nb
                if not (p.is_contiguous() and p.is_pinned()):
                    pin_nb += p_nb
            if not ok or gpu_nb == 0:
                continue
            if gpu_nb > max_cpu:
                max_cpu = gpu_nb
            if pin_nb > max_unpinned:
                max_unpinned = pin_nb
        if max_cpu < (256 << 10):
            return
        try:
            free_b = int(torch.cuda.mem_get_info(self.dev)[0])
        except Exception:
            free_b = 0
        vram_cap = max(0, free_b // 4) if free_b else (160 << 20)
        if vram_cap < (4 << 20):
            return
        cap = min(max_cpu, 160 << 20, vram_cap)
        if cap < (256 << 10):
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
        plan = compat.staging_plan(t)
        if plan is None:
            return t.to(self.dev)
        parts, rebuild = plan
        if all(p.is_cuda and p.device == self.dev for p in parts):
            return t
        if self.gpu_buf is None:
            return rebuild([p if (p.is_cuda and p.device == self.dev) else p.to(self.dev) for p in parts])
        align16 = lambda n: (n + 15) & ~15
        parts = list(parts)
        offsets = []
        total = pin_total = 0
        can_pack = True
        for pi, p in enumerate(parts):
            if p.is_cuda:
                if p.device != self.dev:
                    can_pack = False
                    break
                offsets.append(None)
                continue
            if pi == 0 and not p.is_contiguous():
                can_pack = False
                break
            if pi > 0 and not p.is_contiguous():
                p = parts[pi] = p.contiguous()
            nb = p.numel() * p.element_size()
            if nb == 0:
                can_pack = False
                break
            if p.is_pinned():
                offsets.append((total, nb, p.dtype, p.shape, None))
            else:
                offsets.append((total, nb, p.dtype, p.shape, pin_total))
                pin_total = align16(pin_total + nb)
            total = align16(total + nb)
        if (
            not can_pack
            or total == 0
            or total > self.buf_nb
            or (pin_total > 0 and (self.pins is None or pin_total > self.pins[0].numel()))
        ):
            return rebuild([p if (p.is_cuda and p.device == self.dev) else p.to(self.dev) for p in parts])
        gbuf = self.gpu_buf
        s = None
        if pin_total > 0:
            s = self.slot
            self.slot = s ^ 1
            if self.slot_used[s]:
                self.ev_dma[s].synchronize()
            pbuf = self.pins[s]
            for p, spec in zip(parts, offsets):
                if spec is not None and spec[4] is not None:
                    _, nb, dt, sh, poff = spec
                    pbuf[poff : poff + nb].view(dt).view(sh).copy_(p)
        if pin_total == total:
            gbuf[:total].copy_(self.pins[s][:total], non_blocking=True)
        else:
            for p, spec in zip(parts, offsets):
                if spec is not None and spec[4] is None:
                    goff, nb, dt, sh, _ = spec
                    gbuf[goff : goff + nb].view(dt).view(sh).copy_(p, non_blocking=True)
            if pin_total > 0:
                pbuf = self.pins[s]
                deltas = {spec[0] - spec[4] for spec in offsets if spec is not None and spec[4] is not None}
                if len(deltas) == 1:
                    base_off = next(iter(deltas))
                    gbuf[base_off : base_off + pin_total].copy_(pbuf[:pin_total], non_blocking=True)
                else:
                    for spec in offsets:
                        if spec is not None and spec[4] is not None:
                            goff, nb, _, _, poff = spec
                            gbuf[goff : goff + nb].copy_(pbuf[poff : poff + nb], non_blocking=True)
        if s is not None:
            self.ev_dma[s].record(torch.cuda.current_stream(self.dev))
            self.slot_used[s] = True
        gviews = [
            p if spec is None else gbuf[spec[0] : spec[0] + spec[1]].view(spec[2]).view(spec[3])
            for p, spec in zip(parts, offsets)
        ]
        return rebuild(gviews)

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
