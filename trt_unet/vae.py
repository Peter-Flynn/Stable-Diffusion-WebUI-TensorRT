# The VAE's decoder and encoder as weightless TensorRT engines, refitted from whichever VAE is loaded. This module: which
# VAEs the engines support, the shapes and precisions they're built for, and the engine records on disk
# (models/Unet-trt/<UNet arch id>/vae/). vae_build.py builds them; vae_run.py decodes and encodes on them.
#
# Each runs in three parts: a "pre" engine up to the mid-block attention's q/k/v, that attention on PyTorch (one head
# 512 channels wide, which TensorRT's fused attention doesn't cover: unfused it would hold the whole score matrix,
# ~17 GB in FP32 at 2048x2048), and a "post" engine for the rest. The encoder's are "encode-pre" and "encode-post".
#
# Whole-image engines run in reForge's VAE precision (BF16, or FP32 where that would be FP16), never FP16: VAE
# activations exceed FP16's range (SDXL's do), and computing each GroupNorm in FP32 inside an FP16 graph is only fast in
# TensorRT with an FP32 copy of the tensor, which at whole-image sizes costs as much as FP32 or doesn't fit (measured:
# GroupNorm was 85% of decode time without it).
#
# Strip engines (strips.py) run in FP16 where the GPU is fast at it, with the GroupNorm split across engines. They can't
# overflow, because every nonlinearity (SiLU, the attention softmax) directly follows a GroupNorm, whose output doesn't
# depend on its input's scale. Everything between two GroupNorms is therefore linear, and runs multiplied by 2^-EXPONENT
# if the conv biases added along the way are multiplied too (scaled_bias). The last GroupNorm removes the scale.
import contextlib
import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import torch

from . import common, store, strips
from .common import logger

PRE, POST = "pre", "post"
ENCODE_PRE, ENCODE_POST = "encode-pre", "encode-post"
ENCODER = (ENCODE_PRE, ENCODE_POST)
ENCODE_STRIP = "encode-strip-"
MAX_BATCH = 16
# FP16 holds magnitudes up to 2^16 with precision down to 2^-24, so a tensor keeps full FP16 precision until its typical
# value nears 2^-13 (measured: a decoder with values around 0.5 decodes identically scaled by 1 through 2^-12). At 2^-6,
# values up to ~4M fit and values down to 0.01 keep their precision. On NaN/Inf the VAE's exponent rises by
# EXPONENT_STEP, up to EXPONENT_MAX; past that it doesn't run in strips.
EXPONENT, EXPONENT_STEP, EXPONENT_MAX = 6, 3, 12


def loaded_vae():
    from . import arch as arch_mod

    return getattr(getattr(arch_mod.loaded_model(), "forge_objects", None), "vae", None)


def is_decoder_weight(name: str) -> bool:
    return name.startswith("decoder.") or name.startswith("post_quant_conv.")


def unsupported(vae) -> str:
    fsm = getattr(vae, "first_stage_model", None)
    decoder = getattr(fsm, "decoder", None)
    if decoder is None or type(decoder).__name__ != "Decoder":
        return "not an SD or SDXL VAE"
    if getattr(vae, "latent_dim", 2) != 2 or getattr(vae, "upscale_ratio", 8) != 8 or getattr(vae, "extra_1d_channel", None) is not None:
        return "not an image VAE with 8x upscaling"
    if getattr(vae, "packed_latent_channels", None) is not None or getattr(vae, "latent_bn_running_mean", None) is not None:
        return "packed latents"
    if getattr(vae, "latent_scale_factor", 1.0) != 1.0 or getattr(vae, "latent_shift_factor", 0.0) != 0.0:
        return "scaled latents"
    if decoder.give_pre_end or decoder.tanh_out or not isinstance(decoder.conv_in, torch.nn.Conv2d):
        return "an unusual decoder"
    if any(len(up.attn) for up in decoder.up):
        return "attention outside the mid block"
    if any(k for k, v in (getattr(getattr(vae, "patcher", None), "patches", None) or {}).items() if v):
        return "patched VAE weights"
    return ""


def config(vae) -> dict:
    fsm = vae.first_stage_model
    d = fsm.decoder
    pqc = getattr(fsm, "post_quant_conv", None)
    return {
        "z_channels": int(d.conv_in.in_channels),
        "latent_channels": int(pqc.in_channels) if pqc is not None else int(d.conv_in.in_channels),
        "post_quant_conv": pqc is not None,
        "ch": int(d.ch),
        "ch_mult": [int(d.up[i].block[-1].out_channels) // int(d.ch) for i in range(d.num_resolutions)],
        "num_res_blocks": int(d.num_res_blocks),
        "out_ch": int(d.conv_out.out_channels),
        "mid_channels": int(d.mid.attn_1.in_channels),
        "padding": {n: m.padding_mode for n, m in fsm.named_modules() if is_decoder_weight(n + ".") and isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"},
    }


def digest(vae) -> str:
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_decoder_weight(k))
    return hashlib.sha256(json.dumps([sorted(config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def is_encoder_weight(name: str) -> bool:
    return name.startswith("encoder.") or name.startswith("quant_conv.")


_ENCODES = {"AutoencodingEngine.encode": False, "AutoencodingEngineLegacy.encode": True}  # reForge encode -> applies quant_conv


def encoder_unsupported(vae) -> str:
    fsm = getattr(vae, "first_stage_model", None)
    encoder = getattr(fsm, "encoder", None)
    if encoder is None or type(encoder).__name__ != "Encoder":
        return "not an SD or SDXL VAE's encoder"
    how = getattr(getattr(type(fsm), "encode", None), "__qualname__", "")
    if how not in _ENCODES or not callable(getattr(fsm, "regularization", None)):
        return "a first stage model that encodes its own way"
    if _ENCODES[how] and not isinstance(getattr(fsm, "quant_conv", None), torch.nn.Conv2d):
        return "an unusual encoder"
    try:
        if getattr(vae, "downscale_ratio", 8) != 8 or 2 ** (encoder.num_resolutions - 1) != 8:
            return "not an image VAE with 8x downscaling"
        if not isinstance(encoder.conv_in, torch.nn.Conv2d) or any(len(down.attn) for down in encoder.down):
            return "an unusual encoder"
        for down in list(encoder.down)[:-1]:
            conv = down.downsample.conv if down.downsample.with_conv else None
            if conv is None or tuple(conv.stride) != (2, 2) or tuple(conv.kernel_size) != (3, 3) or tuple(conv.padding) != (0, 0):
                return "an unusual encoder"
    except AttributeError:
        return "an unusual encoder"
    return ""


def encoder_config(vae) -> dict:
    fsm = vae.first_stage_model
    e = fsm.encoder
    quant = getattr(fsm, "quant_conv", None) if _ENCODES[type(fsm).encode.__qualname__] else None
    out = int(e.conv_out.out_channels)
    return {
        "in_channels": int(e.conv_in.in_channels),
        "ch": int(e.ch),
        "ch_mult": [int(e.down[i].block[-1].out_channels) // int(e.ch) for i in range(e.num_resolutions)],
        "num_res_blocks": int(e.num_res_blocks),
        "out_channels": out,
        "quant_conv": quant is not None,
        "moments": int(quant.out_channels) if quant is not None else out,
        "mid_channels": int(e.mid.attn_1.in_channels),
        "padding": {n: m.padding_mode for n, m in fsm.named_modules() if is_encoder_weight(n + ".") and isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"},
    }


def encoder_digest(vae) -> str:
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_encoder_weight(k))
    return hashlib.sha256(json.dumps([sorted(encoder_config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def engine_dtype(vae):
    return torch.float32 if vae.vae_dtype == torch.float16 else vae.vae_dtype


def strip_dtype(vae):
    dtype = engine_dtype(vae)
    if dtype == torch.float32:
        try:
            mm = common.mm()
            if not getattr(mm.args, "vae_in_fp32", False) and mm.should_use_fp16(mm.get_torch_device()):
                return torch.float16
        except Exception:
            pass
    return dtype


def scaled(dtype) -> bool:
    return dtype == torch.float16


def precisions(dtype, strips_dtype) -> str:
    whole, in_strips = label(dtype_name(dtype)), label(dtype_name(strips_dtype))
    return whole if whole == in_strips else f"{whole} ({in_strips} in strips)"


_SCALED_BIASES = (
    "post_quant_conv.bias", "decoder.conv_in.bias", "decoder.mid.attn_1.proj_out.bias",
    "encoder.conv_in.bias", "encoder.mid.attn_1.proj_out.bias",
)
_SCALED_ENDINGS = (
    ".conv1.bias", ".conv2.bias", ".nin_shortcut.bias", ".conv_shortcut.bias", ".upsample.conv.bias", ".downsample.conv.bias",
)


def scaled_bias(name: str) -> bool:
    # Biases added to scaled tensors, refitted multiplied by the scale: every conv's, except convs whose input comes
    # straight from a GroupNorm (attention q/k/v, the final conv_out, and quant_conv after the encoder's).
    return name in _SCALED_BIASES or name.endswith(_SCALED_ENDINGS)


def weight_bytes(vae, names, dtype) -> int:
    state = vae.first_stage_model.state_dict(keep_vars=True)
    return sum(state[n].numel() for n in names if n in state) * torch.tensor([], dtype=dtype).element_size()


def label(name: str) -> str:
    return {"bfloat16": "BF16", "float32": "FP32", "float16": "FP16"}.get(name, name)


TENSOR_LIMIT = 2**31 - 1  # max elements in one tensor for TensorRT's kernels (Myelin)
RESERVE = 2**30  # VRAM left out of the budget: CUDA context, engine weights, kernel scratch
RETRY_SHARE = 0.75
PLANNED = 1.2  # TensorRT's decode activation memory / peak() (4.32 GiB at 1472x1024 in FP32, TensorRT 11.4)


def gpu_vram():
    try:
        return int(torch.cuda.get_device_properties(common.mm().get_torch_device()).total_memory)
    except Exception:
        return None


def vram_budget(dtype=None, vram=None, share: float = 1.0):
    if dtype is None or not vram:
        return None
    return max(0, int((vram - RESERVE) * share / torch.tensor([], dtype=dtype).element_size()))


def _levels(cfg: dict, h: int, w: int):
    ch, mult = cfg["ch"], cfg["ch_mult"]
    for i in range(len(mult)):
        scale = 2 ** (len(mult) - 1 - i)
        incoming = ch * mult[i + 1] if i + 1 < len(mult) else cfg["mid_channels"]
        yield incoming, ch * mult[i], (h * scale + 2) * (w * scale + 2)


def largest_tensor(cfg: dict, b: int, h: int, w: int) -> int:
    worst = b * cfg["latent_channels"] * (h + 2) * (w + 2)
    return max([worst] + [b * max(incoming, own) * pixels for incoming, own, pixels in _levels(cfg, h, w)])


def peak(cfg: dict, b: int, h: int, w: int) -> int:
    # Max elements alive at once in a decode, over each resolution: a block's input (kept for the shortcut) + its
    # normalized copy + conv output, or the upsampler's input, output and conv. For SD/SDXL VAEs, 256+256+128 channels at
    # full size: 2.5x the largest tensor.
    return max(
        b * max(2 * incoming + own, incoming + 2 * own, 3 * own, 2 * incoming + incoming // 4) * pixels
        for incoming, own, pixels in _levels(cfg, h, w)
    )


def excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    tensor = largest_tensor(cfg, b, h, w) / TENSOR_LIMIT
    memory = peak(cfg, b, h, w) * PLANNED / budget if budget else 0.0
    if budget is not None and budget <= 0:
        memory = math.inf
    return max(tensor, memory), "tensor" if tensor >= memory else "memory"


def why_smaller(cause: str, dtype, vram, what: str = "decode") -> str:
    if cause == "tensor":
        return f"a larger {what}'s largest tensor would pass TensorRT's limit of 2^31 values"
    return f"a larger {what} in {label(dtype_name(dtype))} wouldn't fit in this GPU's {common.gib(vram)}, even alone"


def _shrunk(h: int, w: int, over, step: int = 1) -> tuple:
    ratio = over(h, w)
    if ratio > 1:
        scale = math.sqrt(1 / ratio) if ratio < math.inf else 0
        h, w = max(step, int(h * scale) // step * step), max(step, int(w * scale) // step * step)
        while over(h, w) > 1 and max(h, w) > step:
            h, w = (h - step, w) if h >= w else (h, w - step)
    return h, w


def shapes_for(profiles: list, cfg: dict, dtype=None, vram=None, share: float = 1.0) -> dict:
    # The post engine's max size is capped (keeping the aspect ratio), then its batch, so a decode fits TENSOR_LIMIT and
    # `share` of this GPU's VRAM; larger decodes run in strips. The pre engine takes every size: strips need it too.
    budget = vram_budget(dtype, vram, share)
    lo_h, lo_w = min(p.height[0] for p in profiles) // 8, min(p.width[0] for p in profiles) // 8
    hi_h, hi_w = max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8
    full_h, full_w = hi_h, hi_w
    hi_h, hi_w = _shrunk(hi_h, hi_w, lambda h, w: excess(cfg, 1, h, w, budget)[0])
    fits = TENSOR_LIMIT // largest_tensor(cfg, 1, hi_h, hi_w)
    if budget is not None:
        fits = min(fits, int(budget / (peak(cfg, 1, hi_h, hi_w) * PLANNED)))
    batch = max(1, min(MAX_BATCH, max(p.batch[2] for p in profiles), fits))
    zc, c = cfg["latent_channels"], cfg["mid_channels"]

    def side(ch, hi_h, hi_w):
        low_h, low_w = min(lo_h, hi_h), min(lo_w, hi_w)
        opt_h, opt_w = min(max(profiles[0].height[1] // 8, low_h), hi_h), min(max(profiles[0].width[1] // 8, low_w), hi_w)
        return [[1, ch, low_h, low_w], [1, ch, opt_h, opt_w], [batch, ch, hi_h, hi_w]]

    return {PRE: {"latent": side(zc, full_h, full_w)}, POST: {"x": side(c, hi_h, hi_w), "attention": side(c, hi_h, hi_w)}}


def _encoder_levels(cfg: dict, h: int, w: int):
    ch, mult = cfg["ch"], cfg["ch_mult"]
    for i in range(len(mult)):
        yield ch * (mult[i - 1] if i else 1), ch * mult[i], (h // 2**i + 2) * (w // 2**i + 2)


def encoder_largest(cfg: dict, b: int, h: int, w: int) -> int:
    worst = b * cfg["in_channels"] * (h + 2) * (w + 2)
    return max([worst] + [b * max(incoming, own) * pixels for incoming, own, pixels in _encoder_levels(cfg, h, w)])


def encoder_peak(cfg: dict, b: int, h: int, w: int) -> int:  # a block's input, its normalized copy, and SiLU of that
    return max(b * 3 * max(incoming, own) * pixels for incoming, own, pixels in _encoder_levels(cfg, h, w))


def encoder_excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    tensor = encoder_largest(cfg, b, h, w) / TENSOR_LIMIT
    memory = encoder_peak(cfg, b, h, w) / budget if budget else 0.0
    if budget is not None and budget <= 0:
        memory = math.inf
    return max(tensor, memory), "tensor" if tensor >= memory else "memory"


def encoder_shapes(profiles: list, cfg: dict, dtype=None, vram=None, share: float = 1.0) -> dict:
    budget = vram_budget(dtype, vram, share)
    lo_h, lo_w = min(p.height[0] for p in profiles) // 8 * 8, min(p.width[0] for p in profiles) // 8 * 8
    hi_h, hi_w = max(p.height[2] for p in profiles) // 8 * 8, max(p.width[2] for p in profiles) // 8 * 8
    full_h, full_w = hi_h, hi_w
    hi_h, hi_w = _shrunk(hi_h, hi_w, lambda h, w: encoder_excess(cfg, 1, h, w, budget)[0], 8)
    fits = TENSOR_LIMIT // encoder_largest(cfg, 1, hi_h, hi_w)
    if budget is not None:
        fits = min(fits, int(budget / encoder_peak(cfg, 1, hi_h, hi_w)))
    batch = max(1, min(MAX_BATCH, max(p.batch[2] for p in profiles), fits))
    lo_h, lo_w = min(lo_h, hi_h), min(lo_w, hi_w)
    opt_h, opt_w = min(max(profiles[0].height[1] // 8 * 8, lo_h), hi_h), min(max(profiles[0].width[1] // 8 * 8, lo_w), hi_w)
    c = cfg["mid_channels"]
    # Not capped like the image: encodes in strips feed encode-post at the full size
    mid = [[1, c, lo_h // 8, lo_w // 8], [1, c, opt_h // 8, opt_w // 8], [batch, c, full_h // 8, full_w // 8]]
    image = [[1, cfg["in_channels"], lo_h, lo_w], [1, cfg["in_channels"], opt_h, opt_w], [batch, cfg["in_channels"], hi_h, hi_w]]
    return {ENCODE_PRE: {"image": image}, ENCODE_POST: {"x": mid, "attention": [list(shape) for shape in mid]}}


def strip_shapes(profiles: list, found: list, dtype=torch.float32) -> dict:
    lo_w = min(p.width[0] for p in profiles) // 8
    hi = (max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8)
    opt = (profiles[0].height[1] // 8, profiles[0].width[1] // 8)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return {name: strips.kind_shapes(kind, scales, lo_w, opt, hi, itemsize) for name, (kind, scales) in strips.kinds(found).items()}


def _covers(outer: dict, inner: dict) -> bool:
    for part, shapes in inner.items():
        for name, (lo, _, hi) in shapes.items():
            o_lo, _, o_hi = outer.get(part, {}).get(name, ([1] * 4, None, [0] * 4))
            if any(a < b for a, b in zip(lo, o_lo)) or any(a > b for a, b in zip(hi, o_hi)):
                return False
    return True


def folder(arch_id: str) -> str:
    return os.path.join(store.root(), arch_id, "vae")


def path_prefix(arch_id: str) -> str:
    return os.path.join(folder(arch_id), f"{common.gpu_tag()}-{common.trt_tag()}")


@dataclass
class Record:
    base: str
    meta: dict = field(repr=False)

    @property
    def key(self) -> str:
        return self.meta["key"]

    def plan(self, part: str) -> str:
        return f"{self.base}.{part}.plan"

    @property
    def dtype(self):
        return getattr(torch, self.meta["dtype"])

    @property
    def strips_dtype(self):
        return getattr(torch, self.meta["strips_dtype"])

    @property
    def scaled(self) -> bool:
        return scaled(self.strips_dtype)

    @property
    def loadable(self) -> bool:
        try:
            return self.meta["gpu"] == common.gpu_tag() and common.trt_tag(self.meta["trt"]) == common.trt_tag()
        except Exception:
            return False

    def strip_plan(self, kind: str) -> str:
        return f"{self.base}.strip-{kind}.plan"

    @property
    def encoder(self):
        return self.meta.get("encoder") or None

    @property
    def encoder_strips(self) -> dict:
        return (self.encoder or {}).get("strips") or {}

    def encoder_strip_plan(self, kind: str) -> str:
        return f"{self.base}.encode-strip-{kind}.plan"

    def encoder_strips_cover(self, found: list, h: int, w: int) -> bool:
        return bool(self.encoder_strips) and self._post_encodes(h // 8, w // 8) and _kinds_cover(self.encoder_strips, found, w // 8)

    def _post_encodes(self, h: int, w: int) -> bool:
        lo, _, hi = self.encoder["shapes"][ENCODE_POST]["x"]
        return lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def has_encoder(self, digest: str, c: int) -> bool:
        enc = self.encoder
        return enc is not None and enc.get("digest") == digest and c == enc["shapes"][ENCODE_PRE]["image"][0][1]

    def encodes(self, h: int, w: int) -> bool:
        lo, _, hi = self.encoder["shapes"][ENCODE_PRE]["image"]
        return lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    @property
    def encode_batch(self) -> int:
        return int(self.encoder["shapes"][ENCODE_PRE]["image"][2][0])

    def encode_reach(self) -> str:
        b, _, h, w = self.encoder["shapes"][ENCODE_PRE]["image"][2]
        text = f"{w}x{h}, {b} at a time"
        if self.encoder_strips:
            _, _, ph, pw = self.encoder["shapes"][ENCODE_POST]["x"][2]
            text += f", and in strips up to {pw * 8}x{ph * 8}"
        return text

    @property
    def strips(self) -> dict:
        return self.meta.get("strips") or {}

    def _pre_covers(self, c: int, h: int, w: int) -> bool:
        lo, _, hi = self.meta["shapes"][PRE]["latent"]
        return c == lo[1] and lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def covers(self, b: int, c: int, h: int, w: int) -> bool:
        lo, _, hi = self.meta["shapes"][POST]["x"]
        return self._pre_covers(c, h, w) and lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def strips_cover(self, found: list, c: int, h: int, w: int) -> bool:
        return bool(self.strips) and self._pre_covers(c, h, w) and _kinds_cover(self.strips, found, w)

    @property
    def max_batch(self) -> int:
        return int(self.meta["shapes"][POST]["x"][2][0])

    def reach(self) -> str:
        b, _, h, w = self.meta["shapes"][POST]["x"][2]
        text = f"{w * 8}x{h * 8}, {b} at a time"
        if self.strips:
            _, _, ph, pw = self.meta["shapes"][PRE]["latent"][2]
            text += f", and in strips up to {pw * 8}x{ph * 8}"
        return text


def _kinds_cover(kinds: dict, found: list, w: int) -> bool:
    for s in found:
        entry = kinds.get(s.kind.name)
        if entry is None or not strips.engine_rows(entry["shapes"], s.kind, w * s.scale):
            return False
    return True


def records(arch_id: str) -> list:
    found = []
    for path in glob.glob(glob.escape(path_prefix(arch_id)) + "-*.json"):
        try:
            base = path[: -len(".json")]
            meta = common.read_json(path)  # shared cache: copy before changing
            r = Record(base, meta)
            if r.loadable and all(os.path.isfile(r.plan(p)) for p in (PRE, POST)):
                if not all(os.path.isfile(r.strip_plan(k)) for k in r.strips):
                    meta = {**meta, "strips": {}}
                if r.encoder is not None and not all(os.path.isfile(r.plan(p)) for p in ENCODER):
                    meta = {**meta, "encoder": None}
                elif r.encoder is not None and not all(os.path.isfile(r.encoder_strip_plan(k)) for k in r.encoder_strips):
                    meta = {**meta, "encoder": {**r.encoder, "strips": {}}}
                found.append(Record(base, meta))
        except Exception as e:
            logger.debug(f"Ignoring {path}, which couldn't be read: {e}")
    return found


def record_for(arch_id: str, vae):
    try:
        wanted = digest(vae)
        matching = [r for r in records(arch_id) if r.meta.get("digest") == wanted]
    except Exception as e:
        logger.debug(f"Couldn't look up the VAE engines: {e}")
        return None
    wanted = (engine_dtype(vae), strip_dtype(vae))
    return next((r for r in matching if (r.dtype, r.strips_dtype) == wanted), matching[0] if matching else None)


def remove_record(base: str):
    for path in [base + ".json"] + glob.glob(glob.escape(base) + ".*.plan"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def delete_for(arch_id: str, gpu: str, trt: str):
    for path in glob.glob(os.path.join(glob.escape(folder(arch_id)), f"{gpu}-{common.trt_tag(trt)}-*.json")):
        remove_record(path[: -len(".json")])


def available() -> bool:
    try:
        common.trt_version()
        return True
    except Exception:
        return False


def needs_build(arch, profiles: list, vae=None) -> str:
    vae = vae if vae is not None else loaded_vae()
    if arch is None or not profiles or vae is None or unsupported(vae) or not available():
        return ""
    try:
        cfg, dtype, sdtype = config(vae), engine_dtype(vae), strip_dtype(vae)
        shapes = shapes_for(profiles, cfg, dtype, gpu_vram())
        strip_wanted = strips_wanted(vae, profiles, cfg, sdtype)
        enc_wanted = encoder_wanted(vae, profiles, dtype, sdtype, gpu_vram())
        lacking = [parts_missing(r, shapes, strip_wanted, enc_wanted) for r in records_matching(arch.id, vae, dtype, sdtype)]
        if any(not parts for parts in lacking):
            return ""
    except Exception as e:
        logger.debug(f"Couldn't check which VAE engines are built: {e}")
        return ""
    only_encoder = any(all(part in ENCODER or part.startswith(ENCODE_STRIP) for part in parts) for parts in lacking)
    what = "VAE encoder" if only_encoder else "VAE decoder and encoder" if enc_wanted else "VAE decoder"
    return f"the {what}, in {precisions(dtype, sdtype)}"


def encoder_wanted(vae, profiles: list, dtype, sdtype, vram) -> dict:
    try:
        if encoder_unsupported(vae):
            return None
        cfg = encoder_config(vae)
        return {
            "digest": encoder_digest(vae), "config": cfg, "shapes": encoder_shapes(profiles, cfg, dtype, vram),
            "strips": strips_wanted(vae, profiles, cfg, sdtype, encoding=True),
        }
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def records_matching(arch_id: str, vae, dtype, sdtype) -> list:
    wanted = digest(vae)
    return [r for r in records(arch_id) if r.meta.get("digest") == wanted and (r.dtype, r.strips_dtype) == (dtype, sdtype)]


def strips_wanted(vae, profiles: list, cfg: dict, dtype, encoding: bool = False) -> dict:
    try:
        lists = (strips.encoder_every_way if encoding else strips.every_way)(vae.first_stage_model, cfg["padding"])
    except (ValueError, KeyError, AttributeError, StopIteration):
        return {}
    return {} if any(strips.supported(found) for found in lists) else strip_shapes(profiles, lists, dtype)


def parts_missing(record: Record, shapes: dict, strip_wanted: dict, enc_wanted: dict = None) -> list:
    # Compared against "asked" (what an engine was built for), not "shapes": one TensorRT had to build smaller still
    # counts as built. Parts whose build already failed for these shapes ("*_error", "*_asked") count as built too.
    meta = record.meta
    out = []
    if not _covers(meta["shapes"], {PRE: shapes[PRE]}):
        out.append(PRE)
    if not _covers(meta["asked"], {POST: shapes[POST]}):
        out.append(POST)
    if enc_wanted is not None:
        have = record.encoder
        same = have is not None and have.get("digest") == enc_wanted["digest"]
        built = same and _covers(have["asked"], enc_wanted["shapes"])
        tried = meta.get("encoder_asked") if meta.get("encoder_error") else None
        if not built and not (tried and tried.get("digest") == enc_wanted["digest"] and _covers(tried["shapes"], enc_wanted["shapes"])):
            out += list(ENCODER)
        if same:
            kinds = have.get("strips") or {}
            failed = have.get("strips_asked") if have.get("strips_error") else None
            out += [
                ENCODE_STRIP + name for name, wanted in enc_wanted["strips"].items()
                if not (name in kinds and _covers({name: kinds[name]["shapes"]}, {name: wanted}))
                and not (failed and name in failed and _covers({name: failed[name]}, {name: wanted}))
            ]
        elif ENCODE_PRE in out:
            out += [ENCODE_STRIP + name for name in enc_wanted["strips"]]
    failed = meta.get("strips_asked") if meta.get("strips_error") else None
    for name, kind_shapes in strip_wanted.items():
        have = record.strips.get(name)
        if have is not None and _covers({name: have["shapes"]}, {name: kind_shapes}):
            continue
        if failed and name in failed and _covers({name: failed[name]}, {name: kind_shapes}):
            continue
        out.append(name)
    return out
