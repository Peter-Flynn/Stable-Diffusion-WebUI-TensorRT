"""The VAE on TensorRT: its decoder and its encoder as weightless engines like the UNet's, refitted from
whichever VAE is loaded. This module says which VAEs the engines run, what sizes and precisions they're
built for, and which are on disk; vae_build.py builds them, vae_run.py decodes and encodes on them.

The engines that take a whole image (the decoder's "pre" and "post", the encoder's two) run in the
precision reForge itself runs that VAE in (BF16 where the GPU has it, else FP32; FP32 where it's FP16):
plain graphs, their GroupNorms TensorRT's own. Not FP16: a VAE's values outgrow it (SDXL's pass its
65504), and the way around that (below) needs each GroupNorm computed in FP32 inside an FP16 graph, which
TensorRT only does fast with an FP32 copy of the tensor. At a whole image's sizes those copies cost what
FP32 would or don't fit at all, and without them its kernels run several times slower than FP32's
(measured: 85% of a decode's time).

The engines that decode and encode in strips (strips.py) run in FP16 where the GPU runs it well, at half
the VRAM and on its tensor cores: a strip's GroupNorms are split between engines (the statistics in one,
applied in the next), which TensorRT runs fast at any size. They can't overflow, whatever the VAE, because
of where its nonlinearities are: every one (SiLU, the attention's softmax) comes right after a GroupNorm,
and a GroupNorm's output doesn't depend on its input's scale. So everything between two GroupNorms is
linear, and can run multiplied by any constant as long as what's added along the way (the convs' biases)
is multiplied too. FP16 strips run it multiplied by a power of two, 2^-EXPONENT, which costs floating
point nothing: what comes in and each conv's input after a GroupNorm are multiplied by it, the biases
added to scaled tensors are refitted multiplied by it, the statistics are taken in FP32, and the result
comes out as it is, the last GroupNorm having taken the scale out. Should it still come out NaN/Inf, the
exponent goes up for that VAE and it runs again.

The decoder runs in three parts: a "pre" engine up to its mid-block attention's queries, keys and values;
that attention on PyTorch, with reForge's own implementation; and a "post" engine for the rest. The
attention has one head 512 channels wide, which TensorRT's fused attention doesn't cover, so in an engine
it would hold its whole score matrix (about 17 GB in FP32 at 2048x2048); reForge's memory-efficient
kernels hold a fraction of that. The encoder runs the same three parts the other way round: an
"encode-pre" engine from the image down to its mid-block attention's queries, keys and values, that
attention on PyTorch, and an "encode-post" engine for the rest, whose output reForge's own code makes the
latent of.

Engines live in models/Unet-trt/<UNet architecture id>/vae/, built with that architecture's UNet engines
for the sizes its profiles cover."""

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
ENCODE_STRIP = "encode-strip-"  # an encode in strips' kind's engine: its ONNX graph's and plan's names start so
MAX_BATCH = 16  # images per engine call, at most
# FP16 engines (those for strips) scale the decoder and the encoder by 2^-EXPONENT. FP16 holds magnitudes up to 2^16 in steps no finer than
# 2^-24, so a tensor keeps FP16's precision scaled down until its typical value nears 2^-13 (measured: a
# decoder whose tensors are around 0.5 decodes as closely scaled by anything from 1 to 2^-12). At 6,
# tensors the VAE makes as large as 4 million fit, and ones as small as 0.01 keep their precision. A
# decode or encode that overflows anyway raises it by EXPONENT_STEP for that VAE's decoder or encoder, up
# to EXPONENT_MAX; past that, it runs on PyTorch.
EXPONENT, EXPONENT_STEP, EXPONENT_MAX = 6, 3, 12


# --- Which VAEs


def loaded_vae():
    from . import arch as arch_mod

    return getattr(getattr(arch_mod.loaded_model(), "forge_objects", None), "vae", None)


def is_decoder_weight(name: str) -> bool:
    return name.startswith("decoder.") or name.startswith("post_quant_conv.")


def unsupported(vae) -> str:
    """Why the engines can't decode for this VAE ("" if they can): they hold an SD 1.x/2.x/SDXL-style
    decoder, without attention past its mid block."""
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
    """What the decoder engines are built from: the decoder's layout, and its padding where it isn't zeros
    (reForge's "reflective padding" option)."""
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
    """The decoder's architecture: its config and its weights' shapes, not their values."""
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_decoder_weight(k))
    return hashlib.sha256(json.dumps([sorted(config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def is_encoder_weight(name: str) -> bool:
    return name.startswith("encoder.") or name.startswith("quant_conv.")


_ENCODES = {"AutoencodingEngine.encode": False, "AutoencodingEngineLegacy.encode": True}  # reForge's encodes the engines stand in for: whether it applies quant_conv


def encoder_unsupported(vae) -> str:
    """Why the engines can't encode for a VAE they decode for ("" if they can): they hold an SD 1.x/2.x/
    SDXL-style encoder, halving the image three times, without attention before its mid block, in a
    first stage model that encodes the way reForge's own do (the encoder, quant_conv if it has one, then
    its regularization)."""
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
    """What the encoder engines are built from: the encoder's layout, whether quant_conv follows it, and
    its padding where it isn't zeros."""
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
    """The encoder's architecture: its config and its weights' shapes, not their values."""
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_encoder_weight(k))
    return hashlib.sha256(json.dumps([sorted(encoder_config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def engine_dtype(vae):
    """The precision the whole-image engines (the decoder's pre and post, the encoder's two) run a VAE in:
    reForge's own for it, or FP32 where that's FP16 (see the module)."""
    return torch.float32 if vae.vae_dtype == torch.float16 else vae.vae_dtype


def strip_dtype(vae):
    """The precision the engines of a decode or an encode in strips run in: FP16 (scaled, so it can't
    overflow) where the whole-image engines' is FP32, reForge wasn't told to keep the VAE in FP32, and the
    GPU runs FP16 well; else theirs."""
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
    """Whether engines for strips in this precision run the VAE scaled: FP16's range is the one VAEs outgrow
    (BF16 and FP32 share one 2^127 wide)."""
    return dtype == torch.float16


def precisions(dtype, strips_dtype) -> str:
    """For people: "FP32 (FP16 in strips)", "BF16", ..."""
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
    """Whether a weight of the decoder or the encoder is a bias added to a scaled tensor (so it's refitted
    scaled): all the convs' but those whose inputs a GroupNorm made, or what one did make (the attention's
    queries', keys' and values', the last conv's, and quant_conv's after the encoder's)."""
    return name in _SCALED_BIASES or name.endswith(_SCALED_ENDINGS)


def weight_bytes(vae, names, dtype) -> int:
    """What the VAE's weights `names` take in this precision."""
    state = vae.first_stage_model.state_dict(keep_vars=True)
    return sum(state[n].numel() for n in names if n in state) * torch.tensor([], dtype=dtype).element_size()


def label(name: str) -> str:
    """For people: "BF16", "FP32", ..."""
    return {"bfloat16": "BF16", "float32": "FP32", "float16": "FP16"}.get(name, name)


TENSOR_LIMIT = 2**31 - 1  # elements a tensor can hold in TensorRT's kernels (its Myelin backend)
RESERVE = 2**30  # of the GPU's VRAM, for the CUDA context, the engines' weights and their kernels' scratch
RETRY_SHARE = 0.75  # of that budget, for a second try when TensorRT finds no kernel that fits the largest size
PLANNED = 1.2  # TensorRT's activation memory for a decode, as a multiple of peak() (4.32 GiB for 1472x1024 in FP32, TensorRT 11.4)


def gpu_vram():
    """The GPU's total VRAM, or None if it can't tell."""
    try:
        return int(torch.cuda.get_device_properties(common.mm().get_torch_device()).total_memory)
    except Exception:
        return None


def vram_budget(dtype=None, vram=None, share: float = 1.0):
    """Elements a decode may hold at once on this GPU in this precision (None if it can't tell): its VRAM
    less RESERVE, or a share of that."""
    if dtype is None or not vram:
        return None
    return max(0, int((vram - RESERVE) * share / torch.tensor([], dtype=dtype).element_size()))


def _levels(cfg: dict, h: int, w: int):
    """(channels coming in, channels of the blocks, pixels padded by a row and column on each side, as a
    conv's padding may hold them) at each of the decoder's resolutions, for latents of h x w."""
    ch, mult = cfg["ch"], cfg["ch_mult"]
    for i in range(len(mult)):
        scale = 2 ** (len(mult) - 1 - i)  # level i's size, in latents
        incoming = ch * mult[i + 1] if i + 1 < len(mult) else cfg["mid_channels"]
        yield incoming, ch * mult[i], (h * scale + 2) * (w * scale + 2)


def largest_tensor(cfg: dict, b: int, h: int, w: int) -> int:
    """Elements in the decoder's largest tensor for b latents of h x w: at each resolution, the most
    channels there, those of the blocks or of what's upsampled into them. For SD and SDXL VAEs, the 256
    channels upsampled to full size."""
    worst = b * cfg["latent_channels"] * (h + 2) * (w + 2)
    return max([worst] + [b * max(incoming, own) * pixels for incoming, own, pixels in _levels(cfg, h, w)])


def peak(cfg: dict, b: int, h: int, w: int) -> int:
    """Elements a decode of b latents of h x w holds at once, at its worst: at each resolution, the first
    block's input (kept for its shortcut) with its normalized copy and its first conv's output, or with two
    of the block's own; three of the later blocks' own; or the upsampling's input, its output and its conv's.
    For SD and SDXL VAEs, 256 + 256 + 128 channels at full size: 2.5 times the largest tensor."""
    return max(
        b * max(2 * incoming + own, incoming + 2 * own, 3 * own, 2 * incoming + incoming // 4) * pixels
        for incoming, own, pixels in _levels(cfg, h, w)
    )


def excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    """(how far over a decode of b latents of h x w is, as a ratio of elements (<= 1 if it fits), why):
    over TensorRT's limit for a tensor, or over the VRAM budget."""
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
    """h x w, or the largest size below it (keeping its aspect, in multiples of `step`) that over(h, w)
    puts at 1 or less."""
    ratio = over(h, w)
    if ratio > 1:
        scale = math.sqrt(1 / ratio) if ratio < math.inf else 0
        h, w = max(step, int(h * scale) // step * step), max(step, int(w * scale) // step * step)
        while over(h, w) > 1 and max(h, w) > step:
            h, w = (h - step, w) if h >= w else (h, w - step)
    return h, w


def shapes_for(profiles: list, cfg: dict, dtype=None, vram=None, share: float = 1.0) -> dict:
    """The engines' input shapes for the sizes a list of profiles covers: from the smallest to the largest
    of them, optimal at the first profile's optimal size, one image at a time up to the largest batch.
    The post engine's are capped so a decode fits TensorRT's limit for a tensor and, with the precision
    and the GPU's VRAM, so it could run at all (vram_budget, `share` of it): the size first (scaled down,
    keeping its aspect), then the batch at that size; larger decodes run in strips. The pre engine (at
    the latents' size, small) takes every size, for those."""
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
    """(channels coming in, channels of the blocks, pixels padded by a row and column on each side) at each
    of the encoder's resolutions, for images of h x w."""
    ch, mult = cfg["ch"], cfg["ch_mult"]
    for i in range(len(mult)):
        yield ch * (mult[i - 1] if i else 1), ch * mult[i], (h // 2**i + 2) * (w // 2**i + 2)


def encoder_largest(cfg: dict, b: int, h: int, w: int) -> int:
    """Elements in the encoder's largest tensor for b images of h x w. For SD and SDXL VAEs, the 128
    channels at full size."""
    worst = b * cfg["in_channels"] * (h + 2) * (w + 2)
    return max([worst] + [b * max(incoming, own) * pixels for incoming, own, pixels in _encoder_levels(cfg, h, w)])


def encoder_peak(cfg: dict, b: int, h: int, w: int) -> int:
    """Elements an encode of b images of h x w holds at once, at its worst: at each resolution, a block's
    input (kept for its shortcut) with its normalized copy and that one's SiLU. For SD and SDXL VAEs, three
    times the 128 channels at full size. (The decoder's, peak() times PLANNED, comes to the same count:
    three of its largest tensor.)"""
    return max(b * 3 * max(incoming, own) * pixels for incoming, own, pixels in _encoder_levels(cfg, h, w))


def encoder_excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    """As excess, for an encode of b images of h x w."""
    tensor = encoder_largest(cfg, b, h, w) / TENSOR_LIMIT
    memory = encoder_peak(cfg, b, h, w) / budget if budget else 0.0
    if budget is not None and budget <= 0:
        memory = math.inf
    return max(tensor, memory), "tensor" if tensor >= memory else "memory"


def encoder_shapes(profiles: list, cfg: dict, dtype=None, vram=None, share: float = 1.0) -> dict:
    """The encoder engines' input shapes for the sizes a list of profiles covers, as shapes_for: the image
    from the smallest of them to the largest (capped so an encode fits TensorRT's limit for a tensor and
    could run at all on this GPU); and what the mid block takes at an 8th of that, up to the largest's
    whatever the image's cap (an encode in strips gives the post engine that)."""
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
    mid = [[1, c, lo_h // 8, lo_w // 8], [1, c, opt_h // 8, opt_w // 8], [batch, c, full_h // 8, full_w // 8]]
    image = [[1, cfg["in_channels"], lo_h, lo_w], [1, cfg["in_channels"], opt_h, opt_w], [batch, cfg["in_channels"], hi_h, hi_w]]
    return {ENCODE_PRE: {"image": image}, ENCODE_POST: {"x": mid, "attention": [list(shape) for shape in mid]}}


def strip_shapes(profiles: list, found: list, dtype=torch.float32) -> dict:
    """{kind name: its engine's shapes} for the stripped decode, in this precision, of the sizes a list of
    profiles covers (of any height: only a strip's rows go through an engine at once)."""
    lo_w = min(p.width[0] for p in profiles) // 8
    hi = (max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8)
    opt = (profiles[0].height[1] // 8, profiles[0].width[1] // 8)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return {name: strips.kind_shapes(kind, scales, lo_w, opt, hi, itemsize) for name, (kind, scales) in strips.kinds(found).items()}


def _covers(outer: dict, inner: dict) -> bool:
    """Whether the shapes `outer` accepts include all of `inner`'s."""
    for part, shapes in inner.items():
        for name, (lo, _, hi) in shapes.items():
            o_lo, _, o_hi = outer.get(part, {}).get(name, ([1] * 4, None, [0] * 4))
            if any(a < b for a, b in zip(lo, o_lo)) or any(a > b for a, b in zip(hi, o_hi)):
                return False
    return True


# --- Engines on disk


def folder(arch_id: str) -> str:
    return os.path.join(store.root(), arch_id, "vae")


def path_prefix(arch_id: str) -> str:
    return os.path.join(folder(arch_id), f"{common.gpu_tag()}-{common.trt_tag()}")


@dataclass
class Record:
    base: str  # path without extension
    meta: dict = field(repr=False)

    @property
    def key(self) -> str:
        return self.meta["key"]

    def plan(self, part: str) -> str:
        return f"{self.base}.{part}.plan"

    @property
    def dtype(self):
        """The whole-image engines' precision."""
        return getattr(torch, self.meta["dtype"])

    @property
    def strips_dtype(self):
        """The precision of the engines for strips."""
        return getattr(torch, self.meta["strips_dtype"])

    @property
    def scaled(self) -> bool:
        """Whether the engines for strips run scaled (FP16 ones do)."""
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
        """The encoder engines' {"digest", "config", "shapes", "asked", "weights", "stripped", "weight_bytes",
        "strips"}, or None if the record has none."""
        return self.meta.get("encoder") or None

    @property
    def encoder_strips(self) -> dict:
        """{kind name: {"shapes", "weights", "stripped"}}: the engines that encode in strips."""
        return (self.encoder or {}).get("strips") or {}

    def encoder_strip_plan(self, kind: str) -> str:
        return f"{self.base}.encode-strip-{kind}.plan"

    def encoder_strips_cover(self, found: list, h: int, w: int) -> bool:
        """Whether an encode in strips of these stages runs images of h x w on these engines (and the
        encode-post engine takes their latents)."""
        return bool(self.encoder_strips) and self._post_encodes(h // 8, w // 8) and _kinds_cover(self.encoder_strips, found, w // 8)

    def _post_encodes(self, h: int, w: int) -> bool:
        lo, _, hi = self.encoder["shapes"][ENCODE_POST]["x"]
        return lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def has_encoder(self, digest: str, c: int) -> bool:
        """Whether it has engines for an encoder of this architecture, for images of c channels."""
        enc = self.encoder
        return enc is not None and enc.get("digest") == digest and c == enc["shapes"][ENCODE_PRE]["image"][0][1]

    def encodes(self, h: int, w: int) -> bool:
        """Whether the encoder engines take images of h x w whole."""
        lo, _, hi = self.encoder["shapes"][ENCODE_PRE]["image"]
        return lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    @property
    def encode_batch(self) -> int:
        return int(self.encoder["shapes"][ENCODE_PRE]["image"][2][0])

    def encode_reach(self) -> str:
        """For people: "1824x1824, 1 at a time, and in strips up to 2944x2944"."""
        b, _, h, w = self.encoder["shapes"][ENCODE_PRE]["image"][2]
        text = f"{w}x{h}, {b} at a time"
        if self.encoder_strips:
            _, _, ph, pw = self.encoder["shapes"][ENCODE_POST]["x"][2]
            text += f", and in strips up to {pw * 8}x{ph * 8}"
        return text

    @property
    def strips(self) -> dict:
        """{kind name: {"kind", "shapes", "weights", "stripped"}}: the stripped decode's engines."""
        return self.meta.get("strips") or {}

    def _pre_covers(self, c: int, h: int, w: int) -> bool:
        lo, _, hi = self.meta["shapes"][PRE]["latent"]
        return c == lo[1] and lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def covers(self, b: int, c: int, h: int, w: int) -> bool:
        """Whether the post engine decodes latents of h x w (as many at a time as max_batch)."""
        lo, _, hi = self.meta["shapes"][POST]["x"]
        return self._pre_covers(c, h, w) and lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    def strips_cover(self, found: list, c: int, h: int, w: int) -> bool:
        """Whether the stripped decode of these stages runs latents of h x w on these engines."""
        return bool(self.strips) and self._pre_covers(c, h, w) and _kinds_cover(self.strips, found, w)

    @property
    def max_batch(self) -> int:
        return int(self.meta["shapes"][POST]["x"][2][0])

    def reach(self) -> str:
        """For people: "1824x1824, 1 at a time, and in strips up to 2944x2944"."""
        b, _, h, w = self.meta["shapes"][POST]["x"][2]
        text = f"{w * 8}x{h * 8}, {b} at a time"
        if self.strips:
            _, _, ph, pw = self.meta["shapes"][PRE]["latent"][2]
            text += f", and in strips up to {pw * 8}x{ph * 8}"
        return text


def _kinds_cover(kinds: dict, found: list, w: int) -> bool:
    """Whether engines of these kinds ({name: {"shapes", ...}}) run every stage found, for latents w wide."""
    for s in found:
        entry = kinds.get(s.kind.name)
        if entry is None or not strips.engine_rows(entry["shapes"], s.kind, w * s.scale):
            return False
    return True


def records(arch_id: str) -> list:
    """This GPU's and TensorRT version's VAE engines for an architecture."""
    found = []
    for path in glob.glob(glob.escape(path_prefix(arch_id)) + "-*.json"):
        try:
            base = path[: -len(".json")]
            meta = common.read_json(path)  # (shared: changed by copying)
            r = Record(base, meta)
            if r.loadable and all(os.path.isfile(r.plan(p)) for p in (PRE, POST)):
                if not all(os.path.isfile(r.strip_plan(k)) for k in r.strips):
                    meta = {**meta, "strips": {}}  # decodes on the post engine only; the tab asks to build them again
                if r.encoder is not None and not all(os.path.isfile(r.plan(p)) for p in ENCODER):
                    meta = {**meta, "encoder": None}  # likewise: it encodes on PyTorch
                elif r.encoder is not None and not all(os.path.isfile(r.encoder_strip_plan(k)) for k in r.encoder_strips):
                    meta = {**meta, "encoder": {**r.encoder, "strips": {}}}  # encodes whole only; likewise
                found.append(Record(base, meta))
        except Exception as e:
            logger.debug(f"Ignoring {path}, which couldn't be read: {e}")
    return found


def record_for(arch_id: str, vae):
    """The engines that decode for this VAE: its decoder's architecture, in its precision if there are
    engines in several."""
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
    """Deletes an architecture's VAE engines for one GPU and TensorRT version (with its UNet engines)."""
    for path in glob.glob(os.path.join(glob.escape(folder(arch_id)), f"{gpu}-{common.trt_tag(trt)}-*.json")):
        remove_record(path[: -len(".json")])


def available() -> bool:
    """Whether TensorRT is installed and new enough (the tab says when it isn't)."""
    try:
        common.trt_version()
        return True
    except Exception:
        return False


def needs_build(arch, profiles: list, vae=None) -> str:
    """What of the VAE's engines building these profiles builds too ("" if nothing): none for the loaded
    VAE's decoder and precision cover the profiles' sizes, or they lack its encoder's."""
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
    """{"digest", "config", "shapes", "strips"} of the encoder engines for the profiles, "strips" being the
    encode in strips' kinds and their shapes (None if the engines can't encode for this VAE)."""
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
    """The records for this VAE's decoder in these precisions (the whole-image engines', the strips')."""
    wanted = digest(vae)
    return [r for r in records(arch_id) if r.meta.get("digest") == wanted and (r.dtype, r.strips_dtype) == (dtype, sdtype)]


def strips_wanted(vae, profiles: list, cfg: dict, dtype, encoding: bool = False) -> dict:
    """The kinds of the decode in strips (or the encode's) and their shapes for the profiles ({} if it can't
    run for this VAE)."""
    try:
        lists = (strips.encoder_every_way if encoding else strips.every_way)(vae.first_stage_model, cfg["padding"])
    except (ValueError, KeyError, AttributeError, StopIteration):
        return {}
    return {} if any(strips.supported(found) for found in lists) else strip_shapes(profiles, lists, dtype)


def parts_missing(record: Record, shapes: dict, strip_wanted: dict, enc_wanted: dict = None) -> list:
    """The parts (PRE, POST, the ENCODER's two together, the decode in strips' kind names, the encode's
    prefixed ENCODE_STRIP) a record lacks for these shapes. A post or encoder engine that had to be built
    smaller than asked counts as built (its "asked"), and so do the encoder's or the kinds' whose build
    failed for these shapes. The kinds for encoding in strips are kept whatever becomes of the encoder's
    two, as long as they're for the same encoder."""
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
        if same:  # its strips' kinds, unless built or their build failed for these shapes
            kinds = have.get("strips") or {}
            failed = have.get("strips_asked") if have.get("strips_error") else None
            out += [
                ENCODE_STRIP + name for name, wanted in enc_wanted["strips"].items()
                if not (name in kinds and _covers({name: kinds[name]["shapes"]}, {name: wanted}))
                and not (failed and name in failed and _covers({name: failed[name]}, {name: wanted}))
            ]
        elif ENCODE_PRE in out:  # another encoder's, or none: all of them, with its two
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
