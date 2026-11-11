"""The VAE on TensorRT: its decoder and its encoder as weightless engines like the UNet's, refitted from
whichever VAE is loaded, in the precision reForge itself picks for that VAE when they're built (BF16
where the GPU has it, or what a --vae-in-* flag asks for), except where it picks FP32 only for want of
BF16: there the engines run in FP16, at half the VRAM and on the GPU's tensor cores.

VAEs overflow in FP16 (SDXL's makes values past its 65504), which is why reForge avoids it. The engines
don't, whatever the VAE, because of where its nonlinearities are: every one (SiLU, the attention's
softmax) comes right after a GroupNorm, and a GroupNorm's output doesn't depend on its input's scale.
So everything between two GroupNorms is linear, and can run multiplied by any constant as long as
what's added along the way (the convs' biases) is multiplied too. FP16 engines run it multiplied by a
power of two, 2^-EXPONENT, which costs floating point nothing: what comes in (the latent, or the image
to encode) and each resnet block's conv inputs are multiplied by it (an input of the graphs, "scale"),
the biases added to scaled tensors are refitted multiplied by it, GroupNorms compute in FP32, and the
result comes out as it is, the last GroupNorm having taken the scale out. Should it still come out
NaN/Inf, the exponent goes up for that VAE and it runs again.

The decoder runs in three parts: a "pre" engine up to its mid-block attention's queries, keys and values;
that attention on PyTorch, with reForge's own implementation; and a "post" engine for the rest. The
attention has one head 512 channels wide, which TensorRT's fused attention doesn't cover, so in an engine
it would hold its whole score matrix (about 17 GB in FP32 at 2048x2048); reForge's memory-efficient
kernels hold a fraction of that.

Each decode takes the fastest way there can be room for, decided before it starts: the engines, as many
images at a time as fit; else the post part in strips (strips.py) on engines of its own, holding in RAM
what doesn't fit in VRAM; else reForge's own decode. Room is made for exactly what that takes, the VAE's
engines it doesn't run on going first, then other models by reForge's manager, a UNet engine last and
only if that makes the room. Each engine loads when it's first used.

The encoder (an img2img's or an inpaint's image, a Hires. fix's upscaled one) runs the same three parts
the other way round: an "encode-pre" engine from the image down to its mid-block attention's queries,
keys and values, that attention on PyTorch, and an "encode-post" engine for the rest, whose output
reForge's own code makes the latent of. An encode there can't be room for whole runs in strips (strips.py's
Encode, on engines of its own) up to the mid block's attention, which with its GroupNorm and its queries, keys
and values runs on PyTorch at the latent's size, then the encode-post engine; failing that, on PyTorch.

Engines live in models/Unet-trt/<UNet architecture id>/vae/, built with that architecture's UNet engines
for the sizes its profiles cover."""

import contextlib
import gc
import glob
import hashlib
import inspect
import json
import math
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field

import torch

from . import common, gpu, store, strips
from .common import log, notify

FORMAT = 1
PRE, POST = "pre", "post"
ENCODE_PRE, ENCODE_POST = "encode-pre", "encode-post"
ENCODER = (ENCODE_PRE, ENCODE_POST)
ENCODE_STRIP = "encode-strip-"  # an encode in strips' kind's engine: its ONNX graph's and plan's names start so
MAX_BATCH = 16  # images per engine call, at most
# FP16 engines scale the decoder and the encoder by 2^-EXPONENT. FP16 holds magnitudes up to 2^16 in steps no finer than
# 2^-24, so a tensor keeps FP16's precision scaled down until its typical value nears 2^-13 (measured: a
# decoder whose tensors are around 0.5 decodes as closely scaled by anything from 1 to 2^-12). At 6,
# tensors the VAE makes as large as 4 million fit, and ones as small as 0.01 keep their precision. A
# decode or encode that overflows anyway raises it by EXPONENT_STEP for that VAE's decoder or encoder, up
# to EXPONENT_MAX; past that, it runs on PyTorch.
EXPONENT, EXPONENT_STEP, EXPONENT_MAX = 6, 3, 12
# The scaled graphs' version, in a record's "scaled": 2, each GroupNorm PyTorch's own entirely in FP32. Engines
# of an older one (True: the GroupNorm's scale and shift after a cast inside its sequence) are built again.
SCALED = 2


# --- Which VAEs


def loaded_vae():
    from . import arch as arch_mod

    return getattr(getattr(arch_mod.loaded_model(), "forge_objects", None), "vae", None)


def _relevant(name: str) -> bool:
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
        "padding": {n: m.padding_mode for n, m in fsm.named_modules() if _relevant(n + ".") and isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"},
    }


def digest(vae) -> str:
    """The decoder's architecture: its config and its weights' shapes, not their values."""
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if _relevant(k))
    return hashlib.sha256(json.dumps([sorted(config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def _encoder_relevant(name: str) -> bool:
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
        "padding": {n: m.padding_mode for n, m in fsm.named_modules() if _encoder_relevant(n + ".") and isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"},
    }


def encoder_digest(vae) -> str:
    """The encoder's architecture: its config and its weights' shapes, not their values."""
    shapes = sorted((k, list(v.shape)) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if _encoder_relevant(k))
    return hashlib.sha256(json.dumps([sorted(encoder_config(vae).items()), shapes], sort_keys=True).encode()).hexdigest()[:10]


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def engine_dtype(vae):
    """The precision the engines run a VAE in: reForge's for it, or FP16 (scaled, so it can't overflow)
    where reForge fell back to FP32 without being asked to and the GPU runs FP16 well."""
    dtype = vae.vae_dtype
    if dtype == torch.float32:
        try:
            mm = common.mm()
            if not getattr(mm.args, "vae_in_fp32", False) and mm.should_use_fp16(mm.get_torch_device()):
                return torch.float16
        except Exception:
            pass
    return dtype


def scaled(dtype) -> bool:
    """Whether engines in this precision run the VAE scaled: FP16's range is the one VAEs outgrow (BF16
    and FP32 share one 2^127 wide)."""
    return dtype == torch.float16


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


def label(name: str) -> str:
    """For people: "BF16", "FP32", ..."""
    return {"bfloat16": "BF16", "float32": "FP32", "float16": "FP16"}.get(name, name)


TENSOR_LIMIT = 2**31 - 1  # elements a tensor can hold in TensorRT's kernels (its Myelin backend)
RESERVE = 2**30  # of the GPU's VRAM, for the CUDA context, the engines' weights and their kernels' scratch
RETRY_SHARE = 0.75  # of that budget, for a second try when TensorRT finds no kernel that fits the largest size
PLANNED = 1.2  # TensorRT's activation memory for a decode, as a multiple of peak() (4.32 GiB for 1472x1024 in FP32, TensorRT 11.4)


def _vram():
    """The GPU's total VRAM, or None if it can't tell."""
    try:
        return int(torch.cuda.get_device_properties(common.mm().get_torch_device()).total_memory)
    except Exception:
        return None


def _budget(dtype=None, vram=None, share: float = 1.0):
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


def _excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    """(how far over a decode of b latents of h x w is, as a ratio of elements (<= 1 if it fits), why):
    over TensorRT's limit for a tensor, or over the VRAM budget."""
    tensor = largest_tensor(cfg, b, h, w) / TENSOR_LIMIT
    memory = peak(cfg, b, h, w) * PLANNED / budget if budget else 0.0
    if budget is not None and budget <= 0:
        memory = math.inf
    return max(tensor, memory), "tensor" if tensor >= memory else "memory"


def _why(cause: str, dtype, vram, what: str = "decode") -> str:
    if cause == "tensor":
        return f"TensorRT can't hold a larger {what}'s largest tensor (2^31 elements)"
    return f"a larger {what} wouldn't fit in this GPU's {common.gib(vram)} in {label(dtype_name(dtype))}, even alone"


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
    and the GPU's VRAM, so it could run at all (_budget, `share` of it): the size first (scaled down,
    keeping its aspect), then the batch at that size; larger decodes run in strips. The pre engine (at
    the latents' size, small) takes every size, for those."""
    budget = _budget(dtype, vram, share)
    lo_h, lo_w = min(p.height[0] for p in profiles) // 8, min(p.width[0] for p in profiles) // 8
    hi_h, hi_w = max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8
    full_h, full_w = hi_h, hi_w
    hi_h, hi_w = _shrunk(hi_h, hi_w, lambda h, w: _excess(cfg, 1, h, w, budget)[0])
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


def _encoder_excess(cfg: dict, b: int, h: int, w: int, budget) -> tuple:
    """As _excess, for an encode of b images of h x w."""
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
    budget = _budget(dtype, vram, share)
    lo_h, lo_w = min(p.height[0] for p in profiles) // 8 * 8, min(p.width[0] for p in profiles) // 8 * 8
    hi_h, hi_w = max(p.height[2] for p in profiles) // 8 * 8, max(p.width[2] for p in profiles) // 8 * 8
    full_h, full_w = hi_h, hi_w
    hi_h, hi_w = _shrunk(hi_h, hi_w, lambda h, w: _encoder_excess(cfg, 1, h, w, budget)[0], 8)
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


def _prefix(arch_id: str) -> str:
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
        return getattr(torch, self.meta["dtype"])

    @property
    def scaled(self) -> bool:
        """Whether the engines run the decoder scaled (FP16 ones do: their graphs take the scale)."""
        return self.meta.get("scaled") == SCALED

    @property
    def stale(self) -> bool:
        """Scaled engines of an older graph: built again rather than run."""
        return bool(self.meta.get("scaled")) and not self.scaled

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
    for path in glob.glob(glob.escape(_prefix(arch_id)) + "-*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            r = Record(path[: -len(".json")], meta)
            if r.stale:
                if r.base not in _told:
                    _told.add(r.base)
                    log("The VAE's FP16 engines were built with an older graph; build again in the TensorRT tab to replace them.")
                continue
            if meta.get("vae_format") == FORMAT and r.loadable and all(os.path.isfile(r.plan(p)) for p in (PRE, POST)):
                if not all(os.path.isfile(r.strip_plan(k)) for k in r.strips):
                    meta["strips"] = {}  # decodes on the post engine only; the tab asks to build them again
                if r.encoder is not None and not all(os.path.isfile(r.plan(p)) for p in ENCODER):
                    meta["encoder"] = None  # likewise: it encodes on PyTorch
                if r.encoder is not None and not all(os.path.isfile(r.encoder_strip_plan(k)) for k in r.encoder_strips):
                    meta["encoder"]["strips"] = {}  # encodes whole only; likewise
                found.append(r)
        except Exception as e:
            log(f"Ignoring {path}: {e}")
    return found


def record_for(arch_id: str, vae):
    """The engines that decode for this VAE: its decoder's architecture, in its precision if there are
    engines in several."""
    try:
        wanted = digest(vae)
        matching = [r for r in records(arch_id) if r.meta.get("digest") == wanted]
    except Exception as e:
        log(f"Could not look up the VAE decoder engines: {e}")
        return None
    wanted = engine_dtype(vae)
    return next((r for r in matching if r.dtype == wanted and r.scaled == scaled(wanted)), matching[0] if matching else None)


def _remove(base: str):
    for path in [base + ".json"] + glob.glob(glob.escape(base) + ".*.plan"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def delete_for(arch_id: str, gpu: str, trt: str):
    """Deletes an architecture's VAE engines for one GPU and TensorRT version (with its UNet engines)."""
    for path in glob.glob(os.path.join(glob.escape(folder(arch_id)), f"{gpu}-{common.trt_tag(trt)}-*.json")):
        _remove(path[: -len(".json")])


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
        cfg, dtype = config(vae), engine_dtype(vae)
        shapes = shapes_for(profiles, cfg, dtype, _vram())
        strip_wanted = _strips_wanted(vae, profiles, cfg, dtype)
        enc_wanted = _encoder_wanted(vae, profiles, dtype, _vram())
        lacking = [_missing(r, shapes, strip_wanted, enc_wanted) for r in _matching(arch.id, vae, dtype)]
        if any(not parts for parts in lacking):
            return ""
    except Exception as e:
        log(f"Could not check the VAE's engines: {e}")
        return ""
    only_encoder = any(all(part in ENCODER for part in parts) for parts in lacking)
    what = "VAE encoder" if only_encoder else "VAE decoder and encoder" if enc_wanted else "VAE decoder"
    return f"the {what}, in {label(dtype_name(dtype))}"


def _encoder_wanted(vae, profiles: list, dtype, vram) -> dict:
    """{"digest", "config", "shapes", "strips"} of the encoder engines for the profiles, "strips" being the
    encode in strips' kinds and their shapes (None if the engines can't encode for this VAE)."""
    try:
        if encoder_unsupported(vae):
            return None
        cfg = encoder_config(vae)
        return {
            "digest": encoder_digest(vae), "config": cfg, "shapes": encoder_shapes(profiles, cfg, dtype, vram),
            "strips": _strips_wanted(vae, profiles, cfg, dtype, encoding=True),
        }
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _matching(arch_id: str, vae, dtype) -> list:
    """The records for this VAE's decoder in this precision (and scaled, if that precision is)."""
    wanted, name = digest(vae), dtype_name(dtype)
    return [r for r in records(arch_id) if r.meta.get("digest") == wanted and r.meta.get("dtype") == name and r.scaled == scaled(dtype)]


def _strips_wanted(vae, profiles: list, cfg: dict, dtype, encoding: bool = False) -> dict:
    """The kinds of the decode in strips (or the encode's) and their shapes for the profiles ({} if it can't
    run for this VAE)."""
    try:
        lists = (strips.encoder_every_way if encoding else strips.every_way)(vae.first_stage_model, cfg["padding"])
    except (ValueError, KeyError, AttributeError, StopIteration):
        return {}
    return {} if any(strips.supported(found) for found in lists) else strip_shapes(profiles, lists, dtype)


def _missing(record: Record, shapes: dict, strip_wanted: dict, enc_wanted: dict = None) -> list:
    """The parts (PRE, POST, the ENCODER's two together, kind names) a record lacks for these shapes. A post
    or encoder engine that had to be built smaller than asked counts as built (its "asked"), and so do the
    encoder's or the kinds' whose build failed for these shapes."""
    meta = record.meta
    out = []
    if not _covers(meta["shapes"], {PRE: shapes[PRE]}):
        out.append(PRE)
    if not _covers(meta.get("asked", meta["shapes"]), {POST: shapes[POST]}):
        out.append(POST)
    if enc_wanted is not None:
        have = record.encoder
        built = have is not None and have.get("digest") == enc_wanted["digest"] and _covers(have.get("asked", have["shapes"]), enc_wanted["shapes"])
        if built:  # and its strips, or a failed build of them for these shapes
            kinds = have.get("strips") or {}
            failed = have.get("strips_asked") if have.get("strips_error") else None
            built = all(
                (name in kinds and _covers({name: kinds[name]["shapes"]}, {name: wanted}))
                or (failed and name in failed and _covers({name: failed[name]}, {name: wanted}))
                for name, wanted in enc_wanted["strips"].items()
            )
        tried = meta.get("encoder_asked") if meta.get("encoder_error") else None
        if not built and not (tried and tried.get("digest") == enc_wanted["digest"] and _covers(tried["shapes"], enc_wanted["shapes"])):
            out += list(ENCODER)
    failed = meta.get("strips_asked") if meta.get("strips_error") else None
    for name, kind_shapes in strip_wanted.items():
        have = record.strips.get(name)
        if have is not None and _covers({name: have["shapes"]}, {name: kind_shapes}):
            continue
        if failed and name in failed and _covers({name: failed[name]}, {name: kind_shapes}):
            continue
        out.append(name)
    return out


# --- Export and build


class _Scale:
    """What a scaled decoder or encoder computes with: the scale its blocks multiply by (the graph's input,
    while it's traced)."""

    value = None


def _scaled_norm(norm, holder: _Scale):
    """A GroupNorm's forward on a scaled tensor: all of it in FP32 (its scale and shift too, `norm` being
    FP32), on its input divided by the scale, then cast back. FP32 has the range for the values as the VAE
    makes them, their squares included, and so the GroupNorm's epsilon is added to the variance it was meant
    for (to a scaled one, it would be thousands of times too large).

    The GroupNorm itself is PyTorch's own: it exports as the sequence the FP32 engines' GroupNorms are
    (Reshape, InstanceNormalization, Reshape, Mul, Add), with nothing inside it; the casts and the division
    come before and after."""

    def forward(x):
        normed = torch.nn.functional.group_norm(x.float() / holder.value.float(), norm.num_groups, norm.weight, norm.bias, norm.eps)
        return normed.to(x.dtype)

    return forward


def scale_decoder(module) -> _Scale:
    """Makes a decoder or encoder (vae.fresh's, or a stand-in) compute scaled, for tracing: each resnet
    block's convs take their input times the scale (the returned holder's value), and every GroupNorm
    computes on its input divided by it (_scaled_norm). With what comes in multiplied by the scale (_Pre,
    _Post), and the biases refitted scaled (scaled_bias), the tensors between GroupNorms are all scaled
    alike."""
    holder = _Scale()
    for m in module.modules():
        if isinstance(m, torch.nn.GroupNorm):
            m.float()  # its scale and shift, as the engines take them (a refit gives them in FP32)
            m.forward = _scaled_norm(m, holder)
        elif all(hasattr(m, part) for part in ("norm1", "conv1", "norm2", "conv2")):
            for conv in (m.conv1, m.conv2):
                conv.forward = (lambda inner: lambda x: inner(x * holder.value))(conv.forward)
    return holder


class _Pre(torch.nn.Module):
    """The decoder up to its mid-block attention: (that attention's input, its queries, keys, values).
    With a scale's holder (scale_decoder), it takes the scale too, and the attention's input comes out
    scaled."""

    def __init__(self, fsm, holder: _Scale = None):
        super().__init__()
        self.post_quant_conv = getattr(fsm, "post_quant_conv", None)
        self.decoder = fsm.decoder
        self.holder = holder

    def forward(self, latent, scale=None):
        if self.holder is not None:
            self.holder.value = scale
            latent = latent * scale
        z = self.post_quant_conv(latent) if self.post_quant_conv is not None else latent
        d = self.decoder
        x = d.mid.block_1(d.conv_in(z), None)
        attn = d.mid.attn_1
        h = attn.norm(x)
        return x, attn.q(h), attn.k(h), attn.v(h)


class _Post(torch.nn.Module):
    """The rest of the decoder, from the mid-block attention's input and output: the image. With a scale's
    holder, it takes the scale too, and the attention's input scaled (its output as it is)."""

    def __init__(self, fsm, holder: _Scale = None):
        super().__init__()
        self.decoder = fsm.decoder
        self.holder = holder

    def forward(self, x, attention, scale=None):
        if self.holder is not None:
            self.holder.value = scale
            attention = attention * scale
        d = self.decoder
        h = d.mid.block_2(x + d.mid.attn_1.proj_out(attention), None)
        for i_level in reversed(range(d.num_resolutions)):
            for i_block in range(d.num_res_blocks + 1):
                h = d.up[i_level].block[i_block](h, None)
            if i_level != 0:
                h = d.up[i_level].upsample(h)
        h = d.norm_out(h)
        return d.conv_out(h * torch.sigmoid(h))  # reForge's nonlinearity


class _EncodePre(torch.nn.Module):
    """The encoder up to its mid-block attention: (that attention's input, its queries, keys, values). With
    a scale's holder, it takes the scale too, and the attention's input comes out scaled."""

    def __init__(self, fsm, holder: _Scale = None):
        super().__init__()
        self.encoder = fsm.encoder
        self.holder = holder

    def forward(self, image, scale=None):
        if self.holder is not None:
            self.holder.value = scale
            image = image * scale
        e = self.encoder
        h = e.conv_in(image)
        for i_level in range(e.num_resolutions):
            for i_block in range(e.num_res_blocks):
                h = e.down[i_level].block[i_block](h, None)
            if i_level != e.num_resolutions - 1:
                h = e.down[i_level].downsample(h)
        x = e.mid.block_1(h, None)
        attn = e.mid.attn_1
        h = attn.norm(x)
        return x, attn.q(h), attn.k(h), attn.v(h)


class _EncodePost(torch.nn.Module):
    """The rest of the encoder, from the mid-block attention's input and output, and quant_conv where the
    first stage model applies it: the latent's distribution ("moments"), as it is. With a scale's holder,
    it takes the scale too, and the attention's input scaled."""

    def __init__(self, fsm, holder: _Scale = None):
        super().__init__()
        self.encoder = fsm.encoder
        self.quant_conv = getattr(fsm, "quant_conv", None)
        self.holder = holder

    def forward(self, x, attention, scale=None):
        if self.holder is not None:
            self.holder.value = scale
            attention = attention * scale
        e = self.encoder
        h = e.mid.block_2(x + e.mid.attn_1.proj_out(attention), None)
        h = e.norm_out(h)
        h = e.conv_out(h * torch.sigmoid(h))  # reForge's nonlinearity
        return self.quant_conv(h) if self.quant_conv is not None else h


def fresh(cfg: dict, device, dtype):
    """A decoder of this layout from reForge's own classes, with none of a loaded one's offloading hooks."""
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.model import Decoder

    module = torch.nn.Module()
    module.decoder = Decoder(
        ch=cfg["ch"], out_ch=cfg["out_ch"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=3, resolution=256, z_channels=cfg["z_channels"], dropout=0.0,
    )
    if cfg["post_quant_conv"]:
        module.post_quant_conv = ops.disable_weight_init.Conv2d(cfg["latent_channels"], cfg["z_channels"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def fresh_encoder(cfg: dict, device, dtype):
    """An encoder of this layout from reForge's own classes, likewise (with quant_conv where it's applied)."""
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.model import Encoder

    module = torch.nn.Module()
    module.encoder = Encoder(
        ch=cfg["ch"], out_ch=cfg["in_channels"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=cfg["in_channels"], resolution=256, z_channels=cfg["out_channels"], double_z=False, dropout=0.0,
    )
    if cfg["quant_conv"]:
        module.quant_conv = ops.disable_weight_init.Conv2d(cfg["out_channels"], cfg["moments"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def _export(module, halves: dict, work: str, dtype, device, first, loaded: dict) -> dict:
    """Traces a fresh module's two halves ({part: (the half's class, its inputs' names, its outputs', their
    dynamic axes)}, the first one fed `first`, the second its first output and one like it) to
    work/vae_<part>.onnx, scaled if the precision is: {part: {weight name: shape}}."""
    from . import export

    holder = scale_decoder(module) if scaled(dtype) else None
    for t in module.state_dict(keep_vars=True).values():  # random: tracing doesn't depend on the weights, and
        if t.is_floating_point():  # distinct values keep the exporter from merging identical ones (GroupNorm's)
            t.normal_(0, 0.02)
    scale = (torch.full((1, 1, 1, 1), 2.0**-EXPONENT, dtype=dtype, device=device),) if holder is not None else ()
    more = ["scale"] if holder is not None else []
    (one, two), parts = list(halves), {part: spec[0](module, holder) for part, spec in halves.items()}
    x = parts[one](first, *scale)[0]
    args = {one: (first, *scale), two: (x, torch.randn_like(x), *scale)}
    kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    exported = {k: list(v.shape) for k, v in module.state_dict(keep_vars=True).items()}
    result = {}
    for part, (_, inputs, outputs, axes) in halves.items():
        path = os.path.join(work, f"vae_{part}.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(parts[part], args[part], path, input_names=inputs + more, output_names=outputs, dynamic_axes=axes, **kwargs)
        result[part] = export._weightless_graph(path, inputs + more, (exported, loaded))
    return result


_AXES = {0: "batch", 2: "height", 3: "width"}
_LARGER = {0: "batch", 2: "image_height", 3: "image_width"}


@common.inference_mode
def export_onnx(vae, work: str, dtype, device) -> dict:
    """Writes work/vae_pre.onnx and work/vae_post.onnx (scaled, in FP16); returns {part: {weight name:
    shape}, "config"}."""
    cfg = config(vae)
    halves = {
        PRE: (_Pre, ["latent"], ["x", "q", "k", "v"], {n: _AXES for n in ("latent", "x", "q", "k", "v")}),
        POST: (_Post, ["x", "attention"], ["image"], {"x": _AXES, "attention": _AXES, "image": _LARGER}),
    }
    latent = torch.randn(1, cfg["latent_channels"], 32, 32, dtype=dtype, device=device)
    loaded = {k: list(v.shape) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if _relevant(k)}
    return {"config": cfg, **_export(fresh(cfg, device, dtype), halves, work, dtype, device, latent, loaded)}


@common.inference_mode
def export_encoder_onnx(vae, work: str, dtype, device) -> dict:
    """Writes work/vae_encode-pre.onnx and work/vae_encode-post.onnx (scaled, in FP16); returns {part:
    {weight name: shape}, "config"}."""
    cfg = encoder_config(vae)
    halves = {
        ENCODE_PRE: (_EncodePre, ["image"], ["x", "q", "k", "v"], {"image": _LARGER, **{n: _AXES for n in ("x", "q", "k", "v")}}),
        ENCODE_POST: (_EncodePost, ["x", "attention"], ["moments"], {n: _AXES for n in ("x", "attention", "moments")}),
    }
    image = torch.randn(1, cfg["in_channels"], 128, 128, dtype=dtype, device=device)
    loaded = {k: list(v.shape) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if _encoder_relevant(k)}
    return {"config": cfg, **_export(fresh_encoder(cfg, device, dtype), halves, work, dtype, device, image, loaded)}


@common.inference_mode
def export_strips(vae, work: str, dtype, device, names, encoding: bool = False) -> dict:
    """Writes the ONNX graphs of the kinds named, of the decode in strips (work/strip-<kind>.onnx) or the
    encode's (work/encode-strip-<kind>.onnx); returns {kind name (prefixed so, for the encode's): {weight name:
    shape}}. Each from a stage of that kind in a fresh decoder or encoder of the VAE's layout."""
    from . import export

    if encoding:
        cfg = encoder_config(vae)
        module, every_way, prefix, key = fresh_encoder(cfg, device, dtype), strips.encoder_every_way, ENCODE_STRIP, ENCODE_STRIP
    else:
        cfg = config(vae)
        module, every_way, prefix, key = fresh(cfg, device, dtype), strips.every_way, "strip-", ""
    for t in module.state_dict(keep_vars=True).values():  # random, as in export_onnx
        if t.is_floating_point():
            t.normal_(0, 0.02)
    found = [s for each in every_way(module, cfg["padding"]) for s in each]
    loaded = vae.first_stage_model.state_dict(keep_vars=True)
    kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    result = {}
    for name in names:
        stage = next(s for s in found if s.kind.name == name)
        kind, part = stage.kind, strips.StageModule(stage, module)
        path = os.path.join(work, f"{prefix}{name}.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(
                part, strips.example_inputs(kind, dtype, device), path, input_names=list(kind.inputs()),
                output_names=list(kind.outputs()), dynamic_axes=strips.dynamic_axes(kind), **kwargs,
            )
        own = {k: list(v.shape) for k, v in part.state_dict(keep_vars=True).items()}
        theirs = {k: list(loaded[stage.weights[k]].shape) for k in own}
        result[key + name] = export._weightless_graph(path, kind.inputs(), (own, theirs), quiet=True)
    del module, found
    log(f"Exported {len(names)} graphs for {'en' if encoding else 'de'}coding in strips.")
    return result


def _weight_bytes(vae, names, dtype) -> int:
    state = vae.first_stage_model.state_dict(keep_vars=True)
    return sum(state[n].numel() for n in names if n in state) * torch.tensor([], dtype=dtype).element_size()


def build(arch, profiles: list, work: str, optimization_level=None, progress=None, vae=None):
    """Builds the VAE engines these profiles' sizes lack, for the loaded VAE and its precision: the
    decoder's pre and post engines, the encoder's two with those that encode in strips, and the stripped
    decode's (strips.py), keeping what
    this GPU's and TensorRT version's engines for the architecture have that covers them, and deleting the
    rest. Returns (the Record or None, why it failed or None, how many kernels TensorRT skipped for want of
    VRAM). If only the encoder's engines or the stripped decode's fail, the record has the others (and says
    why in "encoder_error" or "strips_error")."""
    from . import export

    vae = vae if vae is not None else loaded_vae()
    reason = "no VAE is loaded" if vae is None else unsupported(vae)
    if reason:
        return None, reason, 0
    skipped = 0
    try:
        dtype = engine_dtype(vae)
        cfg, vram, device = config(vae), _vram(), common.mm().get_torch_device()
        asked = shapes_for(profiles, cfg, dtype, vram)
        strip_wanted = _strips_wanted(vae, profiles, cfg, dtype)
        enc_wanted = _encoder_wanted(vae, profiles, dtype, vram)
        everything = [PRE, POST] + (list(ENCODER) if enc_wanted else []) + list(strip_wanted)
        old = next(iter(_matching(arch.id, vae, dtype)), None)
        todo = _missing(old, asked, strip_wanted, enc_wanted) if old is not None else list(everything)
        if old is not None and old.meta.get("strips_error"):  # building anyway: try those again
            todo += [k for k in strip_wanted if k not in old.strips and k not in todo]
        if old is not None and old.meta.get("encoder_error") and enc_wanted and ENCODE_PRE not in todo:
            todo += list(ENCODER)
        kinds_todo = [k for k in todo if k not in (PRE, POST) + ENCODER]
        shapes = {PRE: asked[PRE], POST: asked[POST]}
        post_asked = asked[POST]
        if old is not None:
            if PRE not in todo:
                shapes[PRE] = old.meta["shapes"][PRE]
            if POST not in todo:
                shapes[POST], post_asked = old.meta["shapes"][POST], old.meta.get("asked", old.meta["shapes"])[POST]
        if POST in todo:
            wanted_h, wanted_w = max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8
            b, _, h, w = asked[POST]["x"][2]
            if (h, w) != (wanted_h, wanted_w):
                cause = _excess(cfg, 1, wanted_h, wanted_w, _budget(dtype, vram))[1]
                further = " and further in strips" if strip_wanted else "; larger decodes run on PyTorch"
                log(f"The VAE decoder's post engine goes up to {w * 8}x{h * 8} ({b} at a time), not the profiles' {wanted_w * 8}x{wanted_h * 8}: {_why(cause, dtype, vram)}. It decodes up to there{further}.")
        if ENCODE_PRE in todo:
            wanted_h, wanted_w = max(p.height[2] for p in profiles) // 8 * 8, max(p.width[2] for p in profiles) // 8 * 8
            b, _, h, w = enc_wanted["shapes"][ENCODE_PRE]["image"][2]
            if (h, w) != (wanted_h, wanted_w):
                cause = _encoder_excess(enc_wanted["config"], 1, wanted_h, wanted_w, _budget(dtype, vram))[1]
                log(f"The VAE encoder's engines go up to {w}x{h} ({b} at a time), not the profiles' {wanted_w}x{wanted_h}: {_why(cause, dtype, vram, 'encode')}. Larger encodes run on PyTorch.")
        if old is not None and sorted(todo) != sorted(everything):
            log(f"Keeping the VAE engines that cover these sizes; building {len(todo)} of {len(everything)}.")
        if progress:
            progress(0.0, "exporting the graphs")
        graphs = {}
        if PRE in todo or POST in todo:
            graphs.update(export_onnx(vae, work, dtype, device))
        if ENCODE_PRE in todo:
            graphs.update({part: names for part, names in export_encoder_onnx(vae, work, dtype, device).items() if part in ENCODER})
            if enc_wanted["strips"]:
                graphs.update(export_strips(vae, work, dtype, device, list(enc_wanted["strips"]), encoding=True))
        if kinds_todo:
            graphs.update(export_strips(vae, work, dtype, device, kinds_todo))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the trace's VRAM back to the driver, for the builder's process
        built, done, tries = {}, 0, 0
        total = len(todo) + (len(enc_wanted["strips"]) if ENCODE_PRE in todo else 0)  # engines to build

        def job(part, plan_shapes):
            onnx_name = f"strip-{part}" if part in kinds_todo else part if part.startswith(ENCODE_STRIP) else f"vae_{part}"
            return {
                "onnx": os.path.join(work, f"{onnx_name}.onnx"),
                "plan": os.path.join(work, f"{onnx_name}.plan"),
                "weights": sorted(graphs[part]),
                "profiles": [plan_shapes],
                "timing_cache": store.timing_cache_path(),
                "optimization_level": optimization_level,
            }

        def step(what):
            n = done
            return (lambda f, t: progress((n + f) / max(1, total), f"{what}: {t}")) if progress else None

        def build_kinds(kinds: dict, prefix: str, what: str) -> tuple:
            """Builds the engines of these kinds ({name: shapes}) for `what` in strips, their graphs and plans
            named `prefix` and the kind: ({name: its meta} of those built, why one failed or None, stopping
            there). Raises _Cancelled."""
            nonlocal done, total, skipped
            metas = {}
            for i, name in enumerate(kinds):
                if export.CANCEL.is_set():
                    raise _Cancelled()
                result = export._build_plan(job(prefix + name, kinds[name]), work, step(f"engine {i + 1} of {len(kinds)} for {what} in strips"))
                skipped += result.get("memory_warnings") or 0
                if not result["ok"]:
                    if export.CANCEL.is_set() or str(result.get("error")) == "cancelled":
                        raise _Cancelled()
                    total -= len(kinds) - i
                    log(f"Building the VAE's engines for {what} in strips failed ({result.get('error')}); larger ones run on PyTorch.")
                    return metas, str(result.get("error"))
                built[prefix + name] = result
                metas[name] = {"shapes": kinds[name], "weights": sorted(graphs[prefix + name]), "stripped": result["stripped"]}
                done += 1
            return metas, None

        parts = [p for p in (POST, PRE) if p in todo]  # the post engine first: TensorRT may find no kernels for it
        while parts:
            part = parts[0]
            result = export._build_plan(job(part, shapes[part]), work, step(f"{part} engine"))
            warned = result.get("memory_warnings") or 0
            skipped += warned
            if not result["ok"]:
                smaller = shapes_for(profiles, cfg, dtype, vram, RETRY_SHARE) if dtype is not None and vram else asked
                if part != POST or tries or not warned or smaller[POST] == shapes[POST]:
                    return None, str(result.get("error")), skipped
                tries, shapes[POST] = 1, smaller[POST]
                b, _, h, w = shapes[POST]["x"][2]
                log(f"TensorRT found no kernels with the VRAM to spare at the largest size; trying again up to {w * 8}x{h * 8} ({b} at a time).")
                continue
            built[part] = result
            parts.pop(0)
            done += 1
        # The encoder's: kept if they cover these sizes, else built, the pre one again smaller like the post
        enc_meta = old.encoder if old is not None and enc_wanted and ENCODE_PRE not in todo else None
        enc_error = None
        if ENCODE_PRE in todo:
            enc_shapes, tries = enc_wanted["shapes"], 0
            parts = list(ENCODER)
            while parts and not enc_error:
                if export.CANCEL.is_set():
                    return None, "cancelled", skipped
                part = parts[0]
                result = export._build_plan(job(part, enc_shapes[part]), work, step(f"{part} engine"))
                warned = result.get("memory_warnings") or 0
                skipped += warned
                if result["ok"]:
                    built[part] = result
                    parts.pop(0)
                    done += 1
                    continue
                smaller = encoder_shapes(profiles, enc_wanted["config"], dtype, vram, RETRY_SHARE) if dtype is not None and vram else enc_shapes
                if export.CANCEL.is_set() or str(result.get("error")) == "cancelled":
                    return None, "cancelled", skipped
                if part != ENCODE_PRE or tries or not warned or smaller == enc_shapes:
                    enc_error = str(result.get("error"))
                    break
                tries, enc_shapes = 1, smaller
                b, _, h, w = enc_shapes[ENCODE_PRE]["image"][2]
                log(f"TensorRT found no kernels with the VRAM to spare at the largest size; trying again up to {w}x{h} ({b} at a time).")
            if enc_error:
                done = len([p for p in todo if p in (PRE, POST) + ENCODER])
                total -= len(enc_wanted["strips"])
                log(f"Building the VAE encoder's engines failed ({enc_error}); it encodes on PyTorch.")
            else:
                enc_meta = {
                    "digest": enc_wanted["digest"],
                    "config": enc_wanted["config"],
                    "shapes": enc_shapes,
                    "asked": enc_wanted["shapes"],  # what they were built for: the shapes, unless TensorRT had to go smaller
                    "weights": {part: sorted(graphs[part]) for part in ENCODER},
                    "weight_bytes": _weight_bytes(vae, sorted(set(graphs[ENCODE_PRE]) | set(graphs[ENCODE_POST])), dtype),
                    "stripped": {part: built[part]["stripped"] for part in ENCODER},
                    "strips": {},
                }
                kinds = enc_wanted["strips"]  # the encode in strips' (a failure leaves encodes whole only)
                enc_meta["strips"], error = build_kinds(kinds, ENCODE_STRIP, "encoding")
                if error:
                    enc_meta.update(strips={}, strips_error=error, strips_asked=kinds)
        strip_meta = {k: v for k, v in old.strips.items() if k in strip_wanted and k not in todo} if old is not None else {}
        built_kinds, strips_error = build_kinds({name: strip_wanted[name] for name in kinds_todo}, "", "decoding")
        strip_meta.update(built_kinds)
        if strips_error:
            strip_meta = {}  # a stripped decode needs every kind
        identity = [digest(vae), dtype_name(dtype), scaled(dtype), shapes, sorted((k, v["shapes"]) for k, v in strip_meta.items())]
        if enc_meta is not None:
            identity.append([enc_meta["digest"], enc_meta["shapes"], sorted((k, v["shapes"]) for k, v in (enc_meta.get("strips") or {}).items())])
        key = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
        weights = {part: sorted(graphs[part]) if part in built else old.meta["weights"][part] for part in (PRE, POST)}
        meta = {
            "vae_format": FORMAT,
            "key": key,
            "arch_id": arch.id,
            "digest": digest(vae),
            "dtype": dtype_name(dtype),
            "scaled": SCALED if scaled(dtype) else False,
            "config": cfg,
            "shapes": shapes,
            "asked": {PRE: shapes[PRE], POST: post_asked},  # what they were built for: the shapes, unless TensorRT had to go smaller
            "weights": weights,
            "weight_bytes": _weight_bytes(vae, weights[PRE] + weights[POST], dtype),
            "stripped": {part: built[part]["stripped"] if part in built else old.meta["stripped"][part] for part in (PRE, POST)},
            "strips": strip_meta,
            "encoder": enc_meta,
            "trt": common.trt_version(),
            "gpu": common.gpu_tag(),
            "gpu_name": torch.cuda.get_device_name(),
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        if strips_error:
            meta["strips_error"], meta["strips_asked"] = strips_error, strip_wanted
        if enc_error:
            meta["encoder_error"], meta["encoder_asked"] = enc_error, {"digest": enc_wanted["digest"], "shapes": enc_wanted["shapes"]}
        os.makedirs(folder(arch.id), exist_ok=True)
        base = f"{_prefix(arch.id)}-{key}"
        new = Record(base, meta)
        for part in (PRE, POST) + (ENCODER if enc_meta is not None else ()):
            source = os.path.join(work, f"vae_{part}.plan") if part in built else old.plan(part)
            if source != new.plan(part):
                os.replace(source, new.plan(part))
        for name in strip_meta:
            source = os.path.join(work, f"strip-{name}.plan") if name in built else old.strip_plan(name)
            if source != new.strip_plan(name):
                os.replace(source, new.strip_plan(name))
        for name in (enc_meta or {}).get("strips") or {}:
            source = os.path.join(work, f"{ENCODE_STRIP}{name}.plan") if ENCODE_STRIP + name in built else old.encoder_strip_plan(name)
            if source != new.encoder_strip_plan(name):
                os.replace(source, new.encoder_strip_plan(name))
        for path in glob.glob(glob.escape(_prefix(arch.id)) + "-*.json"):  # what these replace
            if path[: -len(".json")] != base:
                _remove(path[: -len(".json")])
        for path in glob.glob(glob.escape(base) + ".strip-*.plan"):  # this record's own, from before, it no longer has
            if path[len(base) + len(".strip-"): -len(".plan")] not in strip_meta:
                os.remove(path)
        for path in glob.glob(glob.escape(base) + f".{ENCODE_STRIP}*.plan"):
            if path[len(base) + len(ENCODE_STRIP) + 1: -len(".plan")] not in ((enc_meta or {}).get("strips") or {}):
                os.remove(path)
        if enc_meta is None:
            for part in ENCODER:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(new.plan(part))
        with open(base + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(base + ".json.tmp", base + ".json")
        return new, None, skipped
    except _Cancelled:
        return None, "cancelled", skipped
    except Exception as e:
        traceback.print_exc()
        return None, str(e), skipped


class _Cancelled(Exception):
    """The build was cancelled."""


# --- Decoding


class TensorRTVAE:
    """What the model manager holds for the VAE's engines (its name shows in reForge's log)."""


class _Side:
    """The VAE's decoder's engines, or its encoder's: the two parts of the one that runs it whole (PRE and
    POST, or ENCODER's) and the engines of the stages of the one that runs it in strips. Each loads when it's
    first used, with exactly the room loading it (or one of its kind) took the last time made for it (its
    weights' size, the first time), and holds the weights of `key` (the VAE's decoder's or encoder's) scaled
    by 2^-exponent, refitted when it's used holding others."""

    def __init__(self, model: "VaeModel", encoding: bool):
        self.model, self.encoding = model, encoding
        self.strips_key = "encode-strips" if encoding else "strips"  # its stages' engines, in VaeModel.using
        self.key = None  # the key of the weights its engines are to hold (decode() or encode() sets it)
        self.engines = {}  # a part, or a Stage.identity -> its Engine
        self.names = {}  # likewise -> {the engine's weight: the VAE's}
        self.lasts = {}  # Stage.identity -> a context on its engine for the stage's last strips
        self.probes = {}  # kind name -> a context for activation sizes, on the engine of a stage of that kind
        self.applied = {}  # a part, or a Stage.identity -> (key, exponent) of the weights its engine holds
        self.sizes = {}  # likewise -> what its engine held once loaded
        self.peaks = {}  # a part, or a kind name -> the most loading its engine (one of its stages') took
        self.activations = {}  # (kind, rows, width) -> TensorRT's activation memory for a strip (its plan's own figure)
        self.loads = self.refits = 0  # since the last report(), and the seconds they took
        self.load_seconds = self.refit_seconds = 0.0

    @property
    def exponent(self) -> int:
        """What scaled engines scale this side by, as 2^-exponent (0: they aren't scaled)."""
        return (_exponents.get(self.key) or EXPONENT) if self.model.record.scaled else 0

    @property
    def scale(self) -> float:
        return 2.0 ** -self.exponent

    @property
    def kinds(self) -> dict:
        """{kind name: {"shapes", "weights", "stripped"}}: the engines of the one that runs in strips."""
        return self.model.record.encoder_strips if self.encoding else self.model.record.strips

    def scale_input(self) -> dict:
        """The engines' "scale" input, if they take one."""
        record = self.model.record
        if not record.scaled:
            return {}
        return {"scale": torch.full((1, 1, 1, 1), self.scale, dtype=record.dtype, device=self.model.load_device)}

    def engine(self, part: str):
        """The engine of a part of the one that runs whole, loaded, holding the current weights."""
        record = self.model.record
        listed = (record.encoder if self.encoding else record.meta)["weights"][part]
        return self._get(part, record.plan(part), None, listed, part)

    def stage(self, stage) -> tuple:
        """(The engine of a stage of the one that runs in strips, loaded, holding the current weights; a context
        on it for the stage's last strips, so neither changes shapes from one decode of a size to the next.)"""
        record, kind = self.model.record, stage.kind.name
        plan = record.encoder_strip_plan(kind) if self.encoding else record.strip_plan(kind)
        e = self._get(stage.identity, plan, stage.weights, stage.weights.values(), kind)
        if stage.identity not in self.lasts:
            self.lasts[stage.identity] = e.twin()
        return e, self.lasts[stage.identity]

    def probe(self, stage):
        """A context of the stage's kind, for TensorRT's activation sizes."""
        if stage.kind.name not in self.probes:
            self.probes[stage.kind.name] = self.stage(stage)[0].twin()
        return self.probes[stage.kind.name]

    def runner_for(self, stage):
        return strips.EngineRunner(lambda: self.stage(stage), lambda: self.probe(stage), self.kinds[stage.kind.name]["shapes"], stage.kind)

    def _get(self, which, plan: str, weights, listed, peak_key):
        """`which`'s engine (a part's, whose weights are named as the VAE's, or a stage's, `weights` naming
        them: {the engine's: the VAE's}): loaded from `plan` if it isn't, with room made for what loading one
        of `peak_key` (the part, or the stage's kind) took, or the `listed` weights' size the first time (a
        load that runs out anyway is tried once more with all the room there could be); refitted if it holds
        other weights than the current ones."""
        from .engine import Engine

        model = self.model
        e = self.engines.get(which)
        if e is not None and e.loaded:
            if self.applied.get(which) != (self.key, self.exponent):
                start = time.time()
                self._refit(which, e)
                self.refits, self.refit_seconds = self.refits + 1, self.refit_seconds + time.time() - start
            return e
        start = time.time()
        room = self.peaks.get(peak_key) or _weight_bytes(model.vae, listed, model.record.dtype)
        e = Engine(plan, model.arena, model.owner)
        for attempt in (0, 1):
            model.make_room(room)
            before, load = gpu.held(model.owner), {}
            try:
                with gpu.loading(model.owner) as load:
                    e.load()
                    state = model.vae.first_stage_model.state_dict(keep_vars=True)
                    own = weights if weights is not None else {n: n for n in e.weight_names if n in state}
                    self.names[which] = {n: own[n] for n in e.weight_names if n in own}
                    self._refit(which, e)
                break
            except Exception:
                e.unload()
                self.names.pop(which, None)
                self.applied.pop(which, None)
                if attempt or not load.get("failures"):
                    raise
                room = model.obtainable()
                log(f"Ran out of VRAM loading a VAE engine ({os.path.basename(plan)}); making all the room there could be and trying again.")
        self.peaks[peak_key] = max(self.peaks.get(peak_key, 0), load["peak"] - before)
        self.sizes[which] = gpu.held(model.owner) - before
        self.engines[which] = e
        self.loads, self.load_seconds = self.loads + 1, self.load_seconds + time.time() - start
        return e

    def _refit(self, which, e):
        get, names = self.model.weights(self.scale), self.names[which]
        torch.cuda.current_stream().synchronize()  # nothing in flight on the weights it holds now
        e.refit(list(names), lambda name, device: get(names[name], device))
        self.applied[which] = (self.key, self.exponent)

    def _kept(self, which, keep) -> bool:
        return which in keep if isinstance(which, str) else self.strips_key in keep

    def spare(self, keep) -> int:
        """What its engines that aren't in `keep` hold (VaeModel.using's keys)."""
        return sum(self.sizes.get(w, 0) for w in self.engines if not self._kept(w, keep))

    def drop(self, keep=()):
        """Unloads its engines that aren't in `keep` (VaeModel.using's keys), and the contexts on them."""
        if self.strips_key not in keep:
            for e in list(self.probes.values()) + list(self.lasts.values()):
                e.unload()  # the contexts on an engine before it
            self.probes, self.lasts = {}, {}
        for which in [w for w in self.engines if not self._kept(w, keep)]:
            self.engines.pop(which).unload()
            gpu.unloaded(self.model.owner, self.sizes.get(which, 0))
            for d in (self.names, self.applied, self.sizes):
                d.pop(which, None)

    def stage_count(self) -> tuple:
        """(How many of its stages' engines are loaded, what they hold.)"""
        stages = [w for w in self.engines if not isinstance(w, str)]
        return len(stages), sum(self.sizes.get(w, 0) for w in stages)

    def report(self) -> str:
        """For the log: the engines it loaded and refitted since the last report."""
        parts = []
        if self.loads:
            parts.append(f"loaded {self.loads} engine{'s' if self.loads > 1 else ''} in {self.load_seconds:.1f}s")
        if self.refits:
            parts.append(f"refitted {self.refits} for other weights in {self.refit_seconds:.1f}s")
        self.loads = self.refits = 0
        self.load_seconds = self.refit_seconds = 0.0
        return ", ".join(parts)


class VaeModel:
    """The VAE's engines as reForge's model manager sees a model: held as loaded while a decode or an encode
    runs on them (common.register), and evicted like any other, which unloads them all (each comes back with
    a small refit). Its size is what its engines hold in VRAM, as counted (gpu.py). They load as they're
    first used (_Side), and those the decode or encode under way doesn't run on (not in `using`) are the
    first to go when it needs room: their weights are copies of those of the ones it runs on."""

    parent = None

    def __init__(self, record: Record):
        from .engine import Arena

        mm = common.mm()
        self.record = record
        self.owner = f"VAE engines ({os.path.basename(record.base)})"  # whose TensorRT memory it is (gpu.py)
        self.model = TensorRTVAE()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.arena = Arena()  # every engine's activation memory: one runs at a time
        self.vae = None  # the VAE to take the weights from, while a decode or an encode runs
        self.active = False  # held as loaded by reForge's manager
        self.using = set()  # the engines the decode or encode under way runs on (_Side.drop's keys)
        self.decoder, self.encoder = _Side(self, False), _Side(self, True)
        self.needs = {}  # (encoding, images, height, width, ...) -> what need() or need_encode() found
        self.attention = {}  # (encoding, images, height, width) -> the attention's working memory, as it took

    @property
    def loaded(self) -> bool:
        return self.active

    def weights(self, scale: float):
        """(name, device) -> the VAE's weight as engines scaled by `scale` take it: the biases added to scaled
        tensors, scaled."""
        state = self.vae.first_stage_model.state_dict(keep_vars=True)

        def get(name, device):
            t = state[name].data.to(device)
            return t.float() * scale if scale != 1.0 and scaled_bias(name) else t

        return get

    def model_size(self):
        return gpu.held(self.owner)

    def loaded_size(self):
        return self.model_size() if self.active else 0

    def current_loaded_device(self):
        return self.load_device if self.active else self.offload_device

    def model_dtype(self):
        return self.record.dtype

    def model_patches_to(self, *args):
        pass

    def lowvram_patch_counter(self):
        return 0

    def is_clone(self, other):
        return other is self

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def partially_load(self, device, extra_memory=0, force_patch_weights=False):
        """Held as loaded: its engines load as they're first used, each with its room made (_Side)."""
        self.active = True
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all:
            self.decoder.drop()
            self.encoder.drop()
            self.arena.release()
            self.active = False
            gpu.forget(self.owner)
        return self.model

    def use(self, side: _Side, parts=(), strips: bool = False):
        """What's under way runs on these parts of `side`'s whole one, loaded now (each with its room), and
        with `strips`, on its stages' engines (loaded as they're planned): the room made for it is beside
        them."""
        self.using = set(parts) | ({side.strips_key} if strips else set())
        for part in parts:
            side.engine(part)

    def obtainable(self, engines: bool = True) -> int:
        """The most VRAM there could be beside the reserve for what's under way: what's free, what every other
        model holds (the UNet engines' too, unless not `engines`), and what this one's engines that it doesn't
        run on hold."""
        spare = self.decoder.spare(self.using) + self.encoder.spare(self.using)
        return common.obtainable(self.load_device, keep=[self], engines=engines) - common.reserve() + spare

    def make_room(self, nbytes: int) -> int:
        """Makes `nbytes` free beside the reserve: this model's engines that what's under way doesn't run on go
        first (their weights are copies, and they're the cheapest to bring back), then other models, by
        reForge's manager (the UNet engines last). Returns what's free then, beside the reserve."""
        if common.free(self.load_device) < nbytes:
            self.decoder.drop(self.using)
            self.encoder.drop(self.using)
        return common.make_room(nbytes, self.load_device, _own(self))

    def _attending(self, lat: int, work: int) -> int:
        """The most the mid block's attention holds at once (_attend), for queries, keys and values of `lat`
        bytes each and `work` bytes of its own working memory: them, its working memory and its output;
        beside scaled engines, them and their FP32 copies while those are made, then the copies, its working
        memory and its FP32 output, then that output and its cast back."""
        if not self.record.scaled:
            return 3 * lat + work + lat
        wide = lat * 4 // self.record.dtype.itemsize
        return max(3 * lat + 3 * wide, 3 * wide + work + wide, wide + lat)

    def need(self, n: int, h: int, w: int, out_gpu: bool = False, whole: bool = True) -> tuple:
        """(The most VRAM a decode of n latents of h x w on the pre and post engines holds beside what's
        loaded, their activation memory): TensorRT's activation memory for those shapes (one arena, the
        larger engine's), and beside it the tensors held at each step: the latents and what the pre engine
        makes of them; then the mid block's input and the attention's (_attending) with its working memory
        as it took at this size before; then that input, the attention's output, the image, and its FP32
        copy if the VAE outputs on the GPU. Not whole: the part of a decode in strips before them (the pre
        engine's activation memory, the first two steps)."""
        key = (False, n, h, w, out_gpu, whole)
        if key not in self.needs:
            cfg, size = self.record.meta["config"], self.record.dtype.itemsize
            stream = torch.cuda.current_stream().cuda_stream
            c, zc = cfg["mid_channels"], cfg["latent_channels"]
            act = self.decoder.engine(PRE).estimate_activation_bytes(0, {"latent": (n, zc, h, w)}, stream)
            lat, z = n * c * h * w * size, n * zc * h * w * (size + 4)  # the latents, and their FP32 form on the way
            steps = [z + 4 * lat, lat + self._attending(lat, self.attention.get((False, n, h, w), 0))]
            if whole:
                act = max(act, self.decoder.engine(POST).estimate_activation_bytes(0, {"x": (n, c, h, w), "attention": (n, c, h, w)}, stream))
                image = n * cfg["out_ch"] * 64 * h * w
                steps.append(2 * lat + image * size + (image * 4 if out_gpu else 0))
            self.needs[key] = (act + max(steps), act)
        return self.needs[key]

    def need_encode(self, n: int, h: int, w: int, pixels_gpu: bool = False) -> tuple:
        """(The most VRAM an encode of n images of h x w on the encoder's engines holds beside what's
        loaded, their activation memory), likewise: the images (and their FP32 form, if they come from the
        GPU) with what the encode-pre engine makes of them; then the attention's; then the mid block's input,
        the attention's output, the moments and their FP32 copy."""
        c, mid = self.record.encoder["config"]["mid_channels"], (h // 8, w // 8)
        key = (True, n, *mid, pixels_gpu, h, w)  # the attention's size first (see _attention)
        if key not in self.needs:
            cfg, size = self.record.encoder["config"], self.record.dtype.itemsize
            stream = torch.cuda.current_stream().cuda_stream
            act = max(
                self.encoder.engine(ENCODE_PRE).estimate_activation_bytes(0, {"image": (n, cfg["in_channels"], h, w)}, stream),
                self.encoder.engine(ENCODE_POST).estimate_activation_bytes(0, {"x": (n, c, *mid), "attention": (n, c, *mid)}, stream),
            )
            lat, image = n * c * mid[0] * mid[1] * size, n * cfg["in_channels"] * h * w
            moments = n * cfg["moments"] * mid[0] * mid[1]
            steps = (
                image * size + (image * 4 if pixels_gpu else 0) + 4 * lat,
                lat + self._attending(lat, self.attention.get((True, n, *mid), 0)),
                2 * lat + moments * (size + 4),
            )
            self.needs[key] = (act + max(steps), act)
        return self.needs[key]

    def need_after_strips(self, h: int, w: int) -> int:
        """What an encode in strips holds for an image once its strips are done, beyond their output (the mid
        block's input, of a latent of h x w): the attention's GroupNorm's output in FP32, with the queries,
        keys and values made of it (_encode_strips); then those, the attention's working memory (as it took at
        this size before) and its output, and that cast to the engines' precision; then that, the
        encode-post engine's activation memory, the moments and their FP32 copy."""
        cfg, size = self.record.encoder["config"], self.record.dtype.itemsize
        c, moments = cfg["mid_channels"], cfg["moments"] * h * w
        wide, lat = c * h * w * 4, c * h * w * size
        shapes = {"x": (1, c, h, w), "attention": (1, c, h, w)}
        act = self.encoder.engine(ENCODE_POST).estimate_activation_bytes(0, shapes, torch.cuda.current_stream().cuda_stream)
        work = self.attention.get((True, 1, h, w), 0)
        return max(4 * wide + work, 4 * wide + lat, act + lat + moments * (size + 4))

    def release(self):
        self.arena.release()


_models = {}  # Record.base -> VaeModel
_told = set()  # what was logged once already: records with another precision, architectures without engines
_exponents = {}  # the key of a VAE's decoder's or encoder's weights -> its exponent, where it outgrew EXPONENT (None: every one)


class _Overflow(Exception):
    """A scaled decode or encode produced NaN/Inf: its scale wasn't small enough for this VAE."""


_weights_keys = {}  # whether the encoder's -> (the VAE's model, the generation, the key of its decoder's or encoder's weights)


def _key(vae, encoder: bool = False) -> str:
    """A key for the decoder's weights (or the encoder's): their names, types, shapes and checksums. Once
    per generation: a VAE's weights can change in place. The checksums are computed where the weights are
    (VRAM, once reForge's own decode or encode has run), so keying them copies nothing to RAM."""
    from .weights import _Same, checksum

    fsm = vae.first_stage_model
    known = _weights_keys.get(encoder)
    if known is not None and known[0] == _Same(fsm) and known[1] == common.generation:
        return known[2]
    state = fsm.state_dict(keep_vars=True)
    h = hashlib.sha1()
    for name in sorted(n for n in state if (_encoder_relevant(n) if encoder else _relevant(n))):
        t = state[name].data
        h.update(f"{name} {t.dtype} {tuple(t.shape)} {checksum(t)};".encode())
    key = h.hexdigest()
    _weights_keys[encoder] = (_Same(fsm), common.generation, key)
    return key


def model_for(record: Record) -> VaeModel:
    model = _models.get(record.base)
    if model is None or model.record.meta != record.meta:
        if model is not None:
            _unload(model)
        model = _models[record.base] = VaeModel(record)
    return model


@common.inference_mode
def _unload(model: VaeModel):
    mm = common.mm()
    for i in reversed(range(len(mm.current_loaded_models))):
        if mm.current_loaded_models[i].model is model:
            mm.current_loaded_models.pop(i).model_unload()
    model.detach()


def release_all():
    for model in list(_models.values()):
        _unload(model)
    _models.clear()
    strips.release_pinned()  # the RAM that decodes in strips held their tensors in


def _fit(model: VaeModel, top: int, need_for) -> tuple:
    """(The most latents a decode on the pre and post engines takes at once, up to `top` (or images, an
    encode on the encoder's), 0 if not even one can; what that holds at its most; the engines' activation
    memory for it; the most room there could be.) `need_for(n)` gives what n hold, the engines being loaded
    (VaeModel.use). Room is made for exactly that (this model's engines that it doesn't run on first, then by
    reForge's manager, the UNet engines last). Several at a time only where that takes no UNet engine's
    room: one at a time is barely slower, and bringing an engine back costs seconds. The activation memory
    is allocated for it, once."""
    model.release()
    can = model.obtainable()
    beside = model.obtainable(engines=False)
    n = next((n for n in range(top, 0, -1) if need_for(n)[0] <= beside), 0) or (1 if need_for(1)[0] <= can else 0)
    if not n:
        return 0, need_for(1)[0], 0, can
    need, act = need_for(n)
    free = model.make_room(need)
    if free < need:
        log(f"Making room for {common.gib(need)} left {common.gib(free)} free: the models loaded hold more than reForge counts.")
        return 0, need, act, can
    model.arena.get(act)
    return n, need, act, can


PLANNINGS = 8  # plans made in a row at most while the stage engines they run load (_settled)


def _settled(planner, h: int, w: int, budget):
    """The planner's plan for latents of h x w in `budget()` bytes, once the stage engines it runs are loaded:
    planning loads engines of its own (for TensorRT's activation sizes), and a plan may run others, so it's
    planned again until one runs no engine that isn't loaded and none loaded while it was made. None if not
    even the smallest strips fit (or, which planning converges well before, PLANNINGS plans weren't enough)."""
    for _ in range(PLANNINGS):
        if not gpu.attached():  # TensorRT allocates from the driver: PyTorch's cache isn't free to it
            torch.cuda.empty_cache()
        room = budget()
        plan = planner.plan(h, w, room)
        if plan is None:
            return None
        missing = [r for r in plan.runners if r.engines is None]
        for r in missing:
            r.engines = r.load()
        if not missing and budget() >= room:
            return plan
    log(f"Planning strips didn't settle in {PLANNINGS} plans while their engines loaded.")
    return None


def _plan_strips(model: VaeModel, planner, h: int, w: int, beside: int):
    """The fastest plan in strips there can be room for, with exactly that room made: planned for the most
    there could be (VaeModel.obtainable), room made for what it holds and `beside` (what's held beside the
    strips that isn't allocated yet when they're planned), then planned for what's free then, which gives
    that plan unless reForge's manager couldn't make the room."""
    best = _settled(planner, h, w, lambda: model.obtainable() - beside)
    if best is None:
        return None
    model.make_room(best.vram + beside)
    return _settled(planner, h, w, lambda: common.free(model.load_device) - beside)


class _Peak:
    """The most PyTorch's allocator held during a block beyond what it held when the block began (with
    gpu.py's allocator, TensorRT's allocations are PyTorch's), through the resets nested blocks make."""

    carried = 0  # what an enclosing block saw before a nested one reset the allocator's peak

    def __init__(self):
        self.base = self.held = 0

    def __enter__(self):
        _Peak.carried = max(_Peak.carried, torch.cuda.max_memory_allocated())
        self.base = torch.cuda.memory_allocated()
        self.outer = _Peak.carried
        _Peak.carried = 0
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        self.held = max(_Peak.carried, torch.cuda.max_memory_allocated()) - self.base
        _Peak.carried = max(self.outer, _Peak.carried, torch.cuda.max_memory_allocated())
        return False


def _counted(held: int, need: int) -> str:
    """For the log: what a decode or an encode held at most, beside what was counted for it."""
    if held > need:
        return f"held {common.gib(held)}, {common.gib(held - need)} more than the {common.gib(need)} counted"
    return f"held {common.gib(held)} of the {common.gib(need)} counted"


def _did(side: _Side) -> str:
    """For the log: "; " and the engines the side loaded and refitted, if any."""
    said = side.report()
    return f"; {said}" if said else ""


def _own(model: VaeModel) -> list:
    mm = common.mm()
    return [lm for lm in mm.current_loaded_models if lm.model is model]


_copy_streams = {}


def _copy_stream(device):
    if device not in _copy_streams:
        _copy_streams[device] = torch.cuda.Stream(device=device)
    return _copy_streams[device]


def decode(vae, samples_in):
    """The decoded images as VAE.decode returns them, or None to decode on PyTorch."""
    from . import arch as arch_mod

    if unsupported(vae) or samples_in.ndim != 4:
        return None
    arch = arch_mod.detect()
    if arch is None:
        return None
    record = record_for(arch.id, vae)
    if record is None:
        if arch.id not in _told and store.engines(arch.id):  # the tab shows it as to build, too
            _told.add(arch.id)
            log("The VAE decoder has no engines for this architecture yet: build in the TensorRT tab to add them.")
        return None
    b, c, h, w = samples_in.shape
    found = None
    if record.strips:
        try:
            found = strips.stages(vae.first_stage_model, record.meta["config"]["padding"])
            if strips.supported(found) or not record.strips_cover(found, c, h, w):
                found = None
        except (ValueError, KeyError, AttributeError, StopIteration):
            found = None
    whole = record.covers(b, c, h, w)
    if not whole and found is None:
        return None
    wanted = engine_dtype(vae)
    if (record.dtype != wanted or record.scaled != scaled(wanted)) and record.base not in _told:
        _told.add(record.base)
        how = "" if record.scaled or not scaled(record.dtype) else ", where it can overflow"
        log(f"The VAE decoder engines were built in {label(record.meta['dtype'])}{how}; they'd now be built in {label(dtype_name(wanted))}. Build again to match.")
    model = model_for(record)
    side = model.decoder
    model.vae, side.key = vae, _key(vae)
    try:
        if record.scaled and _exponents.get(side.key, EXPONENT) is None:
            return None  # no scale kept this VAE's decoder finite: said when found out
        while True:
            try:
                with torch.inference_mode():
                    return _decode(model, vae, samples_in, whole, found)
            except _Overflow:
                exponent = side.exponent
                if exponent + EXPONENT_STEP > EXPONENT_MAX:
                    _exponents[side.key] = None
                    notify(f"The VAE decoder engines produced NaN/Inf values in FP16 even scaled by 2^-{exponent}; this VAE decodes on PyTorch")
                    return None
                _exponents[side.key] = exponent + EXPONENT_STEP
                log(f"This VAE's decoder outgrew FP16 scaled by 2^-{exponent}: decoding again scaled by 2^-{exponent + EXPONENT_STEP}, as it will from now on.")
    except torch.cuda.OutOfMemoryError as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        where = f"{os.path.basename(frame.filename)}:{frame.lineno}, {frame.name}"
        del e
        try:
            vram = f"; {common.gib(torch.cuda.mem_get_info()[0])} free, {common.gib(torch.cuda.memory_allocated())} of PyTorch's {common.gib(torch.cuda.memory_reserved())} in use"
        except Exception:
            vram = ""
        log(f"Ran out of VRAM decoding on the VAE engines ({where}{vram}); decoding on PyTorch.")
        return None
    finally:
        model.release()
        model.vae, model.using = None, set()  # don't keep this VAE alive
        _profiled("VAE decode")


def _profiled(where: str):
    from .engine import profile_report

    profile_report(where)


def _decode(model: VaeModel, vae, samples_in, whole: bool, found):
    """One try at a decode: on the pre and post engines if there can be room for them (VaeModel.need), else
    in strips (`found`, the stages) with exactly the room the fastest plan there can be room for takes, else
    None. Raises _Overflow if a scaled decode's values outgrew its scale."""
    b, _, h, w = samples_in.shape
    side = model.decoder
    how = f"{label(model.record.meta['dtype'])}" + (f", scaled by 2^-{side.exponent}" if model.record.scaled else "")
    common.register(model)
    out_gpu = torch.device(vae.output_device).type == "cuda"
    held_out = b * model.record.meta["config"]["out_ch"] * 64 * h * w * 4 if out_gpu else 0  # the images, all of them, as returned

    def need_for(n):
        need, act = model.need(n, h, w, out_gpu)
        return need + held_out, act

    n = need = act = can = 0
    if whole:
        model.use(side, (PRE, POST))
        n, need, act, can = _fit(model, min(b, model.record.max_batch), need_for)
        common.memory(
            f"VAE decode of {w * 8}x{h * 8}: {'whole, ' + str(n) + ' at a time' if n else 'not whole'}",
            f"whole holds {common.gib(need)} at most (TensorRT activations {common.gib(act)}); there could be room for {common.gib(can)}",
        )
    if n:
        start = time.time()
        with _Peak() as peak:
            pixels = _run(model, vae, samples_in, n)  # ends with a copy to the CPU, so the time is the decode's
        if pixels is not None:
            at_once = f", {n} at a time" if n < b else ""
            log(f"Decoded {b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8} on the VAE decoder engines in {time.time() - start:.2f}s ({how}{at_once}; {_counted(peak.held + act, need)}{_did(side)})")
        return pixels
    if found is None:
        log(f"The VAE decoder engines hold {common.gib(need)} at most for {w * 8}x{h * 8}, there could be room for {common.gib(can)}: decoding on PyTorch.")
        return None
    model.use(side, (PRE,), strips=True)
    model.release()
    common.memory(f"VAE decode of {w * 8}x{h * 8}: before planning strips")
    start = time.time()
    pixels, report = _run_strips(model, vae, samples_in, found, out_gpu)
    if pixels is not None:
        why = f"whole it holds {common.gib(need)}, there could be room for {common.gib(can)}; " if whole else ""
        log(f"Decoded {b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8} in strips on the VAE decoder engines in {time.time() - start:.2f}s ({how}; {why}{report}{_did(side)})")
    return pixels


def _attend(model: VaeModel, vae, mid: dict, encoding: bool = False):
    """The mid block's attention on what the pre engine made of its input (popped from `mid`), on PyTorch
    with reForge's own implementation, as PyTorch decodes or encodes. Beside scaled engines in FP32: its
    scores are products of what a GroupNorm made, which no scale keeps in FP16, and at the latent's size it
    costs little. The working memory it takes is kept, per size, for VaeModel.need and need_encode."""
    q, k, v = mid.pop("q"), mid.pop("k"), mid.pop("v")
    if model.record.scaled:
        q, k, v = q.float(), k.float(), v.float()
    return _attention(model, vae, q, k, v, encoding)


def _attention(model: VaeModel, vae, q, k, v, encoding: bool = False):
    """reForge's own attention of the mid block on these queries, keys and values, in the engines'
    precision; its working memory kept, per size (see _attend)."""
    fsm = vae.first_stage_model
    attention = (fsm.encoder if encoding else fsm.decoder).mid.attn_1.optimized_attention
    key = (encoding, q.shape[0], q.shape[2], q.shape[3])
    with _Peak() as peak:
        out = attention(q, k, v)
    work = max(0, peak.held - out.numel() * out.element_size())
    if work > model.attention.get(key, 0):  # what need() or need_encode() counted at this size is short
        model.attention[key] = work
        model.needs = {kept: need for kept, need in model.needs.items() if kept[:4] != key}
    return out.to(model.record.dtype)


def _check(model: VaeModel, image, first: bool) -> bool:
    """Whether a decoded image is usable. A scaled decode that isn't finite outgrew its scale (_Overflow);
    others are checked once per decode, as PyTorch's would be as wrong."""
    if model.record.scaled:
        if not torch.isfinite(image).all():
            raise _Overflow()
        return True
    if first and not torch.isfinite(image).all():
        notify(f"The VAE decoder engines produced NaN/Inf values in {label(model.record.meta['dtype'])}; decoding on PyTorch")
        return False
    return True


def _run(model: VaeModel, vae, samples_in, n: int):
    side = model.decoder
    dtype, device = model.record.dtype, model.load_device
    stream = torch.cuda.current_stream().cuda_stream
    to_latent = getattr(vae, "_to_vae_latent", lambda t: t)  # SD and SDXL VAEs: as is
    pre, post, scale = side.engine(PRE), side.engine(POST), side.scale_input()
    pixels = None
    for start in range(0, samples_in.shape[0], n):
        z = to_latent(samples_in[start:start + n]).to(device=device, dtype=dtype)
        mid = pre.infer({"latent": z, **scale}, 0, stream)
        del z
        h = _attend(model, vae, mid)
        image = post.infer({"x": mid.pop("x"), "attention": h, **scale}, 0, stream)["image"]
        del h
        if not _check(model, image, pixels is None):
            return None
        if pixels is None:
            pixels = torch.empty((samples_in.shape[0],) + tuple(image.shape[1:]), device=vae.output_device)
        pixels[start:start + n] = vae.process_output(image.to(vae.output_device).float())
        del image
    return pixels.movedim(1, -1)


def _run_strips(model: VaeModel, vae, samples_in, found: list, out_gpu: bool):
    """(The images, decoded one at a time with the post part in strips, or None; how it went for the first,
    for the log.) The part before the strips gets exactly its room (VaeModel.need, not whole), the strips
    theirs (_plan_strips). A plan that runs out of VRAM anyway raises (decode() says so): it was counted
    wrong."""
    side, record = model.decoder, model.record
    dtype, device = record.dtype, model.load_device
    stream = torch.cuda.current_stream().cuda_stream
    to_latent = getattr(vae, "_to_vae_latent", lambda t: t)
    fsm, padding = vae.first_stage_model, record.meta["config"]["padding"]
    decoder = strips.Decode(fsm, padding, side.runner_for, dtype, device, _copy_stream(device), model.arena, side.activations, set(side.kinds), side.scale)
    scale = side.scale_input()
    b, _, h, w = samples_in.shape
    image = record.meta["config"]["out_ch"] * 64 * h * w  # an image's values
    # Held beside the strips but not yet when they're planned: the image they make, copied out of them; on
    # the GPU, also its FP32 copy and the images decoded so far
    beside = image * dtype.itemsize + (image * 4 * (b + 1) if out_gpu else 0)
    pixels = plan = None
    report = ""
    for i in range(b):
        if i == 0:
            model.make_room(model.need(1, h, w, whole=False)[0])
        z = to_latent(samples_in[i:i + 1]).to(device=device, dtype=dtype)
        mid = side.engine(PRE).infer({"latent": z, **scale}, 0, stream)
        del z
        x = mid.pop("x")
        attended = _attend(model, vae, mid)
        del mid
        model.release()  # the pre engine's activations: the strips size their own
        if plan is None:
            start = time.time()
            plan = _plan_strips(model, decoder, h, w, beside)
            if plan is None:
                log(f"Not even a strip of the VAE decoder fits in the {common.gib(model.obtainable() - beside)} of VRAM there could be for {w * 8}x{h * 8}: decoding on PyTorch.")
                return None, ""
            planned = time.time() - start
        with _Peak() as peak:
            image_out = decoder.run(x, attended, plan, timed=i == 0)
        del x, attended
        if not _check(model, image_out, pixels is None):
            return None, ""
        if pixels is None:
            report = f"{_strips_report(plan, planned, side)}; {_counted(peak.held, plan.vram + beside)}"
            common.memory(f"VAE decode of {w * 8}x{h * 8}: strips done", decoder.memory_report(plan))
            pixels = torch.empty((b,) + tuple(image_out.shape[1:]), device=vae.output_device)
        pixels[i:i + 1] = vae.process_output(image_out.to(vae.output_device).float())
        del image_out
    return pixels.movedim(1, -1), report


def encode(vae, pixel_samples):
    """The latents as VAE.encode returns them, or None to encode on PyTorch."""
    from . import arch as arch_mod

    if unsupported(vae) or pixel_samples.ndim != 4 or encoder_unsupported(vae):
        return None
    arch = arch_mod.detect()
    record = record_for(arch.id, vae) if arch is not None else None
    if record is None or record.encoder is None:
        return None  # (vae.decode says when an architecture has no engines)
    pixels = vae.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
    b, c, h, w = pixels.shape
    if not record.has_encoder(encoder_digest(vae), c):
        return None
    found = None
    if record.encoder_strips:
        try:
            found = strips.encoder_stages(vae.first_stage_model, record.encoder["config"]["padding"])
            if strips.supported(found) or not record.encoder_strips_cover(found, h, w):
                found = None
        except (ValueError, KeyError, AttributeError, StopIteration):
            found = None
    if not record.encodes(h, w) and found is None:
        return None
    model = model_for(record)
    side = model.encoder
    model.vae, side.key = vae, _key(vae, encoder=True)
    try:
        if record.scaled and _exponents.get(side.key, EXPONENT) is None:
            return None  # no scale kept this VAE's encoder finite: said when found out
        while True:
            try:
                with torch.inference_mode():
                    return _encode(model, vae, pixels, found)
            except _Overflow:
                exponent = side.exponent
                if exponent + EXPONENT_STEP > EXPONENT_MAX:
                    _exponents[side.key] = None
                    notify(f"The VAE encoder engines produced NaN/Inf values in FP16 even scaled by 2^-{exponent}; this VAE encodes on PyTorch")
                    return None
                _exponents[side.key] = exponent + EXPONENT_STEP
                log(f"This VAE's encoder outgrew FP16 scaled by 2^-{exponent}: encoding again scaled by 2^-{exponent + EXPONENT_STEP}, as it will from now on.")
    except torch.cuda.OutOfMemoryError as e:
        del e
        log("Ran out of VRAM encoding on the VAE engines; encoding on PyTorch.")
        return None
    finally:
        model.release()
        model.vae, model.using = None, set()  # don't keep this VAE alive
        _profiled("VAE encode")


def _encode(model: VaeModel, vae, pixels, found):
    """One try at an encode of images (b, channels, h, w) in 0..1 on the encoder's engines: whole if there
    can be room for it (VaeModel.need_encode), else in strips (`found`, the stages) with exactly the room the
    fastest plan there can be room for takes, else None. Raises _Overflow if a scaled encode's values
    outgrew its scale."""
    b, _, h, w = pixels.shape
    side, record = model.encoder, model.record
    how = f"{label(record.meta['dtype'])}" + (f", scaled by 2^-{side.exponent}" if record.scaled else "")
    images = f"{b} image{'' if b == 1 else 's'} of {w}x{h}"
    common.register(model)
    whole = record.encodes(h, w)
    pixels_gpu = pixels.is_cuda
    n = need = act = can = 0
    if whole:
        model.use(side, ENCODER)
        n, need, act, can = _fit(model, min(b, record.encode_batch), lambda n: model.need_encode(n, h, w, pixels_gpu))
        common.memory(
            f"VAE encode of {w}x{h}: {'whole, ' + str(n) + ' at a time' if n else 'not whole'}",
            f"whole holds {common.gib(need)} at most (TensorRT activations {common.gib(act)}); there could be room for {common.gib(can)}",
        )
    start = time.time()
    if n:
        samples = None
        with _Peak() as peak:
            for at in range(0, b, n):
                latent = _encode_images(model, vae, vae.process_input(pixels[at:at + n]))
                if latent is None:
                    return None
                out = latent.to(vae.output_device).float()
                out = vae._from_vae_latent(out) if hasattr(vae, "_from_vae_latent") else out
                if samples is None:
                    samples = torch.empty((b,) + tuple(out.shape[1:]), device=vae.output_device)
                samples[at:at + n] = out
                del latent, out
        at_once = f", {n} at a time" if n < b else ""
        log(f"Encoded {images} on the VAE encoder engines in {time.time() - start:.2f}s ({how}{at_once}; {_counted(peak.held + act, need)}{_did(side)})")
        return samples
    why = f"whole it holds {common.gib(need)}, there could be room for {common.gib(can)}" if whole else "larger than they take whole"
    if found is None:
        log(f"The VAE encoder engines can't encode {w}x{h} ({why}): encoding on PyTorch.")
        return None
    model.use(side, (ENCODE_POST,), strips=True)
    model.release()
    common.memory(f"VAE encode of {w}x{h}: before planning strips")
    samples, report = _encode_strips(model, vae, pixels, found)
    if samples is not None:
        log(f"Encoded {images} in strips on the VAE encoder engines in {time.time() - start:.2f}s ({how}; {why}; {report}{_did(side)})")
    return samples


def _encode_strips(model: VaeModel, vae, pixels, found: list):
    """(The latents of images (b, channels, h, w) in 0..1, encoded one at a time in strips up to the mid
    block's attention, or None; how it went for the first, for the log.) Then, on PyTorch at the latent's
    size, the attention's GroupNorm (from the statistics the strips gave), its queries, keys and values (in
    FP32, as a GroupNorm's output is), and the attention; then the encode-post engine, as a whole encode.
    The image gets exactly its room, the strips theirs (_plan_strips), and the rest its own
    (VaeModel.need_after_strips). Raises _Overflow if a scaled encode's values outgrew its scale."""
    side, record, device = model.encoder, model.record, model.load_device
    dtype, stream = record.dtype, torch.cuda.current_stream().cuda_stream
    cfg = record.encoder["config"]
    fsm = vae.first_stage_model
    encoder = strips.Encode(fsm, cfg["padding"], side.runner_for, dtype, device, _copy_stream(device), model.arena, side.activations, set(side.kinds), side.scale)
    norm = strips.norms_of(fsm, [strips.ATTENTION_NORM], device)[strips.ATTENTION_NORM]
    a = fsm.encoder.mid.attn_1
    convs = [(c.weight.detach().to(device, torch.float32), c.bias.detach().to(device, torch.float32)) for c in (a.q, a.k, a.v)]
    scale = side.scale_input()
    b, c, h, w = pixels.shape
    mh, mw = h // 8, w // 8
    beside = cfg["mid_channels"] * mh * mw * dtype.itemsize  # the strips' output, copied out of them
    samples = plan = None
    report = ""
    for i in range(b):
        if i == 0:
            model.make_room(c * h * w * (4 + dtype.itemsize))  # the image, and its FP32 form on the way
        image = vae.process_input(pixels[i:i + 1]).to(device=device, dtype=dtype)
        if plan is None:
            start = time.time()
            plan = _plan_strips(model, encoder, mh, mw, beside)
            if plan is None:
                log(f"Not even a strip of the VAE encoder fits in the {common.gib(model.obtainable() - beside)} of VRAM there could be for {w}x{h}: encoding on PyTorch.")
                return None, ""
            planned = time.time() - start
        with _Peak() as peak:
            x, stats = encoder.run(image, plan, timed=i == 0)
        del image
        model.release()  # the strips' activations
        if i == 0:
            report = f"{_strips_report(plan, planned, side)}; {_counted(peak.held, plan.vram + beside)}"
            common.memory(f"VAE encode of {w}x{h}: strips done", encoder.memory_report(plan))
            model.make_room(model.need_after_strips(mh, mw))
        center, factor, shift = stats.affine(norm, side.scale)
        normed = (x.float() - center) * factor + shift
        q, k, v = (torch.nn.functional.conv2d(normed, weight, bias) for weight, bias in convs)
        del normed
        attended = _attention(model, vae, q, k, v, encoding=True)
        del q, k, v
        moments = side.engine(ENCODE_POST).infer({"x": x, "attention": attended, **scale}, 0, stream)["moments"]
        del x, attended
        if not torch.isfinite(moments).all():
            if record.scaled:
                raise _Overflow()
            notify(f"The VAE encoder engines produced NaN/Inf values in {label(record.meta['dtype'])}; encoding on PyTorch")
            return None, ""
        out = fsm.regularization(moments.float())[0].to(vae.output_device).float()
        out = vae._from_vae_latent(out) if hasattr(vae, "_from_vae_latent") else out
        if samples is None:
            samples = torch.empty((b,) + tuple(out.shape[1:]), device=vae.output_device)
        samples[i:i + 1] = out
        del moments, out
    return samples, report


def _encode_images(model: VaeModel, vae, image):
    """The latents (in VRAM, FP32) of images as the encoder takes them (-1..1), on the engines; None if
    they came out NaN/Inf (said). Raises _Overflow instead if the engines are scaled."""
    side, record = model.encoder, model.record
    stream = torch.cuda.current_stream().cuda_stream
    scale = side.scale_input()
    image = image.to(device=model.load_device, dtype=record.dtype)
    mid = side.engine(ENCODE_PRE).infer({"image": image, **scale}, 0, stream)
    del image
    attended = _attend(model, vae, mid, encoding=True)
    moments = side.engine(ENCODE_POST).infer({"x": mid.pop("x"), "attention": attended, **scale}, 0, stream)["moments"]
    del attended, mid
    if not torch.isfinite(moments).all():
        if record.scaled:
            raise _Overflow()
        notify(f"The VAE encoder engines produced NaN/Inf values in {label(record.meta['dtype'])}; encoding on PyTorch")
        return None
    return vae.first_stage_model.regularization(moments.float())[0]  # reForge's own: the distribution's mode


def _strips_report(plan, planned: float, side: _Side) -> str:
    """For the log: the VRAM, the blocks' ways, RAM, planning and the stages' time on the GPU, and the
    stage engines."""
    parts = [f"{common.gib(plan.vram)} of VRAM"]
    count, held = side.stage_count()
    if count:
        parts.append(f"{count} stage engines holding {common.gib(held)}")
    if plan.describe():
        parts.append(plan.describe())
    if plan.host:
        kind, nbytes, seconds = strips.host_memory()
        parts.append(f"{len(plan.host)} outputs in {kind} RAM ({common.gib(plan.moved)} copied; {common.gib(nbytes)} had in {seconds:.1f}s)")
    parts.append(f"planned in {planned * 1000:.0f} ms")
    if plan.times:
        try:
            times = {n: a.elapsed_time(b) / 1000 for n, (a, b) in plan.times.items()}
            slowest = max(times, key=times.get)
            parts.append(f"stages {sum(times.values()):.2f}s on the GPU, the slowest {plan.stages[slowest].kind.name} {times[slowest]:.2f}s")
        except Exception:
            pass
    return "; ".join(parts)
