"""Decoding latents on TensorRT: the VAE's decoder as weightless engines like the UNet's, refitted from
whichever VAE is loaded, in the precision reForge itself picks for that VAE when they're built (BF16
where the GPU has it, or what a --vae-in-* flag asks for), except where it picks FP32 only for want of
BF16: there the engines run in FP16, at half the VRAM and on the GPU's tensor cores.

VAEs overflow in FP16 (SDXL's makes values past its 65504), which is why reForge avoids it. The engines
don't, whatever the VAE, because of where the decoder's nonlinearities are: every one (SiLU, the
attention's softmax) comes right after a GroupNorm, and a GroupNorm's output doesn't depend on its
input's scale. So everything between two GroupNorms is linear, and can run multiplied by any constant
as long as what's added along the way (the convs' biases) is multiplied too. FP16 engines run it
multiplied by a power of two, 2^-EXPONENT, which costs floating point nothing: the latent and each
resnet block's conv inputs are multiplied by it (an input of the graphs, "scale"), the biases added to
scaled tensors are refitted multiplied by it, GroupNorms compute in FP32, and the image comes out as it
is, its last GroupNorm having taken the scale out. Should a decode still produce NaN/Inf, the exponent
goes up for that VAE and the decode runs again.

The decoder runs in three parts: a "pre" engine up to its mid-block attention's queries, keys and values;
that attention on PyTorch, with reForge's own implementation; and a "post" engine for the rest. The
attention has one head 512 channels wide, which TensorRT's fused attention doesn't cover, so in an engine
it would hold its whole score matrix (about 17 GB in FP32 at 2048x2048); reForge's memory-efficient
kernels hold a fraction of that.

Each decode takes the fastest way that fits in the VRAM free, decided before it starts: the engines, as
many images at a time as fit; else the post part in strips (strips.py) on engines of its own, holding in
RAM what doesn't fit in VRAM; else reForge's own decode. The engines never take a UNet engine's VRAM.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from . import common, store, strips
from .common import log, notify

FORMAT = 1
PRE, POST = "pre", "post"
HEADROOM = 256 * 2**20
MAX_BATCH = 16  # images per engine call, at most
# FP16 engines scale the decoder by 2^-EXPONENT. FP16 holds magnitudes up to 2^16 in steps no finer than
# 2^-24, so a tensor keeps FP16's precision scaled down until its typical value nears 2^-13 (measured: a
# decoder whose tensors are around 0.5 decodes as closely scaled by anything from 1 to 2^-12). At 6,
# tensors the VAE makes as large as 4 million fit, and ones as small as 0.01 keep their precision. A
# decode that overflows anyway raises it by EXPONENT_STEP for that VAE, up to EXPONENT_MAX; past that the
# VAE decodes on PyTorch.
EXPONENT, EXPONENT_STEP, EXPONENT_MAX = 6, 3, 12


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


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def engine_dtype(vae):
    """The precision the engines run a VAE's decoder in: reForge's for it, or FP16 (scaled, so it can't
    overflow) where reForge fell back to FP32 without being asked to and the GPU runs FP16 well."""
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
    """Whether engines in this precision run the decoder scaled: FP16's range is the one VAEs outgrow (BF16
    and FP32 share one 2^127 wide)."""
    return dtype == torch.float16


_SCALED_BIASES = ("post_quant_conv.bias", "decoder.conv_in.bias", "decoder.mid.attn_1.proj_out.bias")
_SCALED_ENDINGS = (".conv1.bias", ".conv2.bias", ".nin_shortcut.bias", ".conv_shortcut.bias", ".upsample.conv.bias")


def scaled_bias(name: str) -> bool:
    """Whether a decoder weight is a bias added to a scaled tensor (so it's refitted scaled): all the convs'
    but the last one's and the attention's queries', keys' and values', whose inputs a GroupNorm made."""
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


def _why(cause: str, dtype, vram) -> str:
    if cause == "tensor":
        return "TensorRT can't hold a larger decode's largest tensor (2^31 elements)"
    return f"a larger decode wouldn't fit in this GPU's {common.gib(vram)} in {label(dtype_name(dtype))}, even alone"


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
    over = _excess(cfg, 1, hi_h, hi_w, budget)[0]
    if over > 1:
        scale = math.sqrt(1 / over) if over < math.inf else 0
        hi_h, hi_w = max(1, int(hi_h * scale)), max(1, int(hi_w * scale))
        while _excess(cfg, 1, hi_h, hi_w, budget)[0] > 1 and max(hi_h, hi_w) > 1:
            hi_h, hi_w = (hi_h - 1, hi_w) if hi_h >= hi_w else (hi_h, hi_w - 1)
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
        return bool(self.meta.get("scaled"))

    @property
    def loadable(self) -> bool:
        try:
            return self.meta["gpu"] == common.gpu_tag() and common.trt_tag(self.meta["trt"]) == common.trt_tag()
        except Exception:
            return False

    def strip_plan(self, kind: str) -> str:
        return f"{self.base}.strip-{kind}.plan"

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
        kinds = self.strips
        if not kinds or not self._pre_covers(c, h, w):
            return False
        for s in found:
            entry = kinds.get(s.kind.name)
            if entry is None or not strips.engine_rows(entry["shapes"], s.kind, w * s.scale):
                return False
        return True

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


def records(arch_id: str) -> list:
    """This GPU's and TensorRT version's decoder engines for an architecture."""
    found = []
    for path in glob.glob(glob.escape(_prefix(arch_id)) + "-*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            r = Record(path[: -len(".json")], meta)
            if meta.get("vae_format") == FORMAT and r.loadable and all(os.path.isfile(r.plan(p)) for p in (PRE, POST)):
                if not all(os.path.isfile(r.strip_plan(k)) for k in r.strips):
                    meta["strips"] = {}  # decodes on the post engine only; the tab asks to build them again
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
    for path in [base + ".json", f"{base}.{PRE}.plan", f"{base}.{POST}.plan"] + glob.glob(glob.escape(base) + ".strip-*.plan"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def delete_for(arch_id: str, gpu: str, trt: str):
    """Deletes an architecture's decoder engines for one GPU and TensorRT version (with its UNet engines)."""
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
    """Why building these profiles builds the decoder engines too ("" if there's nothing to build): none
    for the loaded VAE's decoder and precision cover the profiles' sizes."""
    vae = vae if vae is not None else loaded_vae()
    if arch is None or not profiles or vae is None or unsupported(vae) or not available():
        return ""
    try:
        cfg, dtype = config(vae), engine_dtype(vae)
        shapes = shapes_for(profiles, cfg, dtype, _vram())
        strip_wanted = _strips_wanted(vae, profiles, cfg, dtype)
        if any(not _missing(r, shapes, strip_wanted) for r in _matching(arch.id, vae, dtype)):
            return ""
    except Exception as e:
        log(f"Could not check the VAE decoder engines: {e}")
        return ""
    return f"the VAE decoder, in {label(dtype_name(dtype))}"


def _matching(arch_id: str, vae, dtype) -> list:
    """The records for this VAE's decoder in this precision (and scaled, if that precision is)."""
    wanted, name = digest(vae), dtype_name(dtype)
    return [r for r in records(arch_id) if r.meta.get("digest") == wanted and r.meta.get("dtype") == name and r.scaled == scaled(dtype)]


def _strips_wanted(vae, profiles: list, cfg: dict, dtype) -> dict:
    """The stripped decode's kinds and their shapes for the profiles ({} if it can't run for this VAE)."""
    try:
        lists = strips.every_way(vae.first_stage_model, cfg["padding"])
    except (ValueError, KeyError, AttributeError, StopIteration):
        return {}
    return {} if any(strips.supported(found) for found in lists) else strip_shapes(profiles, lists, dtype)


def _missing(record: Record, shapes: dict, strip_wanted: dict) -> list:
    """The parts (PRE, POST, kind names) a record lacks for these shapes. A post engine that had to be built
    smaller than asked counts as built (its "asked"), and so do kinds whose build failed for these shapes."""
    meta = record.meta
    out = []
    if not _covers(meta["shapes"], {PRE: shapes[PRE]}):
        out.append(PRE)
    if not _covers(meta.get("asked", meta["shapes"]), {POST: shapes[POST]}):
        out.append(POST)
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
    """The scale a scaled decoder's blocks multiply by: the graph's input, while it's traced."""

    value = None


def scale_decoder(module) -> _Scale:
    """Makes a decoder (vae.fresh's, or a stand-in) compute scaled, for tracing: each resnet block's convs
    take their input times the scale (the returned holder's value), and every GroupNorm computes in FP32
    (its weights too) on its input divided by the scale. FP32 has the range for the values as the VAE
    makes them, their squares included, and so the GroupNorm's epsilon is added to the variance it was
    meant for (to a scaled one, it would be thousands of times too large). With _Pre and _Post multiplying
    what comes in, and the biases refitted scaled (scaled_bias), the decoder's tensors between GroupNorms
    are all scaled alike."""
    holder = _Scale()
    for m in module.modules():
        if isinstance(m, torch.nn.GroupNorm):
            m.float()
            m.forward = (lambda inner: lambda x: inner(x.float() / holder.value.float()).to(x.dtype))(m.forward)
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


@common.inference_mode
def export_onnx(vae, work: str, dtype, device) -> dict:
    """Writes work/vae_pre.onnx and work/vae_post.onnx (scaled, in FP16); returns {part: {weight name:
    shape}, "config"}."""
    from . import export

    cfg = config(vae)
    module = fresh(cfg, device, dtype)
    holder = scale_decoder(module) if scaled(dtype) else None
    for t in module.state_dict(keep_vars=True).values():  # random: tracing doesn't depend on the weights, and
        if t.is_floating_point():  # distinct values keep the exporter from merging identical ones (GroupNorm's)
            t.normal_(0, 0.02)
    parts = {PRE: _Pre(module, holder), POST: _Post(module, holder)}
    latent = torch.randn(1, cfg["latent_channels"], 32, 32, dtype=dtype, device=device)
    scale = (torch.full((1, 1, 1, 1), 2.0**-EXPONENT, dtype=dtype, device=device),) if holder is not None else ()
    more = ["scale"] if holder is not None else []
    x, _, _, _ = parts[PRE](latent, *scale)
    kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    lat_axes = {0: "batch", 2: "height", 3: "width"}
    specs = {
        PRE: ((latent, *scale), ["latent"] + more, ["x", "q", "k", "v"], {n: lat_axes for n in ("latent", "x", "q", "k", "v")}),
        POST: ((x, torch.randn_like(x), *scale), ["x", "attention"] + more, ["image"], {"x": lat_axes, "attention": lat_axes, "image": {0: "batch", 2: "image_height", 3: "image_width"}}),
    }
    exported = {k: list(v.shape) for k, v in module.state_dict(keep_vars=True).items()}
    loaded = {k: list(v.shape) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if _relevant(k)}
    result = {"config": cfg}
    for part, (args, inputs, outputs, axes) in specs.items():
        path = os.path.join(work, f"vae_{part}.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(parts[part], args, path, input_names=inputs, output_names=outputs, dynamic_axes=axes, **kwargs)
        result[part] = export._weightless_graph(path, inputs, (exported, loaded))
    del module, parts, latent, x
    return result


@common.inference_mode
def export_strips(vae, work: str, dtype, device, names) -> dict:
    """Writes work/strip-<kind>.onnx for the stripped decode's kinds named; returns {kind name: {weight name:
    shape}}. Each from a stage of that kind in a fresh decoder of the VAE's layout."""
    from . import export

    cfg = config(vae)
    module = fresh(cfg, device, dtype)
    for t in module.state_dict(keep_vars=True).values():  # random, as in export_onnx
        if t.is_floating_point():
            t.normal_(0, 0.02)
    found = [s for each in strips.every_way(module, cfg["padding"]) for s in each]
    loaded = vae.first_stage_model.state_dict(keep_vars=True)
    kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    result = {}
    for name in names:
        stage = next(s for s in found if s.kind.name == name)
        kind, part = stage.kind, strips.StageModule(stage, module)
        path = os.path.join(work, f"strip-{name}.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(
                part, strips.example_inputs(kind, dtype, device), path, input_names=list(kind.inputs()),
                output_names=list(kind.outputs()), dynamic_axes=strips.dynamic_axes(kind), **kwargs,
            )
        own = {k: list(v.shape) for k, v in part.state_dict(keep_vars=True).items()}
        theirs = {k: list(loaded[stage.weights[k]].shape) for k in own}
        result[name] = export._weightless_graph(path, kind.inputs(), (own, theirs), quiet=True)
    del module, found
    log(f"Exported {len(names)} graphs for decoding in strips.")
    return result


def _weight_bytes(vae, names, dtype) -> int:
    state = vae.first_stage_model.state_dict(keep_vars=True)
    return sum(state[n].numel() for n in names if n in state) * torch.tensor([], dtype=dtype).element_size()


def build(arch, profiles: list, work: str, optimization_level=None, progress=None, vae=None):
    """Builds the decoder engines these profiles' sizes lack, for the loaded VAE's decoder and precision:
    the pre and post engines and the stripped decode's (strips.py), keeping what this GPU's and TensorRT
    version's engines for the architecture have that covers them, and deleting the rest. Returns (the
    Record or None, why it failed or None, how many kernels TensorRT skipped for want of VRAM). If only the
    stripped decode's engines fail, the record has the others (and says why in "strips_error")."""
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
        old = next(iter(_matching(arch.id, vae, dtype)), None)
        todo = _missing(old, asked, strip_wanted) if old is not None else [PRE, POST] + list(strip_wanted)
        if old is not None and old.meta.get("strips_error"):  # building anyway: try those again
            todo += [k for k in strip_wanted if k not in old.strips and k not in todo]
        kinds_todo = [k for k in todo if k not in (PRE, POST)]
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
        if old is not None and todo != [PRE, POST] + list(strip_wanted):
            log(f"Keeping the VAE decoder engines' parts that cover these sizes; building {len(todo)} of them.")
        if progress:
            progress(0.0, "exporting the graphs")
        graphs = {}
        if PRE in todo or POST in todo:
            graphs.update(export_onnx(vae, work, dtype, device))
        if kinds_todo:
            graphs.update(export_strips(vae, work, dtype, device, kinds_todo))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the trace's VRAM back to the driver, for the builder's process
        built, done, tries = {}, 0, 0

        def job(part, plan_shapes):
            onnx_name = f"strip-{part}" if part in kinds_todo else f"vae_{part}"
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
            return (lambda f, t: progress((n + f) / max(1, len(todo)), f"{what}: {t}")) if progress else None

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
        strips_error = None
        strip_meta = {k: v for k, v in old.strips.items() if k in strip_wanted and k not in todo} if old is not None else {}
        for i, name in enumerate(kinds_todo):
            if export.CANCEL.is_set():
                return None, "cancelled", skipped
            result = export._build_plan(job(name, strip_wanted[name]), work, step(f"engine {i + 1} of {len(kinds_todo)} for decoding in strips"))
            skipped += result.get("memory_warnings") or 0
            if not result["ok"]:
                strips_error = str(result.get("error"))
                log(f"Building the VAE decoder's engines for decoding in strips failed ({strips_error}); larger decodes run on PyTorch.")
                break
            built[name] = result
            strip_meta[name] = {"shapes": strip_wanted[name], "weights": sorted(graphs[name]), "stripped": result["stripped"]}
            done += 1
        if strips_error:
            strip_meta = {}  # a stripped decode needs every kind
        key = hashlib.sha1(json.dumps([digest(vae), dtype_name(dtype), scaled(dtype), shapes, sorted((k, v["shapes"]) for k, v in strip_meta.items())], sort_keys=True).encode()).hexdigest()[:10]
        weights = {part: sorted(graphs[part]) if part in built else old.meta["weights"][part] for part in (PRE, POST)}
        meta = {
            "vae_format": FORMAT,
            "key": key,
            "arch_id": arch.id,
            "digest": digest(vae),
            "dtype": dtype_name(dtype),
            "scaled": scaled(dtype),
            "config": cfg,
            "shapes": shapes,
            "asked": {PRE: shapes[PRE], POST: post_asked},  # what they were built for: the shapes, unless TensorRT had to go smaller
            "weights": weights,
            "weight_bytes": _weight_bytes(vae, weights[PRE] + weights[POST], dtype),
            "stripped": {part: built[part]["stripped"] if part in built else old.meta["stripped"][part] for part in (PRE, POST)},
            "strips": strip_meta,
            "trt": common.trt_version(),
            "gpu": common.gpu_tag(),
            "gpu_name": torch.cuda.get_device_name(),
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        if strips_error:
            meta["strips_error"], meta["strips_asked"] = strips_error, strip_wanted
        os.makedirs(folder(arch.id), exist_ok=True)
        base = f"{_prefix(arch.id)}-{key}"
        new = Record(base, meta)
        for part in (PRE, POST):
            source = os.path.join(work, f"vae_{part}.plan") if part in built else old.plan(part)
            if source != new.plan(part):
                os.replace(source, new.plan(part))
        for name in strip_meta:
            source = os.path.join(work, f"strip-{name}.plan") if name in built else old.strip_plan(name)
            if source != new.strip_plan(name):
                os.replace(source, new.strip_plan(name))
        for path in glob.glob(glob.escape(_prefix(arch.id)) + "-*.json"):  # what these replace
            if path[: -len(".json")] != base:
                _remove(path[: -len(".json")])
        for path in glob.glob(glob.escape(base) + ".strip-*.plan"):  # this record's own, from before, it no longer has
            if path[len(base) + len(".strip-"): -len(".plan")] not in strip_meta:
                os.remove(path)
        with open(base + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(base + ".json.tmp", base + ".json")
        return new, None, skipped
    except Exception as e:
        traceback.print_exc()
        return None, str(e), skipped


# --- Decoding


class TensorRTVAE:
    """What the model manager holds for the decoder engines (its name shows in reForge's log)."""


class VaeModel:
    """The decoder engines as reForge's model manager sees a model. Cheap to bring back (a small refit), so
    evicted like any other; their activation memory is only held during a decode."""

    parent = None

    def __init__(self, record: Record):
        from .engine import Arena, Engine

        mm = common.mm()
        self.record = record
        self.model = TensorRTVAE()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.arena = Arena()
        self.engines = {part: Engine(record.plan(part), self.arena) for part in (PRE, POST)}
        self.vae = self.key = None  # the VAE to refit from, and its weights' key
        self.applied = None  # the key of the weights in the engines
        self.measured = 0
        self.stage_engines = {}  # Stage.identity -> (its engine, a context for last strips, its weights, their bytes)
        self.probes = {}  # kind name -> a context for activation sizes
        self.strips_applied = None  # the key of the weights in the stage engines
        self.activations = {}  # (kind, rows, width) -> the activation memory its engine needs
        self.stage_vram = 0  # what loading the stage engines took from the driver (their weights and the rest)

    @property
    def loaded(self) -> bool:
        return all(e.loaded for e in self.engines.values())

    @property
    def exponent(self) -> int:
        """What scaled engines scale this VAE's decoder by, as 2^-exponent (0: they aren't scaled)."""
        return (_exponents.get(self.key) or EXPONENT) if self.record.scaled else 0

    @property
    def scale(self) -> float:
        return 2.0 ** -self.exponent

    def scale_input(self) -> dict:
        """The engines' "scale" input, if they take one."""
        if not self.record.scaled:
            return {}
        return {"scale": torch.full((1, 1, 1, 1), self.scale, dtype=self.record.dtype, device=self.load_device)}

    def _weights(self):
        """(name, device) -> the VAE's weight as the engines take it: the biases added to scaled tensors,
        scaled."""
        state, scale = self.vae.first_stage_model.state_dict(keep_vars=True), self.scale

        def get(name, device):
            t = state[name].data.to(device)
            return t.float() * scale if scale != 1.0 and scaled_bias(name) else t

        return state, get

    def model_size(self):
        size = max(self.measured, int(self.record.meta["weight_bytes"]))  # what the driver showed can be less
        return size + sum(e[3] for e in self.stage_engines.values())

    def loaded_size(self):
        return self.model_size() if self.loaded else 0

    def current_loaded_device(self):
        return self.load_device if self.loaded else self.offload_device

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
        if self.loaded:
            return 0
        if self.vae is None:
            raise RuntimeError("no VAE to take the decoder engines' weights from")
        start, free, reserved = time.time(), torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
        try:
            for e in self.engines.values():
                e.load()
            self.applied = None
            self.sync(quiet=True)
        except Exception:
            self.detach()
            raise
        used = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
        self.measured = max(self.measured, used)
        log(f"Loaded the VAE decoder engines ({common.gib(self.model_size())}, {label(self.record.meta['dtype'])}{_beyond(used, int(self.record.meta['weight_bytes']))}) in {time.time() - start:.1f}s")
        return self.model_size()

    def detach(self, unpatch_all=True):
        if unpatch_all and self.loaded:
            for e in self.engines.values():
                e.unload()
            self.applied = None
        if unpatch_all:
            self._unload_strips()
        return self.model

    def _unload_strips(self):
        for e in list(self.probes.values()) + [e[1] for e in self.stage_engines.values()] + [e[0] for e in self.stage_engines.values()]:
            e.unload()  # the contexts on an engine before it
        self.stage_engines, self.probes, self.strips_applied, self.activations, self.stage_vram = {}, {}, None, {}, 0

    def stage_engine(self, stage) -> tuple:
        """(The engine for a stage of the stripped decode, loaded with its weights; a context for its last strips.)"""
        from .engine import Engine

        if stage.identity not in self.stage_engines:
            free, reserved = _driver_free()
            e = Engine(self.record.strip_plan(stage.kind.name), self.arena)
            e.load()
            try:
                self._refit_stage(e, stage.weights)
            except Exception:
                e.unload()
                raise
            nbytes = _weight_bytes(self.vae, stage.weights.values(), self.record.dtype)
            self.stage_engines[stage.identity] = (e, e.twin(), stage.weights, nbytes)
            if free is not None:
                now, now_reserved = _driver_free()
                self.stage_vram += max(0, free - now - (now_reserved - reserved))
        return self.stage_engines[stage.identity][:2]

    def probe(self, stage):
        """A context of the stage's kind for activation sizes (on its first stage's engine)."""
        if stage.kind.name not in self.probes:
            self.probes[stage.kind.name] = self.stage_engine(stage)[0].twin()
        return self.probes[stage.kind.name]

    def runner_for(self, stage):
        shapes = self.record.strips[stage.kind.name]["shapes"]
        return strips.EngineRunner(lambda: self.stage_engine(stage), lambda: self.probe(stage), shapes, stage.kind)

    def _refit_stage(self, engine, weights: dict):
        _, get = self._weights()
        engine.refit([n for n in engine.weight_names if n in weights], lambda name, device: get(weights[name], device))

    def sync_strips(self):
        """Refits the stage engines loaded with another VAE's weights, or scaled otherwise (those loaded
        later get what's current)."""
        wanted = (self.key, self.exponent)
        if self.strips_applied == wanted:
            return
        if self.stage_engines:
            start = time.time()
            torch.cuda.current_stream().synchronize()
            for e, _, weights, _ in self.stage_engines.values():
                self._refit_stage(e, weights)
            log(f"Refitted the VAE decoder's {len(self.stage_engines)} engines for decoding in strips in {time.time() - start:.1f}s")
        self.strips_applied = wanted

    def sync(self, quiet=False):
        """Refits both engines from self.vae unless they hold its weights already, scaled as now."""
        wanted = (self.key, self.exponent)
        if self.applied == wanted:
            return
        start = time.time()
        state, get = self._weights()
        torch.cuda.current_stream().synchronize()
        for e in self.engines.values():
            e.refit([n for n in e.weight_names if n in state], get)
        if not quiet:
            log(f"Refitted the VAE decoder engines in {time.time() - start:.1f}s")
        self.applied = wanted

    def release(self):
        self.arena.release()

    def need(self, n: int, h: int, w: int) -> tuple:
        """(VRAM a decode of n latents of h x w takes on the engines, their activation memory): their
        activations (one engine at a time), the tensors between them, the attention's own working memory,
        and the image."""
        cfg, size = self.record.meta["config"], self.record.dtype.itemsize
        stream = torch.cuda.current_stream().cuda_stream
        c, zc = cfg["mid_channels"], cfg["latent_channels"]
        act = max(
            self.engines[PRE].estimate_activation_bytes(0, {"latent": (n, zc, h, w)}, stream),
            self.engines[POST].estimate_activation_bytes(0, {"x": (n, c, h, w), "attention": (n, c, h, w)}, stream),
        )
        lat, image = n * c * h * w * size, n * cfg["out_ch"] * 64 * h * w * size
        attending = lat * (3 if self.record.scaled else 1)  # in FP32 beside scaled engines, and their copies
        return max(act + 5 * attending, 8 * attending, act + 2 * lat + image) + HEADROOM, act


_models = {}  # Record.base -> VaeModel
_told = set()  # what was logged once already: records with another precision, architectures without engines
_exponents = {}  # the key of a VAE's decoder's weights -> its exponent, where a decode outgrew EXPONENT (None: every one)


class _Overflow(Exception):
    """A scaled decode produced NaN/Inf: its scale wasn't small enough for this VAE."""
_weights_key = None  # (the VAE's model, the generation, the key of its decoder's weights)


def _key(vae) -> str:
    """A hash of the decoder's weights, once per generation: a VAE's weights can change in place."""
    global _weights_key
    from .weights import _bytes, _Same

    fsm = vae.first_stage_model
    if _weights_key is not None and _weights_key[0] == _Same(fsm) and _weights_key[1] == common.generation:
        return _weights_key[2]
    state = fsm.state_dict(keep_vars=True)
    names = sorted(n for n in state if _relevant(n))
    with ThreadPoolExecutor() as pool:
        hashes = list(pool.map(lambda n: hashlib.sha1(_bytes(state[n])).digest(), names))
    key = hashlib.sha1(b"".join(hashes)).hexdigest()
    _weights_key = (_Same(fsm), common.generation, key)
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


def _fit(model: VaeModel, b: int, h: int, w: int) -> tuple:
    """(The most latents a decode on the pre and post engines can take at once in the VRAM free, after
    other models make room (the caller keeps the UNet engines), or 0 if not even one fits; the VRAM one
    takes; what's free.) Sizes their activation memory for it, so it's allocated once."""
    mm = common.mm()
    model.release()
    top = min(b, model.record.max_batch)
    free = mm.get_free_memory(model.load_device)
    fits = next((n for n in range(top, 0, -1) if model.need(n, h, w)[0] <= free), 0)
    need = model.need(max(1, fits), h, w)[0]
    if not fits:
        mm.free_memory(need, model.load_device, keep_loaded=_own(model))
        free = mm.get_free_memory(model.load_device)
        fits = 1 if need <= free else 0
        if not fits:
            return 0, need, free
    model.arena.get(model.need(fits, h, w)[1])
    return fits, need, free


def _own(model: VaeModel) -> list:
    mm = common.mm()
    return [lm for lm in mm.current_loaded_models if lm.model is model]


_copy_streams = {}


def _copy_stream(device):
    if device not in _copy_streams:
        _copy_streams[device] = torch.cuda.Stream(device=device)
    return _copy_streams[device]


def decode(vae, samples_in):
    """The decoded images as VAE.decode returns them, or None to decode on PyTorch. The caller keeps the
    UNet engines loaded (runtime._keeping)."""
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
    model.vae, model.key = vae, _key(vae)
    try:
        if record.scaled and _exponents.get(model.key, EXPONENT) is None:
            return None  # no scale kept this VAE's decoder finite: said when found out
        while True:
            try:
                with torch.inference_mode():
                    return _decode(model, vae, samples_in, whole, found)
            except _Overflow:
                exponent = model.exponent
                if exponent + EXPONENT_STEP > EXPONENT_MAX:
                    _exponents[model.key] = None
                    notify(f"The VAE decoder engines produced NaN/Inf values in FP16 even scaled by 2^-{exponent}; this VAE decodes on PyTorch")
                    return None
                _exponents[model.key] = exponent + EXPONENT_STEP
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
        model.vae = None  # don't keep this VAE alive


def _decode(model: VaeModel, vae, samples_in, whole: bool, found):
    """One try at a decode: on the pre and post engines if they fit, else in strips (`found`, the stages),
    else None. Raises _Overflow if a scaled decode's values outgrew its scale."""
    mm = common.mm()
    b, _, h, w = samples_in.shape
    how = f"{label(model.record.meta['dtype'])}" + (f", scaled by 2^-{model.exponent}" if model.record.scaled else "")
    mm.load_models_gpu([model])
    model.sync()
    n, need, free = _fit(model, b, h, w) if whole else (0, 0, 0)
    if n:
        start = time.time()
        pixels = _run(model, vae, samples_in, n)  # ends with a copy to the CPU, so the time is the decode's
        if pixels is not None:
            at_once = f", {n} at a time" if n < b else ""
            log(f"Decoded {b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8} on the VAE decoder engines in {time.time() - start:.2f}s ({how}{at_once})")
        return pixels
    if found is None:
        log(f"The VAE decoder engines need {common.gib(need)} of VRAM for {w * 8}x{h * 8}, {common.gib(free)} is free: decoding on PyTorch.")
        return None
    if not whole:  # the post engine doesn't reach this size: what the strips can have
        mm.free_memory(_vram() or 2**62, model.load_device, keep_loaded=_own(model))
    model.release()
    model.sync_strips()
    start = time.time()
    pixels, report = _run_strips(model, vae, samples_in, found)
    if pixels is not None:
        why = f"whole it needs {common.gib(need)}, {common.gib(free)} was free; " if whole else ""
        log(f"Decoded {b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8} in strips on the VAE decoder engines in {time.time() - start:.2f}s ({how}; {why}{report})")
    return pixels


def _attend(model: VaeModel, vae, mid: dict):
    """The mid block's attention on what the pre engine made of its input (popped from `mid`), on PyTorch
    with reForge's own implementation, as PyTorch decodes. Beside scaled engines in FP32: its scores are
    products of what a GroupNorm made, which no scale keeps in FP16, and at the latent's size it costs
    little."""
    attention = vae.first_stage_model.decoder.mid.attn_1.optimized_attention
    q, k, v = mid.pop("q"), mid.pop("k"), mid.pop("v")
    if model.record.scaled:
        return attention(q.float(), k.float(), v.float()).to(model.record.dtype)
    return attention(q, k, v)


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
    dtype, device = model.record.dtype, model.load_device
    stream = torch.cuda.current_stream().cuda_stream
    to_latent = getattr(vae, "_to_vae_latent", lambda t: t)  # SD and SDXL VAEs: as is
    scale = model.scale_input()
    pixels = None
    for start in range(0, samples_in.shape[0], n):
        z = to_latent(samples_in[start:start + n]).to(device=device, dtype=dtype)
        mid = model.engines[PRE].infer({"latent": z, **scale}, 0, stream)
        del z
        h = _attend(model, vae, mid)
        image = model.engines[POST].infer({"x": mid.pop("x"), "attention": h, **scale}, 0, stream)["image"]
        del h
        if not _check(model, image, pixels is None):
            return None
        if pixels is None:
            pixels = torch.empty((samples_in.shape[0],) + tuple(image.shape[1:]), device=vae.output_device)
        pixels[start:start + n] = vae.process_output(image.to(vae.output_device).float())
        del image
    return pixels.to(vae.output_device).movedim(1, -1)


def _run_strips(model: VaeModel, vae, samples_in, found: list):
    """(The images, decoded one at a time with the post part in strips, or None; how it went for the first,
    for the log.)"""
    dtype, device = model.record.dtype, model.load_device
    mm = common.mm()
    stream = torch.cuda.current_stream().cuda_stream
    to_latent = getattr(vae, "_to_vae_latent", lambda t: t)
    fsm, padding = vae.first_stage_model, model.record.meta["config"]["padding"]
    decoder = strips.Decode(
        fsm, padding, model.runner_for, dtype, device, _copy_stream(device), model.arena, model.activations, set(model.record.strips), model.scale,
    )
    scale = model.scale_input()
    b, _, h, w = samples_in.shape
    pixels = plan = None
    report = ""
    for i in range(b):
        z = to_latent(samples_in[i:i + 1]).to(device=device, dtype=dtype)
        mid = model.engines[PRE].infer({"latent": z, **scale}, 0, stream)
        del z
        x = mid.pop("x")
        attended = _attend(model, vae, mid)
        del mid
        model.release()  # the pre engine's activations: the strips size their own
        if plan is None:
            start = time.time()
            plan = _plan_strips(decoder, mm, device, h, w)
            if plan is None:
                log(f"Not even a strip of the VAE decoder fits in the {common.gib(mm.get_free_memory(device))} of VRAM free for {w * 8}x{h * 8}: decoding on PyTorch.")
                return None, ""
            planned = time.time() - start
        try:
            image = decoder.run(x, attended, plan, timed=i == 0)
        except torch.cuda.OutOfMemoryError:  # not as planned: once more, planning for 3/4 of that
            torch.cuda.empty_cache()
            log(f"The VAE decoder's strips didn't fit in the {common.gib(plan.vram)} of VRAM planned; planning for less.")
            plan = decoder.plan(h, w, (mm.get_free_memory(device) * 3) // 4)
            if plan is None:
                return None, ""
            image = decoder.run(x, attended, plan)
        del x, attended
        if not _check(model, image, pixels is None):
            return None, ""
        if pixels is None:
            report = _strips_report(plan, planned, model)
            pixels = torch.empty((b,) + tuple(image.shape[1:]), device=vae.output_device)
        pixels[i:i + 1] = vae.process_output(image.to(vae.output_device).float())
        del image
    return pixels.to(vae.output_device).movedim(1, -1), report


def _plan_strips(decoder, mm, device, h: int, w: int):
    """The plan for what's free once the engines it runs are loaded (planning loads some, for their
    activation sizes): planned, the rest loaded, planned again, up to three times."""
    plan = None
    for _ in range(3):
        torch.cuda.empty_cache()  # PyTorch's cached blocks back to the driver: what's free is what's free
        plan = decoder.plan(h, w, mm.get_free_memory(device))
        if plan is None:
            return None
        missing = [r for r in plan.runners if r.engines is None]
        if not missing:
            return plan
        for r in missing:
            r.engines = r.load()
    return plan


def _driver_free() -> tuple:
    """(The driver's free VRAM, PyTorch's reserved), or (None, None)."""
    try:
        return torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
    except Exception:
        return None, None


def _beyond(used: int, weights: int) -> str:
    """For the log: what loading engines took from the driver beyond their weights (kernels and the rest),
    when the driver showed more."""
    return f"; {common.gib(used - weights)} beyond the weights" if used > weights + 2**20 else ""


def _strips_report(plan, planned: float, model=None) -> str:
    """For the log: the VRAM, the blocks' ways, RAM, planning and the stages' time on the GPU, and what the
    stage engines took (beyond their weights: kernels and the rest)."""
    parts = [f"{common.gib(plan.vram)} of VRAM"]
    if model is not None and model.stage_engines:
        weights = sum(e[3] for e in model.stage_engines.values())
        parts.append(f"{len(model.stage_engines)} stage engines in {common.gib(max(model.stage_vram, weights))}{_beyond(model.stage_vram, weights)}")
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
