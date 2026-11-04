"""Decoding latents on TensorRT: the VAE's decoder as weightless engines like the UNet's, refitted from
whichever VAE is loaded, in the precision reForge itself picks for that VAE when they're built: BF16 where
the GPU has it, FP32 below, or what a --vae-in-* flag asks for. So the engines overflow only where
PyTorch would, whatever the VAE.

The decoder runs in three parts: a "pre" engine up to its mid-block attention's queries, keys and values;
that attention on PyTorch, with reForge's own implementation; and a "post" engine for the rest. The
attention has one head 512 channels wide, which TensorRT's fused attention doesn't cover, so in an engine
it would hold its whole score matrix (about 17 GB in FP32 at 2048x2048); reForge's memory-efficient
kernels hold a fraction of that.

Each decode takes the fastest way that fits in the VRAM free, decided before it starts: the engines, as
many images at a time as fit, else reForge's own decode. The engines never take a UNet engine's VRAM.

Engines live in models/Unet-trt/<UNet architecture id>/vae/, built with that architecture's UNet engines
for the sizes its profiles cover."""

import contextlib
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

from . import common, store
from .common import log, notify

FORMAT = 1
PRE, POST = "pre", "post"
HEADROOM = 256 * 2**20
MAX_BATCH = 16  # images per engine call, at most


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


def label(name: str) -> str:
    """For people: "BF16", "FP32", ..."""
    return {"bfloat16": "BF16", "float32": "FP32", "float16": "FP16"}.get(name, name)


def shapes_for(profiles: list, cfg: dict) -> dict:
    """The engines' input shapes for the sizes a list of profiles covers: from the smallest to the largest
    of them, optimal at the first profile's optimal size, one image at a time up to the largest batch."""
    lo_h, lo_w = min(p.height[0] for p in profiles) // 8, min(p.width[0] for p in profiles) // 8
    hi_h, hi_w = max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8
    opt_h, opt_w = profiles[0].height[1] // 8, profiles[0].width[1] // 8
    batch = min(MAX_BATCH, max(p.batch[2] for p in profiles))
    zc, c = cfg["latent_channels"], cfg["mid_channels"]
    side = lambda ch: [[1, ch, lo_h, lo_w], [1, ch, opt_h, opt_w], [batch, ch, hi_h, hi_w]]  # noqa: E731
    return {PRE: {"latent": side(zc)}, POST: {"x": side(c), "attention": side(c)}}


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
    def loadable(self) -> bool:
        try:
            return self.meta["gpu"] == common.gpu_tag() and common.trt_tag(self.meta["trt"]) == common.trt_tag()
        except Exception:
            return False

    def covers(self, b: int, c: int, h: int, w: int) -> bool:
        lo, _, hi = self.meta["shapes"][PRE]["latent"]
        return c == lo[1] and lo[2] <= h <= hi[2] and lo[3] <= w <= hi[3]

    @property
    def max_batch(self) -> int:
        return int(self.meta["shapes"][PRE]["latent"][2][0])


def records(arch_id: str) -> list:
    """This GPU's and TensorRT version's decoder engines for an architecture."""
    found = []
    for path in glob.glob(glob.escape(_prefix(arch_id)) + "-*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            r = Record(path[: -len(".json")], meta)
            if meta.get("vae_format") == FORMAT and r.loadable and all(os.path.isfile(r.plan(p)) for p in (PRE, POST)):
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
    return next((r for r in matching if r.dtype == getattr(vae, "vae_dtype", None)), matching[0] if matching else None)


def _remove(base: str):
    for path in (base + ".json", f"{base}.{PRE}.plan", f"{base}.{POST}.plan"):
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
        cfg, wanted, dtype = config(vae), digest(vae), dtype_name(vae.vae_dtype)
        shapes = shapes_for(profiles, cfg)
        if any(r.meta.get("digest") == wanted and r.meta.get("dtype") == dtype and _covers(r.meta["shapes"], shapes) for r in records(arch.id)):
            return ""
    except Exception as e:
        log(f"Could not check the VAE decoder engines: {e}")
        return ""
    return f"the VAE decoder, in {label(dtype)}"


# --- Export and build


class _Pre(torch.nn.Module):
    """The decoder up to its mid-block attention: (that attention's input, its queries, keys, values)."""

    def __init__(self, fsm):
        super().__init__()
        self.post_quant_conv = getattr(fsm, "post_quant_conv", None)
        self.decoder = fsm.decoder

    def forward(self, latent):
        z = self.post_quant_conv(latent) if self.post_quant_conv is not None else latent
        d = self.decoder
        x = d.mid.block_1(d.conv_in(z), None)
        attn = d.mid.attn_1
        h = attn.norm(x)
        return x, attn.q(h), attn.k(h), attn.v(h)


class _Post(torch.nn.Module):
    """The rest of the decoder, from the mid-block attention's input and output: the image."""

    def __init__(self, fsm):
        super().__init__()
        self.decoder = fsm.decoder

    def forward(self, x, attention):
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
    """Writes work/vae_pre.onnx and work/vae_post.onnx; returns {part: {weight name: shape}, "config"}."""
    from . import export

    cfg = config(vae)
    module = fresh(cfg, device, dtype)
    for t in module.state_dict(keep_vars=True).values():  # random: tracing doesn't depend on the weights, and
        if t.is_floating_point():  # distinct values keep the exporter from merging identical ones (GroupNorm's)
            t.normal_(0, 0.02)
    parts = {PRE: _Pre(module), POST: _Post(module)}
    latent = torch.randn(1, cfg["latent_channels"], 32, 32, dtype=dtype, device=device)
    x, _, _, _ = parts[PRE](latent)
    kwargs = dict(export_params=False, opset_version=17, do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    lat_axes = {0: "batch", 2: "height", 3: "width"}
    specs = {
        PRE: ((latent,), ["latent"], ["x", "q", "k", "v"], {n: lat_axes for n in ("latent", "x", "q", "k", "v")}),
        POST: ((x, torch.randn_like(x)), ["x", "attention"], ["image"], {"x": lat_axes, "attention": lat_axes, "image": {0: "batch", 2: "image_height", 3: "image_width"}}),
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


def build(arch, profiles: list, work: str, optimization_level=None, progress=None, vae=None):
    """Builds the decoder engines for these profiles' sizes, for the loaded VAE's decoder and precision, and
    deletes this GPU's and TensorRT version's other decoder engines for the architecture. Returns (the
    Record, None) or (None, why it failed)."""
    from . import export

    vae = vae if vae is not None else loaded_vae()
    reason = "no VAE is loaded" if vae is None else unsupported(vae)
    if reason:
        return None, reason
    try:
        dtype = vae.vae_dtype
        cfg = config(vae)
        shapes = shapes_for(profiles, cfg)
        if progress:
            progress(0.0, "exporting the graphs")
        graphs = export_onnx(vae, work, dtype, common.mm().get_torch_device())
        stripped = {}
        for n, part in enumerate((PRE, POST)):
            step = (lambda f, t, n=n, part=part: progress((n + f) / 2, f"{part} engine: {t}")) if progress else None
            job = {
                "onnx": os.path.join(work, f"vae_{part}.onnx"),
                "plan": os.path.join(work, f"vae_{part}.plan"),
                "weights": sorted(graphs[part]),
                "profiles": [shapes[part]],
                "timing_cache": store.timing_cache_path(),
                "optimization_level": optimization_level,
            }
            result = export._build_plan(job, work, step)
            if not result["ok"]:
                return None, str(result.get("error"))
            stripped[part] = result["stripped"]
        key = hashlib.sha1(json.dumps([digest(vae), dtype_name(dtype), shapes], sort_keys=True).encode()).hexdigest()[:10]
        size = torch.tensor([], dtype=dtype).element_size()
        meta = {
            "vae_format": FORMAT,
            "key": key,
            "arch_id": arch.id,
            "digest": digest(vae),
            "dtype": dtype_name(dtype),
            "config": cfg,
            "shapes": shapes,
            "weights": {part: sorted(graphs[part]) for part in (PRE, POST)},
            "weight_bytes": sum(math.prod(s) * size for part in (PRE, POST) for s in graphs[part].values()),
            "stripped": stripped,
            "trt": common.trt_version(),
            "gpu": common.gpu_tag(),
            "gpu_name": torch.cuda.get_device_name(),
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        os.makedirs(folder(arch.id), exist_ok=True)
        base = f"{_prefix(arch.id)}-{key}"
        for path in glob.glob(glob.escape(_prefix(arch.id)) + "-*.json"):  # what these replace
            _remove(path[: -len(".json")])
        for part in (PRE, POST):
            os.replace(os.path.join(work, f"vae_{part}.plan"), f"{base}.{part}.plan")
        with open(base + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(base + ".json.tmp", base + ".json")
        return Record(base, meta), None
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


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
        arena = Arena()
        self.engines = {part: Engine(record.plan(part), arena) for part in (PRE, POST)}
        self.vae = self.key = None  # the VAE to refit from, and its weights' key
        self.applied = None  # the key of the weights in the engines
        self.measured = 0

    @property
    def loaded(self) -> bool:
        return all(e.loaded for e in self.engines.values())

    def model_size(self):
        return self.measured or int(self.record.meta["weight_bytes"])

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
        log(f"Loaded the VAE decoder engines ({common.gib(self.model_size())}, {label(self.record.meta['dtype'])}) in {time.time() - start:.1f}s")
        return self.model_size()

    def detach(self, unpatch_all=True):
        if unpatch_all and self.loaded:
            for e in self.engines.values():
                e.unload()
            self.applied = None
        return self.model

    def sync(self, quiet=False):
        """Refits both engines from self.vae unless they hold its weights already."""
        if self.applied == self.key:
            return
        start, state = time.time(), self.vae.first_stage_model.state_dict(keep_vars=True)
        torch.cuda.current_stream().synchronize()
        for e in self.engines.values():
            e.refit([n for n in e.weight_names if n in state], lambda name, device: state[name].data.to(device))
        if not quiet:
            log(f"Refitted the VAE decoder engines in {time.time() - start:.1f}s")
        self.applied = self.key

    def release(self):
        self.engines[PRE].release_activations()

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
        return max(act + 5 * lat, 8 * lat, act + 2 * lat + image) + HEADROOM, act


_models = {}  # Record.base -> VaeModel
_told = set()  # what was logged once already: records with another precision, architectures without engines
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


def _fit(model: VaeModel, b: int, h: int, w: int) -> int:
    """The most latents a decode on the engines can take at once in the VRAM free, after other models make
    room (the caller keeps the UNet engines), or 0 if not even one fits. Sizes their activation memory for
    it, so it's allocated once."""
    mm = common.mm()
    model.release()
    top = min(b, model.record.max_batch)
    free = mm.get_free_memory(model.load_device)
    fits = next((n for n in range(top, 0, -1) if model.need(n, h, w)[0] <= free), 0)
    if not fits:
        need = model.need(1, h, w)[0]
        mm.free_memory(need, model.load_device, keep_loaded=[lm for lm in mm.current_loaded_models if lm.model is model])
        free = mm.get_free_memory(model.load_device)
        fits = 1 if need <= free else 0
        if not fits:
            log(f"The VAE decoder engines need {common.gib(need)} of VRAM for {w * 8}x{h * 8}, {common.gib(free)} is free: decoding on PyTorch.")
            return 0
    model.engines[PRE].arena.get(model.need(fits, h, w)[1])
    return fits


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
    if not record.covers(b, c, h, w):
        return None
    if dtype_name(vae.vae_dtype) != record.meta["dtype"] and record.base not in _told:
        _told.add(record.base)
        log(f"The VAE decoder engines were built in {label(record.meta['dtype'])}; the VAE now runs in {label(dtype_name(vae.vae_dtype))}. Build again to match.")
    model = model_for(record)
    model.vae, model.key = vae, _key(vae)
    mm = common.mm()
    try:
        with torch.inference_mode():
            mm.load_models_gpu([model])
            model.sync()
            n = _fit(model, b, h, w)
            if not n:
                return None
            start = time.time()
            pixels = _run(model, vae, samples_in, n)  # ends with a copy to the CPU, so the time is the decode's
            if pixels is not None:
                at_once = f", {n} at a time" if n < b else ""
                log(f"Decoded {b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8} on the VAE decoder engines in {time.time() - start:.2f}s{at_once}")
            return pixels
    except torch.cuda.OutOfMemoryError:
        log("Ran out of VRAM decoding on the VAE engines; decoding on PyTorch.")
        return None
    finally:
        model.release()
        model.vae = None  # don't keep this VAE alive


def _run(model: VaeModel, vae, samples_in, n: int):
    dtype, device = model.record.dtype, model.load_device
    attention = vae.first_stage_model.decoder.mid.attn_1.optimized_attention  # reForge's own, as PyTorch decodes
    stream = torch.cuda.current_stream().cuda_stream
    to_latent = getattr(vae, "_to_vae_latent", lambda t: t)  # SD and SDXL VAEs: as is
    pixels = None
    for start in range(0, samples_in.shape[0], n):
        z = to_latent(samples_in[start:start + n]).to(device=device, dtype=dtype)
        mid = model.engines[PRE].infer({"latent": z}, 0, stream)
        del z
        h = attention(mid.pop("q"), mid.pop("k"), mid.pop("v"))
        image = model.engines[POST].infer({"x": mid.pop("x"), "attention": h}, 0, stream)["image"]
        del h
        if pixels is None:
            if not torch.isfinite(image).all():
                notify(f"The VAE decoder engines produced NaN/Inf values in {label(model.record.meta['dtype'])}; decoding on PyTorch")
                return None
            pixels = torch.empty((samples_in.shape[0],) + tuple(image.shape[1:]), device=vae.output_device)
        pixels[start:start + n] = vae.process_output(image.to(vae.output_device).float())
        del image
    return pixels.to(vae.output_device).movedim(1, -1)
