import hashlib
import json
import os
import weakref
from dataclasses import asdict, dataclass

from . import compat

_IGNORED_CONFIG = {"dtype", "device", "operations", "disable_unet_model_creation"}
# Left out of the architecture hash: scaled-FP8 / MixedPrecisionOps checkpoints carry these per-layer quantization
# metadata keys, which weights.py dequantizes/folds before refitting, so quantized checkpoints share engines with FP16.
_IGNORED_WEIGHTS = (
    ".scale_weight", ".scale_input", ".weight_scale", ".weight_scale_2",
    ".comfy_quant", ".input_scale", ".pre_quant_scale", ".weight_s_rel", ".weight_s_channel", ".weight_codebook",
    ".weight_correction",
)
NAMES = {
    "SD15": "SD 1.5",
    "SD20": "SD 2.x",
    "SDXL": "SDXL",
    "SDXLRefiner": "SDXL Refiner",
    "SSD1B": "SSD-1B",
    "Anima": "Anima",
    "MOSSImage": "MOSS-Image",
}
KINDS = ("unet", "anima", "moss_image")
_cache = {}


@dataclass(frozen=True)
class Arch:
    id: str
    family: str
    in_channels: int
    context_dim: int
    adm_channels: int
    multiple: int
    kind: str = "unet"
    downscale: int = 8
    latent_ndim: int = 4

    @property
    def name(self) -> str:
        return NAMES.get(self.family, self.family)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.id.rsplit('-', 1)[-1]})"

    @property
    def is_dit(self) -> bool:
        return self.kind != "unet"

    @property
    def allow_context_repeat(self) -> bool:
        # MOSS-Image applies RoPE directly to text tokens, so tiling context changes positional attention.
        return self.kind != "moss_image"

    @property
    def default_context(self) -> int:
        if self.kind == "anima":
            return 512
        if self.kind == "moss_image":
            return 1
        return 77

    def spatial(self, shape) -> tuple:
        if len(shape) == 5:
            return int(shape[0]), int(shape[1]), int(shape[3]), int(shape[4])
        return int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])

    def latent_shape(self, batch: int, height: int, width: int, channels: int = None) -> tuple:
        c = self.in_channels if channels is None else int(channels)
        if self.latent_ndim == 5:
            return (int(batch), c, 1, int(height), int(width))
        return (int(batch), c, int(height), int(width))

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep legacy UNet metadata byte-identical to existing on-disk JSON files.
        if self.kind == "unet" and self.downscale == 8 and self.latent_ndim == 4:
            for k in ("kind", "downscale", "latent_ndim"):
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Arch":
        if not isinstance(d, dict):
            raise TypeError("arch must be a dict")
        id_, family = d["id"], d["family"]
        if (
            not isinstance(id_, str)
            or not id_
            or id_ in (".", "..")
            or "/" in id_
            or "\\" in id_
            or os.path.basename(id_) != id_
            or not isinstance(family, str)
            or not family
        ):
            raise ValueError("invalid arch id or family")
        ints = {}
        for k, lo in (("in_channels", 1), ("context_dim", 1), ("adm_channels", 0), ("multiple", 1)):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"invalid arch field {k}: {v!r}")
            ints[k] = v
        kind = d.get("kind", "unet")
        if kind not in KINDS:
            raise ValueError(f"invalid arch kind: {kind!r}")
        downscale = d.get("downscale", 8)
        if isinstance(downscale, bool) or not isinstance(downscale, int) or downscale < 1:
            raise ValueError(f"invalid arch downscale: {downscale!r}")
        latent_ndim = d.get("latent_ndim", 4)
        if isinstance(latent_ndim, bool) or latent_ndim not in (4, 5):
            raise ValueError(f"invalid arch latent_ndim: {latent_ndim!r}")
        return cls(id=id_, family=family, kind=kind, downscale=downscale, latent_ndim=latent_ndim, **ints)


def loaded_model():
    from modules import sd_models

    sd_model = getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    return sd_model if compat.is_loaded_sd_model(sd_model) else None


def unet_patcher(sd_model=None):
    sd_model = sd_model if sd_model is not None else loaded_model()
    return getattr(getattr(sd_model, "forge_objects", None), "unet", None)


def unet_config(patcher) -> dict:
    mc = compat.model_config(patcher)
    return {k: v for k, v in mc.unet_config.items() if k not in _IGNORED_CONFIG}


def _supported(module, patcher):
    mc = compat.model_config(patcher)
    if mc is None or not hasattr(mc, "unet_config"):
        return None
    mod_type = type(module)
    if mod_type is compat.unet_class() and not getattr(module, "predict_codebook_ids", False):
        # Profiles and runtime assume an 8x latent-to-pixel VAE ratio (SD 1.5/2.x/SDXL); reject 16x models like Mugen.
        if getattr(getattr(mc, "latent_format", None), "spacial_downscale_ratio", 8) != 8:
            return None
        config = unet_config(patcher)
        in_ch, ctx = config.get("in_channels"), config.get("context_dim")
        if (
            config.get("dims", 2) != 2
            or config.get("num_classes") not in (None, "sequential")
            or config.get("use_temporal_attention")
            or config.get("use_temporal_resblock")
            or isinstance(in_ch, bool)
            or not isinstance(in_ch, int)
            or in_ch < 1
            or isinstance(ctx, bool)
            or not isinstance(ctx, int)
            or ctx < 1
        ):
            return None
        return "unet", config

    anima_cls = compat.anima_class()
    if anima_cls is not None and mod_type is anima_cls:
        if getattr(getattr(mc, "latent_format", None), "spacial_downscale_ratio", 8) != 8:
            return None
        config = unet_config(patcher)
        in_ch = getattr(module, "in_channels", config.get("in_channels"))
        out_ch = getattr(module, "out_channels", config.get("out_channels"))
        ctx = config.get("crossattn_emb_channels", 1024)
        patch_s = getattr(module, "patch_spatial", config.get("patch_spatial", 2))
        patch_t = getattr(module, "patch_temporal", config.get("patch_temporal", 1))
        if (
            isinstance(in_ch, bool) or not isinstance(in_ch, int) or in_ch < 1
            or isinstance(out_ch, bool) or not isinstance(out_ch, int) or out_ch < 1
            or isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1
            or patch_s != 2 or patch_t != 1
            or not getattr(module, "blocks", None)
        ):
            return None
        return "anima", config

    moss_cls = compat.moss_class()
    if moss_cls is not None and mod_type is moss_cls:
        if getattr(getattr(mc, "latent_format", None), "spacial_downscale_ratio", 16) != 16:
            return None
        config = unet_config(patcher)
        in_ch = getattr(module, "latent_channels", config.get("latent_channels", 128))
        ctx = _moss_context_dim(module, config)
        if (
            isinstance(in_ch, bool) or not isinstance(in_ch, int) or in_ch < 1
            or isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1
            or not getattr(module, "layers", None)
        ):
            return None
        return "moss_image", config

    return None


def _moss_context_dim(module, config: dict):
    text_proj = getattr(getattr(module, "text_embedder", None), "proj", None)
    if text_proj is not None:
        in_f = getattr(text_proj, "in_features", None)
        if isinstance(in_f, int) and not isinstance(in_f, bool) and in_f >= 1:
            return in_f
        w = getattr(text_proj, "weight", None)
        if w is not None and len(w.shape) >= 2:
            return int(w.shape[1])
    return config.get("text_dim", 4096)


def detect(patcher=None):
    patcher = patcher if patcher is not None else unet_patcher()
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if module is None:
        return None
    cached = _cache.get(id(module))
    if cached is not None and cached[0]() is module:
        return cached[1]
    arch = None
    found = _supported(module, patcher)
    if found is not None:
        kind, config = found
        # Overlay named_parameters() on state_dict() so QuantizedTensor / ParameterGGUF report their logical shapes.
        sd = {**module.state_dict(keep_vars=True), **dict(module.named_parameters())}
        shapes = sorted((k, list(v.shape)) for k, v in sd.items() if not k.endswith(_IGNORED_WEIGHTS))
        family = type(compat.model_config(patcher)).__name__
        digest = hashlib.sha256(json.dumps([sorted((k, repr(v)) for k, v in config.items()), shapes]).encode()).hexdigest()
        if kind == "unet":
            adm = (config.get("adm_in_channels") or 0) if config.get("num_classes") == "sequential" else 0
            arch = Arch(
                id=f"{family.lower()}-{digest[:10]}",
                family=family,
                in_channels=int(config["in_channels"]),
                context_dim=int(config["context_dim"]),
                adm_channels=int(adm),
                multiple=8 * 2 ** (len(config.get("channel_mult", (1, 2, 4, 4))) - 1),  # VAE 8x, then 2x per UNet downsampling
            )
        elif kind == "anima":
            patch_s = int(getattr(module, "patch_spatial", config.get("patch_spatial", 2)))
            arch = Arch(
                id=f"{family.lower()}-{digest[:10]}",
                family=family,
                in_channels=int(getattr(module, "in_channels", config.get("in_channels", 16))),
                context_dim=int(config.get("crossattn_emb_channels", 1024)),
                adm_channels=0,
                multiple=8 * patch_s,
                kind="anima",
                downscale=8,
                latent_ndim=5,
            )
        elif kind == "moss_image":
            ctx = int(_moss_context_dim(module, config))
            arch = Arch(
                id=f"{family.lower()}-{digest[:10]}",
                family=family,
                in_channels=int(getattr(module, "latent_channels", config.get("latent_channels", 128))),
                context_dim=ctx,
                adm_channels=0,
                multiple=16,
                kind="moss_image",
                downscale=16,
                latent_ndim=4,
            )
    _cache.clear()
    _cache[id(module)] = (weakref.ref(module), arch)
    return arch
