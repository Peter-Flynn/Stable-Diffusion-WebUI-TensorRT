import hashlib
import json
import os
import weakref
from dataclasses import asdict, dataclass

_IGNORED_CONFIG = {"dtype", "device", "operations", "disable_unet_model_creation"}
# Left out of the architecture hash: scaled-FP8 checkpoints carry these per-layer scales, which weights.py folds
# into the weights before refitting, so those checkpoints share engines with unscaled ones.
_IGNORED_WEIGHTS = (".scale_weight", ".scale_input")
NAMES = {"SD15": "SD 1.5", "SD20": "SD 2.x", "SDXL": "SDXL", "SDXLRefiner": "SDXL Refiner", "SSD1B": "SSD-1B"}
_cache = {}


@dataclass(frozen=True)
class Arch:
    id: str
    family: str
    in_channels: int
    context_dim: int
    adm_channels: int
    multiple: int

    @property
    def name(self) -> str:
        return NAMES.get(self.family, self.family)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.id.rsplit('-', 1)[-1]})"

    def to_dict(self) -> dict:
        return asdict(self)

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
        return cls(id=id_, family=family, **ints)


def loaded_model():
    from modules import sd_models

    return getattr(getattr(sd_models, "model_data", None), "sd_model", None)


def unet_patcher(sd_model=None):
    sd_model = sd_model if sd_model is not None else loaded_model()
    return getattr(getattr(sd_model, "forge_objects", None), "unet", None)


def unet_config(patcher) -> dict:
    return {k: v for k, v in patcher.model.model_config.unet_config.items() if k not in _IGNORED_CONFIG}


def _supported(module, patcher):
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

    if type(module) is not UNetModel or getattr(module, "predict_codebook_ids", False):
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
    return config


def detect(patcher=None):
    patcher = patcher if patcher is not None else unet_patcher()
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if module is None:
        return None
    cached = _cache.get(id(module))
    if cached is not None and cached[0]() is module:
        return cached[1]
    arch = None
    config = _supported(module, patcher)
    if config is not None:
        shapes = sorted((k, list(v.shape)) for k, v in module.state_dict(keep_vars=True).items() if not k.endswith(_IGNORED_WEIGHTS))
        family = type(patcher.model.model_config).__name__
        digest = hashlib.sha256(json.dumps([sorted((k, repr(v)) for k, v in config.items()), shapes]).encode()).hexdigest()
        adm = (config.get("adm_in_channels") or 0) if config.get("num_classes") == "sequential" else 0
        arch = Arch(
            id=f"{family.lower()}-{digest[:10]}",
            family=family,
            in_channels=int(config["in_channels"]),
            context_dim=int(config["context_dim"]),
            adm_channels=int(adm),
            multiple=8 * 2 ** (len(config.get("channel_mult", (1, 2, 4, 4))) - 1),  # VAE 8x, then 2x per UNet downsampling
        )
    _cache.clear()
    _cache[id(module)] = (weakref.ref(module), arch)
    return arch
