"""A UNet's architecture: what engines are compiled for. Every checkpoint with the same architecture
(fine-tunes, merges, v-prediction variants) runs on the same engines with its own weights."""

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass

_IGNORED_CONFIG = {"dtype", "device", "operations", "disable_unet_model_creation"}
NAMES = {"SD15": "SD 1.5", "SD20": "SD 2.x", "SDXL": "SDXL", "SDXLRefiner": "SDXL Refiner", "SSD1B": "SSD-1B"}
_cache = {}


@dataclass(frozen=True)
class Arch:
    id: str
    family: str
    in_channels: int
    context_dim: int
    adm_channels: int
    multiple: int  # image sizes are multiples of this

    @property
    def name(self) -> str:
        """For people: "SD 1.5", "SDXL", ... (reForge's model config class otherwise)."""
        return NAMES.get(self.family, self.family)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.id.rsplit('-', 1)[-1]})"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Arch":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def loaded_model():
    """The loaded checkpoint, without loading one."""
    from modules import sd_models

    return getattr(getattr(sd_models, "model_data", None), "sd_model", None)


def unet_patcher(sd_model=None):
    sd_model = sd_model if sd_model is not None else loaded_model()
    return getattr(getattr(sd_model, "forge_objects", None), "unet", None)


def unet_config(patcher) -> dict:
    return {k: v for k, v in patcher.model.model_config.unet_config.items() if k not in _IGNORED_CONFIG}


def _supported(module) -> bool:
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

    return isinstance(module, UNetModel)


def detect(patcher=None):
    """The Arch of a UNet patcher (default: the loaded checkpoint's), or None if engines can't run it."""
    patcher = patcher if patcher is not None else unet_patcher()
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if module is None:
        return None
    cached = _cache.get(id(module))
    if cached is not None and cached[0]() is module:
        return cached[1]
    arch = None
    if _supported(module):
        config = unet_config(patcher)
        shapes = sorted((k, list(v.shape)) for k, v in module.state_dict(keep_vars=True).items())
        family = type(patcher.model.model_config).__name__
        digest = hashlib.sha256(json.dumps([sorted((k, repr(v)) for k, v in config.items()), shapes]).encode()).hexdigest()
        adm = (config.get("adm_in_channels") or 0) if config.get("num_classes") == "sequential" else 0
        arch = Arch(
            id=f"{family.lower()}-{digest[:10]}",
            family=family,
            in_channels=int(config["in_channels"]),
            context_dim=int(config.get("context_dim") or 0),
            adm_channels=int(adm),
            multiple=8 * 2 ** (len(config.get("channel_mult", (1, 2, 4, 4))) - 1),
        )
    _cache.clear()
    _cache[id(module)] = (weakref.ref(module), arch)
    return arch
