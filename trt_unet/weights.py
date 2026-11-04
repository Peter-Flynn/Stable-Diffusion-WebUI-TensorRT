"""The weights an engine runs with: those PyTorch would use for the same UNet patcher, i.e. the loaded
checkpoint's with the patcher's patches (LoRAs, LyCORIS, ...) applied the way reForge applies them."""

import hashlib
import weakref

import torch

from . import common

PREFIX = "diffusion_model."


class _Same:
    """Equal only to a _Same of the very same object while it exists. Unlike id(), which an object made
    later can reuse (e.g. another LoRA of the same rank), and without keeping the object alive."""

    __slots__ = ("ref",)

    def __init__(self, obj):
        try:
            self.ref = weakref.ref(obj)
        except TypeError:  # not weak-referenceable: keep it
            self.ref = lambda: obj

    def __eq__(self, other):
        return isinstance(other, _Same) and self.ref() is not None and self.ref() is other.ref()

    __hash__ = None


def _identity(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", _Same(value), tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return tuple(_identity(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _identity(v)) for k, v in value.items()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return ("object", _Same(value))


class EffectiveWeights:
    def __init__(self, patcher):
        self.patcher = patcher
        self.module = patcher.model.diffusion_model
        self.patches = getattr(patcher, "patches", {}) or {}
        # Layers that store a weight in another form than they use it, e.g. scaled FP8 checkpoints' Linear
        # layers (weight * scale_weight). reForge converts their weights before patching them; so does this.
        self.converters = {
            f"{n}.{p}": getattr(m, f"convert_{p}")
            for n, m in self.module.named_modules()
            for p, _ in m.named_parameters(recurse=False)
            if callable(getattr(m, f"convert_{p}", None))
        }
        self.refresh()
        self.base_version = (_Same(self.module), self._fingerprint())

    def refresh(self):
        """Loading and unloading replace the module's parameters: read them again before a refit."""
        self.params = self.module.state_dict(keep_vars=True)
        self.backup = getattr(self.patcher, "backup", {}) or {}

    def has(self, name: str) -> bool:
        return name in self.params or PREFIX + name in self.backup

    def base(self, name: str) -> torch.Tensor:
        """The unpatched weight (reForge keeps the original of one it patched in place). `.data` reads it
        even if moving it outside inference mode broke its version counter."""
        backup = self.backup.get(PREFIX + name)
        if backup is None and name not in self.params:
            raise RuntimeError(f"The engine has a weight this UNet doesn't ({name}); rebuild the engine")
        return (getattr(backup, "weight", backup) if backup is not None else self.params[name]).data

    def _fingerprint(self) -> str:
        """Tells checkpoints apart when one is loaded into the same module. Every weight counts: two
        checkpoints can differ in some layers only (e.g. one is the other with a LoRA merged in)."""
        heads = [self.base(n).reshape(-1)[:16].float().cpu() for n in sorted(self.params)]
        return hashlib.sha256(torch.cat(heads).numpy().tobytes()).hexdigest() if heads else ""

    def is_patched(self, name: str) -> bool:
        return bool(self.patches.get(PREFIX + name))

    def signature(self, name: str):
        patches = self.patches.get(PREFIX + name)
        return self.base_version, _identity(patches) if patches else None

    def tensor(self, name: str) -> torch.Tensor:
        """The weight PyTorch runs with: as stored, or, if it's converted or patched, computed in FP32 on
        the GPU (converted, then patched, like reForge does) and kept in FP16."""
        base = self.base(name)
        patches = self.patches.get(PREFIX + name)
        convert = self.converters.get(name)
        if not patches and convert is None:
            return base
        weight = base.to(common.mm().get_torch_device(), torch.float32, copy=True)
        if convert is not None:
            weight = convert(weight, inplace=True)
        if patches:
            try:
                from ldm_patched.modules.lora import calculate_weight as calculate
            except ImportError:
                calculate = self.patcher.calculate_weight
            weight = calculate(patches, weight, PREFIX + name)
        return weight.to(torch.float16).cpu()  # in RAM until the refit: patched SDXL weights add up to GBs
