"""The weights an engine runs with: those PyTorch would use for the same UNet patcher, i.e. the loaded
checkpoint's with the patcher's patches (LoRAs, LyCORIS, ...) applied the way reForge applies them."""

import torch

from . import common

PREFIX = "diffusion_model."


def _identity(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value), tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return tuple(_identity(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _identity(v)) for k, v in value.items()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return ("object", id(value))


class EffectiveWeights:
    def __init__(self, patcher):
        self.patcher = patcher
        self.module = patcher.model.diffusion_model
        self.patches = getattr(patcher, "patches", {}) or {}
        self.refresh()
        self.base_version = (id(self.module), self._fingerprint())

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

    def _fingerprint(self):
        """Tells checkpoints apart when one is loaded into the same module."""
        names = sorted(self.params)
        sample = [names[int(i * (len(names) - 1) / 7)] for i in range(8)] if names else []
        return tuple(float(self.base(n).reshape(-1)[:64].float().sum()) for n in sample)

    def signature(self, name: str):
        patches = self.patches.get(PREFIX + name)
        return self.base_version, _identity(patches) if patches else None

    def tensor(self, name: str) -> torch.Tensor:
        base = self.base(name)
        patches = self.patches.get(PREFIX + name)
        if not patches:
            return base
        try:
            from ldm_patched.modules.lora import calculate_weight as calculate
        except ImportError:
            calculate = self.patcher.calculate_weight
        weight = base.to(common.mm().get_torch_device(), torch.float32, copy=True)
        weight = calculate(patches, weight, PREFIX + name)
        return weight.to(torch.float16).cpu()  # in RAM until the refit: patched SDXL weights add up to GBs
