import hashlib
import weakref

import torch

from . import common

PREFIX = "diffusion_model."


class _Same:
    # Identity of an object without keeping it alive. id() isn't enough: a later object (e.g. another LoRA of the
    # same rank) can reuse a freed object's id.
    __slots__ = ("ref",)

    def __init__(self, obj):
        try:
            self.ref = weakref.ref(obj)
        except TypeError:
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


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def checksum(t: torch.Tensor) -> int:
    # The tensor's raw bits summed as int64 words, on its own device: exact for every dtype (FP8, BF16 included),
    # changes with any value, much faster than hashing, and no copy from VRAM to RAM.
    flat = t.detach().reshape(-1)
    size = flat.element_size()
    per = 8 // size
    if flat.storage_offset() % per:
        flat = flat.clone()  # .view(torch.int64) needs a word-aligned storage offset
    whole = flat.numel() // per * per
    total = flat[:whole].view(torch.int64).sum()
    if whole < flat.numel():
        total = total + flat[whole:].view(_BITS[size]).sum(dtype=torch.int64)
    return int(total)


class EffectiveWeights:
    def __init__(self, patcher):
        self.patcher = patcher
        self.module = patcher.model.diffusion_model
        self.patches = getattr(patcher, "patches", {}) or {}
        # Layers that store a weight in another form than they compute with (scaled-FP8 Linear: weight * scale_weight).
        # reForge converts these before applying patches, and so does tensor().
        self.converters = {
            f"{n}.{p}": getattr(m, f"convert_{p}")
            for n, m in self.module.named_modules()
            for p, _ in m.named_parameters(recurse=False)
            if callable(getattr(m, f"convert_{p}", None))
        }
        self.refresh()
        self.base_version = (_Same(self.module), self._fingerprint())

    def refresh(self):  # reForge replaces the module's parameters when it loads or unloads it
        self.params = self.module.state_dict(keep_vars=True)
        self.backup = getattr(self.patcher, "backup", {}) or {}

    def has(self, name: str) -> bool:
        return name in self.params or PREFIX + name in self.backup

    def base(self, name: str) -> torch.Tensor:
        backup = self.backup.get(PREFIX + name)
        if backup is None and name not in self.params:
            raise RuntimeError(f"The engine has a weight this UNet doesn't ({name}); rebuild the engine")
        # reForge keeps the original of a weight it patched in place in `backup`. `.data` because a tensor moved outside
        # inference mode has a broken version counter, and reading it directly raises.
        return (getattr(backup, "weight", backup) if backup is not None else self.params[name]).data

    def _fingerprint(self) -> str:  # samples every weight: two checkpoints can differ in only a few layers (a merged LoRA)
        heads = [self.base(n).reshape(-1)[:16].float().cpu() for n in sorted(self.params)]
        return hashlib.sha256(torch.cat(heads).numpy().tobytes()).hexdigest() if heads else ""

    def is_patched(self, name: str) -> bool:
        return bool(self.patches.get(PREFIX + name))

    def signature(self, name: str):
        patches = self.patches.get(PREFIX + name)
        return self.base_version, _identity(patches) if patches else None

    def tensor(self, name: str, device="cpu") -> torch.Tensor:
        base = self.base(name)
        patches = self.patches.get(PREFIX + name)
        convert = self.converters.get(name)
        if not patches and convert is None:
            return base.to(device)
        # Two steps: one .to(device, float32) would convert on the CPU first. copy=True because convert and patch work in place.
        weight = base.to(common.mm().get_torch_device()).to(torch.float32, copy=True)
        if convert is not None:
            weight = convert(weight, inplace=True)
        if patches:
            from ldm_patched.modules.lora import calculate_weight

            weight = calculate_weight(patches, weight, PREFIX + name)
        return weight.to(torch.float16).to(device)
