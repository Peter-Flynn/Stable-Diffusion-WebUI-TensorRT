"""The weights an engine runs with: those PyTorch would use for the same UNet patcher, i.e. the loaded
checkpoint's with the patcher's patches (LoRAs, LyCORIS, ...) applied the way reForge applies them.

Weights that have to be computed (patched or converted) stay in RAM once computed (_computed), so an engine
that loads again (after reForge evicted it, or another profile's engine for a Hires. fix) is refitted from
them instead of computing them again."""

import hashlib
import weakref

import torch

from . import common

PREFIX = "diffusion_model."
# weight name -> (key, the computed FP16 weight in RAM): the latest of each. The key is of the tensors' contents
# it was computed from (see EffectiveWeights.key) rather than naming them: extensions can change a UNet's weights
# in place, so only their contents say whether the computed weight is still the one PyTorch would use.
_computed = {}


def clear_computed():
    _computed.clear()


def computed_bytes() -> int:
    return sum(t.numel() * t.element_size() for _, t in _computed.values())


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


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def checksum(t: torch.Tensor) -> int:
    """The sum of a tensor's bits taken as 64-bit words (wrapping), computed on its own device, for any dtype
    (FP8 and BF16 included). Exact, so the same wherever and however the tensor is stored, and changed by any
    change to a value. Many times quicker than hashing its bytes, and a tensor in VRAM isn't copied to RAM
    for it."""
    flat = t.detach().reshape(-1)
    size = flat.element_size()
    per = 8 // size  # values a word
    if flat.storage_offset() % per:
        flat = flat.clone()  # (a view can only start on a word of its storage)
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
        self.reused = 0  # computed weights taken from RAM
        for name in [n for n in _computed if not self.computed(n)]:  # no longer patched or converted
            del _computed[name]

    def refresh(self):
        """Loading and unloading replace the module's parameters: read them again before a refit."""
        self.params = self.module.state_dict(keep_vars=True)
        self.backup = getattr(self.patcher, "backup", {}) or {}
        self._hashes = {}
        self._layers = {}
        for n in self.params:
            self._layers.setdefault(n.rpartition(".")[0], []).append(n)

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

    def computed(self, name: str) -> bool:
        """Whether the weight PyTorch runs with is computed from the stored one: patched or converted."""
        return self.is_patched(name) or name in self.converters

    def signature(self, name: str):
        patches = self.patches.get(PREFIX + name)
        return self.base_version, _identity(patches) if patches else None

    def _hash(self, name: str) -> bytes:
        """A hash of what every tensor of the weight's layer holds (its own and e.g. an FP8 scale a
        conversion uses), unpatched: their names, types, shapes and checksums."""
        h = hashlib.sha1()
        for n in sorted(self._layers.get(name.rpartition(".")[0], [name])):
            t = self.base(n)
            h.update(f"{n} {t.dtype} {tuple(t.shape)} {checksum(t)};".encode())
        return h.digest()

    def prepare(self, names):
        """Hashes the layers of the computed weights among `names`, ahead of a refit that asks for them one
        at a time."""
        for n in names:
            if self.computed(n) and n not in self._hashes:
                self._hashes[n] = self._hash(n)

    def key(self, name: str):
        """What a computed weight is computed from: its layer's contents and its patches (by identity)."""
        if name not in self._hashes:
            self._hashes[name] = self._hash(name)
        patches = self.patches.get(PREFIX + name)
        return self._hashes[name], _identity(patches) if patches else None

    def tensor(self, name: str, device="cpu") -> torch.Tensor:
        """The weight PyTorch runs with, on `device`: as stored, or, if it's converted or patched, computed
        in FP32 on the GPU (converted, then patched, like reForge does) and kept in FP16, in RAM too."""
        base = self.base(name)
        patches = self.patches.get(PREFIX + name)
        convert = self.converters.get(name)
        if not patches and convert is None:
            return base.to(device)
        key = self.key(name)
        kept = _computed.get(name)
        if kept is not None and kept[0] == key:
            self.reused += 1
            return kept[1].to(device)
        _computed.pop(name, None)  # not kept alongside the new one
        weight = base.to(common.mm().get_torch_device(), torch.float32, copy=True)
        if convert is not None:
            weight = convert(weight, inplace=True)
        if patches:
            try:
                from ldm_patched.modules.lora import calculate_weight as calculate
            except ImportError:
                calculate = self.patcher.calculate_weight
            weight = calculate(patches, weight, PREFIX + name)
        weight = weight.to(torch.float16)
        host = weight.to("cpu")
        _computed[name] = (key, host)
        return host if torch.device(device).type == "cpu" else weight.to(device)
