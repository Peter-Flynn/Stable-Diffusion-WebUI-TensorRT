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
    if not flat.is_contiguous() or flat.storage_offset() % per:
        flat = flat.clone(memory_format=torch.contiguous_format)  # .view(torch.int64) needs contiguous, word-aligned storage
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

    _LORA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def needs_gpu(self, name: str) -> bool:
        return bool(self.patches.get(PREFIX + name) or self.converters.get(name) is not None)

    def fast_lora(self, name: str):
        # Standard LoRA weights (up @ down) can be staged alongside base in pinned memory and merged via in-place FP32
        # addmm_, avoiding per-weight synchronous H2D copies and intermediate lora_diff allocations. Anything unusual
        # (converters, DoRA, LoCon mid weights, offsets, custom functions, strength_model != 1, LoHa/LoKr) returns None.
        if self.converters.get(name) is not None:
            return None
        patches = self.patches.get(PREFIX + name)
        if not patches:
            return None
        base = self.base(name)
        if type(base) is not torch.Tensor or base.ndim < 2 or base.shape[0] <= 0:
            return None
        rows, total = base.shape[0], base.numel()
        items = []
        for entry in patches:
            if not isinstance(entry, (tuple, list)) or len(entry) < 5:
                return None
            strength, v, strength_model, offset, func = entry[:5]
            if strength_model != 1.0 or offset is not None or func is not None:
                return None
            if type(v).__name__ == "LoRAAdapter" and type(v).__module__.endswith(".lora") and hasattr(v, "weights"):
                ws = v.weights
            elif isinstance(v, tuple) and len(v) == 2 and v[0] == "lora":
                ws = v[1]
            else:
                return None
            if not isinstance(ws, (tuple, list)) or len(ws) < 6:
                return None
            up, down, alpha, mid, dora_scale, reshape = ws[:6]
            if mid is not None or dora_scale is not None or reshape is not None:
                return None
            if type(up) is not torch.Tensor or type(down) is not torch.Tensor:
                return None
            if up.dtype not in self._LORA_DTYPES or down.dtype not in self._LORA_DTYPES:
                return None
            if up.ndim < 2 or down.ndim < 2 or down.shape[0] <= 0 or up.shape[0] != rows:
                return None
            rank = up.numel() // rows
            in_cols = down.numel() // down.shape[0]
            if rank != down.shape[0] or rows * in_cols != total:
                return None
            scale = float(strength) * (float(alpha) / float(down.shape[0]) if alpha is not None else 1.0)
            items.append((up, down, scale))
        return items

    @staticmethod
    def apply_fast_lora(gpu_base: torch.Tensor, gpu_items: list, dtype=torch.float16) -> torch.Tensor:
        weight = gpu_base.to(torch.float32, copy=True, memory_format=torch.contiguous_format)
        flat = weight.reshape(weight.shape[0], -1)
        for up, down, scale in gpu_items:
            if scale != 0.0:
                flat.addmm_(up.to(torch.float32).flatten(start_dim=1), down.to(torch.float32).flatten(start_dim=1), alpha=scale)
        return weight.to(dtype)

    def apply_on_gpu(self, name: str, gpu_base: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
        items = self.fast_lora(name)
        if items is not None:
            dev = gpu_base.device
            return self.apply_fast_lora(gpu_base, [(u.to(dev), d.to(dev), s) for u, d, s in items], dtype)
        # copy=True because convert and patch work in place.
        weight = gpu_base.to(torch.float32, copy=True)
        convert = self.converters.get(name)
        if convert is not None:
            weight = convert(weight, inplace=True)
        patches = self.patches.get(PREFIX + name)
        if patches:
            from ldm_patched.modules.lora import calculate_weight

            weight = calculate_weight(patches, weight, PREFIX + name)
        return weight.to(dtype)

    def tensor(self, name: str, device="cpu", dtype=torch.float16) -> torch.Tensor:
        base = self.base(name)
        if not self.needs_gpu(name):
            return base if device is None else base.to(device)
        # Two steps: one .to(device, float32) would convert on the CPU first.
        weight = self.apply_on_gpu(name, base.to(common.device()), dtype)
        return weight if device is None else weight.to(device)
