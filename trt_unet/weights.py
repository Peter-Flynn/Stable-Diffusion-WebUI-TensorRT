import hashlib
import weakref

import torch

from . import common, compat, quant

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


def _head_bytes(t: torch.Tensor, limit: int) -> torch.Tensor:
    raw = t.detach()
    if raw.element_size() == 1:
        raw = raw.view(torch.uint8)
    return raw.reshape(-1)[:limit].contiguous().view(torch.uint8).cpu()


def _fingerprint_head(t) -> torch.Tensor:
    qdata = getattr(t, "_qdata", None)
    if isinstance(qdata, torch.Tensor):
        parts = [_head_bytes(qdata, 16)]
        params = getattr(t, "_params", None)
        if params is not None:
            for attr in ("scale", "block_scale", "s_channel", "codebook", "correction"):
                val = getattr(params, attr, None)
                if isinstance(val, torch.Tensor):
                    parts.append(_head_bytes(val, 4))
        return torch.cat(parts) if len(parts) > 1 else parts[0]
    return _head_bytes(t.data, 16)


class EffectiveWeights:
    def __init__(self, patcher):
        self.patcher = patcher
        self.module = patcher.model.diffusion_model
        self.patches = getattr(patcher, "patches", {}) or {}
        self.wrappers = getattr(patcher, "weight_wrapper_patches", {}) or {}
        # Layers that store a weight in another form than they compute with (scaled-FP8 / MixedPrecisionOps Linear).
        # reForge/Neo converts these before applying patches, and so does tensor().
        self.converters = {
            f"{n}.{p}": getattr(m, f"convert_{p}")
            for n, m in self.module.named_modules()
            for p, _ in m.named_parameters(recurse=False)
            if callable(getattr(m, f"convert_{p}", None))
        }
        self.refresh()
        self.base_version = (_Same(self.module), self._fingerprint())

    def refresh(self):  # reForge/Neo replaces the module's parameters when it loads or unloads it
        # Overlay named_parameters() so QuantizedTensor / ParameterGGUF wrappers aren't unpacked to raw bytes.
        self.params = {**self.module.state_dict(keep_vars=True), **dict(self.module.named_parameters())}
        self.backup = getattr(self.patcher, "backup", {}) or {}
        self.patches = getattr(self.patcher, "patches", {}) or {}
        self.wrappers = getattr(self.patcher, "weight_wrapper_patches", {}) or {}

    def _patches_for(self, name: str):
        base_name = quant.base_weight_name(name)
        full = PREFIX + base_name
        baked = self.patches.get(full)
        wrappers = self.wrappers.get(full)
        if not wrappers:
            return baked
        online = [entry for w in wrappers if compat.is_online_lora_patch(w) for entry in w.patch]
        if not online:
            return baked
        return (list(baked) if baked else []) + online

    def has(self, name: str) -> bool:
        base_name = quant.base_weight_name(name)
        return base_name in self.params or PREFIX + base_name in self.backup

    def base(self, name: str) -> torch.Tensor:
        base_name = quant.base_weight_name(name)
        backup = self.backup.get(PREFIX + base_name)
        if backup is None and base_name not in self.params:
            raise RuntimeError(f"The engine has a weight this model doesn't ({base_name}); rebuild the engine")
        # reForge/Neo keeps the original of a weight it patched in place in `backup`. Keep QuantizedTensor/ParameterGGUF
        # wrappers intact so dequantization sees their metadata; use `.data` on plain Parameters for inference-mode safety.
        src = getattr(backup, "weight", backup) if backup is not None else self.params[base_name]
        if compat.is_quantized_weight(src):
            return src
        return src.data

    def _fingerprint(self) -> str:  # samples every weight: two checkpoints can differ in only a few layers (a merged LoRA)
        heads = [_fingerprint_head(self.base(n)) for n in sorted(self.params)]
        return hashlib.sha256(torch.cat(heads).numpy().tobytes()).hexdigest() if heads else ""

    def is_patched(self, name: str) -> bool:
        if quant.is_any_scale_weight(name):
            return False
        return bool(self._patches_for(name))

    def signature(self, name: str):
        if quant.is_act_scale_weight(name):
            return self.base_version, "act_scale"
        patches = self._patches_for(name)
        return self.base_version, _identity(patches) if patches else None

    _LORA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def needs_gpu(self, name: str) -> bool:
        if quant.is_act_scale_weight(name):
            return False
        if quant.is_scale_weight(name):
            return True
        base_name = quant.base_weight_name(name)
        return bool(self._patches_for(base_name) or self.converters.get(base_name) is not None or compat.is_quantized_weight(self.base(base_name)))

    def act_scale(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self, "_int8_act_scales"):
            from . import arch as arch_mod

            arch = arch_mod.detect(self.patcher)
            self._int8_act_scales = quant.calibrate_int8_scales(arch, self)
        base_name = quant.base_weight_name(name)
        val = self._int8_act_scales.get(base_name)
        if val is None or not (val > 0.0):
            val = quant.fallback_int8_act_scale(base_name)
        min_scale = 1e-4 if dtype == torch.float16 else 1e-12
        return torch.tensor(max(min_scale, float(val)), dtype=dtype)

    def fast_lora(self, name: str):
        # Standard LoRA weights (up @ down) can be staged alongside base in pinned memory and merged via in-place FP32
        # addmm_, avoiding per-weight synchronous H2D copies and intermediate lora_diff allocations. Anything unusual
        # (converters, quantized weights, DoRA, LoCon mid weights, offsets, custom functions, strength_model != 1, LoHa/LoKr) returns None.
        base_name = quant.base_weight_name(name)
        if self.converters.get(base_name) is not None:
            return None
        patches = self._patches_for(base_name)
        if not patches:
            return None
        base = self.base(base_name)
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
    def apply_fast_lora(gpu_base: torch.Tensor, gpu_items: list, dtype=torch.float16, out: torch.Tensor = None) -> torch.Tensor:
        weight = gpu_base.to(torch.float32, copy=True, memory_format=torch.contiguous_format)
        flat = weight.reshape(weight.shape[0], -1)
        for up, down, scale in gpu_items:
            if scale != 0.0:
                flat.addmm_(up.to(torch.float32).flatten(start_dim=1), down.to(torch.float32).flatten(start_dim=1), alpha=scale)
        if out is not None:
            out.copy_(weight)
            return out
        return weight if dtype == torch.float32 else weight.to(dtype)

    def apply_fp32_on_gpu(self, name: str, gpu_base: torch.Tensor, gpu_lora: list = None) -> torch.Tensor:
        base_name = quant.base_weight_name(name)
        if gpu_lora is not None:
            return self.apply_fast_lora(gpu_base, gpu_lora, torch.float32)
        items = self.fast_lora(base_name)
        if items is not None:
            dev = gpu_base.device
            return self.apply_fast_lora(gpu_base, [(u.to(dev), d.to(dev), s) for u, d, s in items], torch.float32)
        weight = compat.dequantize_weight(gpu_base, self.converters.get(base_name))
        patches = self._patches_for(base_name)
        if patches:
            weight = compat.merge_weight(patches, weight, PREFIX + base_name)
        return weight if weight.dtype == torch.float32 else weight.float()

    def quantize_on_gpu(self, name: str, gpu_base: torch.Tensor, quant_kind: str, scale_dtype: torch.dtype, gpu_lora: list = None) -> tuple:
        w_fp32 = self.apply_fp32_on_gpu(name, gpu_base, gpu_lora=gpu_lora)
        can_inplace = w_fp32 is not gpu_base and (
            type(gpu_base) is not torch.Tensor or w_fp32.untyped_storage().data_ptr() != gpu_base.untyped_storage().data_ptr()
        )
        return quant.quantize_for_refit(
            w_fp32,
            quant_kind,
            scale_dtype,
            transposed=quant.is_transposed_weight(name),
            in_place=can_inplace,
        )

    def apply_on_gpu(self, name: str, gpu_base: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
        base_name = quant.base_weight_name(name)
        items = self.fast_lora(base_name)
        if items is not None:
            dev = gpu_base.device
            return self.apply_fast_lora(gpu_base, [(u.to(dev), d.to(dev), s) for u, d, s in items], dtype)
        weight = compat.dequantize_weight(gpu_base, self.converters.get(base_name))
        patches = self._patches_for(base_name)
        if patches:
            weight = compat.merge_weight(patches, weight, PREFIX + base_name)
        return weight.to(dtype)

    def tensor(self, name: str, device="cpu", dtype=torch.float16) -> torch.Tensor:
        if quant.is_act_scale_weight(name):
            t = self.act_scale(name, dtype)
            return t if device is None else t.to(device)
        base = self.base(name)
        if not self.needs_gpu(name):
            return base if device is None else base.to(device)
        # Two steps: one .to(device, float32) would convert on the CPU first.
        weight = self.apply_on_gpu(name, base.to(common.device()), dtype)
        return weight if device is None else weight.to(device)
