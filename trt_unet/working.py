# Working VRAM of a UNet call, measured from the architecture alone: reForge's UNet is built on the meta device
# (shapes, no memory), run once, and the bytes of all tensors alive at once are tracked. The peak per pixel per image
# is the same at every size, batch and prompt length, so one figure per architecture (per_pixel) serves for all. It
# peaks in the first up-path transformer's feed-forward at the highest-resolution attention level: FP16 SDXL 157.5
# bytes/pixel, SD 1.5 260. Attention counts as one op with no score matrix, as TensorRT fuses it.
import threading
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .common import logger

# Measured on SDXL (1024x1472 at batch 1 and 2, 1536x2208): TensorRT's activation memory is 1.28x the figure, 1.31x with
# inputs and outputs, and up to 1.48x with what the sampler holds between calls. 1.40 x common.BUFFER (1.10) = 1.54.
CALL_FACTOR = 1.40
FALLBACK = 260  # bytes/pixel if measuring fails: the highest measured (SD 1.5), so errors lean towards PyTorch
PROBE = 64

_figures = {}
_lock = threading.Lock()


class _Alive(TorchDispatchMode):
    # Tracks bytes alive at once. An op output declared as aliasing its input (a view, an in-place op) shares that
    # input's memory block, which is freed when the last tensor using it is.
    def __init__(self):
        super().__init__()
        self.now = self.most = 0
        self.blocks = {}

    def _holds(self, t, block):
        block[1] += 1
        self.blocks[id(t)] = block
        weakref.finalize(t, self._gone, id(t), block)

    def _gone(self, key, block):
        if self.blocks.get(key) is block:
            del self.blocks[key]
        block[1] -= 1
        if not block[1]:
            self.now -= block[0]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        returns = func._schema.returns
        given = next((a for a in args if isinstance(a, torch.Tensor)), None)
        for i, t in enumerate(out if isinstance(out, (tuple, list)) else (out,)):
            if not isinstance(t, torch.Tensor) or t is given:
                continue
            declared = returns[min(i, len(returns) - 1)] if returns else None
            if declared is not None and declared.alias_info is not None:
                block = self.blocks.get(id(given)) if given is not None else None
                if block is not None:
                    self._holds(t, block)
                continue
            block = [t.numel() * t.element_size(), 0]
            self.now += block[0]
            self.most = max(self.most, self.now)
            self._holds(t, block)
        return out


def _attention(q, k, v, heads, *args, **kwargs):  # stand-in: just the output's memory, like TensorRT's fused attention
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def measure(config: dict, in_channels: int, context_dim: int, adm_channels: int, dtype=torch.float16) -> float:
    import ldm_patched.ldm.modules.attention as attention
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

    side = PROBE
    unet = UNetModel(**config, dtype=dtype, device="meta", operations=ops.disable_weight_init).eval()
    feed = [
        torch.empty(1, in_channels, side, side, dtype=dtype, device="meta"),
        torch.empty(1, dtype=dtype, device="meta"),
        torch.empty(1, 77, context_dim, dtype=dtype, device="meta"),
    ]
    y = torch.empty(1, adm_channels, dtype=dtype, device="meta") if adm_channels else None
    originals = (attention.optimized_attention, attention.optimized_attention_masked)
    alive = _Alive()
    attention.optimized_attention = attention.optimized_attention_masked = _attention
    try:
        with torch.no_grad(), alive:
            unet(*feed, y=y)
    finally:
        attention.optimized_attention, attention.optimized_attention_masked = originals
    return alive.most / (side * 8 * side * 8)


def per_pixel(arch, patcher) -> float:
    with _lock:
        if arch.id not in _figures:
            from .arch import unet_config

            try:
                _figures[arch.id] = measure(unet_config(patcher), arch.in_channels, arch.context_dim, arch.adm_channels)
                logger.debug(f"A {arch.name} UNet call takes {_figures[arch.id]:.1f} bytes of VRAM per pixel of each image.")
            except Exception as e:
                _figures[arch.id] = FALLBACK
                logger.warning(
                    f"Couldn't work out how much VRAM a {arch.name} UNet call takes ({type(e).__name__}: {e}), so it's assumed to be "
                    f"{FALLBACK} bytes per pixel, the most of any known architecture. Some passes that would fit may run on PyTorch."
                )
        return _figures[arch.id]


def call_bytes(arch, patcher, batch: int, width: int, height: int) -> int:
    return int(batch * width * height * per_pixel(arch, patcher) * CALL_FACTOR)
