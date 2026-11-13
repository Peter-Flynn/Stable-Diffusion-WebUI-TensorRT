"""The working VRAM of a UNet call, from the architecture up.

A call's working memory is its activations: every tensor the UNet makes on the way, for as long as something
still needs it. Their sizes follow from the architecture and the image's size alone, so they're counted
here without an engine, without weights and without VRAM: reForge's own UNet is built from its
configuration on PyTorch's meta device (shapes only), one call is run through it, and the bytes of every
tensor alive at each point are added up. The most that was alive at once, for each pixel of each image the
call runs, is the architecture's figure (`per_pixel`). It's the same at every image size, batch size and
prompt length (every tensor that matters scales with the image), which is why one figure does for all of
them. A call runs the two halves of the guidance as one batch where a profile takes them, so twice the
images of the pass (runtime.call_bytes has how many).

Where it comes from: the most is alive in the feed-forward layer of the first transformer block of the up
path, at the highest resolution that has attention (it widens to eight times the level's channels), with
the skip connections the up path hasn't used yet. In FP16, for each image a call runs: SDXL (attention
from half resolution down), 157.5 bytes a pixel; SD 1.5 (attention at full resolution), 260.

Attention is counted as one operation, as TensorRT runs it: its output, and no map of every query against
every key (which would grow with the square of the image; an engine's memory doesn't).

That figure is the least a call can do with. A pass takes more (CALL_FACTOR), as seen for SDXL at
1024x1472 (batch sizes 1 and 2) and 1536x2208: the engine's activation memory is 1.28 times it (TensorRT
plans its memory its own way, and runs normalizations in FP32), its inputs and outputs bring that to
1.31, and what the sampler holds of its own between calls to 1.45 to 1.48 at its most. CALL_FACTOR and the
buffer on every estimate (common.BUFFER) together cover that: 1.40 x 1.10 = 1.54."""

import threading
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .common import logger

CALL_FACTOR = 1.40  # a pass's working memory, as a multiple of its UNet's activations (see the module)
FALLBACK = 260  # bytes a pixel where the architecture can't be run through: the most of those that have been (SD 1.5's), so it errs towards PyTorch
PROBE = 64  # the latent's side the architecture is run through at (the figure doesn't depend on it)

_figures = {}  # architecture id -> bytes a pixel of image, for each image a call runs
_lock = threading.Lock()


class _Alive(TorchDispatchMode):
    """Adds up the bytes of the tensors alive as operations make them: `most` is the most at once. What an
    operation returns that is (part of) what it was given, a view or its input changed in place (by the
    operation's own declaration), is that tensor's memory, which stays alive as long as either does."""

    def __init__(self):
        super().__init__()
        self.now = self.most = 0
        self.blocks = {}  # id of a tensor alive -> its memory: [bytes, the tensors alive that are (parts of) it]

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


def _attention(q, k, v, heads, *args, **kwargs):
    """Attention as one operation: its output (a value for every query, as wide as the queries)."""
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def measure(config: dict, in_channels: int, context_dim: int, adm_channels: int, dtype=torch.float16) -> float:
    """The architecture's figure: the bytes a UNet call has alive at its most, for each pixel of each image
    it runs. Takes no VRAM, and a few hundred milliseconds."""
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
    """The architecture's figure (measure), worked out once for each architecture."""
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
    """The working VRAM of a pass whose UNet calls run `batch` images of width x height at once: the
    architecture's figure for them, and what a pass holds around its calls (CALL_FACTOR)."""
    return int(batch * width * height * per_pixel(arch, patcher) * CALL_FACTOR)
