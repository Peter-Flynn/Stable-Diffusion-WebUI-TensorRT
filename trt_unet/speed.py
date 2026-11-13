# Speed Hacks: DeepCache with its refresh interval stretched by AdaptiveDiffusion's smoothness test.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. Most evaluations run only the shallow engine, on
# the deep feature from the last full evaluation.
#
# AdaptiveDiffusion (Ye et al., 2024): while the sampler's latents move smoothly (third-order difference small next to
# the previous first-order one), the noise prediction barely changes. Here that test lets a due full evaluation wait,
# in a pass's second half only: on SDXL a shallow call costs a few percent of a full one, so running it is as cheap as
# reusing the last prediction, and more accurate.
import torch

BRANCH = 0  # shallow engine = input blocks 0..BRANCH and the output blocks from deep_block(BRANCH) on
INTERVAL = 3
MAX_STRETCH = 2
SMOOTHNESS = 0.01  # AdaptiveDiffusion's threshold for SD 1.5 and SDXL
WARMUP = 0.4
TAIL = 0.05
STRETCH_FROM = 0.5
MIN_STEPS = 12
MAX_AGE = INTERVAL - 1 + MAX_STRETCH


def deep_block(unet, branch: int = BRANCH) -> int:
    return len(unet.input_blocks) - 1 - branch


def _embedding(unet, x, timesteps, y):
    from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding

    emb = unet.time_embed(timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(x.dtype))
    if unet.num_classes is not None:
        emb = emb + unet.label_emb(y)
    return emb


# full_forward and shallow_forward mirror reForge's UNetModel._forward, minus patches and ControlNet (runtime.py sends
# those passes to PyTorch). Keep them in sync with it.
def full_forward(unet, x, timesteps, context, y=None, branch: int = BRANCH):
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed

    options = {"original_shape": list(x.shape), "transformer_index": 0}
    emb = _embedding(unet, x, timesteps, y)
    hs, h = [], x
    for module in unet.input_blocks:
        h = forward_timestep_embed(module, h, emb, context, options)
        hs.append(h)
    if unet.middle_block is not None:
        h = forward_timestep_embed(unet.middle_block, h, emb, context, options)
    deep, at = None, deep_block(unet, branch)
    for i, module in enumerate(unet.output_blocks):
        if i == at:
            deep = h
        h = torch.cat([h, hs.pop()], dim=1)
        h = forward_timestep_embed(module, h, emb, context, options, hs[-1].shape if hs else None)
    return unet.out(h.type(x.dtype)), deep


def shallow_forward(unet, x, timesteps, context, y, deep, branch: int = BRANCH):
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed

    options = {"original_shape": list(x.shape), "transformer_index": 0}
    emb = _embedding(unet, x, timesteps, y)
    hs, h = [], x
    for module in unet.input_blocks[: branch + 1]:
        h = forward_timestep_embed(module, h, emb, context, options)
        hs.append(h)
    h = deep
    for module in unet.output_blocks[deep_block(unet, branch):]:
        h = torch.cat([h, hs.pop()], dim=1)
        h = forward_timestep_embed(module, h, emb, context, options, hs[-1].shape if hs else None)
    return unet.out(h.type(x.dtype))


def shallow_shapes(shapes: dict, deep: dict) -> dict:
    scale, channels = deep["scale"], deep["channels"]
    return {**shapes, "deep": [[b, channels, h // scale, w // scale] for b, _, h, w in shapes["sample"]]}


class Schedule:
    def __init__(self, warmup: float = WARMUP, tail: float = TAIL):
        self.warmup, self.tail = warmup, tail
        self.index = None
        self.decision = "full"
        self.last_full = None
        self.stretched = 0
        self.latents = []
        self.slot = 0
        # (slot, sample shape) -> (deep feature, evaluation index). Keyed by shape too: with "Negative Guidance minimum
        # sigma", every other evaluation drops the unconds, so one slot alternates between two batch shapes.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0

    def begin(self, index: int, total, steps, x):
        self.index, self.slot = index, 0
        self.latents = (self.latents + [x.detach().clone()])[-4:]
        self.decision = self._decide(index, total, steps)
        if self.decision == "full":
            self.last_full, self.stretched = index, 0

    def _decide(self, index, total, steps) -> str:
        if not total or not steps or steps < MIN_STEPS:
            return "full"
        if index < round(self.warmup * total) or index >= total - max(1, round(self.tail * total)) or self.last_full is None:
            return "full"
        if index - self.last_full < INTERVAL:
            return "shallow"
        # total == steps: the test compares consecutive steps, so not for samplers that evaluate more than once per step
        if index >= STRETCH_FROM * total and self.stretched < MAX_STRETCH and total == steps and self._smooth():
            self.stretched += 1
            return "shallow"
        return "full"

    def _smooth(self) -> bool:
        if len(self.latents) < 4:
            return False
        x3, x2, x1, x0 = self.latents  # oldest first: the paper's x_{i+2}, x_{i+1}, x_i, x_{i-1}
        previous = x1 - x2  # Eq. 4's threshold is relative to this difference, not the newest one
        third = (x0 - x1) - 2 * previous + (x2 - x3)
        norms = torch.stack([third.float().norm(), previous.float().norm()]).tolist()
        return norms[0] <= SMOOTHNESS * norms[1]

    def call(self, shape) -> tuple:
        if self.index is None:
            self.full_calls += 1
            return None, None
        slot, self.slot = self.slot, self.slot + 1
        kept = self.deep.get((slot, tuple(shape)))
        if self.decision == "shallow" and kept is not None and self.index - kept[1] <= MAX_AGE:
            self.shallow_calls += 1
            return slot, kept[0]
        self.full_calls += 1
        return slot, None

    def keep(self, slot: int, shape, deep):
        self.deep[(slot, tuple(shape))] = (deep, self.index)

    def release(self):
        self.deep.clear()
        self.latents = []
