"""Speed Hacks: DeepCache's split UNet, refreshed on a schedule that AdaptiveDiffusion's test stretches.

DeepCache (Ma et al., 2023): between denoising steps the UNet's deep features change little; its outermost
blocks matter more. Every engine is built in two parts: the UNet, which also outputs the feature entering
its last output block (the "deep" feature), and a shallow engine of only its outermost blocks, which takes
that feature. On most evaluations only the shallow engine runs, on the deep feature of the last full one.

AdaptiveDiffusion (Ye et al., 2024): while the sampler's latents move smoothly (their third-order
difference is small next to the first-order one), the noise prediction barely changes, so a step can
reuse the last one. Here that test decides when a due full evaluation can wait: on SDXL a shallow call
costs a few percent of a full one, so it's no slower than reusing the prediction, and closer to it.

The first and last evaluations of a pass always run in full, and passes of fewer evaluations than
MIN_EVALUATIONS (e.g. few-step LoRAs) don't use any of it."""

import torch

BRANCH = 0  # the shallow part: the input blocks up to this one and the output blocks from its partner on
INTERVAL = 3  # a full evaluation every 3 at least...
MAX_STRETCH = 2  # ...or up to this many later while the sampler's trajectory is smooth
SMOOTHNESS = 0.01  # AdaptiveDiffusion's threshold (its SD 1.5 and SDXL setting)
WARMUP = 2  # evaluations run in full at the start of a pass
TAIL = 0.1  # the share of a pass's evaluations run in full at its end
MIN_EVALUATIONS = 12


# --- The two halves, as exported. Like UNetModel._forward without what an engine can't run anyway
# (patches, ControlNet), which runtime.py leaves to PyTorch.


def deep_block(unet, branch: int = BRANCH) -> int:
    """The output block whose input is the deep feature: the partner of input block `branch`."""
    return len(unet.input_blocks) - 1 - branch


def _embedding(unet, x, timesteps, y):
    from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding

    emb = unet.time_embed(timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(x.dtype))
    if unet.num_classes is not None:
        emb = emb + unet.label_emb(y)
    return emb


def full_forward(unet, x, timesteps, context, y=None, branch: int = BRANCH):
    """The UNet's output, and its deep feature."""
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
    """The UNet's output from its outermost blocks and a deep feature."""
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
    """A profile's UNet input shapes for the shallow engine: the UNet's, and the deep feature's."""
    scale, channels = deep["scale"], deep["channels"]
    return {**shapes, "deep": [[b, channels, h // scale, w // scale] for b, _, h, w in shapes["sample"]]}


# --- When to run which


class Schedule:
    """Decides, for each evaluation of a sampling pass (a call of reForge's CFG denoiser, which may run
    several UNet calls: cond and uncond, or more), whether its UNet calls run in full or shallow, and keeps
    each call's deep feature. A UNet call that has no deep feature of its shape to use runs in full."""

    def __init__(self):
        self.decision = "full"
        self.last_full = None  # the last evaluation decided full
        self.stretched = 0  # evaluations it was put off by since it was due
        self.latents = []  # the last 4 evaluations' sampler latents
        self.slot = 0  # UNet calls so far in this evaluation
        # (slot, sample shape) -> deep feature: e.g. with "Negative Guidance minimum sigma", every other
        # evaluation runs its conds without the unconds, so a slot's calls alternate between two shapes
        self.deep = {}
        self.shallow_calls = self.full_calls = 0

    def begin(self, index: int, total, steps, x):
        """Evaluation `index` of `total` (the sampler takes `steps` steps, so more evaluations than steps
        means it evaluates more than once per step) starts, at sampler latent `x`."""
        self.slot = 0
        self.latents = (self.latents + [x.detach().clone()])[-4:]
        self.decision = self._decide(index, total, steps)
        if self.decision == "full":
            self.last_full, self.stretched = index, 0

    def _decide(self, index, total, steps) -> str:
        if not total or total < MIN_EVALUATIONS:
            return "full"
        if index < WARMUP or index >= total - max(1, round(TAIL * total)) or self.last_full is None:
            return "full"
        if index - self.last_full < INTERVAL:
            return "shallow"
        # AdaptiveDiffusion's test compares consecutive steps, so not for samplers that evaluate more than once per step
        if self.stretched < MAX_STRETCH and total == steps and self._smooth():
            self.stretched += 1
            return "shallow"
        return "full"

    def _smooth(self) -> bool:
        if len(self.latents) < 4:
            return False
        x3, x2, x1, x0 = self.latents  # oldest first
        first = x0 - x1
        third = first - 2 * (x1 - x2) + (x2 - x3)
        norms = torch.stack([third.float().norm(), first.float().norm()]).tolist()
        return norms[0] <= SMOOTHNESS * norms[1]

    def call(self, shape) -> tuple:
        """(slot, the deep feature to run it shallow on, or None to run it in full) for the next UNet call."""
        slot, self.slot = self.slot, self.slot + 1
        cached = self.deep.get((slot, tuple(shape)))
        if self.decision == "shallow" and cached is not None:
            self.shallow_calls += 1
            return slot, cached
        self.full_calls += 1
        return slot, None

    def keep(self, slot: int, shape, deep):
        self.deep[(slot, tuple(shape))] = deep
