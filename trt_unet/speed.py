# Speed Hacks: DeepCache with its refresh interval stretched while a measured error allows.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. Most evaluations run only the shallow engine, on the
# deep feature from the last full evaluation: INTERVAL - 1 in a row, more while checks allow.
#
# Checks: a full call from CHECKS calls before the first shallow call until the full steps at the end also runs the
# shallow engine as the shallow call at its index would have run, on the deep feature it replaces, and compares
# (fresh.py). That is the error a shallow call of that age actually makes, for any sampler, noise included; it costs a
# shallow call, a few percent of a full one on SDXL. A due full call waits while the latest check's error, scaled
# linearly to the age the deep feature would reach, stays within budget(quality) times the error of the first check
# after warmup (the error DeepCache's own interval made there). Only checks at a sampler step's first call count: DPM++
# SDE and Heun call the UNet twice per step (reForge's total_steps / steps), at two kinds of point whose errors differed
# ~50x in simulation. This replaces AdaptiveDiffusion's test (Ye et al., 2024), which judged staleness from how smoothly
# x moves: noise the sampler adds and the kink at each refresh both read as roughness.
#
# Samplers that add noise between calls (ancestral, SDE): fresh.py, built into the shallow engine.
import math

import torch

BRANCH = 0  # shallow engine = input blocks 0..BRANCH and the output blocks from deep_block(BRANCH) on
INTERVAL = 3
WARMUP = 0.4
TAIL = 0.05
QUALITY = 0.9
# Checked warmup calls before the first shallow call: fresh.py's noise share is measured at checks and averaged (EMA
# 0.5). Simulated: the last 3 of 18 warmup calls gave 0.0065 distance to the all-full image, all 18 0.0059, the last 1
# 0.0133.
CHECKS = 3
MIN_STEPS = 12


def budget(quality: float) -> float:
    # The error a stretched call may reach, as a multiple of the reference: 0 at quality 1 (DeepCache alone), infinite at
    # 0 (every call between the full shares shallow). Simulated (13 samplers, Karras and DDIM schedules, 20 to 44 steps,
    # a shallow call costing 5% of a full one), the speed over DeepCache alone came to about x1.03 at 0.9, x1.16 at 0.5,
    # x1.26 at 0.1 and x1.34 at 0.
    if quality >= 1:
        return 0.0
    if quality <= 0:
        return math.inf
    return 2 * ((1 - quality) / quality) ** 0.75


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
    def __init__(self, warmup: float = WARMUP, tail: float = TAIL, quality: float = QUALITY):
        self.warmup, self.tail = warmup, tail
        self.budget = budget(quality)
        self.index = None
        self.decision = "full"
        self.check = False
        self.last_full = None
        self.slot = 0
        self.start = 0
        self.per = 1  # UNet calls per sampler step
        # (slot, sample shape) -> (deep feature, evaluation index). Keyed by shape too: with "Negative Guidance minimum
        # sigma", every other evaluation drops the unconds, so one slot alternates between two batch shapes.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        self.rate = self.reference = None  # error per call of age: the latest check's and the first after warmup's
        self._pending = []  # (error in pinned memory, CUDA event, age) of checks not read yet

    def begin(self, index: int, total, steps):
        self.index, self.slot = index, 0
        self._collect()
        self.decision, self.check = self._decide(index, total, steps)
        if self.decision == "full":
            self.last_full = index

    def _decide(self, index, total, steps) -> tuple:
        # (decision, whether a full call is checked)
        if not total or not steps or steps < MIN_STEPS:
            return "full", False
        self.per = max(1, total // steps)
        self.start, end = round(self.warmup * total), total - max(1, round(self.tail * total))
        if index < self.start or index >= end or self.last_full is None:
            return "full", self.start - CHECKS <= index < end
        age = index - self.last_full
        if age < INTERVAL or self.budget == math.inf:
            return "shallow", False
        if self.budget > 0 and self.rate is not None and self.rate * age <= self.budget * self.reference * INTERVAL:
            return "shallow", False
        return "full", True

    def checked(self, error, origin: int):
        # origin: the call index of the deep feature checked. The error is copied to pinned memory asynchronously and
        # read once its event has completed, a few calls later at the earliest, so the CPU never waits for the GPU here.
        # Warmup's checks are for fresh.py's noise share only.
        if self.index < self.start or self.index % self.per:
            return
        age = self.index - origin
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(error.reshape(()), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending.append((host, event, age))

    def _collect(self):
        while self._pending and self._pending[0][1].query():
            host, _, age = self._pending.pop(0)
            self.rate = float(host) / age
            if self.reference is None:
                self.reference = self.rate

    def call(self, shape) -> tuple:
        # (slot, deep feature or None, index of the full call that computed it, (deep feature, index) this checked full
        # call replaces or None)
        if self.index is None:
            self.full_calls += 1
            return None, None, None, None
        slot, self.slot = self.slot, self.slot + 1
        kept = self.deep.get((slot, tuple(shape)))
        if self.decision == "shallow" and kept is not None:
            self.shallow_calls += 1
            return slot, kept[0], kept[1], None
        self.full_calls += 1
        return slot, None, None, kept if self.check and slot == 0 else None

    def keep(self, slot: int, shape, deep):
        self.deep[(slot, tuple(shape))] = (deep, self.index)

    def release(self):
        self.deep.clear()
        self._pending = []
