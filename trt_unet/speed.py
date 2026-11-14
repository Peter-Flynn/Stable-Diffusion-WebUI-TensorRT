# Speed Hacks: DeepCache, with due full calls waiting while a measured error allows.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. After a full call, the next INTERVAL - 1 calls run
# only the shallow engine, on its deep feature. The noise ancestral and SDE samplers add: fresh.py, in the shallow
# engine.
#
# Waiting (in place of AdaptiveDiffusion, Ye et al., 2024, whose test of how smoothly x moves reads that noise as
# roughness, so it never passed with those samplers): a due full call also runs the shallow engine as the shallow call
# at its index would have, on the deep feature it replaces (a "check", the cost of a shallow call), which gives the
# error that shallow call would have made, for any sampler. Due full calls wait while the latest check's error, scaled
# linearly to the age the deep feature would reach, stays within budget(quality) times the pass's first check's. Only
# a sampler step's first call is checked: DPM++ SDE and Heun call the UNet twice per step (reForge's total_steps /
# steps), at points whose errors aren't comparable.
import math

import torch

BRANCH = 0  # shallow engine = input blocks 0..BRANCH and the output blocks from deep_block(BRANCH) on
INTERVAL = 3
WARMUP = 0.4
TAIL = 0.05
QUALITY = 0.9
MIN_STEPS = 12


def budget(quality: float) -> float:
    # The error a call past DeepCache's interval may reach, in units of the pass's first check (the error of a call at
    # DeepCache's interval there). At 1 nothing waits. Down to 0.9 the budget rises to 1, where a waiting call is no
    # worse than that first check. Below 0.9 it's 0.9 / quality: at a steady error rate a call waits until age
    # INTERVAL * budget, so the share of full calls falls in proportion to quality, from DeepCache's 1 in INTERVAL at
    # 0.9 to none at 0.
    if quality >= 1:
        return 0.0
    if quality <= 0:
        return math.inf
    return 10 * (1 - quality) if quality >= 0.9 else 0.9 / quality


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
        # (slot, sample shape) -> (deep feature, evaluation index). Keyed by shape too: with "Negative Guidance minimum
        # sigma", every other evaluation drops the unconds, so one slot alternates between two batch shapes.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        self.rate = self.reference = None  # a check's error per call of age: the latest read, and the pass's first
        self.checks = []  # (index, age, error) of the checks read
        self._pending = []  # (index, age, error in pinned memory, CUDA event) of checks not read yet

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
        start, end = round(self.warmup * total), total - max(1, round(self.tail * total))
        if index < start or index >= end or self.last_full is None:
            return "full", False
        age = index - self.last_full
        if age < INTERVAL or self.budget == math.inf:
            return "shallow", False
        if self.budget == 0:
            return "full", False
        if self.rate is not None and self.rate * age <= self.budget * self.reference * INTERVAL:
            return "shallow", False
        return "full", index % max(1, total // steps) == 0

    def checked(self, error, age: int):
        # The error is copied to pinned memory asynchronously and read once its event has completed, a call or more
        # later, so the CPU never waits for the GPU here.
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(error.reshape(()), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending.append((self.index, age, host, event))

    def _collect(self):
        while self._pending and self._pending[0][3].query():
            index, age, host, _ = self._pending.pop(0)
            error = float(host)
            self.checks.append((index, age, error))
            self.rate = error / age
            if self.reference is None:
                self.reference = self.rate

    def call(self, shape) -> tuple:
        # (slot, deep feature or None, index of the call that computed it, (deep feature, index) this checked full call
        # replaces or None)
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
