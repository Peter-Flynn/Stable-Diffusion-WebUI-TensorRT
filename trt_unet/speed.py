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
# error that shallow call would have made, for any sampler. Due full calls wait while drift(), the error the shallow
# calls since the last full call are predicted to have made in total, stays within tolerance(quality). The tolerance is
# a fixed amount of error (relative to the denoised value's norm), not one relative to other checks, so the start and
# end shares don't change what a quality means. Only a sampler step's first call is checked: DPM++ SDE and Heun call the
# UNet twice per step (reForge's total_steps / steps), at points whose errors aren't comparable.
#
# Measurements behind the constants: SDXL (novaOrangeXL v1.4, 1024x1024, CFG 3), Euler, Euler a, DPM++ 2M, 2M SDE,
# 3M SDE, SDE and Heun, Karras/Simple/Exponential/SGM Uniform, 20 to 100 steps, 3 prompts; every call's shallow error
# against the full UNet's output at the same x, on runs with DeepCache every 3rd and 6th call and with every call
# shallow between the start and end shares.
import math

import torch

BRANCH = 0  # shallow engine = input blocks 0..BRANCH and the output blocks from deep_block(BRANCH) on
INTERVAL = 3
WARMUP = 0.4
TAIL = 0.05
QUALITY = 0.9
MIN_STEPS = 12


# A shallow call's error grew as age ** 0.42 (deterministic samplers) and age ** 0.25 (ancestral and SDE) from age 3 to
# 12 in the measurements: 0.5 slightly over-predicts both.
AGE_EXPONENT = 0.5


def tolerance(quality: float) -> float:
    # The drift() a stretch may reach before a full call. 1.9 * (1 - q) / q, fitted to the measurements: at 30 steps
    # with 10% of steps full at the start, it keeps the share of full calls between the start and end shares near
    # quality times DeepCache's (1 in INTERVAL); at more steps, where calls change less, it runs fewer (0.9 runs 5% fewer
    # full calls than 1 at 30 steps, 19% at 100). Of c * ((1 - q) / q) ** p, p = 1 fitted best.
    if quality >= 1:
        return 0.0
    if quality <= 0:
        return math.inf
    return 1.9 * (1 - quality) / quality


def drift(check: tuple, age: int) -> float:
    # check: (error, age) of the latest check
    error, checked = check
    return sum(error * (a / checked) ** AGE_EXPONENT for a in range(1, age + 1))


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
        self.tolerance = tolerance(quality)
        self.index = None
        self.decision = "full"
        self.check = False
        self.last_full = None
        self.slot = 0
        # (slot, sample shape) -> (deep feature, evaluation index). Keyed by shape too: with "Negative Guidance minimum
        # sigma", every other evaluation drops the unconds, so one slot alternates between two batch shapes.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        self.latest = None  # (error, age) of the latest check read
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
        per = max(1, total // steps)
        # Samplers that call the UNet twice per step (Heun, DPM++ SDE, DPM++ 2S a) call it once at their last step, so
        # their final call is at total - 2.
        start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
        if index < start or index >= end or self.last_full is None:
            return "full", False
        age = index - self.last_full
        if age < INTERVAL or self.tolerance == math.inf:
            return "shallow", False
        if self.tolerance == 0:
            return "full", False
        if self.latest is not None and drift(self.latest, age) <= self.tolerance:
            return "shallow", False
        return "full", index % per == 0

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
            self.latest = (error, age)

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
