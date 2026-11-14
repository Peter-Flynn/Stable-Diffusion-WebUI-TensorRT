# Speed Hacks: DeepCache with its refresh interval stretched while a measured error allows.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. Most evaluations run only the shallow engine, on
# the deep feature from the last full evaluation: INTERVAL - 1 in a row, more while checks allow.
#
# Checks: a full call from CHECKS calls before the first shallow call until the full steps at the end also runs the
# shallow engine as the shallow call at its index would have run, on the deep feature it replaces, and compares
# (fresh.py). That is the error a shallow call of that age actually makes, for any sampler, noise included; it costs a
# shallow call, a few percent of a full one on SDXL. A due full call waits while the error predicted at the age the deep
# feature would reach stays within budget(quality) times that of the first two checks after warmup (the error
# DeepCache's own interval made there), and while fresh.py can still separate the noise drawn since the deep feature's
# call (an ancestral or SDE sampler replaces all of x's noise eventually). The prediction is the latest two checks'
# error per call of age, times the age, times (age / the latest check's age)^(EXTRAPOLATION - 1) past that age. This
# replaces AdaptiveDiffusion's test (Ye et al., 2024), which judged staleness from how smoothly x moves: noise the
# sampler adds and the kink at each refresh both read as roughness.
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
# Errors grow faster than linearly past the ages checks have measured: simulated (ancestral and SDE samplers, 30 Karras
# steps) roughly as age^2.3 from age 3 to 7.
EXTRAPOLATION = 2
MIN_STEPS = 12


def budget(quality: float) -> float:
    # The error a stretched call may reach, as a multiple of the reference: 0 at quality 1, infinite at 0. The distance
    # to the all-full image grew about 0.6 times DeepCache's own per doubling of the budget in simulation, so the budget
    # is log-linear in quality: 0.5 at the default 0.9, 64 at 0.1. Simulated (13 samplers, Karras and DDIM schedules,
    # 20 to 44 steps), the distance exceeded DeepCache alone's by +5% on average at 0.9, +1.2x at 0.5, +2.3x at 0.1 and
    # +6x at 0.
    if quality >= 1:
        return 0.0
    if quality <= 0:
        return math.inf
    return 5.66 * ((1 - quality) / quality) ** 1.1


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
        # (slot, sample shape) -> (deep feature, evaluation index). Keyed by shape too: with "Negative Guidance minimum
        # sigma", every other evaluation drops the unconds, so one slot alternates between two batch shapes.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        # Error per call of age of each check after warmup, in order, and the latest one's age. Averaging pairs evens out
        # samplers whose calls alternate in kind (DPM++ SDE's: checks' errors alternated 1.09, 0.02 in simulation).
        self.rates, self.verified = [], 0
        self.lost = False  # fresh.py flagged the noise since the current deep feature's call as no longer separable
        # (value in pinned memory, CUDA event, age of a check's error or None for a shallow call's flag, origin, index)
        self._pending = []
        # For the debug log: per call F (checked full), f (full), s (shallow), S (stretched shallow); (index, age, error)
        # of each check after warmup.
        self.trace, self.checks = [], []

    def begin(self, index: int, total, steps):
        self.index, self.slot = index, 0
        self._collect()
        self.decision, self.check = self._decide(index, total, steps)
        if self.decision == "full":
            self.trace.append("F" if self.check else "f")
            self.last_full, self.lost = index, False
        else:
            self.trace.append("s" if index - self.last_full < INTERVAL else "S")

    def _decide(self, index, total, steps) -> tuple:
        # (decision, whether a full call is checked)
        if not total or not steps or steps < MIN_STEPS:
            return "full", False
        self.start, end = round(self.warmup * total), total - max(1, round(self.tail * total))
        if index < self.start or index >= end or self.last_full is None:
            return "full", self.start - CHECKS <= index < end
        age = index - self.last_full
        if age < INTERVAL:
            return "shallow", False
        if self.lost:
            return "full", True
        if self.budget == math.inf:
            return "shallow", False
        # A stretch ends in a checked full call, not in the full steps at the end: those aren't checked, and a stretch
        # running into them went unmeasured. Simulated (DDIM, 20 steps, quality 0.6): DPM++ 2M SDE's distance to the
        # all-full image grew 10.2x running into them, 3M SDE's 7.4x; ending one call earlier, -0.3x and 0.5x.
        if self.budget > 0 and len(self.rates) >= 2 and index + 1 < end:
            rate, reference = self.rates[-2] + self.rates[-1], self.rates[0] + self.rates[1]
            if rate * age * max(1.0, age / self.verified) ** (EXTRAPOLATION - 1) <= self.budget * reference * INTERVAL:
                return "shallow", False
        return "full", True

    def checked(self, error, age: int):
        if self.index >= self.start:  # warmup's checks are for fresh.py's noise share only
            self._send(error, age)

    def shallow_done(self, unknown, origin: int):
        # unknown: the shallow call's per-image flags (fresh.py)
        self._send(unknown.amax(), None, origin)

    def _send(self, value, age, origin=None):
        # Copied to pinned memory asynchronously and read once its event has completed (a call later at the earliest),
        # so the CPU never waits for the GPU here; release() waits for the rest at the end of the pass.
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(value.reshape(()), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending.append((host, event, age, origin, self.index))

    def _collect(self, wait=False):
        while self._pending and (wait or self._pending[0][1].query()):
            host, event, age, origin, index = self._pending.pop(0)
            if wait:
                event.synchronize()
            if age is None:
                self.lost = self.lost or (origin == self.last_full and float(host) > 0.5)
            else:
                self.rates.append(float(host) / age)
                self.verified = age
                self.checks.append((index, age, float(host)))

    def summary(self) -> str:
        checks = ", ".join(f"{i}:{a}:{e:.4f}" for i, a, e in self.checks)
        return f"calls {''.join(self.trace)}; checks (call:age:error) {checks or 'none'}"

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
        self._collect(wait=True)
