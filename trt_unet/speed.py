# Speed Hacks: DeepCache, with due full calls waiting while a measured error allows.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. After a full call, the next INTERVAL - 1 calls run
# only the shallow engine, on its deep feature, if their batch and prompts are the deep feature's (Schedule.deep): a
# prompt edit gets a full call, and each prompt of an alternation its own deep feature. The noise ancestral and SDE
# samplers add: fresh.py, in the shallow engine.
#
# Waiting (in place of AdaptiveDiffusion, Ye et al., 2024, whose test of how smoothly x moves reads DeepCache's own
# call-to-call unevenness, and the noise of ancestral and SDE samplers, as roughness: in the measurements below it never
# passed where it would have let a full call wait): a due full call also runs the shallow engine as the shallow call
# at its index would have, on the deep feature it replaces (a "check", the cost of a shallow call), which gives the
# error that shallow call would have made, for any sampler. Due full calls wait while drift(), the error the shallow
# calls since the deep feature's call are predicted to have made in total, stays within tolerance(quality). The
# tolerance is a fixed amount of error (relative to the denoised value's norm), not one relative to other checks, so the
# start and end shares don't change what a quality means. Checks cost a shallow call each and errors change slowly along
# a pass, so a due full call checks only once the latest check is CHECKS steps old, or 3 times that while the latest put
# waiting far out of reach (drift at age INTERVAL above FAR times the tolerance). Only a sampler step's first denoiser
# call is checked, in all of its UNet calls: Heun, DPM++ SDE, SEEDS, HeunPP2 and others call the denoiser 2-3 times per
# step, at points whose errors aren't comparable. The last full call before the first shallow one also runs the shallow
# engine (a "probe"), whose status tells fresh.Tracker whether the sampler adds noise before any shallow call runs.
#
# The start and end shares are of the steps of the sampler's schedule, and each call is placed on a step (_place) by its
# sigma and the sampler's callbacks. Counting calls instead misplaces them for the samplers reForge doesn't flag
# second_order (its total_steps counts 1 call per step for all of reForge's "alter" samplers, which make 2-3) and for
# the adaptive ones (DPM adaptive, the ODE samplers), whose error control sets the number of calls. Timestep samplers
# (DDIM, PLMS, UniPC), which make 1 call per step, are placed by call index against total_steps.
#
# Measurements behind the constants: SDXL (novaOrangeXL v1.4, 1024x1024, CFG 1, 3 and 7), Euler, Euler a, DPM++ 2M,
# 2M SDE, 3M SDE, SDE and Heun, Karras/Simple/Exponential/SGM Uniform, 20/30/40/50/70/100 steps, 3 prompts; every
# call's shallow error against the full UNet's output at the same x, on runs with DeepCache every 3rd and 6th call and
# with every call shallow between the start and end shares. The backtests replay those errors through this schedule
# with a start share of 0.4.
import bisect
import math

import torch

from .common import Readback

BRANCH = 0  # shallow engine = input blocks 0..BRANCH and the output blocks from deep_block(BRANCH) on
INTERVAL = 3
WARMUP = 0.15
TAIL = 0.05
QUALITY = 0.8
MIN_STEPS = 12
# Check spacing (see the header). Backtested: at any accumulated error, within 0.03% of or cheaper than a check at
# every due full call (up to 1% less near quality 0.9 for a shallow call costing 3-10% of a full one), with 4.5 checks
# per pass at quality 0.9 instead of 7.8.
CHECKS = 4
FAR = 2
# A kept deep feature no call has used for this many evaluations is dropped (the old prompt's, after a prompt edit).
STALE = 8
# A prompt that moved by more than this share of its norm since the deep feature's call makes the next call full.
MOVED = 0.01


# A shallow call's error grew as age ** 0.5-0.8 (deterministic samplers, more at more steps) and age ** 0.25-0.4
# (ancestral and SDE) from age 3 to 6 in the measurements. Backtested, any exponent from 0.35 to 0.8 changed the cost
# at a given accumulated error by under 0.3%.
AGE_EXPONENT = 0.5


def tolerance(quality: float) -> float:
    # The drift() a stretch may reach before a full call. Fitted on the backtests: at 30 steps, quality q skips about
    # 1 - q of the full calls between the start and end shares that quality 0 skips; at more steps, where calls change
    # less, more of them (0.9: 8% at 30 steps, 17% at 50, 27% at 100).
    if quality >= 1:
        return 0.0
    if quality <= 0:
        return math.inf
    return 1.5 * ((1 - quality) / quality) ** (2 / 3)


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
        self.warmup, self.tail, self.quality = warmup, tail, quality
        self.tolerance = tolerance(quality)
        self.index = None
        self.region = False  # this evaluation is between the start and end shares
        self.idle = False  # the pass has no shallow region (too short, or start >= end)
        self.done = False  # past the shallow region (in the tail)
        self.kind = None  # "check", "probe" or None: what this evaluation's full calls also run the shallow engine for
        self._undecided = None  # (total, steps, step, sigma) until the evaluation's first call() decides region and kind
        # The sigmas KDiffusionSampler.get_sigmas returned for this pass (runtime.py), or None (timestep samplers)
        self.sigmas = None
        self._levels = None  # the sigmas of the steps this pass samples, negated (ascending), once known
        self._probed = False
        self._checked_at = -math.inf  # evaluation index of the latest check
        # The sampler's steps, from its callback (reForge's state.sampling_step): the evaluation before each callback is
        # taken as a step's first, and the evaluations between two callbacks as the evaluations per step.
        self._step = None
        self._anchor = None  # index of the latest evaluation followed by a callback
        self.per = None
        self.slot = 0
        # (slot, key) -> [deep feature, evaluation index, the prompt it saw ([tensor, its norm once compare() needs it]
        # per prompt tensor), last evaluation index that used it]. key: (sample shape, reForge's cond_or_uncond, prompt shape, which scheduled
        # prompts the evaluation used). With "Negative Guidance minimum sigma", every other evaluation drops the
        # unconds, so one slot alternates between two batch shapes; cond_or_uncond keeps a cond row from reusing an
        # uncond row's deep feature; and the prompts give a prompt edit ([a:b:0.5]) a full call, and each prompt of an
        # alternation ([a|b]) its own deep feature.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        self.latest = None  # (error, age) of the latest check read
        self.checks = []  # (index, age, error) of the checks read
        # Checks' errors, tagged ("check", evaluation index, age), and prompt movements, tagged ("move", (slot, key), deep
        # feature's index)
        self._readback = Readback()
        # A prompt that moves again within STALE evaluations (an extension blending or alternating prompts every step)
        # is seen a call late each time, so its calls run in full for the rest of the pass.
        self._moved = {}  # (slot, key) -> evaluation index of the latest move read
        self.volatile = set()  # (slot, key)

    def begin(self, index: int, total, steps, step, sigma):
        # step: reForge's state.sampling_step; sigma: the denoiser's, a GPU tensor read in the first call()
        self.index, self.slot = index, 0
        self._collect()
        self._follow(index, step)
        if self._levels is None and self.sigmas is not None and steps and len(self.sigmas) > steps:
            self._levels = [-v for v in self.sigmas[-(steps + 1):]]  # img2img samples the schedule's last steps
        self.idle = not total or not steps or steps < MIN_STEPS
        if not self.idle:
            if self._levels is not None:
                n = len(self._levels) - 1
                start, end = round(self.warmup * n), n - max(1, round(self.tail * n))
                if start >= end:
                    self.idle = True
                elif (step + 1 if index else 0) >= end:
                    self.done = True
            else:
                per = max(1, total // steps)
                start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
                if start >= end:
                    self.idle = True
                elif index >= end:
                    self.done = True
        self.region, self.kind = False, None
        self._undecided = None if self.idle or self.done else (total, steps, step, sigma)
        if self.done:
            self.deep.clear()
        else:
            for k in [k for k, kept in self.deep.items() if index - kept[3] > STALE]:
                del self.deep[k]

    def _follow(self, index, step):
        if self._step is None:
            self._step = step
        elif step != self._step:
            self._step = step
            if self._anchor is not None and index - 1 > self._anchor:
                self.per = index - 1 - self._anchor
            self._anchor = index - 1

    def _first(self, index) -> bool:
        return self.per is not None and (index - self._anchor) % self.per == 0

    def _decide(self, index, total, steps, step, sigma) -> tuple:
        if self._levels is None and self.sigmas is not None and len(self.sigmas) > steps:
            self._levels = [-v for v in self.sigmas[-(steps + 1):]]  # img2img samples the schedule's last steps
        if self._levels is not None:
            # A sync, in the evaluation's first UNet call right before the engine runs (as fresh.Tracker.valid's)
            t = torch.as_tensor(sigma)
            region, probe = self._place(index, float(t) if t.numel() == 1 else float(t.max()), step)
        else:
            # Calls against reForge's total_steps, which counts 2 per step only for samplers it flags second_order:
            # their final call is at total - 2.
            per = max(1, total // steps)
            start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
            region, probe = start <= index < end, index == start - 1 and index >= 1
        if probe:
            return False, "probe"
        if not region:
            return False, None
        return True, "check" if 0 < self.tolerance < math.inf and self._first(index) and self._check_due(index) else None

    def _place(self, index, sigma: float, step) -> tuple:
        # (region, probe) by the call's step: the later of the last step whose sigma the call's hasn't gone below and the
        # steps the sampler's callbacks have reported. Callback i sets state.sampling_step to i after step i's first
        # call, and a later call is taken as step i + 1's, which places the rest of step i's calls (Heun's, DPM++ SDE's)
        # up to a step late. Callbacks alone miss DPM fast's calls (1 callback per 3) and the adaptive samplers'; sigma
        # alone places the calls of samplers that raise sigma before calling the model (s_churn; the Dy samplers by
        # sqrt(2)) steps early, which would leave the last steps in the region.
        n = len(self._levels) - 1
        start, end = round(self.warmup * n), n - max(1, round(self.tail * n))
        # Sigmas a sampler recomputes from log sigma (DPM++ SDE, the RES samplers) differ from the schedule's by FP32
        # rounding.
        position = max(bisect.bisect_right(self._levels, -sigma * (1 - 1e-4)) - 1, step + 1 if index else 0)
        region = start <= position < end
        # The probe is the call before the first shallow one, or the first call placed past start if none was
        # placed just before it
        if 2 <= start and not self._probed and (region or position == start - 1):
            self._probed = True
            return False, True
        return region, False

    def _check_due(self, index) -> bool:
        gap = index - self._checked_at
        if gap < CHECKS * self.per:
            return False
        far = self.latest is not None and drift(self.latest, INTERVAL) > FAR * self.tolerance
        return not far or gap >= 3 * CHECKS * self.per

    def _wait(self, age: int) -> bool:
        # Whether a call may reuse a deep feature computed age evaluations ago
        if age < INTERVAL or self.tolerance == math.inf:
            return True
        return self.tolerance > 0 and self.latest is not None and drift(self.latest, age) <= self.tolerance

    def checked(self, error, age: int):
        self._checked_at = self.index
        self._readback.send(("check", self.index, age), error)

    def _collect(self):
        for (kind, a, b), ((value,),) in self._readback.ready():
            if kind == "check":
                self.checks.append((a, b, value))
                self.latest = (value, b)
                continue
            where, origin = a, b
            if value <= MOVED:
                continue
            if self.index - self._moved.get(where, -math.inf) <= STALE:
                self.volatile.add(where)
            self._moved[where] = self.index
            kept = self.deep.get(where)
            if kept is not None and kept[1] == origin:
                del self.deep[where]

    def call(self, key: tuple) -> tuple:
        # (slot, deep feature or None, index of the call that computed it, (deep feature, index) that this checked or
        # probed full call replaces or None, "check"/"probe"/None)
        if self.index is None:
            self.full_calls += 1
            return None, None, None, None, None
        if self._undecided is not None:
            self.region, self.kind = self._decide(self.index, *self._undecided)
            self._undecided = None
        slot, self.slot = self.slot, self.slot + 1
        kept = self.deep.get((slot, key))
        if kept is None or (slot, key) in self.volatile:
            self.full_calls += 1
            return slot, None, None, None, None
        kept[3] = self.index
        if self.region and self._wait(self.index - kept[1]):
            self.shallow_calls += 1
            return slot, kept[0], kept[1], None, None
        self.full_calls += 1
        return slot, None, None, (kept[0], kept[1]) if self.kind else None, self.kind

    def compare(self, slot: int, key: tuple, prompt: tuple):
        # For a shallow call: how far the prompt (its tensors: context and SDXL's y) moved from the one its deep feature
        # saw, which callbacks can change without the scheduled prompts changing. Read like checks; a move past MOVED
        # drops the deep feature.
        kept = self.deep.get((slot, key))
        if kept is None:
            return
        moved = None
        for t, reference in zip(prompt, kept[2]):
            r = reference[0]
            if t is not None and r is not None:
                if reference[1] is None:
                    reference[1] = r.float().norm().clamp_min(1e-30)
                m = torch.dist(t.float(), r.float()) / reference[1]
                moved = m if moved is None else torch.maximum(moved, m)
        if moved is not None:
            self._readback.send(("move", (slot, key), kept[1]), moved)

    def ran_full(self):
        # A call the schedule gave a deep feature that the runtime ran in full instead
        self.shallow_calls -= 1
        self.full_calls += 1

    def keep(self, slot: int, key: tuple, deep, prompt: tuple):
        if (slot, key) not in self.volatile:
            self.deep[(slot, key)] = [deep, self.index, [[t, None] for t in prompt], self.index]

    def release(self):
        self.deep.clear()
        self._readback.clear()
