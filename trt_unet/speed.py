# Speed Hacks: DeepCache, with due full calls waiting while a measured error allows.
#
# DeepCache (Ma et al., 2023): the UNet's deep features change little between steps. Every engine is built twice: the
# full UNet, which also outputs the input of output block deep_block() (the "deep" feature), and a shallow engine of
# only the outermost blocks, which takes that feature as an input. After a full call, the next INTERVAL - 1 calls run
# only the shallow engine, on its deep feature, if their batch and prompts are the deep feature's (Schedule.deep): a
# prompt edit gets a full call, and each prompt of an alternation its own deep feature. The noise ancestral and SDE
# samplers add: fresh.py, in the shallow engine.
#
# Waiting (in place of AdaptiveDiffusion, Ye et al., 2024, whose test of how smoothly x moves reads that noise as
# roughness, so it never passed with those samplers): a due full call also runs the shallow engine as the shallow call
# at its index would have, on the deep feature it replaces (a "check", the cost of a shallow call), which gives the
# error that shallow call would have made, for any sampler. Due full calls wait while drift(), the error the shallow
# calls since the deep feature's call are predicted to have made in total, stays within tolerance(quality). The
# tolerance is a fixed amount of error (relative to the denoised value's norm), not one relative to other checks, so the
# start and end shares don't change what a quality means. Only a sampler step's first denoiser call is checked, in all
# of its UNet calls: DPM++ SDE and Heun call the denoiser twice per step (reForge's total_steps / steps), at points
# whose errors aren't comparable. The last full call before the first shallow one also runs the shallow engine (a
# "probe"), whose status tells fresh.Tracker whether the sampler adds noise before any shallow call runs.
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
# A kept deep feature no call has used for this many evaluations is dropped (the old prompt's, after a prompt edit).
STALE = 8
# A prompt that moved by more than this share of its norm since the deep feature's call makes the next call full.
MOVED = 0.01


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
        self.region = False  # this evaluation is between the start and end shares
        self.kind = None  # "check", "probe" or None: what this evaluation's full calls also run the shallow engine for
        self.slot = 0
        # (slot, key) -> [deep feature, evaluation index, the prompt it saw, last evaluation index that used it]. key:
        # (sample shape, reForge's cond_or_uncond, prompt shape, which scheduled prompts the evaluation used). With
        # "Negative Guidance minimum sigma", every other evaluation drops the unconds, so one slot alternates between two
        # batch shapes; cond_or_uncond keeps a cond row from reusing an uncond row's deep feature; and the prompts give
        # a prompt edit ([a:b:0.5]) a full call, and each prompt of an alternation ([a|b]) its own deep feature.
        self.deep = {}
        self.shallow_calls = self.full_calls = 0
        self.latest = None  # (error, age) of the latest check read
        self.checks = []  # (index, age, error) of the checks read
        self._pending = []  # (index, age, error in pinned memory, CUDA event) of checks not read yet
        self._moves = []  # ((slot, key), deep feature's index, prompt movement in pinned memory, CUDA event)
        # A prompt that moves again within STALE evaluations (an extension blending or alternating prompts every step)
        # is seen a call late each time, so its calls run in full for the rest of the pass.
        self._moved = {}  # (slot, key) -> evaluation index of the latest move read
        self.volatile = set()  # (slot, key)

    def begin(self, index: int, total, steps):
        self.index, self.slot = index, 0
        self._collect()
        self.region, self.kind = self._decide(index, total, steps)
        for k in [k for k, kept in self.deep.items() if index - kept[3] > STALE]:
            del self.deep[k]

    def _decide(self, index, total, steps) -> tuple:
        if not total or not steps or steps < MIN_STEPS:
            return False, None
        per = max(1, total // steps)
        # Samplers that call the UNet twice per step (Heun, DPM++ SDE, DPM++ 2S a) call it once at their last step, so
        # their final call is at total - 2.
        start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
        if index == start - 1 and index >= 1:
            return False, "probe"
        if index < start or index >= end:
            return False, None
        return True, "check" if 0 < self.tolerance < math.inf and index % per == 0 else None

    def _wait(self, age: int) -> bool:
        # Whether a call may reuse a deep feature computed age evaluations ago
        if age < INTERVAL or self.tolerance == math.inf:
            return True
        return self.tolerance > 0 and self.latest is not None and drift(self.latest, age) <= self.tolerance

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
        while self._moves and self._moves[0][3].query():
            where, origin, host, _ = self._moves.pop(0)
            if float(host) <= MOVED:
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

    def compare(self, slot: int, key: tuple, prompt):
        # For a shallow call: how far the prompt moved from the one its deep feature saw, which callbacks can change
        # without the scheduled prompts changing. Read like checks; a move past MOVED drops the deep feature.
        kept = self.deep.get((slot, key))
        if kept is None or kept[2] is None:
            return
        reference = kept[2].float()
        moved = (prompt.float() - reference).norm() / reference.norm().clamp_min(1e-30)
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(moved, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._moves.append(((slot, key), kept[1], host, event))

    def ran_full(self):
        # A call the schedule gave a deep feature that the runtime ran in full instead
        self.shallow_calls -= 1
        self.full_calls += 1

    def keep(self, slot: int, key: tuple, deep, prompt):
        self.deep[(slot, key)] = [deep, self.index, prompt, self.index]

    def release(self):
        self.deep.clear()
        self._pending = []
        self._moves = []
