# Fresh noise for Speed Hacks. Ancestral, SDE and other noise-adding samplers add new noise to x between model calls. A
# shallow call reuses the deep feature of an earlier call, computed before that noise was drawn, and the shallow UNet
# alone attributes little of the new noise to its noise prediction: the denoised value keeps it as image content, a
# burst of latent noise at the first shallow call. The shallow engine's graph (shallow() below) removes it.
#
# For every k-diffusion sampler, the x it passes to the model is a linear combination (per-image scalar coefficients) of
# earlier calls' x and denoised values, plus the noise it draws. Here x and sigma are what reForge's apply_model got in
# the step's first UNet call, after any edits by the CFG denoiser, its callbacks and conditioning modifiers. For a
# timestep sampler such as DDIM, that is its x rescaled to k-diffusion's and the sigma of its timestep, which keeps the
# combination linear. The denoised value is what the CFG denoiser returned, after any guidance (computed from the eps a
# timestep sampler gets back), and the uncond rows' denoised value is a column of its own for CFG++ samplers: the fit
# doesn't depend on how they were computed. For the calls after the deep feature's call j, write x_i = shadow_i + N_i,
# N_i being the noise drawn since j as it stands in x_i (N_i = 0 for i <= j). A shallow call's denoised value is
# computed on its shadow and holds none of N, so x = (a combination of the last WINDOW calls' shadows and denoised
# values) + N, with N independent of all of those: least squares of x on them leaves N as the residual. A call whose N
# isn't known (it ran on another deep feature, or its fit failed) leaves its x out: x_i is itself a combination of
# earlier x and denoised values plus noise, so the fit still holds while an earlier call's x or shadow is in the window.
# Every kind of model sampling reForge has scales x as noise_scaling does, alpha(sigma) * latent + sigma * noise, with
# alpha = 1 but for flow models' CONST (1 - sigma): x - N holds noise sqrt(sigma^2 - var(N)) at the same alpha, which is
# the point sigma_eff of the schedule with that ratio of noise to alpha, after scaling x - N by alpha(sigma_eff) /
# alpha(sigma). The shallow UNet runs there, and its output is converted to the one that gives the same denoised value
# at the call's real x and sigma. The sampler's x keeps N, which the next full call sees whole.
#
# The model sampling's own timestep, calculate_input, calculate_denoised and noise_scaling (linear in their tensor
# arguments for every kind reForge has) come in as tables (see tables()), so the graph serves any of them. The fit is
# classical Gram-Schmidt applied twice (CGS2) in FP32: TensorRT has no FP64 or QR, and consecutive calls' values are
# nearly parallel, which rules out normal equations in FP32. The correction applies per image only where the graph finds
# its assumptions hold: the UNet's input rows are calculate_input(sigma, x) (not in a later UNet call of the step on
# another x), and the residual is white noise of the same variance across the image (a sampler the window doesn't cover
# leaves structure; inpainting's "mask before denoising" adds fresh noise outside the inpainted area only). Elsewhere
# the call runs the plain shallow UNet, and on a pass where noise was found Tracker.safe() has the runtime run such
# calls in full instead.
#
# Engine inputs: x and sigma per image (see above); tables (see tables()); hx/hr/hv/hd/hu<m>: the m-th previous call's
# x, N (zeros up to the deep feature's call), that call's reason per image (0 where its N is known; see below), denoised
# value and uncond denoised value (zeros if none). Outputs: latent (the UNet output rows), removed (this call's N),
# reason (per image: 0 where N was removed or is negligible, as with deterministic samplers, whose residual is FP32
# rounding and may fail the evenness test; else why not: 1 the UNet input rows aren't calculate_input(sigma, x), 2 the
# residual isn't white noise of even variance, 3 the residual exceeds sigma^2, or alpha is below LATENT), share (the
# residual's variance less FLOOR times x's mean square, / sigma^2, white or not: a residual the fit can't explain may
# hold noise).
import math
from collections import deque

import numpy
import torch

from .common import Readback

# SA-Solver's corrector uses the last 4 calls' denoised values and DPM++ 3M SDE's update the last 3; DPM++ SDE, DPM++
# 2S a and Heun step from the x of 2 calls back.
WINDOW = 4
# CFG++ samplers step with the uncond denoised value of the current call and, for DPM++ 2M CFG++ and res_multistep
# CFG++, the previous one: only the newest UNCOND calls' are columns of the fit.
UNCOND = 2
# Lag-1 autocorrelation / energy of the residual: ~1 / sqrt(pixels) for noise (under 0.01 at 512x512), near 1 for
# image structure.
WHITE = 0.1
# The residual's variance in every TILE x TILE patch of latent pixels must be within a factor UNEVEN of its mean. A
# patch of 8 x 8 pixels and 4 channels holds 256 values, whose variance has a relative standard deviation of 9%.
TILE = 8
UNEVEN = 2.0
# CGS2 drops a column with less than this share of its norm outside the earlier columns' span. FP32 rounding is ~1e-7.
DROP = 1e-4
# Most var(N) may exceed sigma^2 by. A variance over d values has relative standard deviation sqrt(2 / d), 2.2% for
# a 32x32 latent of 4 channels, and LCM's sampler redraws all of x's noise every call (var(N) = sigma^2).
CEILING = 1.1
# The UNet's input rows match calculate_input(sigma, x) to FP16 rounding, 2^-11 of the largest value.
GATE = 0.01
# A residual share of sigma^2 above which a call's x holds noise or structure the deep feature's call didn't see:
# deterministic samplers leave FP32 rounding (~1e-12), ancestral and SDE samplers 0.01 to 0.5 (SDXL measurements).
NOISY = 1e-4
# The fit's own FP32 error, as a variance relative to x's mean square, subtracted before comparing with NOISY. Measured
# up to (3.5e-5)^2 on deterministic samplers down to sigma 0.0015 (flow schedules), where consecutive calls' values are
# nearly parallel; under (2.5e-7)^2 above sigma 0.03. Ancestral noise at a share of 0.01 exceeds it above sigma 0.003
# for latents of unit mean square.
FLOOR = 1e-7
# Table points over log sigma, from sigma_min / 2 to sigma_max * 2: 0.0065 apart for a zero-terminal-SNR model
# (sigma_max 4096), where linear interpolation of the smooth columns is exact to ~1e-5.
GRID = 2048
# The most evaluations between probes while they keep finding the fit failing (Tracker.probe)
PROBES = 8
# A latent scale alpha(sigma) below which a call's noise isn't removed: a flow model's first steps, near sigma = 1
LATENT = 0.01
# Engines record this; one built with another has a different shallow graph. Raise the version when shallow() changes.
SIGNATURE = {"version": 11, "window": WINDOW, "uncond": UNCOND, "grid": GRID}

IMAGES = [f"{name}{m}" for m in range(1, WINDOW + 1) for name in ("hx", "hr", "hd", "hu")[: 4 if m <= UNCOND else 3]]
VECTORS = [f"hv{m}" for m in range(1, WINDOW + 1)]
INPUTS = ["x", "sigma", "tables", *IMAGES, *VECTORS]
OUTPUTS = ["latent", "removed", "reason", "share"]


def tables(model_sampling, device) -> torch.Tensor:
    # (8, GRID). Rows 0-5, at log sigma evenly spaced: log sigma, timestep, the coefficients of calculate_input
    # (input = c_in * x) and calculate_denoised (denoised = a * x + b * output), and alpha (see the header). Rows 6-7, at
    # log(sigma / alpha) evenly spaced over the sigmas where alpha >= LATENT: log(sigma / alpha), and sigma.
    lo, hi = float(model_sampling.sigma_min), float(model_sampling.sigma_max)
    grid = torch.linspace(math.log(lo / 2), math.log(hi * 2), GRID, dtype=torch.float64)
    sigma = grid.exp().float().to(device)
    ones, zeros = torch.ones(GRID, 1, device=device), torch.zeros(GRID, 1, device=device)
    columns = [
        model_sampling.timestep(sigma),
        model_sampling.calculate_input(sigma, ones),
        model_sampling.calculate_denoised(sigma, zeros, ones),
        model_sampling.calculate_denoised(sigma, ones, zeros),
        model_sampling.noise_scaling(sigma, zeros.clone(), ones) if hasattr(model_sampling, "noise_scaling") else ones,
    ]
    columns = [c.reshape(-1).double().cpu() for c in columns]
    alpha = columns[-1]
    valid = alpha >= LATENT
    ratio = grid[valid] - alpha[valid].log()
    if not bool((ratio[1:] > ratio[:-1]).all()):
        raise RuntimeError(f"{type(model_sampling).__name__}'s noise to latent ratio doesn't rise with sigma")
    inverse = torch.linspace(float(ratio[0]), float(ratio[-1]), GRID, dtype=torch.float64)
    at = torch.from_numpy(numpy.interp(inverse.numpy(), ratio.numpy(), grid[valid].exp().numpy()))
    return torch.stack([grid, *columns, inverse, at]).float().to(device)


def _lookup(grid, rows, value):
    # rows at each value, linear in log value between the points of grid (log values, evenly spaced)
    position = ((value.log() - grid[0]) / (grid[1] - grid[0])).clamp(0, GRID - 1.001)
    index = position.floor()
    weight = position - index
    index = index.long()
    lo, hi = rows.index_select(1, index), rows.index_select(1, index + 1)
    return lo + (hi - lo) * weight


def _view(v):
    return v.reshape(-1, 1, 1, 1)


def _residual(y, columns):
    n = y.shape[0]
    basis = None
    for column in columns:
        a = column.reshape(n, -1, 1)
        v = a
        if basis is not None:
            for _ in range(2):
                v = v - basis @ (basis.transpose(1, 2) @ v)
        length = (v * v).sum(1, keepdim=True).sqrt()
        q = torch.where(length > DROP * (a * a).sum(1, keepdim=True).sqrt(), v / length.clamp_min(1e-30), torch.zeros_like(v))
        basis = q if basis is None else torch.cat([basis, q], 2)
    r = y.reshape(n, -1, 1)
    for _ in range(2):
        r = r - basis @ (basis.transpose(1, 2) @ r)
    return r.reshape(y.shape)


def _white(r):
    squares = r * r
    energy = squares.sum((1, 2, 3))
    lag = torch.maximum((r[:, :, 1:] * r[:, :, :-1]).sum((1, 2, 3)).abs(), (r[..., 1:] * r[..., :-1]).sum((1, 2, 3)).abs())
    patches = torch.nn.functional.avg_pool2d(squares.mean(1, keepdim=True), TILE).flatten(1)
    mean = patches.mean(1)
    even = (patches.amax(1) < UNEVEN * mean) & (patches.amin(1) > mean / UNEVEN)
    return (lag < WHITE * energy) & even


# run(sample, timesteps) returns the shallow UNet's output rows. sample's rows are blocks of x's images (reForge's
# calc_cond_batch puts each cond's copy of the batch after the other); its first x.shape[1] channels are the latent and
# any further ones the model's concat conditioning (inpainting and edit models). inputs: INPUTS by name.
def shallow(run, sample, timesteps, inputs: dict):
    x, sigma, tables = inputs["x"], inputs["sigma"], inputs["tables"]
    n, c = x.shape[0], x.shape[1]
    _, c_in, a, b, alpha = _lookup(tables[0], tables[1:6], sigma)
    blocks = sample.reshape(-1, n, *sample.shape[1:])
    scaled = _view(c_in) * x
    gap = (blocks[:, :, :c].float() - scaled.unsqueeze(0)).abs().amax((0, 2, 3, 4))
    consistent = gap <= GATE * scaled.abs().amax((1, 2, 3))
    columns = []
    for m in range(1, WINDOW + 1):
        shadow = inputs[f"hx{m}"] - inputs[f"hr{m}"]
        columns += [torch.where(_view(inputs[f"hv{m}"] < 0.5), shadow, torch.zeros_like(shadow)), inputs[f"hd{m}"]]
        if m <= UNCOND:
            columns.append(inputs[f"hu{m}"])
    noise = _residual(x, columns)
    variance = (noise * noise).mean((1, 2, 3))
    excess = (variance - FLOOR * (x * x).mean((1, 2, 3))).clamp_min(0)
    s2 = sigma * sigma
    white = _white(noise)
    below = (variance < CEILING * s2) & (alpha >= LATENT)
    share = excess / s2
    # A negligible residual (deterministic samplers) runs the plain shallow UNet, at the sampler's own timestep
    negligible = share <= NOISY
    apply = consistent & white & below & ~negligible
    # When N is nearly all of x's noise (a long wait with an ancestral sampler), x - N is close to the previous
    # denoised value at almost no noise, which the shallow UNet returns nearly as is.
    left = (s2 - variance).clamp_min(1e-6 * s2).sqrt()
    sigma_eff = _lookup(tables[6], tables[7:8], left / alpha.clamp_min(LATENT))[0]
    t_eff, c_eff, a_eff, b_eff, alpha_eff = _lookup(tables[0], tables[1:6], sigma_eff)
    x_eff = (x - noise) * _view(alpha_eff / alpha.clamp_min(LATENT))
    lanes = apply.reshape(1, -1, 1, 1, 1)
    latent = torch.where(lanes, (_view(c_eff) * x_eff).to(sample.dtype).unsqueeze(0), blocks[:, :, :c])
    if sample.shape[1] > c:
        latent = torch.cat([latent, blocks[:, :, c:]], 2)
    out = run(
        latent.reshape(sample.shape),
        torch.where(apply.reshape(1, -1), t_eff.to(timesteps.dtype).reshape(1, -1), timesteps.reshape(-1, n)).reshape(timesteps.shape),
    )
    rows = out.reshape(-1, n, *out.shape[1:])
    fixed = (_view(a_eff) * x_eff - _view(a) * x + _view(b_eff) * rows.float()) / _view(b)
    latent = torch.where(lanes, fixed.to(out.dtype), rows).reshape(out.shape)
    removed = torch.where(_view(apply), noise, torch.zeros_like(noise))
    one = torch.ones_like(variance)
    known = apply | negligible
    reason = torch.where(known, 0 * one, torch.where(~consistent, one, torch.where(~white, 2 * one, 3 * one)))
    return latent, removed, reason, share


def dynamic_axes() -> dict:
    image, vector = {0: "images", 2: "height", 3: "width"}, {0: "images"}
    axes = {"x": image, "sigma": vector, "removed": image, "reason": vector, "share": vector}
    axes.update({name: image for name in IMAGES})
    axes.update({name: vector for name in VECTORS})
    return axes


def example_inputs(sample, channels: int) -> list:
    rows, _, h, w = sample.shape
    n = max(1, rows // 2)
    kw = {"device": sample.device}
    grid = torch.linspace(-4, 9, GRID, **kw)
    values = {"x": torch.randn(n, channels, h, w, **kw), "sigma": torch.rand(n, **kw) * 10 + 0.1}
    values["tables"] = torch.stack([grid, grid * 100, torch.rsqrt(grid.exp() ** 2 + 1), torch.ones_like(grid), -grid.exp(), torch.ones_like(grid), grid, grid.exp()])
    values.update({name: torch.randn(n, channels, h, w, **kw) for name in IMAGES})
    values.update({name: torch.zeros(n, **kw) for name in VECTORS})
    return [values[name] for name in INPUTS]


def input_shapes(unet_shapes: dict, channels: int) -> dict:
    # Images range from 1 to the profile's most UNet rows: a sampler without uncond sends one row per image.
    (r_lo, _, h_lo, w_lo), (r_opt, _, h_opt, w_opt), (r_hi, _, h_hi, w_hi) = unet_shapes["sample"]
    i_opt = max(1, r_opt // 2)
    image = [[1, channels, h_lo, w_lo], [i_opt, channels, h_opt, w_opt], [r_hi, channels, h_hi, w_hi]]
    vector = [[1], [i_opt], [r_hi]]
    shapes = {"x": image, "sigma": vector, "tables": [[8, GRID]] * 3}
    shapes.update({name: image for name in IMAGES})
    shapes.update({name: vector for name in VECTORS})
    return shapes


class Entry:
    # One CFG denoiser call: the shape of its images, x and sigma (see the header), the denoised value, the uncond rows'
    # denoised value, and for a step that ran shallow its N, reasons and the index of the deep feature's call.
    __slots__ = ("index", "shape", "x", "sigma", "denoised", "uncond", "removed", "reason", "origin")

    def __init__(self, index, shape):
        self.index, self.shape = index, tuple(shape)
        self.x = self.sigma = self.denoised = self.uncond = self.removed = self.reason = self.origin = None


class Tracker:
    # One sampling pass's state between Speed Hacks engine calls. Calls are numbered by speed.Schedule's index, as deep
    # features' origins are. Statuses (reason, share) of shallow-graph runs are read back without a sync, a call or more
    # later.
    def __init__(self, model_sampling, device, channels: int):
        self.sampling = model_sampling
        self.device = device
        self.channels = channels
        self.tables = tables(model_sampling, device)
        self.history = deque(maxlen=WINDOW)  # complete Entries, newest last
        self.current = None  # the Entry of the denoiser call in progress
        self.noisy = False  # a fit in this pass left a residual above NOISY
        self.read = False  # a status has been read
        self.blocked = False  # the latest status read had noise left in on some image
        self.check = None  # (Entry, age, [full output rows], [checked rows]) until the denoiser's output is in
        self._statuses = Readback()  # (reasons, shares) of shallow-graph runs
        self._probe_at, self._probe_gap = 0, 1
        self._constants = {}

    def _constant(self, shape, value=0.0, dtype=torch.float32):
        key = (tuple(shape), value, dtype)
        if key not in self._constants:
            self._constants[key] = torch.full(shape, value, dtype=dtype, device=self.device)
        return self._constants[key]

    def begin(self, index: int, x):
        self._collect()
        newest = self.history[-1] if self.history else None
        if newest is not None and (newest.index != index - 1 or newest.shape != tuple(x.shape)):
            self.history.clear()
        self.current = Entry(index, x.shape)

    def see(self, x, sigma):
        # apply_model's arguments, for every UNet call. The step's first full-size call's are copied: a sampler may write
        # to its x in place.
        entry = self.current
        if entry is None or entry.x is not None or x.shape[0] % entry.shape[0] or tuple(x.shape[1:]) != entry.shape[1:]:
            return
        n = entry.shape[0]
        entry.x = x[:n].to(torch.float32, memory_format=torch.contiguous_format, copy=True)
        sigma = sigma.reshape(-1).to(torch.float32)
        entry.sigma = (sigma[:n] if sigma.shape[0] >= n else sigma.expand(n)).clone(memory_format=torch.contiguous_format)

    def call(self, sample, rows: int):
        # The Entry a UNet call belongs to, if its rows are blocks of the denoiser's images: not crops, as sliding window
        # guidance and area conds send. rows: the real ones; a batch padded to a profile's minimum repeats them, which
        # keeps the padded rows in blocks too.
        entry = self.current
        if entry is None or entry.x is None:
            return None
        padded, channels, h, w = sample.shape
        n, c = entry.shape[:2]
        if rows % n or padded % n or channels < c or (h, w) != entry.shape[2:]:
            return None
        return entry

    def safe(self, entry, origin) -> bool:
        # Whether a shallow call may run. One the fit can't follow (entry None) only once a status showed the pass adds no
        # noise. A tracked one always on a pass without noise (before the first status, its graph removes what noise it
        # finds), and with noise while the window holds the calls the fit needs and no status says it fails.
        self._collect()
        if entry is None:
            return self.read and not self.noisy
        if not self.noisy:
            return True
        newest = self.history[-1] if self.history else None
        if self.blocked or newest is None or newest.index != entry.index - 1:
            return False
        return any(p.index <= origin or p.origin == origin for p in self.history)

    def probe(self, index: int) -> bool:
        # Whether a call that runs in full because safe() refused also runs the shallow engine, whose status can show the
        # fit working again. While statuses keep showing it failing, each probe waits twice as long as the last (up to
        # PROBES evaluations), so a pass the fit can't follow costs little more than one without Speed Hacks.
        if index < self._probe_at:
            return False
        self._probe_at, self._probe_gap = index + self._probe_gap, min(2 * self._probe_gap, PROBES)
        return True

    def inputs(self, entry, origin, sample) -> dict:
        if entry is None:
            return self._untracked(sample)
        n, c, h, w = entry.shape
        values = {"x": entry.x, "sigma": entry.sigma, "tables": self.tables}
        zeros, known, unknown = self._constant((n, c, h, w)), self._constant((n,)), self._constant((n,), 1.0)
        past = list(self.history)[::-1]
        for m in range(1, WINDOW + 1):
            p = past[m - 1] if m <= len(past) else None
            if p is None:
                values.update({f"hx{m}": zeros, f"hr{m}": zeros, f"hv{m}": unknown, f"hd{m}": zeros})
            elif p.index <= origin:
                values.update({f"hx{m}": p.x, f"hr{m}": zeros, f"hv{m}": known, f"hd{m}": p.denoised})
            elif p.origin == origin and p.removed is not None:
                values.update({f"hx{m}": p.x, f"hr{m}": p.removed, f"hv{m}": p.reason, f"hd{m}": p.denoised})
            else:
                values.update({f"hx{m}": p.x, f"hr{m}": zeros, f"hv{m}": unknown, f"hd{m}": p.denoised})
            if m <= UNCOND:
                values[f"hu{m}"] = p.uncond if p is not None and p.uncond is not None else zeros
        return values

    def _untracked(self, sample) -> dict:
        # One image of zeros, which every UNet row is a block of: the rows fail the gate, so shallow() returns the plain
        # shallow UNet's output, after fitting that one image rather than one per row
        _, _, h, w = sample.shape
        image = self._constant((1, self.channels, h, w))
        values = {"x": image, "sigma": self._constant((1,), 1.0), "tables": self.tables}
        values.update({name: image for name in IMAGES})
        values.update({name: self._constant((1,), 1.0) for name in VECTORS})
        return values

    def after_call(self, entry, out, cond_or_uncond):
        # Every tracked UNet call: the uncond rows' denoised value, which CFG++ samplers step with
        n = entry.shape[0]
        if entry.uncond is None and cond_or_uncond and 1 in cond_or_uncond and len(cond_or_uncond) * n == out.shape[0]:
            i = cond_or_uncond.index(1)
            entry.uncond = self.sampling.calculate_denoised(entry.sigma, out[i * n:(i + 1) * n].float(), entry.x)

    def after_shallow(self, entry, outputs: dict, origin):
        if entry.removed is None:
            entry.removed, entry.reason, entry.origin = outputs["removed"], outputs["reason"], origin
        self.status(outputs)

    def status(self, outputs: dict):
        self._statuses.send(None, outputs["reason"], outputs["share"])

    def _collect(self):
        for _, (reasons, shares) in self._statuses.ready():
            self.read = True
            self.noisy = self.noisy or any(s > NOISY for s in shares)
            self.blocked = any(r > 0 for r in reasons)
            if not self.blocked:
                self._probe_at, self._probe_gap = 0, 1

    def add_check(self, entry, full, checked, age: int):
        if self.check is None or self.check[0] is not entry:
            self.check = (entry, age, [], [])
        self.check[2].append(full)
        self.check[3].append(checked)

    def finish(self, output, x=None, sigma=None, eps: bool = False):
        # After the denoiser returned its output (eps for a timestep sampler) for x and sigma, its arguments: completes the
        # step's Entry, and returns a pending check's (error, age) or None
        entry, self.current = self.current, None
        check, self.check = self.check, None
        if entry is None:
            return None
        if tuple(output.shape) != entry.shape or (entry.x is None and (eps or x is None)):
            self.history.clear()
            return None
        if entry.x is None:
            # No UNet call of the step was on the whole image (tiled diffusion sends tiles): the denoiser's own x and
            # sigma, and the fit in PyTorch until a status says whether the pass adds noise
            entry.x = x.to(torch.float32, memory_format=torch.contiguous_format, copy=True)
            entry.sigma = sigma.reshape(-1).to(torch.float32).expand(entry.shape[0]).clone(memory_format=torch.contiguous_format)
            if not self.read and not self._statuses.pending:
                self._measure(entry)
        if eps:
            entry.denoised = entry.x - entry.sigma.reshape(-1, 1, 1, 1) * output.float()
        else:
            # Copied: LCM's sampler adds its noise to the denoised value in place
            entry.denoised = output.to(torch.float32, memory_format=torch.contiguous_format, copy=True)
        self.history.append(entry)
        if check is None or check[0] is not entry:
            return None
        return self._error(entry, torch.cat(check[2]), torch.cat(check[3])), check[1]

    def _measure(self, entry):
        columns = []
        for m, p in enumerate(reversed(self.history), 1):
            columns += [p.x, p.denoised] + ([p.uncond] if p.uncond is not None and m <= UNCOND else [])
        if not columns:
            return
        noise = _residual(entry.x, columns)
        excess = ((noise * noise).mean((1, 2, 3)) - FLOOR * (entry.x * entry.x).mean((1, 2, 3))).clamp_min(0)
        share = excess / entry.sigma ** 2
        reason = torch.where(_white(noise) | (share <= NOISY), 0.0, 2.0)
        self.status({"reason": reason, "share": share})

    def _error(self, entry, full, checked):
        # A check's error in the denoiser's output, relative to it. How the output depends on each block of UNet rows
        # comes from a least-squares fit of the output on the blocks' denoised values, per image: CFG's 1 - s and s, and
        # for other guidance (APG, sliding window guidance) its local weights. linalg.solve_ex skips the error check, so
        # nothing here waits for the GPU.
        n = entry.shape[0]
        k = full.shape[0] // n
        sigma, x = entry.sigma.repeat(k), entry.x.repeat(k, 1, 1, 1)
        blocks = self.sampling.calculate_denoised(sigma, full.float(), x).reshape(k, n, -1).permute(1, 2, 0).double()
        changed = self.sampling.calculate_denoised(sigma, checked.float(), x).reshape(k, n, -1).permute(1, 2, 0).double()
        output = entry.denoised.reshape(n, -1, 1).double()
        gram = blocks.transpose(1, 2) @ blocks
        ridge = 1e-12 * gram.diagonal(dim1=1, dim2=2).sum(1).reshape(n, 1, 1) * torch.eye(k, dtype=gram.dtype, device=gram.device)
        weights = torch.linalg.solve_ex(gram + ridge, blocks.transpose(1, 2) @ output)[0]
        error = (changed - blocks) @ weights
        return (error.norm() / output.norm().clamp_min(1e-30)).float()
