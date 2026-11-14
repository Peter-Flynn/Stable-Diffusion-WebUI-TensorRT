# Fresh noise for Speed Hacks. Ancestral and SDE samplers add new noise to x between model calls. The shallow engine
# reuses a deep feature computed from x before that noise was drawn, so its eps holds the old noise pattern, and
# denoised = x - sigma * eps carries the new noise as if it were image content: a burst of latent noise at the first
# shallow step, larger the higher sigma is there. Everything below is one graph, in the Speed Hacks (shallow) engine.
#
# Finding the noise: a k-diffusion-style sampler's x is a linear combination of the last few calls' x and denoised
# values, plus whatever noise it drew since the last call. Per image, x is fitted by least squares on WINDOW previous
# calls' x and the denoised values of their first and last block of UNet rows (cond and uncond); the residual is that
# noise. The fit is classical Gram-Schmidt applied twice (CGS2) in FP32: the columns are nearly dependent (condition
# numbers ~1e8, simulated), which rules out normal equations. A residual that isn't white is structure outside the
# fit's span: the call is flagged "structured", and shallow calls whose deep feature predates it get the plain shallow
# UNet's output.
#
# Shallow call: the shallow engine reuses the deep feature of an earlier full call (speed.Schedule keeps one per UNet
# batch shape). x minus the noise drawn since that call (_remove) is where the sampler would be without it, at noise
# level sigma_eff = sqrt(sigma^2 - removed variance). The shallow UNet runs there, at the timestep
# model_sampling.timestep() gives sigma_eff (nearest in reForge's log-sigma table). The share of the removed noise that
# a full UNet's denoised output would contain is added back to the shallow UNet's denoised value, per radial frequency
# band (DFTs as matrix products: TensorRT has no FFT). The output is the eps that gives that denoised value at the
# call's real x and sigma.
#
# _remove: x is built from the last WINDOW calls' x (the fit), so its noise since the deep feature's call is this call's
# noise plus a combination of theirs, which each of those calls removed and output. That combination is fitted jointly.
# One total carried from the previous call isn't enough: DPM++ SDE builds x from the x before its substep, whose noise
# never reaches the next call. Simulated (30 Karras steps, every full step skipped that could be): DPM++ SDE ended 1.02
# from the all-full image with one total, 0.17 with the joint fit.
#
# Check (a full call that speed.Schedule checks): the graph runs as the shallow call at that index would have, on the
# deep feature that call replaces, and is also given the full call's output. It outputs the relative error of the
# denoised values the shallow output gives, and measures the noise share: regressing the full call's denoised rows on
# the noise drawn just before it, band by band, gives the share S of that noise in them (unbiased: the noise is
# independent of everything else in them). S is stored as signal power P = S sigma^2 / (1 - S), the Wiener relation,
# averaged over checks, and converted back to S = P / (P + sigma^2) at a shallow call's sigma.
#
# "Simulated" figures below: transcriptions of reForge's sampler update rules (Euler a, Euler a CFG++, DPM++ 2M SDE,
# DPM++ 3M SDE, DPM++ SDE, DPM++ 2S a, Euler with churn, DPM++ 2M, Heun) run against an exact Wiener denoiser,
# 1x4x128x128 latents, 20 Karras steps, CFG with cond and uncond rows.
#
# Engine inputs, per image (B = UNet rows / row blocks): x and sigma as reForge's apply_model received them (FP32),
# hx/hs/he/hr<m> the m-th previous call's x, sigma, UNet output rows and removed total (zeros unless that call shared
# this call's deep feature), power (B, BINS + 1): P per band and the number of measurements, broken: 1 where the noise
# since the deep feature's call is unknown, measure: 1 at a check, out: the full call's output rows at a check (else
# zeros).
import math
from collections import deque

import torch

WINDOW = 5  # iPNDM's 4th-order steps use the last 4 calls' x and denoised values; one more as a margin
BINS = 48  # simulated: 24 bands overshot the high frequencies of steep spectra, 48 didn't
FLOOR = 1e-4  # residual / |x|: deterministic samplers' (FP32 rounding) simulated at ~3e-7, fresh noise's above 1e-2
WHITE = 0.1  # lag-1 autocorrelation / energy of the residual: ~1 / sqrt(pixels) for noise, near 1 for image structure
# CGS2 drops a column with less than this share of its norm outside the other columns' span; 1e-6 to 1e-3 simulated
# alike.
DROP = 1e-4
# Removing at most this share of sigma^2 keeps sigma_eff above a tenth of sigma. LCM's sampler redraws all of x's
# noise every call, so its share reaches 1. 0.95 left some ancestral and SDE calls uncorrected in simulation.
RENOISE = 0.99
EMA = 0.5  # weight of earlier measurements in P; simulated slightly better than 0 (latest measurement only)
SHARE_MAX = 0.999  # P = S sigma^2 / (1 - S) is infinite at S = 1
# Engines record this; a graph built with another signature takes other inputs. "inputs" changes with the graph's
# inputs or outputs.
SIGNATURE = {"inputs": 3, "window": WINDOW, "bins": BINS}

HISTORY = [name for m in range(1, WINDOW + 1) for name in (f"hx{m}", f"hs{m}", f"he{m}", f"hr{m}")]
SHALLOW_INPUTS = ["x", "sigma", "log_sigmas", "pred", *HISTORY, "power", "broken", "measure", "out"]
# removed / unknown: this call's total and flags, for the next shallow calls. error: the relative error of the denoised
# values latent gives, against out's.
SHALLOW_OUTPUTS = ["latent", "removed", "unknown", "power_out", "error"]


def coefficients(sigma, pred):
    # reForge's calculate_denoised as a * x + b * output: EPS (pred 0) and V_PREDICTION (pred 1), sigma_data 1
    s2 = sigma * sigma
    v = pred > 0.5
    a = torch.where(v, 1 / (s2 + 1), torch.ones_like(sigma))
    b = torch.where(v, -sigma * torch.rsqrt(s2 + 1), -sigma)
    return a.reshape(-1, 1, 1, 1), b.reshape(-1, 1, 1, 1)


def _blocks(x, rows):  # (blocks * B, ...) -> (blocks, B, ...)
    return rows.reshape(-1, *x.shape)


def _ends(x, sigma, pred, out):
    a, b = coefficients(sigma, pred)
    d = a * x + b * _blocks(x, out.float())
    return d[0], d[-1:].reshape(x.shape)


def _residual(x, columns):
    n = x.shape[0]
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
    y = x.reshape(n, -1, 1)
    for _ in range(2):
        y = y - basis @ (basis.transpose(1, 2) @ y)
    return y.reshape(x.shape)


def _history_residual(x, pred, history):
    return _residual(x, [t for hx, hs, he in history for t in (hx, *_ends(hx, hs, pred, he))])


def _noise(x, r):  # (r where it's fresh noise, else zeros; "structured" flag as 1.0 or 0.0)
    energy = (r * r).sum((1, 2, 3))
    lag = torch.maximum((r[:, :, 1:] * r[:, :, :-1]).sum((1, 2, 3)).abs(), (r[..., 1:] * r[..., :-1]).sum((1, 2, 3)).abs())
    significant = energy > FLOOR * FLOOR * (x * x).sum((1, 2, 3))
    white = lag < WHITE * energy
    noise = torch.where((significant & white).reshape(-1, 1, 1, 1), r, torch.zeros_like(r))
    return noise, (significant & ~white).float()


def _dft(n, like):
    k = torch.arange(n, dtype=torch.float32, device=like.device)
    kk = k[:, None] * k[None, :]
    angle = (kk - n * torch.floor(kk / n)) * (2 * math.pi / n)  # k * j mod n first: exact in FP32 below 2^24
    return torch.cos(angle), torch.sin(angle)


def _dft2(rows, ch, sh, cw, sw):
    a, b = rows @ cw, rows @ sw
    return ch @ a - sh @ b, -(sh @ a + ch @ b)


def _idft2(re, im, ch, sh, cw, sw):
    u, v = re @ cw - im @ sw, re @ sw + im @ cw
    return (ch @ u - sh @ v) / (ch.shape[0] * cw.shape[0])


def _bands(h, w, like):  # (h * w, BINS): one-hot radial band of each DFT frequency
    def freq(n):
        k = torch.arange(n, dtype=torch.float32, device=like.device)
        return torch.minimum(k, n - k) / n

    radius = (freq(h)[:, None] ** 2 + freq(w)[None, :] ** 2).sqrt()
    band = torch.clamp(torch.floor(radius * (BINS / math.sqrt(0.5))), max=BINS - 1)
    return (band.reshape(-1, 1) == torch.arange(BINS, dtype=torch.float32, device=like.device)).float()


def _measure(re, im, noise, sigma, power, onehot):  # re, im: spectra of the denoised first and last rows, the noise
    n = noise.shape[0]
    cross = ((re[0] + re[1]) * re[2] + (im[0] + im[1]) * im[2]).sum(1).reshape(n, -1) @ onehot
    energy = (2 * (re[2] * re[2] + im[2] * im[2])).sum(1).reshape(n, -1) @ onehot
    share = (cross / energy.clamp_min(1e-30)).clamp(max=SHARE_MAX)
    measured = share * (sigma * sigma).reshape(-1, 1) / (1 - share)
    power_, count = power[:, :BINS], power[:, BINS:]
    averaged = torch.where(count > 0, EMA * power_ + (1 - EMA) * measured, measured)
    drawn = ((noise * noise).sum((1, 2, 3)) > 0).reshape(-1, 1)
    return torch.cat([torch.where(drawn, averaged, power_), count + drawn.float()], 1)


def _remove(re, im, noise, totals, sigma, power, onehot, dft):
    # re, im: spectra of x, this call's noise and the totals. This call's noise is in x exactly once (x = the fit + it).
    # The totals' combination in x is its projection onto them by generalized least squares, each DFT frequency
    # weighted by 1 / (P + sigma^2): image content is uncorrelated with the noise and adds to the estimate's error with
    # spectrum P. Simulated: up to a third less distance to the all-full image than plain inner products. Returns (the
    # noise's total in x, the share of that total a full UNet's denoised output would contain).
    n, c, h, w = noise.shape
    s2 = (sigma * sigma).reshape(-1, 1)
    power_ = power[:, :BINS].clamp_min(0)
    measured = power[:, BINS:] > 0
    weight = torch.where(measured, s2 / (power_ + s2), torch.ones_like(power_))
    keep = torch.where(measured, power_ / (power_ + s2), torch.zeros_like(power_))
    weight = (weight @ onehot.t()).reshape(n, 1, h, w)
    keep = (keep @ onehot.t()).reshape(n, 1, h, w)

    def inner(a, b):
        return (weight * (a[0] * b[0] + a[1] * b[1])).sum((1, 2, 3), keepdim=True)

    basis = []
    for k in range(2, re.shape[0]):
        a = v = (re[k], im[k])
        for _ in range(2):
            for q in basis:
                coef = inner(v, q)
                v = (v[0] - coef * q[0], v[1] - coef * q[1])
        length = inner(v, v)
        scale = torch.where(length > DROP * DROP * inner(a, a), torch.rsqrt(length.clamp_min(1e-30)), torch.zeros_like(length))
        basis.append((v[0] * scale, v[1] * scale))
    x = (re[0], im[0])
    since_re, since_im = torch.zeros_like(re[0]), torch.zeros_like(im[0])
    for q in basis:
        coef = inner(x, q)
        since_re, since_im = since_re + coef * q[0], since_im + coef * q[1]
    removed = noise + _idft2(since_re.reshape(-1, h, w), since_im.reshape(-1, h, w), *dft).reshape(noise.shape)
    kept_re = ((re[1] + since_re) * keep).reshape(-1, h, w)
    kept_im = ((im[1] + since_im) * keep).reshape(-1, h, w)
    return removed, _idft2(kept_re, kept_im, *dft).reshape(noise.shape)


# run(sample, timesteps) returns the shallow UNet's eps rows. history: [(x, sigma, UNet output rows, removed total)],
# latest first.
def shallow(run, sample, timesteps, x, sigma, log_sigmas, pred, history, power, broken, measure, out):
    n, c, h, w = x.shape
    noise, structured = _noise(x, _history_residual(x, pred, [entry[:3] for entry in history]))
    first, last = _ends(x, sigma, pred, out)
    ch, sh = _dft(h, x)
    cw, sw = _dft(w, x)
    totals = [entry[3] for entry in history]
    re, im = _dft2(torch.cat([x, noise, first, last, *totals]).reshape(-1, h, w), ch, sh, cw, sw)
    re, im = re.reshape(-1, n, c, h, w), im.reshape(-1, n, c, h, w)
    onehot = _bands(h, w, x)
    unknown = broken + structured
    picked = [0, 1, *range(4, 4 + len(totals))]
    removed, kept = _remove(re[picked], im[picked], noise, totals, sigma, power, onehot, (ch, sh, cw, sw))
    variance = (removed * removed).mean((1, 2, 3))
    s2 = sigma * sigma
    apply = (unknown < 0.5) & (variance > 0) & (variance < RENOISE * s2)
    sigma_eff = (s2 - variance).clamp_min(1e-20).sqrt()
    x_det = x - removed
    t_eff = (sigma_eff.log()[:, None] - log_sigmas[None, :]).abs().argmin(1).to(timesteps.dtype)
    lanes = apply.reshape(1, -1, 1, 1, 1)
    scaled = (x_det * torch.rsqrt(sigma_eff * sigma_eff + 1).reshape(-1, 1, 1, 1)).to(sample.dtype)
    sample_in = torch.where(lanes, scaled.unsqueeze(0), _blocks(x, sample)).reshape(sample.shape)
    t_in = torch.where(apply.reshape(1, -1), t_eff.reshape(1, -1), timesteps.reshape(-1, x.shape[0])).reshape(timesteps.shape)
    eps = run(sample_in, t_in)
    a, b = coefficients(sigma, pred)
    a_eff, b_eff = coefficients(sigma_eff, pred)
    target = a_eff * x_det + kept - a * x
    fixed = (target.unsqueeze(0) + b_eff * _blocks(x, eps.float())) / b
    latent = torch.where(lanes, fixed.to(eps.dtype), _blocks(x, eps)).reshape(eps.shape)

    measured = _measure(re[[2, 3, 1]], im[[2, 3, 1]], noise, sigma, power, onehot)
    power = torch.where(measure.reshape(1, 1) > 0.5, measured, power)
    # Measured on the denoised values, the image the sampler steps toward: eps errors grow as sigma falls (eps = (x -
    # denoised) / sigma) while the step that carries them into x shrinks.
    denoised = a * x + b * _blocks(x, out.float())
    difference = b * (_blocks(x, latent.float()) - _blocks(x, out.float()))
    error = ((difference * difference).sum() / (denoised * denoised).sum().clamp_min(1e-30)).sqrt().reshape(1)
    return latent, removed, (unknown > 0.5).float(), power, error


def group(names: list, values: list) -> dict:
    return dict(zip(names, values))


def unpack(inputs: dict):
    return [(inputs[f"hx{m}"], inputs[f"hs{m}"], inputs[f"he{m}"], inputs[f"hr{m}"]) for m in range(1, WINDOW + 1)]


def dynamic_axes() -> dict:
    image, vector = {0: "images", 2: "height", 3: "width"}, {0: "images"}
    axes = {"x": image, "sigma": vector, "log_sigmas": {0: "steps"}, "power": vector, "broken": vector}
    axes.update({"out": {0: "batch", 2: "height", 3: "width"}, "removed": image, "unknown": vector, "power_out": vector})
    for m in range(1, WINDOW + 1):
        axes.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": {0: f"rows{m}", 2: "height", 3: "width"}, f"hr{m}": image})
    return axes


def example_inputs(sample) -> list:
    rows, c, h, w = sample.shape
    n = max(1, rows // 2)
    kw = {"device": sample.device}
    image = lambda: torch.randn(n, c, h, w, **kw)
    sigma = lambda: torch.rand(n, **kw) * 10 + 0.1
    values = {"x": image(), "sigma": sigma(), "pred": torch.zeros(1, **kw), "power": torch.zeros(n, BINS + 1, **kw), "broken": torch.zeros(n, **kw)}
    values.update({"log_sigmas": torch.linspace(-3.5, 2.7, 1000, **kw), "measure": torch.ones(1, **kw)})
    values["out"] = torch.randn(rows, c, h, w, dtype=sample.dtype, **kw)
    for m in range(1, WINDOW + 1):
        values.update({f"hx{m}": image(), f"hs{m}": sigma(), f"he{m}": torch.randn(rows, c, h, w, dtype=sample.dtype, **kw), f"hr{m}": image()})
    return [values[name] for name in SHALLOW_INPUTS]


def input_shapes(unet_shapes: dict) -> dict:
    # Images range from 1 to the profile's most UNet rows: a sampler without uncond sends one row per image, and so do
    # Tracker.dummy_inputs.
    (r_lo, c, h_lo, w_lo), (r_opt, _, h_opt, w_opt), (r_hi, _, h_hi, w_hi) = unet_shapes["sample"]
    i_opt = max(1, r_opt // 2)
    image = [[1, c, h_lo, w_lo], [i_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    rows = [[1, c, h_lo, w_lo], [r_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    vector = [[1], [i_opt], [r_hi]]
    shapes = {"x": image, "sigma": vector, "pred": [[1]] * 3, "power": [[1, BINS + 1], [i_opt, BINS + 1], [r_hi, BINS + 1]]}
    shapes.update({"log_sigmas": [[2], [1000], [4096]], "broken": vector, "measure": [[1]] * 3, "out": rows})
    for m in range(1, WINDOW + 1):
        shapes.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": rows, f"hr{m}": image})
    return shapes


# The graph's pred input for this model sampling (0 EPS, 1 V_PREDICTION), or None for kinds it doesn't handle.
def supported(model_sampling):
    from ldm_patched.modules import model_sampling as ms

    if not isinstance(model_sampling, ms.ModelSamplingDiscrete) or type(model_sampling).timestep is not ms.ModelSamplingDiscrete.timestep:
        return None
    if getattr(model_sampling, "sigma_data", None) != 1.0 or not isinstance(model_sampling, ms.EPS):
        return None
    if isinstance(model_sampling, tuple(c for c in (getattr(ms, "EDM", None), getattr(ms, "X0", None)) if c is not None)):
        return None
    return 1.0 if isinstance(model_sampling, ms.V_PREDICTION) else 0.0


class Call:
    __slots__ = ("x", "sigma", "index", "rows")

    def __init__(self, x, sigma, index, rows):
        self.x, self.sigma, self.index, self.rows = x, sigma, index, rows


class Tracker:
    # One sampling pass's state between engine calls: references to tensors only. Every computation on them is in the
    # Speed Hacks engine's graph, so none runs in Python or on the CPU. Calls are numbered by speed.Schedule's index, as
    # the deep features' origins are.
    def __init__(self, log_sigmas, pred, device):
        self.active = pred is not None
        self.device = device
        if self.active:
            self.log_sigmas = log_sigmas.to(device=device, dtype=torch.float32)
            self.pred = torch.full((1,), float(pred), device=device)
        # (x, sigma, UNet output rows, removed total or None, index of the deep feature's call or None), newest last
        self.history = deque(maxlen=WINDOW)
        self.unknown = None  # (deep feature's call index, call index, unknown flags) of the last shallow call
        self.gap_at = None  # index of the latest call whose noise is unknown
        self.key = None
        self.power = None
        self.stash = None  # (x, sigma) of the UNet rows, from BaseModel.apply_model
        self._constants = {}

    def _constant(self, shape, value=0.0, dtype=torch.float32):
        key = (tuple(shape), value, dtype)
        if key not in self._constants:
            self._constants[key] = torch.full(shape, value, dtype=dtype, device=self.device)
        return self._constants[key]

    def gap(self, index):
        self.history.clear()
        self.unknown = None
        self.gap_at = index

    def call(self, sample, chunks: int, padded: bool, slot, index):
        stash, self.stash = self.stash, None
        rows, c, h, w = sample.shape
        if not self.active or stash is None or padded or slot not in (None, 0) or not chunks or rows % chunks:
            self.gap(index)
            return None
        x, sigma = stash
        images = rows // chunks
        if tuple(x.shape) != (rows, c, h, w) or sigma.shape[0] != rows:
            self.gap(index)
            return None
        key = (images, c, h, w)
        if key != self.key:
            self.gap(index)
            self.key = key
        return Call(x[:images].float(), sigma[:images].float(), index, rows)

    def _history(self, call, origin) -> dict:
        n, c, h, w = call.x.shape
        values = {}
        recent = list(self.history)[::-1]
        zeros = self._constant((n, c, h, w))
        for m in range(1, WINDOW + 1):
            if m <= len(recent):
                hx, hs, he, hr, since = recent[m - 1]
                hr = hr if hr is not None and origin is not None and since == origin else zeros
            else:
                hx, hs, he, hr = zeros, self._constant((n,)), self._constant((n, c, h, w), dtype=torch.float16), zeros
            values.update({f"hx{m}": hx, f"hs{m}": hs, f"he{m}": he, f"hr{m}": hr})
        return values

    def _power(self, n):
        return self.power if self.power is not None and self.power.shape[0] == n else self._constant((n, BINS + 1))

    def shallow_inputs(self, call, origin, out=None) -> dict:
        # origin: the index of the full call whose deep feature this call uses; None for a check on the checked call's
        # own deep feature, which holds this call's noise already.
        n, c, h, w = call.x.shape
        ones = self._constant((n,), 1.0)
        if origin is None:
            broken = ones
        elif self.unknown is not None and self.unknown[:2] == (origin, call.index - 1):
            broken = self.unknown[2]
        elif call.index - 1 == origin and not (self.gap_at is not None and self.gap_at > origin):
            broken = self._constant((n,))
        else:
            broken = ones
        values = {"x": call.x, "sigma": call.sigma, "log_sigmas": self.log_sigmas, "pred": self.pred, "power": self._power(n)}
        values.update({"broken": broken, "measure": self._constant((1,), float(out is not None))})
        values["out"] = out if out is not None else self._constant((call.rows, c, h, w), dtype=torch.float16)
        values.update(self._history(call, origin))
        return values

    def dummy_inputs(self, sample, out=None) -> dict:
        # For a call without tracking: one image per UNet row and broken = 1, for which fresh.shallow returns the plain
        # shallow UNet's output.
        rows, c, h, w = sample.shape
        image, zeros, ones = self._constant((rows, c, h, w)), self._constant((rows,)), self._constant((rows,), 1.0)
        values = {"x": image, "sigma": ones, "log_sigmas": self._constant((2,)), "pred": self._constant((1,))}
        values.update({"power": self._constant((rows, BINS + 1)), "broken": ones, "measure": self._constant((1,))})
        values["out"] = out if out is not None else self._constant((rows, c, h, w), dtype=torch.float16)
        for m in range(1, WINDOW + 1):
            values.update({f"hx{m}": image, f"hs{m}": ones, f"he{m}": self._constant((rows, c, h, w), dtype=torch.float16), f"hr{m}": image})
        return values

    def after_shallow(self, call, outputs: dict, origin):
        self.history.append((call.x, call.sigma, outputs["latent"], outputs["removed"], origin))
        self.unknown = (origin, call.index, outputs["unknown"])

    def after_full(self, call, out, outputs=None):
        # outputs: the check's, which measured the noise share at this call
        self.history.append((call.x, call.sigma, out, None, None))
        self.unknown = None
        if outputs is not None:
            self.power = outputs["power_out"]
