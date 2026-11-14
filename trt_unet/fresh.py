# Fresh noise for Speed Hacks. Ancestral and SDE samplers add new noise to x between model calls. The shallow engine
# reuses a deep feature computed from x before that noise was drawn, so its eps holds the old noise pattern, and
# denoised = x - sigma * eps carries the new noise as if it were image content: a burst of latent noise at the first
# shallow step, larger the higher sigma is there.
#
# Finding the noise (both graphs below): a k-diffusion-style sampler's x is a linear combination of the last few
# calls' x and denoised values, plus whatever noise it drew since the last call. Per image, x is fitted by least
# squares on WINDOW previous calls' x and the denoised values of their first and last block of UNet rows (cond and
# uncond); the residual is that noise. The fit is classical Gram-Schmidt applied twice (CGS2) in FP32: the columns are
# nearly dependent (condition numbers ~1e8, simulated), which rules out normal equations. A residual that isn't white
# is structure outside the fit's span: the chain of calls is then "broken" until the next full call, and shallow calls
# get the plain shallow UNet's output.
#
# Shallow call: x minus the noises found since the last full call, each scaled to its amount in x (_remove), is where
# the sampler would be without them, at noise level sigma_eff = sqrt(sigma^2 - removed variance). The shallow UNet
# runs there, at the timestep model_sampling.timestep() gives sigma_eff (nearest in reForge's log-sigma table). The
# share of the removed noise that a full UNet's denoised output would contain is added back to the shallow UNet's
# denoised value, per radial frequency band (DFTs as matrix products: TensorRT has no FFT). The output is the eps that
# gives that denoised value at the call's real x and sigma.
#
# Probe (its own small engine, run after each full call): the same fit finds the noise drawn just before the full call.
# Regressing the full call's denoised rows on it, band by band, measures the share S of that noise in them (unbiased:
# the noise is independent of everything else in them). S is stored as signal power P = S sigma^2 / (1 - S), the
# Wiener relation, averaged over full calls, and converted back to S = P / (P + sigma^2) at a shallow call's sigma.
#
# "Simulated" figures below: transcriptions of reForge's sampler update rules (Euler a, Euler a CFG++, DPM++ 2M SDE,
# DPM++ 3M SDE, DPM++ SDE, DPM++ 2S a, Euler with churn, DPM++ 2M, Heun) run against an exact Wiener denoiser,
# 1x4x128x128 latents, 20 Karras steps, CFG with cond and uncond rows.
#
# Engine inputs, per image (B = UNet rows / row blocks): x and sigma as reForge's apply_model received them (FP32),
# hx/hs/he<m> the m-th previous call's x, sigma and UNet output rows, found<f> the noise found by the shallow calls
# since the last full call (zeros if none), power (B, BINS + 1): P per band and the number of measurements, broken.
import math
from collections import deque

import torch

from .speed import MAX_AGE

WINDOW = 5  # iPNDM's 4th-order steps use the last 4 calls' x and denoised values; one more as a margin
FOUND = MAX_AGE  # speed.Schedule runs at most MAX_AGE shallow calls between two full calls
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
SIGNATURE = {"window": WINDOW, "found": FOUND, "bins": BINS}

HISTORY = [name for m in range(1, WINDOW + 1) for name in (f"hx{m}", f"hs{m}", f"he{m}")]
SHALLOW_INPUTS = ["x", "sigma", "log_sigmas", "pred", *HISTORY, *(f"found{f}" for f in range(1, FOUND + 1)), "power", "broken"]
SHALLOW_OUTPUTS = ["latent", "noise", "broken_out"]
PROBE_INPUTS = ["x", "sigma", "pred", "out", *HISTORY, "power", "broken"]
PROBE_OUTPUTS = ["power_out"]


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


def _classify(x, r, broken):
    energy = (r * r).sum((1, 2, 3))
    lag = torch.maximum((r[:, :, 1:] * r[:, :, :-1]).sum((1, 2, 3)).abs(), (r[..., 1:] * r[..., :-1]).sum((1, 2, 3)).abs())
    significant = energy > FLOOR * FLOOR * (x * x).sum((1, 2, 3))
    white = lag < WHITE * energy
    return significant & white & (broken < 0.5), significant & ~white


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


def _measure(first, last, r, sigma, drawn, power):
    n, c, h, w = r.shape
    ch, sh = _dft(h, r)
    cw, sw = _dft(w, r)
    re, im = _dft2(torch.cat([first, last, r]).reshape(-1, h, w), ch, sh, cw, sw)
    re, im = re.reshape(3, n, c, h, w), im.reshape(3, n, c, h, w)
    onehot = _bands(h, w, r)
    cross = ((re[0] + re[1]) * re[2] + (im[0] + im[1]) * im[2]).sum(1).reshape(n, -1) @ onehot
    energy = (2 * (re[2] * re[2] + im[2] * im[2])).sum(1).reshape(n, -1) @ onehot
    share = (cross / energy.clamp_min(1e-30)).clamp(max=SHARE_MAX)
    measured = share * (sigma * sigma).reshape(-1, 1) / (1 - share)
    power_, count = power[:, :BINS], power[:, BINS:]
    averaged = torch.where(count > 0, EMA * power_ + (1 - EMA) * measured, measured)
    drawn = drawn.reshape(-1, 1)
    return torch.cat([torch.where(drawn, averaged, power_), count + drawn.float()], 1)


def _remove(x, sigma, noises, power):
    # Each noise's amount in x by generalized least squares, each DFT frequency weighted by 1 / (P + sigma^2): image
    # content is uncorrelated with the noises and adds to the estimate's error with spectrum P. Simulated: up to a third
    # less distance to the all-full image than plain inner products. Returns (the noises' total in x, the share of that
    # total a full UNet's denoised output would contain).
    n, c, h, w = x.shape
    ch, sh = _dft(h, x)
    cw, sw = _dft(w, x)
    onehot = _bands(h, w, x)
    re, im = _dft2(torch.cat([x, *noises]).reshape(-1, h, w), ch, sh, cw, sw)
    re, im = re.reshape(-1, n, c, h, w), im.reshape(-1, n, c, h, w)
    s2 = (sigma * sigma).reshape(-1, 1)
    power_ = power[:, :BINS].clamp_min(0)
    measured = power[:, BINS:] > 0
    weight = torch.where(measured, s2 / (power_ + s2), torch.ones_like(power_))
    keep = torch.where(measured, power_ / (power_ + s2), torch.zeros_like(power_))
    weight = (weight @ onehot.t()).reshape(n, 1, h, w)
    keep = (keep @ onehot.t()).reshape(n, 1, h, w)
    amount = (weight * (re[:1] * re[1:] + im[:1] * im[1:])).sum((2, 3, 4), keepdim=True)
    amount = amount / (weight * (re[1:] * re[1:] + im[1:] * im[1:])).sum((2, 3, 4), keepdim=True).clamp_min(1e-30)
    removed = (amount * torch.stack(noises)).sum(0)
    kept_re, kept_im = ((amount * re[1:]).sum(0) * keep).reshape(-1, h, w), ((amount * im[1:]).sum(0) * keep).reshape(-1, h, w)
    return removed, _idft2(kept_re, kept_im, ch, sh, cw, sw).reshape(x.shape)


# run(sample, timesteps) returns the shallow UNet's eps rows. history: [(x, sigma, UNet output rows)], latest first.
def shallow(run, sample, timesteps, x, sigma, log_sigmas, pred, history, found, power, broken):
    r = _history_residual(x, pred, history)
    drawn, structured = _classify(x, r, broken)
    broken_out = (broken > 0.5) | structured
    noise = torch.where(drawn.reshape(-1, 1, 1, 1), r, torch.zeros_like(r))
    removed, kept = _remove(x, sigma, [*found, noise], power)
    variance = (removed * removed).mean((1, 2, 3))
    s2 = sigma * sigma
    apply = ~broken_out & (variance > 0) & (variance < RENOISE * s2)
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
    return latent, noise, broken_out.float()


def probe(x, sigma, pred, out, history, power, broken):
    r = _history_residual(x, pred, history)
    drawn, _ = _classify(x, r, broken)
    first, last = _ends(x, sigma, pred, out)
    return _measure(first, last, r, sigma, drawn, power)


def group(names: list, values: list) -> dict:
    return dict(zip(names, values))


def unpack(inputs: dict):
    history = [(inputs[f"hx{m}"], inputs[f"hs{m}"], inputs[f"he{m}"]) for m in range(1, WINDOW + 1)]
    found = [inputs[f"found{f}"] for f in range(1, FOUND + 1)] if "found1" in inputs else []
    return history, found


def dynamic_axes(shallow_graph: bool) -> dict:
    image = {0: "images", 2: "height", 3: "width"}
    axes = {"x": image, "sigma": {0: "images"}, "power": {0: "images"}, "broken": {0: "images"}}
    for m in range(1, WINDOW + 1):
        axes.update({f"hx{m}": image, f"hs{m}": {0: "images"}, f"he{m}": {0: f"rows{m}", 2: "height", 3: "width"}})
    if shallow_graph:
        axes.update({"log_sigmas": {0: "steps"}, "noise": image, "broken_out": {0: "images"}})
        axes.update({f"found{f}": image for f in range(1, FOUND + 1)})
    else:
        axes.update({"out": {0: "batch", 2: "height", 3: "width"}, "power_out": {0: "images"}})
    return axes


def example_inputs(sample, shallow_graph: bool) -> list:
    rows, c, h, w = sample.shape
    n = max(1, rows // 2)
    kw = {"device": sample.device}
    image = lambda: torch.randn(n, c, h, w, **kw)
    sigma = lambda: torch.rand(n, **kw) * 10 + 0.1
    values = {"x": image(), "sigma": sigma(), "pred": torch.zeros(1, **kw), "power": torch.zeros(n, BINS + 1, **kw), "broken": torch.zeros(n, **kw)}
    values["log_sigmas"] = torch.linspace(-3.5, 2.7, 1000, **kw)
    values["out"] = torch.randn(rows, c, h, w, dtype=sample.dtype, **kw)
    for m in range(1, WINDOW + 1):
        values.update({f"hx{m}": image(), f"hs{m}": sigma(), f"he{m}": torch.randn(rows, c, h, w, dtype=sample.dtype, **kw)})
    for f in range(1, FOUND + 1):
        values[f"found{f}"] = image()
    return [values[name] for name in (SHALLOW_INPUTS if shallow_graph else PROBE_INPUTS)]


def input_shapes(unet_shapes: dict, shallow_graph: bool) -> dict:
    # Images range from 1 to the profile's most UNet rows: a sampler without uncond sends one row per image, and so do
    # Tracker.dummy_inputs.
    (r_lo, c, h_lo, w_lo), (r_opt, _, h_opt, w_opt), (r_hi, _, h_hi, w_hi) = unet_shapes["sample"]
    i_opt = max(1, r_opt // 2)
    image = [[1, c, h_lo, w_lo], [i_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    rows = [[1, c, h_lo, w_lo], [r_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    vector = [[1], [i_opt], [r_hi]]
    shapes = {"x": image, "sigma": vector, "pred": [[1]] * 3, "power": [[1, BINS + 1], [i_opt, BINS + 1], [r_hi, BINS + 1]], "broken": vector}
    for m in range(1, WINDOW + 1):
        shapes.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": rows})
    if shallow_graph:
        shapes["log_sigmas"] = [[2], [1000], [4096]]
        shapes.update({f"found{f}": image for f in range(1, FOUND + 1)})
    else:
        shapes["out"] = [list(s) for s in unet_shapes["sample"]]
    return shapes


# The graphs' pred input for this model sampling (0 EPS, 1 V_PREDICTION), or None for kinds they don't handle.
def supported(model_sampling):
    from ldm_patched.modules import model_sampling as ms

    if not isinstance(model_sampling, ms.ModelSamplingDiscrete) or type(model_sampling).timestep is not ms.ModelSamplingDiscrete.timestep:
        return None
    if getattr(model_sampling, "sigma_data", None) != 1.0 or not isinstance(model_sampling, ms.EPS):
        return None
    if isinstance(model_sampling, (ms.EDM, ms.X0)):
        return None
    return 1.0 if isinstance(model_sampling, ms.V_PREDICTION) else 0.0


class Call:
    __slots__ = ("x", "sigma")

    def __init__(self, x, sigma):
        self.x, self.sigma = x, sigma


class Tracker:
    # One sampling pass's state between engine calls: references to tensors only. Every computation on them is in the
    # engines' graphs (shallow, probe), so none runs in Python or on the CPU.
    def __init__(self, log_sigmas, pred, device):
        self.active = pred is not None
        self.device = device
        if self.active:
            self.log_sigmas = log_sigmas.to(device=device, dtype=torch.float32)
            self.pred = torch.full((1,), float(pred), device=device)
        self.history = deque(maxlen=WINDOW)  # (x, sigma, UNet output rows), newest last
        self.key = None
        self.found = []
        self.power = None
        self.broken = None
        self.stash = None  # (x, sigma) of the UNet rows, from BaseModel.apply_model
        self._zeros = {}

    def zeros(self, shape, dtype=torch.float32):
        key = (tuple(shape), dtype)
        if key not in self._zeros:
            self._zeros[key] = torch.zeros(shape, dtype=dtype, device=self.device)
        return self._zeros[key]

    def ones(self, shape):
        key = (tuple(shape), "ones")
        if key not in self._zeros:
            self._zeros[key] = torch.ones(shape, dtype=torch.float32, device=self.device)
        return self._zeros[key]

    def gap(self):
        self.history.clear()
        self.found = []
        self.broken = None

    def call(self, sample, chunks: int, padded: bool, slot):
        stash, self.stash = self.stash, None
        rows, c, h, w = sample.shape
        if not self.active or stash is None or padded or slot not in (None, 0) or not chunks or rows % chunks:
            self.gap()
            return None
        x, sigma = stash
        images = rows // chunks
        if tuple(x.shape) != (rows, c, h, w) or sigma.shape[0] != rows:
            self.gap()
            return None
        key = (images, c, h, w)
        if key != self.key:
            self.gap()
            self.key = key
        x, sigma = x[:images], sigma[:images]
        if x.dtype != torch.float32:
            x = x.float()
        if sigma.dtype != torch.float32:
            sigma = sigma.float()
        return Call(x, sigma)

    def _history(self, call) -> dict:
        n, c, h, w = call.x.shape
        values = {}
        recent = list(self.history)[::-1]
        for m in range(1, WINDOW + 1):
            if m <= len(recent):
                hx, hs, he = recent[m - 1]
            else:
                hx, hs, he = self.zeros((n, c, h, w)), self.zeros((n,)), self.zeros((n, c, h, w), torch.float16)
            values.update({f"hx{m}": hx, f"hs{m}": hs, f"he{m}": he})
        return values

    def _state(self, n):
        power = self.power if self.power is not None and self.power.shape[0] == n else self.zeros((n, BINS + 1))
        broken = self.broken if self.broken is not None and self.broken.shape[0] == n else self.ones((n,))
        return power, broken

    def shallow_inputs(self, call) -> dict:
        n, c, h, w = call.x.shape
        power, broken = self._state(n)
        values = {"x": call.x, "sigma": call.sigma, "log_sigmas": self.log_sigmas, "pred": self.pred, "power": power, "broken": broken}
        values.update(self._history(call))
        for f in range(1, FOUND + 1):
            values[f"found{f}"] = self.found[f - 1] if f <= len(self.found) else self.zeros((n, c, h, w))
        return values

    def dummy_inputs(self, sample) -> dict:
        # For a shallow call without tracking: one image per UNet row and broken = 1, for which fresh.shallow returns
        # the plain shallow UNet's output.
        rows, c, h, w = sample.shape
        image, vector = self.zeros((rows, c, h, w)), self.ones((rows,))
        values = {"x": image, "sigma": vector, "log_sigmas": self.zeros((2,)), "pred": self.zeros((1,)), "power": self.zeros((rows, BINS + 1)), "broken": vector}
        for m in range(1, WINDOW + 1):
            values.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": self.zeros((rows, c, h, w), torch.float16)})
        values.update({f"found{f}": image for f in range(1, FOUND + 1)})
        return values

    def after_shallow(self, call, outputs: dict):
        self.history.append((call.x, call.sigma, outputs["latent"]))
        if len(self.found) >= FOUND:
            self.gap()
            return
        self.found.append(outputs["noise"])
        self.broken = outputs["broken_out"]

    def probe_inputs(self, call, out):
        if not self.history:
            return None
        n = call.x.shape[0]
        power, _ = self._state(n)
        values = {"x": call.x, "sigma": call.sigma, "pred": self.pred, "out": out, "power": power, "broken": self.zeros((n,))}
        values.update(self._history(call))
        return values

    def after_full(self, call, out, power=None):
        n = call.x.shape[0]
        if power is not None:
            self.power = power
        self.history.append((call.x, call.sigma, out))
        self.found = []
        self.broken = self.zeros((n,))
