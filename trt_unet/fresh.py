# Fresh noise for Speed Hacks. Ancestral and SDE samplers add new noise to x between model calls. A shallow call reuses
# the deep feature of an earlier call, computed before that noise was drawn, and the shallow UNet alone attributes
# little of the new noise to its noise prediction: denoised = x - sigma * eps keeps it as image content, a burst of
# latent noise at the first shallow call. The shallow engine's graph (shallow() below) removes it.
#
# A k-diffusion sampler builds x linearly from earlier calls' x and denoised values (per-image scalar coefficients),
# plus the noise it draws. For the calls after the deep feature's call j, write x_i = shadow_i + N_i, N_i being the
# noise drawn since j as it stands in x_i (N_i = 0 for i <= j). A shallow call's denoised value is computed on its
# shadow and holds none of N, so x = (a combination of the last WINDOW calls' shadows and denoised values) + N, and N
# is independent of all of those: least squares of x on them leaves N as the residual. The shallow UNet runs on x - N
# at sigma_eff = sqrt(sigma^2 - var(N)), the noise level that leaves, at the timestep model_sampling.timestep() gives
# sigma_eff, and its denoised value is returned as the output that gives the same denoised value at the call's real x
# and sigma. The sampler's x keeps N, which the next full call sees whole.
#
# The fit is classical Gram-Schmidt applied twice (CGS2) in FP32: TensorRT has no FP64 or QR, and consecutive calls'
# values are nearly parallel, which rules out normal equations in FP32. Its denoised columns are the first and last
# block of UNet rows (cond and uncond), which span CFG's and CFG++'s combinations. A residual that isn't white is
# structure outside the fit's span (a sampler the window doesn't cover): that call runs the plain shallow UNet, and so
# do the later calls on the same deep feature. Deterministic samplers leave N ~ 0, which changes nothing.
#
# Engine inputs, per image: x and sigma as the sampler passed them to reForge's CFG denoiser (FP32); hx/hs/he/hr<m>
# the m-th previous call's x, sigma, UNet output rows and N (zeros up to the deep feature's call); broken: 1 where N is
# unknown. Outputs: latent (the UNet output rows), removed (this call's N), unknown (this call's broken, for the next).
from collections import deque

import torch

# DPM++ 3M SDE's update uses the last 3 calls' denoised values; DPM++ SDE's and DPM++ 2S a's use the x of 2 calls back.
WINDOW = 3
# Lag-1 autocorrelation / energy of the residual: ~1 / sqrt(pixels) for noise (under 0.01 at 512x512), near 1 for
# image structure.
WHITE = 0.1
# CGS2 drops a column with less than this share of its norm outside the earlier columns' span. FP32 rounding is ~1e-7.
DROP = 1e-4
# Most of sigma^2 that N may hold. LCM's sampler redraws all of x's noise every call (share 1), where sigma_eff would
# reach 0.
RENOISE = 0.99
# Engines record this; one built with another takes other inputs.
SIGNATURE = {"inputs": 4, "window": WINDOW}

HISTORY = [f"{name}{m}" for m in range(1, WINDOW + 1) for name in ("hx", "hs", "he", "hr")]
INPUTS = ["x", "sigma", "log_sigmas", "pred", "broken", *HISTORY]
OUTPUTS = ["latent", "removed", "unknown"]


def coefficients(sigma, pred):
    # reForge's calculate_denoised as a * x + b * output, for EPS (pred 0) and V_PREDICTION (pred 1) with sigma_data 1
    s2 = sigma * sigma
    v = pred > 0.5
    a = torch.where(v, 1 / (s2 + 1), torch.ones_like(sigma))
    b = torch.where(v, -sigma * torch.rsqrt(s2 + 1), -sigma)
    return a.reshape(-1, 1, 1, 1), b.reshape(-1, 1, 1, 1)


def _blocks(x, rows):  # (blocks * images, ...) -> (blocks, images, ...)
    return rows.reshape(-1, *x.shape)


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
    energy = (r * r).sum((1, 2, 3))
    lag = torch.maximum((r[:, :, 1:] * r[:, :, :-1]).sum((1, 2, 3)).abs(), (r[..., 1:] * r[..., :-1]).sum((1, 2, 3)).abs())
    return lag < WHITE * energy


# run(sample, timesteps) returns the shallow UNet's output rows. inputs: INPUTS by name.
def shallow(run, sample, timesteps, inputs: dict):
    x, sigma, pred = inputs["x"], inputs["sigma"], inputs["pred"]
    columns = []
    for m in range(1, WINDOW + 1):
        hx, hs, he, hr = (inputs[f"{name}{m}"] for name in ("hx", "hs", "he", "hr"))
        a, b = coefficients(hs, pred)
        denoised = a * hx + b * _blocks(hx, he.float())
        columns += [hx - hr, denoised[0], denoised[-1]]
    noise = _residual(x, columns)
    variance = (noise * noise).mean((1, 2, 3))
    s2 = sigma * sigma
    apply = (inputs["broken"] < 0.5) & _white(noise) & (variance < RENOISE * s2)
    sigma_eff = (s2 - variance).clamp_min((1 - RENOISE) * s2).sqrt()
    x_eff = x - noise
    t_eff = (sigma_eff.log()[:, None] - inputs["log_sigmas"][None, :]).abs().argmin(1).to(timesteps.dtype)
    lanes = apply.reshape(1, -1, 1, 1, 1)
    scaled = (x_eff * torch.rsqrt(sigma_eff * sigma_eff + 1).reshape(-1, 1, 1, 1)).to(sample.dtype)
    out = run(
        torch.where(lanes, scaled.unsqueeze(0), _blocks(x, sample)).reshape(sample.shape),
        torch.where(apply.reshape(1, -1), t_eff.reshape(1, -1), timesteps.reshape(-1, x.shape[0])).reshape(timesteps.shape),
    )
    a, b = coefficients(sigma, pred)
    a_eff, b_eff = coefficients(sigma_eff, pred)
    fixed = (a_eff * x_eff - a * x + b_eff * _blocks(x, out.float())) / b
    latent = torch.where(lanes, fixed.to(out.dtype), _blocks(x, out)).reshape(out.shape)
    removed = torch.where(apply.reshape(-1, 1, 1, 1), noise, torch.zeros_like(noise))
    return latent, removed, (~apply).float()


def dynamic_axes() -> dict:
    image, vector = {0: "images", 2: "height", 3: "width"}, {0: "images"}
    axes = {"x": image, "sigma": vector, "log_sigmas": {0: "sigmas"}, "broken": vector, "removed": image, "unknown": vector}
    for m in range(1, WINDOW + 1):
        axes.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": {0: f"rows{m}", 2: "height", 3: "width"}, f"hr{m}": image})
    return axes


def example_inputs(sample) -> list:
    rows, c, h, w = sample.shape
    n = max(1, rows // 2)
    kw = {"device": sample.device}
    image = lambda: torch.randn(n, c, h, w, **kw)
    sigma = lambda: torch.rand(n, **kw) * 10 + 0.1
    values = {"x": image(), "sigma": sigma(), "log_sigmas": torch.linspace(-3.5, 2.7, 1000, **kw), "pred": torch.zeros(1, **kw)}
    values["broken"] = torch.zeros(n, **kw)
    for m in range(1, WINDOW + 1):
        values.update({f"hx{m}": image(), f"hs{m}": sigma(), f"he{m}": torch.randn(rows, c, h, w, dtype=sample.dtype, **kw), f"hr{m}": image()})
    return [values[name] for name in INPUTS]


def input_shapes(unet_shapes: dict) -> dict:
    # Images range from 1 to the profile's most UNet rows: a sampler without uncond sends one row per image, and
    # Tracker.inputs sends one image per row for a call it doesn't track.
    (r_lo, c, h_lo, w_lo), (r_opt, _, h_opt, w_opt), (r_hi, _, h_hi, w_hi) = unet_shapes["sample"]
    i_opt = max(1, r_opt // 2)
    image = [[1, c, h_lo, w_lo], [i_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    rows = [[1, c, h_lo, w_lo], [r_opt, c, h_opt, w_opt], [r_hi, c, h_hi, w_hi]]
    vector = [[1], [i_opt], [r_hi]]
    shapes = {"x": image, "sigma": vector, "log_sigmas": [[2], [1000], [4096]], "pred": [[1]] * 3, "broken": vector}
    for m in range(1, WINDOW + 1):
        shapes.update({f"hx{m}": image, f"hs{m}": vector, f"he{m}": rows, f"hr{m}": image})
    return shapes


# The graph's pred input for this model sampling (0 EPS, 1 V_PREDICTION), or None for kinds the graph doesn't handle.
def supported(model_sampling):
    from ldm_patched.modules import model_sampling as ms

    if not isinstance(model_sampling, ms.ModelSamplingDiscrete) or type(model_sampling).timestep is not ms.ModelSamplingDiscrete.timestep:
        return None
    if model_sampling.sigma_data != 1.0 or not isinstance(model_sampling, ms.EPS) or isinstance(model_sampling, (ms.EDM, ms.X0)):
        return None
    return 1.0 if isinstance(model_sampling, ms.V_PREDICTION) else 0.0


class Call:
    __slots__ = ("x", "sigma", "index")

    def __init__(self, x, sigma, index):
        self.x, self.sigma, self.index = x, sigma, index


class Tracker:
    # One sampling pass's state between Speed Hacks engine calls: references to tensors only. Calls are numbered by
    # speed.Schedule's index, as deep features' origins are.
    def __init__(self, model_sampling, device):
        pred = supported(model_sampling)
        self.active = pred is not None
        self.device = device
        self.log_sigmas = model_sampling.log_sigmas.to(device=device, dtype=torch.float32) if self.active else torch.zeros(2, device=device)
        self.pred = torch.full((1,), pred or 0.0, device=device)
        # (Call, UNet output rows, N or None, index of the deep feature's call or None), newest last
        self.history = deque(maxlen=WINDOW)
        self.last = None  # (deep feature's call index, call index, unknown flags) of the latest shallow call
        self.gap_at = None  # index of the latest call left out of history
        self.key = None
        self.stash = None  # (x, sigma) the CFG denoiser was called with, for the call's first UNet call
        self._constants = {}

    def _constant(self, shape, value=0.0, dtype=torch.float32):
        key = (tuple(shape), value, dtype)
        if key not in self._constants:
            self._constants[key] = torch.full(shape, value, dtype=dtype, device=self.device)
        return self._constants[key]

    def _gap(self, index):
        self.history.clear()
        self.last = None
        self.gap_at = index

    def call(self, sample, chunks: int, padded: bool, slot, index):
        # A UNet call is tracked when it's the denoiser call's only one: its rows are chunks blocks of the same images.
        stash, self.stash = self.stash, None
        rows, c, h, w = sample.shape
        if stash is None or padded or slot not in (None, 0) or not chunks or rows % chunks:
            self._gap(index)
            return None
        x, sigma = stash
        if tuple(x.shape) != (rows // chunks, c, h, w) or tuple(sigma.shape) != (rows // chunks,):
            self._gap(index)
            return None
        if x.shape != self.key:
            self._gap(index)
            self.key = x.shape
        return Call(x.float(), sigma.float(), index)

    def inputs(self, call, origin, sample) -> dict:
        # origin: the index of the call whose deep feature this call uses
        if call is None:
            return self._untracked(sample)
        n, c, h, w = call.x.shape
        if not self.active:
            broken = self._constant((n,), 1.0)
        elif self.last is not None and self.last[:2] == (origin, call.index - 1):
            broken = self.last[2]
        elif call.index - 1 == origin and (self.gap_at is None or self.gap_at < origin):
            broken = self._constant((n,))
        else:
            broken = self._constant((n,), 1.0)
        values = {"x": call.x, "sigma": call.sigma, "log_sigmas": self.log_sigmas, "pred": self.pred, "broken": broken}
        zeros = self._constant((n, c, h, w))
        recent = list(self.history)[::-1]
        for m in range(1, WINDOW + 1):
            if m <= len(recent):
                past, out, removed, since = recent[m - 1]
                hx, hs, he, hr = past.x, past.sigma, out, removed if removed is not None and since == origin else zeros
            else:
                hx, hs, he, hr = zeros, self._constant((n,)), self._constant((n, c, h, w), dtype=torch.float16), zeros
            values.update({f"hx{m}": hx, f"hs{m}": hs, f"he{m}": he, f"hr{m}": hr})
        return values

    def _untracked(self, sample) -> dict:
        # One image per UNet row and broken = 1, for which shallow() returns the plain shallow UNet's output
        rows, c, h, w = sample.shape
        image, ones = self._constant((rows, c, h, w)), self._constant((rows,), 1.0)
        values = {"x": image, "sigma": ones, "log_sigmas": self._constant((2,)), "pred": self._constant((1,)), "broken": ones}
        for m in range(1, WINDOW + 1):
            values.update({f"hx{m}": image, f"hs{m}": ones, f"he{m}": self._constant((rows, c, h, w), dtype=torch.float16), f"hr{m}": image})
        return values

    def after_shallow(self, call, outputs: dict, origin):
        self.history.append((call, outputs["latent"], outputs["removed"], origin))
        self.last = (origin, call.index, outputs["unknown"])

    def after_full(self, call, out):
        self.history.append((call, out, None, None))
        self.last = None

    def error(self, call, shallow, out):
        # A check's relative error, on the denoised values: eps errors grow as sigma falls (eps = (x - denoised) / sigma)
        # while the step that carries them into x shrinks.
        a, b = coefficients(call.sigma, self.pred)
        full = _blocks(call.x, out.float())
        denoised = a * call.x + b * full
        difference = b * (_blocks(call.x, shallow.float()) - full)
        return ((difference * difference).sum() / (denoised * denoised).sum().clamp_min(1e-30)).sqrt()
