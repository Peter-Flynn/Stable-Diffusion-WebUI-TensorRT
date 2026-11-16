# Fresh noise for Speed Hacks. Ancestral, SDE and other noise-adding samplers add new noise to x between model calls. A
# shallow call reuses the deep feature of an earlier call, computed before that noise was drawn, and the shallow UNet
# alone attributes little of the new noise to its noise prediction: the denoised value keeps it as image content, a
# burst of latent noise at the first shallow call. The shallow engine's graph (shallow() below) removes it.
#
# For every k-diffusion sampler, the x it passes to the model is a linear combination (per-image scalar coefficients)
# of earlier calls' x and denoised values, plus the noise it draws. Here x is what the sampler passed to reForge's CFG
# denoiser and the denoised value is what that returned, after any guidance, with the uncond rows' denoised value as a
# column of its own for CFG++ samplers: the fit doesn't depend on how they were computed. For the calls after the deep
# feature's call j, write x_i = shadow_i + N_i, N_i being the noise drawn since j as it stands in x_i (N_i = 0 for
# i <= j). A shallow call's denoised value is computed on its shadow and holds none of N, so x = (a combination of the
# last WINDOW calls' shadows and denoised values) + N, with N independent of all of those: least squares of x on them
# leaves N as the residual. The shallow UNet runs on x - N at sigma_eff = sqrt(sigma^2 - var(N)), the noise level that
# leaves, and its output is converted to the one that gives the same denoised value at the call's real x and sigma.
# The sampler's x keeps N, which the next full call sees whole.
#
# The model sampling's own timestep, calculate_input and calculate_denoised (linear in their tensor arguments for every
# kind reForge has) come in as tables over log sigma, so the graph serves any of them. The fit is classical
# Gram-Schmidt applied twice (CGS2) in FP32: TensorRT has no FP64 or QR, and consecutive calls' values are nearly
# parallel, which rules out normal equations in FP32. The correction applies per image only where the graph finds its
# assumptions hold: the UNet's input rows are calculate_input(sigma, x) (not where the denoiser edited x, as
# inpainting's "mask before denoising" does, or a timestep-based sampler such as DDIM scales it differently), and the
# residual is white (a sampler the window doesn't cover leaves structure). Elsewhere the call runs the plain shallow
# UNet, and on a pass where noise was found Tracker.safe() has the runtime run such calls in full instead.
#
# Engine inputs: x and sigma per image as the sampler passed them; tables (see tables()); prior: the previous call's
# reason where it ran on the same deep feature, 0 where that call was the deep feature's own, 1 where the shadows below
# aren't valid; hx/hr/hd/hu<m>: the m-th previous call's x, N (zeros up to the deep feature's call), denoised value and
# uncond denoised value (zeros if none). Outputs: latent (the UNet output rows), removed (this call's N), reason (per
# image, why N wasn't removed: 0 it was, 1 the previous call on this deep feature didn't remove its N, 2 the UNet input
# rows aren't calculate_input(sigma, x), 3 the residual isn't white, 4 the residual exceeds sigma^2), share (var(N) /
# sigma^2 where the residual was white, else -1).
import math
from collections import deque

import torch

# SA-Solver's corrector uses the last 4 calls' denoised values and DPM++ 3M SDE's update the last 3; DPM++ SDE, DPM++
# 2S a and Heun step from the x of 2 calls back.
WINDOW = 4
# Lag-1 autocorrelation / energy of the residual: ~1 / sqrt(pixels) for noise (under 0.01 at 512x512), near 1 for
# image structure.
WHITE = 0.1
# CGS2 drops a column with less than this share of its norm outside the earlier columns' span. FP32 rounding is ~1e-7.
DROP = 1e-4
# Most var(N) may exceed sigma^2 by. A variance over d values has relative standard deviation sqrt(2 / d), 2.2% for
# a 32x32 latent of 4 channels, and LCM's sampler redraws all of x's noise every call (var(N) = sigma^2).
CEILING = 1.1
# The UNet's input rows match calculate_input(sigma, x) to FP16 rounding, 2^-11 of the largest value.
GATE = 0.01
# A residual share of sigma^2 above which the sampler adds noise: deterministic samplers leave FP32 rounding (~1e-12),
# ancestral and SDE samplers 0.01 to 0.5 (SDXL measurements).
NOISY = 1e-4
# Table points over log sigma, from sigma_min / 2 to sigma_max * 2: 0.0065 apart for a zero-terminal-SNR model
# (sigma_max 4096), where linear interpolation of the smooth columns is exact to ~1e-5.
GRID = 2048
# Engines record this; one built with another takes other inputs.
SIGNATURE = {"inputs": 5, "window": WINDOW, "grid": GRID}

HISTORY = [f"{name}{m}" for m in range(1, WINDOW + 1) for name in ("hx", "hr", "hd", "hu")]
INPUTS = ["x", "sigma", "tables", "prior", *HISTORY]
OUTPUTS = ["latent", "removed", "reason", "share"]


def tables(model_sampling, device) -> torch.Tensor:
    # (5, GRID): log sigma (evenly spaced), timestep, and the coefficients of calculate_input (input = c_in * x) and
    # calculate_denoised (denoised = a * x + b * output)
    lo, hi = float(model_sampling.sigma_min), float(model_sampling.sigma_max)
    grid = torch.linspace(math.log(lo / 2), math.log(hi * 2), GRID, dtype=torch.float64, device=device)
    sigma = grid.exp().float()
    ones, zeros = torch.ones(GRID, 1, device=device), torch.zeros(GRID, 1, device=device)
    columns = [
        model_sampling.timestep(sigma),
        model_sampling.calculate_input(sigma, ones),
        model_sampling.calculate_denoised(sigma, zeros, ones),
        model_sampling.calculate_denoised(sigma, ones, zeros),
    ]
    return torch.stack([grid.float()] + [c.float().reshape(-1).to(device) for c in columns])


def _lookup(tables, sigma):
    # (4, n): timestep, c_in, a, b at each sigma, linear in log sigma between table points
    grid = tables[0]
    position = ((sigma.log() - grid[0]) / (grid[1] - grid[0])).clamp(0, GRID - 1.001)
    index = position.floor()
    weight = position - index
    index = index.long()
    lo, hi = tables[1:].index_select(1, index), tables[1:].index_select(1, index + 1)
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
    energy = (r * r).sum((1, 2, 3))
    lag = torch.maximum((r[:, :, 1:] * r[:, :, :-1]).sum((1, 2, 3)).abs(), (r[..., 1:] * r[..., :-1]).sum((1, 2, 3)).abs())
    return lag < WHITE * energy


# run(sample, timesteps) returns the shallow UNet's output rows. sample's rows are blocks of x's images (reForge's
# calc_cond_batch puts each cond's copy of the batch after the other); its first x.shape[1] channels are the latent and
# any further ones the model's concat conditioning (inpainting and edit models). inputs: INPUTS by name.
def shallow(run, sample, timesteps, inputs: dict):
    x, sigma, tables = inputs["x"], inputs["sigma"], inputs["tables"]
    n, c = x.shape[0], x.shape[1]
    _, c_in, a, b = _lookup(tables, sigma)
    blocks = sample.reshape(-1, n, *sample.shape[1:])
    scaled = _view(c_in) * x
    gap = (blocks[:, :, :c].float() - scaled.unsqueeze(0)).abs().amax((0, 2, 3, 4))
    consistent = gap <= GATE * scaled.abs().amax((1, 2, 3))
    columns = []
    for m in range(1, WINDOW + 1):
        columns += [inputs[f"hx{m}"] - inputs[f"hr{m}"], inputs[f"hd{m}"], inputs[f"hu{m}"]]
    noise = _residual(x, columns)
    variance = (noise * noise).mean((1, 2, 3))
    s2 = sigma * sigma
    white = _white(noise)
    intact = inputs["prior"] < 0.5
    below = variance < CEILING * s2
    apply = consistent & intact & white & below
    # When N is nearly all of x's noise (a long wait with an ancestral sampler), x - N is close to the previous
    # denoised value at almost no noise, which the shallow UNet returns nearly as is.
    sigma_eff = (s2 - variance).clamp_min(1e-6 * s2).sqrt()
    t_eff, c_eff, a_eff, b_eff = _lookup(tables, sigma_eff)
    x_eff = x - noise
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
    share = torch.where(white, variance / s2, -torch.ones_like(variance))
    one = torch.ones_like(variance)
    reason = torch.where(apply, 0 * one, torch.where(~consistent, 2 * one, torch.where(~intact, one, torch.where(~white, 3 * one, 4 * one))))
    return latent, removed, reason, share


def dynamic_axes() -> dict:
    image, vector = {0: "images", 2: "height", 3: "width"}, {0: "images"}
    axes = {"x": image, "sigma": vector, "prior": vector, "removed": image, "reason": vector, "share": vector}
    axes.update({name: image for name in HISTORY})
    return axes


def example_inputs(sample, channels: int) -> list:
    rows, _, h, w = sample.shape
    n = max(1, rows // 2)
    kw = {"device": sample.device}
    grid = torch.linspace(-4, 9, GRID, **kw)
    values = {"x": torch.randn(n, channels, h, w, **kw), "sigma": torch.rand(n, **kw) * 10 + 0.1, "prior": torch.zeros(n, **kw)}
    values["tables"] = torch.stack([grid, grid * 100, torch.rsqrt(grid.exp() ** 2 + 1), torch.ones_like(grid), -grid.exp()])
    values.update({name: torch.randn(n, channels, h, w, **kw) for name in HISTORY})
    return [values[name] for name in INPUTS]


def input_shapes(unet_shapes: dict, channels: int) -> dict:
    # Images range from 1 to the profile's most UNet rows: a sampler without uncond sends one row per image, and
    # Tracker.inputs sends one image per row for a call it doesn't track.
    (r_lo, _, h_lo, w_lo), (r_opt, _, h_opt, w_opt), (r_hi, _, h_hi, w_hi) = unet_shapes["sample"]
    i_opt = max(1, r_opt // 2)
    image = [[1, channels, h_lo, w_lo], [i_opt, channels, h_opt, w_opt], [r_hi, channels, h_hi, w_hi]]
    vector = [[1], [i_opt], [r_hi]]
    shapes = {"x": image, "sigma": vector, "tables": [[5, GRID]] * 3, "prior": vector}
    shapes.update({name: image for name in HISTORY})
    return shapes


class Entry:
    # One CFG denoiser call: the sampler's x and sigma, the denoiser's output, the uncond rows' denoised value, and for a
    # step that ran shallow its N and the index of the deep feature's call.
    __slots__ = ("index", "x", "sigma", "denoised", "uncond", "removed", "origin")

    def __init__(self, index, x, sigma):
        self.index, self.x, self.sigma = index, x, sigma
        self.denoised = self.uncond = self.removed = self.origin = None


class Tracker:
    # One sampling pass's state between Speed Hacks engine calls. Calls are numbered by speed.Schedule's index, as deep
    # features' origins are. Statuses (reason, share) of shallow-graph runs are read back without a sync, a call or more
    # later.
    def __init__(self, model_sampling, device, channels: int):
        self.sampling = model_sampling
        self.device = device
        self.channels = channels
        try:
            self.tables = tables(model_sampling, device)
            self.active = True
        except Exception:
            self.tables = torch.zeros(5, GRID, device=device)
            self.active = False
        self.history = deque(maxlen=WINDOW)  # complete Entries, newest last
        self.current = None  # the Entry of the denoiser call in progress
        self.last = None  # (deep feature's call index, call index, reasons) of the latest step that ran shallow
        self.noisy = False  # a fit in this pass found the sampler adding noise
        self.blocked = False  # the latest status read had noise left in by reason 2 to 4 on some image
        self.dead = set()  # indices of deep features whose chain broke
        self.check = None  # (Entry, full output rows, checked rows, age) until the denoiser's output is in
        self._statuses = []  # (reasons and shares in pinned memory, CUDA event, deep feature's call index)
        self._constants = {}

    def _constant(self, shape, value=0.0, dtype=torch.float32):
        key = (tuple(shape), value, dtype)
        if key not in self._constants:
            self._constants[key] = torch.full(shape, value, dtype=dtype, device=self.device)
        return self._constants[key]

    def begin(self, index: int, x, sigma):
        self._collect()
        x = x.float()
        sigma = sigma.float().reshape(-1).expand(x.shape[0])
        newest = self.history[-1] if self.history else None
        if newest is not None and (newest.index != index - 1 or newest.x.shape != x.shape):
            self.history.clear()
        self.current = Entry(index, x, sigma) if self.active else None

    def call(self, sample, padded: bool):
        # The Entry a UNet call belongs to, if its rows are blocks of the denoiser's images (not crops, as sliding window
        # guidance and area conds send, nor a padded batch)
        entry = self.current
        if entry is None or padded:
            return None
        rows, channels, h, w = sample.shape
        n, c = entry.x.shape[:2]
        if rows % n or channels < c or (h, w) != tuple(entry.x.shape[2:]):
            return None
        return entry

    def safe(self, entry, origin) -> bool:
        # Whether a shallow call may run: always on a pass without noise, and with noise only where the fit can remove it
        self._collect()
        if not self.noisy:
            return True
        return entry is not None and not self.blocked and origin not in self.dead and self._prior(entry, origin) is not None

    def _prior(self, entry, origin):
        newest = self.history[-1].index if self.history else None
        if origin is None or newest != entry.index - 1:
            return None
        if self.last is not None and self.last[:2] == (origin, newest):
            return self.last[2]
        return self._constant((entry.x.shape[0],)) if newest == origin else None

    def inputs(self, entry, origin, sample) -> dict:
        if entry is None:
            return self._untracked(sample)
        n, c, h, w = entry.x.shape
        prior = self._prior(entry, origin)
        values = {"x": entry.x, "sigma": entry.sigma, "tables": self.tables, "prior": prior if prior is not None else self._constant((n,), 1.0)}
        zeros = self._constant((n, c, h, w))
        past = list(self.history)[::-1]
        for m in range(1, WINDOW + 1):
            if m <= len(past):
                p = past[m - 1]
                removed = p.removed if p.removed is not None and p.origin == origin else zeros
                values.update({f"hx{m}": p.x, f"hr{m}": removed, f"hd{m}": p.denoised, f"hu{m}": p.uncond if p.uncond is not None else zeros})
            else:
                values.update({f"hx{m}": zeros, f"hr{m}": zeros, f"hd{m}": zeros, f"hu{m}": zeros})
        return values

    def _untracked(self, sample) -> dict:
        # One image per UNet row and prior = 1, for which shallow() returns the plain shallow UNet's output
        rows, _, h, w = sample.shape
        image = self._constant((rows, self.channels, h, w))
        values = {"x": image, "sigma": self._constant((rows,), 1.0), "tables": self.tables, "prior": self._constant((rows,), 1.0)}
        values.update({name: image for name in HISTORY})
        return values

    def after_call(self, entry, out, cond_or_uncond):
        # Every tracked UNet call: the uncond rows' denoised value, which CFG++ samplers step with
        n = entry.x.shape[0]
        if entry.uncond is None and cond_or_uncond and 1 in cond_or_uncond and len(cond_or_uncond) * n == out.shape[0]:
            i = cond_or_uncond.index(1)
            entry.uncond = self.sampling.calculate_denoised(entry.sigma, out[i * n:(i + 1) * n].float(), entry.x)

    def after_shallow(self, entry, outputs: dict, origin):
        if entry.removed is None:
            entry.removed, entry.origin = outputs["removed"], origin
            self.last = (origin, entry.index, outputs["reason"])
        self.status(outputs, origin)

    def status(self, outputs: dict, origin):
        host = torch.empty((2, outputs["share"].shape[0]), dtype=torch.float32, pin_memory=True)
        host[0].copy_(outputs["reason"], non_blocking=True)
        host[1].copy_(outputs["share"], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._statuses.append((host, event, origin))

    def _collect(self):
        while self._statuses and self._statuses[0][1].query():
            host, _, origin = self._statuses.pop(0)
            reasons, shares = host.tolist()
            self.noisy = self.noisy or any(s > NOISY for s in shares)
            self.blocked = any(r >= 2 and (s > NOISY or s < 0) for r, s in zip(reasons, shares))
            if any(r == 1 for r in reasons):
                self.dead.add(origin)

    def finish(self, denoised):
        # After the denoiser returned: completes the step's Entry, and returns a pending check's (error, age) or None
        entry, self.current = self.current, None
        if entry is None:
            return None
        if denoised.shape != entry.x.shape:
            self.history.clear()
            self.check = None
            return None
        entry.denoised = denoised.float()
        self.history.append(entry)
        check, self.check = self.check, None
        if check is None or check[0] is not entry:
            return None
        return self._error(entry, check[1], check[2]), check[3]

    def _error(self, entry, full, checked):
        # A check's error in the denoiser's output, relative to it. How the output depends on each block of UNet rows
        # comes from a least-squares fit of the output on the blocks' denoised values, per image: CFG's 1 - s and s, and
        # for other guidance (APG, sliding window guidance) its local weights. linalg.solve_ex skips the error check, so
        # nothing here waits for the GPU.
        n = entry.x.shape[0]
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
