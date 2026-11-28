# Stateless DiT Speed Hacks math, spectral filtering, predictor transforms, trajectory forecasters, and quality-tiered
# policy primitives shared by `dit_speed.py`. Kept pure (no pass state, no GPU readbacks) so future DiT architectures
# can reuse or extend individual forecasting and risk-gating components cleanly.
import math

import numpy as np
import torch

DEFAULT_METHOD = "auto"  # "auto" | "vde" | "freqca" | "hermite" | "taylor" | "chebyshev" | "constant"
HISTORY_LEN = 5
MIN_STEPS = 10
STALE = 8
MOVED = 0.01
ORDER2_DAMP = 0.5
# Canonical scaled physicist Hermite stability point: 2 * sigma^2 == 1.0 gives exact linear reproduction (0% lag)
# and 0.5 * (s^2 - 1) == 0.0 at s == 1 (zero 2nd-difference noise amplification on single-step skips).
HERMITE_SIGMA = 1.0 / math.sqrt(2.0)
CHEB_RIDGE = 0.05
FREQCA_CUTOFF = 0.20
KALMAN_Q = 0.05
KALMAN_R = 0.05
MAX_CONSECUTIVE_CAP = 6

# Per-(H, W, device) cached normalized radial frequency grid for 2D rfft2 (capped at 32 entries).
_RADIAL_CACHE = {}


def _radial_freq_grid(h: int, w: int, device) -> torch.Tensor:
    key = (h, w, str(device))
    grid = _RADIAL_CACHE.get(key)
    if grid is None:
        if len(_RADIAL_CACHE) >= 32:
            _RADIAL_CACHE.clear()
        fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32) * 2.0
        fx = torch.fft.rfftfreq(w, device=device, dtype=torch.float32) * 2.0
        grid = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).detach()
        _RADIAL_CACHE[key] = grid
    return grid


def _lf_mask(h: int, w: int, cutoff: float, device) -> torch.Tensor:
    r = _radial_freq_grid(h, w, device)
    lo, hi = cutoff * 0.8, cutoff * 1.2
    t = ((r - lo) / max(1e-6, hi - lo)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def wiener_filter_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """SeaCache (CVPR 2026) zero-calibration 2D FFT Wiener filter isolating semantic drift from Gaussian noise."""
    xf = x.float()
    h, w = xf.shape[-2], xf.shape[-1]
    if h < 4 or w < 4:
        return xf
    r = _radial_freq_grid(h, w, xf.device)
    sx = 1.0 / (r * r + 0.04)
    s_clamped = max(0.0, min(0.999, float(sigma)))
    a = max(1e-3, 1.0 - s_clamped)
    b = max(1e-3, s_clamped)
    gain = (a * sx) / (a * a * sx + b * b)
    gain = gain / gain.mean().clamp_min(1e-6)
    return torch.fft.irfft2(torch.fft.rfft2(xf) * gain, s=(h, w))


def _broadcast_sigma(sigma, ref: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(sigma, device=ref.device, dtype=torch.float32).reshape(-1)
    if s.numel() == 1:
        s = s.expand(ref.shape[0])
    elif s.numel() != ref.shape[0]:
        s = s[:1].expand(ref.shape[0])
    return s.view(ref.shape[0], *([1] * (ref.ndim - 1)))


def to_x0(predictor, sigma: torch.Tensor, model_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Converts model output (flow velocity / eps / v_pred) to clean-image prediction x0 in FP32."""
    xf = x.float()
    vf = model_output.float()
    sf = _broadcast_sigma(sigma, xf)
    ptype = getattr(predictor, "prediction_type", "const")
    if ptype == "v_prediction":
        sd = float(getattr(predictor, "sigma_data", 1.0) or 1.0)
        denom = sf * sf + sd * sd
        return xf * (sd * sd / denom) - vf * (sf * sd / denom.sqrt())
    # Both Flow Matching ("const") and "epsilon" use x0 = x - sigma * model_output.
    return xf - vf * sf


def from_x0(predictor, sigma: torch.Tensor, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Inverts clean-image prediction x0 back to model output at current noisy x_t and sigma (fresh-noise safe)."""
    xf = x.float()
    x0f = x0.float()
    sf = _broadcast_sigma(sigma, xf)
    ptype = getattr(predictor, "prediction_type", "const")
    if ptype == "v_prediction":
        sd = float(getattr(predictor, "sigma_data", 1.0) or 1.0)
        denom = sf * sf + sd * sd
        return (xf * (sd * sd / denom) - x0f) * denom.sqrt() / (sf * sd).clamp_min(1e-6)
    return (xf - x0f) / sf.clamp_min(1e-6)


def sigma_from_timesteps(predictor, timesteps: torch.Tensor) -> torch.Tensor:
    t = timesteps.reshape(-1).float()
    mult = float(getattr(predictor, "multiplier", 1.0) or 1.0)
    return t / mult


def vde_decompose(x: torch.Tensor, v: torch.Tensor) -> tuple:
    """Orthogonal velocity decomposition v = alpha * x + beta * ||x|| * u_dir (VDE, arXiv:2605.23381)."""
    xf = x.float()
    vf = v.float()
    dims = tuple(range(1, xf.ndim))
    x_sq = (xf * xf).sum(dim=dims, keepdim=True).clamp_min(1e-12)
    alpha = (vf * xf).sum(dim=dims, keepdim=True) / x_sq
    orth = vf - alpha * xf
    orth_norm = (orth * orth).sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    beta = orth_norm / x_sq.sqrt()
    u_dir = orth / orth_norm
    return alpha, beta, u_dir


def vde_predict(vde_us: list, vde_alphas: list, vde_betas: list, vde_dir: torch.Tensor, x_now: torch.Tensor, u_now: float) -> torch.Tensor:
    """Predicts model output at (x_now, u_now) via linearly extrapolated (alpha, beta) and reused unit direction."""
    u2, u1 = float(vde_us[-1]), float(vde_us[-2])
    du1 = u2 - u1
    s = max(-3.0, min(3.0, (float(u_now) - u2) / du1)) if abs(du1) > 1e-8 else 0.0
    a_last = vde_alphas[-1]
    b_last = vde_betas[-1]
    a_hat = a_last + s * (a_last - vde_alphas[-2])
    b_hat = (b_last + s * (b_last - vde_betas[-2])).clamp_min(0.0).minimum((b_last * 2.5).clamp_min(1e-4))
    xf = x_now.float()
    dims = tuple(range(1, xf.ndim))
    x_norm = (xf * xf).sum(dim=dims, keepdim=True).sqrt()
    return a_hat * xf + b_hat * x_norm * vde_dir


def forecast_weights(
    us: list,
    target_u: float,
    method: str = "hermite",
    *,
    order: int = 2,
    damp: float = ORDER2_DAMP,
    hermite_sigma: float = HERMITE_SIGMA,
    ridge: float = CHEB_RIDGE,
) -> list:
    """Computes affine scalar weights `[w_0, ..., w_{K-1}]` (sum == 1) to forecast y(target_u) from (us, ys)."""
    k = len(us)
    if k == 0:
        raise ValueError("empty history")
    if k == 1 or method == "constant":
        return [0.0] * (k - 1) + [1.0]

    u_last = float(us[-1])
    u_prev = float(us[-2])
    du_prev = u_last - u_prev
    du = float(target_u) - u_last
    if abs(du_prev) < 1e-8 or abs(du) < 1e-12:
        return [0.0] * (k - 1) + [1.0]

    # Cap extrapolation step ratio so abrupt schedule jumps cannot amplify noise.
    step_ratio = du / du_prev
    if abs(step_ratio) > 4.0:
        step_ratio = math.copysign(4.0, step_ratio)
        du = step_ratio * du_prev

    if method == "taylor":
        if k == 2 or order <= 1:
            w = [0.0] * k
            w[-1] = 1.0 + step_ratio
            w[-2] = -step_ratio
            return w
        u_prev2 = float(us[-3])
        du_prev2 = u_prev - u_prev2
        du_span = u_last - u_prev2
        if abs(du_prev2) < 1e-8 or abs(du_span) < 1e-8:
            w = [0.0] * k
            w[-1] = 1.0 + step_ratio
            w[-2] = -step_ratio
            return w
        du1 = du_prev
        du0 = du_prev2
        c2 = damp * du * (du + du1) / du_span
        max_c2 = 1.5 * abs(du1)
        if abs(c2) > max_c2:
            c2 = math.copysign(max_c2, c2)
        c1 = du / du1
        w = [0.0] * k
        w[-1] = 1.0 + c1 + c2 / du1
        w[-2] = -c1 - c2 * (1.0 / du1 + 1.0 / du0)
        w[-3] = c2 / du0
        return w

    if method in ("hermite", "hicache", "freqca", "auto"):
        # True scaled physicist Hermite basis (HiCache, ICLR 2026) on non-uniform sigma grids:
        # H1_tilde(s) = 2 * sigma^2 * s,  H2_tilde(s) = (sigma^2 / 2) * (4 * sigma^2 * s^2 - 2) = 2 * sigma^4 * s^2 - sigma^2.
        # At sigma = 1/sqrt(2): H1_tilde(s) = s (exact linear reproduction), H2_tilde(1) = 0 (zero 2nd-order noise at s=1).
        s = step_ratio
        s2 = hermite_sigma * hermite_sigma
        c1 = 2.0 * s2 * s
        w = [0.0] * k
        if k == 2 or order <= 1:
            w[-1] = 1.0 + c1
            w[-2] = -c1
            return w
        u_prev2 = float(us[-3])
        du0 = u_prev - u_prev2
        if abs(du0) < 1e-8:
            w[-1] = 1.0 + c1
            w[-2] = -c1
            return w
        r = max(0.25, min(4.0, du_prev / du0))
        c2_raw = 2.0 * (s2 * s2) * (s * s) - s2
        c2 = max(-0.5, min(2.5, c2_raw)) * (2.0 / (1.0 + r))
        # Delta_1 = y2 - y1, Delta_2 = (y2 - y1) - r * (y1 - y0) (identically 0 on any linear trajectory).
        w[-1] = 1.0 + c1 + c2
        w[-2] = -c1 - c2 * (1.0 + r)
        w[-3] = c2 * r
        return w

    if method == "chebyshev":
        u_arr = np.asarray(us, dtype=np.float64)
        u_min, u_max = float(u_arr.min()), float(u_arr.max())
        span = u_max - u_min
        if span < 1e-8:
            return [0.0] * (k - 1) + [1.0]
        tau = 2.0 * (u_arr - u_min) / span - 1.0
        t_tgt = 2.0 * (float(target_u) - u_min) / span - 1.0
        t_tgt = max(-2.5, min(2.5, t_tgt))
        deg = min(order, k - 1, 2)
        A = np.zeros((k, deg + 1), dtype=np.float64)
        phi = np.zeros(deg + 1, dtype=np.float64)
        A[:, 0] = 1.0
        phi[0] = 1.0
        if deg >= 1:
            A[:, 1] = tau
            phi[1] = t_tgt
        if deg >= 2:
            A[:, 2] = 2.0 * tau * tau - 1.0
            phi[2] = 2.0 * t_tgt * t_tgt - 1.0
        pen = np.zeros((deg + 1, deg + 1), dtype=np.float64)
        for j in range(1, deg + 1):
            pen[j, j] = ridge * (j * j)
        ata = A.T @ A + pen
        try:
            coeffs = np.linalg.solve(ata, A.T)
            w = (phi @ coeffs).tolist()
            s_w = sum(w)
            if abs(s_w) > 1e-8:
                w = [v / s_w for v in w]
            return w
        except np.linalg.LinAlgError:
            return [0.0] * (k - 1) + [1.0]

    raise ValueError(f"Unknown DiT forecast method {method!r}")


def eval_linear_combination(tensors: list, weights: list) -> torch.Tensor:
    """Evaluates sum_i w_i * tensors[i] in FP32 with minimal GPU kernel launches (single GEMV for K >= 2)."""
    active = [(t, w) for t, w in zip(tensors, weights) if abs(w) > 1e-12]
    if not active:
        return tensors[-1].float()
    if len(active) == 1:
        t, w = active[0]
        return t.float() if abs(w - 1.0) < 1e-7 else t.float() * w
    shape = active[0][0].shape
    dev = active[0][0].device
    stack = torch.stack([t.float().reshape(-1) for t, _ in active], dim=0)
    w_t = torch.tensor([w for _, w in active], device=dev, dtype=torch.float32)
    return torch.matmul(w_t, stack).reshape(shape)


def freqca_forecast(us: list, ys: list, target_u: float, *, order: int = 2, cutoff: float = FREQCA_CUTOFF) -> torch.Tensor:
    """FreqCa (arXiv:2510.08669): directly reuses low spatial frequencies of x0 and forecasts high frequencies."""
    y_last = ys[-1].float()
    if len(ys) < 2:
        return y_last
    w = forecast_weights(us, target_u, method="hermite", order=order)
    y_fc = eval_linear_combination(ys, w)
    h, w_dim = y_last.shape[-2], y_last.shape[-1]
    if h < 4 or w_dim < 4:
        return y_fc
    m_lf = _lf_mask(h, w_dim, cutoff, y_last.device)
    f_last = torch.fft.rfft2(y_last)
    f_fc = torch.fft.rfft2(y_fc)
    return torch.fft.irfft2(m_lf * f_last + (1.0 - m_lf) * f_fc, s=(h, w_dim))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def tier_policy(quality: float) -> dict:
    """Smooth continuous quality-tiered policy across quality in [0.0, 1.0], calibrated to real Flow DiTs.

    - Tier 1 [0.85, 1.00]: Near-lossless dual VDE/FreqCa skips; conservative tail drop below q=0.98; no FasterCache.
    - Tier 2 [0.65, 0.85): High Quality (default q=0.80); adds late Limited-Interval Guidance & guarded FasterCache.
    - Tier 3 [0.40, 0.65): Fast / Balanced; smoothly widens Wiener + tolerance + FasterCache + Adaptive Guidance.
    - Tier 4 [0.00, 0.40): Draft / Turbo; convex ramp eliminates the q=0.2->0.0 cliff and unlocks a usable 2.5-3.2x
      preview band at q=0.15-0.35 while keeping anchor refreshes (consec_cap <= 6).
    """
    q = max(0.0, min(1.0, float(quality)))
    if q >= 0.85:
        t = (1.0 - q) / 0.15
        return {
            "tolerance": _lerp(0.06, 0.15, t),
            "consec_cap": 1 if q >= 0.96 else 2,
            "tail_uncond_drop": 0.0 if q >= 0.98 else _lerp(0.0, 0.13, t),
            "fc_max_consec": 0,
            "fc_start": 1.0,
            "fc_tol": 0.0,
            "adaptive_cos": 0.0,
            "wiener_thresh": _lerp(0.20, 0.25, t),
        }
    if q >= 0.65:
        t = (0.85 - q) / 0.20
        return {
            "tolerance": _lerp(0.15, 0.27, t),
            "consec_cap": 3,
            "tail_uncond_drop": _lerp(0.13, 0.29, t),
            "fc_max_consec": 1,
            "fc_start": _lerp(0.38, 0.26, t),
            "fc_tol": _lerp(0.60, 0.88, t),
            "adaptive_cos": 0.0,
            "wiener_thresh": _lerp(0.25, 0.34, t),
        }
    if q >= 0.40:
        t = (0.65 - q) / 0.25
        return {
            "tolerance": _lerp(0.27, 0.48, t),
            "consec_cap": 3 if q >= 0.55 else 4,
            "tail_uncond_drop": _lerp(0.29, 0.38, t),
            "fc_max_consec": 2,
            "fc_start": _lerp(0.26, 0.20, t),
            "fc_tol": _lerp(0.88, 1.25, t),
            "adaptive_cos": _lerp(0.97, 0.945, t),
            "wiener_thresh": _lerp(0.34, 0.56, t),
        }
    t = (0.40 - q) / 0.40
    t_convex = t * (0.55 + 0.45 * t)
    return {
        "tolerance": math.inf if q <= 0.0 else _lerp(0.48, 1.80, t_convex),
        "consec_cap": 4 if q > 0.28 else (5 if q > 0.12 else MAX_CONSECUTIVE_CAP),
        "tail_uncond_drop": _lerp(0.38, 0.46, t),
        "fc_max_consec": 3 if q > 0.15 else 4,
        "fc_start": _lerp(0.20, 0.15, t),
        "fc_tol": math.inf if q <= 0.0 else _lerp(1.25, 2.40, t_convex),
        "adaptive_cos": _lerp(0.945, 0.90, t),
        "wiener_thresh": math.inf if q <= 0.10 else _lerp(0.56, 1.15, (0.40 - q) / 0.30),
    }


def dit_tolerance(quality: float) -> float:
    """Base closed-loop x0 forecast drift tolerance for quality in [0, 1] (modulated by propagation gain g(r))."""
    return tier_policy(quality)["tolerance"]


def propagation_gain(progress: float) -> float:
    """RA-CFGCache / EpaCache analytic local-to-final error propagation gain g(r): tighter early, looser late."""
    rem = max(0.05, 1.0 - max(0.0, min(1.0, float(progress))))
    return 0.35 + 1.45 * (rem ** 1.2)


def _kalman_update(prev_val, prev_var: float, obs: float) -> tuple:
    if prev_val is None:
        return obs, 0.10
    p_pred = prev_var + KALMAN_Q
    k = p_pred / (p_pred + KALMAN_R)
    return (1.0 - k) * prev_val + k * obs, (1.0 - k) * p_pred
