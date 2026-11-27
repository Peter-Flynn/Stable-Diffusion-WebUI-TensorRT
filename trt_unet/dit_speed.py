# DiT-native Speed Hacks: quality-tiered, zero-extra-engine acceleration for Anima & MOSS-Image.
#
# Combines three empirically validated, zero-rebuild SOTA layers across the single `quality` slider (0.0–1.0):
#   1. Self-Selecting Dual Forecaster ("auto"): tracks both VDE (input-adaptive orthogonal velocity decomposition,
#      arXiv:2605.23381) and FreqCa (2D FFT low-frequency reuse + scaled physicist Hermite / HiCache high-frequency
#      forecasting in clean-image x0-space, arXiv:2510.08669 & arXiv:2508.16984) via dual 1D NaviCache Kalman error
#      trackers (arXiv:2606.26795), automatically selecting whichever fits the current model, sampler, and noise regime
#      best (VDE excels on MOSS ODE; FreqCa in x0-space excels on Anima and keeps ancestral/SDE samplers fresh-noise safe).
#   2. Closed-Loop Risk Gating: RA-CFGCache / EpaCache analytic propagation gain g(r) (tighter early, looser late),
#      SeaCache 2D FFT Wiener latent drift gate (arXiv:2602.18993), and GuideCache spatial guidance-share weighting
#      (arXiv:2611.25382) so rendered text/signage regions automatically tighten error checks without slowing scenic prompts.
#   3. Orthogonal CFG-Branch Acceleration (quality-gated): late-step Limited-Interval Guidance (u = c, exact w_eff = 1.0,
#      active even inside tail full-eval steps, arXiv:2404.07724), mid-to-late FasterCache CFG-Cache (u_hat = c_t + Delta,
#      canceling common-mode scene drift via fresh c_t, arXiv:2410.19355) with Coupled skip_all, and Adaptive Guidance cosine
#      cutoff in lower quality tiers. Strictly execution-order guarded: requires fresh c_t from the current step.
import bisect
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import speed
from .common import Readback

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


def _broadcast_sigma(sigma, ref: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(sigma, device=ref.device, dtype=torch.float32).reshape(-1)
    if s.numel() == 1:
        s = s.expand(ref.shape[0])
    elif s.numel() != ref.shape[0]:
        s = s[:1].expand(ref.shape[0])
    return s.view(ref.shape[0], *([1] * (ref.ndim - 1)))


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
    - Tier 3 [0.40, 0.65): Fast / Balanced; widens Limited-Interval + FasterCache + Adaptive Guidance cosine cutoff.
    - Tier 4 [0.00, 0.40): Draft / Turbo; pushes up to ~3.5-4.0x while keeping anchor refreshes (consec_cap <= 6) so
      images stay sharp instead of collapsing into blur.
    """
    q = max(0.0, min(1.0, float(quality)))
    if q >= 0.85:
        t = (1.0 - q) / 0.15
        return {
            "tolerance": _lerp(0.06, 0.145, t),
            "consec_cap": 1 if q >= 0.96 else 2,
            "tail_uncond_drop": 0.0 if q >= 0.98 else _lerp(0.0, 0.12, t),
            "fc_max_consec": 0,
            "fc_start": 1.0,
            "fc_tol": 0.0,
            "adaptive_cos": 0.0,
            "wiener_thresh": _lerp(0.20, 0.23, t),
        }
    if q >= 0.65:
        t = (0.85 - q) / 0.20
        return {
            "tolerance": _lerp(0.145, 0.255, t),
            "consec_cap": 3,
            "tail_uncond_drop": _lerp(0.12, 0.28, t),
            "fc_max_consec": 1,
            "fc_start": _lerp(0.40, 0.28, t),
            "fc_tol": _lerp(0.55, 0.82, t),
            "adaptive_cos": 0.0,
            "wiener_thresh": _lerp(0.23, 0.28, t),
        }
    if q >= 0.40:
        t = (0.65 - q) / 0.25
        return {
            "tolerance": _lerp(0.255, 0.36, t),
            "consec_cap": 4,
            "tail_uncond_drop": _lerp(0.28, 0.36, t),
            "fc_max_consec": 2,
            "fc_start": _lerp(0.28, 0.22, t),
            "fc_tol": _lerp(0.82, 1.08, t),
            "adaptive_cos": _lerp(0.97, 0.95, t),
            "wiener_thresh": _lerp(0.28, 0.38, t),
        }
    t = (0.40 - q) / 0.40
    return {
        "tolerance": math.inf if q <= 0.0 else _lerp(0.36, 0.62, t),
        "consec_cap": 5 if q > 0.15 else MAX_CONSECUTIVE_CAP,
        "tail_uncond_drop": _lerp(0.36, 0.45, t),
        "fc_max_consec": 3,
        "fc_start": _lerp(0.22, 0.15, t),
        "fc_tol": math.inf if q <= 0.0 else _lerp(1.08, 1.75, t),
        "adaptive_cos": _lerp(0.95, 0.92, t),
        "wiener_thresh": math.inf if q <= 0.10 else _lerp(0.38, 0.55, t),
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


class _BranchState:
    __slots__ = (
        "us",
        "ys",
        "prompt",
        "prompt_norm",
        "last_full_index",
        "last_seen_index",
        "last_full_u",
        "last_err",
        "last_err_du",
        "consec_skips",
        "accum_err",
        # Dual 1D NaviCache Kalman error-rate trackers (per unit delta-sigma)
        "kf_primary",
        "kf_primary_var",
        "kf_vde",
        "kf_vde_var",
        # VDE orthogonal state (1 extra latent-sized unit direction tensor)
        "vde_us",
        "vde_alphas",
        "vde_betas",
        "vde_dir",
        # SeaCache 2D FFT Wiener latent anchor
        "wiener_x",
        "wiener_norm",
    )

    def __init__(self):
        self.us = []
        self.ys = []
        self.prompt = None
        self.prompt_norm = None
        self.last_full_index = -1
        self.last_seen_index = -1
        self.last_full_u = None
        self.last_err = None
        self.last_err_du = None
        self.consec_skips = 0
        self.accum_err = 0.0
        self.kf_primary = None
        self.kf_primary_var = 0.25
        self.kf_vde = None
        self.kf_vde_var = 0.25
        self.vde_us = []
        self.vde_alphas = []
        self.vde_betas = []
        self.vde_dir = None
        self.wiener_x = None
        self.wiener_norm = None

    def reset(self):
        self.us.clear()
        self.ys.clear()
        self.prompt = None
        self.prompt_norm = None
        self.last_full_index = -1
        self.last_full_u = None
        self.last_err = None
        self.last_err_du = None
        self.consec_skips = 0
        self.accum_err = 0.0
        self.kf_primary = None
        self.kf_primary_var = 0.25
        self.kf_vde = None
        self.kf_vde_var = 0.25
        self.vde_us.clear()
        self.vde_alphas.clear()
        self.vde_betas.clear()
        self.vde_dir = None
        self.wiener_x = None
        self.wiener_norm = None


def _skey_of(bkey: tuple) -> tuple:
    """Canonical CFG-pair key (occ, chunk, spatial, prompt_ids) from branch key."""
    return (bkey[1], bkey[2], bkey[3], bkey[5])


def _same_latent(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact latent identity check (O(1) pointer fast-path when unsplit; exact tensor equality when sliced)."""
    if a is b:
        return True
    if a.shape != b.shape or a.device != b.device:
        return False
    if a.data_ptr() == b.data_ptr() and a.stride() == b.stride():
        return True
    return bool(torch.equal(a, b))


class DiTSchedule:
    """Quality-tiered closed-loop DiT Speed Hacks scheduler (self-selecting VDE/FreqCa + orthogonal CFG stack)."""

    def __init__(
        self,
        predictor,
        warmup: float = speed.WARMUP,
        tail: float = speed.TAIL,
        quality: float = speed.QUALITY,
        method: str = None,
        order: int = 2,
        enable_cfg_tier1: bool = True,
    ):
        self.predictor = predictor
        self.warmup, self.tail, self.quality = warmup, tail, quality
        self.method = DEFAULT_METHOD if method is None else method
        self.order = order
        self.enable_cfg_tier1 = enable_cfg_tier1

        pol = tier_policy(quality)
        self.policy = pol
        self.tolerance = pol["tolerance"]
        self.consec_cap = pol["consec_cap"]
        # Respect users who explicitly raise `Full end` (tail) above the 5% default by trimming tail_uncond_drop.
        extra_tail = max(0.0, float(tail) - speed.TAIL)
        self.tail_uncond_drop = max(0.0, pol["tail_uncond_drop"] - extra_tail) if enable_cfg_tier1 else 0.0
        self.fc_max_consec = pol["fc_max_consec"] if enable_cfg_tier1 else 0
        self.fc_start = pol["fc_start"]
        self.fc_tol = pol["fc_tol"]
        self.adaptive_cos = pol["adaptive_cos"] if enable_cfg_tier1 else 0.0
        self.wiener_thresh = pol["wiener_thresh"]

        self.index = None
        self.region = False
        self.idle = False
        self.done = False
        self.sigmas = None
        self._levels = None
        self._undecided = None
        self._step = None
        self._start = None
        self._anchor = None
        self.per = None
        self._progress = 0.0
        self._total_steps = 20
        self._cond_scale = 5.0
        self._asymmetric_cfg = False
        self._step_multi_cond = False
        self._mismatched_latents = False
        self._has_side_call = False
        self._step_used_cfg_shortcut = False
        self._prev_u = None
        self._step_u = None

        self.branches = {}  # branch_key -> _BranchState
        self._tag_counts = {}
        self._step_skip = {}  # group_key -> bool (per-tile / per-image / multi-cond phase-lock)
        self.volatile = set()
        self._volatile_skeys = set()
        self._volatile_gkeys = set()
        self._moved = {}
        self._readback = Readback()

        # Orthogonal CFG acceleration & GuideCache text-protection state
        self._fresh_cond = {}  # skey -> (step_index, out_tensor, is_real_full, x_tensor)
        self._last_uncond = {}  # skey -> (step_index, out_tensor, x_tensor)
        self._cfg_delta = {}  # skey -> u - c (FP32)
        self._cfg_delta_kf = {}  # skey -> (rate_est, variance)
        self._cfg_delta_consec = {}  # skey -> consecutive FasterCache reuses
        self._cfg_cos = {}  # skey -> cos(c, u)
        self._guide_weight_map = {}  # skey -> spatial guidance-share weight (mean ~ 1.0)

        # Counters compatible with runtime.py's end() reporting & infotext
        self.full_calls = 0
        self.shallow_calls = 0  # full-step skips (skip_all, 0 TRT calls)
        self.cfg_unbatched_calls = 0  # unbatched UNCOND calls saved via Limited-Interval / FasterCache (0 TRT calls)
        self.cfg_batched_calls = 0  # batched [COND, UNCOND] calls run at batch=1 (50% FLOPs)
        self.checks = []  # (index, age, rel_err)

    @property
    def tier1_calls(self) -> int:
        return self.cfg_unbatched_calls + self.cfg_batched_calls

    def note_multi_cond(self):
        """Flags composable/multi-cond prompts (e.g. `A AND B`) so orthogonal CFG shortcuts stay disabled."""
        self._step_multi_cond = True
        if not self._asymmetric_cfg:
            self._asymmetric_cfg = True
            self._clear_all_cfg()

    def note_side_call(self):
        """Disables skipping when an out-of-profile sibling branch ran on PyTorch in coexist mode."""
        self._has_side_call = True
        self._asymmetric_cfg = True
        self._step_skip.clear()
        self._clear_all_cfg()

    def _group_key(self, bkey: tuple) -> tuple:
        """Phase-lock group key: omits `occ` and `chunk` under asymmetric/multi-cond CFG so all branches lock together."""
        if self._asymmetric_cfg or self._step_multi_cond:
            return (bkey[3], bkey[5])
        return _skey_of(bkey)

    def _cfg_shortcuts_allowed(self) -> bool:
        return (
            self.enable_cfg_tier1
            and not self.idle
            and self._cond_scale > 1.01
            and not self._asymmetric_cfg
            and not self._step_multi_cond
            and not self._mismatched_latents
            and not self._has_side_call
        )

    def tail_cfg_active(self) -> bool:
        """Whether Limited-Interval Guidance can still drop UNCOND during tail (respecting user-raised Full end)."""
        return self._cfg_shortcuts_allowed() and self.tail_uncond_drop > 0.0

    def begin(self, index: int, total, steps, step, sigma, cond_scale=None):
        # Detect asymmetric multi-cond / Perp-Neg (tag>=2) / mismatched chunk partitions from prior step.
        if self._tag_counts:
            asym_before = self._asymmetric_cfg
            if any(t not in (0, 1) for t in self._tag_counts):
                self._asymmetric_cfg = True
            c_cnt = self._tag_counts.get(0, 0)
            u_cnt = self._tag_counts.get(1, 0)
            if c_cnt > 0 and u_cnt > 0:
                if c_cnt != u_cnt:
                    self._asymmetric_cfg = True
                elif not self._step_used_cfg_shortcut:
                    real_cond = {k for k, v in self._fresh_cond.items() if v[2] and k not in self._volatile_skeys}
                    real_uncond = {k for k in self._last_uncond if k not in self._volatile_skeys}
                    if real_cond and real_uncond and real_cond != real_uncond:
                        self._mismatched_latents = True
                        self._asymmetric_cfg = True
            if self._asymmetric_cfg and not asym_before:
                self._clear_all_cfg()
        self.index = index
        if self._step_u is not None:
            self._prev_u = self._step_u
        self._step_u = None
        self._tag_counts.clear()
        self._step_skip.clear()
        self._fresh_cond.clear()
        self._last_uncond.clear()
        while len(self._cfg_delta) > 8:
            self._clear_skey_cfg(next(iter(self._cfg_delta)))
        self._step_multi_cond = False
        self._step_used_cfg_shortcut = False
        if cond_scale is not None:
            try:
                if isinstance(cond_scale, torch.Tensor):
                    if cond_scale.numel():
                        self._cond_scale = float(cond_scale.max().item())
                else:
                    self._cond_scale = float(cond_scale)
            except (TypeError, ValueError):
                pass
        if steps:
            self._total_steps = int(steps)
        prog_step = (step + 1 if index else 0) / max(1, self._total_steps)
        if self._levels is None and total:
            prog_step = max(prog_step, index / max(1, int(total)))
        self._progress = min(1.0, max(0.0, prog_step))
        self._collect()
        self._follow(index, step)
        if self._levels is None and self.sigmas is not None and steps and len(self.sigmas) > steps:
            self._levels = [-v for v in self.sigmas[-(steps + 1) :]]
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
        self.region = False
        self._undecided = None if self.idle or self.done else (total, steps, step, sigma)
        if self.done:
            self.branches.clear()
            self._clear_all_cfg()
        else:
            for k in [k for k, st in self.branches.items() if index - st.last_seen_index > STALE]:
                del self.branches[k]

    def _follow(self, index, step):
        if self._step is None:
            self._step, self._start = step, index
        elif step != self._step:
            self._step = step
            if self._anchor is not None and index - 1 > self._anchor:
                self.per = index - 1 - self._anchor
            self._anchor = index - 1

    def _ensure_decided(self):
        if self._undecided is None:
            return
        total, steps, step, sigma = self._undecided
        self._undecided = None
        if self._levels is None and self.sigmas is not None and len(self.sigmas) > steps:
            self._levels = [-v for v in self.sigmas[-(steps + 1) :]]
        if self._levels is not None:
            t = torch.as_tensor(sigma)
            sig = float(t) if t.numel() == 1 else float(t.max())
            n = len(self._levels) - 1
            start, end = round(self.warmup * n), n - max(1, round(self.tail * n))
            position = max(bisect.bisect_right(self._levels, -sig * (1 - 1e-4)) - 1, step + 1 if self.index else 0)
            self.region = start <= position < end
            self._progress = min(1.0, max(0.0, position / max(1, n)))
        else:
            per = max(1, total // steps)
            start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
            self.region = start <= self.index < end

    def _clear_skey_cfg(self, skey):
        self._cfg_delta.pop(skey, None)
        self._cfg_delta_kf.pop(skey, None)
        self._cfg_delta_consec.pop(skey, None)
        self._cfg_cos.pop(skey, None)
        self._guide_weight_map.pop(skey, None)

    def _clear_all_cfg(self):
        self._cfg_delta.clear()
        self._cfg_delta_kf.clear()
        self._cfg_delta_consec.clear()
        self._cfg_cos.clear()
        self._guide_weight_map.clear()

    def _collect(self):
        for tag, ((value,),) in self._readback.ready():
            kind = tag[0]
            if kind == "err":
                _, bkey, idx, du, age = tag
                val = float(value)
                st = self.branches.get(bkey)
                if st is not None:
                    st.last_err = val
                    st.last_err_du = float(du)
                    rate = val / max(1e-4, float(du))
                    st.kf_primary, st.kf_primary_var = _kalman_update(st.kf_primary, st.kf_primary_var, rate)
                self.checks.append((idx, age, val))
            elif kind == "err_vde":
                _, bkey, du = tag
                st = self.branches.get(bkey)
                if st is not None:
                    rate = float(value) / max(1e-4, float(du))
                    st.kf_vde, st.kf_vde_var = _kalman_update(st.kf_vde, st.kf_vde_var, rate)
            elif kind == "delta_err":
                _, skey, prog = tag
                # Ignore high-noise step-0/step-1 jumps so early transients never lock out mid-pass FasterCache.
                if prog >= 0.15 and skey not in self._volatile_skeys:
                    prev_est, prev_var = self._cfg_delta_kf.get(skey, (None, 0.25))
                    self._cfg_delta_kf[skey] = _kalman_update(prev_est, prev_var, float(value))
            elif kind == "cfg_cos":
                _, skey, prog = tag
                # Ignore early high-noise cosine similarity (where shared noise makes cos(c, u) artificially high).
                if prog >= 0.50 and skey not in self._volatile_skeys:
                    self._cfg_cos[skey] = float(value)
            elif kind == "move":
                _, bkey, origin = tag
                if value <= MOVED:
                    continue
                skey = _skey_of(bkey)
                prev = self._moved.get(bkey)
                if prev is None or prev[1] != origin:
                    if prev is not None and self.index is not None and self.index - prev[0] <= STALE:
                        self.volatile.add(bkey)
                        self._volatile_skeys.add(skey)
                        self._volatile_gkeys.add((bkey[3], bkey[5]))
                        self._clear_skey_cfg(skey)
                    self._moved[bkey] = (self.index, origin)
                st = self.branches.get(bkey)
                if st is not None and st.last_full_index == origin:
                    st.reset()
                    self._clear_skey_cfg(skey)

    def _effective_tol(self, base_tol: float) -> float:
        if base_tol <= 0.0 or base_tol == math.inf:
            return base_tol
        return base_tol / propagation_gain(self._progress)

    def _active_forecaster(self, st: _BranchState, bkey: tuple = None) -> str:
        """Selects VDE vs FreqCa in 'auto' mode using dual Kalman rates, phase-locked across CFG siblings."""
        if self.method != "auto":
            return self.method
        states = [st]
        if bkey is not None:
            gkey = self._group_key(bkey)
            for k, other in self.branches.items():
                if k != bkey and self._group_key(k) == gkey and self.index is not None and self.index - other.last_seen_index <= STALE:
                    states.append(other)
        # Pick VDE only when ALL active CFG sibling branches are VDE-ready and VDE matches/beats FreqCa jointly,
        # guaranteeing COND and UNCOND (and all multi-cond terms) never mix forecaster bases inside (c_hat - u_hat).
        if all(len(s.vde_us) >= 2 and s.vde_dir is not None and s.kf_vde is not None for s in states):
            vde_sum = sum(s.kf_vde for s in states)
            pri_vals = [s.kf_primary for s in states if s.kf_primary is not None]
            if len(pri_vals) < len(states) or vde_sum <= sum(pri_vals) * 1.02:
                return "vde"
        return "freqca"

    def _best_rate(self, st: _BranchState, bkey: tuple = None):
        """Returns the Kalman error rate for the exact forecaster that _active_forecaster will execute."""
        if self._active_forecaster(st, bkey) == "vde":
            return st.kf_vde
        return st.kf_primary

    def _pred_step_err(self, st: _BranchState, du: float, bkey: tuple = None) -> float:
        rate = self._best_rate(st, bkey)
        if rate is not None:
            return rate * du
        ratio = max(0.25, du / st.last_err_du)
        return st.last_err * (ratio ** 1.5)

    def _wiener_ok(self, st: _BranchState, x_slice: torch.Tensor, u_now: float) -> bool:
        if self.wiener_thresh == math.inf or st.wiener_x is None or st.wiener_norm is None:
            return True
        if st.wiener_x.shape != x_slice.shape:
            return True
        wx = wiener_filter_latent(x_slice, u_now)
        drift = float((torch.dist(wx, st.wiener_x) / st.wiener_norm).item())
        return drift <= self._effective_tol(self.wiener_thresh)

    def _branch_volatile(self, bkey: tuple) -> bool:
        if bkey in self.volatile or _skey_of(bkey) in self._volatile_skeys:
            return True
        if (self._asymmetric_cfg or self._step_multi_cond) and (bkey[3], bkey[5]) in self._volatile_gkeys:
            return True
        return False

    def _is_upward_jump(self, u_now: float, st: _BranchState = None) -> bool:
        if self._prev_u is not None and u_now > self._prev_u + 1e-7:
            return True
        return st is not None and bool(st.us) and u_now > st.us[-1] + 1e-7

    def _branch_can_skip(self, bkey: tuple, is_uncond: bool, u_now: float, x_slice=None) -> bool:
        if not self.region or self.idle or self.done or self._mismatched_latents or self._has_side_call or self._branch_volatile(bkey):
            return False
        st = self.branches.get(bkey)
        if st is None or len(st.ys) < 2 or st.last_full_u is None:
            return False
        # Do not forecast across upward sigma jumps (e.g. s_churn / Dy / SMEA) or zero-delta repeat probes.
        if self._is_upward_jump(u_now, st):
            return False
        du = st.last_full_u - u_now
        if du <= 1e-7:
            return False
        if st.consec_skips >= self.consec_cap:
            return False
        if self.tolerance == math.inf:
            return True
        if x_slice is not None and not self._wiener_ok(st, x_slice, u_now):
            return False
        tol = self._effective_tol(self.tolerance)
        if self._best_rate(st, bkey) is None and (st.last_err is None or st.last_err_du is None or st.last_err_du <= 1e-8):
            return len(st.ys) >= 3 and st.consec_skips == 0
        step_err = self._pred_step_err(st, du, bkey)
        return (st.accum_err + step_err) <= tol

    def _can_tail_drop_uncond(self, skey=None) -> bool:
        if not self._cfg_shortcuts_allowed() or self.tail_uncond_drop <= 0.0 or skey in self._volatile_skeys:
            return False
        # Strictly respect user's Full start (warmup): only fire in active region or tail (done).
        if not (self.region or self.done):
            return False
        return self._progress >= (1.0 - self.tail_uncond_drop) - 1e-6

    def _can_adaptive_drop_uncond(self, skey) -> bool:
        # Strictly confined to active region (respects both Full start and user-raised Full end) & non-volatile prompts.
        if (
            not self._cfg_shortcuts_allowed()
            or not self.region
            or self.done
            or self.adaptive_cos <= 0.0
            or self._progress < 0.50
            or skey in self._volatile_skeys
        ):
            return False
        c = self._cfg_cos.get(skey)
        return c is not None and c >= self.adaptive_cos

    def _can_fastercache_uncond(self, skey) -> bool:
        if (
            not self._cfg_shortcuts_allowed()
            or self.done
            or not self.region
            or self.fc_max_consec <= 0
            or self._progress < self.fc_start - 1e-6
            or skey in self._volatile_skeys
        ):
            return False
        if skey not in self._cfg_delta:
            return False
        if self._cfg_delta_consec.get(skey, 0) >= self.fc_max_consec:
            return False
        if self.fc_tol < math.inf:
            kf = self._cfg_delta_kf.get(skey)
            if kf is not None and kf[0] is not None and kf[0] > self._effective_tol(self.fc_tol):
                return False
        return True

    def _check_prompt_move(self, bkey: tuple, st: _BranchState, prompt_tensor: torch.Tensor) -> bool:
        if st.prompt is None or prompt_tensor is None:
            return True
        if st.prompt.shape != prompt_tensor.shape:
            st.reset()
            self._clear_skey_cfg(_skey_of(bkey))
            return False
        if st.prompt_norm is None:
            st.prompt_norm = st.prompt.float().norm().clamp_min(1e-30)
        moved = torch.dist(prompt_tensor.float(), st.prompt.float()) / st.prompt_norm
        self._readback.send(("move", bkey, st.last_full_index), moved)
        return True

    def plan_call(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, cond_or_uncond, prompt_ids: tuple) -> dict:
        """Plans how to handle a DiT call across unbatched ([0]/[1]) and batched ([0, 1]) CFG layouts."""
        if self.index is None:
            return {"mode": "full", "groups": [], "u": 0.0, "b": x.shape[0], "skey": None}
        self._ensure_decided()
        b = x.shape[0]
        u_now = float(torch.as_tensor(sigma).reshape(-1)[0].item())
        self._step_u = u_now
        spatial = tuple(x.shape[1:])
        tags = list(cond_or_uncond) if cond_or_uncond else [0]
        n_groups = len(tags)
        if n_groups == 0 or b % n_groups != 0:
            tags = [0]
            n_groups = 1
        chunk = b // n_groups

        groups = []
        occs = []
        upward_jump = self._is_upward_jump(u_now)
        for gi, tag in enumerate(tags):
            start, end = gi * chunk, (gi + 1) * chunk
            itag = int(tag)
            if itag not in (0, 1):
                # 3+ conditioning tag layouts (e.g. Perp-Neg [cond, uncond, empty]) disable 2-branch CFG shortcuts.
                if not self._asymmetric_cfg:
                    self._asymmetric_cfg = True
                    self._clear_all_cfg()
            occ = self._tag_counts.get(itag, 0)
            self._tag_counts[itag] = occ + 1
            occs.append(occ)
            is_uncond = itag == 1
            bkey = (itag, occ, chunk, spatial, tuple(context.shape[1:]), prompt_ids)
            st = self.branches.get(bkey)
            if st is not None:
                st.last_seen_index = self.index
                if self._is_upward_jump(u_now, st):
                    upward_jump = True
            can_skip = False if self.done else self._branch_can_skip(bkey, is_uncond, u_now, x_slice=x[start:end])
            groups.append((slice(start, end), bkey, is_uncond, can_skip))

        # Key CFG state by per-tile/per-image occurrence `occ` AND verify exact latent identity before pairing.
        occ_match = len(set(occs)) == 1
        skey = (occs[0], chunk, spatial, prompt_ids) if occ_match else None
        if n_groups == 1 and groups[0][2] and skey is not None:
            fc_chk = self._fresh_cond.get(skey)
            if fc_chk is not None and fc_chk[0] == self.index and not _same_latent(x[0:chunk], fc_chk[3]):
                # Cross-image partition mismatch (e.g. batch_size >= 3 variable-length prompts or area crops): disable pairing!
                self._mismatched_latents = True
                self._asymmetric_cfg = True
                self._clear_all_cfg()

        batched_1to1 = (
            occ_match
            and n_groups == 2
            and not self._asymmetric_cfg
            and not self._step_multi_cond
            and not self._mismatched_latents
            and not self._has_side_call
            and sum(1 for g in groups if g[1][0] == 0) == 1
            and sum(1 for g in groups if g[1][0] == 1) == 1
            and _same_latent(x[groups[0][0]], x[groups[1][0]])
        )
        pair_cfg_ok = occ_match and (n_groups == 1 or batched_1to1)

        tail_drop = pair_cfg_ok and not upward_jump and (self._can_tail_drop_uncond(skey) or self._can_adaptive_drop_uncond(skey))
        fc_ok = pair_cfg_ok and not upward_jump and self._can_fastercache_uncond(skey)

        for gi, (slc, bkey, is_u, can_skip) in enumerate(groups):
            st = self.branches.get(bkey)
            if (can_skip or fc_ok) and st is not None:
                if not self._check_prompt_move(bkey, st, context[slc]):
                    groups[gi] = (slc, bkey, is_u, False)
        if fc_ok and skey not in self._cfg_delta:
            fc_ok = False

        # Phase-lock full-skip per group key across ALL split calls (unbatched MOSS, multi-cond A AND B, Perp-Neg).
        if not batched_1to1 and not self.done:
            group_keys = {g[1] for g in groups}
            for gi, (slc, bkey, is_u, can_skip) in enumerate(groups):
                gkey = self._group_key(bkey)
                if gkey not in self._step_skip:
                    if n_groups == 1 and bkey[0] == 0 and (tail_drop or fc_ok):
                        self._step_skip[gkey] = can_skip
                    else:
                        same_g_ok = all(g[3] for g in groups if self._group_key(g[1]) == gkey)
                        siblings = [
                            k for k, st in self.branches.items()
                            if k not in group_keys
                            and self._group_key(k) == gkey
                            and self.index - st.last_seen_index <= STALE
                        ]
                        # Require matching chunk size so sibling Wiener drift is evaluated on the exact same batch slice.
                        self._step_skip[gkey] = same_g_ok and all(
                            k[2] == chunk
                            and self._branch_can_skip(
                                k, k[0] == 1, u_now, x_slice=x[slc] if not self._mismatched_latents else None
                            )
                            for k in siblings
                        )
                if not self._step_skip[gkey]:
                    groups[gi] = (slc, bkey, is_u, False)
                elif n_groups == 1 and is_u and skey is not None and (tail_drop or fc_ok):
                    fc = self._fresh_cond.get(skey)
                    if fc is not None and fc[0] == self.index and _same_latent(x[slc], fc[3]):
                        groups[gi] = (slc, bkey, is_u, True)

        # 1. Full step skip (0 TRT calls), including Coupled FasterCache / tail_drop for 1:1 batched [COND, UNCOND].
        if not self.done and not self._mismatched_latents and not self._has_side_call:
            if all(g[3] for g in groups):
                return {"mode": "skip_all", "groups": groups, "u": u_now, "b": b, "skey": skey, "tail_drop": tail_drop, "fc_ok": fc_ok}
            if batched_1to1 and (tail_drop or fc_ok):
                cond_groups = [g for g in groups if g[1][0] == 0]
                if len(cond_groups) == 1 and cond_groups[0][3]:
                    return {"mode": "skip_all", "groups": groups, "u": u_now, "b": b, "skey": skey, "tail_drop": tail_drop, "fc_ok": fc_ok}

        # 2. Orthogonal CFG Acceleration (Limited-Interval Guidance OR FasterCache CFG-Cache)
        # Unbatched UNCOND (e.g. MOSS-Image): strictly requires REAL full COND from THIS exact step with IDENTICAL latent!
        if n_groups == 1 and groups[0][2] and skey is not None:
            fc = self._fresh_cond.get(skey)
            if fc is not None and fc[0] == self.index and fc[2] and _same_latent(x[0:chunk], fc[3]):
                if tail_drop:
                    self._step_used_cfg_shortcut = True
                    return {"mode": "unbatched_cfg_drop", "groups": groups, "u": u_now, "b": b, "skey": skey}
                if fc_ok:
                    self._step_used_cfg_shortcut = True
                    return {"mode": "unbatched_fastercache", "groups": groups, "u": u_now, "b": b, "skey": skey}

        # Batched 1:1 [COND, UNCOND] (e.g. Anima): run batch=1 COND on TRT, synthesize UNCOND from fresh COND.
        if batched_1to1:
            if tail_drop:
                self._step_used_cfg_shortcut = True
                return {"mode": "batched_cfg_drop", "groups": groups, "u": u_now, "b": b, "skey": skey}
            if fc_ok:
                self._step_used_cfg_shortcut = True
                return {"mode": "batched_fastercache", "groups": groups, "u": u_now, "b": b, "skey": skey}

        return {"mode": "full", "groups": groups, "u": u_now, "b": b, "skey": skey}

    def synthesize_uncond(self, skey, mode: str, cond_out: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
        """Synthesizes UNCOND from fresh COND via Limited-Interval (u=c) or FasterCache (u=c + Delta_cfg)."""
        if mode in ("unbatched_cfg_drop", "batched_cfg_drop"):
            return cond_out.to(x_dtype)
        # Exact unboosted residual u_hat = c_t + (u_a - c_a) cancels common-mode scene drift inside CFG!
        u_hat = cond_out.float() + self._cfg_delta[skey]
        self._cfg_delta_consec[skey] = self._cfg_delta_consec.get(skey, 0) + 1
        return u_hat.to(x_dtype)

    def forecast_step(self, plan: dict, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Executes a skip_all plan (forecasting COND first so UNCOND can couple via tail_drop / FasterCache)."""
        skey = plan["skey"]
        tail_drop = bool(plan.get("tail_drop", False))
        fc_ok = bool(plan.get("fc_ok", False))
        groups = plan["groups"]
        u_now = plan["u"]
        assembled = [None] * len(groups)
        can_couple = skey is not None and (
            len(groups) == 1 or (len(groups) == 2 and sum(1 for g in groups if g[1][0] == 0) == 1 and sum(1 for g in groups if g[1][0] == 1) == 1)
        )

        c_hat = None
        c_x = None
        for gi, (slc, bkey, is_u, _) in enumerate(groups):
            if not is_u:
                pred_out = self.forecast_slice(bkey, x[slc], sigma[slc], u_now)
                assembled[gi] = pred_out
                if bkey[0] == 0 and can_couple:
                    c_hat = pred_out
                    c_x = x[slc]
                    # Mark is_real_full=False so record_full can NEVER mistake forecasted c_hat for a full eval!
                    self._fresh_cond[skey] = (self.index, c_hat.detach(), False, c_x.detach())
        for gi, (slc, bkey, is_u, _) in enumerate(groups):
            if is_u:
                fc_entry = self._fresh_cond.get(skey) if can_couple else None
                if can_couple and c_hat is not None and _same_latent(x[slc], c_x):
                    c_ref = c_hat
                elif fc_entry is not None and fc_entry[0] == self.index and _same_latent(x[slc], fc_entry[3]):
                    c_ref = fc_entry[1]
                else:
                    c_ref = None
                if c_ref is not None and tail_drop:
                    assembled[gi] = c_ref.to(x.dtype)
                elif c_ref is not None and fc_ok and skey in self._cfg_delta:
                    assembled[gi] = self.synthesize_uncond(skey, "unbatched_fastercache", c_ref, x.dtype)
                else:
                    assembled[gi] = self.forecast_slice(bkey, x[slc], sigma[slc], u_now)
        return assembled[0] if len(assembled) == 1 else torch.cat(assembled, dim=0)

    def forecast_slice(self, bkey: tuple, x_slice: torch.Tensor, sigma_slice: torch.Tensor, u_now: float) -> torch.Tensor:
        """Forecasts model output for a skipped branch slice using the active forecaster (auto VDE/FreqCa or explicit)."""
        st = self.branches[bkey]
        active = self._active_forecaster(st, bkey)
        if active == "vde" and len(st.vde_us) >= 2 and st.vde_dir is not None:
            v_pred = vde_predict(st.vde_us, st.vde_alphas, st.vde_betas, st.vde_dir, x_slice, u_now)
        elif active == "freqca":
            x0_pred = freqca_forecast(st.us, st.ys, u_now, order=self.order)
            v_pred = from_x0(self.predictor, sigma_slice, x0_pred, x_slice)
        else:
            weights = forecast_weights(st.us, u_now, method=active, order=self.order)
            x0_pred = eval_linear_combination(st.ys, weights)
            v_pred = from_x0(self.predictor, sigma_slice, x0_pred, x_slice)

        du = max(1e-7, (st.last_full_u or u_now) - u_now)
        if self._best_rate(st, bkey) is not None or (st.last_err is not None and st.last_err_du and st.last_err_du > 1e-8):
            st.accum_err += self._pred_step_err(st, du, bkey)
        st.consec_skips += 1
        return v_pred.to(x_slice.dtype)

    def _weighted_rel_err(self, pred: torch.Tensor, target: torch.Tensor, skey) -> torch.Tensor:
        wmap = self._guide_weight_map.get(skey)
        if wmap is not None and wmap.shape[-2:] == target.shape[-2:] and wmap.shape[0] == target.shape[0]:
            num = ((pred - target).pow(2) * wmap).sum().sqrt()
            den = (target.pow(2) * wmap).sum().sqrt().clamp_min(1e-6)
            return num / den
        return torch.dist(pred, target) / target.norm().clamp_min(1e-6)

    def record_full(self, bkey: tuple, x_slice: torch.Tensor, sigma_slice: torch.Tensor, out_slice: torch.Tensor, context_slice: torch.Tensor, u_now: float):
        """Records a full evaluation, updates dual Kalman & CFG-Cache state, and runs free closed-loop error checks."""
        if self.index is None or bkey in self.volatile:
            return
        tag = bkey[0]
        skey = _skey_of(bkey)
        if tag == 0:
            self._fresh_cond[skey] = (self.index, out_slice.detach(), True, x_slice.detach())
        elif tag == 1:
            self._last_uncond[skey] = (self.index, out_slice.detach(), x_slice.detach())

        # During tail (self.done), we only need fresh_cond recorded for unbatched tail CFG drop!
        if self.done:
            return

        st = self.branches.get(bkey)
        if st is None:
            st = self.branches[bkey] = _BranchState()

        # Strict upward sigma jump resets stale trajectory & pre-jump CFG cache BEFORE storing fresh post-jump CFG state.
        if self._is_upward_jump(u_now, st):
            st.us.clear()
            st.ys.clear()
            st.vde_us.clear()
            st.vde_alphas.clear()
            st.vde_betas.clear()
            st.vde_dir = None
            self._clear_skey_cfg(skey)

        # When BOTH COND (tag==0) and UNCOND (tag==1) ran REAL full evals on this step WITH IDENTICAL LATENT, update CFG state.
        if tag in (0, 1) and skey not in self._volatile_skeys and not self._asymmetric_cfg and not self._step_multi_cond and not self._mismatched_latents:
            fc = self._fresh_cond.get(skey)
            fu = self._last_uncond.get(skey)
            if fc is not None and fu is not None and fc[0] == self.index and fc[2] and fu[0] == self.index:
                if not _same_latent(fc[3], fu[2]):
                    self._mismatched_latents = True
                    self._asymmetric_cfg = True
                    self._clear_all_cfg()
                else:
                    c_out = fc[1].float()
                    u_out = fu[1].float()
                    delta = u_out - c_out
                    prev_delta = self._cfg_delta.get(skey)
                    if prev_delta is not None and prev_delta.shape == delta.shape:
                        d_err = self._weighted_rel_err(prev_delta, delta, skey)
                        self._readback.send(("delta_err", skey, self._progress), d_err)
                    if self.adaptive_cos > 0.0 and self._progress >= 0.50:
                        cos_val = F.cosine_similarity(c_out.flatten(1), u_out.flatten(1), dim=1).min()
                        self._readback.send(("cfg_cos", skey, self._progress), cos_val)
                    if skey not in self._cfg_delta and len(self._cfg_delta) >= 8:
                        # Never evict a tile whose COND already ran on this step and whose UNCOND is still pending.
                        victim = next((k for k in self._cfg_delta if k not in self._fresh_cond or k in self._last_uncond), None)
                        if victim is not None:
                            self._clear_skey_cfg(victim)
                    self._cfg_delta[skey] = delta
                    self._cfg_delta_consec[skey] = 0
                    # GuideCache (arXiv:2611.25382): spatial guidance-share map boosts error weight on high-guidance text/faces.
                    if self._cond_scale > 1.01:
                        w_cfg = self._cond_scale
                        guided = u_out + w_cfg * (c_out - u_out)
                        cdim = tuple(range(1, c_out.ndim - 2))
                        g_mag = (w_cfg * (c_out - u_out)).pow(2).sum(dim=cdim, keepdim=True).sqrt()
                        v_mag = guided.pow(2).sum(dim=cdim, keepdim=True).sqrt().clamp_min(1e-4)
                        rho = g_mag / v_mag
                        rho = rho / rho.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                        self._guide_weight_map[skey] = 1.0 + 0.75 * (rho - 1.0).clamp_min(0.0)

        target = to_x0(self.predictor, sigma_slice, out_slice, x_slice)

        # Free closed-loop accuracy checks (dual checks in "auto" mode: FreqCa + VDE, both GuideCache-weighted).
        if len(st.ys) >= 2 and st.last_full_u is not None and (st.last_full_u - u_now) > 1e-7:
            du = st.last_full_u - u_now
            age = st.consec_skips + 1
            idx_now = self.index
            if self.method in ("auto", "freqca"):
                pred_primary = freqca_forecast(st.us, st.ys, u_now, order=self.order)
                rel_primary = self._weighted_rel_err(pred_primary, target, skey)
                self._readback.send(("err", bkey, idx_now, du, age), rel_primary)
            elif self.method != "vde":
                weights = forecast_weights(st.us, u_now, method=self.method, order=self.order)
                pred_primary = eval_linear_combination(st.ys, weights)
                rel_primary = self._weighted_rel_err(pred_primary, target, skey)
                self._readback.send(("err", bkey, idx_now, du, age), rel_primary)

            if self.method in ("auto", "vde") and len(st.vde_us) >= 2 and st.vde_dir is not None:
                v_vde = vde_predict(st.vde_us, st.vde_alphas, st.vde_betas, st.vde_dir, x_slice, u_now)
                pred_vde = to_x0(self.predictor, sigma_slice, v_vde, x_slice)
                rel_vde = self._weighted_rel_err(pred_vde, target, skey)
                if self.method == "vde":
                    self._readback.send(("err", bkey, idx_now, du, age), rel_vde)
                self._readback.send(("err_vde", bkey, du), rel_vde)

        # Equal-sigma repeat (Heun) updates in place; otherwise append (upward jumps were already cleared above).
        if st.us and abs(u_now - st.us[-1]) <= 1e-7:
            st.us[-1] = u_now
            st.ys[-1] = target
        else:
            st.us.append(u_now)
            st.ys.append(target)
            if len(st.us) > HISTORY_LEN:
                st.us.pop(0)
                st.ys.pop(0)

        # Update VDE orthogonal state
        if self.method in ("auto", "vde"):
            alpha, beta, u_dir = vde_decompose(x_slice, out_slice)
            if st.vde_us and abs(u_now - st.vde_us[-1]) <= 1e-7:
                st.vde_us[-1], st.vde_alphas[-1], st.vde_betas[-1] = u_now, alpha, beta
            else:
                st.vde_us.append(u_now)
                st.vde_alphas.append(alpha)
                st.vde_betas.append(beta)
                if len(st.vde_us) > 3:
                    st.vde_us.pop(0)
                    st.vde_alphas.pop(0)
                    st.vde_betas.pop(0)
            st.vde_dir = u_dir

        # Update SeaCache 2D FFT Wiener latent anchor
        if self.wiener_thresh < math.inf:
            wx = wiener_filter_latent(x_slice, u_now)
            st.wiener_x = wx
            st.wiener_norm = wx.norm().clamp_min(1e-6)

        st.prompt = context_slice.detach()
        st.prompt_norm = None
        st.last_full_index = self.index
        st.last_seen_index = st.last_full_index
        st.last_full_u = u_now
        st.consec_skips = 0
        st.accum_err = 0.0

    def note_counts(self, mode: str):
        if mode == "skip_all":
            self.shallow_calls += 1
        elif mode in ("unbatched_cfg_drop", "unbatched_fastercache"):
            self.cfg_unbatched_calls += 1
        elif mode in ("batched_cfg_drop", "batched_fastercache", "split_cfg"):
            self.cfg_batched_calls += 1
            self.full_calls += 1
        else:
            self.full_calls += 1

    def release(self):
        self.branches.clear()
        self._tag_counts.clear()
        self._step_skip.clear()
        self._readback.clear()
        self._fresh_cond.clear()
        self._last_uncond.clear()
        self._clear_all_cfg()
        self._prev_u = None
        self._step_u = None
