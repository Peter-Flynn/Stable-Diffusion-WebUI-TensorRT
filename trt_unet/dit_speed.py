# DiT-native Speed Hacks: closed-loop x0-space trajectory forecasting (damped Hermite / Taylor / Chebyshev)
# for Anima & MOSS-Image with zero extra engine or VRAM overhead.
#
# Forecasting in clean-image x0-space (x0 = predictor.calculate_denoised(sigma, v, x)) and inverting back to velocity
# (v_pred = (x_t - x0_pred) / sigma) makes skips automatically fresh-noise-safe on ancestral and SDE samplers (Euler a,
# DPM++ 2M SDE, etc.): 100% of freshly injected noise in x_t is attributed to velocity with zero ghosting or blur.
# Parameterizing by actual sigma handles arbitrary non-uniform schedules (Beta, Karras, SGM Uniform, Shifted Flow).
import bisect
import math

import numpy as np
import torch

from . import speed
from .common import Readback

DEFAULT_METHOD = "hermite"  # "constant" | "taylor" | "hermite" | "chebyshev"
HISTORY_LEN = 5
MIN_STEPS = 10
STALE = 8
MOVED = 0.01
# Uncond tracks smoother trajectories than cond; independent drift tracking naturally triggers CFG Tier 1
# (exact cond at half batch + forecasted uncond) whenever cond drifts faster than uncond.
UNCOND_TOL_MULT = 1.35
ORDER2_DAMP = 0.5
HERMITE_SIGMA = 0.70
CHEB_RIDGE = 0.05
# Hard cap on consecutive skipped evaluations per branch even at low quality (quality=0 still skips full middle).
MAX_CONSECUTIVE_CAP = 6


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


def forecast_weights(
    us: list,
    target_u: float,
    method: str = DEFAULT_METHOD,
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
            # Linear: y_last + step_ratio * (y_last - y_prev)
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
        # Second-order divided-difference Newton polynomial with stability damping on curvature term:
        # y(u) = y2 + du * (y2 - y1)/du1 + damp * du * (u - u1) * [(y2 - y1)/du1 - (y1 - y0)/du0] / (u2 - u0)
        du1 = du_prev
        du0 = du_prev2
        # Limit effective second-order leverage for large extrapolation steps.
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

    if method == "hermite":
        # HiCache-inspired damped Hermite contraction in normalized step s = du / du_prev.
        s = step_ratio
        w = [0.0] * k
        if k == 2 or order <= 1:
            c1 = hermite_sigma * s
            w[-1] = 1.0 + c1
            w[-2] = -c1
            return w
        u_prev2 = float(us[-3])
        du0 = u_prev - u_prev2
        if abs(du0) < 1e-8:
            c1 = hermite_sigma * s
            w[-1] = 1.0 + c1
            w[-2] = -c1
            return w
        # Scale 2nd difference by step-spacing ratio so non-uniform schedules remain consistent.
        r = du_prev / du0
        c1 = hermite_sigma * s
        c2 = (hermite_sigma ** 2) * r * s * (s + 1.0) / (1.0 + r)
        # Delta_1 = y2 - y1, Delta_2 = (y2 - y1) - r * (y1 - y0)
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
        # Map observed points to [-1, 1]; clamp target tau to avoid runaway polynomial extrapolation.
        tau = 2.0 * (u_arr - u_min) / span - 1.0
        t_tgt = 2.0 * (float(target_u) - u_min) / span - 1.0
        t_tgt = max(-2.5, min(2.5, t_tgt))
        deg = min(order, k - 1, 2)
        # Chebyshev basis T_0(t)=1, T_1(t)=t, T_2(t)=2t^2-1
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
        # Zero penalty on T_0 guarantees sum(w) == 1 (exact reproduction of constant fields).
        pen = np.zeros((deg + 1, deg + 1), dtype=np.float64)
        for j in range(1, deg + 1):
            pen[j, j] = ridge * (j * j)
        ata = A.T @ A + pen
        try:
            coeffs = np.linalg.solve(ata, A.T)  # (deg+1, k)
            w = (phi @ coeffs).tolist()
            # Enforce exact affine sum == 1 against tiny float rounding.
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
    # Stack into (K, N) and run one matmul (1, K) @ (K, N) -> ~0.05 ms on GPU.
    shape = active[0][0].shape
    dev = active[0][0].device
    stack = torch.stack([t.float().reshape(-1) for t, _ in active], dim=0)
    w_t = torch.tensor([w for _, w in active], device=dev, dtype=torch.float32)
    return torch.matmul(w_t, stack).reshape(shape)


def dit_tolerance(quality: float) -> float:
    """Relative L2 x0 forecast drift tolerance for quality in [0, 1]."""
    if quality >= 1.0:
        return 0.0
    if quality <= 0.0:
        return math.inf
    # Calibrated around DiT x0 relative L2 errors (typically 0.005–0.06 per step in mid-trajectory):
    # q=0.95 -> ~0.015, q=0.9 -> ~0.025, q=0.8 -> ~0.045, q=0.6 -> ~0.095, q=0.4 -> ~0.165.
    return 0.09 * ((1.0 - quality) / quality) ** 0.75


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


class DiTSchedule:
    """Adaptive closed-loop DiT Speed Hacks scheduler with phase-locked CFG and x0-space trajectory forecasting."""

    def __init__(
        self,
        predictor,
        warmup: float = speed.WARMUP,
        tail: float = speed.TAIL,
        quality: float = speed.QUALITY,
        method: str = None,
        order: int = 2,
        enable_cfg_tier1: bool = False,
    ):
        self.predictor = predictor
        self.warmup, self.tail, self.quality = warmup, tail, quality
        self.method = DEFAULT_METHOD if method is None else method
        self.order = order
        self.enable_cfg_tier1 = enable_cfg_tier1
        self.tolerance = dit_tolerance(quality)

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

        self.branches = {}  # branch_key -> _BranchState
        self._tag_counts = {}  # per-step tag occurrence counters (cleared in begin())
        self._step_skip = None  # phase-locked skip decision across unbatched CFG calls in the same step
        self.volatile = set()
        self._moved = {}
        self._readback = Readback()

        # Counters compatible with runtime.py's end() reporting & infotext
        self.full_calls = 0
        self.shallow_calls = 0  # full-skip (Tier 2) or shallow-engine calls
        self.tier1_calls = 0  # CFG-split calls (cond full, uncond forecasted)
        self.checks = []  # (index, consec_or_age, rel_err)

    def begin(self, index: int, total, steps, step, sigma):
        self.index = index
        self._tag_counts.clear()
        self._step_skip = None
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
        else:
            per = max(1, total // steps)
            start, end = round(self.warmup * total), total - max(per, round(self.tail * total))
            self.region = start <= self.index < end

    def _collect(self):
        for tag, ((value,),) in self._readback.ready():
            kind = tag[0]
            if kind == "err":
                _, bkey, idx, du, age = tag
                st = self.branches.get(bkey)
                if st is not None:
                    st.last_err = float(value)
                    st.last_err_du = float(du)
                self.checks.append((idx, age, float(value)))
            elif kind == "move":
                _, bkey, origin = tag
                if value <= MOVED:
                    continue
                prev = self._moved.get(bkey)
                if prev is None or prev[1] != origin:
                    if prev is not None and self.index is not None and self.index - prev[0] <= STALE:
                        self.volatile.add(bkey)
                    self._moved[bkey] = (self.index, origin)
                st = self.branches.get(bkey)
                if st is not None and st.last_full_index == origin:
                    st.reset()

    def _branch_can_skip(self, bkey: tuple, is_uncond: bool, u_now: float) -> bool:
        if not self.region or self.idle or self.done or bkey in self.volatile:
            return False
        st = self.branches.get(bkey)
        if st is None or len(st.ys) < 2 or st.last_full_u is None:
            return False
        # Do not forecast across upward sigma jumps (e.g. s_churn / Dy / SMEA) or zero-delta repeat probes.
        du = st.last_full_u - u_now
        if du <= 1e-7:
            return False
        if self.tolerance == math.inf:
            return True
        if self.tolerance <= 0.0:
            # At quality == 1.0: allow only conservative single-step skip when history >= 3 and measured error < 1%.
            return (
                len(st.ys) >= 3
                and st.consec_skips == 0
                and st.last_err is not None
                and st.last_err <= 0.01
            )
        if st.consec_skips >= MAX_CONSECUTIVE_CAP:
            return False
        tol = self.tolerance * (UNCOND_TOL_MULT if (is_uncond and self.enable_cfg_tier1) else 1.0)
        if st.last_err is None or st.last_err_du is None or st.last_err_du <= 1e-8:
            # Before first measured error is collected, allow 1 probe skip once we have >= 3 full points so the
            # following full step measures true forecast accuracy and calibrates the controller.
            need_pts = 2 if (is_uncond and self.enable_cfg_tier1) else 3
            return len(st.ys) >= need_pts and st.consec_skips == 0
        ratio = max(0.25, du / st.last_err_du)
        # Forecast error grows ~O(ratio^1.5) for order-1/2 extrapolation.
        step_err = st.last_err * (ratio ** 1.5)
        return (st.accum_err + step_err) <= tol

    def _check_prompt_move(self, bkey: tuple, st: _BranchState, prompt_tensor: torch.Tensor) -> bool:
        if st.prompt is None or prompt_tensor is None:
            return True
        if st.prompt.shape != prompt_tensor.shape:
            st.reset()
            return False
        if st.prompt_norm is None:
            st.prompt_norm = st.prompt.float().norm().clamp_min(1e-30)
        moved = torch.dist(prompt_tensor.float(), st.prompt.float()) / st.prompt_norm
        self._readback.send(("move", bkey, st.last_full_index), moved)
        return True

    def plan_call(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, cond_or_uncond, prompt_ids: tuple) -> dict:
        """Plans how to handle a DiT call, supporting unbatched ([0]/[1]) and batched ([0, 1]) CFG layouts.

        When enable_cfg_tier1 is False (default), cond and uncond branches are phase-locked so common-mode forecast
        error cancels inside `uncond + cfg * (cond - uncond)` rather than being amplified by cfg_scale.
        """
        self._ensure_decided()
        b = x.shape[0]
        u_now = float(torch.as_tensor(sigma).reshape(-1)[0].item())
        spatial = tuple(x.shape[1:])
        tags = list(cond_or_uncond) if cond_or_uncond else [0]
        n_groups = len(tags)
        if n_groups == 0 or b % n_groups != 0:
            tags = [0]
            n_groups = 1
        chunk = b // n_groups

        groups = []
        for gi, tag in enumerate(tags):
            start, end = gi * chunk, (gi + 1) * chunk
            itag = int(tag)
            occ = self._tag_counts.get(itag, 0)
            self._tag_counts[itag] = occ + 1
            is_uncond = itag == 1
            # Include cond/uncond tag, per-step tag occurrence, batch chunk, spatial shape, context shape, and prompt ids.
            bkey = (itag, occ, chunk, spatial, tuple(context.shape[1:]), prompt_ids)
            st = self.branches.get(bkey)
            if st is not None:
                st.last_seen_index = self.index if self.index is not None else st.last_seen_index
            can_skip = self._branch_can_skip(bkey, is_uncond, u_now)
            if can_skip and st is not None:
                if not self._check_prompt_move(bkey, st, context[start:end]):
                    can_skip = False
            groups.append((slice(start, end), bkey, is_uncond, can_skip))

        # Phase-lock unbatched CFG calls (e.g. MOSS-Image cond and uncond in separate calls at the same step):
        # skip only if all sibling branches active on the previous step agree to skip.
        if not self.enable_cfg_tier1 and n_groups == 1:
            if self._step_skip is None:
                cur_key = groups[0][1]
                siblings = [
                    k for k, st in self.branches.items()
                    if k != cur_key
                    and k[3] == spatial
                    and k[5] == prompt_ids
                    and self.index is not None
                    and st.last_seen_index >= self.index - 1
                ]
                self._step_skip = groups[0][3] and all(self._branch_can_skip(k, k[0] == 1, u_now) for k in siblings)
            if not self._step_skip:
                slc, bkey, is_u, _ = groups[0]
                groups[0] = (slc, bkey, is_u, False)

        if all(g[3] for g in groups):
            mode = "skip_all"
        elif self.enable_cfg_tier1 and n_groups > 1 and any(g[3] for g in groups) and not all(g[3] for g in groups):
            mode = "split_cfg"
        else:
            mode = "full"

        return {"mode": mode, "groups": groups, "u": u_now, "b": b}

    def forecast_slice(self, bkey: tuple, x_slice: torch.Tensor, sigma_slice: torch.Tensor, u_now: float) -> torch.Tensor:
        """Forecasts model output for a skipped branch slice in x0-space and inverts to velocity."""
        st = self.branches[bkey]
        weights = forecast_weights(st.us, u_now, method=self.method, order=self.order)
        x0_pred = eval_linear_combination(st.ys, weights)
        v_pred = from_x0(self.predictor, sigma_slice, x0_pred, x_slice)
        # Update accumulated drift estimate for closed-loop capping.
        du = max(1e-7, (st.last_full_u or u_now) - u_now)
        if st.last_err is not None and st.last_err_du and st.last_err_du > 1e-8:
            ratio = max(0.25, du / st.last_err_du)
            st.accum_err += st.last_err * (ratio ** 1.5)
        st.consec_skips += 1
        return v_pred.to(x_slice.dtype)

    def record_full(self, bkey: tuple, x_slice: torch.Tensor, sigma_slice: torch.Tensor, out_slice: torch.Tensor, context_slice: torch.Tensor, u_now: float):
        """Records a full evaluation in x0-space and measures free closed-loop forecast error."""
        if bkey in self.volatile:
            return
        st = self.branches.get(bkey)
        if st is None:
            st = self.branches[bkey] = _BranchState()

        target = to_x0(self.predictor, sigma_slice, out_slice, x_slice)

        # Free closed-loop accuracy check: compare what forecasting WOULD have predicted at u_now against true target.
        if len(st.ys) >= 2 and st.last_full_u is not None and (st.last_full_u - u_now) > 1e-7:
            weights = forecast_weights(st.us, u_now, method=self.method, order=self.order)
            pred = eval_linear_combination(st.ys, weights)
            denom = target.norm().clamp_min(1e-6)
            rel_err = torch.dist(pred, target) / denom
            du = st.last_full_u - u_now
            age = st.consec_skips + 1
            self._readback.send(("err", bkey, self.index if self.index is not None else 0, du, age), rel_err)

        # Strict upward sigma jump resets history; equal-sigma repeat (e.g. Heun corrector -> next step) updates in place.
        if st.us and u_now > st.us[-1] + 1e-7:
            st.us.clear()
            st.ys.clear()
            st.us.append(u_now)
            st.ys.append(target)
        elif st.us and abs(u_now - st.us[-1]) <= 1e-7:
            st.us[-1] = u_now
            st.ys[-1] = target
        else:
            st.us.append(u_now)
            st.ys.append(target)
            if len(st.us) > HISTORY_LEN:
                st.us.pop(0)
                st.ys.pop(0)
        st.prompt = context_slice.detach()
        st.prompt_norm = None
        st.last_full_index = self.index if self.index is not None else 0
        st.last_seen_index = st.last_full_index
        st.last_full_u = u_now
        st.consec_skips = 0
        st.accum_err = 0.0

    def note_counts(self, mode: str):
        if mode == "skip_all":
            self.shallow_calls += 1
        elif mode == "split_cfg":
            self.tier1_calls += 1
            self.full_calls += 1
        else:
            self.full_calls += 1

    def release(self):
        self.branches.clear()
        self._tag_counts.clear()
        self._readback.clear()
