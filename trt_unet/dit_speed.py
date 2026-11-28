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

import torch
import torch.nn.functional as F

from . import speed
from .common import Readback
from .dit_math import (  # noqa: F401 — re-exported for callers & compatibility
    CHEB_RIDGE,
    DEFAULT_METHOD,
    FREQCA_CUTOFF,
    HERMITE_SIGMA,
    HISTORY_LEN,
    KALMAN_Q,
    KALMAN_R,
    MAX_CONSECUTIVE_CAP,
    MIN_STEPS,
    MOVED,
    ORDER2_DAMP,
    STALE,
    _RADIAL_CACHE,
    _broadcast_sigma,
    _kalman_update,
    _lerp,
    _lf_mask,
    _radial_freq_grid,
    dit_tolerance,
    eval_linear_combination,
    forecast_weights,
    freqca_forecast,
    from_x0,
    propagation_gain,
    sigma_from_timesteps,
    tier_policy,
    to_x0,
    vde_decompose,
    vde_predict,
    wiener_filter_latent,
)


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


class _CfgCache:
    """Encapsulates per-tile / per-image CFG-pairing state (FasterCache, Adaptive Guidance, GuideCache)."""

    __slots__ = (
        "fresh_cond",    # skey -> (step_index, out_tensor, is_real_full, x_tensor)
        "last_uncond",   # skey -> (step_index, out_tensor, x_tensor)
        "delta",         # skey -> u - c (FP32)
        "delta_kf",      # skey -> (rate_est, variance)
        "delta_consec",  # skey -> consecutive FasterCache reuses
        "cos",           # skey -> cos(c, u)
        "guide_weight",  # skey -> spatial guidance-share weight (mean ~ 1.0)
        "boost_skeys",   # immutable per-step skeys eligible for boosted FasterCache
    )

    def __init__(self):
        self.fresh_cond = {}
        self.last_uncond = {}
        self.delta = {}
        self.delta_kf = {}
        self.delta_consec = {}
        self.cos = {}
        self.guide_weight = {}
        self.boost_skeys = set()

    def clear_skey(self, skey):
        # Intentionally preserves fresh_cond / last_uncond for same-step partition mismatch checks.
        self.delta.pop(skey, None)
        self.delta_kf.pop(skey, None)
        self.delta_consec.pop(skey, None)
        self.cos.pop(skey, None)
        self.guide_weight.pop(skey, None)
        self.boost_skeys.discard(skey)

    def clear_all(self):
        self.delta.clear()
        self.delta_kf.clear()
        self.delta_consec.clear()
        self.cos.clear()
        self.guide_weight.clear()
        self.boost_skeys.clear()

    def clear_step(self):
        self.fresh_cond.clear()
        self.last_uncond.clear()
        self.boost_skeys.clear()
        while len(self.delta) > 8:
            self.clear_skey(next(iter(self.delta)))

    def release(self):
        self.fresh_cond.clear()
        self.last_uncond.clear()
        self.clear_all()

    def fresh_cond_for(self, skey, step_index: int, require_real_full: bool = True, x_slice=None):
        entry = self.fresh_cond.get(skey)
        if entry is None or entry[0] != step_index:
            return None
        if require_real_full and not entry[2]:
            return None
        if x_slice is not None and not _same_latent(x_slice, entry[3]):
            return None
        return entry[1]

    def update_pair(
        self,
        skey,
        step_index: int,
        progress: float,
        cond_scale: float,
        adaptive_cos: float,
        readback,
        weight_err_fn,
    ) -> bool:
        """Updates CFG delta / GuideCache / readbacks when both real COND & UNCOND ran on this step.

        Returns False if COND and UNCOND latents mismatch (caller must latch asymmetric/mismatched state).
        """
        fc = self.fresh_cond.get(skey)
        fu = self.last_uncond.get(skey)
        if fc is None or fu is None or fc[0] != step_index or not fc[2] or fu[0] != step_index:
            return True
        if not _same_latent(fc[3], fu[2]):
            return False
        c_out = fc[1].float()
        u_out = fu[1].float()
        delta = u_out - c_out
        prev_delta = self.delta.get(skey)
        if prev_delta is not None and prev_delta.shape == delta.shape:
            readback.send(("delta_err", skey, progress), weight_err_fn(prev_delta, delta, skey))
        if adaptive_cos > 0.0 and progress >= 0.50:
            cos_val = F.cosine_similarity(c_out.flatten(1), u_out.flatten(1), dim=1).min()
            readback.send(("cfg_cos", skey, progress), cos_val)
        if skey not in self.delta and len(self.delta) >= 8:
            # Never evict a tile whose COND already ran on this step and whose UNCOND is still pending.
            victim = next((k for k in self.delta if k not in self.fresh_cond or k in self.last_uncond), None)
            if victim is not None:
                self.clear_skey(victim)
        self.delta[skey] = delta
        self.delta_consec[skey] = 0
        # GuideCache (arXiv:2611.25382): spatial guidance-share map boosts error weight on high-guidance text/faces.
        if cond_scale > 1.01:
            guided = u_out + cond_scale * (c_out - u_out)
            cdim = tuple(range(1, c_out.ndim - 2))
            g_mag = (cond_scale * (c_out - u_out)).pow(2).sum(dim=cdim, keepdim=True).sqrt()
            v_mag = guided.pow(2).sum(dim=cdim, keepdim=True).sqrt().clamp_min(1e-4)
            rho = g_mag / v_mag
            rho = rho / rho.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            self.guide_weight[skey] = 1.0 + 0.75 * (rho - 1.0).clamp_min(0.0)
        return True


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
        self._cfg = _CfgCache()

        # Counters compatible with runtime.py's end() reporting & infotext
        self.full_calls = 0
        self.shallow_calls = 0  # full-step skips (skip_all, 0 TRT calls)
        self.cfg_unbatched_calls = 0  # unbatched UNCOND calls saved via Limited-Interval / FasterCache (0 TRT calls)
        self.cfg_batched_calls = 0  # batched [COND, UNCOND] calls run at batch=1 (50% FLOPs)
        self.checks = []  # (index, age, rel_err)

    @property
    def _fresh_cond(self):
        return self._cfg.fresh_cond

    @property
    def _last_uncond(self):
        return self._cfg.last_uncond

    @property
    def _cfg_delta(self):
        return self._cfg.delta

    @property
    def _cfg_delta_kf(self):
        return self._cfg.delta_kf

    @property
    def _cfg_delta_consec(self):
        return self._cfg.delta_consec

    @property
    def _cfg_cos(self):
        return self._cfg.cos

    @property
    def _guide_weight_map(self):
        return self._cfg.guide_weight

    @property
    def tier1_calls(self) -> int:
        return self.cfg_unbatched_calls + self.cfg_batched_calls

    def fresh_cond_for(self, skey, require_real_full: bool = True, x_slice=None):
        """Returns the current step's fresh COND output for `skey` if valid, else None."""
        if self.index is None or skey is None:
            return None
        return self._cfg.fresh_cond_for(skey, self.index, require_real_full=require_real_full, x_slice=x_slice)

    def has_cfg_delta(self, skey) -> bool:
        return skey is not None and skey in self._cfg.delta

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
                    real_cond = {k for k, v in self._cfg.fresh_cond.items() if v[2] and k not in self._volatile_skeys}
                    real_uncond = {k for k in self._cfg.last_uncond if k not in self._volatile_skeys}
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
        self._cfg.clear_step()
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
        self._refresh_fc_boost()

    def _refresh_fc_boost(self):
        """Computes immutable per-step FasterCache boost skeys once per step (before Call 1 & Call 2).

        Closed-loop relaxation unlocks higher speed without weakening quality gates:
        - Condition A: verified ultra-smooth CFG delta (`delta_err <= 0.45 * eff_fc_tol`, reclaiming unused headroom).
        - Condition B: smooth CFG delta (`<= 0.75 * eff_fc_tol`) while trajectory skipping is Kalman-blocked
          (e.g. stochastic `Euler a` / `DPM++ 2M SDE`, where common-mode noise cancels inside `u - c`).
        """
        self._cfg.boost_skeys.clear()
        if (
            not self.region
            or self.done
            or not self._cfg_shortcuts_allowed()
            or self.fc_max_consec <= 0
            or self.fc_tol >= math.inf
        ):
            return
        eff_fc_tol = self._effective_tol(self.fc_tol)
        eff_tol = self._effective_tol(self.tolerance)
        for skey, (est, _) in self._cfg.delta_kf.items():
            if est is None or skey in self._volatile_skeys or skey not in self._cfg.delta:
                continue
            if est <= eff_fc_tol * 0.45:
                self._cfg.boost_skeys.add(skey)
                continue
            if est <= eff_fc_tol * 0.75 and eff_tol < math.inf:
                blocked = False
                for bkey, st in self.branches.items():
                    if _skey_of(bkey) != skey or self.index - st.last_seen_index > STALE:
                        continue
                    rate = self._best_rate(st, bkey)
                    if rate is not None and st.last_err_du is not None and st.last_err_du > 1e-8:
                        if rate * st.last_err_du > eff_tol:
                            blocked = True
                            break
                    elif st.last_err is not None and st.last_err > eff_tol:
                        blocked = True
                        break
                if blocked:
                    self._cfg.boost_skeys.add(skey)

    def _clear_skey_cfg(self, skey):
        self._cfg.clear_skey(skey)

    def _clear_all_cfg(self):
        self._cfg.clear_all()

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
                    prev_est, prev_var = self._cfg.delta_kf.get(skey, (None, 0.25))
                    self._cfg.delta_kf[skey] = _kalman_update(prev_est, prev_var, float(value))
            elif kind == "cfg_cos":
                _, skey, prog = tag
                # Ignore early high-noise cosine similarity (where shared noise makes cos(c, u) artificially high).
                if prog >= 0.50 and skey not in self._volatile_skeys:
                    self._cfg.cos[skey] = float(value)
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
        c = self._cfg.cos.get(skey)
        return c is not None and c >= self.adaptive_cos

    def _can_fastercache_uncond(self, skey) -> bool:
        if (
            not self._cfg_shortcuts_allowed()
            or self.done
            or not self.region
            or self.fc_max_consec <= 0
            or skey in self._volatile_skeys
            or skey not in self._cfg.delta
        ):
            return False
        boosted = skey in self._cfg.boost_skeys
        fc_max = min(4, self.fc_max_consec + 1) if boosted else self.fc_max_consec
        fc_start = min(self.fc_start, max(self.warmup + 0.05, self.fc_start - 0.10)) if boosted else self.fc_start
        if self._progress < fc_start - 1e-6:
            return False
        if self._cfg.delta_consec.get(skey, 0) >= fc_max:
            return False
        if self.fc_tol < math.inf:
            kf = self._cfg.delta_kf.get(skey)
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
            fc_chk = self._cfg.fresh_cond.get(skey)
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
        if fc_ok and skey not in self._cfg.delta:
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
                    if self.fresh_cond_for(skey, require_real_full=False, x_slice=x[slc]) is not None:
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
        # Unbatched UNCOND (e.g. MOSS-Image / split Anima): strictly requires REAL full COND from THIS exact step!
        if n_groups == 1 and groups[0][2] and skey is not None:
            if self.fresh_cond_for(skey, require_real_full=True, x_slice=x[0:chunk]) is not None:
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
        u_hat = cond_out.float() + self._cfg.delta[skey]
        self._cfg.delta_consec[skey] = self._cfg.delta_consec.get(skey, 0) + 1
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
                    self._cfg.fresh_cond[skey] = (self.index, c_hat.detach(), False, c_x.detach())
        for gi, (slc, bkey, is_u, _) in enumerate(groups):
            if is_u:
                if can_couple and c_hat is not None and _same_latent(x[slc], c_x):
                    c_ref = c_hat
                elif can_couple:
                    c_ref = self.fresh_cond_for(skey, require_real_full=False, x_slice=x[slc])
                else:
                    c_ref = None
                if c_ref is not None and tail_drop:
                    assembled[gi] = c_ref.to(x.dtype)
                elif c_ref is not None and fc_ok and skey in self._cfg.delta:
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
        wmap = self._cfg.guide_weight.get(skey)
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
            self._cfg.fresh_cond[skey] = (self.index, out_slice.detach(), True, x_slice.detach())
        elif tag == 1:
            self._cfg.last_uncond[skey] = (self.index, out_slice.detach(), x_slice.detach())

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
            if not self._cfg.update_pair(
                skey, self.index, self._progress, self._cond_scale, self.adaptive_cos, self._readback, self._weighted_rel_err
            ):
                self._mismatched_latents = True
                self._asymmetric_cfg = True
                self._clear_all_cfg()

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
        self._cfg.release()
        self._prev_u = None
        self._step_u = None
