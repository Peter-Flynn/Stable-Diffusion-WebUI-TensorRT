"""Shape profiles: what an engine accepts, in image terms for people and in UNet input shapes for TensorRT."""

import math
from dataclasses import dataclass

TOKENS_PER_CHUNK = 75
CONTEXT_PER_CHUNK = 77
FIELDS = ("width", "height", "batch", "tokens")


def context_length(tokens: int) -> int:
    return max(1, math.ceil(tokens / TOKENS_PER_CHUNK)) * CONTEXT_PER_CHUNK


@dataclass(frozen=True)
class Profile:
    width: tuple  # (min, opt, max) in pixels
    height: tuple
    batch: tuple  # images per batch
    tokens: tuple  # prompt tokens

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(*(tuple(int(v) for v in d[f]) for f in FIELDS))

    def to_dict(self) -> dict:
        return {f: list(getattr(self, f)) for f in FIELDS}

    def validated(self, multiple: int) -> "Profile":
        """Snapped to the sizes the model takes (outward, so the range still covers what was asked),
        each range ordered and containing its optimum."""

        def fix(values, step, lowest):
            lo, opt, hi = values
            lo, opt, hi = (max(lowest, v * step) for v in (math.floor(lo / step), round(opt / step), math.ceil(hi / step)))
            return min(lo, opt), opt, max(hi, opt)

        return Profile(
            fix(self.width, multiple, multiple),
            fix(self.height, multiple, multiple),
            fix(self.batch, 1, 1),
            fix(self.tokens, TOKENS_PER_CHUNK, TOKENS_PER_CHUNK),
        )

    @property
    def static(self) -> bool:
        return all(getattr(self, f)[0] == getattr(self, f)[2] for f in FIELDS)

    def describe(self) -> str:
        def span(values, unit=""):
            lo, opt, hi = values
            return f"{opt}{unit}" if lo == hi else f"{lo}-{hi}{unit} (opt {opt})"

        size = f"{self.width[1]}x{self.height[1]}" if self.width[0] == self.width[2] and self.height[0] == self.height[2] else (
            f"{span(self.width)} x {span(self.height)}"
        )
        return f"{size}, batch {span(self.batch)}, {span(self.tokens)} tokens{' (static)' if self.static else ''}"

    def unet_shapes(self, in_channels: int, context_dim: int, adm_channels: int) -> dict:
        """{input: [min, opt, max] shapes}. The UNet batch holds cond and uncond (2 per image); dynamic
        batches also take one per image (CFG 1), static ones are padded to it."""
        lo, opt, hi = self.batch
        batch = (2 * lo,) * 3 if lo == hi else (lo, 2 * opt, 2 * hi)
        shapes = {
            "sample": [[b, in_channels, h // 8, w // 8] for b, h, w in zip(batch, self.height, self.width)],
            "timesteps": [[b] for b in batch],
            "encoder_hidden_states": [[b, context_length(t), context_dim] for b, t in zip(batch, self.tokens)],
        }
        if adm_channels:
            shapes["y"] = [[b, adm_channels] for b in batch]
        return shapes


def fit(shapes: dict, batch: int, height: int, width: int, context: int):
    """(score, padded batch, context repeats) if the profile can run these UNet inputs, else None.
    Smaller batches are padded by repetition and the output sliced back; a context repeated k times
    gives the same cross-attention, so it can stand in for a longer one."""
    (b_lo, _, h_lo, w_lo), (b_opt, _, h_opt, w_opt), (b_hi, _, h_hi, w_hi) = shapes["sample"]
    if not (h_lo <= height <= h_hi and w_lo <= width <= w_hi and batch <= b_hi):
        return None
    c_lo, c_hi = shapes["encoder_hidden_states"][0][1], shapes["encoder_hidden_states"][2][1]
    repeats = max(1, math.ceil(c_lo / context))
    if repeats * context > c_hi:
        return None
    padded = max(batch, b_lo)
    spread = math.log(h_hi * w_hi / (h_lo * w_lo)) + math.log(b_hi / b_lo) + math.log(c_hi / c_lo)  # narrower ranges run faster
    score = (padded / batch, repeats, abs(math.log(height * width / (h_opt * w_opt))), abs(math.log(padded / b_opt)), spread)
    return score, padded, repeats


def select(profile_shapes: list, batch: int, height: int, width: int, context: int):
    """(profile index, padded batch, context repeats) of the best profile for these inputs, or None."""
    fits = [(f, i) for i, s in enumerate(profile_shapes) if (f := fit(s, batch, height, width, context))]
    if not fits:
        return None
    (_, padded, repeats), index = min(fits)
    return index, padded, repeats


def covers_size(profile_shapes: list, height: int, width: int, batch: int = 1) -> bool:
    return any(
        s["sample"][0][2] <= height <= s["sample"][2][2] and s["sample"][0][3] <= width <= s["sample"][2][3] and batch <= s["sample"][2][0]
        for s in profile_shapes
    )


def _p(w, h, batch=(1, 1, 1), tokens=(75, 75, 75)):
    return Profile(w if isinstance(w, tuple) else (w,) * 3, h if isinstance(h, tuple) else (h,) * 3, batch, tokens)


PRESETS = {
    "xl": {
        "1024x1024": _p(1024, 1024),
        "832x1216": _p(832, 1216),
        "1216x832": _p(1216, 832),
        "768-1344 (any SDXL aspect ratio)": _p((768, 1024, 1344), (768, 1024, 1344), tokens=(75, 75, 150)),
        "1536x1536 (Hires. fix 1.5x of 1024)": _p(1536, 1536),
        "1024-2048 (Hires. fix up to 2x)": _p((1024, 1536, 2048), (1024, 1536, 2048), tokens=(75, 75, 150)),
    },
    "sd": {
        "512x512": _p(512, 512),
        "512x768": _p(512, 768),
        "768x512": _p(768, 512),
        "512-768, batch 1-4": _p((512, 512, 768), (512, 512, 768), (1, 1, 4), (75, 75, 150)),
        "768x768": _p(768, 768),
        "1024x1024 (Hires. fix 2x of 512)": _p(1024, 1024),
    },
}


def presets_for(adm_channels: int) -> dict:
    return PRESETS["xl" if adm_channels else "sd"]


def default_profiles(adm_channels: int) -> list:
    return [next(iter(presets_for(adm_channels).values()))]
