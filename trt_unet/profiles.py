import math
from dataclasses import dataclass

TOKENS_PER_CHUNK = 75
CONTEXT_PER_CHUNK = 77  # 75 prompt tokens + CLIP's start and end tokens
FIELDS = ("width", "height", "batch", "tokens")


def context_length(tokens: int) -> int:
    return max(1, math.ceil(tokens / TOKENS_PER_CHUNK)) * CONTEXT_PER_CHUNK


@dataclass(frozen=True)
class Profile:
    width: tuple
    height: tuple
    batch: tuple
    tokens: tuple

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(*(tuple(int(v) for v in d[f]) for f in FIELDS))

    def to_dict(self) -> dict:
        return {f: list(getattr(self, f)) for f in FIELDS}

    def validated(self, multiple: int) -> "Profile":
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
    def fixed_size(self) -> bool:
        return all(getattr(self, f)[0] == getattr(self, f)[2] for f in ("width", "height", "batch"))

    def describe(self) -> str:
        def span(values, unit=""):
            lo, opt, hi = values
            return f"{opt}{unit}" if lo == hi else f"{lo}-{hi}{unit} (opt {opt})"

        size = f"{self.width[1]}x{self.height[1]}" if self.width[0] == self.width[2] and self.height[0] == self.height[2] else (
            f"{span(self.width)} x {span(self.height)}"
        )
        return f"{size}, batch {span(self.batch)}, {span(self.tokens)} tokens{' (fixed size)' if self.fixed_size else ''}"

    def unet_shapes(self, in_channels: int, context_dim: int, adm_channels: int) -> dict:
        # Cond and uncond make 2 UNet rows per image. The minimum stays at 1 so CFG-1 samplers (Lightning, DMD2,
        # Turbo), which send 1 row per image, aren't padded to 2.
        lo, opt, hi = self.batch
        batch = (lo, 2 * opt, 2 * hi)
        shapes = {
            "sample": [[b, in_channels, h // 8, w // 8] for b, h, w in zip(batch, self.height, self.width)],
            "timesteps": [[b] for b in batch],
            "encoder_hidden_states": [[b, context_length(t), context_dim] for b, t in zip(batch, self.tokens)],
        }
        if adm_channels:
            shapes["y"] = [[b, adm_channels] for b in batch]
        return shapes


def fit(shapes: dict, batch: int, height: int, width: int, context: int):
    # A batch below the profile's minimum is padded by repeating it (the caller slices the output back). A context
    # shorter than the minimum is tiled k times, which leaves cross-attention's output unchanged.
    (b_lo, _, h_lo, w_lo), (b_opt, _, h_opt, w_opt), (b_hi, _, h_hi, w_hi) = shapes["sample"]
    if not (h_lo <= height <= h_hi and w_lo <= width <= w_hi and batch <= b_hi):
        return None
    c_lo, c_hi = shapes["encoder_hidden_states"][0][1], shapes["encoder_hidden_states"][2][1]
    repeats = max(1, math.ceil(c_lo / context))
    if repeats * context > c_hi:
        return None
    padded = max(batch, b_lo)
    spread = math.log(h_hi * w_hi / (h_lo * w_lo)) + math.log(b_hi / b_lo) + math.log(c_hi / c_lo)
    score = (padded / batch, repeats, abs(math.log(height * width / (h_opt * w_opt))), abs(math.log(padded / b_opt)), spread)
    return score, padded, repeats


def select(profile_shapes: list, batch: int, height: int, width: int, context: int, prefer=()):
    fits = [(f, i) for i, s in enumerate(profile_shapes) if (f := fit(s, batch, height, width, context))]
    if not fits:
        return None
    best = min(fits)
    # A profile of the engine already loaded wins unless the best needs less padding or fewer repeats: loading
    # another engine costs far more than running on a profile tuned for other sizes.
    preferred = [f for f in fits if f[1] in prefer]
    if preferred and min(preferred)[0][0][:2] <= best[0][0][:2]:
        best = min(preferred)
    (_, padded, repeats), index = best
    return index, padded, repeats


def covers_size(profile_shapes: list, height: int, width: int, batch: int = 1) -> bool:
    return any(
        s["sample"][0][2] <= height <= s["sample"][2][2] and s["sample"][0][3] <= width <= s["sample"][2][3] and batch <= s["sample"][2][0]
        for s in profile_shapes
    )


def _p(w, h, batch=(1, 1, 1), tokens=(75, 300, 750)):  # a wide prompt-length range costs almost nothing
    return Profile(w if isinstance(w, tuple) else (w,) * 3, h if isinstance(h, tuple) else (h,) * 3, batch, tokens)


PRESETS = {
    "xl": {
        "640-1536 (every SDXL resolution)": _p((640, 1024, 1536), (640, 1024, 1536)),
        "1024x1024": _p(1024, 1024),
        "832x1216": _p(832, 1216),
        "1216x832": _p(1216, 832),
        "640-3072 (Hires. fix up to 2x of every SDXL resolution)": _p((640, 1536, 3072), (640, 1536, 3072)),
        "640-1536, batch 1-4": _p((640, 1024, 1536), (640, 1024, 1536), (1, 1, 4)),
    },
    "sd": {
        "512-768 (every SD 1.5 resolution)": _p((512, 512, 768), (512, 512, 768)),
        "512x512": _p(512, 512),
        "512x768": _p(512, 768),
        "768x512": _p(768, 512),
        "512-1536 (Hires. fix up to 2x of every SD 1.5 resolution)": _p((512, 1024, 1536), (512, 1024, 1536)),
        "512-768, batch 1-4": _p((512, 512, 768), (512, 512, 768), (1, 1, 4)),
    },
}
# The Hires. fix profiles start at the smallest base size, not the largest: an upscaled portrait or landscape
# (640x1536 at 1.5x is 960x2304) only exceeds the base profile on one side.
DEFAULTS = {
    "xl": ("640-1536 (every SDXL resolution)", "640-3072 (Hires. fix up to 2x of every SDXL resolution)"),
    "sd": ("512-768 (every SD 1.5 resolution)", "512-1536 (Hires. fix up to 2x of every SD 1.5 resolution)"),
}


def _family(adm_channels: int) -> str:
    return "xl" if adm_channels else "sd"


def presets_for(adm_channels: int) -> dict:
    return PRESETS[_family(adm_channels)]


def default_profiles(adm_channels: int) -> list:
    family = _family(adm_channels)
    return [PRESETS[family][name] for name in DEFAULTS[family]]
