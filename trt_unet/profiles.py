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
        if not isinstance(d, dict):
            raise TypeError("profile must be a dict")
        values = []
        for f in FIELDS:
            raw = d[f]
            if not isinstance(raw, (list, tuple)) or len(raw) != 3 or any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
                raise ValueError(f"invalid profile field {f}")
            lo, opt, hi = raw
            lowest = 8 if f in ("width", "height") else 1
            if not (lowest <= lo <= opt <= hi):
                raise ValueError(f"invalid bounds for {f}: {(lo, opt, hi)}")
            values.append((lo, opt, hi))
        return cls(*values)

    def to_dict(self) -> dict:
        return {f: list(getattr(self, f)) for f in FIELDS}

    def validated(self, multiple: int, *, token_step: int = TOKENS_PER_CHUNK, token_lowest: int = TOKENS_PER_CHUNK) -> "Profile":
        def fix(values, step, lowest):
            lo, opt, hi = values
            lo, opt, hi = (max(lowest, v * step) for v in (math.floor(lo / step), round(opt / step), math.ceil(hi / step)))
            return min(lo, opt), opt, max(hi, opt)

        lowest = max(multiple, 64)  # fresh.TILE is 8 latent pixels (64 image pixels)
        return Profile(
            fix(self.width, multiple, lowest),
            fix(self.height, multiple, lowest),
            fix(self.batch, 1, 1),
            fix(self.tokens, token_step, token_lowest),
        )

    def validated_for(self, arch) -> "Profile":
        if arch is None:
            return self.validated(64)
        if getattr(arch, "is_dit", False):
            return self.validated(arch.multiple, token_step=1, token_lowest=1)
        return self.validated(arch.multiple)

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

    def engine_shapes(self, arch, module=None) -> dict:
        if not getattr(arch, "is_dit", False):
            return self.unet_shapes(arch.in_channels, arch.context_dim, arch.adm_channels)
        lo, opt, hi = self.batch
        if arch.kind == "anima":
            batch = (lo, 2 * opt, 2 * hi)
            head_dim = int(getattr(getattr(getattr(module, "blocks", [None])[0], "self_attn", None), "head_dim", 128)) if module is not None else 128
            rope_half = head_dim // 2
            return {
                "sample": [[b, arch.in_channels, 1, h // 8, w // 8] for b, h, w in zip(batch, self.height, self.width)],
                "timesteps": [[b] for b in batch],
                "encoder_hidden_states": [[b, t, arch.context_dim] for b, t in zip(batch, self.tokens)],
                "rope": [[1, h // 16, w // 16, 1, rope_half, 2] for h, w in zip(self.height, self.width)],
            }
        if arch.kind == "moss_image":
            # MOSS-Image cond and uncond almost always have distinct Qwen token lengths (unbatched batch=opt calls),
            # while equal-length prompts still batch to 2 * hi.
            batch = (lo, opt, 2 * hi)
            head_dim = 128
            if module is not None and getattr(module, "layers", None):
                q_norm = getattr(module.layers[0], "q_norm", None)
                if q_norm is not None and getattr(q_norm, "weight", None) is not None:
                    head_dim = int(q_norm.weight.shape[0])
            rope_half = head_dim // 2
            return {
                "sample": [[b, arch.in_channels, h // 16, w // 16] for b, h, w in zip(batch, self.height, self.width)],
                "timesteps": [[b] for b in batch],
                "encoder_hidden_states": [[b, t, arch.context_dim] for b, t in zip(batch, self.tokens)],
                "rope_img": [[1, h // 16, w // 16, 1, rope_half, 2] for h, w in zip(self.height, self.width)],
                "rope_txt": [[1, t, 1, rope_half, 2] for t in self.tokens],
            }
        raise ValueError(f"Unsupported architecture kind {arch.kind!r}")


def fit(shapes: dict, batch: int, height: int, width: int, context, *, allow_context_repeat: bool = True):
    # A batch below the profile's minimum is padded by repeating it (the caller slices the output back). A context
    # shorter than the minimum is tiled k times when allow_context_repeat is True (safe for unpositioned cross-attn,
    # disabled for RoPE-positioned text tokens like MOSS-Image). When context is None (pre-call sizing / memory query),
    # skip context bounds and assume repeats=1.
    s_lo, s_opt, s_hi = shapes["sample"]
    b_lo, h_lo, w_lo = s_lo[0], s_lo[-2], s_lo[-1]
    b_opt, h_opt, w_opt = s_opt[0], s_opt[-2], s_opt[-1]
    b_hi, h_hi, w_hi = s_hi[0], s_hi[-2], s_hi[-1]
    if not (h_lo <= height <= h_hi and w_lo <= width <= w_hi and batch <= b_hi):
        return None
    c_lo, c_hi = shapes["encoder_hidden_states"][0][1], shapes["encoder_hidden_states"][2][1]
    if context is None:
        repeats = 1
    elif allow_context_repeat:
        repeats = max(1, math.ceil(c_lo / context))
        if repeats * context > c_hi:
            return None
    else:
        if not (c_lo <= context <= c_hi):
            return None
        repeats = 1
    padded = max(batch, b_lo)
    spread = math.log(h_hi * w_hi / (h_lo * w_lo)) + math.log(b_hi / b_lo) + math.log(c_hi / c_lo)
    score = (padded / batch, repeats, abs(math.log(height / h_opt)) + abs(math.log(width / w_opt)), abs(math.log(padded / b_opt)), spread)
    return score, padded, repeats


def select(profile_shapes: list, batch: int, height: int, width: int, context, prefer=(), *, allow_context_repeat: bool = True):
    fits = [(f, i) for i, s in enumerate(profile_shapes) if (f := fit(s, batch, height, width, context, allow_context_repeat=allow_context_repeat))]
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
        s["sample"][0][-2] <= height <= s["sample"][2][-2] and s["sample"][0][-1] <= width <= s["sample"][2][-1] and batch <= s["sample"][2][0]
        for s in profile_shapes
    )


def _p(w, h, batch=(1, 1, 1), tokens=(75, 300, 750)):  # a wide prompt-length range costs almost nothing
    return Profile(w if isinstance(w, tuple) else (w,) * 3, h if isinstance(h, tuple) else (h,) * 3, batch, tokens)


PRESETS = {
    "xl": {
        "320-1536 (every SDXL resolution, and half sizes)": _p((320, 1024, 1536), (320, 1024, 1536)),
        "1024x1024": _p(1024, 1024),
        "832x1216": _p(832, 1216),
        "1216x832": _p(1216, 832),
        "320-3072 (Hires. fix up to 2x of every SDXL resolution)": _p((320, 1536, 3072), (320, 1536, 3072)),
        "320-1536, batch 1-4": _p((320, 1024, 1536), (320, 1024, 1536), (1, 1, 4)),
    },
    "sd": {
        "256-768 (every SD 1.5 resolution, and half sizes)": _p((256, 512, 768), (256, 512, 768)),
        "512x512": _p(512, 512),
        "512x768": _p(512, 768),
        "768x512": _p(768, 512),
        "256-1536 (Hires. fix up to 2x of every SD 1.5 resolution)": _p((256, 1024, 1536), (256, 1024, 1536)),
        "256-768, batch 1-4": _p((256, 512, 768), (256, 512, 768), (1, 1, 4)),
    },
    "anima": {
        "512-1536 (every Anima resolution)": _p((512, 1024, 1536), (512, 1024, 1536), tokens=(1, 512, 1024)),
        "1024x1024 (Anima)": _p(1024, 1024, tokens=(1, 512, 1024)),
        "832x1216 (Anima)": _p(832, 1216, tokens=(1, 512, 1024)),
        "1216x832 (Anima)": _p(1216, 832, tokens=(1, 512, 1024)),
        "512-2048 (Anima Hires. fix up to 2K)": _p((512, 1536, 2048), (512, 1536, 2048), tokens=(1, 512, 1024)),
        "512-1536, batch 1-4 (Anima)": _p((512, 1024, 1536), (512, 1024, 1536), (1, 1, 4), tokens=(1, 512, 1024)),
    },
    "moss_image": {
        "512-1536 (every MOSS-Image resolution)": _p((512, 1024, 1536), (512, 1024, 1536), tokens=(1, 256, 1024)),
        "1024x1024 (MOSS-Image)": _p(1024, 1024, tokens=(1, 256, 1024)),
        "896x1152 (MOSS-Image)": _p(896, 1152, tokens=(1, 256, 1024)),
        "1152x896 (MOSS-Image)": _p(1152, 896, tokens=(1, 256, 1024)),
        "512-2048 (MOSS-Image Hires. fix up to 2K)": _p((512, 1536, 2048), (512, 1536, 2048), tokens=(1, 256, 1024)),
        "512-1536, batch 1-4 (MOSS-Image)": _p((512, 1024, 1536), (512, 1024, 1536), (1, 1, 4), tokens=(1, 256, 1024)),
    },
}
# The size ranges start at half the smallest base size: Euler Dy, Euler SMEA Dy, Euler Negative Dy and their CFG++
# versions call the UNet at half size in steps 0-3, and DPM++ 2S Ancestral CFG++ Intern in every step, and a call no
# profile takes sends the rest of the pass to PyTorch. That also covers the Hires. fix profile's own reason to start
# low: an upscaled portrait or landscape (640x1536 at 1.5x is 960x2304) only exceeds the base profile on one side.
DEFAULTS = {
    "xl": ("320-1536 (every SDXL resolution, and half sizes)", "320-3072 (Hires. fix up to 2x of every SDXL resolution)"),
    "sd": ("256-768 (every SD 1.5 resolution, and half sizes)", "256-1536 (Hires. fix up to 2x of every SD 1.5 resolution)"),
    "anima": ("512-1536 (every Anima resolution)",),
    "moss_image": ("512-1536 (every MOSS-Image resolution)",),
}


def _family(arch_or_adm) -> str:
    if hasattr(arch_or_adm, "kind"):
        if arch_or_adm.kind in PRESETS:
            return arch_or_adm.kind
        return "xl" if arch_or_adm.adm_channels else "sd"
    return "xl" if arch_or_adm else "sd"


def presets_for(arch_or_adm) -> dict:
    return PRESETS[_family(arch_or_adm)]


def default_profiles(arch_or_adm) -> list:
    family = _family(arch_or_adm)
    return [PRESETS[family][name] for name in DEFAULTS[family]]
