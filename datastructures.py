import copy
from dataclasses import dataclass, fields
from enum import Enum
from json import JSONEncoder


class SDVersion(Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    Unknown = -1

    def __str__(self):
        return self.name

    @classmethod
    def from_str(cls, s):
        return cls.__members__.get(s, cls.Unknown)

    def match(self, sd_model):
        return self == SDVersion.Unknown or bool(getattr(sd_model, f"is_{self.name.lower()}", False))


@dataclass
class ModelConfig:
    profile: dict
    static_shapes: bool
    fp32: bool
    inpaint: bool
    refit: bool
    lora: bool
    vram: int
    unet_hidden_dim: int = 4
    trt_version: str = ""

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def min_batch(self) -> int:
        return self.profile["sample"][0][0]

    def min_tokens(self) -> int:
        return self.profile["encoder_hidden_states"][0][1]

    def is_compatible_from_dict(self, shapes: dict):
        """(fits, distance from the profile's optimum) for {input name: tensor or shape}. A batch
        below the minimum, and a context length dividing the minimum, fit by repetition (see
        TrtUnet._pad_to_profile)."""
        distance = 0.0
        exact = True
        for name, shape in shapes.items():
            if name not in self.profile:
                continue
            lo, opt, hi = self.profile[name]
            shape = list(getattr(shape, "shape", shape))
            exact = exact and shape == list(opt)
            if shape[0] < lo[0]:
                shape[0] = lo[0]
                distance += 500
            if name == "encoder_hidden_states" and shape[1] < lo[1] and lo[1] % shape[1] == 0:
                shape[1] = lo[1]
                distance += 500
            if any(s < l or s > h for s, l, h in zip(shape, lo, hi)):
                return False, distance
            distance += sum(abs(o - s) + 0.5 * (s - l) + 0.25 * (h - s) for s, l, o, h in zip(shape, lo, opt, hi))
        if self.static_shapes:
            distance += -1000 if exact else -100
        else:
            distance += 50
        return True, distance

    def is_compatible(self, width: int, height: int, batch: int, tokens: int):
        """is_compatible_from_dict for the UNet inputs of an image size (batch: cond + uncond)."""
        shapes = {}
        for name, (_, opt, _) in self.profile.items():
            shape = [batch, *opt[1:]]
            if name == "sample":
                shape[2:] = [height // 8, width // 8]
            elif name == "encoder_hidden_states":
                shape[1] = tokens
            shapes[name] = shape
        return self.is_compatible_from_dict(shapes)


class ModelConfigEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


@dataclass
class ProfileSettings:
    bs_min: int
    bs_opt: int
    bs_max: int
    h_min: int
    h_opt: int
    h_max: int
    w_min: int
    w_opt: int
    w_max: int
    t_min: int
    t_opt: int
    t_max: int
    static_shape: bool = False

    def __str__(self):
        return (
            f"Batch Size: {self.bs_min}-{self.bs_opt}-{self.bs_max}\nHeight: {self.h_min}-{self.h_opt}-{self.h_max}\n"
            f"Width: {self.w_min}-{self.w_opt}-{self.w_max}\nToken Count: {self.t_min}-{self.t_opt}-{self.t_max}"
        )

    def out(self):
        return tuple(getattr(self, f.name) for f in fields(self)[:12])

    def token_to_dim(self, static_shapes: bool):
        """Prompt tokens -> context length (77 per 75 tokens); static: everything at its optimum."""
        self.t_min, self.t_opt, self.t_max = ((t // 75) * 77 for t in (self.t_min, self.t_opt, self.t_max))
        for axis in ("t", "bs", "h", "w"):
            opt = getattr(self, f"{axis}_opt")
            if static_shapes:
                lo = hi = opt
            else:
                lo, hi = min(getattr(self, f"{axis}_min"), opt), max(getattr(self, f"{axis}_max"), opt)
            setattr(self, f"{axis}_min", lo)
            setattr(self, f"{axis}_max", hi)
        self.static_shape = self.static_shape or static_shapes

    def get_latent_dim(self):
        return self.h_min // 8, self.h_opt // 8, self.h_max // 8, self.w_min // 8, self.w_opt // 8, self.w_max // 8

    def get_batch_dim(self):
        # Usually cond + uncond (2B); B with CFG 1 or when they run separately, which static engines pad.
        if self.static_shape or (self.bs_min == self.bs_max == self.bs_opt and self.h_min == self.h_max and self.w_min == self.w_max):
            return self.bs_opt * 2, self.bs_opt * 2, self.bs_opt * 2
        return self.bs_min, self.bs_opt * 2, self.bs_max * 2


class ProfilePresets:
    def __init__(self):
        self.profile_presets = {
            "512x512 | Batch Size 1 (Static)": ProfileSettings(1, 1, 1, 512, 512, 512, 512, 512, 512, 75, 75, 75),
            "768x768 | Batch Size 1 (Static)": ProfileSettings(1, 1, 1, 768, 768, 768, 768, 768, 768, 75, 75, 75),
            "1024x1024 | Batch Size 1 (Static)": ProfileSettings(1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75),
            "1024x1536 | Batch Size 1 (Static)": ProfileSettings(1, 1, 1, 1536, 1536, 1536, 1024, 1024, 1024, 75, 75, 75),
            "1536x1024 | Batch Size 1 (Static)": ProfileSettings(1, 1, 1, 1024, 1024, 1024, 1536, 1536, 1536, 75, 75, 75),
            "256x256 - 512x512 | Batch Size 1-4": ProfileSettings(1, 1, 4, 256, 512, 512, 256, 512, 512, 75, 75, 150),
            "512x512 - 768x768 | Batch Size 1-4": ProfileSettings(1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150),
            "768x768 - 1024x1024 | Batch Size 1-4": ProfileSettings(1, 1, 4, 768, 1024, 1024, 768, 1024, 1024, 75, 75, 150),
            "1024x1024 - 1024x1536 | Batch Size 1-2": ProfileSettings(1, 1, 2, 1024, 1536, 1536, 1024, 1024, 1024, 75, 75, 150),
        }
        self.default = ProfileSettings(1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150)
        self.default_xl = ProfileSettings(1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75)

    def get_settings_from_version(self, version: str):
        if version == "Default":
            return *self.default.out(), False
        return *self.profile_presets[version].out(), "Static" in version

    def get_choices(self):
        return list(self.profile_presets) + ["Default"]

    def get_default(self, is_xl: bool):
        return copy.copy(self.default_xl if is_xl else self.default)  # callers mutate it
