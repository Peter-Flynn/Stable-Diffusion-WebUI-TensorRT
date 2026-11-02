import copy
from dataclasses import dataclass
from enum import Enum
from json import JSONEncoder
import torch


class SDVersion(Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    Unknown = -1

    def __str__(self):
        return self.name

    @classmethod
    def from_str(cls, str):
        try:
            return cls[str]
        except KeyError:
            return cls.Unknown

    def match(self, sd_model):
        if sd_model.is_sd1 and self == SDVersion.SD1:
            return True
        elif sd_model.is_sd2 and self == SDVersion.SD2:
            return True
        elif sd_model.is_sdxl and self == SDVersion.SDXL:
            return True
        elif self == SDVersion.Unknown:
            return True
        else:
            return False


class ModelType(Enum):
    UNET = 0
    CONTROLNET = 1
    LORA = 2
    UNDEFINED = -1

    @classmethod
    def from_string(cls, s):
        return getattr(cls, s.upper(), None)

    def __str__(self):
        return self.name.lower()


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
    trt_version: str = ""  # TensorRT version the engine was built with ("" = unknown / legacy)

    def min_batch(self) -> int:
        return self.profile["sample"][0][0]

    def min_tokens(self) -> int:
        return self.profile["encoder_hidden_states"][0][1]

    def is_compatible_from_dict(self, feed_dict: dict):
        distance = 0.0
        for k, v in feed_dict.items():
            if k not in self.profile:
                continue
            _min, _opt, _max = self.profile[k]
            v_shape = list(v.shape) if hasattr(v, "shape") else list(v)
            # Inputs below the profile minimum can be padded by repetition (see TrtUnet.forward):
            # the batch always, the token axis when the minimum is a multiple of it.
            if v_shape and v_shape[0] < _min[0]:
                v_shape[0] = _min[0]
                distance += 500.0
            if k == "encoder_hidden_states" and v_shape[1] < _min[1] and _min[1] % v_shape[1] == 0:
                v_shape[1] = _min[1]
                distance += 500.0
            v_tensor = torch.Tensor(v_shape)
            r_min = torch.Tensor(_max) - v_tensor
            r_opt = (torch.Tensor(_opt) - v_tensor).abs()
            r_max = v_tensor - torch.Tensor(_min)
            if torch.any(r_min < 0) or torch.any(r_max < 0):
                return (False, distance)
            distance += float(r_opt.sum().item() + 0.5 * (r_max.sum().item() + 0.5 * r_min.sum().item()))

        if self.static_shapes:
            is_exact = True
            for k, v in feed_dict.items():
                if k in self.profile:
                    opt_shape = self.profile[k][1]
                    v_shape = list(v.shape) if hasattr(v, "shape") else list(v)
                    if v_shape != list(opt_shape):
                        is_exact = False
                        break
            distance += -1000.0 if is_exact else -100.0
        else:
            distance += 50.0

        return (True, distance)

    def is_compatible(
        self, width: int, height: int, batch_size: int, max_embedding: int
    ):
        distance = 0
        sample = self.profile["sample"]
        embedding = self.profile["encoder_hidden_states"]

        batch_size *= 2
        width = width // 8
        height = height // 8

        _min, _opt, _max = sample
        if _min[0] > batch_size or _max[0] < batch_size:
            return (False, distance)
        if _min[2] > height or _max[2] < height:
            return (False, distance)
        if _min[3] > width or _max[3] < width:
            return (False, distance)

        _min_em, _opt_em, _max_em = embedding
        if _min_em[1] > max_embedding or _max_em[1] < max_embedding:
            return (False, distance)

        # 1. Spatial distance: difference between optimal dims and requested dims
        dist_spatial = (
            abs(_opt[0] - batch_size) * 10
            + abs(_opt[2] - height)
            + abs(_opt[3] - width)
            + 0.25 * (abs(_max[2] - height) + abs(_max[3] - width))
        )

        # 2. Token count distance: difference between optimal token count and prompt tokens
        dist_token = abs(_opt_em[1] - max_embedding) * 0.1

        # 3. Static shape priority:
        # If this engine has static shapes and matches requested resolution and batch,
        # grant a major priority bonus so specialized static engines are preferred over dynamic fallbacks.
        if self.static_shapes:
            if _opt[0] == batch_size and _opt[2] == height and _opt[3] == width:
                dist_static = -1000.0
            else:
                dist_static = -100.0
        else:
            dist_static = 50.0

        distance = dist_spatial + dist_token + dist_static
        return (True, distance)


class ModelConfigEncoder(JSONEncoder):
    def default(self, o: ModelConfig):
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

    def __str__(self) -> str:
        return "Batch Size: {}-{}-{}\nHeight: {}-{}-{}\nWidth: {}-{}-{}\nToken Count: {}-{}-{}".format(
            self.bs_min,
            self.bs_opt,
            self.bs_max,
            self.h_min,
            self.h_opt,
            self.h_max,
            self.w_min,
            self.w_opt,
            self.w_max,
            self.t_min,
            self.t_opt,
            self.t_max,
        )

    def out(self):
        return (
            self.bs_min,
            self.bs_opt,
            self.bs_max,
            self.h_min,
            self.h_opt,
            self.h_max,
            self.w_min,
            self.w_opt,
            self.w_max,
            self.t_min,
            self.t_opt,
            self.t_max,
        )

    def token_to_dim(self, static_shapes: bool):
        self.t_min = (self.t_min // 75) * 77
        self.t_opt = (self.t_opt // 75) * 77
        self.t_max = (self.t_max // 75) * 77

        if static_shapes:
            self.t_min = self.t_max = self.t_opt
            self.bs_min = self.bs_max = self.bs_opt
            self.h_min = self.h_max = self.h_opt
            self.w_min = self.w_max = self.w_opt
            self.static_shape = True
        else:
            self.t_max = max(self.t_max, self.t_opt)
            self.t_min = min(self.t_min, self.t_opt)
            self.bs_max = max(self.bs_max, self.bs_opt)
            self.bs_min = min(self.bs_min, self.bs_opt)
            self.h_max = max(self.h_max, self.h_opt)
            self.h_min = min(self.h_min, self.h_opt)
            self.w_max = max(self.w_max, self.w_opt)
            self.w_min = min(self.w_min, self.w_opt)

    def get_latent_dim(self):
        return (
            self.h_min // 8,
            self.h_opt // 8,
            self.h_max // 8,
            self.w_min // 8,
            self.w_opt // 8,
            self.w_max // 8,
        )

    def get_a1111_batch_dim(self):
        # Cond and uncond are normally denoised together (batch 2B). A batch of B happens
        # with CFG 1 (reForge skips the uncond pass) or when cond/uncond run separately;
        # dynamic engines cover it natively, static ones get padded at runtime.
        if self.static_shape or self.bs_min == self.bs_max == self.bs_opt and self.h_min == self.h_max and self.w_min == self.w_max:
            return (self.bs_opt * 2, self.bs_opt * 2, self.bs_opt * 2)
        return (self.bs_min, self.bs_opt * 2, self.bs_max * 2)


class ProfilePrests:
    def __init__(self):
        self.profile_presets = {
            "512x512 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 512, 512, 512, 512, 512, 512, 75, 75, 75
            ),
            "768x768 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 768, 768, 768, 768, 768, 768, 75, 75, 75
            ),
            "1024x1024 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75
            ),
            "1024x1536 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 1536, 1536, 1536, 1024, 1024, 1024, 75, 75, 75
            ),
            "1536x1024 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 1024, 1024, 1024, 1536, 1536, 1536, 75, 75, 75
            ),
            "256x256 - 512x512 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 256, 512, 512, 256, 512, 512, 75, 75, 150
            ),
            "512x512 - 768x768 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150
            ),
            "768x768 - 1024x1024 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 768, 1024, 1024, 768, 1024, 1024, 75, 75, 150
            ),
            "1024x1024 - 1024x1536 | Batch Size 1-2": ProfileSettings(
                1, 1, 2, 1024, 1536, 1536, 1024, 1024, 1024, 75, 75, 150
            ),
        }
        self.default = ProfileSettings(
            1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150
        )
        self.default_xl = ProfileSettings(
            1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75
        )

    def get_settings_from_version(self, version: str):
        static = False
        if version == "Default":
            return *self.default.out(), static
        if "Static" in version:
            static = True
        return *self.profile_presets[version].out(), static

    def get_choices(self):
        return list(self.profile_presets.keys()) + ["Default"]

    def get_default(self, is_xl: bool):
        # Return a copy: callers mutate the settings (token_to_dim).
        return copy.copy(self.default_xl if is_xl else self.default)
