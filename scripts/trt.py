import contextlib
import math
import os
import re
from typing import List

import gradio as gr
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open

from modules import devices, prompt_parser, script_callbacks, scripts, sd_models, sd_unet, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

import ui_trt
import unet_memory
from model_manager import TRT_MODEL_DIR, modelmanager, onnx_path
from unet_memory import TrtFallback
from utilities import Engine

AUTO_PROFILE = "Auto (Best Match / SD Unet default)"  # saved UI settings reference this text
EXTRA_NETWORK_TAG = re.compile(r"<(\w+):([^>]+)>")
HEADROOM = 256 * 2**20


class NoProfileError(RuntimeError):
    pass


class MissingTrtLora(Exception):
    def __init__(self, names):
        super().__init__(f"LoRA {', '.join(names)} hasn't been converted for TensorRT (TensorRT tab -> TensorRT LoRA)")


def _notify(message: str):
    print(f"[TensorRT] {message}", flush=True)
    try:
        gr.Warning(f"TensorRT: {message}")
    except Exception:
        pass


def _free_cuda_cache():
    if unet_memory.IS_FORGE_BACKEND:
        unet_memory.forge_model_management().soft_empty_cache(force=True)
    torch.cuda.empty_cache()


def current_model_name() -> str:
    return shared.sd_model.sd_checkpoint_info.model_name


def hires_size(p):
    """The Hires. fix size, which the WebUI only works out after process()."""
    rx, ry = getattr(p, "hr_resize_x", 0), getattr(p, "hr_resize_y", 0)
    if not (rx or ry):
        return int(p.width * p.hr_scale), int(p.height * p.hr_scale)
    if not ry:
        return rx, rx * p.height // p.width
    if not rx:
        return ry * p.width // p.height, ry
    return rx, ry


def format_profile_summary(config) -> str:
    # Part of SD Unet labels, which settings store: keep the format stable.
    (bmin, _, hmin, wmin), (bopt, _, hopt, wopt), (bmax, _, hmax, wmax) = config.profile["sample"]
    tok = f", tok={config.profile['encoder_hidden_states'][1][1]}"
    bs_opt, bs_min, bs_max = max(1, bopt // 2), max(1, (bmin + 1) // 2), max(1, bmax // 2)
    opt = f"{wopt * 8}x{hopt * 8}"
    if config.static_shapes or (hmin, wmin, bs_min) == (hmax, wmax, bs_max):
        return f"{opt} (bs={bs_opt}{tok}, Static)"
    if (hmin, wmin) == (hmax, wmax):
        return f"{opt} (bs={bs_min}-{bs_max}{tok}, Dynamic)"
    return f"{wmin * 8}x{hmin * 8}-{wmax * 8}x{hmax * 8} (opt {opt}, bs={bs_min}-{bs_max}{tok}, Dynamic)"


def prompt_tokens(text: str, steps: int) -> int:
    """Context length of a prompt: 77 per chunk of 75 tokens."""
    text = EXTRA_NETWORK_TAG.sub("", text or "")
    try:
        variants = {t for schedule in prompt_parser.get_learned_conditioning_prompt_schedules([text], steps) for _, t in schedule}
        chunks, _ = shared.sd_model.cond_stage_model.process_texts(sorted(variants))
        return max(len(c) for c in chunks) * 77
    except Exception:
        words = re.findall(r"\w+|[^\w\s]", text)
        return max(1, math.ceil(sum(1 + len(w) // 5 for w in words) / 75)) * 77


def batched_token_count(pos: int, neg: int) -> int:
    """reForge repeats cond and uncond to the LCM of their lengths if that's at most 4x the shorter
    (else they run separately); A1111 pads to the longer one."""
    if unet_memory.IS_FORGE_BACKEND:
        lcm = math.lcm(pos, neg)
        if lcm // min(pos, neg) <= 4:
            return lcm
    return max(pos, neg)


def prompt_token_counts(p):
    pos = max(prompt_tokens(t, p.steps) for t in set(p.all_prompts or [p.prompt]))
    neg = max(prompt_tokens(t, p.steps) for t in set(p.all_negative_prompts or [p.negative_prompt]))
    return pos, neg


def unet_shape(p, cfg: float, pos: int, neg: int):
    """(batch, context length) of the UNet input; reForge skips uncond at CFG 1."""
    if unet_memory.IS_FORGE_BACKEND and (cfg < 0 or math.isclose(cfg, 1)):
        return p.batch_size, pos
    return p.batch_size * 2, batched_token_count(pos, neg)


def parse_lora_tags(prompt: str):
    """(name, UNet multiplier) per tag: <lora:name:te:unet>, te=/unet=."""
    loras = []
    for kind, args in EXTRA_NETWORK_TAG.findall(prompt or ""):
        if kind != "lora":
            continue
        parts = args.split(":")
        positional = [x for x in parts if "=" not in x]
        named = dict(x.split("=", 1) for x in parts if "=" in x)
        te = float(named.get("te", positional[1] if len(positional) > 1 else 1.0))
        loras.append((positional[0], float(named.get("unet", positional[2] if len(positional) > 2 else te))))
    return loras


def resolve_trt_lora(tag_name: str, available: dict):
    """Converted LoRAs are named after the file; tags may use the alias from its metadata."""
    if tag_name in available:
        return available[tag_name]
    try:
        from exporter import import_lora_networks

        networks = import_lora_networks()
        if not networks.available_network_aliases:
            networks.list_available_networks()
        entry = networks.available_network_aliases.get(tag_name)
        if entry is not None:
            for key in (entry.name, os.path.splitext(os.path.basename(entry.filename))[0]):
                if key in available:
                    return available[key]
    except Exception as e:
        print(f"[TensorRT] Could not resolve LoRA '{tag_name}': {e}", flush=True)
    return None


def load_lora_refit(base_onnx: str, loras: list) -> dict:
    """Engine weights with the LoRAs [(path, multiplier)] applied, for the weights they change.
    One weight at a time: SDXL LoRAs change gigabytes of them."""
    with contextlib.ExitStack() as stack:
        files = []
        for path, scale in loras:
            f = stack.enter_context(safe_open(path, framework="pt"))
            files.append((f, set(f.keys()), scale))
        missing = set().union(*(keys for _, keys, _ in files))
        total = len(missing)
        refit = {}
        onnx_dir = os.path.dirname(base_onnx)
        for init in onnx.load(base_onnx, load_external_data=False).graph.initializer:
            if init.name not in missing:
                continue
            weight = torch.from_numpy(numpy_helper.to_array(init, base_dir=onnx_dir).astype(np.float32))
            for f, keys, scale in files:
                if init.name in keys:
                    weight += f.get_tensor(init.name).float() * scale
            refit[init.name] = weight.half()
            missing.discard(init.name)
    if len(missing) > 0.02 * total:
        raise RuntimeError(
            f"{len(missing)} of {total} LoRA weights aren't in {os.path.basename(base_onnx)} (a LoRA for another model "
            "type, or converted before the model was exported again): convert it again in the TensorRT LoRA tab"
        )
    return refit


def rank_profiles(configs: List[dict], check) -> dict:
    """{profile index: distance} for the profiles `check(config)` accepts."""
    ranked = {}
    for i, entry in enumerate(configs):
        fits, distance = check(entry["config"])
        if fits:
            ranked[i] = distance
    return ranked


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, model_name: str, configs: List[dict], label: str, forced_profile_idx: int = None):
        self.model_name = model_name
        self.configs = configs
        self.label = label
        self.forced_profile_idx = forced_profile_idx

    def create_unet(self):
        return TrtUnet(self.model_name, self.configs)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict]):
        super().__init__()
        self.model_name = model_name
        self.configs = configs
        self.profile_idx = 0
        self.is_auto = True
        self.loaded_config = None
        self.engine = None
        self.refit_dict = {}
        self.refitted_keys = set()
        self._last_signature = None
        self._selected = False  # this generation's engine was chosen from real inputs
        self._check_output = False
        self._warned_controlnet = False
        self._fallback_reason = None  # set: the rest of this generation runs on PyTorch
        self._torch_restored = False
        self._input_shape = None

    def configure(self, profile_idx: int, is_auto: bool, refit_dict: dict):
        """Settings for a new generation. Engines load at the first UNet call, once the text encoder is done."""
        self._fallback_reason = None
        self._last_signature = None
        self._selected = False
        self.is_auto = is_auto
        reload = self.engine is None or profile_idx != self.profile_idx or not self.refitted_keys.issubset(refit_dict)
        changed = refit_dict is not self.refit_dict
        self.profile_idx, self.refit_dict = profile_idx, refit_dict
        if reload:
            self.release_engine()  # a fresh engine also undoes weights refitted for other LoRAs
        elif changed:
            self._refit()

    def _enter_fallback(self, reason: str, message: str):
        """The rest of this generation runs like with SD Unet = None."""
        self._fallback_reason = reason
        self._last_signature = None
        self.release_engine()
        _free_cuda_cache()
        # Mid-sampling, sampling_prepare already ran without the UNet.
        unet_memory.restore_torch_unet(self._input_shape if unet_memory.in_trt_sampling else None)
        self._torch_restored = True
        _notify(message)

    def forward(self, x, timesteps=None, context=None, *args, **kwargs):
        self._input_shape = tuple(x.shape)
        if self._fallback_reason is None and self.engine is None:
            self.activate()
        if self._fallback_reason is not None:
            raise TrtFallback(self._fallback_reason)
        if kwargs.get("control") is not None and not self._warned_controlnet:
            self._warned_controlnet = True
            print("[TensorRT] Warning: TensorRT engines ignore ControlNet. Set SD Unet to None to use it.", flush=True)

        timesteps = kwargs.get("timesteps") if timesteps is None else timesteps
        context = kwargs.get("context", kwargs.get("encoder_hidden_states")) if context is None else context
        y = kwargs.get("y", args[0] if args else None)
        b = x.shape[0]
        timesteps = timesteps.reshape(-1)
        feed_dict = {
            "sample": x,
            "timesteps": timesteps if timesteps.shape[0] == b else timesteps.expand(b),
            "encoder_hidden_states": context if context.shape[0] == b else context.expand(b, -1, -1),
        }
        if y is not None:
            feed_dict["y"] = y if y.shape[0] == b else y.expand(b, -1)
        feed_dict = {k: v if v.is_cuda else v.to(devices.device) for k, v in feed_dict.items()}

        signature = tuple((k, tuple(v.shape)) for k, v in feed_dict.items())
        if signature != self._last_signature:
            try:
                self._select_profile(feed_dict)
            except NoProfileError as e:
                self._enter_fallback("unsupported input shapes", f"{e} Finishing this generation with PyTorch.")
            if self._fallback_reason is not None:
                raise TrtFallback(self._fallback_reason)
            self._last_signature = signature
            self._check_output = True

        feed_dict = self._pad_to_profile(feed_dict)
        stream = torch.cuda.current_stream().cuda_stream
        try:
            out = self.engine.infer(feed_dict, stream)["latent"]
        except torch.cuda.OutOfMemoryError:
            print("[TensorRT] Out of GPU memory; freeing VRAM and retrying...", flush=True)
            unet_memory.make_room(self.engine.needed_activation_bytes() + HEADROOM)
            _free_cuda_cache()
            try:
                out = self.engine.infer(feed_dict, stream)["latent"]
            except torch.cuda.OutOfMemoryError:
                self._enter_fallback(
                    "out of GPU memory",
                    "Out of GPU memory running the engine; finishing this generation with PyTorch. "
                    "A smaller engine profile or freeing VRAM avoids this.",
                )
                raise TrtFallback(self._fallback_reason)
        if self._check_output:
            self._check_output = False
            if not torch.isfinite(out).all():
                self._enter_fallback(
                    "non-finite output",
                    "The engine produced NaN/Inf values; finishing this generation with PyTorch. Rebuild the engine "
                    "(Force Rebuild), and report it if that doesn't help.",
                )
                raise TrtFallback(self._fallback_reason)
        return out[:b] if out.shape[0] != b else out

    def _select_profile(self, feed_dict):
        current = self.loaded_config["config"] if self.loaded_config else None
        compatible = current is not None and current.is_compatible_from_dict(feed_dict)[0]
        # Keep a chosen engine that fits: shapes change mid-generation (CFG schedules, NGMS, ...).
        if compatible and (self._selected or not self.is_auto):
            return
        self._selected = True
        candidates = rank_profiles(self.configs, lambda c: c.is_compatible_from_dict(feed_dict))
        if not candidates:
            if compatible:
                return
            x, ctx = feed_dict["sample"], feed_dict["encoder_hidden_states"]
            raise NoProfileError(
                f"No engine profile of {self.model_name} covers {x.shape[3] * 8}x{x.shape[2] * 8}, "
                f"batch {x.shape[0]}, {ctx.shape[1]} tokens."
            )
        best = min(candidates, key=candidates.get)
        if best != self.profile_idx:
            self.profile_idx = best
            self.switch_engine()

    def _pad_to_profile(self, feed_dict):
        """Pads by repetition, which leaves the result unchanged: the output is sliced back, and a
        repeated context leaves cross-attention unchanged."""
        config = self.loaded_config["config"]
        b, min_b = feed_dict["sample"].shape[0], config.min_batch()
        if b < min_b:
            reps = -(-min_b // b)
            feed_dict = {k: v.repeat(reps, *([1] * (v.dim() - 1)))[:min_b] for k, v in feed_dict.items()}
        ctx, min_t = feed_dict["encoder_hidden_states"], config.min_tokens()
        if ctx.shape[1] < min_t and min_t % ctx.shape[1] == 0:
            feed_dict["encoder_hidden_states"] = ctx.repeat(1, min_t // ctx.shape[1], 1)
        return feed_dict

    def _refit(self):
        self.refitted_keys = set()
        if not self.refit_dict or self.engine is None:
            return
        try:
            self.engine.refit(self.refit_dict)
            self.refitted_keys = set(self.refit_dict)
        except Exception as e:
            self._enter_fallback("the LoRA couldn't be applied", f"Couldn't apply the LoRA to the engine: {e} Using PyTorch for this generation.")

    def switch_engine(self):
        self.loaded_config = self.configs[self.profile_idx]
        if self._fallback_reason is None:
            self._last_signature = None
            self.activate()

    def activate(self):
        self.loaded_config = self.configs[self.profile_idx]
        path = os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
        if self.engine is None:
            self.engine = Engine(path)
        else:
            self.engine.reset(path)
        self.engine.before_allocate = lambda size: unet_memory.make_room(size + HEADROOM)
        if self._torch_restored:
            unet_memory.offload_torch_unet()
            self._torch_restored = False
        with contextlib.suppress(OSError):
            unet_memory.make_room(os.path.getsize(path) + HEADROOM)
        _free_cuda_cache()
        try:
            self.engine.load()
            self.engine.activate()
        except Exception as e:
            self._enter_fallback("the engine couldn't be loaded", f"{e}\nUsing PyTorch for this generation.")
            return
        print(f"[TensorRT] Loaded Profile {self.profile_idx}:\n{self.engine}", flush=True)
        self._refit()

    def release_activation_memory(self):
        if self.engine is not None and self.engine.engine is not None:
            self.engine.release_activation_memory()

    def release_engine(self):
        if self.engine is not None:
            self.engine.reset()
        self.engine = None
        self.loaded_config = None
        self.refitted_keys = set()

    def deactivate(self):
        self.release_engine()
        unet_memory.restore_torch_unet()


def _trt_active() -> bool:
    unet = sd_unet.current_unet
    return isinstance(unet, TrtUnet) and unet._fallback_reason is None


def _release_activation_memory():
    unet = sd_unet.current_unet
    if isinstance(unet, TrtUnet):
        unet.release_activation_memory()


def _loaded_engine():
    unet = sd_unet.current_unet
    if isinstance(unet, TrtUnet) and unet.engine is not None and unet.engine.engine is not None:
        return unet
    return None


def _before_model_load(models):
    """reForge can't reclaim an engine's VRAM: if weights it's about to load won't fully fit next to
    the engine (partially loaded, SDXL's text encoder breaks), the engine makes way until sampling."""
    unet = _loaded_engine()
    if unet is None or unet_memory.in_trt_sampling:
        return
    needed = unet_memory.weights_to_load(models)
    if needed <= 0 or needed + unet_memory.minimum_inference_memory() <= unet_memory.vram_available_for(models):
        return
    names = sorted({type(getattr(m, "model", m)).__name__ for m in models if m is not None and not unet_memory.is_unet(m)})
    print(f"[TensorRT] Releasing the engine to make room for {', '.join(names)}; it's reloaded for sampling.", flush=True)
    unet.release_engine()
    _free_cuda_cache()


unet_memory.callbacks.update(
    trt_active=_trt_active,
    sampling_done=_release_activation_memory,
    before_model_load=_before_model_load,
)


def profile_choices() -> List[str]:
    model = getattr(getattr(sd_models, "model_data", None), "sd_model", None)  # not shared.sd_model: that loads one
    configs = modelmanager.available_models().get(model.sd_checkpoint_info.model_name if model else None, [])
    return [AUTO_PROFILE] + [f"Profile {i}: {format_profile_summary(c['config'])}" for i, c in enumerate(configs)]


def select_profiles(p, configs: List[dict], pos: int, neg: int, is_img2img: bool):
    """(base profile, Hires. fix profile); None runs that pass on PyTorch."""

    def candidates(width, height, cfg):
        batch, tokens = unet_shape(p, cfg, pos, neg)
        return rank_profiles(configs, lambda c: c.is_compatible(width, height, batch, tokens))

    base = candidates(p.width, p.height, p.cfg_scale)
    if not base:
        return None, None
    best = min(base, key=base.get)
    if is_img2img or not p.enable_hr:
        return best, best
    hires = candidates(*hires_size(p), getattr(p, "hr_cfg", 0) or p.cfg_scale)
    both = [i for i in base if i in hires]
    if both:
        best = min(both, key=lambda i: base[i] + hires[i])
        return best, best
    return best, min(hires, key=hires.get) if hires else None


class TensorRTScript(scripts.Script):
    def __init__(self):
        self.idx = None
        self.hr_idx = None
        self.is_auto = True
        self.lora_key = None
        self.lora_refit_dict = {}

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion("TensorRT Profile", open=False):
            with gr.Row():
                override = gr.Dropdown(
                    label="Profile Selection Override",
                    choices=profile_choices(),
                    value=AUTO_PROFILE,
                    elem_id=f"trt_profile_override_{tab}",
                    info="Engines exported for the current checkpoint. Refresh after changing checkpoints or exporting.",
                )
                refresh = ToolButton(value=refresh_symbol, elem_id=f"trt_profile_refresh_{tab}")

        def refresh_choices(current):
            choices = profile_choices()
            return gr.update(choices=choices, value=current if current in choices else AUTO_PROFILE)

        refresh.click(refresh_choices, inputs=[override], outputs=[override])
        return [override]

    def before_process(self, p, *args):
        if shared.opts.sd_unet == "None":
            return
        sizes = [(p.width, p.height)] + ([hires_size(p)] if not self.is_img2img and p.enable_hr else [])
        if any(w % 64 or h % 64 for w, h in sizes):
            gr.Warning("TensorRT: image sizes should be divisible by 64.")

    def get_loras(self, p, model_name: str):
        loras = parse_lora_tags((p.all_prompts or [p.prompt])[0])  # the WebUI applies the first prompt's LoRAs
        key = (model_name, loras)
        if key == self.lora_key:
            return
        refit = {}
        if loras:
            available = modelmanager.available_loras()
            paths = [resolve_trt_lora(name, available) for name, _ in loras]
            missing = [name for (name, _), path in zip(loras, paths) if path is None]
            if missing:
                raise MissingTrtLora(missing)
            print(f"[TensorRT] Applying LoRAs: {', '.join(f'{n}:{s}' for n, s in loras)}", flush=True)
            refit = load_lora_refit(onnx_path(model_name), [(path, s) for path, (_, s) in zip(paths, loras)])
        self.lora_key, self.lora_refit_dict = key, refit

    def process(self, p, *args):
        unet_memory.install_hooks()
        unet_memory.in_trt_sampling = False  # an interrupted generation never cleans up
        option = sd_unet.get_unet_option()
        model_name = current_model_name()
        if option is None and shared.opts.sd_unet == "Automatic" and model_name in modelmanager.available_models():
            sd_unet.list_unets()
            option = sd_unet.get_unet_option()
        if not isinstance(option, TrtUnetOption):
            return

        self.idx = None
        if option.model_name != model_name:
            _notify(f"The selected engine is for {option.model_name}, but {model_name} is loaded; using PyTorch. Select Automatic in SD Unet.")
            return self.apply_unet(option)

        override = re.match(r"Profile (\d+)", args[0]) if args and isinstance(args[0], str) else None
        forced = int(override.group(1)) if override else option.forced_profile_idx
        self.is_auto = forced is None or forced >= len(option.configs)
        if forced is not None and self.is_auto:
            gr.Warning(f"TensorRT: Profile {forced} doesn't exist; selecting a profile automatically.")
        if self.is_auto:
            pos, neg = prompt_token_counts(p)
            self.idx, self.hr_idx = select_profiles(p, option.configs, pos, neg, self.is_img2img)
            if self.idx is None:
                _notify(
                    f"No engine profile for {model_name} at {p.width}x{p.height}, batch size {p.batch_size}, "
                    f"{pos} prompt / {neg} negative tokens; using PyTorch. Export a matching engine from the TensorRT tab."
                )
                return self.apply_unet(option)
            print(f"[TensorRT] Profile {self.idx} (Hires. fix: {self.hr_idx}) for {pos} prompt / {neg} negative tokens", flush=True)
        else:
            self.idx = self.hr_idx = forced

        try:
            self.get_loras(p, model_name)
        except Exception as e:
            self.idx = None
            _notify(f"{e}; using PyTorch for this generation.")
        self.apply_unet(option)

    def apply_unet(self, option):
        """TensorRT with profile self.idx, or PyTorch if it's None."""
        current = sd_unet.current_unet
        sd_unet.current_unet_option = option  # so the WebUI's own apply_unet keeps this
        if self.idx is not None and isinstance(current, TrtUnet) and current.option is option:
            return current.configure(self.idx, self.is_auto, self.lora_refit_dict)
        if current is not None:
            getattr(current, "release_engine", current.deactivate)()
        if self.idx is None:
            sd_unet.current_unet = None
            unet_memory.restore_torch_unet()
            return
        unet_memory.offload_torch_unet()
        unet = sd_unet.current_unet = option.create_unet()
        unet.option = option
        print(f"[TensorRT] Activating {option.label}", flush=True)
        unet.configure(self.idx, self.is_auto, self.lora_refit_dict)

    def post_sample(self, p, ps, *args):
        _release_activation_memory()

    def before_hr(self, p, *args):
        unet = sd_unet.current_unet
        if isinstance(unet, TrtUnet) and self.hr_idx is not None and unet.profile_idx != self.hr_idx:
            unet.profile_idx = self.hr_idx
            unet.switch_engine()


def list_unets(options):
    # The same labels however many engines a checkpoint has: settings store them.
    for model_name, configs in modelmanager.available_models().items():
        options.append(TrtUnetOption(model_name, configs, f"[TRT] {model_name} (Auto)"))
        for i, entry in enumerate(configs):
            label = f"[TRT] {model_name} [Profile {i}: {format_profile_summary(entry['config'])}]"
            options.append(TrtUnetOption(model_name, configs, label, forced_profile_idx=i))
    try:  # checkpoints with one engine used to be labelled "[TRT] name"
        saved, labels = shared.opts.sd_unet, {o.label for o in options}
        if saved.startswith("[TRT] ") and saved not in labels and f"{saved} (Auto)" in labels:
            shared.opts.data["sd_unet"] = f"{saved} (Auto)"
    except Exception:
        pass


unet_memory.install_hooks()
script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
