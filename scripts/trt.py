import os
import sys
import re
import math
import importlib.util
from typing import List

# Compatibility patch for newer onnx versions (>= 1.17) where float32_to_bfloat16 was removed from onnx.helper
try:
    import onnx
    import onnx.helper
    import struct

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        def _float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
            ival = int.from_bytes(struct.pack("<f", float(fval)), "little")
            if truncate:
                return ival >> 16
            return (ival + 0x8000 + ((ival >> 16) & 1)) >> 16

        onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

    if not hasattr(onnx.helper, "bfloat16_to_float32"):
        def _bfloat16_to_float32(ival: int) -> float:
            return struct.unpack("<f", (int(ival) << 16).to_bytes(4, "little"))[0]

        onnx.helper.bfloat16_to_float32 = _bfloat16_to_float32
except Exception:
    pass

import numpy as np
import torch
from torch.cuda import nvtx
import gradio as gr
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from modules import script_callbacks, sd_unet, devices, scripts, shared

import ui_trt
from utilities import Engine
from model_manager import TRT_MODEL_DIR, modelmanager
from datastructures import ModelType
from scripts.lora import apply_loras
import unet_memory


def format_profile_summary(config) -> str:
    try:
        sample = config.profile.get("sample", None)
        emb = config.profile.get("encoder_hidden_states", None)
        if not sample:
            return "Static" if getattr(config, "static_shapes", False) else "Dynamic"
        _min, _opt, _max = sample
        opt_bs = max(1, _opt[0] // 2)
        opt_h = _opt[2] * 8
        opt_w = _opt[3] * 8
        tok_str = ""
        if emb and len(emb) > 1 and len(emb[1]) > 1:
            tok_str = f", tok={emb[1][1]}"

        if getattr(config, "static_shapes", False):
            return f"{opt_w}x{opt_h} (bs={opt_bs}{tok_str}, Static)"
        else:
            min_w, max_w = _min[3] * 8, _max[3] * 8
            min_h, max_h = _min[2] * 8, _max[2] * 8
            # sample batch = 2 x batch size (cond + uncond); the minimum may be 1x for CFG 1
            min_bs, max_bs = max(1, (_min[0] + 1) // 2), max(1, _max[0] // 2)
            if min_w == max_w and min_h == max_h and min_bs == max_bs:
                return f"{opt_w}x{opt_h} (bs={opt_bs}{tok_str}, Static)"
            if min_w == max_w and min_h == max_h:
                return f"{opt_w}x{opt_h} (bs={min_bs}-{max_bs}{tok_str}, Dynamic)"
            return f"{min_w}x{min_h}-{max_w}x{max_h} (opt {opt_w}x{opt_h}, bs={min_bs}-{max_bs}{tok_str}, Dynamic)"
    except Exception:
        return "Static" if getattr(config, "static_shapes", False) else "Dynamic"


def count_clip_tokens(text: str) -> int:
    if not text or not str(text).strip():
        return 77
    text = re.sub(r"<[^>]+>", "", str(text).strip())

    # 1. Try official tokenizer from loaded model
    try:
        if hasattr(shared, "sd_model") and shared.sd_model is not None:
            cond_stage = getattr(shared.sd_model, "cond_stage_model", None)
            if cond_stage is not None:
                if hasattr(cond_stage, "tokenize"):
                    rem = cond_stage.tokenize([text])
                    if isinstance(rem, list) and len(rem) > 0:
                        tok_len = len(rem[0]) if isinstance(rem[0], list) else len(rem)
                        chunks = max(1, (tok_len + 74) // 75)
                        return chunks * 77
                tok = getattr(cond_stage, "tokenizer", None) or getattr(
                    getattr(cond_stage, "wrapped", None), "tokenizer", None
                )
                if tok is not None and hasattr(tok, "encode"):
                    tok_len = len(tok.encode(text))
                    chunks = max(1, (tok_len + 74) // 75)
                    return chunks * 77
            if getattr(shared, "sd_model", None) is not None and hasattr(shared.sd_model, "forge_objects"):
                # ForgeSD is a plain object (.unet/.clip/.vae), not a dict.
                clip = getattr(shared.sd_model.forge_objects, "clip", None)
                if clip is not None:
                    tok = getattr(clip, "tokenizer", None)
                    if tok is not None and hasattr(tok, "tokenize_with_weights"):
                        # {clip_name: [chunk, ...]}, each chunk being one 77-token window
                        tokens = tok.tokenize_with_weights(text)
                        chunks = max([len(v) for v in tokens.values()]) if tokens else 0
                        if chunks > 0:
                            return chunks * 77
    except Exception:
        pass

    # 2. Heuristic CLIP BPE tokenizer estimation
    tokens_raw = re.findall(r"\w+|[^\w\s]", text)
    bpe_estimate = 0.0
    for t in tokens_raw:
        if len(t) <= 3:
            bpe_estimate += 1.0
        elif len(t) <= 7:
            bpe_estimate += 1.3
        elif len(t) <= 12:
            bpe_estimate += 2.0
        else:
            bpe_estimate += 3.0
    total = int(bpe_estimate) + 2  # BOS and EOS tokens
    chunks = max(1, (total + 74) // 75)
    return chunks * 77


IS_FORGE_BACKEND = importlib.util.find_spec("ldm_patched") is not None


def batched_token_count(pos_tokens: int, neg_tokens: int) -> int:
    """Token length the UNet will actually see when cond and uncond are batched.

    reForge/Forge (ComfyUI sampling) pad both to the least common multiple of their
    lengths by repetition, as long as that is at most 4x the shorter one; otherwise they
    run separately. A1111 pads to the longer one."""
    pos_tokens, neg_tokens = max(pos_tokens, 77), max(neg_tokens, 77)
    if IS_FORGE_BACKEND:
        lcm = math.lcm(pos_tokens, neg_tokens)
        if lcm // min(pos_tokens, neg_tokens) <= 4:
            return lcm
    return max(pos_tokens, neg_tokens)


def get_max_prompt_token_count(p) -> (int, dict):
    pos_texts = []
    if hasattr(p, "prompt") and p.prompt:
        pos_texts.append(str(p.prompt))
    if hasattr(p, "all_prompts") and p.all_prompts:
        pos_texts.extend([str(x) for x in p.all_prompts if x])

    neg_texts = []
    if hasattr(p, "negative_prompt") and p.negative_prompt:
        neg_texts.append(str(p.negative_prompt))
    if hasattr(p, "all_negative_prompts") and p.all_negative_prompts:
        neg_texts.extend([str(x) for x in p.all_negative_prompts if x])

    pos_tokens = max([count_clip_tokens(t) for t in pos_texts]) if pos_texts else 77
    neg_tokens = max([count_clip_tokens(t) for t in neg_texts]) if neg_texts else 77
    final_tokens = batched_token_count(pos_tokens, neg_tokens)

    has_wildcards = any("__" in t or "{" in t for t in pos_texts + neg_texts)
    info = {
        "pos": pos_tokens,
        "neg": neg_tokens,
        "final": final_tokens,
        "has_wildcards": has_wildcards,
    }
    return final_tokens, info


class MissingTrtLora(Exception):
    def __init__(self, names):
        super().__init__(f"LoRA not converted for TensorRT: {', '.join(names)}")
        self.names = names


def resolve_trt_lora(tag_name: str, available: dict):
    """Path of the converted .lora for a prompt tag. Tags may use the file name or the alias
    from the LoRA's metadata (reForge's LoRA cards insert the alias by default), so resolve them
    through the LoRA extension like the WebUI does. Converted files are named after the file."""
    if tag_name in available:
        return available[tag_name]
    try:
        from exporter import _import_lora_networks

        networks = _import_lora_networks()
        entry = networks.available_network_aliases.get(tag_name)
        if entry is None and not networks.available_network_aliases:
            networks.list_available_networks()
            entry = networks.available_network_aliases.get(tag_name)
        if entry is not None:
            for key in (entry.name, os.path.splitext(os.path.basename(entry.filename))[0]):
                if key in available:
                    return available[key]
    except Exception as e:
        print(f"[TensorRT] Could not resolve LoRA alias '{tag_name}': {e}", flush=True)
    return None


class NoProfileError(RuntimeError):
    pass


class TrtFallback(Exception):
    """Raised by TrtUnet.forward to have the hooked UNet forward run PyTorch instead."""


def _notify(message: str):
    print(f"[TensorRT] {message}", flush=True)
    try:
        gr.Warning(f"TensorRT: {message}")
    except Exception:
        pass


def _free_cuda_cache():
    try:
        import ldm_patched.modules.model_management as model_management

        model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    torch.cuda.empty_cache()


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(
        self,
        name: str,
        filename: List[dict],
        forced_profile_idx: int = None,
        custom_label: str = None,
    ):
        self.model_name = name
        self.configs = filename
        self.forced_profile_idx = forced_profile_idx
        if custom_label:
            self.label = custom_label
        elif forced_profile_idx is not None:
            self.label = f"[TRT] {name} [Profile {forced_profile_idx}]"
        elif len(filename) > 1:
            self.label = f"[TRT] {name} (Auto)"
        else:
            self.label = f"[TRT] {name}"

    def create_unet(self):
        unet = TrtUnet(self.model_name, self.configs)
        unet.is_auto = (self.forced_profile_idx is None)
        if self.forced_profile_idx is not None:
            unet.profile_idx = self.forced_profile_idx
        return unet


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = None
        self.model_name = model_name
        self.configs = configs

        self.profile_idx = 0
        self.loaded_config = None
        self.is_auto = True

        self.refitted_keys = set()
        self._refit_dict = {}  # active LoRA weights, re-applied whenever an engine is (re)loaded

        self.engine = None
        self.step_idx = 0
        self._last_signature = None
        self._warned_unsupported = set()
        self._fallback_reason = None  # set for the rest of a generation after an unrecoverable OOM
        self._unsupported_signature = None  # input shapes no profile covers -> run PyTorch for them

    def reset_generation_state(self):
        self.step_idx = 0
        self._fallback_reason = None
        self._unsupported_signature = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._fallback_reason is not None:
            raise TrtFallback(self._fallback_reason)
        if self.engine is None or getattr(self.engine, "engine", None) is None:
            # The engine was released after an OOM fallback: free the PyTorch UNet's VRAM again
            # (as apply_unet does on activation) and reload the engine.
            unet_memory.offload_torch_unet()
            self.activate()

        nvtx.range_push("forward")

        self.step_idx = getattr(self, "step_idx", 0) + 1

        # Features the TensorRT engine cannot express: tell the user once instead of failing silently.
        if kwargs.get("control") is not None and "control" not in self._warned_unsupported:
            self._warned_unsupported.add("control")
            print(
                "[TensorRT] Warning: ControlNet inputs are ignored by TensorRT engines. "
                "Set SD Unet to 'None' to use ControlNet.",
                flush=True,
            )

        if timesteps is None and "timesteps" in kwargs:
            timesteps = kwargs["timesteps"]
        if context is None:
            if "context" in kwargs:
                context = kwargs["context"]
            elif "encoder_hidden_states" in kwargs:
                context = kwargs["encoder_hidden_states"]

        b = x.shape[0]
        if timesteps is not None:
            if timesteps.dim() == 0:
                timesteps = timesteps.unsqueeze(0).repeat(b)
            elif timesteps.shape[0] == 1 and b > 1:
                timesteps = timesteps.repeat(b)
            elif timesteps.shape[0] != b:
                timesteps = timesteps.expand(b)

        if context is not None and context.shape[0] == 1 and b > 1:
            context = context.repeat(b, 1, 1)

        feed_dict = {
            "sample": x,
            "timesteps": timesteps,
            "encoder_hidden_states": context,
        }

        y = kwargs.get("y", None)
        if y is None and len(args) > 0 and args[0] is not None:
            y = args[0]
        if y is not None:
            if y.shape[0] == 1 and b > 1:
                y = y.repeat(b, 1)
            feed_dict["y"] = y

        for k, v in list(feed_dict.items()):
            if isinstance(v, torch.Tensor) and v.device.type != "cuda":
                feed_dict[k] = v.to(device=devices.device)

        # Re-check the profile whenever the input shapes change (first step, Hires. fix pass,
        # CFG 1 steps, ...), not only on the first step.
        signature = tuple((k, tuple(v.shape)) for k, v in feed_dict.items() if v is not None)
        if signature == self._unsupported_signature:
            nvtx.range_pop()
            raise TrtFallback("unsupported input shapes")
        if signature != self._last_signature:
            try:
                self._select_profile(feed_dict)
            except NoProfileError as e:
                self._unsupported_signature = signature
                nvtx.range_pop()
                _notify(f"{e} Using PyTorch for these steps.")
                raise TrtFallback(str(e))
            self._last_signature = signature

        feed_dict = self._pad_to_profile(feed_dict)

        out = None
        for attempt in range(2):
            try:
                stream = torch.cuda.current_stream().cuda_stream
                out = self.engine.infer(feed_dict, stream)["latent"]
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == 0:
                    print("[TensorRT] Out of GPU memory; freeing cached memory and retrying...", flush=True)
                    _free_cuda_cache()
                    continue
                # Give the engine's memory back so PyTorch can run, and stay on PyTorch for this generation.
                self._fallback_reason = "out of GPU memory"
                self.engine.reset()
                self._last_signature = None
                _free_cuda_cache()
                nvtx.range_pop()
                _notify(
                    "Out of GPU memory running the engine; finishing this generation with PyTorch. "
                    "A smaller engine profile or freeing VRAM avoids this."
                )
                raise TrtFallback(self._fallback_reason)

        nvtx.range_pop()
        return out[:b] if out.shape[0] != b else out

    def _select_profile(self, feed_dict: dict):
        cur_config = self.loaded_config["config"] if self.loaded_config else None
        is_compat = cur_config is not None and cur_config.is_compatible_from_dict(feed_dict)[0]
        if is_compat and not self.is_auto:
            return

        valid_models, distances, idx = modelmanager.get_valid_models_from_dict(self.model_name, feed_dict)
        if not valid_models:
            if is_compat:
                return
            x = feed_dict["sample"]
            ctx = feed_dict.get("encoder_hidden_states")
            raise NoProfileError(
                f"No engine profile of {self.model_name} covers {x.shape[3] * 8}x{x.shape[2] * 8}, "
                f"batch {x.shape[0]}, {ctx.shape[1] if ctx is not None else '?'} tokens."
            )
        best_idx = idx[int(np.argmin(distances))]
        if best_idx != self.profile_idx or self.loaded_config is None:
            x = feed_dict["sample"]
            ctx = feed_dict.get("encoder_hidden_states")
            print(
                f"[TensorRT] Switching to Profile {best_idx} for {x.shape[3] * 8}x{x.shape[2] * 8}, "
                f"batch {x.shape[0]}, {ctx.shape[1] if ctx is not None else '?'} tokens",
                flush=True,
            )
            self.profile_idx = best_idx
            self.switch_engine()

    def _pad_to_profile(self, feed_dict: dict) -> dict:
        """Pad inputs below the profile minimum by repetition, which doesn't change the result:
        the batch (e.g. CFG 1 with a static 2B engine) and the token axis (repeating all
        context tokens leaves cross-attention unchanged)."""
        config = self.loaded_config["config"]
        b = feed_dict["sample"].shape[0]
        min_b = config.min_batch()
        if b < min_b:
            reps = -(-min_b // b)
            feed_dict = {
                k: (v.repeat(reps, *([1] * (v.dim() - 1)))[:min_b] if v is not None else v)
                for k, v in feed_dict.items()
            }
        ctx = feed_dict.get("encoder_hidden_states")
        min_t = config.min_tokens()
        if ctx is not None and ctx.shape[1] < min_t and min_t % ctx.shape[1] == 0:
            feed_dict["encoder_hidden_states"] = ctx.repeat(1, min_t // ctx.shape[1], 1)
        return feed_dict

    def apply_loras(self, refit_dict: dict):
        self._refit_dict = refit_dict or {}
        if not self.refitted_keys.issubset(set(self._refit_dict.keys())):
            # Weights refitted before but not part of this LoRA set: reload the engine to reset them.
            # (switch_engine re-applies self._refit_dict after loading.)
            self.switch_engine()
            return
        self._refit_current_engine()

    def _refit_current_engine(self):
        if not self._refit_dict:
            self.refitted_keys = set()
            return
        try:
            self.engine.refit_from_dict(self._refit_dict, is_fp16=True)
            self.refitted_keys = set(self._refit_dict.keys())
        except Exception as e:
            self.refitted_keys = set()
            _notify(f"Couldn't apply LoRA weights to the engine ({e}). Generating without the LoRA.")

    def switch_engine(self):
        self.loaded_config = self.configs[self.profile_idx]
        engine_path = os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
        print(f"[TensorRT] Switching to Profile {self.profile_idx}: {engine_path}", flush=True)
        if self.engine is not None:
            self.engine.reset(engine_path)
        else:
            self.engine = Engine(engine_path)
        self.step_idx = 0
        self._last_signature = None
        self.activate()

    def activate(self):
        self.loaded_config = self.configs[self.profile_idx]
        engine_path = os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
        if self.engine is None:
            self.engine = Engine(engine_path)
        elif getattr(self.engine, "engine", None) is None:
            self.engine.engine_path = engine_path
        # TensorRT allocates with cudaMalloc, outside PyTorch: VRAM PyTorch has cached but isn't
        # using is unavailable to it, so return that to the driver before loading the engine.
        _free_cuda_cache()
        self.engine.load()
        print(f"[TensorRT] Loaded Profile {self.profile_idx}:\n{self.engine}", flush=True)
        self.engine.activate()
        # A freshly loaded engine has the base weights: re-apply the active LoRAs, otherwise any
        # profile switch (step-1 check, Hires. fix, reload after a fallback) silently drops them.
        self.refitted_keys = set()
        self._refit_current_engine()

    def release_activation_memory(self):
        if self.engine is not None and getattr(self.engine, "engine", None) is not None:
            self.engine.release_activation_memory()

    def release_engine(self):
        """Free the engine's VRAM without touching the PyTorch UNet."""
        if self.engine is not None:
            self.engine.reset()
        self.engine = None

    def deactivate(self):
        """Called by the WebUI when switching away from TensorRT: free the engine, then give the
        PyTorch UNet back its normal (fully loaded) state."""
        self.release_engine()
        unet_memory.restore_torch_unet()


_patched_unet_classes = set()


def _trt_active() -> bool:
    unet = sd_unet.current_unet
    return isinstance(unet, TrtUnet) and unet._fallback_reason is None


def _release_activation_memory():
    unet = sd_unet.current_unet
    if isinstance(unet, TrtUnet):
        unet.release_activation_memory()


def patch_unet_forward():
    """
    Hooks UNetModel.forward in ldm_patched, ldm, and sgm to redirect to sd_unet.current_unet when active.
    This is required for Stable Diffusion WebUI reForge (and Forge), which uses ldm_patched as its
    execution backend and does not have the classic A1111 sd_hijack unet forward patch enabled.
    """
    unet_memory.install_sampling_hook(_trt_active, on_sampling_done=_release_activation_memory)

    targets = [
        ("ldm_patched.ldm.modules.diffusionmodules.openaimodel", "UNetModel"),
        ("ldm.modules.diffusionmodules.openaimodel", "UNetModel"),
        ("sgm.modules.diffusionmodules.openaimodel", "UNetModel"),
    ]

    for mod_name, cls_name in targets:
        try:
            import importlib
            mod = sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
            cls = getattr(mod, cls_name, None)
            if cls is None or getattr(cls, "_trt_patched", False):
                continue

            orig_forward = cls.forward

            def make_replacement(original_fn, module_label):
                def unet_forward(self, x, timesteps=None, context=None, *args, **kwargs):
                    trt_unet = sd_unet.current_unet
                    if trt_unet is None:
                        return original_fn(self, x, timesteps, context, *args, **kwargs)
                    try:
                        return trt_unet.forward(x, timesteps, context, *args, **kwargs)
                    except TrtFallback:
                        pass
                    # PyTorch fallback. The UNet weights may have been moved off the GPU for TensorRT.
                    unet_memory.ensure_module_on(self, x.device)
                    # Hide the TensorRT UNet so wrapped forwards (e.g. A1111's own hook) don't call it again.
                    sd_unet.current_unet = None
                    try:
                        return original_fn(self, x, timesteps, context, *args, **kwargs)
                    finally:
                        sd_unet.current_unet = trt_unet
                return unet_forward

            cls._orig_trt_forward = orig_forward
            cls.forward = make_replacement(orig_forward, mod_name)
            cls._trt_patched = True
            print(f"[TensorRT] Successfully hijacked {mod_name}.{cls_name}.forward -> sd_unet.current_unet", flush=True)
        except Exception:
            pass


AUTO_PROFILE = "Auto (Best Match / SD Unet default)"  # same text as before, so saved UI settings stay valid


def profile_choices(model_name: str = None) -> List[str]:
    """Profile override choices: Auto plus the engines that actually exist for the checkpoint."""
    if model_name is None:
        info = getattr(getattr(shared, "sd_model", None), "sd_checkpoint_info", None)
        model_name = getattr(info, "model_name", None)
    choices = [AUTO_PROFILE]
    try:
        for i, entry in enumerate(modelmanager.available_models().get(model_name, [])):
            choices.append(f"Profile {i}: {format_profile_summary(entry['config'])}")
    except Exception:
        pass
    return choices


class TensorRTScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        self.lora_hash = ""
        self.update_lora = False
        self.lora_refit_dict = {}
        self.idx = None
        self.hr_idx = None
        self.torch_unet = False
        self.is_auto = True

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion("TensorRT Profile", open=False):
            with gr.Row():
                profile_override = gr.Dropdown(
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

        refresh.click(refresh_choices, inputs=[profile_override], outputs=[profile_override])
        return [profile_override]

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):  # 1
        if getattr(shared.opts, "sd_unet", "Automatic") == "None":
            return
        # TensorRT engines need latents divisible by 8 (images divisible by 64).
        if p.width % 64 or p.height % 64:
            gr.Warning("TensorRT: resolution should be divisible by 64 in both dimensions.")

        if self.is_img2img:
            return
        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                gr.Warning(
                    "TensorRT: Hires. fix resolution should be divisible by 64 in both dimensions."
                )

    def get_profile_idx(self, p, model_name: str, model_type: ModelType, prompt_tokens: int = None) -> (int, int):
        best_hr = None

        if self.is_img2img:
            hr_scale = 1
        else:
            hr_scale = p.hr_scale if p.enable_hr else 1

        if prompt_tokens is None:
            prompt_tokens, _ = get_max_prompt_token_count(p)

        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            model_name,
            p.width,
            p.height,
            p.batch_size,
            prompt_tokens,
        )
        if len(valid_models) == 0 and prompt_tokens != 77:
            (
                valid_models,
                distances,
                idx,
            ) = modelmanager.get_valid_models(
                model_name,
                p.width,
                p.height,
                p.batch_size,
                77,
            )

        if len(valid_models) == 0:
            gr.Warning(
                f"TensorRT: no engine profile for {model_name} at {p.width}x{p.height}, batch {p.batch_size}, "
                f"{prompt_tokens} tokens. Using PyTorch instead; export a matching engine from the TensorRT tab."
            )
            return None, None

        best = idx[np.argmin(distances)]
        best_hr = best

        print(
            f"[TensorRT] Candidate profiles for {p.width}x{p.height} bs={p.batch_size} (prompt tokens ~{prompt_tokens}):",
            flush=True,
        )
        for cand_idx, cand_dist in zip(idx, distances):
            cand_conf = valid_models[idx.index(cand_idx)]["config"]
            cand_desc = format_profile_summary(cand_conf)
            is_chosen = " <-- SELECTED" if cand_idx == best else ""
            print(f"  - Profile {cand_idx} [{cand_desc}]: distance score = {cand_dist:.1f}{is_chosen}", flush=True)

        if hr_scale != 1:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                model_name,
                hr_w,
                hr_h,
                p.batch_size,
                prompt_tokens,
            )
            if len(valid_models_hr) == 0 and prompt_tokens != 77:
                valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                    model_name,
                    hr_w,
                    hr_h,
                    p.batch_size,
                    77,
                )
            if len(valid_models_hr) == 0:
                print(f"[TensorRT] No engine profile covers the Hires. fix resolution {hr_w}x{hr_h}.", flush=True)
            merged_idx = [i for i, id in enumerate(idx) if id in idx_hr]
            if len(merged_idx) == 0:
                gr.Warning(
                    "No model available for both ({}) LOWRES ({}x{}) and HIRES ({}x{}). This will slow-down inference.".format(
                        model_name, p.width, p.height, hr_w, hr_h
                    )
                )
                return None, None
            else:
                _distances = [distances[i] for i in merged_idx]
                best_hr = merged_idx[np.argmin(_distances)]
                best = best_hr

        return best, best_hr

    def get_loras(self, p):
        lora_pathes = []
        lora_scales = []

        # get lora from prompt
        _prompt = p.prompt
        extra_networks = re.findall(r"<(.*?)>", _prompt)
        loras = [net for net in extra_networks if net.startswith("lora")]

        # Avoid that extra networks will be loaded
        for lora in loras:
            _prompt = _prompt.replace(f"<{lora}>", "")
        p.prompt = _prompt

        # check if lora config has changes
        if self.lora_hash != "".join(loras):
            self.lora_hash = "".join(loras)
            self.update_lora = True
            if self.lora_hash == "":
                self.lora_refit_dict = {}
                return
        else:
            return

        # Get pathes
        print("[TensorRT] Applying LoRAs: " + str(loras))
        available = modelmanager.available_loras()
        missing = []
        for lora in loras:
            # <lora:name> or <lora:name:scale> (extra fields after the scale are ignored)
            lora_parts = lora.split(":")[1:]
            if not lora_parts or not lora_parts[0]:
                raise ValueError(f"Invalid LoRA tag: <{lora}>")
            lora_name = lora_parts[0]
            lora_scale = float(lora_parts[1]) if len(lora_parts) > 1 and lora_parts[1] else 1.0
            path = resolve_trt_lora(lora_name, available)
            if path is None:
                missing.append(lora_name)
                continue
            lora_scales.append(lora_scale)
            lora_pathes.append(path)

        if missing:
            self.lora_hash = ""  # retry next time (e.g. after converting the LoRA)
            raise MissingTrtLora(missing)

        # Merge lora refit dicts
        base_name, base_path = modelmanager.get_onnx_path(p.sd_model_name)
        refit_dict = apply_loras(base_path, lora_pathes, lora_scales)

        self.lora_refit_dict = refit_dict

    def process(self, p, *args):
        patch_unet_forward()

        # Until a TensorRT UNet is set up below, this generation uses PyTorch.
        self.torch_unet = True

        # before unet_init
        sd_unet_option = sd_unet.get_unet_option()
        if sd_unet_option is None:
            # Respect an explicit "None" (PyTorch) selection; only auto-detect for "Automatic".
            if getattr(shared.opts, "sd_unet", "Automatic") != "Automatic":
                return
            # Try auto-detecting matching engine if unet_options was not yet populated or set to None
            avail = modelmanager.available_models()
            model_name = getattr(p, "sd_model_name", None) or getattr(
                getattr(shared, "sd_model", None), "sd_checkpoint_info", None
            )
            model_name = getattr(model_name, "model_name", str(model_name))
            
            for opt in getattr(sd_unet, "unet_options", []):
                if getattr(opt, "model_name", None) == model_name:
                    sd_unet_option = opt
                    break

            if sd_unet_option is None and model_name in avail:
                sd_unet.list_unets()
                for opt in getattr(sd_unet, "unet_options", []):
                    if getattr(opt, "model_name", None) == model_name:
                        sd_unet_option = opt
                        break
                if sd_unet_option is None:
                    sd_unet_option = TrtUnetOption(model_name, avail[model_name])
            if sd_unet_option is None:
                return

        if not sd_unet_option.model_name == p.sd_model_name:
            raise gr.Error(
                """Selected torch model ({}) does not match the selected TensorRT U-Net ({}). 
                Please ensure that both models are the same or select Automatic from the SD UNet dropdown.""".format(
                    p.sd_model_name, sd_unet_option.model_name
                )
            )

        # Check for UI override from accordion dropdown
        profile_override = None
        if len(args) > 0 and isinstance(args[0], str):
            match = re.match(r"Profile (\d+)", args[0].strip())
            if match:
                profile_override = int(match.group(1))

        if profile_override is not None:
            self.idx = profile_override
            self.hr_idx = profile_override
            self.is_auto = False
            print(f"[TensorRT] UI override: forced Profile {self.idx}", flush=True)
        elif getattr(sd_unet_option, "forced_profile_idx", None) is not None:
            self.idx = sd_unet_option.forced_profile_idx
            self.hr_idx = sd_unet_option.forced_profile_idx
            self.is_auto = False
            print(f"[TensorRT] SD Unet option: forced Profile {self.idx}", flush=True)
        else:
            self.is_auto = True
            prompt_tokens, token_info = get_max_prompt_token_count(p)
            self.idx, self.hr_idx = self.get_profile_idx(p, p.sd_model_name, ModelType.UNET, prompt_tokens=prompt_tokens)
            print(
                f"[TensorRT] Auto selected Profile: {self.idx} (HR: {self.hr_idx}) "
                f"[tokens: {prompt_tokens} (pos: {token_info['pos']}, neg: {token_info['neg']})]",
                flush=True,
            )

        num_profiles = len(sd_unet_option.configs) if hasattr(sd_unet_option, "configs") else 0
        if self.idx is not None and num_profiles > 0 and self.idx >= num_profiles:
            gr.Warning(f"[TensorRT] Selected Profile {self.idx} is out of range ({num_profiles} available). Reverting to Auto.")
            self.is_auto = True
            self.idx, self.hr_idx = self.get_profile_idx(p, p.sd_model_name, ModelType.UNET)

        self.torch_unet = self.idx is None or self.hr_idx is None

        try:
            if not self.torch_unet:
                self.get_loras(p)
        except MissingTrtLora as e:
            # PyTorch handles LoRAs natively: use it for this generation instead of failing.
            _notify(
                f"LoRA {', '.join(e.names)} hasn't been converted for TensorRT (TensorRT tab -> TensorRT LoRA); "
                "using PyTorch for this generation."
            )
            self.torch_unet = True
        except Exception as e:
            raise gr.Error(str(e))

        self.apply_unet(sd_unet_option)
        if sd_unet.current_unet is not None:
            sd_unet.current_unet.reset_generation_state()

    def apply_unet(self, sd_unet_option):
        if (
            sd_unet_option == sd_unet.current_unet_option
            and sd_unet.current_unet is not None
            and not self.torch_unet
        ):
            sd_unet.current_unet.is_auto = getattr(self, "is_auto", True)
            if self.idx is not None and sd_unet.current_unet.profile_idx != self.idx:
                print(
                    f"[TensorRT] Switching active engine from Profile {sd_unet.current_unet.profile_idx} to Profile {self.idx}",
                    flush=True,
                )
                sd_unet.current_unet.profile_idx = self.idx
                sd_unet.current_unet.switch_engine()
            return

        if sd_unet.current_unet is not None:
            # Just free the old engine; the PyTorch UNet is restored or offloaded below.
            release = getattr(sd_unet.current_unet, "release_engine", None)
            (release or sd_unet.current_unet.deactivate)()

        if self.torch_unet:
            print("[TensorRT] Using PyTorch for this generation.", flush=True)
            sd_unet.current_unet = None
            sd_unet.current_unet_option = sd_unet_option
            unet_memory.restore_torch_unet()
            return
        else:
            unet_memory.offload_torch_unet()
            if self.lora_refit_dict:
                self.update_lora = True
        sd_unet.current_unet = sd_unet_option.create_unet()
        sd_unet.current_unet.profile_idx = self.idx
        sd_unet.current_unet.is_auto = getattr(self, "is_auto", True)
        sd_unet.current_unet.option = sd_unet_option
        sd_unet.current_unet_option = sd_unet_option

        print(f"Activating unet: {sd_unet.current_unet.option.label} (Profile {self.idx})", flush=True)
        sd_unet.current_unet.activate()

    def process_batch(self, p, *args, **kwargs):
        # Called for each batch count
        if self.torch_unet or not isinstance(sd_unet.current_unet, TrtUnet):
            return super().process_batch(p, *args, **kwargs)

        # Check if prompts were expanded by wildcard extensions during process()
        if getattr(self, "is_auto", True):
            prompt_tokens, token_info = get_max_prompt_token_count(p)
            new_idx, new_hr_idx = self.get_profile_idx(p, p.sd_model_name, ModelType.UNET, prompt_tokens=prompt_tokens)
            if new_idx is not None and new_idx != self.idx:
                print(
                    f"[TensorRT] process_batch: updated profile to {new_idx} after prompt expansion ({token_info})",
                    flush=True,
                )
                self.idx = new_idx
                self.hr_idx = new_hr_idx

        if sd_unet.current_unet is not None and self.idx is not None and self.idx != sd_unet.current_unet.profile_idx:
            print(f"[TensorRT] process_batch: switching to Profile {self.idx}", flush=True)
            sd_unet.current_unet.profile_idx = self.idx
            sd_unet.current_unet.switch_engine()

    def post_sample(self, p, ps, *args):
        # Free the engine's activation memory before the VAE decode (reForge also does this
        # after every sampler run via the sampling_cleanup hook).
        _release_activation_memory()

    def before_hr(self, p, *args):
        if (
            sd_unet.current_unet is not None
            and self.hr_idx is not None
            and sd_unet.current_unet.profile_idx != self.hr_idx
        ):
            print(f"[TensorRT] HR Fix: switching to Profile {self.hr_idx}", flush=True)
            sd_unet.current_unet.profile_idx = self.hr_idx
            sd_unet.current_unet.switch_engine()

        return super().before_hr(p, *args)  # 4 (Only when HR starts.....)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        if self.update_lora and not self.torch_unet:
            self.update_lora = False
            sd_unet.current_unet.apply_loras(self.lora_refit_dict)


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        if not v or v[0]["config"].lora:
            continue
        base_label = "{} ({})".format(k, v[0]["base_model"]) if v[0]["config"].lora else k

        # Always provide base/auto option
        auto_label = f"[TRT] {base_label} (Auto)" if len(v) > 1 else f"[TRT] {base_label}"
        l.append(
            TrtUnetOption(
                name=base_label,
                filename=v,
                forced_profile_idx=None,
                custom_label=auto_label,
            )
        )
        # If multiple profiles exist, also expose each profile individually in the dropdown
        if len(v) > 1:
            for idx, conf in enumerate(v):
                summary = format_profile_summary(conf.get("config", None))
                profile_label = f"[TRT] {base_label} [Profile {idx}: {summary}]"
                l.append(
                    TrtUnetOption(
                        name=base_label,
                        filename=v,
                        forced_profile_idx=idx,
                        custom_label=profile_label,
                    )
                )


patch_unet_forward()
script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
