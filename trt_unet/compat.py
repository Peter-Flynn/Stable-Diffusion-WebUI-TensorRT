# Compatibility layer between reForge (ldm_patched + modules_forge.forge_sampler) and Forge Neo
# (Haoming02/sd-webui-forge-classic `neo` branch: backend.*). Every module in trt_unet talks to the host WebUI
# through these helpers so both forks share one code path with zero reForge regressions.
import contextlib
import importlib
import importlib.util
import math
import sys

import torch

_is_neo = None
_mm = None


def is_neo() -> bool:
    global _is_neo
    if _is_neo is None:
        if "ldm_patched.modules.model_management" in sys.modules:
            _is_neo = False
        elif "backend.memory_management" in sys.modules:
            _is_neo = True
        else:
            has_ldm = importlib.util.find_spec("ldm_patched") is not None
            has_backend = importlib.util.find_spec("backend") is not None
            _is_neo = has_backend and not has_ldm
    return _is_neo


def mm():
    global _mm
    if _mm is not None:
        return _mm
    try:
        import ldm_patched.modules.model_management as model_management
    except ImportError:
        import backend.memory_management as model_management
    _mm = model_management
    return _mm


def setup_logger(logger):
    # Forge Neo configures the root logger to ERROR; backend.logging.setup_logger registers TqdmRichHandler at INFO
    # (or --loglevel) so TensorRT INFO/WARNING messages stay visible on Neo.
    if not is_neo():
        return
    try:
        from backend.logging import setup_logger as neo_setup

        neo_setup(logger)
    except Exception:
        pass


def device(mm_mod=None) -> torch.device:
    # reForge records the selected GPU on the main thread in model_management.current_device; Forge Neo stores it in
    # modules.devices.device; get_torch_device() reads torch.cuda.current_device() which is per-thread and defaults to 0
    # on Gradio and TensorRT worker threads.
    try:
        m = mm_mod if mm_mod is not None else mm()
        dev = getattr(m, "current_device", None)
        if dev is None and is_neo():
            with contextlib.suppress(Exception):
                from modules import devices

                dev = getattr(devices, "device", None)
        if dev is None:
            dev = m.get_torch_device()
        dev = torch.device(dev)
        if dev.type == "cuda" and dev.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return dev
    except Exception:
        return torch.device("cuda", torch.cuda.current_device())


def is_loaded_sd_model(sd_model) -> bool:
    # Forge Neo initializes sd_models.model_data.sd_model to FakeInitialModel() before any checkpoint is loaded.
    return (
        sd_model is not None
        and hasattr(sd_model, "forge_objects")
        and hasattr(sd_model, "sd_checkpoint_info")
    )


def model_config(patcher):
    model = getattr(patcher, "model", None)
    if model is None:
        return None
    cfg = getattr(model, "model_config", None)
    return cfg if cfg is not None else getattr(model, "config", None)


def model_sampling(model):
    sampling = getattr(model, "model_sampling", None)
    return sampling if sampling is not None else getattr(model, "predictor", None)


def unet_class():
    try:
        from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

        return UNetModel
    except ImportError:
        from backend.nn.unet import IntegratedUNet2DConditionModel

        return IntegratedUNet2DConditionModel


def anima_class():
    try:
        from backend.nn.anima import Anima

        return Anima
    except ImportError:
        return None


def moss_class():
    try:
        from backend.nn.moss_image import MOSSImage

        return MOSSImage
    except ImportError:
        return None


def fresh_unet(config: dict, device, dtype=torch.float16):
    try:
        import ldm_patched.modules.ops as ops
        from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

        return UNetModel(**config, dtype=dtype, device=device, operations=ops.disable_weight_init).eval()
    except ImportError:
        from backend.nn.unet import IntegratedUNet2DConditionModel

        with torch.device("meta"):
            unet = IntegratedUNet2DConditionModel(**config)
        unet = unet.to(dtype=dtype)
        if torch.device(device).type != "meta":
            unet = unet.to_empty(device=device)
        return unet.eval()


def fresh_dit(kind: str, config: dict, device, dtype=torch.bfloat16):
    cfg = {k: v for k, v in config.items() if k != "image_model"}
    if kind == "anima":
        cls = anima_class()
    elif kind == "moss_image":
        cls = moss_class()
    else:
        raise ValueError(f"Unsupported DiT kind {kind!r}")
    if cls is None:
        raise RuntimeError(f"Could not import Forge Neo's {kind} model class")
    with torch.device("meta"):
        model = cls(**cfg)
    model = model.to(dtype=dtype)
    if torch.device(device).type != "meta":
        model = model.to_empty(device=device)
    return model.eval()


def timestep_embedding(timesteps, dim, repeat_only: bool = False):
    try:
        from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding as _te
    except ImportError:
        from backend.nn.unet import timestep_embedding as _te
    return _te(timesteps, dim, repeat_only=repeat_only)


def forward_block(module, h, emb, context, options, output_shape=None):
    try:
        from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed

        return forward_timestep_embed(module, h, emb, context, options, output_shape)
    except ImportError:
        return module(h, emb, context, options, output_shape)


@contextlib.contextmanager
def patch_attention(fn):
    swaps = []
    try:
        import ldm_patched.ldm.modules.attention as attention

        swaps = [(attention, name, fn) for name in ("optimized_attention", "optimized_attention_masked")]
    except ImportError:
        import backend.attention as attention
        import backend.nn.unet as neo_unet

        # Forge Neo modules bind `from backend.attention import attention_function` at import time, so patch each.
        swaps = [(attention, "attention_function", fn), (neo_unet, "attention_function", fn)]
        for mod_name in ("backend.nn.anima", "backend.nn.moss_image"):
            mod = sys.modules.get(mod_name)
            if mod is None:
                with contextlib.suppress(ImportError):
                    mod = importlib.import_module(mod_name)
            if mod is not None and hasattr(mod, "attention_function"):
                swaps.append((mod, "attention_function", fn))
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
    for owner, name, replacement in swaps:
        setattr(owner, name, replacement)
    try:
        yield
    finally:
        for owner, name, orig in originals:
            setattr(owner, name, orig)


def vae_class():
    try:
        from ldm_patched.modules.sd import VAE

        return VAE
    except ImportError:
        from backend.patcher.vae import VAE

        return VAE


def sampling_module():
    try:
        from modules_forge import forge_sampler

        return forge_sampler
    except ImportError:
        from backend.sampling import sampling_function

        return sampling_function


def fresh_vae_decoder(cfg: dict, device, dtype):
    try:
        import ldm_patched.modules.ops as ops
        from ldm_patched.ldm.modules.diffusionmodules.model import Decoder

        conv_cls = ops.disable_weight_init.Conv2d
    except ImportError:
        from backend.nn.vae import Decoder

        conv_cls = torch.nn.Conv2d

    module = torch.nn.Module()
    module.decoder = Decoder(
        ch=cfg["ch"], out_ch=cfg["out_ch"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=3, resolution=256, z_channels=cfg["z_channels"], dropout=0.0,
    )
    if cfg["post_quant_conv"]:
        module.post_quant_conv = conv_cls(cfg["latent_channels"], cfg["z_channels"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def fresh_vae_encoder(cfg: dict, device, dtype):
    try:
        import ldm_patched.modules.ops as ops
        from ldm_patched.ldm.modules.diffusionmodules.model import Encoder

        conv_cls = ops.disable_weight_init.Conv2d
    except ImportError:
        from backend.nn.vae import Encoder

        conv_cls = torch.nn.Conv2d

    module = torch.nn.Module()
    module.encoder = Encoder(
        ch=cfg["ch"], out_ch=cfg["in_channels"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=cfg["in_channels"], resolution=256, z_channels=cfg["out_channels"], double_z=False, dropout=0.0,
    )
    if cfg["quant_conv"]:
        module.quant_conv = conv_cls(cfg["out_channels"], cfg["moments"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def vae_nonlinearity_module():
    for name in ("ldm_patched.ldm.modules.diffusionmodules.model", "backend.nn.vae"):
        try:
            mod = importlib.import_module(name)
            if callable(getattr(mod, "nonlinearity", None)):
                return mod
        except ImportError:
            continue
    return None


def vae_attention(fsm, encoding: bool = False):
    attn = (fsm.encoder if encoding else fsm.decoder).mid.attn_1
    fn = getattr(attn, "optimized_attention", None)
    if callable(fn):
        return fn
    try:
        from backend.attention import attention_function_vae

        return attention_function_vae
    except ImportError:
        pass

    def _sdpa_vae(q, k, v):
        b, c, h, w = q.shape
        qf, kf, vf = (t.reshape(b, 1, c, h * w).transpose(2, 3).contiguous() for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf)
        return out.transpose(2, 3).reshape(b, c, h, w)

    return _sdpa_vae


def vae_regularize(fsm, moments: torch.Tensor) -> torch.Tensor:
    reg = getattr(fsm, "regularization", None)
    if callable(reg):
        return reg(moments.float())[0]
    from backend.nn.vae import DiagonalGaussianDistribution

    return DiagonalGaussianDistribution(moments.float()).sample()


def vae_to_latent(vae, samples: torch.Tensor) -> torch.Tensor:
    fn = getattr(vae, "_to_vae_latent", None)
    return fn(samples) if callable(fn) else samples


def vae_from_latent(vae, latents: torch.Tensor) -> torch.Tensor:
    fn = getattr(vae, "_from_vae_latent", None)
    return fn(latents) if callable(fn) else latents


def vae_crop_pixels(vae, pixels: torch.Tensor) -> torch.Tensor:
    fn = getattr(vae, "vae_encode_crop_pixels", None)
    return fn(pixels) if callable(fn) else pixels


def vae_in_fp32(mm_mod) -> bool:
    args = getattr(mm_mod, "args", None)
    return bool(getattr(args, "vae_in_fp32", False) or getattr(args, "fp32_vae", False))


def pin_shared_memory(mm_mod) -> bool:
    return bool(getattr(mm_mod, "PIN_SHARED_MEMORY", False)) or getattr(mm_mod, "MAX_PINNED_MEMORY", 0) > 0


def merge_weight(patches, weight: torch.Tensor, key: str) -> torch.Tensor:
    try:
        from ldm_patched.modules.lora import calculate_weight

        return calculate_weight(patches, weight, key)
    except ImportError:
        from backend.patcher.lora import merge_lora_to_weight

        return merge_lora_to_weight(patches, weight, key, computation_dtype=torch.float32)


def is_online_lora_patch(obj) -> bool:
    return type(obj).__name__ == "OnlineLoRAPatch" and isinstance(getattr(obj, "patch", None), list)


_lora_control_cls = None
_lora_control_scan = None


def has_scheduled_lora() -> bool:
    global _lora_control_cls, _lora_control_scan
    if _lora_control_cls is None:
        try:
            from modules import scripts

            data = getattr(scripts, "scripts_data", None)
            sig = (id(data), len(data)) if data is not None else None
            if data is not None and sig != _lora_control_scan:
                for item in data:
                    cls = getattr(item, "script_class", None)
                    if cls is not None and cls.__name__ == "LoRAControl" and hasattr(cls, "mapping"):
                        _lora_control_cls = cls
                        break
                _lora_control_scan = sig
        except Exception:
            pass
    return bool(getattr(_lora_control_cls, "mapping", None)) if _lora_control_cls is not None else False


def is_quantized_weight(t) -> bool:
    if hasattr(t, "gguf_cls"):
        return t.gguf_cls is not None
    if hasattr(t, "_qdata") and callable(getattr(t, "dequantize", None)):
        return True
    if type(t) in (torch.Tensor, torch.nn.Parameter):
        return False
    return bool(getattr(t, "is_quantized", False)) or (
        type(t).__name__ == "QuantizedTensor" and callable(getattr(t, "dequantize", None))
    )


def dequantize_weight(t, converter=None) -> torch.Tensor:
    gguf_cls = getattr(t, "gguf_cls", None)
    if gguf_cls is not None:
        from backend.operations_gguf import dequantize_tensor

        if not getattr(t, "baked", True):
            if callable(getattr(t, "copy_with_data", None)):
                t = t.copy_with_data(t.data.clone())
            gguf_cls.bake(t)
        return dequantize_tensor(t).to(torch.float32, copy=True)
    if is_quantized_weight(t):
        # Neo's QuantizedTensor (MixedPrecisionOps) must be dequantized before casting to FP32. Ensure any extra
        # layout parameter tensors (e.g. block_scale, s_channel, codebook) live on the same device as _qdata.
        qdata = getattr(t, "_qdata", None)
        params = getattr(t, "_params", None)
        if isinstance(qdata, torch.Tensor) and params is not None:
            qdev = qdata.device
            pdict = getattr(params, "__dict__", None)
            if pdict and any(isinstance(v, torch.Tensor) and v.device != qdev for v in pdict.values()):
                if callable(getattr(params, "to_device", None)):
                    params = params.to_device(qdev)
                    pdict = getattr(params, "__dict__", None)
                if pdict and any(isinstance(v, torch.Tensor) and v.device != qdev for v in pdict.values()):
                    import copy as _copy

                    params = _copy.copy(params)
                    for k, v in vars(params).items():
                        if isinstance(v, torch.Tensor) and v.device != qdev:
                            setattr(params, k, v.to(qdev))
                t = type(t)(qdata, t._layout_cls, params)
        out = converter(t, inplace=True) if converter is not None else t.dequantize()
        return out.to(torch.float32, copy=True)
    raw = t.data if isinstance(t, torch.nn.Parameter) else t
    weight = raw.to(torch.float32, copy=True)
    if converter is not None:
        weight = converter(weight, inplace=True)
    return weight


def layer_quant_format(weight, module=None) -> tuple:
    """Classifies a Linear weight's storage quantization format as `(code, display_label)` without model assumptions.

    Returns `(None, None)` for standard floating-point (FP32 / FP16 / BF16) weights.
    """
    gguf_cls = getattr(weight, "gguf_cls", None)
    if gguf_cls is not None:
        qname = getattr(getattr(weight, "tensor_type", None), "name", None) or getattr(gguf_cls, "__name__", "GGUF")
        if qname == "NVFP4":
            return "fp4", "GGUF NVFP4"
        if qname == "BF16":
            return None, None
        return f"gguf:{qname}", f"GGUF {qname}"
    qdata = getattr(weight, "_qdata", None)
    layout = getattr(weight, "_layout_cls", None)
    if isinstance(qdata, torch.Tensor) and layout is not None:
        layout_s = layout.__name__ if isinstance(layout, type) else str(layout)
        params = getattr(weight, "_params", None)
        if layout_s == "TensorCoreNVFP4Layout":
            return "fp4", "NVFP4"
        if layout_s in ("TensorCoreFP8E4M3Layout", "TensorCoreFP8Layout"):
            return "fp8", "FP8"
        if layout_s == "TensorCoreFP8E5M2Layout":
            return "fp8_e5m2", "FP8 E5M2"
        if layout_s == "TensorCoreMXFP8Layout":
            return "mxfp8", "MXFP8"
        if layout_s == "TensorWiseINT8Layout":
            if getattr(params, "convrot", False):
                return "convrot_int8", "INT8 ConvRot"
            return "int8", "INT8"
        if layout_s == "TensorCoreConvRotW4A4Layout":
            return "convrot_w4a4", "ConvRot W4A4"
        if layout_s == "AsymW4A8Int8Layout":
            fmt = getattr(module, "quant_format", None)
            return fmt or "asym_w4a8_int8", "W6A8" if fmt == "w6a8_int8" else "W4A8"
        return f"other:{layout_s}", layout_s
    raw = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if fp8_e4m3 is not None and isinstance(raw, torch.Tensor) and raw.dtype == fp8_e4m3:
        if module is not None and callable(getattr(module, "convert_weight", None)) and not hasattr(module, "scale_weight"):
            return "other:fp8_custom", "FP8"
        return "fp8", "FP8"
    if fp8_e5m2 is not None and isinstance(raw, torch.Tensor) and raw.dtype == fp8_e5m2:
        return "fp8_e5m2", "FP8 E5M2"
    return None, None


def _passthrough_per_channel(qdata: torch.Tensor, scale, kind: str, scale_dtype: torch.dtype):
    rows = int(qdata.shape[0])
    dev = qdata.device
    min_scale = 1e-4 if scale_dtype == torch.float16 else 1e-12
    if scale is None:
        # Clone into owned storage: Engine.refit() passes views into reusable ping-pong gpu_bufs[slot] staging buffers.
        return qdata.clone(memory_format=torch.contiguous_format), torch.ones((rows,), device=dev, dtype=scale_dtype)
    if not isinstance(scale, torch.Tensor):
        return None
    if scale.numel() == 1:
        s_f = scale.to(device=dev, dtype=torch.float32).reshape(1).expand(rows)
    elif tuple(scale.shape) in ((rows,), (rows, 1)):
        s_f = scale.to(device=dev, dtype=torch.float32).reshape(rows)
    else:
        return None
    if scale_dtype == torch.float16:
        nonzero_fin = torch.isfinite(s_f) & (s_f != 0)
        if bool((nonzero_fin & ((s_f.abs() < 1e-4) | (s_f.abs() > 65504.0))).any()):
            return None
    bad = ~torch.isfinite(s_f) | (s_f == 0)
    neg = (~bad) & (s_f < 0)
    safe_s = torch.where(bad, s_f.new_full((), min_scale), s_f.abs().clamp_min(min_scale)).to(scale_dtype).contiguous()
    if kind == "fp8":
        q_u8 = qdata.contiguous().view(torch.uint8)
        q_u8 = torch.where(bad.unsqueeze(1), q_u8.new_zeros(()), q_u8)
        q_u8 = torch.where(neg.unsqueeze(1), torch.bitwise_xor(q_u8, 0x80), q_u8)
        return q_u8.view(torch.float8_e4m3fn).contiguous(), safe_s
    if kind == "int8":
        q_i8 = qdata.contiguous()
        q_neg = torch.where(q_i8 == -128, q_i8.new_full((), 127), -q_i8)
        q_out = torch.where(neg.unsqueeze(1), q_neg, q_i8)
        q_out = torch.where(bad.unsqueeze(1), q_out.new_zeros(()), q_out).contiguous()
        return q_out, safe_s
    return None


def passthrough_quant(
    gpu_base,
    kind: str,
    scale_dtype: torch.dtype,
    *,
    gscale_dtype: torch.dtype = None,
    module=None,
):
    """Directly adapts a natively pre-quantized checkpoint weight to TRT refit tensors with zero re-quantization.

    Supports Forge Neo `TensorCoreNVFP4Layout`, raw `ParameterGGUF` (`quants.NVFP4`), `TensorCoreFP8E4M3Layout` /
    `TensorCoreFP8Layout`, non-ConvRot `TensorWiseINT8Layout`, reForge `scaled_fp8_op` / `fp8_ops`, and Forge Neo
    plain `float8_e4m3fn` storage. Returns `None` on format mismatch or unsupported layouts so callers fall back
    cleanly to dequantize -> quantize. Never mutates `gpu_base` or its parameter buffers.
    """
    from . import quant

    gguf_cls = getattr(gpu_base, "gguf_cls", None)
    if gguf_cls is not None:
        if kind != quant.FP4:
            return None
        qname = getattr(getattr(gpu_base, "tensor_type", None), "name", None) or getattr(gguf_cls, "__name__", "")
        if qname != "NVFP4":
            return None
        shape = tuple(getattr(gpu_base, "shape", ()))
        if len(shape) != 2:
            return None
        return quant.gguf_nvfp4_to_trt(gpu_base.data, int(shape[0]), int(shape[1]), gscale_dtype=gscale_dtype)

    qdata = getattr(gpu_base, "_qdata", None)
    layout = getattr(gpu_base, "_layout_cls", None)
    params = getattr(gpu_base, "_params", None)
    if isinstance(qdata, torch.Tensor) and layout is not None and params is not None:
        layout_s = layout.__name__ if isinstance(layout, type) else str(layout)
        if getattr(params, "transposed", False):
            return None
        orig_shape = tuple(getattr(params, "orig_shape", ()) or getattr(gpu_base, "shape", ()))
        if len(orig_shape) != 2:
            return None
        rows, cols = int(orig_shape[0]), int(orig_shape[1])
        if rows <= 0 or cols <= 0:
            return None
        if kind == quant.FP4 and layout_s == "TensorCoreNVFP4Layout":
            if (
                not hasattr(torch, "float8_e4m3fn")
                or cols % quant.FP4_BLOCK_SIZE != 0
                or qdata.ndim != 2
                or qdata.dtype != torch.uint8
                or qdata.shape[0] < rows
                or qdata.shape[1] < cols // 2
            ):
                return None
            block_scale = getattr(params, "block_scale", None)
            gscale = getattr(params, "scale", None)
            if not isinstance(block_scale, torch.Tensor) or not isinstance(gscale, torch.Tensor) or gscale.numel() != 1:
                return None
            if block_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
                return None
            bs = block_scale.to(device=qdata.device)
            if bs.dtype == torch.uint8:
                bs = bs.view(torch.float8_e4m3fn)
            w_sf = quant.unswizzle_nvfp4_scales(bs, rows, cols // quant.FP4_BLOCK_SIZE)
            if w_sf is None:
                return None
            q_trt = quant.swap_fp4_nibbles(qdata[:rows, : cols // 2])
            return quant.sanitize_fp4_block_scales(q_trt, w_sf, rows, cols, gscale, gscale_dtype=gscale_dtype)
        if kind == quant.FP8 and layout_s in ("TensorCoreFP8E4M3Layout", "TensorCoreFP8Layout"):
            if (
                not hasattr(torch, "float8_e4m3fn")
                or qdata.dtype != torch.float8_e4m3fn
                or qdata.ndim != 2
                or qdata.shape[0] < rows
                or qdata.shape[1] < cols
            ):
                return None
            return _passthrough_per_channel(qdata[:rows, :cols], getattr(params, "scale", None), quant.FP8, scale_dtype)
        if kind == quant.INT8 and layout_s == "TensorWiseINT8Layout" and not getattr(params, "convrot", False):
            if qdata.dtype != torch.int8 or qdata.ndim != 2 or qdata.shape[0] < rows or qdata.shape[1] < cols:
                return None
            return _passthrough_per_channel(qdata[:rows, :cols], getattr(params, "scale", None), quant.INT8, scale_dtype)
        return None

    raw = gpu_base.data if isinstance(gpu_base, torch.nn.Parameter) else gpu_base
    if (
        kind == quant.FP8
        and hasattr(torch, "float8_e4m3fn")
        and isinstance(raw, torch.Tensor)
        and raw.dtype == torch.float8_e4m3fn
        and raw.ndim == 2
    ):
        if module is not None and callable(getattr(module, "convert_weight", None)) and not hasattr(module, "scale_weight"):
            return None
        scale = getattr(module, "scale_weight", None) if module is not None else None
        return _passthrough_per_channel(raw, scale, quant.FP8, scale_dtype)
    return None


def native_input_scale(module, weight, target_kind: str):
    """Extracts a checkpoint-calibrated scalar activation scale `(raw_val, trt_val)` if compatible with `target_kind`."""
    if module is None:
        return None
    sc = getattr(module, "input_scale", None)
    if sc is None:
        sc = getattr(module, "scale_input", None)
    if sc is None:
        return None
    if isinstance(sc, torch.Tensor):
        if sc.device.type == "meta" or sc.numel() != 1:
            return None
        raw_val = float(sc.detach().float().item())
    elif isinstance(sc, (int, float)):
        raw_val = float(sc)
    else:
        return None
    if not math.isfinite(raw_val) or raw_val <= 0.0:
        return None
    layer_fmt, _ = layer_quant_format(weight, module)
    if layer_fmt == "fp8" and target_kind == "fp8":
        return raw_val, raw_val
    if layer_fmt == "fp4" and target_kind == "fp4":
        return raw_val, raw_val
    # NVFP4 engines keep separate V projections in FP8: scale_fp8 = amax/448 = (amax/2688) * 6.
    if layer_fmt == "fp4" and target_kind == "fp8":
        return raw_val, raw_val * 6.0
    return None


def load_list_entries(patcher):
    # reForge yields 4-tuples (module_mem, n, m, params); Forge Neo yields 5-tuples
    # (module_offload_mem, module_mem, n, m, params) where params is a dict. Preserves partially_unload() sort order.
    load_list_fn = getattr(patcher, "_load_list", None)
    if not callable(load_list_fn):
        return None
    out = []
    for entry in sorted(load_list_fn()):
        if len(entry) >= 5:
            _, _, prefix, mod, param_names = entry[:5]
        else:
            _, prefix, mod, param_names = entry[:4]
        out.append((prefix, mod, list(param_names)))
    return out


SNAPSHOT_ATTRS = (
    "ldm_patched_cast_weights",
    "prev_ldm_patched_cast_weights",
    "ldm_patched_patched_weights",
    "parameters_manual_cast",
    "prev_parameters_manual_cast",
    "forge_patched_weights",
    "forge_force_cast_weights",
)


def has_cast_weights(mod) -> bool:
    return hasattr(mod, "ldm_patched_cast_weights") or hasattr(mod, "parameters_manual_cast")


def is_patched_weights(mod) -> bool:
    return bool(getattr(mod, "ldm_patched_patched_weights", False) or getattr(mod, "forge_patched_weights", False))


def mark_patched_weights(mod):
    if hasattr(mod, "parameters_manual_cast") or is_neo():
        mod.forge_patched_weights = True
    else:
        mod.ldm_patched_patched_weights = True
