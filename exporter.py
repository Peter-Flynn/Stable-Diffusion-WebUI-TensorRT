import gc
import os
import sys
import math
import time
import shutil
import json
from pathlib import Path
from logging import info, error
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import uses_external_data


def check_model_uses_external_data(model) -> bool:
    return any(uses_external_data(init) for init in model.graph.initializer)


from modules import shared

from utilities import Engine
from datastructures import ProfileSettings
from model_helper import UNetModel
import unet_memory


def _import_lora_networks():
    try:
        import networks  # importable once the built-in Lora extension is loaded

        return networks
    except ImportError:
        pass
    import importlib

    sys.path.append(os.path.join("extensions-builtin", "Lora"))
    try:
        return importlib.import_module("networks")
    except Exception as e:
        raise RuntimeError(f"The built-in Lora extension could not be imported: {e}")


def _forge_patched_state_dict(unet: torch.nn.Module):
    """reForge/Forge attach LoRAs to the UNet's ModelPatcher instead of merging them into
    the torch weights, so the module's state_dict() never changes. Compute the merged
    weights for the patched keys without modifying the live model. Returns None on A1111."""
    try:
        import ldm_patched.modules.lora as forge_lora

        patcher = shared.sd_model.forge_objects.unet
        patches = patcher.patches
    except Exception:
        return None
    if not patches:
        return None

    state_dict = dict(unet.state_dict())
    prefix = "diffusion_model."
    merged = 0
    for key, key_patches in patches.items():
        name = key[len(prefix):] if key.startswith(prefix) else key
        if name not in state_dict:
            continue
        weight = state_dict[name].detach().to(dtype=torch.float32, copy=True)
        state_dict[name] = forge_lora.calculate_weight(key_patches, weight, key)
        merged += 1
    print(f"[TensorRT] Merged LoRA patches into {merged} UNet weights.", flush=True)
    return state_dict


def apply_lora(model: torch.nn.Module, lora_path: str, inputs: Tuple[torch.Tensor]) -> dict:
    """Load a LoRA through the WebUI and return the UNet state dict with the LoRA merged in."""
    networks = _import_lora_networks()
    lora_name = os.path.splitext(os.path.basename(lora_path))[0]

    model.forward(*inputs)
    networks.load_networks([lora_name], [1.0], [1.0], [None])

    state_dict = _forge_patched_state_dict(model)
    if state_dict is not None:
        # Undo the LoRA selection so it doesn't leak into the next PyTorch generation.
        try:
            networks.load_networks([], [], [], [])
        except Exception:
            pass
        return state_dict

    # A1111: the LoRA is merged into the module weights during forward.
    model.forward(*inputs)
    return model.state_dict()


def get_refit_weights(
    state_dict: dict, onnx_opt_path: str, weight_name_mapping: dict, weight_shape_mapping: dict
) -> dict:
    refit_weights = OrderedDict()
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)
    # Create initializer data hashes
    initializer_hash_mapping = {}
    onnx_data_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(
            initializer, base_dir=onnx_opt_dir
        ).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash
        onnx_data_mapping[initializer.name] = initializer_data

    for torch_name, initializer_name in weight_name_mapping.items():
        initializer_hash = initializer_hash_mapping[initializer_name]
        wt = state_dict[torch_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[torch_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.detach().to(torch.float16).cpu().numpy().data.tobytes())
        if initializer_hash != wt_hash:
            delta = wt.to(torch.float16) - torch.tensor(onnx_data_mapping[initializer_name]).to(wt.device)
            refit_weights[initializer_name] = delta.contiguous()

    return refit_weights


def export_lora(
    modelobj: UNetModel,
    onnx_path: str,
    weights_map_path: str,
    lora_name: str,
    profile: ProfileSettings,
) -> dict:
    info("Exporting LoRA weights...")
    inputs = modelobj.get_sample_input(
        2,
        min(profile.h_opt, TRACE_RESOLUTION) // 8,
        min(profile.w_opt, TRACE_RESOLUTION) // 8,
        profile.t_opt,
    )

    with open(weights_map_path, "r") as fp_wts:
        print(f"[I] Loading weights map: {weights_map_path} ")
        [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

    with torch.inference_mode(), torch.autocast("cuda"):
        state_dict = apply_lora(
            modelobj.unet, os.path.splitext(lora_name)[0], inputs
        )

        refit_dict = get_refit_weights(
            state_dict,
            onnx_path,
            weights_name_mapping,
            weights_shape_mapping,
        )

    return refit_dict


def _math_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)


TRACE_RESOLUTION = 512


def swap_sdpa(func):
    def wrapper(*args, **kwargs):
        old_sdpa = getattr(F, "scaled_dot_product_attention", None)
        # Instead of deleting scaled_dot_product_attention (which causes AttributeError in reForge/ldm_patched),
        # replace it with a pure-math implementation that is 100% exportable to ONNX across all backends.
        try:
            setattr(F, "scaled_dot_product_attention", _math_scaled_dot_product_attention)
        except Exception:
            pass
        try:
            ret = func(*args, **kwargs)
        finally:
            if old_sdpa is not None:
                try:
                    setattr(F, "scaled_dot_product_attention", old_sdpa)
                except Exception:
                    pass
        return ret

    return wrapper


@swap_sdpa
def export_onnx(
    onnx_path: str,
    modelobj: UNetModel,
    profile: ProfileSettings,
    opset: int = 17,
    diable_optimizations: bool = False,
):
    print(f"[TensorRT] Preparing weights for ONNX export...", flush=True)
    # The exported graph has dynamic batch/height/width/token axes, so the trace
    # resolution doesn't affect the engine. Tracing small keeps VRAM use low.
    inputs = modelobj.get_sample_input(
        2,
        min(profile.h_opt, TRACE_RESOLUTION) // 8,
        min(profile.w_opt, TRACE_RESOLUTION) // 8,
        profile.t_opt,
    )

    if not os.path.exists(onnx_path):
        _export_onnx(
            modelobj.unet,
            inputs,
            Path(onnx_path),
            opset,
            modelobj.get_input_names(),
            modelobj.get_output_names(),
            modelobj.get_dynamic_axes(),
            modelobj.optimize if not diable_optimizations else None,
        )
    else:
        print(f"[TensorRT] Found existing ONNX model: {onnx_path} (skipping re-export)", flush=True)


def _export_onnx(
    model: torch.nn.Module, inputs: Tuple[torch.Tensor], path: str, opset: int, in_names: List[str], out_names: List[str], dyn_axes: dict, optimizer=None
):
    tmp_dir = os.path.abspath("onnx_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "model.onnx")
    # Ensure all model parameters and buffers are placed on the input device
    target_device = inputs[0].device if len(inputs) > 0 else torch.device("cuda")
    try:
        model = model.to(target_device)
        for p in model.parameters():
            if p.device != target_device:
                p.data = p.data.to(target_device)
        for b in model.buffers():
            if b.device != target_device:
                b.data = b.data.to(target_device)
    except Exception as e:
        print(f"[TensorRT] Info: Model parameter device verification: {e}", flush=True)

    export_kwargs = {
        "export_params": True,
        "opset_version": max(opset, 17),
        "do_constant_folding": True,
        "input_names": in_names,
        "output_names": out_names,
        "dynamic_axes": dyn_axes,
    }

    # In PyTorch >= 2.2 / 2.4, explicitly set dynamo=False to use TorchScript tracer instead of TorchDynamo/FakeTensor
    import inspect
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    try:
        print(f"[TensorRT] Exporting PyTorch graph to ONNX (working file: {tmp_path})...", flush=True)
        print(f"[TensorRT] This may take 1–3 minutes. Please wait...", flush=True)
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                **export_kwargs,
            )
        print(f"[TensorRT] ONNX graph exported successfully!", flush=True)
    except Exception as e:
        print(f"[TensorRT ERROR] ONNX export failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Error during torch.onnx.export: {e}")

    print(f"[TensorRT] Validating and processing ONNX structure...", flush=True)
    os.makedirs(path.parent, exist_ok=True)
    onnx_model = onnx.load(tmp_path, load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        print(f"[TensorRT] Model exceeds 2GB (using external weight data). Saving...", flush=True)
        onnx_model = onnx.load(tmp_path, load_external_data=True)
        onnx.save(
            onnx_model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
            size_threshold=1024,
        )

    if optimizer is not None:
        try:
            print(f"[TensorRT] Optimizing ONNX nodes (GraphSurgeon)...", flush=True)
            onnx_opt_graph = optimizer("unet", onnx_model)
            onnx.save(onnx_opt_graph, path)
        except Exception as e:
            print(f"[TensorRT ERROR] ONNX optimization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Error during ONNX graph optimization: {e}")

    if not model_uses_external_data and optimizer is None:
        shutil.move(tmp_path, str(path))

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[TensorRT] ONNX model ready at: {path}", flush=True)


def free_gpu_memory():
    """Release as much VRAM as possible before a TensorRT build: unload models held by
    reForge/Forge's model manager (they're reloaded on the next generation) and empty
    PyTorch's cache. TensorRT skips kernels that don't fit, so free memory = faster engines."""
    unet_memory.free_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"[TensorRT] Free GPU memory for the build: {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
    except Exception:
        pass


def export_trt(trt_path: str, onnx_path: str, timing_cache: str, profile: dict, use_fp16: bool, optimization_level: int = None, progress_callback=None):
    print(f"[TensorRT] Starting TensorRT engine compilation...", flush=True)
    print(f"[TensorRT] ONNX Input: {onnx_path}", flush=True)
    print(f"[TensorRT] TRT Output:  {trt_path}", flush=True)
    print(f"[TensorRT] Timing Cache: {timing_cache}", flush=True)
    engine = Engine(trt_path)

    model = None
    # reForge/Forge: free_gpu_memory() unloads everything through the model manager instead;
    # moving the model by hand would desync the manager (device mismatch errors later).
    if not unet_memory.IS_FORGE_BACKEND and getattr(shared, "sd_model", None) is not None:
        try:
            print(f"[TensorRT] Freeing VRAM: moving base model to RAM...", flush=True)
            model = shared.sd_model.cpu()
        except Exception as e:
            print(f"[TensorRT] Warning: failed to move model to CPU: {e}", flush=True)
    free_gpu_memory()

    try:
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_refit=True,
            enable_preview=True,
            timing_cache=timing_cache,
            input_profile=[profile],
            optimization_level=optimization_level,
            progress_callback=progress_callback,
        )
        e = time.time()
        print(f"[TensorRT] TensorRT engine compilation time: {(e-s):.2f}s", flush=True)
        if getattr(engine, "build_memory_messages", 0):
            try:
                import gradio as gr

                gr.Warning(
                    "TensorRT skipped some kernels for lack of GPU memory; the engine may be slower. "
                    "See the console for details."
                )
            except Exception:
                pass
        return ret
    finally:
        if model is not None:
            try:
                print(f"[TensorRT] Restoring model to GPU memory...", flush=True)
                shared.sd_model = model.cuda()
            except Exception as e:
                print(f"[TensorRT] Warning: failed to restore model to GPU: {e}", flush=True)
