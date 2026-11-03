import contextlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from modules import devices, paths_internal, shared

import engine_builder
import unet_memory
from datastructures import ONNX_AUTOCAST, ONNX_FP16, ProfileSettings
from model_helper import UNetModel

TRACE_RESOLUTION = 512


def import_lora_networks():
    try:
        import networks

        return networks
    except ImportError:
        pass
    path = os.path.join(paths_internal.script_path, "extensions-builtin", "Lora")
    if path not in sys.path:
        sys.path.append(path)
    try:
        return importlib.import_module("networks")
    except Exception as e:
        raise RuntimeError(f"The built-in Lora extension could not be imported: {e}")


@unet_memory.in_inference_mode
def _forge_merged_weights(unet: torch.nn.Module) -> dict:
    """reForge keeps LoRAs as patches on the ModelPatcher: merge the patched weights on the CPU."""
    import ldm_patched.modules.lora as forge_lora

    sd_model = shared.sd_model
    patches = getattr(sd_model, "forge_objects_after_applying_lora", sd_model.forge_objects).unet.patches
    state_dict = unet.state_dict()
    merged = {}
    for key, key_patches in patches.items():
        name = key.removeprefix("diffusion_model.")
        if name in state_dict:
            weight = state_dict[name].detach().to(device="cpu", dtype=torch.float32, copy=True)
            merged[name] = forge_lora.calculate_weight(key_patches, weight, key).to(torch.float16)
    print(f"[TensorRT] Merged LoRA patches into {len(merged)} UNet weights.", flush=True)
    return merged


def lora_state_dict(modelobj: UNetModel, lora_name: str) -> dict:
    networks = import_lora_networks()
    if unet_memory.IS_FORGE_BACKEND:
        networks.load_networks([lora_name], [1.0], [1.0], [None])
        try:
            return _forge_merged_weights(modelobj.unet)
        finally:
            networks.load_networks([], [], [], [])
    inputs = modelobj.get_sample_input(2, TRACE_RESOLUTION // 8, TRACE_RESOLUTION // 8, 77)
    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet.forward(*inputs)
        networks.load_networks([lora_name], [1.0], [1.0], [None])
        modelobj.unet.forward(*inputs)  # A1111 merges LoRAs during forward
    return modelobj.unet.state_dict()


def export_lora(modelobj: UNetModel, onnx_path: str, weights_map_path: str, lora_name: str) -> dict:
    """fp16 deltas (merged - base) for the ONNX initializers the LoRA changes."""
    with open(weights_map_path) as f:
        names, shapes = json.load(f)
    state_dict = lora_state_dict(modelobj, lora_name)
    to_torch = {v: k for k, v in names.items()}
    onnx_dir = os.path.dirname(onnx_path)
    deltas = {}
    with torch.inference_mode():
        for init in onnx.load(onnx_path, load_external_data=False).graph.initializer:
            name = to_torch.get(init.name)
            if name not in state_dict:
                continue
            shape, transposed = shapes[name]
            wt = state_dict[name].detach()
            wt = (wt.permute(*range(wt.dim() - 1, -1, -1)) if transposed else wt.reshape(shape)).to(device="cpu", dtype=torch.float16)
            base = torch.from_numpy(numpy_helper.to_array(init, base_dir=onnx_dir).astype(np.float16))
            if not torch.equal(wt, base):
                deltas[init.name] = (wt - base).contiguous()
    return deltas


def _math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    scale = math.sqrt(1.0 / math.sqrt(query.size(-1)) if scale is None else scale)  # split over q and k: no fp16 overflow
    scores = torch.matmul(query * scale, key.transpose(-2, -1) * scale)
    if is_causal:
        mask = torch.ones(query.size(-2), key.size(-2), dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
    return torch.matmul(torch.softmax(scores, dim=-1), value)


def swap_sdpa(func):
    """Trace attention as matmul + softmax (exportable, fused by TensorRT), whatever the WebUI uses."""

    def wrapper(*args, **kwargs):
        swaps = [(F, "scaled_dot_product_attention", _math_sdpa)]
        try:
            import ldm_patched.ldm.modules.attention as attention

            swaps += [(attention, name, attention.attention_pytorch) for name in ("optimized_attention", "optimized_attention_masked")]
        except Exception:
            pass
        originals = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
        for owner, name, replacement in swaps:
            setattr(owner, name, replacement)
        try:
            return func(*args, **kwargs)
        finally:
            for owner, name, original in originals:
                setattr(owner, name, original)

    return wrapper


def onnx_flavor(onnx_path: str):
    """ONNX_FP16, ONNX_AUTOCAST (fp32 inputs, traced under autocast), or None if unreadable."""
    try:
        graph = onnx.load(onnx_path, load_external_data=False).graph
    except Exception:
        return None
    sample = next((i for i in graph.input if i.name == "sample"), None)
    if sample is None:
        return None
    return ONNX_FP16 if sample.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 else ONNX_AUTOCAST


@swap_sdpa
def export_onnx(onnx_path: str, modelobj: UNetModel, profile: ProfileSettings, fp16: bool):
    """fp16: the graph is fp16 throughout, which strongly typed TensorRT builds keep. Otherwise
    traced under autocast, which leaves normalizations, softmax and residuals in fp32."""
    inputs = modelobj.get_sample_input(
        2,
        min(profile.h_opt, TRACE_RESOLUTION) // 8,
        min(profile.w_opt, TRACE_RESOLUTION) // 8,
        profile.t_opt,
        dtype=torch.float16 if fp16 else torch.float32,
    )
    path = Path(onnx_path)
    tmp_dir = path.parent / f".{path.name}.trace"
    stage_dir = path.parent / f".{path.name}.partial"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(stage_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    stage_dir.mkdir()
    tmp_path, staged = tmp_dir / "model.onnx", stage_dir / path.name
    kwargs = {
        "opset_version": 17,
        "do_constant_folding": False,  # runs on the GPU and duplicates weights; TensorRT folds constants itself
        "input_names": modelobj.get_input_names(),
        "output_names": modelobj.get_output_names(),
        "dynamic_axes": modelobj.get_dynamic_axes(),
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    try:
        print(f"[TensorRT] Exporting the UNet to ONNX ({ONNX_FP16 if fp16 else ONNX_AUTOCAST}, 1-3 minutes)...", flush=True)
        model = modelobj.unet.to(inputs[0].device)
        with torch.inference_mode(), contextlib.nullcontext() if fp16 else torch.autocast("cuda"):
            torch.onnx.export(model, inputs, str(tmp_path), **kwargs)
        onnx_model = onnx.load(str(tmp_path))
        # Weights in one data file next to a small graph, whatever the size: cheap to inspect later.
        onnx.save(onnx_model, str(staged), save_as_external_data=True, all_tensors_to_one_file=True, location=path.name + "_data", size_threshold=1024)
        del onnx_model
        # The .onnx goes into place last, so an interrupted export never looks finished.
        for f in sorted(stage_dir.iterdir(), key=lambda f: f.name == path.name):
            os.replace(f, path.parent / f.name)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[TensorRT] ONNX model saved: {path}", flush=True)


def _builder_env() -> dict:
    """Unbuffered UTF-8 output, and the WebUI's GPU as the only visible device."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        device = torch.device(devices.device)
        if device.type != "cuda":
            return env
        index = device.index if device.index is not None else torch.cuda.current_device()
    except Exception:
        return env
    visible = [d.strip() for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    elif index < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[index]
    return env


def _run_builder(job: dict, progress_callback=None):
    """Builds in engine_builder's own process: TensorRT keeps the builder's VRAM until the process
    exits. Returns (success, number of memory allocation failures)."""
    cmd = [sys.executable, "-u", os.path.abspath(engine_builder.__file__), json.dumps(job)]
    result = {}
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_builder_env(), text=True, encoding="utf-8", errors="replace", bufsize=1
    )
    try:
        for line in proc.stdout:
            kind, payload = engine_builder.parse_message(line)
            if kind == "progress":
                if progress_callback is not None:
                    try:
                        progress_callback(payload["fraction"], payload["desc"])
                    except Exception:
                        pass
            elif kind == "result":
                result = payload
            else:
                sys.stdout.write(payload)
                sys.stdout.flush()
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
    if code != 0 and not result:
        print(f"[TensorRT] The engine builder exited unexpectedly (code {code}).", flush=True)
    return bool(result.get("ok")) and code == 0, int(result.get("memory_messages", 0))


def export_trt(trt_path: str, onnx_path: str, timing_cache: str, profile: dict, use_fp16: bool, optimization_level=None, progress_callback=None) -> bool:
    model = shared.sd_model
    move = not unet_memory.IS_FORGE_BACKEND and model is not None and not getattr(model, "lowvram", False)
    if move:
        model.cpu()
    unet_memory.free_vram_for_export()
    unet_memory.report_free_vram("for the build")
    job = {
        "engine_path": trt_path,
        "onnx_path": onnx_path,
        "fp16": use_fp16,
        "profiles": [profile],
        "refit": True,
        "timing_cache": timing_cache,
        "optimization_level": optimization_level,
    }
    try:
        start = time.time()
        ok, memory_messages = _run_builder(job, progress_callback)
        print(f"[TensorRT] Engine build took {time.time() - start:.0f}s", flush=True)
        unet_memory.report_free_vram("after the build")
        if memory_messages:
            import gradio as gr

            gr.Warning("TensorRT skipped some kernels for lack of GPU memory; the engine may be slower. See the console.")
        return ok
    finally:
        if move:
            model.to(devices.device)
