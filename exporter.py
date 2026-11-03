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
from onnx.external_data_helper import uses_external_data

from modules import devices, paths_internal, shared

import engine_builder
import unet_memory
from datastructures import ProfileSettings
from model_helper import UNetModel, optimize_onnx

TRACE_RESOLUTION = 512  # the graph's spatial axes are dynamic; tracing small saves VRAM


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


def _forge_merged_weights(unet: torch.nn.Module) -> dict:
    """reForge keeps LoRAs as patches on the UNet's ModelPatcher, so the module's weights never
    change: merge the patched ones on the CPU without touching the live model."""
    import ldm_patched.modules.lora as forge_lora

    sd_model = shared.sd_model
    patches = getattr(sd_model, "forge_objects_after_applying_lora", sd_model.forge_objects).unet.patches
    state_dict = unet.state_dict()
    merged = {}
    for key, key_patches in patches.items():
        name = key.removeprefix("diffusion_model.")
        if name in state_dict:
            # fp32 math, fp16 result on the CPU: fp32 copies of an SDXL LoRA's weights don't fit in VRAM
            weight = state_dict[name].detach().to(device="cpu", dtype=torch.float32, copy=True)
            merged[name] = forge_lora.calculate_weight(key_patches, weight, key).to(torch.float16)
    print(f"[TensorRT] Merged LoRA patches into {len(merged)} UNet weights.", flush=True)
    return merged


def lora_state_dict(modelobj: UNetModel, lora_name: str) -> dict:
    """UNet weights with the LoRA merged in: the changed ones on reForge, all of them on A1111."""
    networks = import_lora_networks()
    if unet_memory.IS_FORGE_BACKEND:
        networks.load_networks([lora_name], [1.0], [1.0], [None])
        try:
            return _forge_merged_weights(modelobj.unet)
        finally:
            networks.load_networks([], [], [], [])  # don't leave it selected for the next generation
    inputs = modelobj.get_sample_input(2, TRACE_RESOLUTION // 8, TRACE_RESOLUTION // 8, 77)
    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet.forward(*inputs)
        networks.load_networks([lora_name], [1.0], [1.0], [None])
        modelobj.unet.forward(*inputs)  # A1111 merges LoRAs into the weights during forward
    return modelobj.unet.state_dict()


def export_lora(modelobj: UNetModel, onnx_path: str, weights_map_path: str, lora_name: str) -> dict:
    """fp16 deltas (merged weight - base) for the ONNX initializers the LoRA changes."""
    with open(weights_map_path) as f:
        names, shapes = json.load(f)
    state_dict = lora_state_dict(modelobj, lora_name)
    to_torch = {v: k for k, v in names.items()}
    onnx_dir = os.path.dirname(onnx_path)
    deltas = {}
    with torch.inference_mode():
        for init in onnx.load(onnx_path, load_external_data=False).graph.initializer:  # read one at a time
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
    scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(query.size(-1)) if scale is None else scale)
    if is_causal:
        mask = torch.ones(query.size(-2), key.size(-2), dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
    return torch.matmul(torch.softmax(scores, dim=-1), value)


def swap_sdpa(func):
    """Trace attention as matmul + softmax, which exports to ONNX and which TensorRT fuses: the math
    version of PyTorch's fused SDPA, and reForge's PyTorch attention instead of xformers, sage, ..."""

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


@swap_sdpa
def export_onnx(onnx_path: str, modelobj: UNetModel, profile: ProfileSettings, disable_optimizations: bool = False):
    inputs = modelobj.get_sample_input(
        2, min(profile.h_opt, TRACE_RESOLUTION) // 8, min(profile.w_opt, TRACE_RESOLUTION) // 8, profile.t_opt
    )
    optimizer = None if disable_optimizations else optimize_onnx
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
        # PyTorch's constant folding runs on the GPU and can duplicate weights, which doesn't fit for
        # SDXL on 8-12 GB cards; TensorRT folds constants itself.
        "do_constant_folding": optimizer is not None,
        "input_names": modelobj.get_input_names(),
        "output_names": modelobj.get_output_names(),
        "dynamic_axes": modelobj.get_dynamic_axes(),
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # the TorchScript exporter: the dynamo one can't trace these UNets
    try:
        print("[TensorRT] Exporting the UNet to ONNX (1-3 minutes)...", flush=True)
        model = modelobj.unet.to(inputs[0].device)
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(model, inputs, str(tmp_path), **kwargs)

        onnx_model = onnx.load(str(tmp_path), load_external_data=False)
        external = any(uses_external_data(t) for t in onnx_model.graph.initializer)
        if external:  # > 2 GB: exported as one file per tensor, saved as one data file
            onnx_model = onnx.load(str(tmp_path))
        if optimizer is not None:
            onnx_model = optimizer(onnx_model)
        if external:
            onnx.save(onnx_model, str(staged), save_as_external_data=True, all_tensors_to_one_file=True, location=path.name + "_data", size_threshold=1024)
        elif optimizer is not None:
            onnx.save(onnx_model, str(staged))
        else:
            shutil.move(str(tmp_path), str(staged))
        del onnx_model
        # Move into place with the .onnx last: an interrupted export never leaves a partial model
        # that later exports would take as existing.
        for f in sorted(stage_dir.iterdir(), key=lambda f: f.name == path.name):
            os.replace(f, path.parent / f.name)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[TensorRT] ONNX model saved: {path}", flush=True)


def _builder_env() -> dict:
    """The builder process's environment: unbuffered UTF-8 output, and the WebUI's GPU as its only
    visible device (TensorRT builds on the current device, which a new process doesn't share)."""
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
    """Build in engine_builder's own process, so the builder's VRAM goes back to the driver when it
    exits. Returns (success, number of memory allocation failures)."""
    cmd = [sys.executable, "-u", os.path.abspath(engine_builder.__file__), json.dumps(job)]
    result = {}
    # stdin stays open until the builder exits: if the WebUI dies, the builder sees EOF and stops.
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
        if proc.poll() is None:  # interrupted while building
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
    # A1111: everything off the GPU for the build. reForge: unloaded through its model manager below,
    # since moving models behind its back desyncs it.
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
