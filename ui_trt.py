import gc
import json
import os
import traceback

import gradio as gr
import torch
from safetensors.torch import save_file

from modules import devices, sd_hijack, sd_unet, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import FormRow, ToolButton

import model_manager
import unet_memory
from datastructures import ModelConfig, ProfilePresets, ProfileSettings, SDVersion
from exporter import export_lora, export_onnx, export_trt
from model_helper import UNetModel
from model_manager import TRT_MODEL_DIR, modelmanager

profile_presets = ProfilePresets()
LEVELS = (("min", "Min"), ("opt", "Optimal"), ("max", "Max"))
SLIDER_GROUPS = (  # column elem_id, label, (minimum, maximum, step), ProfileSettings prefix, slider elem_id
    ("trt_batch", "{} batch-size", (1, 16, 1), "bs", "trt_{}_batch"),
    ("trt_height", "{} height", (256, 4096, 64), "h", "trt_{}_height"),
    ("trt_width", "{} width", (256, 4096, 64), "w", "trt_{}_width"),
    ("trt_token_count", "{} prompt token count", (75, 750, 75), "t", "trt_opt_token_count_{}"),
)


def clear_vram_for_export():
    """Record what's loaded, then clear VRAM. Pass the result to restore_vram_after_export()."""
    snapshot = unet_memory.snapshot_loaded_models()
    unet_memory.free_vram_for_export()
    unet_memory.report_free_vram("after clearing VRAM for export")
    return snapshot


def restore_vram_after_export(snapshot):
    try:
        unet_memory.free_all_models()
        unet_memory.restore_loaded_models(snapshot)
        sd_hijack.model_hijack.apply_optimizations()  # undo apply_torch_model (A1111; a no-op on reForge)
    except Exception as e:
        print(f"[TensorRT] Could not restore the previous model state: {e}", flush=True)


def load_unet_for_tracing():
    """Only the UNet on the GPU, fully; VRAM must have been cleared."""
    try:
        if unet_memory.IS_FORGE_BACKEND:
            # The original patcher: forge_objects.unet may carry the last generation's LoRA patches,
            # which loading would merge into the traced weights.
            original = getattr(shared.sd_model, "forge_objects_original", shared.sd_model.forge_objects)
            unet_memory.forge_model_management().load_models_gpu([original.unet], force_full_load=True)
        else:
            shared.sd_model.model.diffusion_model.to(devices.device)
    except Exception as e:
        print(f"[TensorRT] Could not load the UNet onto the GPU: {e}", flush=True)
    unet_memory.report_free_vram("after loading the UNet")


def context_dim():
    model = shared.sd_model
    return 2048 if model.is_sdxl else 1024 if model.is_sd2 else 768


def refresh_unet_options():
    """Rebuild the SD Unet options so new or deleted engines take effect, dropping an active TensorRT
    UNet (it holds the old engine list); the next generation creates it again."""
    try:
        if hasattr(sd_unet.current_unet, "configs"):
            sd_unet.current_unet.deactivate()
            sd_unet.current_unet = sd_unet.current_unet_option = None
        sd_unet.list_unets()
    except Exception as e:
        print(f"[TensorRT] Could not refresh the SD Unet list: {e}", flush=True)


# Named parameters: gradio only injects `progress` into a function without *args.
def export_unet_to_trt(
    bs_min, bs_opt, bs_max, h_min, h_opt, h_max, w_min, w_opt, w_max, t_min, t_opt, t_max,
    force_export, static_shapes, preset, optimization_level=3, progress=gr.Progress(),
):
    snapshot = None
    try:
        model = shared.sd_model
        is_xl = model.is_sdxl
        model_name, model_hash = model.sd_checkpoint_info.model_name, model.sd_checkpoint_info.hash
        if preset == "Default":
            settings = profile_presets.get_default(is_xl)
        else:
            settings = ProfileSettings(bs_min, bs_opt, bs_max, h_min, h_opt, h_max, w_min, w_opt, w_max, t_min, t_opt, t_max)
        use_fp32 = 0 < model_manager.cc_major < 7  # no fast FP16 before Volta
        print(f"[TensorRT] Exporting {model_name} ({'SDXL' if is_xl else 'SD 1.5/2.1'}, {'FP32' if use_fp32 else 'FP16'})\n{settings}", flush=True)
        settings.token_to_dim(static_shapes)

        modelobj = UNetModel(model.model.diffusion_model, context_dim(), is_xl)
        profile = modelobj.get_input_profile(settings)
        onnx_path = model_manager.onnx_path(model_name)
        filename = model_manager.trt_filename(model_name, model_hash, profile, static_shapes)
        trt_path = os.path.join(TRT_MODEL_DIR, filename)

        def register(trt_version):
            config = ModelConfig(profile, static_shapes, use_fp32, modelobj.in_channels == 9, True, False, 0, modelobj.in_channels, trt_version)
            modelmanager.add_entry(model_name, filename, config)
            refresh_unet_options()

        if os.path.exists(trt_path) and not force_export:
            if not modelmanager.has_engine(filename):
                register("")  # built before, but missing from model.json
            yield f"### ✅ This engine already exists: `{filename}`\n\nEnable **Force Rebuild** to build it again."
            return

        snapshot = clear_vram_for_export()
        modelobj.apply_torch_model()
        if not os.path.exists(onnx_path):
            yield f"### ⏳ [1/2] Exporting the UNet to ONNX...\n- `{onnx_path}`\n\n*SDXL takes 1-3 minutes. Details are in the console.*"
            load_unet_for_tracing()
            export_onnx(onnx_path, modelobj, settings, disable_optimizations=is_xl)
            gc.collect()
            torch.cuda.empty_cache()

        yield f"### ⚙️ [2/2] Building the TensorRT engine...\n- `{filename}`\n\n*This takes 2-6 minutes. Details are in the console.*"
        ok = export_trt(
            trt_path,
            onnx_path,
            model_manager.timing_cache_path(),
            profile,
            use_fp16=not use_fp32,
            optimization_level=optimization_level,
            progress_callback=lambda fraction, desc: progress(fraction, desc=desc),
        )
        if not ok:
            yield "### ❌ The TensorRT engine build failed.\n\nThe console has the details."
            return
        register(modelmanager.trt_version)
        yield (
            f"### ✅ Engine built: `{filename}`\n\n"
            f"Select it in **SD Unet** (or leave that on `Automatic`) and generate as usual."
        )
    except Exception as e:
        traceback.print_exc()
        yield f"### ❌ Export failed: `{e}`\n\nThe console has the details."
    finally:
        if snapshot is not None:
            restore_vram_after_export(snapshot)


def get_lora_checkpoints() -> dict:
    """{name: {"filename": path relative to the LoRA folder, "version": SDVersion}}"""
    lora_dir = shared.cmd_opts.lora_dir
    loras = {}
    for root, _, files in os.walk(lora_dir, followlinks=True):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext not in (".pt", ".ckpt", ".safetensors"):
                continue
            version = SDVersion.Unknown
            try:  # the LoRA's user metadata from the WebUI
                with open(os.path.join(root, name + ".json"), encoding="utf-8") as fp:
                    version = SDVersion.from_str(json.load(fp).get("sd version"))
            except Exception:
                pass
            loras[name] = {"filename": os.path.relpath(os.path.join(root, f), lora_dir), "version": version}
    return dict(sorted(loras.items(), key=lambda kv: kv[0].lower()))


def get_valid_lora_checkpoints():
    return [f"{name} ({lora['version']})" for name, lora in get_lora_checkpoints().items()]


def export_lora_to_trt(lora_label, force_export):
    lora_name = (lora_label or "").rsplit(" (", 1)[0]  # "<name> (<version>)"
    lora = get_lora_checkpoints().get(lora_name)
    if lora is None:
        return f"## No LoRA model found for {lora_name}"
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    if not lora["version"].match(shared.sd_model):
        print(f"[TensorRT] Warning: {lora_name} is a {lora['version']} LoRA, which doesn't match {model_name}.", flush=True)
    onnx_path = model_manager.onnx_path(model_name)
    if not os.path.exists(onnx_path):
        return f"## Please export the base model ({model_name}) first."
    lora_path = os.path.join(TRT_MODEL_DIR, f"{lora_name}.lora")
    if os.path.exists(lora_path) and not force_export:
        return "## Already converted\n\nEnable **Force Rebuild** to convert it again."

    snapshot = clear_vram_for_export()  # on reForge this also unloads LoRA patches merged into the weights
    try:
        modelobj = UNetModel(shared.sd_model.model.diffusion_model, context_dim(), shared.sd_model.is_sdxl)
        if not unet_memory.IS_FORGE_BACKEND:
            load_unet_for_tracing()  # A1111 merges LoRAs during a forward pass
        weights_map = model_manager.weights_map_path(model_name)
        if not os.path.exists(weights_map):
            modelobj.export_weights_map(onnx_path, weights_map)
        deltas = export_lora(modelobj, onnx_path, weights_map, lora_name)
        save_file(deltas, lora_path + ".tmp")
        os.replace(lora_path + ".tmp", lora_path)
    except Exception as e:
        traceback.print_exc()
        return f"## ❌ Conversion failed: `{e}`\n\nThe console has the details."
    finally:
        restore_vram_after_export(snapshot)
    return f"## Converted {lora_name} ({len(deltas)} weights)"


def profiles_markdown() -> str:
    md = []
    for name, entries in modelmanager.available_models().items():
        md.append(f"<details><summary>{name} ({len(entries)} Profiles)</summary>\n")
        for i, entry in enumerate(entries):
            sample, context = entry["config"].profile["sample"], entry["config"].profile["encoder_hidden_states"]
            rows = (
                ("Height", [s[2] * 8 for s in sample]),
                ("Width", [s[3] * 8 for s in sample]),
                ("Batch Size", [max(s[0] // 2, 1) for s in sample]),
                ("Text-length", [c[1] // 77 * 75 for c in context]),
            )
            md.append(f"#### Profile {i}\n|  | Min | Opt | Max |\n|---|:---:|:---:|:---:|")
            md += [f"| {label} | {' | '.join(map(str, values))} |" for label, values in rows]
            md.append("")
        md.append("</details>\n")
    md.append("\n---\n## LoRA Profiles\n")
    md += [f"- {lora}" for lora in modelmanager.available_loras()]
    return "\n".join(md)


def get_engine_choices() -> dict:
    """Label -> engine filename for every recorded engine (all GPUs and TensorRT versions)."""
    choices = {}
    for cc, name, entry in modelmanager.all_engines():
        config = entry["config"]
        lo, opt, hi = config.profile["sample"]
        res = f"{opt[3] * 8}x{opt[2] * 8} static" if lo == hi else f"{lo[3] * 8}x{lo[2] * 8}-{hi[3] * 8}x{hi[2] * 8}"
        label = f"{name} | {res} | batch {max(lo[0] // 2, 1)}-{max(hi[0] // 2, 1)} | {cc}, TensorRT {config.trt_version or 'unknown'}"
        while label in choices:
            label += " "
        choices[label] = entry["filepath"]
    return choices


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="TensorRT Exporter"):
                        gr.Markdown(value="# TensorRT Exporter")
                        defaults = profile_presets.get_default(is_xl=False)
                        version = gr.Dropdown(label="Preset", choices=profile_presets.get_choices(), elem_id="sd_version", value="Default")

                        with gr.Accordion("Advanced Settings", open=False, visible=False) as advanced_settings:
                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                static_shapes = gr.Checkbox(label="Use static shapes.", value=False, elem_id="trt_static_shapes")
                            profile_sliders = []
                            for column_id, label, (lo, hi, step), field, elem_id in SLIDER_GROUPS:
                                with gr.Column(elem_id=column_id):
                                    for level, level_label in LEVELS:
                                        profile_sliders.append(
                                            gr.Slider(
                                                minimum=lo,
                                                maximum=hi,
                                                step=step,
                                                label=label.format(level_label),
                                                value=getattr(defaults, f"{field}_{level}"),
                                                elem_id=elem_id.format(level),
                                            )
                                        )
                            optimization_level = gr.Slider(
                                minimum=0,
                                maximum=5,
                                step=1,
                                label="Builder optimization level",
                                info="TensorRT default is 3. Higher values search more kernels: much slower builds, sometimes slightly faster engines.",
                                value=3,
                                elem_id="trt_optimization_level",
                            )
                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                force_rebuild = gr.Checkbox(label="Force Rebuild.", value=False, elem_id="trt_force_rebuild")

                        button_export_unet = gr.Button(value="Export Engine", variant="primary", elem_id="trt_export_unet", visible=False)
                        button_export_default_unet = gr.Button(value="Export Default Engine", variant="primary", elem_id="trt_export_default_unet")

                        version.change(profile_presets.get_settings_from_version, version, profile_sliders + [static_shapes])
                        version.change(
                            lambda v: (gr.update(visible=v != "Default"), gr.update(visible=v == "Default"), gr.update(visible=v != "Default")),
                            version,
                            [button_export_unet, button_export_default_unet, advanced_settings],
                        )
                        min_max_sliders = [s for i, s in enumerate(profile_sliders) if i % 3 != 1]
                        static_shapes.change(lambda static: [gr.update(visible=not static)] * len(min_max_sliders), static_shapes, min_max_sliders)

                    with gr.Tab(label="TensorRT LoRA"):
                        gr.Markdown("# Apply LoRA checkpoint to TensorRT model")
                        lora_refresh_button = gr.Button(value="Refresh", variant="primary", elem_id="trt_lora_refresh")
                        trt_lora_dropdown = gr.Dropdown(choices=get_valid_lora_checkpoints(), elem_id="lora_model", label="LoRA Model")
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_lora_force_rebuild = gr.Checkbox(label="Force Rebuild.", value=False, elem_id="trt_lora_force_rebuild")
                        button_export_lora_unet = gr.Button(value="Convert to TensorRT", variant="primary", elem_id="trt_lora_export_unet", visible=False)

                        lora_refresh_button.click(lambda: gr.update(choices=get_valid_lora_checkpoints()), None, trt_lora_dropdown)
                        trt_lora_dropdown.change(lambda lora: gr.update(visible=lora is not None), trt_lora_dropdown, button_export_lora_unet)

            with gr.Column(variant="panel"):
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"), encoding="utf-8") as f:
                    gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=True):
                trt_result = gr.Markdown(elem_id="trt_result", value="")

        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True, variant="compact"):
                button_refresh_profiles = ToolButton(value=refresh_symbol, elem_id="trt_refresh_profiles")
                gr.Markdown(value="## Available TensorRT Engine Profiles")
            with gr.Row(equal_height=True):
                trt_profiles_markdown = gr.Markdown(elem_id="trt_profiles_markdown", value=profiles_markdown())
            with gr.Accordion("Manage engines", open=False):
                gr.Markdown(
                    "Engines only load with the TensorRT version and GPU they were built with. "
                    "Delete ones you no longer need (the file and its entry are removed)."
                )
                with gr.Row(equal_height=True):
                    engine_dropdown = gr.Dropdown(label="Engine", choices=list(get_engine_choices()), elem_id="trt_engine_manage")
                    button_delete_engine = gr.Button(value="Delete engine", variant="stop", elem_id="trt_delete_engine")

        def refresh_profiles():
            return gr.update(value=profiles_markdown()), gr.update(choices=list(get_engine_choices()), value=None)

        def delete_engine(choice):
            filepath = get_engine_choices().get(choice)
            if filepath is not None:
                modelmanager.delete_engine(filepath)
                print(f"[TensorRT] Deleted engine {filepath}", flush=True)
                refresh_unet_options()
            return refresh_profiles()

        button_refresh_profiles.click(refresh_profiles, outputs=[trt_profiles_markdown, engine_dropdown])
        button_delete_engine.click(delete_engine, inputs=[engine_dropdown], outputs=[trt_profiles_markdown, engine_dropdown])

        export_inputs = profile_sliders + [force_rebuild, static_shapes, version, optimization_level]
        button_export_unet.click(export_unet_to_trt, inputs=export_inputs, outputs=[trt_result])
        button_export_default_unet.click(export_unet_to_trt, inputs=export_inputs, outputs=[trt_result])
        button_export_lora_unet.click(export_lora_to_trt, inputs=[trt_lora_dropdown, trt_lora_force_rebuild], outputs=[trt_result])

    return [(trt_interface, "TensorRT", "tensorrt")]
