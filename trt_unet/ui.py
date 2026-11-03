import json
import traceback

import gradio as gr

from modules import sd_unet, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from . import arch as arch_mod
from . import common, store
from .profiles import FIELDS, PRESETS, Profile, default_profiles

JS = "js" if int(gr.__version__.split(".")[0]) >= 4 else "_js"
LEVELS = ("min", "opt", "max")
FAMILY_NAMES = {"xl": "SDXL", "sd": "SD 1.x/2.x"}
HELP = """
**How it works.** An engine is compiled for a UNet *architecture*, not a checkpoint: every SDXL (or SD 1.5, ...) checkpoint with that architecture runs on it with its own weights, and LoRAs work as usual. Engine files hold no weights, so they're small and load quickly.

**Profiles** are the image sizes, batch sizes and prompt lengths an engine runs. Fixed-size profiles are the fastest; ranges cover more sizes. Add one per size you use, e.g. your base size and your Hires. fix size: switching between them costs nothing. Anything no profile covers runs on PyTorch.

**Use it** with `SD Unet` set to `Automatic` (or the `[TRT]` engine); `None` uses PyTorch. ControlNet and other UNet patches can't run on engines, so those generations use PyTorch.
"""


def _preset_values():
    out = {}
    for family, presets in PRESETS.items():
        for name, p in presets.items():
            out[f"{FAMILY_NAMES[family]}: {name}"] = [v for f in FIELDS for v in getattr(p, f)]
    return out


def _current():
    """(sd_model, patcher, arch, record) for the loaded checkpoint; any may be None."""
    sd_model = arch_mod.loaded_model()
    patcher = arch_mod.unet_patcher(sd_model)
    arch = arch_mod.detect(patcher) if patcher is not None else None
    record = None
    if arch is not None:
        try:
            record = store.find(arch.id)
        except Exception:
            pass
    return sd_model, patcher, arch, record


def _status(sd_model, arch, record) -> str:
    try:
        version = common.trt_version()
    except Exception as e:
        return f"### TensorRT isn't available\n`{e}`. Install it (`pip install tensorrt`) and restart the WebUI."
    if sd_model is None:
        return "No checkpoint is loaded yet. Load one, then refresh."
    name = sd_model.sd_checkpoint_info.model_name
    if arch is None:
        return f"`{name}` isn't a UNet model TensorRT engines can run."
    lines = [f"**Checkpoint:** `{name}` · **Architecture:** {arch.label} · TensorRT {version}"]
    if record is None:
        lines.append("No engine for this architecture yet: pick profiles below and build one.")
    else:
        lines.append(f"**Engine:** {len(record.profiles)} profiles, {common.gib(record.disk_bytes)}, built {record.meta['created']}.")
        setting = getattr(shared.opts, "sd_unet", "Automatic")
        if setting == "Automatic" or setting == f"[TRT] {arch.label}":
            lines.append(f"`SD Unet` is `{setting}`: every {arch.family} checkpoint runs on this engine.")
        else:
            lines.append(f"**`SD Unet` is `{setting}`**, so generations don't use this engine. Set it to `Automatic` (Settings, Stable Diffusion).")
    return "\n\n".join(lines)


def _profiles_table(draft: list) -> str:
    if not draft:
        return "*No profiles yet.*"
    rows = ["| # | Profile |", "|---|---|"]
    rows += [f"| {i} | {Profile.from_dict(d).describe()} |" for i, d in enumerate(draft)]
    return "\n".join(rows)


def _engines_table() -> str:
    records = store.records()
    if not records:
        return "*No engines yet.*"
    rows = ["| Engine | Profiles | Size | Built for | |", "|---|---|---|---|---|"]
    for r in records:
        rows.append(f"| {r.arch.label} | {len(r.profiles)} | {common.gib(r.disk_bytes)} | {r.meta['gpu']}, TensorRT {r.meta['trt']} | {'' if r.loadable else 'not loadable here'} |")
    return "\n".join(rows)


def _engine_choices():
    return {f"{r.arch.label} · {r.meta['gpu']} · TensorRT {r.meta['trt']}": r for r in store.records()}


def _legacy_text():
    files = store.legacy_files()
    if not files:
        return "", gr.update(visible=False)
    return f"{len(files)} files from older versions of this extension use {common.gib(store.size_of(files))}. They aren't used anymore.", gr.update(visible=True)


def refresh(reset_draft=True, draft=None):
    sd_model, _, arch, record = _current()
    if reset_draft or draft is None:
        draft = [p.to_dict() for p in (record.profiles if record else default_profiles(arch.adm_channels if arch else 0))] if arch else []
    legacy, legacy_visible = _legacy_text()
    return (
        _status(sd_model, arch, record),
        draft,
        _profiles_table(draft),
        gr.update(choices=[str(i) for i in range(len(draft))], value=None),
        _engines_table(),
        gr.update(choices=list(_engine_choices()), value=None),
        legacy,
        legacy_visible,
    )


def _refresh_unet_list():
    from . import runtime

    runtime.release_all()
    if hasattr(sd_unet.current_unet, "option") and isinstance(sd_unet.current_unet.option, runtime.TrtUnetOption):
        sd_unet.current_unet = sd_unet.current_unet_option = None
    sd_unet.list_unets()


def add_profile(draft, *values):
    _, _, arch, _ = _current()
    try:
        numbers = [int(v) for v in values]
    except (TypeError, ValueError):
        return draft, _profiles_table(draft), gr.update(), "Fill in every field with a number."
    raw = Profile(*(tuple(numbers[i * 3:i * 3 + 3]) for i in range(len(FIELDS))))
    profile = raw.validated(arch.multiple if arch else 64)
    draft = list(draft) + [profile.to_dict()]
    note = "" if profile == raw else f"Adjusted to {profile.describe()}."
    return draft, _profiles_table(draft), gr.update(choices=[str(i) for i in range(len(draft))], value=None), note


def remove_profile(draft, index):
    if index not in (None, ""):
        draft = [d for i, d in enumerate(draft) if i != int(index)]
    return draft, _profiles_table(draft), gr.update(choices=[str(i) for i in range(len(draft))], value=None)


def build(draft, optimization_level, progress=gr.Progress()):
    from modules.call_queue import queue_lock

    from . import export

    message = ""
    try:
        with queue_lock:
            _, patcher, arch, _ = _current()
            if arch is None:
                raise RuntimeError("Load a checkpoint TensorRT engines can run first.")
            if not draft:
                raise RuntimeError("Add at least one profile.")
            _refresh_unet_list()
            result = export.build(arch, patcher, [Profile.from_dict(d) for d in draft], int(optimization_level), lambda f, t: progress(f, desc=t))
            _refresh_unet_list()
        if result["ok"]:
            message = f"### Built the {arch.label} engine\n{result['record'].describe()}."
        else:
            message = f"### The build failed\n`{result.get('error')}`\n\nThe console has the details."
        if result.get("memory_warnings"):
            message += (
                f"\n\nTensorRT skipped {result['memory_warnings']} kernels that need more VRAM than is free at a profile's largest sizes, "
                "so those sizes may run a little slower. Narrower profiles avoid this."
            )
    except Exception as e:
        traceback.print_exc()
        message = f"### The build failed\n`{e}`"
    return (message, *refresh(reset_draft=False, draft=draft))


def delete_engine(choice, draft):
    from modules.call_queue import queue_lock

    record = _engine_choices().get(choice)
    if record is not None:
        with queue_lock:  # not while a generation uses it
            _refresh_unet_list()
            store.delete(record)
            sd_unet.list_unets()
        common.log(f"Deleted {record.plan}")
    return refresh(reset_draft=False, draft=draft)


def delete_legacy(draft):
    from modules.call_queue import queue_lock

    with queue_lock:
        freed = store.remove_legacy_files()
    common.log(f"Deleted files from older versions of the extension ({common.gib(freed)})")
    return refresh(reset_draft=False, draft=draft)


def on_ui_tabs():
    presets = _preset_values()
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            status = gr.Markdown(elem_id="trt_status")
            refresh_button = ToolButton(value=refresh_symbol, elem_id="trt_refresh")
        draft = gr.State([])
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("### Profiles")
                table = gr.Markdown(elem_id="trt_profiles")
                with gr.Row():
                    remove_index = gr.Dropdown(label="Profile #", choices=[], elem_id="trt_remove_index")
                    remove_button = gr.Button("Remove", elem_id="trt_remove")
                with gr.Accordion("Add a profile", open=True):
                    preset = gr.Dropdown(label="Preset", choices=list(presets), elem_id="trt_preset")
                    numbers = []
                    first = next(iter(presets.values()))
                    for f_index, (field, label) in enumerate(zip(FIELDS, ("Width", "Height", "Batch size", "Prompt tokens"))):
                        with gr.Row():
                            for l_index, level in enumerate(LEVELS):
                                numbers.append(gr.Number(label=f"{label} ({level})", value=first[f_index * 3 + l_index], precision=0, elem_id=f"trt_{field}_{level}"))
                    note = gr.Markdown()
                    add_button = gr.Button("Add profile", elem_id="trt_add")
                with gr.Accordion("Advanced", open=False):
                    level = gr.Slider(0, 5, value=3, step=1, label="Builder optimization level", info="Higher searches more kernels: much slower builds, sometimes slightly faster engines.")
                build_button = gr.Button("Build engine", variant="primary", elem_id="trt_build")
                result = gr.Markdown(elem_id="trt_result")
            with gr.Column(variant="panel"):
                gr.Markdown("### Engines")
                engines = gr.Markdown(elem_id="trt_engines")
                with gr.Row():
                    engine_choice = gr.Dropdown(label="Engine", choices=[], elem_id="trt_engine_choice")
                    delete_button = gr.Button("Delete", variant="stop", elem_id="trt_delete")
                legacy = gr.Markdown()
                legacy_button = gr.Button("Delete old files", visible=False, elem_id="trt_legacy")
                gr.Markdown(HELP)

        outputs = [status, draft, table, remove_index, engines, engine_choice, legacy, legacy_button]
        preset.change(None, [preset, *numbers], numbers, **{JS: f"(name, ...current) => ({json.dumps(presets)})[name] || current"})
        refresh_button.click(refresh, None, outputs)
        tab.load(refresh, None, outputs)
        add_button.click(add_profile, [draft, *numbers], [draft, table, remove_index, note])
        remove_button.click(remove_profile, [draft, remove_index], [draft, table, remove_index])
        build_button.click(build, [draft, level], [result, *outputs])
        delete_button.click(delete_engine, [engine_choice, draft], outputs)
        legacy_button.click(delete_legacy, draft, outputs)
    return [(tab, "TensorRT", "tensorrt")]
