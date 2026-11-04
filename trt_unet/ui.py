import json
import re
import threading
import traceback

import gradio as gr

from modules import sd_unet, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from . import arch as arch_mod
from . import common, store
from .profiles import FIELDS, PRESETS, Profile, default_profiles, presets_for

JS = "js" if int(gr.__version__.split(".")[0]) >= 4 else "_js"
NEW = "New profile"
FIELD_LABELS = {"width": "Width", "height": "Height", "batch": "Batch size", "tokens": "Prompt tokens"}
LEVEL_LABELS = {"min": "min", "opt": "optimal", "max": "max"}
BUILDING = threading.Event()
HELP = """
**Engines** are compiled for a UNet architecture, not a checkpoint: every checkpoint with that architecture (fine-tunes, merges, v-prediction ones) runs on the same engines with its own weights, LoRAs included. Engine files hold no weights.

**Profiles** are the image sizes, batch sizes and prompt lengths the engines run, one engine each. TensorRT tunes each profile for its optimal values and accepts anything from min to max; narrower ranges build faster and can run a little faster. Only one engine is loaded at a time, taking about as much VRAM as the checkpoint's UNet. Each batch starts on the engine whose profile fits it best, and its later passes (e.g. Hires. fix) stay on that engine while it covers their sizes; a pass it doesn't cover loads the engine of a profile that does. Anything no profile covers, and ControlNet or other UNet patches, runs on PyTorch.

**Building** makes the engines match the profiles: it builds the profiles that have no engine yet, one at a time, and deletes the engines of profiles that are gone.

**SD Unet** (Settings, Stable Diffusion) decides when engines run: `Automatic` uses the engine of the loaded checkpoint's architecture, `None` always uses PyTorch.
"""


def _span(values) -> str:
    lo, opt, hi = values
    return f"{opt}" if lo == hi else f"{lo}-{hi} ({opt})"


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _preset_values():
    """{name: the 12 form values} of every family's presets (their names are distinct)."""
    return {name: [v for f in FIELDS for v in getattr(p, f)] for presets in PRESETS.values() for name, p in presets.items()}


def _preset_choices(arch):
    """The loaded architecture's presets."""
    return list(_preset_values()) if arch is None else list(presets_for(arch.adm_channels))


def _current():
    """(sd_model, patcher, arch, records) for the loaded checkpoint: records are its architecture's engines
    (maybe none); the others may be None."""
    sd_model = arch_mod.loaded_model()
    patcher = arch_mod.unet_patcher(sd_model)
    arch = arch_mod.detect(patcher) if patcher is not None else None
    records = []
    if arch is not None:
        try:
            records = store.engines(arch.id)
        except Exception:
            pass
    return sd_model, patcher, arch, records


def _stale(arch) -> list:
    """Engines for this architecture that were built for another GPU or TensorRT version (one such build's)."""
    if arch is None:
        return []
    try:
        others = [r for r in store.records() if r.arch.id == arch.id and not r.loadable]
        return [r for r in others if r.group == others[0].group]
    except Exception:
        return []


def _keys(draft) -> list:
    return [store.profile_key(Profile.from_dict(d)) for d in draft]


def _plan(draft, records):
    """(indexes of the draft's profiles that have no engine, how many engines no profile has)."""
    built = {store.profile_key(r.profile) for r in records}
    keys = _keys(draft)
    todo = [i for i, k in enumerate(keys) if k not in built and k not in keys[:i]]
    return todo, len(built - set(keys))


def _status(sd_model, arch, records) -> str:
    try:
        version = common.trt_version()
    except Exception as e:
        return f"### TensorRT isn't available\n`{e}`\n\nRestart the WebUI: the extension installs or updates it."
    if sd_model is None:
        return "No checkpoint is loaded. Load one, then refresh this tab."
    name = sd_model.sd_checkpoint_info.model_name
    if arch is None:
        return f"`{name}` isn't a model TensorRT engines can run (SD 1.x, SD 2.x and SDXL architectures are supported)."
    lines = [f"**Checkpoint:** `{name}` · **Architecture:** {arch.label} · **TensorRT** {version}"]
    if not records:
        stale = _stale(arch)
        if stale:
            lines.append(
                f"**The engines for this architecture were built for {stale[0].meta['gpu_name']} with TensorRT {stale[0].meta['trt']}, "
                "so they can't load here.** Build them again; their profiles are below."
            )
        else:
            lines.append("There are no engines for this architecture yet. The profiles below cover every common size: build them.")
        return "\n\n".join(lines)
    lines.append(
        f"**Engines:** {_n(len(records), 'profile')}, last built {max(r.meta['created'] for r in records)}; "
        f"the one in use takes about {common.gib(records[0].weight_bytes)} of VRAM."
    )
    setting = getattr(shared.opts, "sd_unet", "Automatic")
    if setting in ("Automatic", f"[TRT] {arch.label}"):
        lines.append(f"`SD Unet` is `{setting}`, so every {arch.name} checkpoint runs on them.")
    else:
        lines.append(f"**`SD Unet` is `{setting}`, so generations don't use these engines.** Set it to `Automatic` in Settings, Stable Diffusion.")
    return "\n\n".join(lines)


def _draft_view(draft: list, records) -> str:
    if not draft:
        return "*No profiles. Add at least one to build an engine.*"
    built = {store.profile_key(r.profile) for r in records}
    rows = ["| # | Width | Height | Batch size | Prompt tokens | Engine |", "|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for i, (d, key) in enumerate(zip(draft, _keys(draft))):
        p = Profile.from_dict(d)
        rows.append(f"| {i + 1} | {_span(p.width)} | {_span(p.height)} | {_span(p.batch)} | {_span(p.tokens)} | {'built' if key in built else '**to build**'} |")
    rows.append("\n*Ranges show min-max (optimal).*")
    _, extra = _plan(draft, records)
    if extra:
        rows.append(f"**Building also deletes the {_n(extra, 'engine')} of profiles no longer listed.**")
    return "\n".join(rows)


def _label(i: int, d: dict) -> str:
    p = Profile.from_dict(d)
    return f"#{i + 1}: {_span(p.width)} x {_span(p.height)}"


def _index(choice):
    match = re.match(r"#(\d+):", choice or "")
    return int(match.group(1)) - 1 if match else None


def _draft_outputs(draft, records, select=None):
    """Profile table, profile selector and build button for a draft."""
    choices = [NEW] + [_label(i, d) for i, d in enumerate(draft)]
    selected = _label(select, draft[select]) if select is not None and select < len(draft) else NEW
    todo, extra = _plan(draft, records)
    build_label = f"Build {_n(len(todo), 'engine')}" if todo else "Update engines" if extra else "Engines up to date"
    return (
        _draft_view(draft, records),
        gr.update(choices=choices, value=selected),
        gr.update(value=build_label, interactive=bool(draft) and bool(todo or extra) and not BUILDING.is_set()),
    )


def _groups() -> dict:
    """{label: the engines one architecture has for one GPU and TensorRT version}."""
    groups = {}
    for r in store.records():
        groups.setdefault(f"{r.arch.label} · {r.meta['gpu_name']}, TensorRT {r.meta['trt']}", []).append(r)
    return groups


def _engines_view(arch) -> str:
    groups = _groups()
    if not groups:
        return "*No engines yet.*"
    rows = ["| Architecture | Profiles | VRAM | Built for |", "|---|:-:|:-:|---|"]
    for group in groups.values():
        r = group[0]
        name = f"**{r.arch.label}** · current" if r.loadable and arch is not None and r.arch.id == arch.id else r.arch.label
        built = f"{r.meta['gpu_name']}, TensorRT {r.meta['trt']}" + ("" if r.loadable else " · **can't load here**")
        rows.append(f"| {name} | {len(group)} | {common.gib(r.weight_bytes)} | {built} |")
    return "\n".join(rows)


def refresh(draft=None, draft_arch=None):
    """Everything on the tab, for the loaded checkpoint. Profiles being edited stay unless the architecture changed."""
    sd_model, _, arch, records = _current()
    form = [gr.update()] * (len(FIELDS) * 3)
    if arch is not None and (draft is None or draft_arch != arch.id):
        source = records or _stale(arch)
        draft = [p.to_dict() for p in ([r.profile for r in source] if source else default_profiles(arch.adm_channels))]
        form = [v for f in FIELDS for v in getattr(default_profiles(arch.adm_channels)[0], f)]
    elif arch is None:
        draft = []
    engines = list(_groups())
    return (
        _status(sd_model, arch, records),
        arch.id if arch else None,
        draft,
        *_draft_outputs(draft, records),
        *form,
        gr.update(choices=_preset_choices(arch), value=None),
        _engines_view(arch),
        gr.update(choices=engines, value=None, visible=bool(engines)),
        gr.update(visible=bool(engines)),
        gr.update(visible=BUILDING.is_set()),
    )


def _refresh_unet_list():
    from . import runtime

    runtime.release_all()
    if hasattr(sd_unet.current_unet, "option") and isinstance(sd_unet.current_unet.option, runtime.TrtUnetOption):
        sd_unet.current_unet = sd_unet.current_unet_option = None
    sd_unet.list_unets()


def select_profile(draft, choice):
    index = _index(choice)
    if index is None or index >= len(draft):
        return (*[gr.update()] * 12, gr.update(visible=False), gr.update(visible=False))
    values = [v for f in FIELDS for v in draft[index][f]]
    return (*values, gr.update(visible=True, value=f"Save changes to #{index + 1}"), gr.update(visible=True, value=f"Remove #{index + 1}"))


def _read_profile(values):
    """(Profile, note) from the form, or (None, error)."""
    try:
        numbers = [int(v) for v in values]
    except (TypeError, ValueError):
        return None, "Fill in every field with a whole number."
    if min(numbers) < 1:
        return None, "Every value must be at least 1."
    _, _, arch, _ = _current()
    raw = Profile(*(tuple(numbers[i * 3:i * 3 + 3]) for i in range(len(FIELDS))))
    profile = raw.validated(arch.multiple if arch else 64)
    note = "" if profile == raw else f"Adjusted to sizes the model takes: {_span(profile.width)} x {_span(profile.height)}, batch {_span(profile.batch)}, {_span(profile.tokens)} tokens."
    return profile, note


def _same_as(draft, profile, skip=None):
    """The index of the draft's profile equal to `profile`, `skip` aside, or None: profiles are one engine
    each, so each is listed once."""
    key = store.profile_key(profile)
    return next((i for i, k in enumerate(_keys(draft)) if k == key and i != skip), None)


def add_profile(draft, *values):
    records = _current()[3]
    profile, note = _read_profile(values)
    if profile is None:
        return (draft, *_draft_outputs(draft, records), note)
    same = _same_as(draft, profile)
    if same is not None:
        return (draft, *_draft_outputs(draft, records, select=same), f"That's profile #{same + 1} already.")
    draft = list(draft) + [profile.to_dict()]
    return (draft, *_draft_outputs(draft, records, select=len(draft) - 1), note)


def save_profile(draft, choice, *values):
    records = _current()[3]
    index = _index(choice)
    profile, note = _read_profile(values)
    if profile is None or index is None or index >= len(draft):
        return (draft, *_draft_outputs(draft, records, select=index), note)
    same = _same_as(draft, profile, skip=index)
    if same is not None:
        return (draft, *_draft_outputs(draft, records, select=index), f"That's profile #{same + 1} already; #{index + 1} is unchanged.")
    draft = list(draft)
    draft[index] = profile.to_dict()
    return (draft, *_draft_outputs(draft, records, select=index), note)


def remove_profile(draft, choice):
    index = _index(choice)
    if index is not None and index < len(draft):
        draft = [d for i, d in enumerate(draft) if i != index]
    return (draft, *_draft_outputs(draft, _current()[3]), "")


def _outcome(arch, todo: int, deleted: int, result) -> str:
    built = len(result["records"])
    if result["ok"]:
        message = f"### Engines up to date\nBuilt {_n(built, 'engine')}" + (f", deleted {_n(deleted, 'engine')} of profiles no longer listed" if deleted else "")
        message += f". Every {arch.name} checkpoint runs on them with `SD Unet` set to `Automatic`."
        if any(r.meta.get("stripped") is False for r in result["records"]):
            message += " (TensorRT couldn't leave the weights out of some of them, so their files are as large as the UNet.)"
    else:
        done = f"{_n(built, 'engine')} of {todo} built; the other engines are unchanged."
        if result.get("error") == "cancelled":
            message = f"### Build cancelled\n{done}"
        else:
            message = f"### The build failed\n`{result.get('error')}`\n\nThe console has the details. {done}"
    if result.get("memory_warnings"):
        message += (
            f"\n\nTensorRT skipped {result['memory_warnings']} kernels that need more VRAM than is free at a profile's largest sizes, "
            "so those sizes may run a little slower. Narrower profiles avoid this."
        )
    return message


def build(draft, draft_arch, optimization_level, progress=gr.Progress()):
    from modules.call_queue import queue_lock

    from . import export

    export.CANCEL.clear()
    BUILDING.set()
    try:
        with queue_lock:  # not during a generation
            _, patcher, arch, records = _current()
            if arch is None:
                raise RuntimeError("Load a checkpoint TensorRT engines can run first.")
            if arch.id != draft_arch:  # the checkpoint changed since the tab showed these profiles
                message, draft = "### Nothing built\nThe loaded checkpoint has another architecture now. Check its profiles, then build.", None
            elif not draft:
                raise RuntimeError("Add at least one profile.")
            else:
                profile_list = [Profile.from_dict(d) for d in draft]
                todo, _ = _plan(draft, records)
                _refresh_unet_list()
                result = {"ok": True, "records": []}
                if todo:
                    result = export.build(arch, patcher, [(i, profile_list[i]) for i in todo], int(optimization_level), lambda f, t: progress(f, desc=t))
                deleted = store.keep_only(arch.id, profile_list) if result["ok"] else 0
                _refresh_unet_list()
                message = _outcome(arch, len(todo), deleted, result)
    except Exception as e:
        traceback.print_exc()
        message = f"### The build failed\n`{e}`"
    finally:
        BUILDING.clear()
    return message, draft  # the rest of the tab is refreshed next, without progress shown on all of it


def _build_started():
    return gr.update(interactive=False), gr.update(visible=True), ""


def cancel_build():
    from . import export

    export.CANCEL.set()
    common.log("Cancelling the engine build...")


def delete_engine(choice, draft, draft_arch):
    from modules.call_queue import queue_lock

    group = _groups().get(choice) if choice else None
    if group:
        with queue_lock:  # not while a generation uses them
            _refresh_unet_list()
            store.delete(group)
            sd_unet.list_unets()
        common.log(f"Deleted the {len(group)} engines of {choice}")
    return refresh(draft, draft_arch)


_sd_unet = {"dropdowns": [], "sync": None}  # SD Unet's dropdown (Settings or Quicksettings) and the tab's sync button


def _sd_unet_update():
    from modules import shared_items

    return gr.update(choices=shared_items.sd_unet_items(), value=shared.opts.sd_unet)


def _connect(dropdown):
    _sd_unet["sync"].click(_sd_unet_update, None, dropdown, queue=False)


def on_before_ui():
    _sd_unet.update(dropdowns=[], sync=None)


def on_after_component(component, **kwargs):
    """reForge makes SD Unet's dropdown in Settings before the extension tabs, or in Quicksettings after them."""
    if kwargs.get("elem_id") == "setting_sd_unet":
        _sd_unet["dropdowns"].append(component)
        if _sd_unet["sync"] is not None:
            _connect(component)


def on_ui_tabs():
    presets = _preset_values()
    first = [v for f in FIELDS for v in getattr(default_profiles(2816)[0], f)]
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(equal_height=False):
            status = gr.Markdown(elem_id="trt_status")
            refresh_button = ToolButton(value=refresh_symbol, elem_id="trt_refresh")
        draft, draft_arch = gr.State([]), gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, variant="panel"):
                gr.Markdown("### Profiles")
                table = gr.Markdown(elem_id="trt_profiles")
                with gr.Accordion("Edit profiles", open=False, elem_id="trt_editor"):
                    with gr.Row():
                        selector = gr.Dropdown(label="Profile", choices=[NEW], value=NEW, elem_id="trt_selector")
                        preset = gr.Dropdown(label="Fill in from a preset", choices=[], elem_id="trt_preset")
                    numbers = []
                    for f_index, field in enumerate(FIELDS):
                        with gr.Row():
                            for l_index, level in enumerate(LEVEL_LABELS):
                                numbers.append(
                                    gr.Number(label=f"{FIELD_LABELS[field]} {LEVEL_LABELS[level]}", value=first[f_index * 3 + l_index], precision=0, elem_id=f"trt_{field}_{level}")
                                )
                    with gr.Row():
                        add_button = gr.Button("Add as a new profile", elem_id="trt_add")
                        save_button = gr.Button("Save changes", visible=False, elem_id="trt_save")
                        remove_button = gr.Button("Remove", visible=False, elem_id="trt_remove")
                    note = gr.Markdown(elem_id="trt_note")
                with gr.Accordion("Advanced", open=False):
                    level = gr.Slider(0, 5, value=3, step=1, label="Builder optimization level", info="Higher searches more kernels: much slower builds, sometimes slightly faster engines.")
                with gr.Row():
                    build_button = gr.Button("Build engines", variant="primary", elem_id="trt_build")
                    cancel_button = gr.Button("Cancel build", visible=False, elem_id="trt_cancel")
                result = gr.Markdown(elem_id="trt_result")
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("### Engines")
                engines = gr.Markdown(elem_id="trt_engines")
                engine_choice = gr.Dropdown(label="Engines", choices=[], visible=False, elem_id="trt_engine_choice")
                delete_button = gr.Button("Delete engines", variant="stop", size="sm", visible=False, elem_id="trt_delete")
                with gr.Accordion("How it works", open=False):
                    gr.Markdown(HELP)
        _sd_unet["sync"] = gr.Button(visible=False, elem_id="trt_sd_unet_sync")
        for dropdown in _sd_unet["dropdowns"]:
            _connect(dropdown)

        # SD Unet's dropdown shows the engines (and the setting) of when the page loaded: update it.
        refresh_sd_unet = {JS: "() => { gradioApp().getElementById('trt_sd_unet_sync')?.click(); }"}
        draft_outputs = [draft, table, selector, build_button]
        outputs = [status, draft_arch, *draft_outputs, *numbers, preset, engines, engine_choice, delete_button, cancel_button]
        # In the browser; the preset menu empties again so the same preset can be picked twice.
        preset.change(
            None, [preset], [*numbers, preset],
            **{JS: f"(name, ...current) => [...(({json.dumps(presets)})[name] || current.slice(0, {len(numbers)})), null]"},
        )
        quiet = {"show_progress": "hidden"}  # quick updates: no progress overlay on every component they touch
        refresh_button.click(refresh, [draft, draft_arch], outputs, **quiet)  # also clicked when the tab opens (javascript/tensorrt.js)
        tab.load(refresh, None, outputs, **quiet)
        selector.change(select_profile, [draft, selector], [*numbers, save_button, remove_button], **quiet)
        add_button.click(add_profile, [draft, *numbers], [*draft_outputs, note], **quiet)
        save_button.click(save_profile, [draft, selector, *numbers], [*draft_outputs, note], **quiet)
        remove_button.click(remove_profile, [draft, selector], [*draft_outputs, note], **quiet)
        # The build's progress shows once, in its result.
        build_button.click(_build_started, None, [build_button, cancel_button, result], queue=False, **quiet).then(
            build, [draft, draft_arch, level], [result, draft]
        ).then(refresh, [draft, draft_arch], outputs, **quiet).then(None, None, None, **refresh_sd_unet)
        cancel_button.click(cancel_build, None, None, queue=False)
        delete_button.click(
            delete_engine, [engine_choice, draft, draft_arch], outputs, **quiet,
            **{JS: "(choice, ...rest) => [choice && confirm('Delete the engines of ' + choice + '? Building them again takes a while.') ? choice : null, ...rest.slice(0, 2)]"},
        ).then(None, None, None, **refresh_sd_unet)
    return [(tab, "TensorRT", "tensorrt")]
