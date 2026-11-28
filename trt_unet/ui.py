import contextlib
import json
import re
import threading

import gradio as gr

from modules import shared
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from . import arch as arch_mod
from . import common, compat, quant, store
from . import vae as vae_mod
from .profiles import FIELDS, PRESETS, Profile, default_profiles, presets_for

JS = "js" if int(gr.__version__.split(".")[0]) >= 4 else "_js"
NEW = "New profile"
FIELD_LABELS = {"width": "Width", "height": "Height", "batch": "Batch size", "tokens": "Prompt tokens"}
LEVEL_LABELS = {"min": "min", "opt": "optimal", "max": "max"}
BUILDING = threading.Event()
_BUILD_LOCK = threading.Lock()
# Per process: each WebUI start with no engines asks for the mode again, even if one was saved before.
SETTLED = threading.Event()
MODE_LABELS = {store.PER_PROFILE: "Per-profile engines", store.MERGED: "Merged engine"}
MERGED_VRAM, MERGED_RAM = 24, 64
HELP = """
**Engines** are built per architecture, rather than per checkpoint. They hold no weights – those of the loaded checkpoint, and LoRAs are loaded into them, so every checkpoint of the same architecture (fine-tunes, merges, v-prediction models, and quantized checkpoints such as FP8, INT8, NVFP4, or GGUF) uses the same engines.

**Profiles** set the resolutions, batch sizes, and prompt lengths the engines are built for. Each is tuned for its optimal values, and accepts anything from min to max – narrower ranges build faster, and can run slightly faster. Anything unsupported by the engines (ControlNet, model patches like FreeU, reference-image edit modes, Skip Layer Guidance, or sizes outside every profile) is run on PyTorch instead, with a notice.

**Engine precision** (in **Advanced**, for DiT models such as Anima and MOSS-Image) chooses the execution precision independently of how the checkpoint is stored on disk. `Auto` picks the highest-quality precision supported by your GPU that fits VRAM (`BF16`, `FP8`, `NVFP4` on Blackwell, `INT8`, `INT4`, or `FP16` with an FP32 residual stream on Turing). **Keep engines for other precisions** retains previously built precisions on disk so you can switch between them without rebuilding; uncheck it to remove other precisions on the next build.

**Engine modes** are chosen on first build, and can only be changed while no engines exist:
- *Per-profile engines* – one engine per profile, loaded as needed. Only one is loaded at a time, using about as much VRAM as the checkpoint's model. Each batch starts on the engine that fits it best, and stays on it for any later pass (ex. Hires. fix) it covers.
- *Merged engine* – every profile in one engine, which never needs swapping, but holds every profile's share of TensorRT's memory at once.

Swapped-out engines are kept in RAM (with their weights) when there's room to spare, making them quick to bring back. The system takes that RAM back whenever it needs it.

**Build engines** builds an engine for each profile without one, and deletes those of removed profiles. A merged engine is rebuilt whenever the profiles change.

**SD Unet** (in **Settings → Stable Diffusion**) chooses what runs the model: `Automatic` (the default) uses the engines, `Speed Hacks` uses them with Speed Hacks, and `None` returns to PyTorch. It's also an axis of the X/Y/Z plot script.

**The VAE**'s encoder, and decoder get engines too (for SD 1.x / SD 2.x / SDXL), built alongside the UNet's for the sizes its profiles cover. Whole images run in the precision the WebUI uses for the VAE (BF16, or FP32 on GPUs without it). Large decodes (ex. Hires. fix) are split into strips on engines of their own to save VRAM, with identical results – in FP16 on GPUs without BF16, with the VAE scaled down between its normalizations so nothing overflows. Each decode, and encode takes the first of these with room: whole, in strips, then either with the UNet's engine unloaded – otherwise, it runs on PyTorch.

**Speed Hacks** accelerate denoising steps with adaptive closed-loop error control:
- On **UNets** (SD 1.x / SD 2.x / SDXL), they implement [DeepCache](https://github.com/horseee/DeepCache) – about half of the steps run on a small engine of the outermost blocks using cached deep features, with fresh noise separated out for ancestral/SDE samplers.
- On **DiTs** (Anima / MOSS-Image), they combine quality-tiered trajectory forecasting (self-selecting between VDE and FreqCa in clean-image $\\hat{x}_0$-space with Wiener/guidance-aware error checks) and late/mid-pass CFG acceleration, with zero extra engine or VRAM overhead while keeping ancestral/SDE samplers fresh-noise safe.

The first 15%, and last 5% of steps are always computed in full, as they determine composition, and fine detail. In between, **quality** sets how long a full step may wait while the measured error stays within tolerance. All three are adjustable in **Settings → Stable Diffusion → TensorRT**, and in the X/Y/Z plot script.
"""


def _span(values) -> str:
    lo, opt, hi = values
    return f"{opt}" if lo == hi else f"{lo}-{hi} ({opt})"


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _machine():
    vram = ram = None
    try:
        import torch

        vram = torch.cuda.get_device_properties(common.device()).total_memory
    except Exception:
        pass
    try:
        from . import refits

        ram = refits.memory()[2]
    except Exception:
        pass
    return vram, ram


def _mode_help(selected: str) -> str:
    vram, ram = _machine()
    have = " and ".join(f"{common.gib(n)} of {what}" for n, what in ((vram, "VRAM"), (ram, "RAM")) if n)
    short = [what for n, what, need in ((vram, "VRAM", MERGED_VRAM), (ram, "RAM", MERGED_RAM)) if n and n < need * 10**9]
    machine = f"This machine has {have}" + (f", less {' and '.join(short)} than a merged engine needs." if short else ".") if have else ""
    if short and selected == store.MERGED:
        machine = f"**{machine}** Building or loading a merged engine is likely to run out of memory here."
    return "\n\n".join(line for line in (
        "**Per-profile engines**: an engine for each profile, built one at a time, and only the one a generation's sizes need is "
        "loaded. They need about the VRAM and RAM of one profile. A pass in sizes the loaded engine doesn't cover loads the "
        "engine of a profile that does, which takes a few seconds.",
        f"**Merged engine**: one engine holding every profile of an architecture, so passes in any of their sizes run without "
        f"loading another engine. It holds every profile's share of TensorRT's memory at once, and its build takes them all "
        f"at once: **it needs at least {MERGED_VRAM} GB of VRAM and {MERGED_RAM} GB of RAM.**",
        machine,
        "The mode is for every engine, and can only change while there are none.",
    ) if line)


def _mode_of(label) -> str:
    return next((m for m, text in MODE_LABELS.items() if text == label), store.PER_PROFILE)


def _precision_choices(arch, patcher=None) -> tuple:
    if arch is None or not arch.is_dit:
        return ["Auto"], "Auto"
    auto_p = store.active_precision(arch, patcher, requested=quant.AUTO)
    auto_label = f"Auto ({quant.LABELS.get(auto_p, auto_p)})"
    supported = quant.supported_precisions()
    choices = [auto_label] + [quant.LABELS[p] for p in supported]
    saved = store.saved_precision(arch.id)
    if saved == quant.AUTO or saved not in supported:
        return choices, auto_label
    return choices, quant.LABELS[saved]


def _precision_of(label) -> str:
    if not label or str(label).startswith("Auto"):
        return quant.AUTO
    return next((p for p, text in quant.LABELS.items() if text == label), quant.AUTO)


def _mode_outputs(selected=None):
    have = bool(store.records())
    if have:
        SETTLED.set()
    mode = store.mode()
    asking = not SETTLED.is_set()
    shown = _mode_of(selected) if selected and not have else mode
    if asking:
        title = "### Choose an engine mode\nThere are no engines yet. Choose how they're built:"
    elif have:
        title = f"**Engine mode:** {MODE_LABELS[mode]} · to change it, delete every engine first (in the Engines panel)."
    else:
        title = f"**Engine mode:** {MODE_LABELS[mode]} · there are no engines, so it can still change."
    return (
        title,
        gr.update(value=MODE_LABELS[shown], visible=not have, interactive=not BUILDING.is_set()),
        gr.update(value=_mode_help(shown), visible=not have),
        gr.update(visible=asking),
        gr.update(visible=not asking),
    )


def _preset_values():
    return {name: [v for f in FIELDS for v in getattr(p, f)] for presets in PRESETS.values() for name, p in presets.items()}


def _preset_choices(arch):
    return list(_preset_values()) if arch is None else list(presets_for(arch))


def _current():
    sd_model = arch_mod.loaded_model()
    patcher = arch_mod.unet_patcher(sd_model)
    arch = arch_mod.detect(patcher) if patcher is not None else None
    precision = store.active_precision(arch, patcher) if (arch is not None and arch.is_dit) else None
    records = []
    if arch is not None:
        try:
            records = store.engines(arch.id, precision=precision)
        except Exception:
            pass
    return sd_model, patcher, arch, records, precision


def _stale(arch) -> list:
    if arch is None:
        return []
    try:
        others = [r for r in store.records() if r.arch.id == arch.id and not r.loadable]
        return [r for r in others if r.group == others[0].group]
    except Exception:
        return []


def _other_precision_count(arch, precision) -> int:
    if arch is None or not arch.is_dit or not precision or store.keep_precisions():
        return 0
    try:
        return sum(1 for r in store.engines(arch.id) if r.precision != precision)
    except Exception:
        return 0


def _keys(draft, precision=None) -> list:
    return [f"{precision}-{store.profile_key(Profile.from_dict(d))}" if precision else store.profile_key(Profile.from_dict(d)) for d in draft]


def _plan(draft, records, mode, precision=None):
    keys = _keys(draft, precision)
    wanted = store.wanted_keys([Profile.from_dict(d) for d in draft], mode, precision=precision) if draft else set()
    built = {r.engine_key for r in records if not r.retry_speed_hacks and not r.stale_graph}
    if mode == store.MERGED:
        todo = list(range(len(draft))) if draft and not wanted <= built else []
    else:
        todo = [i for i, k in enumerate(keys) if k not in built and k not in keys[:i]]
    return todo, sum(1 for r in records if r.engine_key not in wanted)


def _pending_neo_checkpoint():
    # Forge Neo updates forge_loading_parameters when a checkpoint is picked, but defers forge_model_reload() until
    # the next generation. Detect when a checkpoint is selected in Neo's dropdown yet not loaded (or changed).
    if not compat.is_neo():
        return None
    try:
        from modules import sd_models

        md = getattr(sd_models, "model_data", None)
        params = getattr(md, "forge_loading_parameters", None) or {}
        info = params.get("checkpoint_info")
        if info is None or not callable(getattr(sd_models, "forge_model_reload", None)):
            return None
        if not compat.is_loaded_sd_model(getattr(md, "sd_model", None)) or getattr(md, "forge_hash", "") != str(params):
            return info
    except Exception:
        pass
    return None


def _status(sd_model, arch, records, precision=None, asking=False) -> str:
    try:
        version = common.trt_version()
    except Exception as e:
        return f"### TensorRT isn't available\n`{e}`\n\nRestart the WebUI: the extension installs or updates it."
    pending = _pending_neo_checkpoint()
    if not compat.is_loaded_sd_model(sd_model):
        if pending is not None:
            return f"Checkpoint `{pending.model_name}` is selected, but Forge Neo hasn't loaded it yet. Click **Load selected checkpoint** (or generate an image first), then build engines."
        return "No checkpoint is loaded. Load one, then refresh this tab."
    name = sd_model.sd_checkpoint_info.model_name
    pending_note = (
        f"*Selected checkpoint `{pending.model_name}` isn't loaded yet — click **Load selected checkpoint** to switch before building.*"
        if pending is not None else ""
    )
    if arch is None:
        base = f"`{name}` isn't a model TensorRT engines can run (SD 1.x, SD 2.x, SDXL, Anima, and MOSS-Image architectures are supported)."
        return f"{base}\n\n{pending_note}" if pending_note else base
    prec_badge = f" · **Precision:** {quant.LABELS.get(precision, precision)}" if arch.is_dit and precision else ""
    lines = [f"**Checkpoint:** `{name}` · **Architecture:** {arch.label}{prec_badge} · **TensorRT** {version}"]
    if pending_note:
        lines.append(pending_note)
    if not records:
        other_precs = []
        if arch.is_dit:
            try:
                other_precs = sorted({quant.LABELS.get(r.precision, r.precision) for r in store.engines(arch.id)})
            except Exception:
                pass
        if other_precs:
            lines.append(
                f"There are no {quant.LABELS.get(precision, precision)} engines for this architecture yet "
                f"(built precisions: {', '.join(other_precs)}). Build them below, or pick a built precision in **Advanced**."
            )
        else:
            stale = _stale(arch)
            if stale:
                lines.append(
                    f"**The engines for this architecture were built for {stale[0].meta['gpu_name']} with TensorRT {stale[0].meta['trt']}, "
                    "so they can't load here.** Build them again; their profiles are below."
                )
            else:
                lines.append(
                    "There are no engines for this architecture yet. "
                    + ("Choose the engine mode below, then build them." if asking else "The profiles below cover every common size: build them.")
                )
        return "\n\n".join(lines)
    held, created = len(store.slots(records)), max(r.meta["created"] for r in records)
    if records[0].merged:
        lines.append(f"**Engine:** merged, {_n(held, 'profile')}, built {created}; it takes at least {common.gib(records[0].weight_bytes)} of VRAM.")
    else:
        lines.append(f"**Engines:** {_n(held, 'profile')}, last built {created}; the one in use takes about {common.gib(records[0].weight_bytes)} of VRAM.")
    setting = getattr(shared.opts, "sd_unet", "Automatic")
    if setting == "Automatic":
        lines.append(f"`SD Unet` is `Automatic`, so every {arch.name} checkpoint runs on them. `Speed Hacks` runs them faster, a little less precisely.")
    elif setting == "Speed Hacks":
        lines.append(f"`SD Unet` is `Speed Hacks`, so every {arch.name} checkpoint runs on them, with Speed Hacks.")
    else:
        lines.append(f"**`SD Unet` is `{setting}`, so generations don't use these engines.** Set it to `Automatic` or `Speed Hacks` in Settings, Stable Diffusion.")
    vae_engines = _vae_status(arch)
    if vae_engines:
        lines.append(vae_engines)
    old_graph = [r for r in records if r.stale_graph]
    if old_graph:
        n = len(old_graph)
        if arch.is_dit:
            lines.append(f"**{_n(n, 'engine')} {'needs' if n == 1 else 'need'} rebuilding for an updated {arch.name} graph.** Building again updates {'it' if n == 1 else 'them'}.")
        else:
            lines.append(f"**{_n(n, 'engine')} {'needs' if n == 1 else 'need'} rebuilding for an updated UNet graph** (fixes TensorRT 11 GroupNorm artifacts). Building again updates {'it' if n == 1 else 'them'}.")
    stale = [r for r in records if r.stale_speed_hacks]
    if stale:
        lines.append(f"Speed Hacks are off for {_n(len(stale), 'engine')}: their Speed Hacks part is from another version of this extension. Building again rebuilds them.")
    failed = [r for r in records if not r.supports_speed_hacks and not r.stale_speed_hacks]
    if failed:
        retry = "building again retries once" if any(r.retry_speed_hacks for r in failed) else "to try again, delete them and build again"
        reason = failed[0].meta.get("speed_hacks_error") or "not built"
        lines.append(f"Speed Hacks couldn't be built for {_n(len(failed), 'engine')} (`{reason}`), so they run without them; {retry}.")
    return "\n\n".join(lines)


def _vae_status(arch) -> str:
    try:
        vae = vae_mod.loaded_vae()
        record = vae_mod.record_for(arch.id, vae) if vae is not None and not vae_mod.unsupported(vae) else None
        if record is None:
            return ""
        text = f"**VAE engines:** {vae_mod.precisions(record.dtype, record.strips_dtype)}; decodes up to {record.reach()}"
        if record.encoder is not None:
            text += f"; encodes up to {record.encode_reach()}"
        return text + "."
    except Exception as e:
        common.notify(f"Couldn't describe the VAE engines: {e}")
        return ""


def _vae_part(draft) -> str:
    try:
        return vae_mod.needs_build(arch_mod.detect(), [Profile.from_dict(d) for d in draft])
    except Exception as e:
        common.notify(f"Couldn't check which VAE engines are built: {e}")
        return ""


def _vae_line(part: str) -> str:
    if not vae_mod.available():
        return ""
    if part:
        return f"**Building also builds the engines for {part}.**"
    vae = vae_mod.loaded_vae()
    reason = "no VAE is loaded" if vae is None else vae_mod.unsupported(vae)
    if reason:
        return f"*The VAE runs on PyTorch: {reason}.*"
    return "*The VAE's engines are built for these sizes.*"


def _draft_view(draft: list, records, mode, vae_part: str = "", precision=None, prune_other: int = 0) -> str:
    if not draft:
        return "*No profiles. Add at least one to build an engine.*"
    todo, extra = _plan(draft, records, mode, precision=precision)
    by_key = {r.engine_key: r for r in records}
    merged_key = next(iter(store.wanted_keys([Profile.from_dict(d) for d in draft], mode, precision=precision))) if mode == store.MERGED else None
    rows = ["| # | Width | Height | Batch size | Prompt tokens | Engine |", "|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for i, (d, key) in enumerate(zip(draft, _keys(draft, precision))):
        p = Profile.from_dict(d)
        rec = by_key.get(merged_key or key)
        if i in todo:
            why = "graph update" if rec is not None and rec.stale_graph else "Speed Hacks"
            state = f"**to rebuild** ({why})" if rec is not None else "**to build**"
        else:
            state = "built"
        rows.append(f"| {i + 1} | {_span(p.width)} | {_span(p.height)} | {_span(p.batch)} | {_span(p.tokens)} | {state} |")
    rows.append("\n*Ranges show min-max (optimal).*")
    if mode == store.MERGED:
        rows.append("*One merged engine holds every profile: changing them builds it again.*")
        if extra and todo:
            rows.append("**Building replaces the merged engine built for other profiles.**")
        elif extra:
            rows.append(f"**Building deletes the {_n(extra, 'engine')} these profiles don't use.**")
    elif extra:
        rows.append(f"**Building also deletes the {_n(extra, 'engine')} of profiles no longer listed.**")
    if prune_other:
        rows.append(f"**Building also deletes the {_n(prune_other, 'engine')} of other precisions.**")
    rows.append(_vae_line(vae_part))
    return "\n".join(rows)


def _label(i: int, d: dict) -> str:
    p = Profile.from_dict(d)
    return f"#{i + 1}: {_span(p.width)} x {_span(p.height)}"


def _index(choice):
    match = re.match(r"#(\d+):", choice or "")
    return int(match.group(1)) - 1 if match else None


def _draft_outputs(draft, records, select=None, arch=None, precision=None):
    mode = store.mode()
    choices = [NEW] + [_label(i, d) for i, d in enumerate(draft)]
    selected = _label(select, draft[select]) if select is not None and select < len(draft) else NEW
    todo, extra = _plan(draft, records, mode, precision=precision)
    prune_other = _other_precision_count(arch, precision)
    vae_part = _vae_part(draft) if draft else ""
    if todo:
        build_label = "Build the merged engine" if mode == store.MERGED else f"Build {_n(len(todo), 'engine')}"
    elif vae_part:
        build_label = "Build the VAE's engines"
    else:
        build_label = "Update engines" if (extra or prune_other) else "Engines up to date"
    return (
        _draft_view(draft, records, mode, vae_part, precision=precision, prune_other=prune_other),
        gr.update(choices=choices, value=selected),
        gr.update(value=build_label, interactive=bool(draft) and bool(todo or extra or prune_other or vae_part) and not BUILDING.is_set()),
    )


def _groups() -> dict:
    by_group = {}
    for r in store.records():
        by_group.setdefault(r.group, []).append(r)
    out = {}
    for rs in by_group.values():
        r = rs[-1]
        prec = f" [{quant.LABELS.get(r.precision, r.precision)}]" if r.arch.is_dit else ""
        out[f"{r.arch.label}{prec} · {r.meta['gpu_name']}, TensorRT {r.meta['trt']}"] = rs
    return out


def _engines_view(arch, precision=None) -> str:
    groups = _groups()
    if not groups:
        return "*No engines yet.*"
    rows = ["| Architecture | Profiles | VRAM | Built for |", "|---|:-:|:-:|---|"]
    for group in groups.values():
        r = group[0]
        label = f"{r.arch.label} [{quant.LABELS.get(r.precision, r.precision)}]" if r.arch.is_dit else r.arch.label
        is_current = r.loadable and arch is not None and r.arch.id == arch.id and (not arch.is_dit or r.precision == precision)
        name = f"**{label}** · current" if is_current else label
        built = f"{r.meta['gpu_name']}, TensorRT {r.meta['trt']}" + ("" if r.loadable else " · **can't load here**")
        held = f"{len(store.slots(group))}" + (" (merged)" if any(e.merged for e in group) else "")
        rows.append(f"| {name} | {held} | {common.gib(r.weight_bytes)} | {built} |")
    return "\n".join(rows)


def refresh(draft=None, draft_arch=None, selected=None):
    sd_model, patcher, arch, records, precision = _current()
    form = [gr.update()] * (len(FIELDS) * 3)
    if arch is not None and (draft is None or draft_arch != arch.id):
        saved = store.load_draft(arch.id)
        other_built = store.engines(arch.id) if arch.is_dit and not records else []
        if other_built:
            other_built = [r for r in other_built if r.group == other_built[0].group]
        source = records or other_built or _stale(arch)
        if saved is not None:
            draft = [p.to_dict() for p in saved]
        else:
            draft = [p.to_dict() for p in (store.profiles_of(source) if source else default_profiles(arch))]
        form = [v for f in FIELDS for v in getattr(default_profiles(arch)[0], f)]
    elif arch is None:
        draft = []
    engines = list(_groups())
    mode_outputs = _mode_outputs(selected)
    pending = _pending_neo_checkpoint()
    is_dit = bool(arch is not None and arch.is_dit)
    prec_choices, prec_value = _precision_choices(arch, patcher)
    return (
        _status(sd_model, arch, records, precision=precision, asking=not SETTLED.is_set()),
        arch.id if arch else None,
        draft,
        *_draft_outputs(draft, records, arch=arch, precision=precision),
        *form,
        gr.update(choices=_preset_choices(arch), value=None),
        _engines_view(arch, precision=precision),
        gr.update(choices=engines, value=None, visible=bool(engines)),
        gr.update(visible=bool(engines)),
        gr.update(visible=BUILDING.is_set()),
        *mode_outputs,
        gr.update(visible=pending is not None and not BUILDING.is_set()),
        gr.update(choices=prec_choices, value=prec_value, visible=is_dit, interactive=not BUILDING.is_set()),
        gr.update(value=store.keep_precisions(), visible=is_dit, interactive=not BUILDING.is_set()),
    )


def load_checkpoint(draft=None, draft_arch=None):
    if compat.is_neo() and not BUILDING.is_set():
        from modules import sd_models
        from modules.call_queue import queue_lock

        reload_fn = getattr(sd_models, "forge_model_reload", None)
        if callable(reload_fn):
            try:
                with queue_lock:
                    reload_fn()
            except Exception as e:
                common.notify(f"Couldn't load the selected checkpoint: {e}", e)
    return refresh(draft, draft_arch)


def choose_mode(label, draft, draft_arch):
    try:
        store.set_mode(_mode_of(label))
        SETTLED.set()
    except Exception as e:
        common.logger.warning(f"Couldn't set the engine mode: {e}")
    return refresh(draft, draft_arch)


def change_mode(label, draft, draft_arch):
    if not SETTLED.is_set():
        return refresh(draft, draft_arch, selected=label)
    if _mode_of(label) != store.mode() and not BUILDING.is_set():
        try:
            store.set_mode(_mode_of(label))
        except Exception as e:
            common.logger.warning(f"Couldn't change the engine mode: {e}")
    return refresh(draft, draft_arch)


def _baseline_draft(arch, records):
    saved = store.load_draft(arch.id)
    if saved is not None:
        return [p.to_dict() for p in saved]
    other_built = store.engines(arch.id) if arch.is_dit and not records else []
    if other_built:
        other_built = [r for r in other_built if r.group == other_built[0].group]
    source = records or other_built or _stale(arch)
    return [p.to_dict() for p in (store.profiles_of(source) if source else default_profiles(arch))]


def change_precision(label, draft, draft_arch):
    if not BUILDING.is_set():
        _, _, arch, old_records, _ = _current()
        if arch is not None and arch.is_dit:
            new_prec = _precision_of(label)
            saved = store.saved_precision(arch.id)
            supported = quant.supported_precisions()
            if new_prec != saved and not (new_prec == quant.AUTO and saved not in supported):
                unmodified = draft is None or draft_arch != arch.id or draft == _baseline_draft(arch, old_records)
                try:
                    store.set_precision(arch.id, new_prec)
                except Exception as e:
                    common.logger.warning(f"Couldn't change the engine precision: {e}")
                else:
                    if unmodified and store.load_draft(arch.id) is None:
                        draft = draft_arch = None
    return refresh(draft, draft_arch)


def change_keep_precisions(keep, draft, draft_arch):
    if not BUILDING.is_set() and bool(keep) != store.keep_precisions():
        try:
            store.set_keep_precisions(bool(keep))
        except Exception as e:
            common.logger.warning(f"Couldn't save precision retention setting: {e}")
    return refresh(draft, draft_arch)


def _release_engines():
    from . import runtime

    runtime.release_all()


def select_profile(draft, choice):
    index = _index(choice)
    if index is None or index >= len(draft):
        return (*[gr.update()] * 12, gr.update(visible=False), gr.update(visible=False))
    values = [v for f in FIELDS for v in draft[index][f]]
    return (*values, gr.update(visible=True, value=f"Save changes to #{index + 1}"), gr.update(visible=True, value=f"Remove #{index + 1}"))


def _read_profile(values):
    try:
        numbers = [int(v) for v in values]
    except (TypeError, ValueError):
        return None, "Fill in every field with a whole number."
    if min(numbers) < 1:
        return None, "Every value must be at least 1."
    _, _, arch, _, _ = _current()
    raw = Profile(*(tuple(numbers[i * 3:i * 3 + 3]) for i in range(len(FIELDS))))
    profile = raw.validated_for(arch)
    note = "" if profile == raw else f"Adjusted to sizes the model takes: {_span(profile.width)} x {_span(profile.height)}, batch {_span(profile.batch)}, {_span(profile.tokens)} tokens."
    return profile, note


def _same_as(draft, profile, skip=None):
    key = store.profile_key(profile)
    return next((i for i, k in enumerate(_keys(draft)) if k == key and i != skip), None)


def _sync_draft(arch, draft_arch, draft, records, precision=None):
    if arch is None or arch.id != draft_arch or store.load_draft(arch.id) is None:
        return
    todo, extra = _plan(draft, records, store.mode(), precision=precision)
    prune_other = _other_precision_count(arch, precision)
    vae_part = _vae_part(draft) if draft else ""
    if not draft or (not todo and not extra and not prune_other and not vae_part):
        store.clear_draft(arch.id)
    else:
        store.save_draft(arch.id, [Profile.from_dict(d) for d in draft])


def add_profile(draft, draft_arch, *values):
    _, _, arch, records, precision = _current()
    profile, note = _read_profile(values)
    if profile is None:
        return (draft, *_draft_outputs(draft, records, arch=arch, precision=precision), note)
    same = _same_as(draft, profile)
    if same is not None:
        return (draft, *_draft_outputs(draft, records, select=same, arch=arch, precision=precision), f"That's profile #{same + 1} already.")
    draft = list(draft) + [profile.to_dict()]
    _sync_draft(arch, draft_arch, draft, records, precision=precision)
    return (draft, *_draft_outputs(draft, records, select=len(draft) - 1, arch=arch, precision=precision), note)


def save_profile(draft, draft_arch, choice, *values):
    _, _, arch, records, precision = _current()
    index = _index(choice)
    profile, note = _read_profile(values)
    if profile is None or index is None or index >= len(draft):
        return (draft, *_draft_outputs(draft, records, select=index, arch=arch, precision=precision), note)
    same = _same_as(draft, profile, skip=index)
    if same is not None:
        return (draft, *_draft_outputs(draft, records, select=index, arch=arch, precision=precision), f"That's profile #{same + 1} already; #{index + 1} is unchanged.")
    draft = list(draft)
    draft[index] = profile.to_dict()
    _sync_draft(arch, draft_arch, draft, records, precision=precision)
    return (draft, *_draft_outputs(draft, records, select=index, arch=arch, precision=precision), note)


def remove_profile(draft, draft_arch, choice):
    _, _, arch, records, precision = _current()
    index = _index(choice)
    if index is not None and index < len(draft):
        draft = [d for i, d in enumerate(draft) if i != index]
        _sync_draft(arch, draft_arch, draft, records, precision=precision)
    return (draft, *_draft_outputs(draft, records, arch=arch, precision=precision), "")


def _outcome(arch, todo: int, deleted: int, result, mode) -> str:
    built, merged = len(result["records"]), mode == store.MERGED
    if result["ok"]:
        if merged:
            message = "### Engine up to date\n" + (f"Built the merged engine, {_n(todo, 'profile')}" if built else "The merged engine holds these profiles")
            message += (f", and deleted {_n(deleted, 'engine')} it replaces" if deleted else "") + f". Every {arch.name} checkpoint runs on it with `SD Unet` set to `Automatic` or `Speed Hacks`."
        else:
            first = f"Built {_n(built, 'engine')}" if built else "The engines were up to date"
            message = f"### Engines up to date\n{first}" + (f", deleted {_n(deleted, 'engine')} no longer needed" if deleted else "")
            message += f". Every {arch.name} checkpoint runs on them with `SD Unet` set to `Automatic` or `Speed Hacks`."
        if any(r.meta.get("stripped") is False for r in result["records"]):
            message += " (TensorRT couldn't leave the weights out of some of them, so their files are as large as the model.)"
        if result.get("hack_errors"):
            retry = "Building again retries once." if any(r.retry_speed_hacks for r in result["records"]) else "That failed before too; to try again, delete the engines and build again."
            message += f"\n\nSpeed Hacks couldn't be built (`{result['hack_errors'][-1]}`), so these engines run without them. {retry} TensorRT's own output is logged with `--loglevel DEBUG`."
    else:
        done = "The engines are unchanged." if merged else f"{_n(built, 'engine')} of {todo} built; the other engines are unchanged."
        if result.get("error") == "cancelled":
            message = f"### Build cancelled\n{done}"
        else:
            message = f"### The build failed\n`{result.get('error')}`\n\n{done} TensorRT's own output is logged with `--loglevel DEBUG`."
            if merged and "memory" in str(result.get("error")).lower():
                message += (
                    f" A merged engine's build needs at least {MERGED_RAM} GB of RAM; per-profile engines build one profile at a "
                    "time (to switch, delete every engine)."
                )
    if result.get("vae") is not None:
        vae_record = result["vae"]
        message += (
            f"\n\nBuilt the VAE's engines, in {vae_mod.precisions(vae_record.dtype, vae_record.strips_dtype)}, for decodes up to {vae_record.reach()}: "
            "decodes run on them whole whenever there can be room, else in strips, with what doesn't fit held in RAM."
            if vae_record.strips else
            f"\n\nBuilt the VAE's engines, in {vae_mod.precisions(vae_record.dtype, vae_record.strips_dtype)}, for decodes up to {vae_record.reach()}: "
            "decodes run on them whenever there can be room."
        )
        if vae_record.meta.get("strips_error"):
            message += f" Its engines for decoding in strips couldn't be built (`{vae_record.meta['strips_error']}`), so larger decodes run on PyTorch."
        if vae_record.encoder is not None:
            message += f" Its encoder's engines encode images whole up to {vae_record.encode_reach()}."
            if vae_record.encoder.get("strips_error"):
                message += f" Its engines for encoding in strips couldn't be built (`{vae_record.encoder['strips_error']}`), so larger encodes run on PyTorch."
        elif vae_record.meta.get("encoder_error"):
            message += f" Its encoder's engines couldn't be built (`{vae_record.meta['encoder_error']}`), so it encodes on PyTorch."
    elif result.get("vae_error") and result["vae_error"] != "cancelled":
        message += f"\n\nThe VAE's engines couldn't be built (`{result['vae_error']}`), so it runs on PyTorch. TensorRT's own output is logged with `--loglevel DEBUG`."
    if result.get("memory_warnings"):
        message += (
            f"\n\nTensorRT skipped {result['memory_warnings']} kernels that need more VRAM than is free at a profile's largest sizes, "
            "so those sizes may run a little slower. Narrower profiles avoid this."
        )
    return message


def build(draft, draft_arch, optimization_level, progress=gr.Progress()):
    from modules.call_queue import queue_lock

    from . import export

    if not _BUILD_LOCK.acquire(blocking=False):
        return "### A build is already running\nWait for it to finish or cancel it.", draft
    BUILDING.set()
    arch = profile_list = None
    try:
        with queue_lock:
            _, patcher, arch, records, prec = _current()
            if arch is None:
                raise RuntimeError("Load a checkpoint TensorRT engines can run first.")
            if arch.id != draft_arch:
                message, draft = "### Nothing built\nThe loaded checkpoint has another architecture now. Check its profiles, then build.", None
            elif not draft:
                raise RuntimeError("Add at least one profile.")
            else:
                mode = store.mode()
                profile_list = [Profile.from_dict(d) for d in draft]
                todo, _ = _plan(draft, records, mode, precision=prec)
                vae_part = _vae_part(draft)
                if export.CANCEL.is_set():
                    result = {"ok": False, "error": "cancelled", "records": []}
                else:
                    _release_engines()
                    result = {"ok": True, "records": []}
                    if todo or vae_part:
                        result = export.build(
                            arch, patcher, [(i, profile_list[i]) for i in todo], int(optimization_level),
                            lambda f, t: progress(f, desc=t), mode, profiles=profile_list, precision=prec,
                        )
                if result["ok"]:
                    store.clear_draft(arch.id)
                    deleted = store.keep_only(
                        arch.id, profile_list, mode, precision=prec, keep_other_precisions=store.keep_precisions(),
                    )
                else:
                    store.save_draft(arch.id, profile_list)
                    deleted = 0
                message = _outcome(arch, len(todo), deleted, result, mode)
    except Exception as e:
        if arch is not None and profile_list:
            with contextlib.suppress(OSError):
                store.save_draft(arch.id, profile_list)
        common.logger.error(f"The build failed: {e}", exc_info=True)
        message = f"### The build failed\n`{e}`"
    finally:
        BUILDING.clear()
        export.CANCEL.clear()
        _BUILD_LOCK.release()
    return message, draft


def _build_started():
    from . import export

    if not BUILDING.is_set() and not _BUILD_LOCK.locked():
        export.CANCEL.clear()
    return (
        gr.update(interactive=False),
        gr.update(visible=True),
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def cancel_build():
    from . import export

    export.CANCEL.set()
    common.logger.info("Cancelling the build.")


def delete_engine(choice, draft, draft_arch):
    from modules.call_queue import queue_lock

    if choice:
        with queue_lock:
            group = _groups().get(choice)
            if group:
                _release_engines()
                dropped = {r.key for r in group}
                base_group = (group[0].meta["arch"]["id"], group[0].meta["gpu"], common.trt_tag(group[0].meta["trt"]))
                if not any(
                    (r.meta["arch"]["id"], r.meta["gpu"], common.trt_tag(r.meta["trt"])) == base_group
                    for r in store.records()
                    if r.key not in dropped
                ):
                    vae_mod.delete_for(group[0].meta["arch"]["id"], group[0].meta["gpu"], group[0].meta["trt"])
                store.delete(group)
                common.logger.info(f"Deleted {_n(len(group), 'engine')}: {choice}.")
    return refresh(draft, draft_arch)


def on_ui_settings():
    from . import runtime, speed

    section, slider = ("tensorrt", "TensorRT"), {"minimum": 0, "maximum": 100, "step": 1}
    shared.opts.add_option(runtime.FULL_START, shared.OptionInfo(
        round(speed.WARMUP * 100), "Speed Hacks: share of steps at the start that run in full (%)", gr.Slider, slider, section=section, category_id="sd",
        infotext=runtime.INFOTEXT[runtime.FULL_START],
    ).info("where composition and colors settle; more keeps images closer to running without Speed Hacks, less runs faster"))
    shared.opts.add_option(runtime.FULL_END, shared.OptionInfo(
        round(speed.TAIL * 100), "Speed Hacks: share of steps at the end that run in full (%)", gr.Slider, slider, section=section, category_id="sd",
        infotext=runtime.INFOTEXT[runtime.FULL_END],
    ).info("where fine detail settles; the last step always runs in full"))
    shared.opts.add_option(runtime.QUALITY, shared.OptionInfo(
        speed.QUALITY, "Speed Hacks: quality", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}, section=section, category_id="sd",
        infotext=runtime.INFOTEXT[runtime.QUALITY],
    ).info("1 runs a full step every third step; 0.9 lets them wait where the image changes very slowly; lower runs faster and softer, down to no full steps between the shares above at 0"))


def add_xyz_axes():
    # reForge's X/Y/Z plot script lists its axes in its module's axis_options. apply_override puts a value in
    # p.override_settings, which processing.process_images applies for that generation and restores after. runtime reads
    # SD Unet from shared.opts at each pass, and process_images_inner calls sd_unet.apply_unet() every generation for
    # other extensions' SD Unet options, so an overridden SD Unet takes effect.
    from modules import scripts, shared_items

    from . import runtime

    xyz = next((s.module for s in scripts.scripts_data if s.script_class.__module__ == "xyz_grid.py"), None)
    if xyz is None:
        return
    labels = {o.label for o in xyz.axis_options}
    for label, kind, key in (
        ("[TensorRT] Speed Hacks: full steps at the start (%)", int, runtime.FULL_START),
        ("[TensorRT] Speed Hacks: full steps at the end (%)", int, runtime.FULL_END),
        ("[TensorRT] Speed Hacks: quality", float, runtime.QUALITY),
    ):
        if label not in labels:
            xyz.axis_options.append(xyz.AxisOption(label, kind, xyz.apply_override(key)))
    if "SD Unet" not in labels:
        def confirm(p, xs):
            known = shared_items.sd_unet_items()
            for x in xs:
                if x not in known:
                    raise RuntimeError(f"Unknown SD Unet: {x}")

        # cost 0.5 (UniPC Order's) makes it an outer loop around free axes like Steps: switching to or from None loads a
        # different UNet
        xyz.axis_options.append(xyz.AxisOption(
            "SD Unet", str, xyz.apply_override("sd_unet"), confirm=confirm, cost=0.5, choices=shared_items.sd_unet_items,
        ))


def on_ui_tabs():
    presets = _preset_values()
    first = [v for f in FIELDS for v in getattr(default_profiles(2816)[0], f)]
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(equal_height=False):
            status = gr.Markdown(elem_id="trt_status")
            load_button = gr.Button("Load selected checkpoint", size="sm", visible=False, elem_id="trt_load_checkpoint")
            refresh_button = ToolButton(value=refresh_symbol, elem_id="trt_refresh")
        draft, draft_arch = gr.State([]), gr.State(None)
        with gr.Column(variant="panel", elem_id="trt_mode"):
            mode_title = gr.Markdown(elem_id="trt_mode_title")
            mode_radio = gr.Radio(list(MODE_LABELS.values()), value=MODE_LABELS[store.PER_PROFILE], show_label=False, elem_id="trt_mode_radio")
            mode_help = gr.Markdown(elem_id="trt_mode_help")
            mode_confirm = gr.Button("Start with this mode", variant="primary", visible=False, elem_id="trt_mode_confirm")
        with gr.Row(equal_height=False) as main:
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
                    precision_dropdown = gr.Dropdown(
                        label="Engine precision",
                        choices=["Auto"],
                        value="Auto",
                        visible=False,
                        info="Auto picks the highest-quality precision supported by your GPU that fits VRAM. Independent of checkpoint quantization.",
                        elem_id="trt_precision",
                    )
                    keep_precisions_cb = gr.Checkbox(
                        label="Keep engines for other precisions",
                        value=store.keep_precisions(),
                        visible=False,
                        info="Retain built engines when building another precision so you can switch without rebuilding. Uncheck to delete other precisions on build.",
                        elem_id="trt_keep_precisions",
                    )
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
        for dynamic in (mode_radio, selector, preset, engine_choice, precision_dropdown, keep_precisions_cb):
            dynamic.do_not_save_to_config = True
        draft_outputs = [draft, table, selector, build_button]
        mode_outputs = [mode_title, mode_radio, mode_help, mode_confirm, main]
        outputs = [
            status, draft_arch, *draft_outputs, *numbers, preset, engines, engine_choice, delete_button, cancel_button,
            *mode_outputs, load_button, precision_dropdown, keep_precisions_cb,
        ]
        # Outputs null to the preset menu last, emptying it, so picking the same preset again still fires .change.
        preset.change(
            None, [preset, *numbers], [*numbers, preset],
            **{JS: f"(name, ...current) => [...(({json.dumps(presets)})[name] || current.slice(0, {len(numbers)})), null]"},
        )
        quiet = {"show_progress": "hidden"}
        # javascript/tensorrt.js clicks this button whenever the tab is opened: the loaded checkpoint may have changed.
        refresh_button.click(refresh, [draft, draft_arch], outputs, **quiet)
        load_button.click(load_checkpoint, [draft, draft_arch], outputs)
        tab.load(refresh, None, outputs, **quiet)
        mode_confirm.click(choose_mode, [mode_radio, draft, draft_arch], outputs, **quiet)
        mode_radio.change(change_mode, [mode_radio, draft, draft_arch], outputs, **quiet)
        precision_dropdown.change(change_precision, [precision_dropdown, draft, draft_arch], outputs, **quiet)
        keep_precisions_cb.change(change_keep_precisions, [keep_precisions_cb, draft, draft_arch], outputs, **quiet)
        selector.change(select_profile, [draft, selector], [*numbers, save_button, remove_button], **quiet)
        add_button.click(add_profile, [draft, draft_arch, *numbers], [*draft_outputs, note], **quiet)
        save_button.click(save_profile, [draft, draft_arch, selector, *numbers], [*draft_outputs, note], **quiet)
        remove_button.click(remove_profile, [draft, draft_arch, selector], [*draft_outputs, note], **quiet)
        build_button.click(
            _build_started, None, [build_button, cancel_button, result, mode_radio, precision_dropdown, keep_precisions_cb], queue=False, **quiet,
        ).then(
            build, [draft, draft_arch, level], [result, draft]
        ).then(refresh, [draft, draft_arch], outputs, **quiet)
        cancel_button.click(cancel_build, None, None, queue=False)
        delete_button.click(
            delete_engine, [engine_choice, draft, draft_arch], outputs, **quiet,
            **{JS: "(choice, ...rest) => [choice && confirm('Delete the engines of ' + choice + '? Building them again takes a while.') ? choice : null, ...rest.slice(0, 2)]"},
        )
    return [(tab, "TensorRT", "tensorrt")]
