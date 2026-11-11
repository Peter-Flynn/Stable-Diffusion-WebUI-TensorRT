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
from . import vae as vae_mod
from .profiles import FIELDS, PRESETS, Profile, default_profiles, presets_for

JS = "js" if int(gr.__version__.split(".")[0]) >= 4 else "_js"
NEW = "New profile"
FIELD_LABELS = {"width": "Width", "height": "Height", "batch": "Batch size", "tokens": "Prompt tokens"}
LEVEL_LABELS = {"min": "min", "opt": "optimal", "max": "max"}
BUILDING = threading.Event()
SETTLED = threading.Event()  # the engine mode was chosen (or engines were there) since the WebUI started
MODE_LABELS = {store.PER_PROFILE: "Per-profile engines", store.MERGED: "Merged engine"}
MERGED_VRAM, MERGED_RAM = 24, 64  # GB a merged engine needs
HELP = """
**Engines** are compiled for a UNet architecture, not a checkpoint: every checkpoint with that architecture (fine-tunes, merges, v-prediction ones) runs on the same engines with its own weights, LoRAs included. Engine files hold no weights.

**Profiles** are the image sizes, batch sizes and prompt lengths the engines run. TensorRT tunes each profile for its optimal values and accepts anything from min to max; narrower ranges build faster and can run a little faster. Anything no profile covers, and ControlNet or other UNet patches, runs on PyTorch.

**Per-profile engines** (the engine mode, above) are one engine per profile. Only one is loaded at a time, taking about as much VRAM as the checkpoint's UNet. Each batch starts on the engine whose profile fits it best, and its later passes (e.g. Hires. fix) stay on that engine while it covers their sizes; a pass it doesn't cover loads the engine of a profile that does. An engine that makes way for another is kept in RAM with its weights while RAM allows, so it comes back for the same checkpoint and LoRAs without refitting them. **A merged engine** holds every profile, so passes in any of their sizes run without loading another engine, but it takes every profile's share of TensorRT's memory at once.

**Building** makes the engines match the profiles: per-profile, it builds the profiles that have no engine yet, one at a time, and deletes the engines of profiles that are gone; merged, it builds the engine again whenever the profiles change.

**SD Unet** (Settings, Stable Diffusion) decides when engines run: `Automatic` uses the engines of the loaded checkpoint's architecture, `Speed Hacks` uses them with Speed Hacks, `None` always uses PyTorch.

**The VAE** gets engines too, for its decoder and its encoder, built with the UNet's for the sizes its profiles cover, in the precision reForge runs the VAE in where the GPU has BF16, and in FP16 where reForge falls back to FP32 for want of it: half the VRAM (a `--vae-in-*` flag at build time decides otherwise). FP16 engines run the VAE scaled down by a power of two between its normalizations, so no VAE overflows in them, and any VAE of the architecture runs on them. Each decode runs on them whole when there can be room for it (reForge's manager makes it, a UNet engine going last), several images at a time where that takes no UNet engine's room. Larger ones (a Hires. fix's) run in strips on engines of their own: the same image, a bit slower. Encodes (img2img's and inpainting's images, a Hires. fix's upscaled one) run on the encoder's engines the same way, whole or in strips. PyTorch does the rest.

**Speed Hacks** run about half of the denoising steps on a small part of the UNet, reusing what the rest of it computed a step or two before (DeepCache), and, late in a pass, wait longer between full steps while the image changes smoothly (AdaptiveDiffusion). The first 30% and last 5% of steps run in full, so composition and detail stay close (both shares are in Settings, Stable Diffusion, TensorRT). Passes of 12 steps or more should run roughly 1.6x faster than on the engines alone (an estimate), with slightly less precise images: a seed gives a somewhat different image than without them. Every engine is built ready for them, which should add about 10-20% to a build.
"""


def _span(values) -> str:
    lo, opt, hi = values
    return f"{opt}" if lo == hi else f"{lo}-{hi} ({opt})"


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _machine():
    """(total VRAM, total RAM) in bytes; either may be None."""
    vram = ram = None
    try:
        import torch

        vram = torch.cuda.get_device_properties(common.mm().get_torch_device()).total_memory
    except Exception:
        pass
    try:
        import psutil

        ram = psutil.virtual_memory().total
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


def _mode_outputs(selected=None):
    """The engine mode panel, and whether the rest of the tab shows: when the WebUI starts with no engines,
    the tab first asks for the mode; while there are engines, it can't change."""
    have = bool(store.records())  # any architecture's, built for any GPU or TensorRT version
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


def _plan(draft, records, mode):
    """(indexes of the draft's profiles to build, how many engines building deletes): per-profile, the
    profiles that have no engine; merged, all of them unless the merged engine holds just these."""
    keys = _keys(draft)
    wanted = store.wanted_keys([Profile.from_dict(d) for d in draft], mode) if draft else set()
    built = {r.engine_key for r in records if not r.outdated and not r.retry_speed_hacks}  # built again: from before Speed Hacks, or retrying them
    if mode == store.MERGED:
        todo = list(range(len(draft))) if draft and not wanted <= built else []
    else:
        todo = [i for i, k in enumerate(keys) if k not in built and k not in keys[:i]]
    return todo, sum(1 for r in records if r.engine_key not in wanted)


def _status(sd_model, arch, records, asking=False) -> str:
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
    outdated = [r for r in records if r.outdated]
    failed = [r for r in records if r.speed_hacks is None and not r.outdated]
    if outdated:
        lines.append(f"**{_n(len(outdated), 'engine')} {'was' if len(outdated) == 1 else 'were'} built before Speed Hacks:** build again to add them.")
    if failed:
        retry = "building again retries once" if any(r.retry_speed_hacks for r in failed) else "to try again, delete them and build again"
        lines.append(f"Speed Hacks couldn't be built for {_n(len(failed), 'engine')} (`{failed[0].meta['speed_hacks_error']}`), so they run without them; {retry}.")
    return "\n\n".join(lines)


def _vae_status(arch) -> str:
    """What the loaded VAE's engines run, for the status ("" if it has none)."""
    try:
        vae = vae_mod.loaded_vae()
        record = vae_mod.record_for(arch.id, vae) if vae is not None and not vae_mod.unsupported(vae) else None
        if record is None:
            return ""
        text = f"**VAE engines:** {vae_mod.label(record.meta['dtype'])}; decodes up to {record.reach()}"
        if record.encoder is not None:
            text += f"; encodes up to {record.encode_reach()}"
        return text + "."
    except Exception as e:
        common.log(f"Could not describe the VAE's engines: {e}")
        return ""


def _vae_part(draft) -> str:
    """What building the draft builds of the VAE's engines (vae.needs_build), "" if nothing."""
    try:
        return vae_mod.needs_build(arch_mod.detect(), [Profile.from_dict(d) for d in draft])
    except Exception as e:
        common.log(f"Could not check the VAE's engines: {e}")
        return ""


def _vae_line(part: str) -> str:
    """The profile table's note on the VAE's engines; `part` is what building builds of them (_vae_part)."""
    if not vae_mod.available():
        return ""
    if part:
        return f"**Building also builds the engines for {part}.**"
    vae = vae_mod.loaded_vae()
    reason = "no VAE is loaded" if vae is None else vae_mod.unsupported(vae)
    if reason:
        return f"*The VAE runs on PyTorch: {reason}.*"
    return "*The VAE's engines are built for these sizes.*"


def _draft_view(draft: list, records, mode, vae_part: str = "") -> str:
    if not draft:
        return "*No profiles. Add at least one to build an engine.*"
    todo, extra = _plan(draft, records, mode)
    exists = {r.engine_key for r in records}
    keys = store.wanted_keys([Profile.from_dict(d) for d in draft], mode) if mode == store.MERGED else None
    rows = ["| # | Width | Height | Batch size | Prompt tokens | Engine |", "|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for i, (d, key) in enumerate(zip(draft, _keys(draft))):
        p = Profile.from_dict(d)
        again = (keys or {key}) <= exists  # built before Speed Hacks, or retrying them
        state = ("**to rebuild** (Speed Hacks)" if again else "**to build**") if i in todo else "built"
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
    rows.append(_vae_line(vae_part))
    return "\n".join(rows)


def _label(i: int, d: dict) -> str:
    p = Profile.from_dict(d)
    return f"#{i + 1}: {_span(p.width)} x {_span(p.height)}"


def _index(choice):
    match = re.match(r"#(\d+):", choice or "")
    return int(match.group(1)) - 1 if match else None


def _draft_outputs(draft, records, select=None):
    """Profile table, profile selector and build button for a draft."""
    mode = store.mode()
    choices = [NEW] + [_label(i, d) for i, d in enumerate(draft)]
    selected = _label(select, draft[select]) if select is not None and select < len(draft) else NEW
    todo, extra = _plan(draft, records, mode)
    vae_part = _vae_part(draft) if draft else ""
    if todo:
        build_label = "Build the merged engine" if mode == store.MERGED else f"Build {_n(len(todo), 'engine')}"
    elif vae_part:
        build_label = "Build the VAE's engines"
    else:
        build_label = "Update engines" if extra else "Engines up to date"
    return (
        _draft_view(draft, records, mode, vae_part),
        gr.update(choices=choices, value=selected),
        gr.update(value=build_label, interactive=bool(draft) and bool(todo or extra or vae_part) and not BUILDING.is_set()),
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
        held = f"{len(store.slots(group))}" + (" (merged)" if any(e.merged for e in group) else "")
        rows.append(f"| {name} | {held} | {common.gib(r.weight_bytes)} | {built} |")
    return "\n".join(rows)


def refresh(draft=None, draft_arch=None, selected=None):
    """Everything on the tab, for the loaded checkpoint. Profiles being edited stay unless the architecture
    changed; `selected` is the engine mode picked but not chosen yet."""
    sd_model, _, arch, records = _current()
    form = [gr.update()] * (len(FIELDS) * 3)
    if arch is not None and (draft is None or draft_arch != arch.id):
        source = records or _stale(arch)
        draft = [p.to_dict() for p in (store.profiles_of(source) if source else default_profiles(arch.adm_channels))]
        form = [v for f in FIELDS for v in getattr(default_profiles(arch.adm_channels)[0], f)]
    elif arch is None:
        draft = []
    engines = list(_groups())
    mode_outputs = _mode_outputs(selected)
    return (
        _status(sd_model, arch, records, asking=not SETTLED.is_set()),
        arch.id if arch else None,
        draft,
        *_draft_outputs(draft, records),
        *form,
        gr.update(choices=_preset_choices(arch), value=None),
        _engines_view(arch),
        gr.update(choices=engines, value=None, visible=bool(engines)),
        gr.update(visible=bool(engines)),
        gr.update(visible=BUILDING.is_set()),
        *mode_outputs,
    )


def choose_mode(label, draft, draft_arch):
    """The first-start prompt's answer."""
    try:
        store.set_mode(_mode_of(label))
        SETTLED.set()
    except Exception as e:  # engines appeared meanwhile: they decide
        common.log(f"Could not set the engine mode: {e}")
    return refresh(draft, draft_arch)


def change_mode(label, draft, draft_arch):
    """The mode picked in the panel: saved right away once the first-start prompt is answered (only while
    there are no engines; the panel hides otherwise), kept as a pick until then. Not during a build: its
    engines will decide the mode."""
    if not SETTLED.is_set():
        return refresh(draft, draft_arch, selected=label)
    if _mode_of(label) != store.mode() and not BUILDING.is_set():
        try:
            store.set_mode(_mode_of(label))
        except Exception as e:
            common.log(f"Could not change the engine mode: {e}")
    return refresh(draft, draft_arch)


def _refresh_unet_list():
    from . import runtime

    runtime.release_all()
    if hasattr(sd_unet.current_unet, "option") and isinstance(sd_unet.current_unet.option, runtime.SpeedHacksOption):
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


def _outcome(arch, todo: int, deleted: int, result, mode) -> str:
    built, merged = len(result["records"]), mode == store.MERGED
    if result["ok"]:
        if merged:
            message = "### Engine up to date\n" + (f"Built the merged engine, {_n(todo, 'profile')}" if built else "The merged engine holds these profiles")
            message += (f", and deleted {_n(deleted, 'engine')} it replaces" if deleted else "") + f". Every {arch.name} checkpoint runs on it with `SD Unet` set to `Automatic` or `Speed Hacks`."
        else:
            first = f"Built {_n(built, 'engine')}" if built else "The UNet engines were up to date"
            message = f"### Engines up to date\n{first}" + (f", deleted {_n(deleted, 'engine')} of profiles no longer listed" if deleted else "")
            message += f". Every {arch.name} checkpoint runs on them with `SD Unet` set to `Automatic` or `Speed Hacks`."
        if any(r.meta.get("stripped") is False for r in result["records"]):
            message += " (TensorRT couldn't leave the weights out of some of them, so their files are as large as the UNet.)"
        if result.get("hack_errors"):
            retry = "Building again retries once." if any(r.retry_speed_hacks for r in result["records"]) else "That failed before too; to try again, delete the engines and build again."
            message += f"\n\nSpeed Hacks couldn't be built (`{result['hack_errors'][-1]}`), so these engines run without them. The console has the details. {retry}"
    else:
        done = "The engines are unchanged." if merged else f"{_n(built, 'engine')} of {todo} built; the other engines are unchanged."
        if result.get("error") == "cancelled":
            message = f"### Build cancelled\n{done}"
        else:
            message = f"### The build failed\n`{result.get('error')}`\n\nThe console has the details. {done}"
            if merged and "memory" in str(result.get("error")).lower():
                message += (
                    f" A merged engine's build needs at least {MERGED_RAM} GB of RAM; per-profile engines build one profile at a "
                    "time (to switch, delete every engine)."
                )
    if result.get("vae") is not None:
        vae_record = result["vae"]
        message += (
            f"\n\nBuilt the VAE's engines, in {vae_mod.label(vae_record.meta['dtype'])}, for decodes up to {vae_record.reach()}: "
            "decodes run on them whole whenever there can be room, else in strips, with what doesn't fit held in RAM."
            if vae_record.strips else
            f"\n\nBuilt the VAE's engines, in {vae_mod.label(vae_record.meta['dtype'])}, for decodes up to {vae_record.reach()}: "
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
        message += f"\n\nThe VAE's engines couldn't be built (`{result['vae_error']}`), so it runs on PyTorch. The console has the details."
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
                mode = store.mode()
                profile_list = [Profile.from_dict(d) for d in draft]
                todo, _ = _plan(draft, records, mode)
                vae_part = _vae_part(draft)
                _refresh_unet_list()
                result = {"ok": True, "records": []}
                if todo or vae_part:
                    result = export.build(
                        arch, patcher, [(i, profile_list[i]) for i in todo], int(optimization_level), lambda f, t: progress(f, desc=t), mode, profiles=profile_list,
                    )
                deleted = store.keep_only(arch.id, profile_list, mode) if result["ok"] else 0
                _refresh_unet_list()
                message = _outcome(arch, len(todo), deleted, result, mode)
    except Exception as e:
        traceback.print_exc()
        message = f"### The build failed\n`{e}`"
    finally:
        BUILDING.clear()
    return message, draft  # the rest of the tab is refreshed next, without progress shown on all of it


def _build_started():
    return gr.update(interactive=False), gr.update(visible=True), "", gr.update(interactive=False)  # the mode too, until its engines exist


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
            vae_mod.delete_for(group[0].meta["arch"]["id"], group[0].meta["gpu"], group[0].meta["trt"])
            sd_unet.list_unets()
        common.log(f"Deleted the {len(group)} engines of {choice}")
    return refresh(draft, draft_arch)


def on_ui_settings():
    """Speed Hacks' full steps, in Settings, Stable Diffusion, TensorRT."""
    from . import runtime, speed

    section, slider = ("tensorrt", "TensorRT"), {"minimum": 0, "maximum": 100, "step": 1}
    shared.opts.add_option(runtime.FULL_START, shared.OptionInfo(
        round(speed.WARMUP * 100), "Speed Hacks: share of steps at the start that run in full (%)", gr.Slider, slider, section=section, category_id="sd",
    ).info("where composition and colors settle; more keeps images closer to running without Speed Hacks, less runs faster"))
    shared.opts.add_option(runtime.FULL_END, shared.OptionInfo(
        round(speed.TAIL * 100), "Speed Hacks: share of steps at the end that run in full (%)", gr.Slider, slider, section=section, category_id="sd",
    ).info("where fine detail settles; the last step always runs in full"))


def on_ui_tabs():
    presets = _preset_values()
    first = [v for f in FIELDS for v in getattr(default_profiles(2816)[0], f)]
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(equal_height=False):
            status = gr.Markdown(elem_id="trt_status")
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
        draft_outputs = [draft, table, selector, build_button]
        mode_outputs = [mode_title, mode_radio, mode_help, mode_confirm, main]
        outputs = [status, draft_arch, *draft_outputs, *numbers, preset, engines, engine_choice, delete_button, cancel_button, *mode_outputs]
        # In the browser; the preset menu empties again so the same preset can be picked twice.
        preset.change(
            None, [preset], [*numbers, preset],
            **{JS: f"(name, ...current) => [...(({json.dumps(presets)})[name] || current.slice(0, {len(numbers)})), null]"},
        )
        quiet = {"show_progress": "hidden"}  # quick updates: no progress overlay on every component they touch
        refresh_button.click(refresh, [draft, draft_arch], outputs, **quiet)  # also clicked when the tab opens (javascript/tensorrt.js)
        tab.load(refresh, None, outputs, **quiet)
        mode_confirm.click(choose_mode, [mode_radio, draft, draft_arch], outputs, **quiet)
        mode_radio.change(change_mode, [mode_radio, draft, draft_arch], outputs, **quiet)
        selector.change(select_profile, [draft, selector], [*numbers, save_button, remove_button], **quiet)
        add_button.click(add_profile, [draft, *numbers], [*draft_outputs, note], **quiet)
        save_button.click(save_profile, [draft, selector, *numbers], [*draft_outputs, note], **quiet)
        remove_button.click(remove_profile, [draft, selector], [*draft_outputs, note], **quiet)
        # The build's progress shows once, in its result.
        build_button.click(_build_started, None, [build_button, cancel_button, result, mode_radio], queue=False, **quiet).then(
            build, [draft, draft_arch, level], [result, draft]
        ).then(refresh, [draft, draft_arch], outputs, **quiet)
        cancel_button.click(cancel_build, None, None, queue=False)
        delete_button.click(
            delete_engine, [engine_choice, draft, draft_arch], outputs, **quiet,
            **{JS: "(choice, ...rest) => [choice && confirm('Delete the engines of ' + choice + '? Building them again takes a while.') ? choice : null, ...rest.slice(0, 2)]"},
        )
    return [(tab, "TensorRT", "tensorrt")]
