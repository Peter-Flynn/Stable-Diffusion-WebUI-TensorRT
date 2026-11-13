"""Building the VAE's engines (vae.py): its decoder's and its encoder's two parts, traced from fresh modules
of the loaded VAE's layout to weightless ONNX graphs, and the stages of decoding and encoding in strips
(strips.py), each built in export.py's builder process. What the engines on disk already cover is kept."""

import contextlib
import gc
import glob
import hashlib
import json
import os
import time

import torch

from . import common, store, strips
from .common import logger
from .vae import (
    ENCODE_POST, ENCODE_PRE, ENCODE_STRIP, ENCODER, POST, PRE, RETRY_SHARE, Record, config, digest, dtype_name, encoder_config,
    encoder_excess, encoder_shapes, encoder_wanted, engine_dtype, excess, folder, gpu_vram, is_decoder_weight, is_encoder_weight,
    loaded_vae, parts_missing, path_prefix, records_matching, remove_record, shapes_for, strip_dtype, strips_wanted, unsupported,
    vram_budget, weight_bytes, why_smaller,
)


class _Pre(torch.nn.Module):
    """The decoder up to its mid-block attention: (that attention's input, its queries, keys, values)."""

    def __init__(self, fsm):
        super().__init__()
        self.post_quant_conv = getattr(fsm, "post_quant_conv", None)
        self.decoder = fsm.decoder

    def forward(self, latent):
        z = self.post_quant_conv(latent) if self.post_quant_conv is not None else latent
        d = self.decoder
        x = d.mid.block_1(d.conv_in(z), None)
        attn = d.mid.attn_1
        h = attn.norm(x)
        return x, attn.q(h), attn.k(h), attn.v(h)


class _Post(torch.nn.Module):
    """The rest of the decoder, from the mid-block attention's input and output: the image."""

    def __init__(self, fsm):
        super().__init__()
        self.decoder = fsm.decoder

    def forward(self, x, attention):
        d = self.decoder
        h = d.mid.block_2(x + d.mid.attn_1.proj_out(attention), None)
        for i_level in reversed(range(d.num_resolutions)):
            for i_block in range(d.num_res_blocks + 1):
                h = d.up[i_level].block[i_block](h, None)
            if i_level != 0:
                h = d.up[i_level].upsample(h)
        h = d.norm_out(h)
        return d.conv_out(h * torch.sigmoid(h))  # reForge's nonlinearity


class _EncodePre(torch.nn.Module):
    """The encoder up to its mid-block attention: (that attention's input, its queries, keys, values)."""

    def __init__(self, fsm):
        super().__init__()
        self.encoder = fsm.encoder

    def forward(self, image):
        e = self.encoder
        h = e.conv_in(image)
        for i_level in range(e.num_resolutions):
            for i_block in range(e.num_res_blocks):
                h = e.down[i_level].block[i_block](h, None)
            if i_level != e.num_resolutions - 1:
                h = e.down[i_level].downsample(h)
        x = e.mid.block_1(h, None)
        attn = e.mid.attn_1
        h = attn.norm(x)
        return x, attn.q(h), attn.k(h), attn.v(h)


class _EncodePost(torch.nn.Module):
    """The rest of the encoder, from the mid-block attention's input and output, and quant_conv where the
    first stage model applies it: the latent's distribution ("moments"), as it is."""

    def __init__(self, fsm):
        super().__init__()
        self.encoder = fsm.encoder
        self.quant_conv = getattr(fsm, "quant_conv", None)

    def forward(self, x, attention):
        e = self.encoder
        h = e.mid.block_2(x + e.mid.attn_1.proj_out(attention), None)
        h = e.norm_out(h)
        h = e.conv_out(h * torch.sigmoid(h))  # reForge's nonlinearity
        return self.quant_conv(h) if self.quant_conv is not None else h


def fresh(cfg: dict, device, dtype):
    """A decoder of this layout from reForge's own classes, with none of a loaded one's offloading hooks."""
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.model import Decoder

    module = torch.nn.Module()
    module.decoder = Decoder(
        ch=cfg["ch"], out_ch=cfg["out_ch"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=3, resolution=256, z_channels=cfg["z_channels"], dropout=0.0,
    )
    if cfg["post_quant_conv"]:
        module.post_quant_conv = ops.disable_weight_init.Conv2d(cfg["latent_channels"], cfg["z_channels"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def fresh_encoder(cfg: dict, device, dtype):
    """An encoder of this layout from reForge's own classes, likewise (with quant_conv where it's applied)."""
    import ldm_patched.modules.ops as ops
    from ldm_patched.ldm.modules.diffusionmodules.model import Encoder

    module = torch.nn.Module()
    module.encoder = Encoder(
        ch=cfg["ch"], out_ch=cfg["in_channels"], ch_mult=cfg["ch_mult"], num_res_blocks=cfg["num_res_blocks"], attn_resolutions=[],
        in_channels=cfg["in_channels"], resolution=256, z_channels=cfg["out_channels"], double_z=False, dropout=0.0,
    )
    if cfg["quant_conv"]:
        module.quant_conv = ops.disable_weight_init.Conv2d(cfg["out_channels"], cfg["moments"], 1)
    for name, m in module.named_modules():
        if name in cfg["padding"]:
            m.padding_mode = cfg["padding"][name]
    return module.to(device=device, dtype=dtype).eval()


def _randomize(module):
    """Random weights: tracing doesn't depend on them, and distinct values keep the exporter from merging
    identical ones (GroupNorm's)."""
    for t in module.state_dict(keep_vars=True).values():
        if t.is_floating_point():
            t.normal_(0, 0.02)
    return module


def _export(module, halves: dict, work: str, first, loaded: dict) -> dict:
    """Traces a fresh module's two halves ({part: (the half's class, its inputs' names, its outputs', their
    dynamic axes)}, the first one fed `first`, the second its first output and one like it) to
    work/vae_<part>.onnx: {part: {weight name: shape}}."""
    from . import export

    _randomize(module)
    (one, two), parts = list(halves), {part: spec[0](module) for part, spec in halves.items()}
    x = parts[one](first)[0]
    args = {one: (first,), two: (x, torch.randn_like(x))}
    exported = {k: list(v.shape) for k, v in module.state_dict(keep_vars=True).items()}
    result = {}
    for part, (_, inputs, outputs, axes) in halves.items():
        path = os.path.join(work, f"vae_{part}.onnx")
        with export.quiet_warnings():
            torch.onnx.export(parts[part], args[part], path, input_names=inputs, output_names=outputs, dynamic_axes=axes, **export.ONNX_KWARGS)
        result[part] = export._weightless_graph(path, inputs, (exported, loaded))
    return result


_AXES = {0: "batch", 2: "height", 3: "width"}
_LARGER = {0: "batch", 2: "image_height", 3: "image_width"}


@common.inference_mode
def export_onnx(vae, work: str, dtype, device) -> dict:
    """Writes work/vae_pre.onnx and work/vae_post.onnx; returns {part: {weight name: shape}, "config"}."""
    cfg = config(vae)
    halves = {
        PRE: (_Pre, ["latent"], ["x", "q", "k", "v"], {n: _AXES for n in ("latent", "x", "q", "k", "v")}),
        POST: (_Post, ["x", "attention"], ["image"], {"x": _AXES, "attention": _AXES, "image": _LARGER}),
    }
    latent = torch.randn(1, cfg["latent_channels"], 32, 32, dtype=dtype, device=device)
    loaded = {k: list(v.shape) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_decoder_weight(k)}
    return {"config": cfg, **_export(fresh(cfg, device, dtype), halves, work, latent, loaded)}


@common.inference_mode
def export_encoder_onnx(vae, work: str, dtype, device) -> dict:
    """Writes work/vae_encode-pre.onnx and work/vae_encode-post.onnx; returns {part: {weight name: shape},
    "config"}."""
    cfg = encoder_config(vae)
    halves = {
        ENCODE_PRE: (_EncodePre, ["image"], ["x", "q", "k", "v"], {"image": _LARGER, **{n: _AXES for n in ("x", "q", "k", "v")}}),
        ENCODE_POST: (_EncodePost, ["x", "attention"], ["moments"], {n: _AXES for n in ("x", "attention", "moments")}),
    }
    image = torch.randn(1, cfg["in_channels"], 128, 128, dtype=dtype, device=device)
    loaded = {k: list(v.shape) for k, v in vae.first_stage_model.state_dict(keep_vars=True).items() if is_encoder_weight(k)}
    return {"config": cfg, **_export(fresh_encoder(cfg, device, dtype), halves, work, image, loaded)}


@common.inference_mode
def export_strips(vae, work: str, dtype, device, names, encoding: bool = False) -> dict:
    """Writes the ONNX graphs of the kinds named, of the decode in strips (work/strip-<kind>.onnx) or the
    encode's (work/encode-strip-<kind>.onnx); returns {kind name (prefixed so, for the encode's): {weight name:
    shape}}. Each from a stage of that kind in a fresh decoder or encoder of the VAE's layout."""
    from . import export

    if encoding:
        cfg = encoder_config(vae)
        module, every_way, prefix, key = fresh_encoder(cfg, device, dtype), strips.encoder_every_way, ENCODE_STRIP, ENCODE_STRIP
    else:
        cfg = config(vae)
        module, every_way, prefix, key = fresh(cfg, device, dtype), strips.every_way, "strip-", ""
    _randomize(module)
    found = [s for each in every_way(module, cfg["padding"]) for s in each]
    loaded = vae.first_stage_model.state_dict(keep_vars=True)
    result = {}
    for name in names:
        stage = next(s for s in found if s.kind.name == name)
        kind, part = stage.kind, strips.StageModule(stage, module)
        path = os.path.join(work, f"{prefix}{name}.onnx")
        with export.quiet_warnings():
            torch.onnx.export(
                part, strips.example_inputs(kind, dtype, device), path, input_names=list(kind.inputs()),
                output_names=list(kind.outputs()), dynamic_axes=strips.dynamic_axes(kind), **export.ONNX_KWARGS,
            )
        own = {k: list(v.shape) for k, v in part.state_dict(keep_vars=True).items()}
        theirs = {k: list(loaded[stage.weights[k]].shape) for k in own}
        result[key + name] = export._weightless_graph(path, kind.inputs(), (own, theirs), quiet=True)
    del module, found
    return result


def build(arch, profiles: list, work: str, optimization_level=None, progress=None, vae=None):
    """Builds the VAE engines these profiles' sizes lack, for the loaded VAE and its precisions
    (engine_dtype, strip_dtype): the decoder's pre and post engines, the encoder's two, and those that
    decode and encode in strips (strips.py), keeping what this GPU's and TensorRT version's engines for the
    architecture have that covers them, and deleting the rest. Returns (the Record or None, why it failed
    or None, how many kernels TensorRT skipped for want of VRAM); a failure is logged here. If only the
    encoder's engines or the stripped decode's fail, the record has the others (and says why in
    "encoder_error" or "strips_error")."""
    from . import export

    vae = vae if vae is not None else loaded_vae()
    reason = "no VAE is loaded" if vae is None else unsupported(vae)
    if reason:
        return None, reason, 0
    skipped = 0
    try:
        dtype, sdtype = engine_dtype(vae), strip_dtype(vae)
        cfg, vram, device = config(vae), gpu_vram(), common.mm().get_torch_device()
        asked = shapes_for(profiles, cfg, dtype, vram)
        strip_wanted = strips_wanted(vae, profiles, cfg, sdtype)
        enc_wanted = encoder_wanted(vae, profiles, dtype, sdtype, vram)
        enc_kinds = [ENCODE_STRIP + k for k in enc_wanted["strips"]] if enc_wanted else []
        everything = [PRE, POST] + (list(ENCODER) if enc_wanted else []) + enc_kinds + list(strip_wanted)
        old = next(iter(records_matching(arch.id, vae, dtype, sdtype)), None)
        todo = parts_missing(old, asked, strip_wanted, enc_wanted) if old is not None else list(everything)
        if old is not None and old.meta.get("strips_error"):  # building anyway: try those again
            todo += [k for k in strip_wanted if k not in old.strips and k not in todo]
        if old is not None and old.meta.get("encoder_error") and enc_wanted and ENCODE_PRE not in todo:
            todo += list(ENCODER) + [k for k in enc_kinds if k not in todo]
        if old is not None and enc_wanted and (old.encoder or {}).get("strips_error"):  # likewise
            todo += [k for k in enc_kinds if k[len(ENCODE_STRIP):] not in old.encoder_strips and k not in todo]
        enc_kinds_todo = [k[len(ENCODE_STRIP):] for k in todo if k.startswith(ENCODE_STRIP)]
        kinds_todo = [k for k in todo if k not in (PRE, POST) + ENCODER and not k.startswith(ENCODE_STRIP)]
        shapes = {PRE: asked[PRE], POST: asked[POST]}
        post_asked = asked[POST]
        if old is not None:
            if PRE not in todo:
                shapes[PRE] = old.meta["shapes"][PRE]
            if POST not in todo:
                shapes[POST], post_asked = old.meta["shapes"][POST], old.meta["asked"][POST]
        if POST in todo:
            wanted_h, wanted_w = max(p.height[2] for p in profiles) // 8, max(p.width[2] for p in profiles) // 8
            b, _, h, w = asked[POST]["x"][2]
            if (h, w) != (wanted_h, wanted_w):
                cause = excess(cfg, 1, wanted_h, wanted_w, vram_budget(dtype, vram))[1]
                further = "in strips" if strip_wanted else "on PyTorch"
                logger.info(
                    f"The VAE decoder's engines take whole images up to {w * 8}x{h * 8} ({b} at a time) rather than "
                    f"{wanted_w * 8}x{wanted_h * 8}: {why_smaller(cause, dtype, vram)}. Larger decodes run {further}."
                )
        if ENCODE_PRE in todo:
            wanted_h, wanted_w = max(p.height[2] for p in profiles) // 8 * 8, max(p.width[2] for p in profiles) // 8 * 8
            b, _, h, w = enc_wanted["shapes"][ENCODE_PRE]["image"][2]
            if (h, w) != (wanted_h, wanted_w):
                cause = encoder_excess(enc_wanted["config"], 1, wanted_h, wanted_w, vram_budget(dtype, vram))[1]
                further = "in strips" if enc_wanted["strips"] else "on PyTorch"
                logger.info(
                    f"The VAE encoder's engines take whole images up to {w}x{h} ({b} at a time) rather than "
                    f"{wanted_w}x{wanted_h}: {why_smaller(cause, dtype, vram, 'encode')}. Larger encodes run {further}."
                )
        if old is not None and sorted(todo) != sorted(everything):
            logger.debug(f"{len(everything) - len(todo)} of the {len(everything)} VAE engines already cover these sizes; building the rest.")
        if progress:
            progress(0.0, "exporting the graphs")
        graphs = {}
        if PRE in todo or POST in todo:
            graphs.update(export_onnx(vae, work, dtype, device))
        if ENCODE_PRE in todo:
            graphs.update({part: names for part, names in export_encoder_onnx(vae, work, dtype, device).items() if part in ENCODER})
        if enc_kinds_todo:
            graphs.update(export_strips(vae, work, sdtype, device, enc_kinds_todo, encoding=True))
        if kinds_todo:
            graphs.update(export_strips(vae, work, sdtype, device, kinds_todo))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the trace's VRAM back to the driver, for the builder's process
        built, done, tries = {}, 0, 0
        total = len(todo)  # engines to build

        def job(part, plan_shapes):
            onnx_name = f"strip-{part}" if part in kinds_todo else part if part.startswith(ENCODE_STRIP) else f"vae_{part}"
            return {
                "onnx": os.path.join(work, f"{onnx_name}.onnx"),
                "plan": os.path.join(work, f"{onnx_name}.plan"),
                "weights": sorted(graphs[part]),
                "profiles": [plan_shapes],
                "timing_cache": store.timing_cache_path(),
                "optimization_level": optimization_level,
            }

        def step(what):
            n = done
            return (lambda f, t: progress((n + f) / max(1, total), f"{what}: {t}")) if progress else None

        def build_kinds(kinds: dict, prefix: str, what: str) -> tuple:
            """Builds the engines of these kinds ({name: shapes}) for `what` in strips, their graphs and plans
            named `prefix` and the kind: ({name: its meta} of those built, why one failed or None, stopping
            there). Raises _Cancelled."""
            nonlocal done, total, skipped
            metas = {}
            for i, name in enumerate(kinds):
                if export.CANCEL.is_set():
                    raise _Cancelled()
                result = export._build_plan(job(prefix + name, kinds[name]), work, step(f"engine {i + 1} of {len(kinds)} for {what} in strips"))
                skipped += result.get("memory_warnings") or 0
                if not result["ok"]:
                    if export.CANCEL.is_set() or str(result.get("error")) == "cancelled":
                        raise _Cancelled()
                    total -= len(kinds) - i
                    larger = "decodes" if what == "decoding" else "encodes"
                    logger.warning(f"Couldn't build the VAE engines for {what} in strips, so larger {larger} run on PyTorch: {result.get('error')}")
                    return metas, str(result.get("error"))
                built[prefix + name] = result
                metas[name] = {"shapes": kinds[name], "weights": sorted(graphs[prefix + name]), "stripped": result["stripped"]}
                done += 1
            return metas, None

        parts = [p for p in (POST, PRE) if p in todo]  # the post engine first: TensorRT may find no kernels for it
        while parts:
            part = parts[0]
            result = export._build_plan(job(part, shapes[part]), work, step(f"{part} engine"))
            warned = result.get("memory_warnings") or 0
            skipped += warned
            if not result["ok"]:
                smaller = shapes_for(profiles, cfg, dtype, vram, RETRY_SHARE) if dtype is not None and vram else asked
                if part != POST or tries or not warned or smaller[POST] == shapes[POST]:
                    logger.warning(f"Couldn't build the VAE engines, so the VAE runs on PyTorch: {result.get('error')}")
                    return None, str(result.get("error")), skipped
                tries, shapes[POST] = 1, smaller[POST]
                b, _, h, w = shapes[POST]["x"][2]
                logger.info(f"TensorRT found no kernels for the VAE decoder that fit in the VRAM free; trying again up to {w * 8}x{h * 8} ({b} at a time).")
                continue
            built[part] = result
            parts.pop(0)
            done += 1
        # The encoder's: kept if they cover these sizes, else built, the pre one again smaller like the post
        enc_meta = old.encoder if old is not None and enc_wanted and ENCODE_PRE not in todo else None
        enc_error = None
        if ENCODE_PRE in todo:
            enc_shapes, tries = enc_wanted["shapes"], 0
            parts = list(ENCODER)
            while parts and not enc_error:
                if export.CANCEL.is_set():
                    return None, "cancelled", skipped
                part = parts[0]
                result = export._build_plan(job(part, enc_shapes[part]), work, step(f"{part} engine"))
                warned = result.get("memory_warnings") or 0
                skipped += warned
                if result["ok"]:
                    built[part] = result
                    parts.pop(0)
                    done += 1
                    continue
                smaller = encoder_shapes(profiles, enc_wanted["config"], dtype, vram, RETRY_SHARE) if dtype is not None and vram else enc_shapes
                if export.CANCEL.is_set() or str(result.get("error")) == "cancelled":
                    return None, "cancelled", skipped
                if part != ENCODE_PRE or tries or not warned or smaller == enc_shapes:
                    enc_error = str(result.get("error"))
                    break
                tries, enc_shapes = 1, smaller
                b, _, h, w = enc_shapes[ENCODE_PRE]["image"][2]
                logger.info(f"TensorRT found no kernels for the VAE encoder that fit in the VRAM free; trying again up to {w}x{h} ({b} at a time).")
            if enc_error:
                done = len([p for p in todo if p in (PRE, POST) + ENCODER])
                total -= len(enc_kinds_todo)
                logger.warning(f"Couldn't build the VAE encoder's engines, so it encodes on PyTorch: {enc_error}")
            else:
                enc_meta = {
                    "digest": enc_wanted["digest"],
                    "config": enc_wanted["config"],
                    "shapes": enc_shapes,
                    "asked": enc_wanted["shapes"],  # what they were built for: the shapes, unless TensorRT had to go smaller
                    "weights": {part: sorted(graphs[part]) for part in ENCODER},
                    "weight_bytes": weight_bytes(vae, sorted(set(graphs[ENCODE_PRE]) | set(graphs[ENCODE_POST])), dtype),
                    "stripped": {part: built[part]["stripped"] for part in ENCODER},
                    "strips": dict(old.encoder_strips) if old is not None and (old.encoder or {}).get("digest") == enc_wanted["digest"] else {},
                }
        if enc_meta is not None:  # the encode in strips' kinds: those kept that are wanted, and those it lacks
            kinds = enc_wanted["strips"]
            enc_meta = dict(enc_meta, strips={k: v for k, v in (enc_meta.get("strips") or {}).items() if k in kinds and k not in enc_kinds_todo})
            if enc_kinds_todo:
                enc_meta.pop("strips_error", None)
                enc_meta.pop("strips_asked", None)
                got, error = build_kinds({k: kinds[k] for k in enc_kinds_todo}, ENCODE_STRIP, "encoding")
                if error:  # encodes run whole only: one in strips needs every kind
                    enc_meta.update(strips={}, strips_error=error, strips_asked=kinds)
                else:
                    enc_meta["strips"].update(got)
        strip_meta = {k: v for k, v in old.strips.items() if k in strip_wanted and k not in todo} if old is not None else {}
        built_kinds, strips_error = build_kinds({name: strip_wanted[name] for name in kinds_todo}, "", "decoding")
        strip_meta.update(built_kinds)
        if strips_error:
            strip_meta = {}  # a stripped decode needs every kind
        identity = [digest(vae), dtype_name(dtype), dtype_name(sdtype), shapes, sorted((k, v["shapes"]) for k, v in strip_meta.items())]
        if enc_meta is not None:
            identity.append([enc_meta["digest"], enc_meta["shapes"], sorted((k, v["shapes"]) for k, v in (enc_meta.get("strips") or {}).items())])
        key = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
        weights = {part: sorted(graphs[part]) if part in built else old.meta["weights"][part] for part in (PRE, POST)}
        meta = {
            "key": key,
            "arch_id": arch.id,
            "digest": digest(vae),
            "dtype": dtype_name(dtype),  # the whole-image engines'
            "strips_dtype": dtype_name(sdtype),  # the engines' for strips (scaled, if FP16)
            "config": cfg,
            "shapes": shapes,
            "asked": {PRE: shapes[PRE], POST: post_asked},  # what they were built for: the shapes, unless TensorRT had to go smaller
            "weights": weights,
            "weight_bytes": weight_bytes(vae, weights[PRE] + weights[POST], dtype),
            "stripped": {part: built[part]["stripped"] if part in built else old.meta["stripped"][part] for part in (PRE, POST)},
            "strips": strip_meta,
            "encoder": enc_meta,
            "trt": common.trt_version(),
            "gpu": common.gpu_tag(),
            "gpu_name": torch.cuda.get_device_name(),
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        if strips_error:
            meta["strips_error"], meta["strips_asked"] = strips_error, strip_wanted
        if enc_error:
            meta["encoder_error"], meta["encoder_asked"] = enc_error, {"digest": enc_wanted["digest"], "shapes": enc_wanted["shapes"]}
        os.makedirs(folder(arch.id), exist_ok=True)
        base = f"{path_prefix(arch.id)}-{key}"
        new = Record(base, meta)
        for part in (PRE, POST) + (ENCODER if enc_meta is not None else ()):
            source = os.path.join(work, f"vae_{part}.plan") if part in built else old.plan(part)
            if source != new.plan(part):
                os.replace(source, new.plan(part))
        for name in strip_meta:
            source = os.path.join(work, f"strip-{name}.plan") if name in built else old.strip_plan(name)
            if source != new.strip_plan(name):
                os.replace(source, new.strip_plan(name))
        for name in (enc_meta or {}).get("strips") or {}:
            source = os.path.join(work, f"{ENCODE_STRIP}{name}.plan") if ENCODE_STRIP + name in built else old.encoder_strip_plan(name)
            if source != new.encoder_strip_plan(name):
                os.replace(source, new.encoder_strip_plan(name))
        for path in glob.glob(glob.escape(path_prefix(arch.id)) + "-*.json"):  # what these replace
            if path[: -len(".json")] != base:
                remove_record(path[: -len(".json")])
        for path in glob.glob(glob.escape(base) + ".strip-*.plan"):  # this record's own, from before, it no longer has
            if path[len(base) + len(".strip-"): -len(".plan")] not in strip_meta:
                os.remove(path)
        for path in glob.glob(glob.escape(base) + f".{ENCODE_STRIP}*.plan"):
            if path[len(base) + len(ENCODE_STRIP) + 1: -len(".plan")] not in ((enc_meta or {}).get("strips") or {}):
                os.remove(path)
        if enc_meta is None:
            for part in ENCODER:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(new.plan(part))
        with open(base + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(base + ".json.tmp", base + ".json")
        return new, None, skipped
    except _Cancelled:
        return None, "cancelled", skipped
    except Exception as e:
        logger.error(f"Couldn't build the VAE engines, so the VAE runs on PyTorch: {e}", exc_info=True)
        return None, str(e), skipped


class _Cancelled(Exception):
    """The build was cancelled."""
