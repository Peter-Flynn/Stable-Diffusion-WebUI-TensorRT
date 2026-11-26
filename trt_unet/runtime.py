# Runs reForge's UNet on TensorRT engines. A hook on UNetModel.forward sends each UNet call of a sampling pass (a Pass)
# to the engine whose profile fits its shapes; anything an engine can't run falls back to PyTorch for the rest of the
# pass. Each engine is a model in reForge's model manager (TrtModel). Multiple engines stay resident in VRAM when there
# is room for them alongside the pass's working memory, evicting least-recently-used engines first when VRAM is needed;
# the manager evicts engines only as a last resort, because loading one means refitting every weight (or restoring it
# from RAM, refits.py). Before anything loads, _fits checks the engine and the pass's working memory (working.py) can fit
# in VRAM at all; if not, the pass runs on PyTorch. During a pass the PyTorch UNet stays patched but unloaded, in RAM.
import collections
import contextlib
import inspect
import math
import os
import sys
import threading
import time
import weakref

import torch

from modules import sd_unet

from . import arch as arch_mod
from . import common, compat, fresh, gpu, profiles, refits, speed, store, strips, working
from . import vae_run
from .common import inference_mode, logger, notify
from .weights import EffectiveWeights

_models = {}
_last_used = {}
_use_order = {}  # record.key -> monotonic use tick for LRU eviction among resident TrtModels
_use_tick = 0
_pass = None
_unet_arenas = {}  # (device.type, device.index) -> shared Arena across co-resident UNet TrtModels
_BackupWeight = collections.namedtuple("Dimension", ["weight", "inplace_update"])


def _unet_arena(dev):
    from .engine import Arena

    d = torch.device(dev)
    key = (d.type, d.index)
    arena = _unet_arenas.get(key)
    if arena is None:
        arena = _unet_arenas[key] = Arena()
    return arena


def _release_unet_arena(dev=None):
    if dev is None:
        for arena in list(_unet_arenas.values()):
            arena.release()
        return
    d = torch.device(dev)
    arena = _unet_arenas.get((d.type, d.index))
    if arena is not None:
        arena.release()


class Fallback(Exception):
    pass


class _CallTorch(Exception):
    # Raised by Pass.run to run a single mid-pass call on PyTorch while keeping the current TRT engine resident in VRAM.
    pass


class TensorRTUNet:
    gguf_baked = True

    def parameters(self):
        return iter(())


class _Shape:
    __slots__ = ("shape",)

    def __init__(self, shape: tuple):
        self.shape = shape


def _took_key(model, x, unet_batch: int = None) -> tuple:
    found = _call_shapes(model, x, unet_batch=unet_batch)
    batch = found[1]["sample"][0] if found is not None else (unet_batch if unet_batch is not None else 2 * x.shape[0])
    return (batch, x.shape[2], x.shape[3])


def call_bytes(model, patcher, x, unet_batch: int = None) -> int:
    batch, h, w = _took_key(model, x, unet_batch=unet_batch)
    return working.call_bytes(model.record.arch, patcher, batch, w * 8, h * 8)


class DoesNotFit(RuntimeError):
    pass


class TrtModel:
    # Implements the parts of ModelPatcher that reForge/Neo's LoadedModel calls, so the model manager can hold an engine
    # (and its Speed Hacks partner) like any model. Its size is what gpu.py measured once loaded, an estimate before.
    parent = None
    evicted_last = True

    def __init__(self, record: store.Record):
        self.record = record
        self.owner = f"{record.arch.label} {record.label} ({record.engine_key})"  # key into gpu.py's per-owner accounting: must be unique
        self.name = f"{record.arch.name} {record.label}"
        self.hacks = record.speed_hacks
        self.model = TensorRTUNet()
        self.load_device = common.device()
        self.offload_device = torch.device("cpu")
        self.engine = self.shallow = None
        self.want_shallow = False
        self.shallow_failed = False
        self.weights = None
        self.applied = {}
        self._plan_bytes = {}
        self.took = {}  # latent (batch, h, w) -> VRAM TensorRT actually needed to run it, recorded when it didn't fit
        self.keep = True  # whether unloading serializes the engine to RAM first (refits.py)

    def _engines(self):
        if self.engine is None:
            from .engine import Engine

            arena = _unet_arena(self.load_device)
            self.engine = Engine(self.record.plan, arena, self.owner)
            if self.hacks:
                self.shallow = Engine(self.record.shallow_plan, arena, self.owner)
        return self.engine

    @property
    def opened(self) -> bool:
        return self.engine is not None and self.engine.loaded

    @property
    def loaded(self) -> bool:
        return self.opened and bool(self.applied.get("unet"))

    @property
    def shallow_loaded(self) -> bool:
        return self.shallow is not None and self.shallow.loaded

    def full(self, shallow: bool, kept=None) -> int:
        total = 0
        for role in ("unet", "shallow") if shallow else ("unet",):
            blob = kept.blobs.get(role) if kept is not None else None
            total += blob.nbytes if blob is not None else self._serialized_size(role)
        return total

    def model_size(self):
        if self.loaded:
            return gpu.held(self.owner)
        return self.full(bool(self.hacks and self.want_shallow))

    def loaded_size(self):
        return self.model_size() if self.loaded else 0

    def current_loaded_device(self):
        return self.load_device if self.loaded else self.offload_device

    @property
    def current_device(self):
        return self.current_loaded_device()

    def model_dtype(self):
        return torch.float16

    def model_patches_to(self, *args):
        pass

    def model_patches_models(self):
        return []

    def lowvram_patch_counter(self):
        return 0

    def is_clone(self, other):
        return other is self

    def partially_load(self, device, extra_memory=0, force_patch_weights=False):
        from . import engine as engine_mod

        if self.loaded:
            return 0
        temp_weights = self.weights is None
        if temp_weights:  # loaded by reForge outside a sampling pass: use the loaded checkpoint's weights
            patcher = arch_mod.unet_patcher()
            if patcher is None:
                raise RuntimeError("no checkpoint to take the engine's weights from")
            self.weights = EffectiveWeights(patcher)
        weights, want_shallow = self.weights, self.want_shallow
        try:
            shallow = bool(self.hacks and want_shallow)
            start, need = time.time(), 0
            if self.opened:  # in VRAM without weights, after a failed refit
                self._discard()
            for attempt in (0, 1):
                self.weights, self.want_shallow = weights, want_shallow
                load, failures = {}, gpu.failures()
                try:
                    kept = self._kept()
                    need = self.full(shallow, kept)
                    if not attempt:
                        common.make_room(common.padded(need), self.load_device)
                    else:
                        with _evictable():
                            common.make_room(common.obtainable(self.load_device) - common.reserve(), self.load_device)
                    with gpu.loading(self.owner) as load:
                        engine = self._engines()
                        self.applied = {}
                        plan_seconds = self._load("unet", engine, kept)
                    with gpu.loading(self.owner) as rest:
                        refit, more, kept = self._load_engines(kept, retry_oom=not attempt)
                    load["failures"] = load.get("failures", 0) + rest.get("failures", 0)
                    plan_seconds += more
                    break
                except Exception as e:
                    past = self.opened
                    if not attempt:
                        e.__traceback__ = None  # its frames reference what TensorRT allocated for the failed load
                    self._discard()
                    if attempt:
                        raise
                    ran_out = isinstance(e, torch.cuda.OutOfMemoryError) or bool(load.get("failures")) or gpu.failures() != failures or engine_mod.LOGGER.out_of_memory
                    if past and not ran_out and engine_mod.own_allocator(failures, self.owner, e):
                        continue
                    if gpu.attached() and not ran_out:  # didn't run out: more room won't help
                        raise
                    logger.info(f"Loading the {self.name} ran out of VRAM with {common.gib(need)} freed ({e}); freeing all there is and trying again.")
            refits.used(self.record.key)
            hacks = " with Speed Hacks" if self.shallow_loaded else ""
            how = ["from RAM" if kept else f"plan {plan_seconds:.1f}s"] + ([_refit_times(refit)] if refit else [])
            logger.info(f"Loaded the {self.name}{hacks} in {time.time() - start:.1f}s ({', '.join(how)}); it holds {common.gib(gpu.held(self.owner))} of VRAM.")
            common.memory(f"after loading the {self.name}")
            return self.model_size()
        finally:
            if temp_weights or not self.loaded:
                self.weights = None
            if not self.loaded:
                self.want_shallow = False

    def _load_engines(self, kept, retry_oom: bool = False) -> tuple:
        from . import engine as engine_mod

        plan_seconds = 0.0
        kept = kept if "unet" in self.applied else None
        if self.want_shallow and self.shallow is not None:
            engine_mod.LOGGER.clear()
            failures = gpu.failures()
            try:
                plan_seconds += self._load("shallow", self.shallow, kept)
            except Exception as e:
                ran_out = isinstance(e, torch.cuda.OutOfMemoryError) or gpu.failures() != failures or engine_mod.LOGGER.out_of_memory
                if retry_oom and ran_out:
                    raise
                notify(f"Couldn't load the Speed Hacks part of the {self.name}, so it runs without Speed Hacks: {e}")
                self.shallow.unload()
                self.applied.pop("shallow", None)
                self.want_shallow, self.shallow_failed = False, True
        refit = self.sync(quiet=True, retry_shallow_oom=retry_oom)
        return refit, plan_seconds, kept

    def _discard(self):
        self.keep = False
        try:
            self.detach()
        finally:
            self.keep = True
        gpu.forget(self.owner)

    def _load(self, role: str, engine, kept) -> float:
        blob = kept.blobs.get(role) if kept is not None else None
        self.applied.pop(role, None)
        if blob is not None:
            with blob.open() as serialized:
                if serialized is None:
                    logger.info(f"The system took back RAM the {self.name} was kept in; loading its plan and refitting it instead.")
                    refits.drop(self.record.key, role)
                else:
                    from .engine import LOGGER

                    failures = gpu.failures()
                    try:
                        seconds = engine.load(serialized)
                        self.applied[role] = dict(kept.signatures[role])
                        return seconds
                    except Exception as e:
                        engine.unload()
                        if gpu.failures() != failures or LOGGER.out_of_memory:
                            raise  # out of VRAM, not a bad copy: partially_load makes more room and loads it again
                        logger.info(f"Couldn't load the {self.name} from RAM ({e}); loading its plan and refitting it instead.")
                        refits.drop(self.record.key, role)
        return engine.load()

    def _kept(self, weights=None):
        weights = weights if weights is not None else self.weights
        entry = refits.get(self.record.key)
        if entry is None or weights is None:
            return None
        wanted = weights.signature
        if "unet" in entry.blobs and all(wanted(name) == sig for sigs in entry.signatures.values() for name, sig in sigs.items()):
            return entry
        refits.drop(self.record.key)
        return None

    def _keep(self):
        loaded = {role: e for role, e in (("unet", self.engine), ("shallow", self.shallow)) if e is not None and e.loaded and self.applied.get(role)}
        if "unet" not in loaded or not self._current():
            why = "it holds no weights" if "unet" not in loaded else "its weights aren't the loaded checkpoint's"
            logger.info(f"Not keeping the {self.name} in RAM: {why}.")
            return
        key, label = self.record.key, self.name
        entry = refits.get(key)
        if entry is not None:
            if entry.signatures.get("unet") != self.applied["unet"]:
                refits.drop(key)
            else:
                for role in [role for role, blob in entry.blobs.items() if not blob.intact()]:
                    refits.drop(key, role)
            entry = refits.get(key)
        todo = [role for role in loaded if entry is None or entry.signatures.get(role) != self.applied[role]]
        if not todo:
            return
        todo.sort(key=lambda r: 0 if r == "shallow" else 1)  # tiny shallow fills in background while unet serializes
        if not refits.room(sum(self._serialized_size(role) for role in todo), key):
            logger.info(f"Not keeping the {label} in RAM: {refits.explain()}.")
            return
        start = time.time()
        torch.cuda.current_stream(self.load_device).synchronize()
        blobs = dict(entry.blobs) if entry is not None else {}
        signatures = dict(entry.signatures) if entry is not None else {}
        made = []
        try:
            for role in todo:

                def _failed(err, blob, k=key, r=role, lbl=label):
                    logger.warning(f"Couldn't keep the {lbl} in RAM, so loading it again will refit it: {err}")
                    refits.drop(k, r, blob=blob)

                made.append(refits.store_async(loaded[role].serialize(), on_error=_failed))
                blobs[role], signatures[role] = made[-1], dict(self.applied[role])
        except BaseException:
            for blob in made:
                blob.free()
            for blob in made:
                blob.wait_fill()
            raise
        kept = refits.Kept(signatures, blobs, label, self._source())
        if not refits.put(key, kept, expected=entry):
            for blob in made:
                blob.free()
            for blob in made:
                blob.wait_fill()
            return
        count, total = refits.held()
        logger.info(f"Kept the {label} in RAM ({common.gib(kept.nbytes)}, {time.time() - start:.1f}s); {count} kept in all, {common.gib(total)}.")

    def _serialized_size(self, role: str) -> int:
        plan, weights = (self.record.plan, self.record.weight_bytes) if role == "unet" else (self.record.shallow_plan, self.hacks["weight_bytes"])
        if role not in self._plan_bytes:
            try:
                self._plan_bytes[role] = os.path.getsize(plan)
            except OSError:
                self._plan_bytes[role] = 0
        return weights + self._plan_bytes[role]

    def _source(self):
        return next(iter(self.applied["unet"].values()))[0][0].ref

    def _current(self) -> bool:
        try:
            module = self._source()()
            return module is not None and module is arch_mod.unet_patcher().model.diffusion_model
        except Exception:
            return False

    def load_shallow(self):
        if not self.loaded or self.shallow is None or self.shallow.loaded:
            return
        kept = self._kept() if self.weights is not None else None
        need = common.padded(self.full(True, kept) - self.full(False, kept))
        keep = _loaded_entries(self)
        if common.make_room(need, self.load_device, keep) < need:
            with _evictable():
                common.make_room(need, self.load_device, keep)
        start = time.time()
        load = {}
        try:
            with gpu.loading(self.owner) as load:
                self._load("shallow", self.shallow, kept)
                self.sync(quiet=True, drop_shallow=False)
        except Exception:
            self.shallow.unload()
            gpu.unloaded(self.owner, load.get("grew", 0))
            self.applied.pop("shallow", None)
            self.want_shallow, self.shallow_failed = False, True
            raise
        logger.info(f"Loaded the Speed Hacks part of the {self.name} in {time.time() - start:.1f}s; the engine holds {common.gib(gpu.held(self.owner))} of VRAM with it.")

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all and self.opened:
            if self.keep and self.loaded:
                try:
                    self._keep()
                except Exception as e:
                    logger.warning(f"Couldn't keep the {self.name} in RAM, so loading it again will refit it: {e}")
            if self.shallow is not None:
                self.shallow.unload()
            self.engine.unload()
            self.applied, self.shallow_failed, self.want_shallow, self.weights = {}, False, False, None
            left = gpu.ledger(self.owner)
            if left:
                logger.debug(f"TensorRT still holds {common.gib(left)} of the {self.name}'s VRAM after unloading it.")
            gpu.forget(self.owner)
            logger.info(f"Unloaded the {self.name}.")
        return self.model

    def sync(self, quiet=False, retry_shallow_oom: bool = False, drop_shallow: bool = True):
        from . import engine as engine_mod

        weights = self.weights
        weights.refresh()
        total = None
        for role, engine in (("unet", self.engine), ("shallow", self.shallow)):
            if engine is None or not engine.loaded:
                continue
            wanted = {name: weights.signature(name) for name in engine.weight_names if weights.has(name)}
            applied = self.applied.get(role, {})
            changed = [name for name in wanted if applied.get(name) != wanted[name]]
            if not changed:
                continue
            self.applied[role] = {}
            refits.drop(self.record.key, role)  # the copy kept in RAM holds the old weights
            torch.cuda.current_stream(self.load_device).synchronize()  # no inference in flight while the weights change
            can_retry_unet = role == "unet" and not gpu.counting(self.owner)
            if can_retry_unet and len(changed) >= 16 and common.free(self.load_device, at_least=1 << 30) < (1 << 30):
                # Free evictable non-engines (CLIP/VAE) before a large live refit so GPU staging fits without touching other UNet engines.
                keep_engines = [
                    lm for lm in common.mm().current_loaded_models
                    if lm.model is not None and (lm.model is self or self.is_clone(lm.model) or getattr(lm.model, "evicted_last", False))
                ]
                common.make_room(1 << 30, self.load_device, keep_engines)
            seconds = None
            dropped = False
            for attempt in (0, 1) if can_retry_unet else (0,):
                engine_mod.LOGGER.clear()
                failures = gpu.failures()
                try:
                    seconds = engine.refit(changed, weights.tensor)
                    break
                except Exception as e:
                    ran_out = isinstance(e, torch.cuda.OutOfMemoryError) or gpu.failures() != failures or engine_mod.LOGGER.out_of_memory
                    if can_retry_unet and ran_out and not attempt:
                        e.__traceback__ = None
                        with _evictable():
                            common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, self.load_device, keep_loaded=_loaded_entries(self))
                        common.trim()
                        continue
                    if role != "shallow" or not drop_shallow:
                        raise
                    if retry_shallow_oom and ran_out:
                        raise
                    notify(f"Couldn't refit the Speed Hacks part of the {self.name}, so it runs without Speed Hacks: {e}")
                    e.__traceback__ = None
                    if gpu.counting(self.owner):
                        engine.unload()
                    elif gpu.attached():
                        before, prev_extra = gpu.ledger(self.owner), engine.extra
                        engine.unload()
                        base_freed = max(0, before - gpu.ledger(self.owner) - prev_extra)
                        if base_freed:
                            gpu.unloaded(self.owner, base_freed)
                    else:
                        engine.release_activations()
                        try:
                            before = gpu.in_use()
                        except Exception:
                            before = None
                        engine.unload()
                        if before is not None:
                            with contextlib.suppress(Exception):
                                base_freed = max(0, before - gpu.in_use())
                                if base_freed:
                                    gpu.unloaded(self.owner, base_freed)
                    self.applied.pop("shallow", None)
                    self.want_shallow, self.shallow_failed = False, True
                    dropped = True
                    break
            if dropped:
                continue
            self.applied[role] = wanted
            refit = {"seconds": seconds, "count": len(changed), "patched": sum(weights.is_patched(name) for name in changed)}
            total = refit if total is None else {k: total[k] + refit[k] for k in refit}
        if total is not None and not quiet:
            logger.info(f"The {self.name} {_refit_times(total)}.")
        return total


def _refit_times(refit: dict) -> str:
    loras = f" ({refit['patched']} with LoRAs)" if refit["patched"] else ""
    return f"refitted {refit['count']} weights{loras} in {refit['seconds']:.1f}s"


def _loaded_entries(*models):
    return [lm for lm in common.mm().current_loaded_models if lm.model is not None and any(lm.model is m or m.is_clone(lm.model) for m in models)]


@inference_mode
def _unload(model: TrtModel, keep: bool = True):
    mm = common.mm()
    model.keep = keep
    try:
        for i in reversed(range(len(mm.current_loaded_models))):
            if mm.current_loaded_models[i].model is model:
                mm.current_loaded_models.pop(i).model_unload()
        model.detach()
    finally:
        model.keep = True


def end_pass():
    global _pass
    p, _pass = _pass, None
    _prompt_seen.clear()
    _prompt_reps.clear()
    if p is not None:
        p.end()


def release_all():
    end_pass()
    for model in list(_models.values()):
        _unload(model, keep=False)
    _models.clear()
    _release_unet_arena()
    _unet_arenas.clear()
    _last_used.clear()
    _use_order.clear()
    vae_run.release_all()
    refits.clear()
    common.mm().soft_empty_cache(force=True)


def model_for(record: store.Record) -> TrtModel:
    model = _models.get(record.key)
    if model is None or model.record.meta != record.meta:
        if model is not None:
            _unload(model, keep=False)
            refits.drop(record.key)
        model = _models[record.key] = TrtModel(record)
    return model


AUTOMATIC, SPEED_HACKS = "Automatic", "Speed Hacks"


def setting() -> str:
    try:
        from modules import shared

        return str(shared.opts.sd_unet)
    except Exception:
        return AUTOMATIC


FULL_START, FULL_END, QUALITY = "trt_speed_hacks_full_start", "trt_speed_hacks_full_end", "trt_speed_hacks_quality"
# Their names in infotext: written for a generation that ran Speed Hacks, and read back by reForge's paste as overrides
INFOTEXT = {FULL_START: "Speed Hacks full start", FULL_END: "Speed Hacks full end", QUALITY: "Speed Hacks quality"}


def speed_settings() -> tuple:
    from modules import shared

    def fraction(key, default, scale=1.0):
        try:
            return min(max(float(getattr(shared.opts, key)) / scale, 0.0), 1.0)
        except Exception:
            return default

    return fraction(FULL_START, speed.WARMUP, 100), fraction(FULL_END, speed.TAIL, 100), fraction(QUALITY, speed.QUALITY)


class SpeedHacksOption(sd_unet.SdUnetOption):
    label = SPEED_HACKS
    model_name = None  # so "Automatic" never selects it

    def create_unet(self):
        return TrtUnet(self)


class TrtUnet(getattr(sd_unet, "SdUnet", object)):
    def __init__(self, option: SpeedHacksOption):
        super().__init__()
        self.option = option

    def activate(self):
        pass

    def deactivate(self):
        pass

    def forward(self, *args, **kwargs):
        raise RuntimeError("TensorRT engines run through the UNet's own forward")


_SPEED_HACKS_OPTION = SpeedHacksOption()  # one instance: reForge compares options by identity when re-listing them


def list_unets(options: list):
    options.append(_SPEED_HACKS_OPTION)


_EXTRA_TRANSFORMER_OPTIONS = (
    "block_modifiers",
    "block_inner_modifiers",
    "group_norm_wrapper",
    "optimized_attention_override",
)


def _unsupported(patcher) -> str:
    if getattr(patcher, "controlnet_linked_list", None) is not None:
        return "ControlNet"
    if getattr(patcher, "extra_concat_condition", None) is not None:
        return "extra input channels (inpainting add-ons)"
    kmodel = getattr(patcher, "model", None)
    if getattr(kmodel, "_torch_compile_config", None) or getattr(getattr(kmodel, "apply_model", None), "_torch_compile_wrapper", False):
        return "torch.compile"
    module = getattr(kmodel, "diffusion_model", None)
    if module is not None:
        paddings = {m.padding_mode for m in module.modules() if isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"}
        if paddings:
            return "tiling (circular padding)" if paddings == {"circular"} else f"non-zero conv padding ({', '.join(sorted(paddings))})"
    options = patcher.model_options.get("transformer_options", {})
    kinds = sorted({k for group in ("patches", "patches_replace") for k, v in options.get(group, {}).items() if v})
    for extra in _EXTRA_TRANSFORMER_OPTIONS:
        if options.get(extra):
            kinds.append(extra)
    if options.get("wrappers", {}).get("diffusion_model"):
        kinds.append("UNet wrappers")
    if any(k.startswith("diffusion_model.") for k in getattr(patcher, "object_patches", {})):
        kinds.append("replaced UNet modules")
    wrappers = [
        w for k, v in (getattr(patcher, "weight_wrapper_patches", None) or {}).items()
        if k.startswith("diffusion_model.") and v for w in v
    ]
    if wrappers:
        if not all(compat.is_online_lora_patch(w) for w in wrappers):
            kinds.append("weight wrappers")
        elif compat.has_scheduled_lora():
            kinds.append("scheduled LoRAs")
    if getattr(patcher, "hook_patches", None):
        kinds.append("scheduled weights")
    return f"UNet patches ({', '.join(kinds)})" if kinds else ""


@inference_mode
def _unload_torch_unet(patcher):
    mm = common.mm()
    unloaded = False
    for lm in _loaded_entries(patcher):
        if lm.model_loaded_memory() > 0:
            lm.model_unload()
            mm.current_loaded_models.remove(lm)
            unloaded = True
    if unloaded:
        mm.soft_empty_cache(force=True)


_evicting = 0  # > 0 inside _evictable(): loads for PyTorch's UNet and unload_all_models evict engines normally
_no_free = 0  # > 0 inside _begin()'s sampling_prepare: keeps engines from being evicted while patches apply
_kept = []  # model tuples appended to keep_loaded by _wrap_free_memory


@contextlib.contextmanager
def _keeping(*models):
    # Loads inside this block can't evict these models. During a pass the UNet must stay loaded (unloading it undoes its
    # patches, e.g. model sampling, which the pass still uses) and so must the engine (it would reload on the next step).
    entry = tuple(models)
    _kept.append(entry)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            _kept.remove(entry)


@contextlib.contextmanager
def _evictable():
    global _evicting
    _evicting += 1
    try:
        yield
    finally:
        _evicting -= 1


def _wrap_free_memory(original):
    # reForge evicts models until its estimate of a load's needs fits. Engines are the costliest to bring back, so they
    # go last, and only if evicting them makes the estimate fit: a worst-case estimate nothing could meet (e.g. a large
    # VAE decode in FP32) no longer takes the engine along; the load makes do with what's free, as it does anyway.
    def free_memory(memory_required, device, keep_loaded=[]):
        if _no_free:
            return []
        keep = list(keep_loaded)
        if _kept:
            extra = _loaded_entries(*(m for group in list(_kept) for m in group))
            keep.extend(lm for lm in extra if lm not in keep)
        mm = common.mm()
        engines = []
        if not getattr(mm, "DISABLE_SMART_MEMORY", False):
            engines = [lm for lm in mm.current_loaded_models if getattr(lm.model, "evicted_last", False) and lm.device == device and lm not in keep]
        if _evicting:
            if len(engines) <= 1:
                return original(memory_required, device, keep)
            unloaded = original(memory_required, device, keep + engines)
            ordered = sorted(engines, key=lambda lm: _use_order.get(getattr(getattr(lm.model, "record", None), "key", None), 0))
            for i in range(1, len(ordered) + 1):
                if mm.get_free_memory(device) >= memory_required:
                    break
                unloaded += original(memory_required, device, keep + ordered[i:])
            return unloaded
        if not engines:
            return original(memory_required, device, keep)
        unloaded = original(memory_required, device, keep + engines)
        short = memory_required - mm.get_free_memory(device)
        if 0 < short <= sum(lm.model_loaded_memory() for lm in engines):
            ordered = sorted(engines, key=lambda lm: _use_order.get(getattr(getattr(lm.model, "record", None), "key", None), 0))
            for i in range(1, len(ordered) + 1):
                unloaded += original(memory_required, device, keep + ordered[i:])
                if mm.get_free_memory(device) >= memory_required:
                    break
        return unloaded

    free_memory._trt_original = original
    return free_memory


def _wrap_get_free_memory(original):
    # Never report less than the driver's free memory. With --cuda-malloc, PyTorch reads its counts from the driver's
    # pool, whose in-use count has been seen 1.3 GiB above what it holds, and reForge then loaded its UNet partially
    # and gave its VAE 128 MiB.
    def get_free_memory(dev=None, torch_free_too=False):
        with common.on_device() if dev is None else contextlib.nullcontext():
            result = original(dev, torch_free_too)
        try:
            device = torch.device(dev if dev is not None else common.device())
            if device.type != "cuda":
                return result
            driver = int(torch.cuda.mem_get_info(device)[0])
        except Exception:
            return result
        if torch_free_too:
            total, cached = result
            return (driver, max(cached, 0)) if total < driver else result
        return max(result, driver)

    get_free_memory._trt_original = original
    return get_free_memory


def _wrap_unload_all(original):
    def unload_all_models(*args, **kwargs):
        with _evictable(), common.on_device():
            return original(*args, **kwargs)

    unload_all_models._trt_original = original
    return unload_all_models


@inference_mode
def _load_torch_unet(patcher, shape):
    estimate = patcher.model_options.get("memory_peak_estimation_modifier", patcher.memory_required)
    required = estimate(list(shape)) + getattr(patcher, "extra_preserved_memory_during_sampling", 0)
    if callable(getattr(patcher, "has_online_lora", None)) and patcher.has_online_lora():
        try:
            from backend import utils as neo_utils

            required += neo_utils.nested_compute_size(
                patcher.weight_wrapper_patches,
                element_size=neo_utils.dtype_to_element_size(patcher.model.computation_dtype),
            )
        except Exception:
            pass
    with _evictable():
        common.mm().load_models_gpu([patcher], memory_required=required)


_MISSING = object()


def _wraps_unet(patcher) -> bool:
    options = getattr(patcher, "model_options", None) or {}
    return bool(options.get("model_function_wrapper") or options.get("tiled_diffusion"))


class Pass:
    def __init__(self, records: list, patcher, x, hacks: bool = False):
        self.patcher = patcher
        model = patcher.model
        self.schedule = self.tracker = None
        if hacks:
            try:
                self.tracker = fresh.Tracker(compat.model_sampling(model), common.device(), model.diffusion_model.out_channels)
                self.schedule = speed.Schedule(*speed_settings())
            except Exception as e:
                notify(f"Speed Hacks failed ({e}), so this pass runs without them.", e)
                self.tracker, hacks = None, False
        self.slots = store.slots(records)
        self.model = None
        self.weights = EffectiveWeights(patcher)
        self.module = patcher.model.diffusion_model
        arch = records[0].arch
        self.arch = arch
        self.arch_id = arch.id
        self.shapes, self.takes_y = [r.profile_shapes[j] for r, j in self.slots], bool(arch.adm_channels)
        self.channels, self.context_dim = arch.in_channels, arch.context_dim
        self._x_shape = tuple(x.shape)
        self.shape = (x.shape[0] * 2, *x.shape[1:])
        self.fallback = None
        self.checked = set()
        self.shallow_checked = set()
        self._keep = None
        options = getattr(patcher, "model_options", None) or {}
        self._deferred = _wraps_unet(patcher)
        self._deferred_seen = False
        self._eval_id = 0
        self._remaining_steps = None
        self._last_step = None
        self._step_evals = 0
        self._evals_per_step = 1
        self._cur_eval_id = -1
        self._cur_eval_shapes = []
        self._anchor = None  # (b, h, w, context_len, b * h * w) of the heaviest shape self.model has run
        self._last_anchor_eval = -1
        self._engine_eval = {}  # record.key -> last _eval_id it ran on
        self._evicts_anchor = set()  # record keys proven to evict the anchor engine when loaded mid-pass
        self._synced = {}  # record.key -> (patches_uuid, shallow_loaded) synced in this pass
        self._torch_partial = False
        self._torch_fast_loaded = False
        self._no_coexist = False
        self._torch_snap = None
        self._torch_mode = False
        self._torch_eval_id = -1
        self._torch_eval_shapes = []
        self._torch_joint_key = None
        self._torch_stable_evals = 0
        self._torch_resumes = 0
        self._torch_logged = False
        if not options.get("tiled_diffusion"):
            self._hint_batching()
        self.started = self.schedule  # for end()'s report once Speed Hacks turn off mid-pass
        self._watching = False
        if hacks:
            self._watch_apply_model(model)

    def _need_for(self, model: TrtModel, want_shallow: bool, call_shape: tuple = None, min_calls: int = 0) -> int:
        if model.loaded:
            load_need = model._serialized_size("shallow") if want_shallow and not model.shallow_loaded else 0
        else:
            load_need = model.full(want_shallow, model._kept(self.weights))
        if call_shape is not None:
            padded, h, w = call_shape
            calls = max(
                working.call_bytes(model.record.arch, self.patcher, padded, w * 8, h * 8),
                model.took.get((padded, h, w), 0),
            )
        else:
            shape = _Shape(self._x_shape)
            calls = max(call_bytes(model, self.patcher, shape), model.took.get(_took_key(model, shape), 0))
        return common.padded(load_need + max(calls, min_calls))

    def _snapshot_torch_cpu(self):
        # Hold references to the UNet's existing CPU weights before partially_load moves them to VRAM so unloading
        # simply restores the CPU pointers (~50ms) instead of copying 4.8 GiB back over PCIe (~3.3s). Skipped on
        # RAM-constrained hosts so we never hold extra RAM pressure when free RAM is tight.
        # Under NO_VRAM with LoRAs, reForge/Neo attaches tiny LowVramPatch hooks (a few KB in loaded_size) while leaving
        # every weight parameter on CPU; verify parameters directly instead of requiring loaded_size() == 0.
        if self._torch_snap is not None or self.patcher.loaded_size() > (16 << 20) or getattr(self.patcher, "backup", None):
            return
        if (
            getattr(self.patcher, "weight_inplace_update", False)
            or getattr(self.patcher, "is_injected", False)
            or getattr(self.patcher, "ejected", False)
            or getattr(self.patcher, "weight_wrapper_patches", None)
            or getattr(self.patcher, "pinned", None)
        ):
            return
        model = getattr(self.patcher, "model", None)
        if not isinstance(model, torch.nn.Module):
            return
        if getattr(model, "current_weight_patches_uuid", None) != getattr(self.patcher, "patches_uuid", None):
            return
        try:
            if refits.memory()[0] < self.patcher.model_size() + (2 << 30):
                return
        except Exception:
            return
        off = torch.device(self.patcher.offload_device)
        if off.type != "cpu":
            return
        params, bufs, attrs = [], [], []
        for mod in model.modules():
            for name, p in mod._parameters.items():
                if p is not None:
                    if p.device != off or type(p) is not torch.nn.Parameter or type(p.data) is not torch.Tensor:
                        return
                    params.append((mod, name, p, p.data))
            for name, b in mod._buffers.items():
                if b is not None and b.device == off:
                    bufs.append((mod, name, b))
            a = {}
            for k in ("weight_function", "bias_function"):
                a[k] = list(mod.__dict__[k]) if k in mod.__dict__ else _MISSING
            for k in compat.SNAPSHOT_ATTRS:
                a[k] = mod.__dict__.get(k, _MISSING)
            if a:
                attrs.append((mod, a))
        state = (
            getattr(model, "model_lowvram", True),
            getattr(model, "lowvram_patch_counter", 0),
            getattr(model, "model_loaded_weight_memory", 0),
            model.__dict__.get("device", _MISSING),
            getattr(model, "current_weight_patches_uuid", None),
            getattr(self.patcher, "patches_uuid", None),
            self.patcher.__dict__.get("current_device", _MISSING),
            model.__dict__.get("model_offload_buffer_memory", _MISSING),
        )
        self._torch_snap = (params, bufs, attrs, state)

    def _restore_torch_snap(self, snap):
        params, bufs, attrs, state = snap
        with torch.inference_mode():
            if hasattr(self.patcher, "unpatch_hooks"):
                self.patcher.unpatch_hooks()
            if hasattr(self.patcher, "unpin_all_weights"):
                self.patcher.unpin_all_weights()
            self.patcher.backup.clear()
            for mod, name, p, cpu_data in params:
                p.data = cpu_data
                mod._parameters[name] = p
            for mod, name, cpu_buf in bufs:
                mod._buffers[name] = cpu_buf
            for mod, a in attrs:
                for k, v in a.items():
                    if v is _MISSING:
                        mod.__dict__.pop(k, None)
                    elif k in ("weight_function", "bias_function"):
                        mod.__dict__[k] = list(v)
                    else:
                        mod.__dict__[k] = v
            m = self.patcher.model
            m.model_lowvram, m.lowvram_patch_counter, m.model_loaded_weight_memory, m_dev, m.current_weight_patches_uuid, _, p_dev, offload_buf = state
            if m_dev is _MISSING:
                m.__dict__.pop("device", None)
            else:
                m.device = m_dev
            if p_dev is not _MISSING:
                self.patcher.current_device = p_dev
            if offload_buf is not _MISSING:
                m.model_offload_buffer_memory = offload_buf

    def _fast_load_torch_unet(self, dev: torch.device):
        # Streams CPU UNet parameters into <=64 MiB GPU slabs (ordered by ModelPatcher._load_list() so partially_unload
        # releases them incrementally) via double-buffered pinned DMA and merges standard LoRAs in-place on GPU, so
        # reForge/Neo's partially_load() only performs bookkeeping instead of 1,680 pageable copies.
        if self._torch_snap is None or dev.type != "cuda":
            return
        if compat.pin_shared_memory(common.mm()):
            return
        entries = compat.load_list_entries(self.patcher)
        if entries is None:
            return
        model = self.patcher.model
        off = torch.device(self.patcher.offload_device)
        if off.type != "cpu":
            return
        patches = getattr(self.patcher, "patches", None) or {}
        wwp = getattr(self.patcher, "weight_wrapper_patches", None) or {}
        self.weights.refresh()
        use_lora = (
            bool(patches)
            and self.weights.patches is patches
            and self.weights.module is getattr(model, "diffusion_model", None)
        )
        prefix_dm = "diffusion_model."
        align256 = lambda n: (n + 255) & ~255
        align16 = lambda n: (n + 15) & ~15
        safe_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8, torch.int8)
        lora_dtypes = (torch.float16, torch.bfloat16, torch.float32)

        param_refs = collections.Counter()
        for mod in model.modules():
            for p in mod._parameters.values():
                if p is not None:
                    param_refs[id(p)] += 1

        slab_cap = 64 << 20
        slab_groups = []
        cur_direct, cur_lora, cur_bytes = [], [], 0
        lora_mods = []
        seen_p = set()

        for prefix, mod, param_names in entries:
            mod_patched = []
            mod_direct = []
            for name in param_names:
                p = mod._parameters.get(name)
                if p is None or id(p) in seen_p:
                    continue
                seen_p.add(id(p))
                full_key = f"{prefix}.{name}" if prefix else name
                if full_key in patches:
                    mod_patched.append((name, full_key, p))
                elif (
                    p.device == off
                    and param_refs[id(p)] == 1
                    and type(p.data) is torch.Tensor
                    and p.data.dtype in safe_dtypes
                    and p.data.is_contiguous()
                    and p.numel() > 0
                    and not callable(getattr(mod, f"convert_{name}", None))
                ):
                    d = p.data
                    nb = d.numel() * d.element_size()
                    mod_direct.append((p, d, nb, align256(nb)))
            # Patched modules without cast_weights/parameters_manual_cast are skipped by ModelPatcher.partially_unload();
            # leave the whole module to reForge/Neo so it can never pin a shared slab.
            if mod_patched and not compat.has_cast_weights(mod):
                continue
            mod_lora = []
            if mod_patched and use_lora and not compat.is_patched_weights(mod):
                cand = []
                ok_mod = True
                for name, full_key, p in mod_patched:
                    if (
                        not full_key.startswith(prefix_dm)
                        or full_key in wwp
                        or p.device != off
                        or param_refs[id(p)] != 1
                        or type(p.data) is not torch.Tensor
                        or p.data.dtype not in lora_dtypes
                        or not p.data.is_contiguous()
                        or p.numel() == 0
                        or callable(getattr(mod, f"convert_{name}", None))
                    ):
                        ok_mod = False
                        break
                    short = full_key[len(prefix_dm):]
                    if self.weights.params.get(short) is not p:
                        ok_mod = False
                        break
                    items = self.weights.fast_lora(short)
                    if items is None or any((u.is_cuda and u.device != dev) or (dn.is_cuda and dn.device != dev) for u, dn, _ in items):
                        ok_mod = False
                        break
                    d = p.data
                    nb = d.numel() * d.element_size()
                    cand.append((full_key, p, d, items, nb, align256(nb)))
                if ok_mod and cand:
                    mod_lora = cand
                    lora_mods.append(mod)
            mod_bytes = sum(x[3] for x in mod_direct) + sum(x[5] for x in mod_lora)
            if mod_bytes == 0:
                continue
            if cur_bytes > 0 and cur_bytes + mod_bytes > slab_cap:
                slab_groups.append((cur_direct, cur_lora))
                cur_direct, cur_lora, cur_bytes = [], [], 0
            cur_direct.extend(mod_direct)
            cur_lora.extend(mod_lora)
            cur_bytes += mod_bytes
        if cur_bytes > 0:
            slab_groups.append((cur_direct, cur_lora))

        slab_sizes = []
        direct_plan = []
        lora_plan = []
        for slab_idx, (s_direct, s_lora) in enumerate(slab_groups):
            s_off = 0
            for p, d, nb, anb in s_direct:
                direct_plan.append((slab_idx, s_off, p, d, nb, anb))
                s_off += anb
            for full_key, p, d, items, nb, anb in s_lora:
                lora_plan.append((slab_idx, s_off, full_key, p, d, items, nb))
                s_off += anb
            slab_sizes.append(s_off)

        total_aligned = sum(slab_sizes)
        if total_aligned < (32 << 20):
            return

        slabs = []
        pins = lora_gbufs = copy_stream = compute_stream = None
        mutated = False
        synced = False
        failed = False
        try:
            with common.on_device(dev):
                for s_total in slab_sizes:
                    slabs.append(torch.empty(s_total, dtype=torch.uint8, device=dev))
                chunk = min(slab_cap, total_aligned)
                pins = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
                max_lora_bytes = 0
                if lora_plan:
                    for _, _, _, _, _, items, _ in lora_plan:
                        l_nb = 0
                        for u, dn, _ in items:
                            if not u.is_cuda:
                                l_nb += align16(u.numel() * u.element_size())
                            if not dn.is_cuda:
                                l_nb += align16(dn.numel() * dn.element_size())
                        if l_nb > max_lora_bytes:
                            max_lora_bytes = l_nb
                    if max_lora_bytes > 0:
                        lora_gbufs = [torch.empty(max_lora_bytes, dtype=torch.uint8, device=dev) for _ in range(2)]
                copy_stream = torch.cuda.Stream(device=dev)
                compute_stream = torch.cuda.current_stream(dev)
                ev_alloc = torch.cuda.Event()
                ev_alloc.record(compute_stream)
                copy_stream.wait_event(ev_alloc)
                ev_copy = [torch.cuda.Event() for _ in range(2)]
                ev_lora_use = [torch.cuda.Event() for _ in range(2)]
                slot_used = [False, False]
                lora_used = [False, False]

                slot = 0
                p_off = 0
                cur_slab = -1
                s_base = 0

                def flush_direct():
                    nonlocal slot, p_off
                    if p_off <= 0:
                        return
                    with torch.cuda.stream(copy_stream):
                        slabs[cur_slab][s_base : s_base + p_off].copy_(pins[slot][:p_off], non_blocking=True)
                        ev_copy[slot].record(copy_stream)
                    slot_used[slot] = True
                    slot ^= 1
                    if slot_used[slot]:
                        ev_copy[slot].synchronize()
                    p_off = 0

                for slab_idx, s_off, p, d, nb, anb in direct_plan:
                    if slab_idx != cur_slab or p_off + anb > chunk:
                        flush_direct()
                        cur_slab = slab_idx
                        s_base = s_off
                    view = slabs[slab_idx][s_off : s_off + nb].view(d.dtype).view(d.shape)
                    if anb > chunk:
                        view.copy_(d)
                        s_base = s_off + anb
                    else:
                        pins[slot][p_off : p_off + nb].view(d.dtype).view(d.shape).copy_(d)
                        p_off += anb
                    mutated = True
                    p.data = view
                flush_direct()

                if lora_plan:
                    def stage_lora(idx):
                        slab_idx, off_s, full_key, p, d, items, nb = lora_plan[idx]
                        sview = slabs[slab_idx][off_s : off_s + nb].view(d.dtype).view(d.shape)
                        base_end = align16(nb)
                        l_specs = []
                        l_total = 0
                        cpu_tensors = []
                        for u, dn, sc in items:
                            pair = []
                            for t in (u, dn):
                                if t.is_cuda:
                                    pair.append(t)
                                else:
                                    tc = t if t.is_contiguous() else t.contiguous()
                                    tnb = tc.numel() * tc.element_size()
                                    pair.append((l_total, tnb, tc.dtype, tc.shape))
                                    cpu_tensors.append((l_total, tnb, tc))
                                    l_total = align16(l_total + tnb)
                            l_specs.append((pair[0], pair[1], sc))
                        if base_end + l_total > chunk or (l_total > 0 and lora_gbufs is None):
                            return None, sview, d, items, full_key, p
                        sl = idx & 1
                        if slot_used[sl]:
                            ev_copy[sl].synchronize()
                        pbuf = pins[sl]
                        pbuf[:nb].view(d.dtype).view(d.shape).copy_(d)
                        for loff, tnb, tc in cpu_tensors:
                            pbuf[base_end + loff : base_end + loff + tnb].view(tc.dtype).view(tc.shape).copy_(tc)
                        with torch.cuda.stream(copy_stream):
                            if lora_used[sl]:
                                copy_stream.wait_event(ev_lora_use[sl])
                            slabs[slab_idx][off_s : off_s + nb].copy_(pbuf[:nb], non_blocking=True)
                            if l_total > 0:
                                lora_gbufs[sl][:l_total].copy_(pbuf[base_end : base_end + l_total], non_blocking=True)
                            ev_copy[sl].record(copy_stream)
                        slot_used[sl] = True
                        gbuf = lora_gbufs[sl] if lora_gbufs is not None else None
                        g_items = []
                        for us, ds, sc in l_specs:
                            ug = us if isinstance(us, torch.Tensor) else gbuf[us[0] : us[0] + us[1]].view(us[2]).view(us[3])
                            dg = ds if isinstance(ds, torch.Tensor) else gbuf[ds[0] : ds[0] + ds[1]].view(ds[2]).view(ds[3])
                            g_items.append((ug, dg, sc))
                        return sl, sview, d, g_items, full_key, p

                    staged = stage_lora(0)
                    for idx in range(len(lora_plan)):
                        sl, sview, d, g_items, full_key, p = staged
                        if idx + 1 < len(lora_plan):
                            staged = stage_lora(idx + 1)
                        if sl is None:
                            sview.copy_(d)
                            g_ready = [(u if u.is_cuda else u.to(dev), dn if dn.is_cuda else dn.to(dev), sc) for u, dn, sc in g_items]
                            EffectiveWeights.apply_fast_lora(sview, g_ready, d.dtype, out=sview)
                            del g_ready
                        else:
                            compute_stream.wait_event(ev_copy[sl])
                            EffectiveWeights.apply_fast_lora(sview, g_items, d.dtype, out=sview)
                            ev_lora_use[sl].record(compute_stream)
                            lora_used[sl] = True
                        mutated = True
                        p.data = sview
                        self.patcher.backup[full_key] = _BackupWeight(d, False)

                copy_stream.synchronize()
                compute_stream.synchronize()
                synced = True
                for mod in lora_mods:
                    compat.mark_patched_weights(mod)
                self._torch_fast_loaded = True
        except BaseException as e:
            failed = True
            if isinstance(e, Exception):
                _clear_traceback(e)
                logger.debug(f"Couldn't fast-upload PyTorch UNet weights; falling back to reForge loader: {e}")
            else:
                raise
        finally:
            if not synced:
                if copy_stream is not None:
                    with contextlib.suppress(Exception):
                        copy_stream.synchronize()
                if compute_stream is not None:
                    with contextlib.suppress(Exception):
                        compute_stream.synchronize()
            if failed:
                if mutated and self._torch_snap is not None:
                    with contextlib.suppress(Exception):
                        self._restore_torch_snap(self._torch_snap)
                slabs.clear()
                slabs = None
            pins = lora_gbufs = None

    def _unload_torch_weights(self, force: bool = False):
        snap, self._torch_snap = self._torch_snap, None
        fast_loaded, self._torch_fast_loaded = self._torch_fast_loaded, False
        if not force and not self._torch_partial and not fast_loaded:
            return
        self._torch_partial = False
        base_loaded = snap[3][2] if snap is not None else 0
        cur_uuid = getattr(self.patcher.model, "current_weight_patches_uuid", None)
        ext_unloaded = (
            snap is not None
            and cur_uuid != snap[3][4]
            and self.patcher.loaded_size() <= base_loaded
            and not _loaded_entries(self.patcher)
        )
        if not force and (ext_unloaded or (not fast_loaded and self.patcher.loaded_size() <= base_loaded)):
            self.weights.refresh()
            return
        restored = False
        if snap is not None and (force or fast_loaded or bool(_loaded_entries(self.patcher))):
            if getattr(self.patcher, "patches_uuid", None) == snap[3][5] and (force or cur_uuid == snap[3][4]):
                try:
                    dev = common.device()
                    if dev.type == "cuda":
                        with contextlib.suppress(Exception):
                            torch.cuda.synchronize(dev)
                    self._restore_torch_snap(snap)
                    restored = True
                except Exception as e:
                    logger.debug(f"Couldn't fast-restore temporary PyTorch UNet weights: {e}")
        snap = None
        if not restored:
            try:
                with torch.inference_mode():
                    self.patcher.partially_unload(self.patcher.offload_device, sys.maxsize)
            except Exception as e:
                logger.debug(f"Couldn't offload temporary PyTorch UNet weights: {e}")
        self.weights.refresh()
        common.mm().soft_empty_cache(force=True)

    def _rollback_torch_load(self):
        if self._torch_snap is not None:
            self._unload_torch_weights(force=True)
            return
        self._torch_partial = False
        self._torch_fast_loaded = False
        with contextlib.suppress(Exception):
            with torch.inference_mode():
                if hasattr(self.patcher, "detach"):
                    self.patcher.detach()
                else:
                    self.patcher.partially_unload(self.patcher.offload_device, sys.maxsize)
        with contextlib.suppress(Exception):
            self.weights.refresh()
        with contextlib.suppress(Exception):
            common.mm().soft_empty_cache(force=True)

    def _select(self, b: int, h: int, w: int, context: int):
        choice = profiles.select(self.shapes, b, h, w, context, prefer=self.preferred())
        if choice is None:
            return None
        if self.model is None or not self.model.loaded or self.slots[choice[0]][0].key == self.model.record.key:
            return choice
        # Prefer another engine already resident in VRAM over loading an unloaded one from RAM or disk.
        loaded = {i for i, (r, _) in enumerate(self.slots) if (m := _models.get(r.key)) is not None and m.loaded}
        if loaded:
            alt = profiles.select(self.shapes, b, h, w, context, prefer=loaded)
            if alt is not None and alt[0] in loaded:
                return alt
        return choice

    def _joint_record(self, shapes: list):
        if not shapes:
            return None
        valid = sorted(
            i for i, (r, _) in enumerate(self.slots)
            if all(any(profiles.fit(s, b, h, w, ctx) is not None for s in r.profile_shapes) for b, h, w, ctx in shapes)
        )
        if not valid:
            return None
        hb, hh, hw, hctx = max(shapes, key=lambda t: t[0] * t[1] * t[2])
        subset = [self.shapes[i] for i in valid]
        pref = {k for k, i in enumerate(valid) if (m := _models.get(self.slots[i][0].key)) is not None and m.loaded}
        choice = profiles.select(subset, hb, hh, hw, hctx, prefer=pref)
        return self.slots[valid[choice[0]]][0] if choice is not None else None

    def _covers_anchor(self, record) -> bool:
        if self._anchor is None:
            return False
        ab, ah, aw, actx, _ = self._anchor
        return any(profiles.fit(s, ab, ah, aw, actx) is not None for s in record.profile_shapes)

    def _anchor_act_bytes(self) -> int:
        if self._anchor is not None and self.model is not None:
            ab, ah, aw, _, _ = self._anchor
            anchor_x = _Shape((ab, self.channels, ah, aw))
            if self.model.loaded and self._covers_anchor(self.model.record):
                anchor_model = self.model
            else:
                anchor_model = next(
                    (
                        m for m in _models.values()
                        if m.loaded and m.record.arch.id == self.arch_id and self._covers_anchor(m.record)
                    ),
                    self.model,
                )
            return max(
                call_bytes(anchor_model, self.patcher, anchor_x, unet_batch=ab),
                anchor_model.took.get(_took_key(anchor_model, anchor_x, unet_batch=ab), 0),
            )
        if self.model is not None:
            xs = _Shape(self._x_shape)
            return max(call_bytes(self.model, self.patcher, xs), self.model.took.get(_took_key(self.model, xs), 0))
        return 0

    def _would_evict_current(self, target: TrtModel, want_shallow: bool, call_shape: tuple) -> bool:
        if target.record.key in self._evicts_anchor:
            return True
        need = self._need_for(target, want_shallow, call_shape, min_calls=self._anchor_act_bytes())
        # Only stale engines from earlier passes count as freely evictable; evicting an active engine of this pass thrashes.
        active = {k for k, ev in self._engine_eval.items() if (self._eval_id - ev) <= 2}
        active.add(self.model.record.key)
        other_engines = sum(
            other.loaded_size() for other in _models.values()
            if other is not target and other.record.key not in active and other.loaded and other.load_device == target.load_device
        )
        arena = self.model.engine.arena.memory if self.model.engine is not None else None
        act_bytes = arena.numel() if arena is not None else 0
        res = common.reserve()
        want = max(0, need - other_engines - act_bytes + res)
        avail = common.obtainable(target.load_device, keep=[target], engines=False, at_least=want) + other_engines + act_bytes - res
        return avail < need

    def _torch_act_bytes(self, shape: tuple) -> int:
        sb, sh, sw, _ = shape
        est = working.call_bytes(self.arch, self.patcher, sb, sw * 8, sh * 8)
        try:
            model = self.patcher.model
            fn = getattr(type(model), "memory_required", None)
            if callable(fn):
                pt = fn(model, [sb, self.channels, sh, sw])
            else:
                saved = model.__dict__.pop("memory_required", _MISSING)
                try:
                    pt = self.patcher.memory_required([sb, self.channels, sh, sw])
                finally:
                    if saved is not _MISSING:
                        model.__dict__["memory_required"] = saved
            if pt and pt != float("inf"):
                est = max(est, int(pt))
        except Exception:
            pass
        return est

    def _can_coexist_torch(self, dev, shape: tuple) -> tuple:
        if self._no_coexist or self.model is None:
            return False, 0, 0
        main_act = self._anchor_act_bytes()
        side_act = self._torch_act_bytes(shape)
        headroom = max(common.padded(main_act), common.padded(side_act), 1024 << 20)
        keep_trt = [m for m in _models.values() if m.loaded]
        missing = 0 if self._torch_partial else max(0, self.patcher.model_size() - self.patcher.loaded_size())
        res = common.reserve()
        want = headroom + missing
        avail = common.obtainable(dev, keep=[self.patcher, *keep_trt], engines=False, at_least=want + res) - res
        return avail >= want, headroom, missing

    def _call_torch(self, why: str, shape: tuple):
        dev = common.device()
        if self.model is not None and self.model.loaded:
            self.model.engine.release_activations()
        coexist, headroom, missing = self._can_coexist_torch(dev, shape)
        if coexist:
            if not self._torch_logged:
                self._torch_logged = True
                logger.info(f"Running side calls on PyTorch while keeping the {self.model.name} in VRAM ({why}).")
            if not self._torch_partial and missing > 0:
                loading = False
                try:
                    keep = [
                        lm for lm in common.mm().current_loaded_models
                        if lm.model is not None and (
                            lm.model is self.patcher
                            or self.patcher.is_clone(lm.model)
                            or getattr(lm.model, "evicted_last", False)
                        )
                    ]
                    common.make_room(headroom + missing, dev, keep)
                    self._snapshot_torch_cpu()
                    loading = True
                    with torch.inference_mode():
                        self._fast_load_torch_unet(dev)
                        # +1 MiB triggers reForge's full_load branch (strict >), avoiding lowvram tail-module loops.
                        loaded = self.patcher.partially_load(dev, extra_memory=missing + (1 << 20))
                    self._torch_partial = self._torch_partial or loaded > 0
                except BaseException as e:
                    if loading:
                        self._rollback_torch_load()
                    if not isinstance(e, Exception):
                        raise
                    _clear_traceback(e)
                    if not loading:
                        self._unload_torch_weights(force=True)
                    logger.debug(f"Couldn't load PyTorch UNet beside the {self.model.name}: {e}")
                    self.fall_back(f"Couldn't run a side call beside the {self.model.name} ({e})", _unexpected(e))
            elif self._torch_partial and common.free(dev, at_least=headroom) < headroom:
                keep = [
                    lm for lm in common.mm().current_loaded_models
                    if lm.model is not None and (
                        lm.model is self.patcher
                        or self.patcher.is_clone(lm.model)
                        or getattr(lm.model, "evicted_last", False)
                    )
                ]
                common.make_room(headroom, dev, keep)
            raise _CallTorch()
        self._enter_torch_mode(why, shape)
        raise _CallTorch()

    def _enter_torch_mode(self, why: str, shape: tuple):
        prev = self.model
        name = prev.name if prev is not None else "engine"
        if not self._torch_logged:
            self._torch_logged = True
            logger.info(f"Pausing the {name} for PyTorch while mixed sizes run ({why}); TensorRT will resume once sizes settle.")
        if self.schedule is not None:
            self.schedule.deep.clear()
            self.schedule._missed.clear()
        self._hold(self.patcher)
        for m in list(_models.values()):
            if m.loaded or m.opened:
                _unload(m, keep=m.loaded)
        self.model = None
        self._synced.clear()
        self._torch_mode = True
        self._torch_eval_id = self._eval_id
        self._torch_eval_shapes = list(self._cur_eval_shapes) if self._cur_eval_id == self._eval_id else []
        if shape not in self._torch_eval_shapes:
            self._torch_eval_shapes.append(shape)
        self._torch_joint_key = None
        self._torch_stable_evals = 0
        dev = common.device()
        missing = max(0, self.patcher.model_size() - self.patcher.loaded_size())
        need_act = max(
            self._torch_act_bytes((self.shape[0], self.shape[2], self.shape[3], 0)),
            self._torch_act_bytes(shape),
        )
        act_room = common.padded(need_act)
        if not self._torch_partial and missing > 0:
            loading = False
            try:
                keep = _loaded_entries(self.patcher)
                had = common.make_room(missing + act_room, dev, keep)
                avail = max(0, had - act_room)
                budget = min(avail, missing)
                if budget > 0:
                    self._snapshot_torch_cpu()
                    extra = missing + (1 << 20) if budget >= missing else budget
                    loading = True
                    with torch.inference_mode():
                        if budget >= missing:
                            self._fast_load_torch_unet(dev)
                        loaded = self.patcher.partially_load(dev, extra_memory=extra)
                    self._torch_partial = self._torch_partial or loaded > 0
            except BaseException as e:
                if loading:
                    self._rollback_torch_load()
                if not isinstance(e, Exception):
                    raise
                _clear_traceback(e)
                if not loading:
                    self._unload_torch_weights(force=True)
                logger.debug(f"Couldn't load PyTorch UNet while {name} was paused: {e}")
                self.fall_back(f"Couldn't run PyTorch while {name} was paused ({e})", _unexpected(e))
        elif self._torch_partial and common.free(dev, at_least=act_room) < act_room:
            common.make_room(act_room, dev, _loaded_entries(self.patcher))

    @contextlib.contextmanager
    def _park_tracker(self):
        tracker = self.tracker
        if tracker is None:
            yield
            return
        dev = tracker.device
        tracker._jumps.wait()
        tracker._statuses.wait()
        tracker._constants.clear()
        moved = []

        def park(obj, attr):
            t = getattr(obj, attr, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                setattr(obj, attr, t.cpu())
                moved.append((obj, attr))

        park(tracker, "tables")
        for entry in (*tracker.history, tracker.current):
            if entry is not None:
                for attr in ("x", "sigma", "denoised", "uncond", "uncond_rows", "removed", "reason"):
                    park(entry, attr)
        last_keys = []
        for k, t in list(tracker._last.items()):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                tracker._last[k] = t.cpu()
                last_keys.append(k)
        ok = False
        try:
            yield
            ok = True
        finally:
            if ok and self.tracker is tracker:
                for obj, attr in moved:
                    t = getattr(obj, attr, None)
                    if isinstance(t, torch.Tensor):
                        setattr(obj, attr, t.to(dev))
                for k in last_keys:
                    if k in tracker._last and isinstance(tracker._last[k], torch.Tensor):
                        tracker._last[k] = tracker._last[k].to(dev)

    def _resume_from_torch_mode(self, record, x, padded: int):
        self._torch_mode = False
        self._torch_joint_key = None
        self._torch_stable_evals = 0
        self._torch_resumes += 1
        self._anchor = None
        self._last_anchor_eval = -1
        self._evicts_anchor.clear()
        self._engine_eval.clear()
        logger.info(f"Resuming TensorRT on the {record.arch.name} {record.label}.")
        try:
            with self._park_tracker():
                self._unload_torch_weights()
                target = model_for(record)
                _fits(target, self, x, unet_batch=padded)
                self.use(record, (padded, x.shape[2], x.shape[3]))
                _make_room_for_activations(self.model, self.patcher, x, unet_batch=padded)
        except Exception as e:
            failed = _models.get(record.key)
            if failed is not None and failed.opened and not failed.loaded:
                _unload(failed, keep=False)
            self.fall_back(f"Couldn't resume the {record.arch.name} {record.label} ({e})", _unexpected(e))

    def use(self, record, call_shape=None):
        global _use_tick
        model = model_for(record)
        want_shallow = bool(self.schedule is not None and model.hacks is not None and not model.shallow_failed)
        if model is not self.model or not model.loaded:
            was_loaded = model.loaded
            if not was_loaded:
                self._synced.pop(record.key, None)
            self._hold()
            for other in list(_models.values()):
                if other is not model and other.opened and not other.loaded:
                    _unload(other, keep=False)
            others = sorted(
                (other for other in _models.values() if other is not model and other.loaded),
                key=lambda m: _use_order.get(m.record.key, 0),
            )
            base_need = self._need_for(model, want_shallow, call_shape) if (others or was_loaded) else 0
            can_reuse_arena = was_loaded and not (want_shallow and not model.shallow_loaded) and model.engine is not None

            def _cur_need() -> int:
                if not can_reuse_arena:
                    return base_need
                arena = model.engine.arena.memory
                return max(0, base_need - (arena.numel() if arena is not None else 0))

            need = _cur_need()
            need_room = need > 0 and (not was_loaded or common.free(model.load_device, at_least=need) < need)
            if need_room and others:
                res = common.reserve()
                while others:
                    need = _cur_need()
                    if need <= 0 or common.obtainable(model.load_device, keep=[model, self.patcher], engines=False, at_least=need + res) - res >= need:
                        break
                    _unload(others.pop(0))
            need = _cur_need()
            if was_loaded and need_room and need > 0 and common.free(model.load_device, at_least=need) < need:
                keep_non_evict = [
                    lm for lm in common.mm().current_loaded_models
                    if lm.model is not None and (
                        lm.model is model
                        or lm.model is self.patcher
                        or self.patcher.is_clone(lm.model)
                        or getattr(lm.model, "evicted_last", False)
                    )
                ]
                common.make_room(need, model.load_device, keep_non_evict)
            model.weights = self.weights
            model.want_shallow = want_shallow
            with torch.inference_mode(), _keeping(self.patcher):
                common.register(model)
            if was_loaded and _use_order.get(record.key) != _use_tick:
                refits.used(record.key)
            self.model = model
            self._hold(model, self.patcher)
        _last_used[self.arch_id] = record.key
        _use_tick += 1
        _use_order[record.key] = _use_tick
        uuid = getattr(self.patcher, "patches_uuid", None)
        if uuid is None or self._synced.get(record.key) != (uuid, model.shallow_loaded) or not model.applied.get("unet"):
            model.sync()
            if uuid is not None:
                self._synced[record.key] = (uuid, model.shallow_loaded)
        if self.schedule is not None and model.hacks and not model.shallow_loaded and not model.shallow_failed:
            model.want_shallow = True
            try:
                with torch.inference_mode():
                    model.load_shallow()
            except Exception as e:
                notify(f"Couldn't load the Speed Hacks part of the {model.name}, so it runs without Speed Hacks: {e}")
            else:
                if uuid is not None and model.shallow_loaded:
                    self._synced[record.key] = (uuid, True)
        return model

    def preferred(self) -> set:
        if self.model is not None and self.model.loaded:
            key = self.model.record.key
            return {i for i, (r, _) in enumerate(self.slots) if r.key == key}
        key = _last_used.get(self.arch_id)
        if key is None:
            return set()
        keys = {key} | {r.key for r, _ in self.slots if (m := _models.get(r.key)) is not None and m.loaded}
        return {i for i, (r, _) in enumerate(self.slots) if r.key in keys}

    def _hold(self, *models):
        if self._keep is not None:
            with contextlib.suppress(ValueError):
                _kept.remove(self._keep)
            self._keep = None
        if models:
            self._keep = tuple(models)
            _kept.append(self._keep)

    def _watch_apply_model(self, model):
        # fresh.Tracker takes x and sigma from apply_model: in FP32, and as the UNet gets them. A wrapper left by a pass
        # that didn't end is removed first, so wrappers don't pile up.
        stale = model.__dict__.get("apply_model")
        if hasattr(stale, "_trt_before"):
            self._put_apply_model(model, stale._trt_before)
        before = model.__dict__.get("apply_model")
        original = model.apply_model

        def apply_model(*args, **kwargs):
            if self.schedule is not None and self.tracker is not None and not self.schedule.idle and not self.schedule.done:
                with self._speed_hacks():
                    x = args[0] if args else kwargs.get("x")
                    t = args[1] if len(args) > 1 else kwargs.get("t")
                    self.tracker.see(x, t)
            return original(*args, **kwargs)

        apply_model._trt_before = before
        model.apply_model = apply_model
        self._watching = True

    @contextlib.contextmanager
    def _speed_hacks(self, rethrow_oom: bool = False):
        from . import engine as engine_mod

        # Speed Hacks' own work: an error in it turns them off for the rest of the pass, and the engine keeps running
        engine_mod.LOGGER.clear()
        failures = gpu.failures()
        try:
            yield
        except torch.cuda.OutOfMemoryError:
            if rethrow_oom:
                raise
            self.drop_speed_hacks("Speed Hacks ran out of VRAM")
        except Exception as e:
            if gpu.failures() != failures or engine_mod.LOGGER.out_of_memory:
                if rethrow_oom:
                    raise torch.cuda.OutOfMemoryError(str(e)) from None
                self.drop_speed_hacks("Speed Hacks ran out of VRAM")
            else:
                self.drop_speed_hacks(f"Speed Hacks failed ({e})", _unexpected(e))

    def drop_speed_hacks(self, reason: str, error: Exception = None):
        notify(f"{reason}, so the rest of this pass runs without them.", error)
        if self.schedule is not None:
            self.schedule.release()
        if self.tracker is not None:
            self.tracker.release()
        self.schedule = self.tracker = None

    @staticmethod
    def _put_apply_model(model, value):
        if value is None:
            model.__dict__.pop("apply_model", None)
        else:
            model.apply_model = value

    def _restore(self):
        self.patcher.model.__dict__.pop("memory_required", None)
        if self._watching:
            self._watching = False
            current = self.patcher.model.__dict__.get("apply_model")
            if hasattr(current, "_trt_before"):
                self._put_apply_model(self.patcher.model, current._trt_before)
        self._hold()
        self._torch_partial = False
        self._torch_fast_loaded = False
        self._no_coexist = False
        self._torch_snap = None
        self._torch_mode = False
        self._torch_joint_key = None
        if self.model is not None and self.model.loaded:
            self.model.engine.release_activations()
        else:
            _release_unet_arena(common.device())
        if self.schedule is not None:
            self.schedule.release()
        if self.tracker is not None:
            self.tracker.release()
        self.schedule = self.tracker = None
        for model in _models.values():
            if model.weights is self.weights:
                model.weights = None  # don't keep this checkpoint alive
        if self.weights is not None:
            self.weights.params.clear()

    def _hint_batching(self):
        # reForge/Neo decides how many conds to batch into one UNet call from the UNet's memory_required(). Overridden for
        # the pass: 0 for any batch some profile runs while TensorRT is active (the engine's memory is already
        # allocated), the model's real estimate while paused in _torch_mode, and infinite whenever no profile covers
        # the combined batch/context so the sampler splits it into profile-compatible calls (allowing TensorRT to resume).
        shapes = self.shapes
        model = self.patcher.model
        base_fn = getattr(type(model), "memory_required", None)
        base_takes_cond = False
        if callable(base_fn):
            try:
                params = list(inspect.signature(base_fn).parameters.values())
                base_takes_cond = len(params) >= 3 or any(
                    p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params
                )
            except (TypeError, ValueError):
                base_takes_cond = False

        def memory_required(input_shape, cond_shapes={}):
            b, _, h, w = input_shape[:4]
            lengths = [s[1] for s in cond_shapes.get("c_crossattn", ()) if len(s) > 1]
            context = math.lcm(*lengths) if lengths else profiles.CONTEXT_PER_CHUNK
            if profiles.select(shapes, b, h, w, context) is None:
                return float("inf")
            if self._torch_mode and callable(base_fn):
                return base_fn(model, input_shape, cond_shapes) if base_takes_cond else base_fn(model, input_shape)
            return 0

        model.memory_required = memory_required

    def end(self):
        self._unload_torch_weights()
        self._restore()
        common.memory("after the sampling pass")
        from .engine import profile_report

        profile_report("sampling pass")
        hacks = self.started
        if hacks is not None and hacks.shallow_calls:
            logger.info(f"Speed Hacks ran {hacks.shallow_calls} of {hacks.shallow_calls + hacks.full_calls} UNet calls on the shallow engine.")
            if hacks.checks:
                logger.debug("Speed Hacks checks (call, age, error): " + ", ".join(f"{i} {a} {e:.4f}" for i, a, e in hacks.checks))
            _note_speed_hacks(hacks)

    def fall_back(self, reason: str, error: Exception = None):
        self.fallback = reason
        notify(f"{reason}, so the rest of this pass runs on PyTorch.", error)
        if self._torch_fast_loaded:
            self._unload_torch_weights()
        self._restore()
        _load_torch_unet(self.patcher, self.shape)
        raise Fallback(reason)

    def run(self, x, timesteps, context, y, cond_or_uncond=None):
        b, c, h, w = x.shape
        if c != self.channels or context.shape[2] != self.context_dim:
            self.fall_back(f"The UNet was given {c} input channels and {context.shape[2]}-wide prompts, but the engine takes {self.channels} and {self.context_dim}")
        ctx_len = context.shape[1]
        cur_shape = (b, h, w, ctx_len)
        if self._deferred:
            cand = (max(b, self._x_shape[0] * 2), c, h, w) if (h, w) == self._x_shape[2:] else tuple(x.shape)
            if not self._deferred_seen or cand[0] * cand[2] * cand[3] >= self.shape[0] * self.shape[2] * self.shape[3]:
                self.shape = cand
                self._deferred_seen = True
        resumed_choice = None
        if self._torch_mode:
            if self._eval_id != self._torch_eval_id:
                prev_shapes = self._torch_eval_shapes
                self._torch_eval_id = self._eval_id
                self._torch_eval_shapes = [cur_shape]
                joint = self._joint_record(prev_shapes)
                if joint is not None:
                    if joint.key == self._torch_joint_key:
                        self._torch_stable_evals += 1
                    else:
                        self._torch_joint_key = joint.key
                        self._torch_stable_evals = 1
                    multi = len({(sh, sw) for _, sh, sw, _ in prev_shapes}) > 1
                    if self._remaining_steps is None:
                        enough_left = True
                    elif multi:
                        enough_left = self._remaining_steps >= 5
                    elif self.schedule is not None:
                        enough_left = self._remaining_steps >= 6
                    else:
                        enough_left = self._remaining_steps * self._evals_per_step >= 16
                    if self._torch_resumes < 2 and enough_left and (multi or self._torch_stable_evals >= 2):
                        fit_cur = profiles.select(joint.profile_shapes, b, h, w, ctx_len)
                        if fit_cur is not None:
                            self._resume_from_torch_mode(joint, x, fit_cur[1])
                            slot_idx = next(i for i, (r, j) in enumerate(self.slots) if r.key == joint.key and j == fit_cur[0])
                            resumed_choice = (slot_idx, fit_cur[1], fit_cur[2])
                        else:
                            self._torch_joint_key = None
                            self._torch_stable_evals = 0
                            raise _CallTorch()
                    else:
                        raise _CallTorch()
                else:
                    self._torch_joint_key = None
                    self._torch_stable_evals = 0
                    raise _CallTorch()
            else:
                if cur_shape not in self._torch_eval_shapes:
                    self._torch_eval_shapes.append(cur_shape)
                raise _CallTorch()
        if self._eval_id != self._cur_eval_id:
            self._cur_eval_id = self._eval_id
            self._cur_eval_shapes = []
        if cur_shape not in self._cur_eval_shapes:
            self._cur_eval_shapes.append(cur_shape)
        choice = resumed_choice if resumed_choice is not None else self._select(b, h, w, ctx_len)
        anchor_recent = (
            self.model is not None
            and self.model.loaded
            and self._anchor is not None
            and (self._eval_id - self._last_anchor_eval) <= 2
        )
        if choice is None:
            if anchor_recent and (h, w) != self._anchor[1:3]:
                self._call_torch(f"no profile runs {w * 8}x{h * 8}", cur_shape)
            tokens = ctx_len // profiles.CONTEXT_PER_CHUNK * profiles.TOKENS_PER_CHUNK
            self.fall_back(
                f"No profile runs {w * 8}x{h * 8} with a UNet batch of {b} and {tokens}-token prompts "
                "(reForge pads the prompt and negative prompt to the same length)"
            )
        index, padded, repeats = choice
        record, profile = self.slots[index]
        prev_model = self.model
        if prev_model is None or prev_model.record.key != record.key or not prev_model.loaded:
            target = model_for(record)
            want_shallow = bool(self.schedule is not None and target.hacks is not None and not target.shallow_failed)
            upgraded = False
            was_anchor_recent = anchor_recent
            returning_to_anchor = prev_model is not None and self._covers_anchor(record) and not self._covers_anchor(prev_model.record)
            stay = None
            if (
                anchor_recent
                and not target.loaded
                and not returning_to_anchor
                and self._would_evict_current(target, want_shallow, (padded, h, w))
            ):
                stay = profiles.select(prev_model.record.profile_shapes, b, h, w, ctx_len)
                if stay is not None:
                    record = prev_model.record
                    profile, padded, repeats = stay
                elif b * h * w <= self._anchor[4] or record.key in self._evicts_anchor:
                    self._call_torch(f"{w * 8}x{h * 8} would evict it", cur_shape)
                else:
                    upgraded = True
                    self._evicts_anchor.add(prev_model.record.key)
                    self._anchor = None
            if stay is None:
                need_load = not target.loaded or (want_shallow and not target.shallow_loaded)
                active_before = {
                    m.record.key: m for m in _models.values()
                    if m.loaded and (m is prev_model or (self._eval_id - self._engine_eval.get(m.record.key, -99)) <= 2)
                }
                try:
                    if need_load and prev_model is not None and prev_model.loaded:
                        prev_model.engine.release_activations()
                    if need_load:
                        self._unload_torch_weights()
                    elif self._torch_partial:
                        need_t = self._need_for(target, want_shallow, (padded, h, w))
                        arena_t = target.engine.arena.memory if target.engine is not None else None
                        if arena_t is not None:
                            need_t = max(0, need_t - arena_t.numel())
                        res_t = common.reserve()
                        if common.obtainable(target.load_device, keep=[target, self.patcher], engines=False, at_least=need_t + res_t) - res_t < need_t:
                            self._no_coexist = True
                            self._unload_torch_weights()
                    if prev_model is None:
                        _fits(target, self, x, unet_batch=padded)
                    self.use(record, (padded, h, w))
                    if prev_model is None:
                        _make_room_for_activations(self.model, self.patcher, x, unet_batch=padded)
                except Exception as e:
                    failed = _models.get(record.key)
                    if failed is not None and failed.opened and not failed.loaded:
                        _unload(failed, keep=False)
                    self.fall_back(f"Couldn't load the {record.arch.name} {record.label} ({e})", _unexpected(e))
                evicted = [k for k, m in active_before.items() if k != record.key and not m.loaded]
                if not upgraded and evicted and was_anchor_recent and self._anchor is not None:
                    if self._covers_anchor(record) or b * h * w > self._anchor[4]:
                        self._evicts_anchor.update(evicted)
                    else:
                        self._evicts_anchor.add(record.key)
                if not anchor_recent and prev_model is not None and prev_model is not self.model:
                    self._anchor = None
                    self._evicts_anchor.clear()
        timesteps = timesteps.reshape(-1)
        feed = {
            "sample": x,
            "timesteps": timesteps if timesteps.shape[0] == b else timesteps.expand(b),
            "encoder_hidden_states": context.repeat(1, repeats, 1) if repeats > 1 else context,
        }
        if y is not None and self.takes_y:
            feed["y"] = y
        if padded > b:
            feed = {k: v.repeat(-(-padded // b), *([1] * (v.dim() - 1)))[:padded] for k, v in feed.items()}
        stream = torch.cuda.current_stream(self.model.load_device).cuda_stream
        hack = None
        if self.schedule is not None and not self.schedule.idle and self.model.shallow_loaded:
            if self.schedule.done:
                self.schedule.full_calls += 1
            else:
                planned_shallow = False
                with self._speed_hacks():
                    prompt = (feed["encoder_hidden_states"], feed.get("y"))
                    key = (tuple(feed["sample"].shape), tuple(cond_or_uncond or ()), tuple(prompt[0].shape), tuple(_prompts))
                    slot, deep, origin, stale, kind = self.schedule.call(key)
                    planned_shallow = deep is not None
                    entry = self.tracker.call(feed["sample"], b)
                    if kind is not None and (not self.tracker.valid(stale[1]) or (entry is not None and not self.tracker.anchored(entry, stale[1]))):
                        stale = kind = None
                    if deep is not None and not (self.tracker.safe(entry, origin) and (entry is not None or self.schedule.step_known())):
                        # Noise the fit can't remove for this call, or for a call the fit can't follow, noise not yet ruled
                        # out. It runs in full, and a tracked one may also probe (fresh.Tracker.probe) when on the same path.
                        probe = (
                            entry is not None
                            and self.tracker.valid(origin)
                            and self.tracker.anchored(entry, origin)
                            and self.schedule.can_probe()
                            and self.tracker.probe(self.schedule.index)
                        )
                        stale, kind = ((deep, origin), "retry") if probe else (None, None)
                        deep = origin = None
                        planned_shallow = False
                        self.schedule.ran_full()
                    if deep is not None:
                        self.schedule.compare(slot, key, prompt)
                    hack = [slot, key, deep, origin, stale, kind, entry, cond_or_uncond, b]
                if hack is None and planned_shallow and self.started is not None:
                    self.started.ran_full()
        vol = b * h * w
        out = self._infer_with_room(feed, profile, stream, hack, vol=vol)
        # A shallow call whose Speed Hacks failed ran on the full engine instead
        shallow = bool(hack) and hack[2] is not None and self.schedule is not None
        engine_key = self.model.record.key
        if not shallow and engine_key not in self.checked:
            self.checked.add(engine_key)
            if not torch.isfinite(out).all():
                self.fall_back("The engine produced NaN/Inf values")
        if shallow and engine_key not in self.shallow_checked:
            self.shallow_checked.add(engine_key)
            if not torch.isfinite(out).all():
                if self.started is not None:
                    self.started.ran_full()
                self.drop_speed_hacks("Speed Hacks produced NaN/Inf values")
                out = self._infer_with_room(feed, profile, stream, vol=vol)
        self._engine_eval[engine_key] = self._eval_id
        expired = self._anchor is not None and (self._eval_id - self._last_anchor_eval) > 2
        if (
            self._anchor is None
            or expired
            or vol > self._anchor[4]
            or (vol == self._anchor[4] and (h, w) == self._anchor[1:3])
        ):
            if expired:
                self._evicts_anchor.clear()
            self._anchor = (b, h, w, ctx_len, vol)
            self._last_anchor_eval = self._eval_id
        elif (h, w) == self._anchor[1:3]:
            self._last_anchor_eval = self._eval_id
        return out[:b].to(x.dtype)

    def _infer_with_room(self, feed, profile, stream, hack=None, vol: int = 0):
        try:
            return self._infer(feed, profile, stream, hack)
        except torch.cuda.OutOfMemoryError as e:
            _clear_traceback(e)
            dev = self.model.load_device
            before_free = common.free(dev)
            self.model.engine.release_activations()
            unloaded_torch = self._torch_partial
            if self._torch_partial and not self._torch_mode:
                self._no_coexist = True
            self._unload_torch_weights()
            active_other = [
                m for m in _models.values()
                if m is not self.model and m.loaded and (self._eval_id - self._engine_eval.get(m.record.key, -99)) <= 2
            ]
            # Stage 1: evict non-engine models (CLIP/VAE) and stale engines while keeping active engines of this pass
            # resident; only touch active engines in Stage 2 if retrying after freeing everything else still OOMs.
            if active_other:
                with _evictable():
                    freed = common.inference_mode(common.mm().free_memory)(
                        sys.maxsize // 2, dev, keep_loaded=_loaded_entries(self.model, self.patcher, *active_other)
                    )
                common.mm().soft_empty_cache(force=True)
                if unloaded_torch or freed or common.free(dev) > before_free:
                    try:
                        return self._infer(feed, profile, stream, hack, retry=False)
                    except torch.cuda.OutOfMemoryError as e1:
                        _clear_traceback(e1)
                        self.model.engine.release_activations()
            active_before = {m.record.key: m for m in active_other if m.loaded}
            with _evictable():
                common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, dev, keep_loaded=_loaded_entries(self.model, self.patcher))
            common.mm().soft_empty_cache(force=True)
            evicted = [k for k, m in active_before.items() if not m.loaded]
            if evicted and self._anchor is not None and (self._eval_id - self._last_anchor_eval) <= 2:
                if self._covers_anchor(self.model.record) or vol > self._anchor[4]:
                    self._evicts_anchor.update(evicted)
                else:
                    self._evicts_anchor.add(self.model.record.key)
            try:
                return self._infer(feed, profile, stream, hack, retry=True)
            except torch.cuda.OutOfMemoryError as e:
                _clear_traceback(e)
                if hack and hack[2] is not None and self.schedule is not None and self.started is not None:
                    self.started.ran_full()
                if hack:
                    hack.clear()
                self.fall_back("The engine ran out of VRAM")

    def _infer(self, feed, profile, stream, hack=None, retry: bool = False):
        from . import engine as engine_mod

        # hack: [slot, Schedule key, deep feature or None, index of the call that computed it, (deep feature, index) to
        # check or probe this full call against or None, "check"/"probe"/"retry"/None, fresh.Entry or None, reForge's
        # cond_or_uncond, rows before padding]
        slot, key, deep, origin, stale, kind, entry, cond_or_uncond, b = hack or (None,) * 9
        engine_mod.LOGGER.clear()
        failures = gpu.failures()
        try:
            if deep is not None and self.schedule is not None:
                # A shallow call that fails runs in full below
                with self._speed_hacks(rethrow_oom=not retry):
                    outputs = self.model.shallow.infer({**feed, "deep": deep, **self.tracker.inputs(entry, origin, feed["sample"])}, profile, stream)
                    if entry is not None:
                        self.tracker.after_shallow(entry, outputs, origin)
                        self.tracker.after_call(entry, outputs["latent"][:b], cond_or_uncond)
                    return outputs["latent"]
                if self.started is not None:
                    self.started.ran_full()
                deep = stale = entry = None
                if hack is not None:
                    hack.clear()
                if retry:
                    self.model.engine.release_activations()
                    common.mm().soft_empty_cache(force=True)
            engine_mod.LOGGER.clear()
            failures = gpu.failures()
            outputs = self.model.engine.infer(feed, profile, stream)
            out = outputs["latent"]
            if slot is not None and "deep" in outputs and self.schedule is not None:
                with self._speed_hacks():
                    self.schedule.keep(slot, key, outputs["deep"], (feed["encoder_hidden_states"], feed.get("y")))
                    if kind is not None and entry is not None:
                        inputs = {**feed, "deep": stale[0], **self.tracker.inputs(entry, stale[1], feed["sample"])}
                        shallow = self.model.shallow.infer(inputs, profile, stream)
                        self.tracker.status(shallow)
                        if kind == "check":
                            self.tracker.add_check(entry, out[:b], shallow["latent"][:b], self.schedule.index - stale[1])
                        elif kind == "probe":
                            self.schedule.mark_probed()
                    if entry is not None:
                        self.tracker.after_call(entry, out[:b], cond_or_uncond)
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            if gpu.failures() != failures or engine_mod.LOGGER.out_of_memory:
                raise torch.cuda.OutOfMemoryError(str(e)) from None
            self.fall_back(f"The engine failed ({e})", _unexpected(e))


def _clear_traceback(error):
    stack, seen = [error], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        for linked in (cur.__cause__, cur.__context__):
            if linked is not None:
                stack.append(linked)
        cur.__traceback__ = cur.__cause__ = cur.__context__ = None


def _engine_records(patcher, x):
    chosen = setting()
    if chosen not in (AUTOMATIC, SPEED_HACKS):
        return None, False
    arch = arch_mod.detect(patcher)
    records = store.engines(arch.id) if arch is not None else []
    if not records:  # with "Automatic" that's normal (e.g. a refiner of another architecture), so no warning
        if chosen == SPEED_HACKS and arch is None:
            notify("TensorRT engines can't run this model, so it runs on PyTorch.")
        elif chosen == SPEED_HACKS:
            notify(f"There are no {arch.name} engines yet, so it runs on PyTorch. Build them in the TensorRT tab.")
        return None, False
    reason = _unsupported(patcher)
    if reason:
        notify(f"TensorRT engines can't apply {reason}, so this pass runs on PyTorch.")
        return None, False
    b, _, h, w = x.shape
    if not _wraps_unet(patcher) and not profiles.covers_size([s for r in records for s in r.profile_shapes], h, w, b):
        notify(f"No {arch.name} profile covers {w * 8}x{h * 8} at batch size {b}, so this pass runs on PyTorch. Add one in the TensorRT tab.")
        return None, False
    stale_graph = sum(r.stale_graph for r in records)
    if stale_graph:
        notify(f"{stale_graph} of the {arch.name} engines were built with an older UNet graph that can produce artifacts on TensorRT 11. Build again in the TensorRT tab to update them.")
    hacks = chosen == SPEED_HACKS and any(r.speed_hacks for r in records)
    stale = sum(r.stale_speed_hacks for r in records) if chosen == SPEED_HACKS else 0
    failed = sum(not r.speed_hacks for r in records) - stale if chosen == SPEED_HACKS else 0
    if stale:
        notify(f"{stale} of the {arch.name} engines have a Speed Hacks part from another version of this extension, so they run without Speed Hacks until they're built again in the TensorRT tab.")
    if failed:
        retry = " Building them again in the TensorRT tab retries once." if any(r.retry_speed_hacks for r in records) else ""
        notify(f"Speed Hacks couldn't be built for {failed} of the {arch.name} engines, so they run without them.{retry}")
    return records, hacks


def _begin(records, hacks, patcher, x, prepare):
    global _no_free
    mm = common.mm()
    with torch.inference_mode():
        _unload_torch_unet(patcher)
        prev_state = mm.vram_state
        # reForge's sampling_prepare applies the UNet's patches; NO_VRAM keeps its weights in RAM while it does
        mm.vram_state = mm.VRAMState.NO_VRAM
        _no_free += 1
        try:
            prepare(patcher, x)
        finally:
            _no_free -= 1
            mm.vram_state = prev_state
        p = Pass(records, patcher, x, hacks)
        if p._deferred:
            # Wrappers such as MultiDiffusion rewrite x into smaller tiles before the UNet runs: wait for the first
            # actual UNet call so we load the tile engine directly instead of loading and immediately evicting the canvas one.
            p._hold(patcher)
            return p
        record = None
        try:
            b, _, h, w = x.shape
            prefer = p.preferred()
            # Load the engine for the likeliest first call: cond and uncond batched, shortest prompts
            choice = profiles.select(p.shapes, 2 * b, h, w, profiles.CONTEXT_PER_CHUNK, prefer) or profiles.select(p.shapes, b, h, w, profiles.CONTEXT_PER_CHUNK, prefer)
            index = choice[0] if choice else next(i for i, s in enumerate(p.shapes) if profiles.covers_size([s], h, w, b))
            record = p.slots[index][0]
            _fits(model_for(record), p, x)
            model = p.use(record)
            _make_room_for_activations(model, patcher, x)
        except Exception as e:
            if record is not None:
                failed = _models.get(record.key)
                if failed is not None and failed.opened and not failed.loaded:
                    _unload(failed, keep=False)
            notify(f"Couldn't run the engine ({e}), so this pass runs on PyTorch.", _unexpected(e))
            p.end()
            _load_torch_unet(patcher, (x.shape[0] * 2, *x.shape[1:]))
            return None
    return p


def _sizes(x) -> str:
    return f"{x.shape[3] * 8}x{x.shape[2] * 8}" + (f", batch size {x.shape[0]}" if x.shape[0] > 1 else "")


def _unexpected(error):  # the error, for its traceback, unless a RuntimeError: those messages are complete on their own
    return None if isinstance(error, RuntimeError) else error


def _fits(model: TrtModel, p: Pass, x, unet_batch: int = None):
    shallow = bool(model.hacks and p.schedule is not None and not model.shallow_failed)
    key = _took_key(model, x, unet_batch=unet_batch)
    calls = max(call_bytes(model, p.patcher, x, unet_batch=unet_batch), model.took.get(key, 0))
    held = model.loaded_size()
    if model.loaded:
        engine = held + (model._serialized_size("shallow") if shallow and not model.shallow_loaded else 0)
    else:
        engine = model.full(shallow, model._kept(p.weights))
    need = held + common.padded(engine - held + calls)  # margin only on what isn't in VRAM yet
    res = common.reserve()
    room = common.obtainable(model.load_device, keep=[model], at_least=max(0, need - held + res)) - res + held
    logger.info(
        f"The {model.name} at {_sizes(x)} needs {common.gib(need)} of VRAM ({common.gib(engine)} for the engine, {common.gib(calls)} "
        f"to run it, {common.gib(need - engine - calls)} margin); {common.gib(room)} can be freed."
    )
    if need > room:
        raise DoesNotFit(f"it needs {common.gib(need)} of VRAM at {_sizes(x)}, and at most {common.gib(room)} can be freed")


def _make_room_for_activations(model: TrtModel, patcher, x, unet_batch: int = None):
    try:
        takes = _call_bytes(model, x, unet_batch=unet_batch)
    except Exception as e:
        logger.debug(f"Couldn't ask TensorRT what running the {model.name} takes: {e}")
        return
    if not takes:
        return
    counted = call_bytes(model, patcher, x, unet_batch=unet_batch)
    if takes > counted:
        most = " (the most for any size its profile takes: it gave no figure for this one)" if model.engine.size_is_most else ""
        logger.debug(f"TensorRT takes {common.gib(takes)} to run the {model.name} at {_sizes(x)}{most}, more than the {common.gib(counted)} estimated.")
    need = common.padded(max(takes, counted))
    keep = _loaded_entries(model, patcher)
    free = common.make_room(need, model.load_device, keep)
    if free < need:
        with _evictable():
            free = common.make_room(need, model.load_device, keep)
    if free < need:
        model.took[_took_key(model, x, unet_batch=unet_batch)] = takes  # so _fits rejects these sizes before loading next time
        raise RuntimeError(f"running it at {_sizes(x)} needs {common.gib(need)} of VRAM besides the engine, and only {common.gib(free)} could be freed")


def _call_shapes(model: TrtModel, x, unet_batch: int = None):
    b, c, h, w = x.shape
    held = model.record.profile_shapes
    if unet_batch is not None:
        choice = profiles.select(held, unet_batch, h, w, profiles.CONTEXT_PER_CHUNK)
    else:
        choice = profiles.select(held, 2 * b, h, w, profiles.CONTEXT_PER_CHUNK) or profiles.select(held, b, h, w, profiles.CONTEXT_PER_CHUNK)
    if choice is None:
        return None
    profile, padded, _ = choice
    s = held[profile]
    shapes = {
        "sample": (padded, c, h, w),
        "timesteps": (padded,),
        "encoder_hidden_states": (padded, s["encoder_hidden_states"][1][1], s["encoder_hidden_states"][0][2]),
    }
    if "y" in s:
        shapes["y"] = (padded, s["y"][0][1])
    return profile, shapes


def _call_bytes(model: TrtModel, x, unet_batch: int = None) -> int:
    found = _call_shapes(model, x, unet_batch=unet_batch)
    if found is None:
        return 0
    profile, shapes = found
    act = model.engine.estimate_activation_bytes(profile, shapes, torch.cuda.current_stream(model.load_device).cuda_stream)
    io = model.engine.io_bytes()
    return act + sum(io.values())


def _note_torch_call(p: Pass, x, out, cond_or_uncond):
    if p.tracker is not None and p.tracker.current is not None and p.schedule is not None and not p.schedule.done:
        with p._speed_hacks():
            entry = p.tracker.call(x, x.shape[0])
            if entry is not None:
                p.tracker.after_call(entry, out[:x.shape[0]], cond_or_uncond)


def _wrap_forward(original):
    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        p = _pass
        if p is not None and p.fallback is None and self is p.module:
            try:
                if control is not None:
                    p.fall_back("TensorRT engines can't apply ControlNet")
                if any(v for g in ("patches", "patches_replace") for v in transformer_options.get(g, {}).values()):
                    p.fall_back("TensorRT engines can't apply UNet patches")
                if any(transformer_options.get(k) for k in _EXTRA_TRANSFORMER_OPTIONS):
                    p.fall_back("TensorRT engines can't apply UNet patches")
                if (transformer_options.get("wrappers") or {}).get("diffusion_model"):
                    p.fall_back("TensorRT engines can't apply UNet wrappers")
                return p.run(x, timesteps, context, y, transformer_options.get("cond_or_uncond"))
            except _CallTorch:
                cond_or_uncond = transformer_options.get("cond_or_uncond")
                try:
                    out = original(self, x, timesteps, context, y, control, transformer_options, **kwargs)
                    _note_torch_call(p, x, out, cond_or_uncond)
                    return out
                except torch.cuda.OutOfMemoryError as e:
                    _clear_traceback(e)
                    if p._torch_partial:
                        if not p._torch_mode:
                            p._no_coexist = True
                        p._unload_torch_weights()
                        try:
                            out = original(self, x, timesteps, context, y, control, transformer_options, **kwargs)
                            _note_torch_call(p, x, out, cond_or_uncond)
                            return out
                        except torch.cuda.OutOfMemoryError as e2:
                            _clear_traceback(e2)
                    try:
                        p.fall_back("Running a side call on PyTorch ran out of VRAM")
                    except Fallback:
                        pass
            except Fallback:
                pass
        return original(self, x, timesteps, context, y, control, transformer_options, **kwargs)

    forward._trt_original = original
    return forward


_latest_sigmas = None
_denoiser_state = None  # active CFGDenoiser.forward call, so a refiner switch inside it can start the new pass on that step


def _begin_step(p: Pass, index: int, total, steps, step, x, sigma):
    try:
        p.schedule.begin(index, total, steps, step, sigma)
        _prompts.clear()
        if p.schedule.done:
            p.tracker.release()
        elif not p.schedule.idle:
            p.tracker.begin(index, x)
    except Exception as e:
        p.drop_speed_hacks(f"Speed Hacks failed ({e})", e)


def _wrap_prepare(original):
    def sampling_prepare(unet, x, *args, **kwargs):
        global _pass, _latest_sigmas, _generation_thread
        end_pass()
        _generation_thread = threading.get_ident()
        if _denoiser_state is None:
            _latest_sigmas = None
        records, hacks = _engine_records(unet, x)
        if records is None:
            with _evictable():
                return original(unet, x, *args, **kwargs)
        _pass = _begin(records, hacks, unet, x, lambda u, v: original(u, v, *args, **kwargs))
        if _pass is not None and _denoiser_state is not None:
            _pass._eval_id = 1
            if _denoiser_state[2]:
                _pass._remaining_steps = max(0, int(_denoiser_state[2]) - int(_denoiser_state[3]))
            if _pass.schedule is not None:
                # Refiner switch mid-denoiser: get_sigmas doesn't run again, and the switch step is already in flight.
                if _latest_sigmas is not None:
                    _pass.schedule.sigmas = list(_latest_sigmas)
                _begin_step(_pass, *_denoiser_state)

    sampling_prepare._trt_original = original
    return sampling_prepare


def _wrap_cleanup(original):
    def sampling_cleanup(unet, *args, **kwargs):
        end_pass()
        return original(unet, *args, **kwargs)

    sampling_cleanup._trt_original = original
    return sampling_cleanup


def _wrap_close(original):
    # reForge/Neo closes the processing object however a generation ended. One that failed mid-pass never reached
    # sampling_cleanup, and its pass would keep the engine and UNet unevictable until the next generation.
    def close(self, *args, **kwargs):
        end_pass()
        return original(self, *args, **kwargs)

    close._trt_original = original
    return close


def _wrap_denoiser(original):
    def forward(self, x, sigma, *args, **kwargs):
        global _denoiser_state
        p = _pass
        if p is not None and p.fallback is None:
            p._eval_id += 1
        current_state = None
        if setting() in (AUTOMATIC, SPEED_HACKS):
            try:
                from modules import shared

                state = shared.state
                current_state = (int(self.step), self.total_steps, state.sampling_steps, state.sampling_step, x, sigma)
                if p is not None and p.fallback is None and state.sampling_steps:
                    step_now = int(state.sampling_step)
                    p._remaining_steps = max(0, int(state.sampling_steps) - step_now)
                    if p._last_step is None:
                        p._last_step, p._step_evals = step_now, 1
                    elif step_now == p._last_step:
                        p._step_evals += 1
                    else:
                        p._evals_per_step = max(1, p._step_evals)
                        p._last_step, p._step_evals = step_now, 1
            except Exception as e:
                if p is not None and p.schedule is not None and p.fallback is None:
                    p.drop_speed_hacks(f"Speed Hacks failed ({e})", e)
            else:
                if p is not None and p.schedule is not None and p.fallback is None:
                    _begin_step(p, *current_state)
        prev_state, _denoiser_state = _denoiser_state, current_state
        try:
            denoised = original(self, x, sigma, *args, **kwargs)
        finally:
            _denoiser_state = prev_state
        active = _pass
        if active is not None and active.schedule is not None and active.tracker is not None and active.tracker.current is not None:
            try:
                tiled = active.tracker.current.x is None
                # Timestep samplers (DDIM, PLMS, UniPC) get eps back
                checked = active.tracker.finish(denoised, x, sigma, eps=bool(getattr(self, "classic_ddim_eps_estimation", False)))
                if (active.tracker.measured or (tiled and active.tracker.noisy)) and active.schedule.kind == "probe":
                    active.schedule.mark_probed()
                if checked is not None:
                    active.schedule.checked(*checked)
            except Exception as e:
                active.drop_speed_hacks(f"Speed Hacks failed ({e})", e)
        return denoised

    forward._trt_original = original
    return forward


def _wrap_get_sigmas(original):
    # The schedule speed.Schedule places its start and end shares on. KDiffusionSampler.sample and sample_img2img call
    # get_sigmas after sampling_prepare, before the first denoiser call; a subclass's override that calls the base
    # class's returns last, and its sigmas are the ones sampled.
    def get_sigmas(self, p, steps, *args, **kwargs):
        global _latest_sigmas
        sigmas = original(self, p, steps, *args, **kwargs)
        try:
            values = [float(s) for s in sigmas.reshape(-1).tolist()]
        except Exception as e:
            _latest_sigmas = None
            if _pass is not None and _pass.schedule is not None and _pass.schedule.index is None:
                _pass.drop_speed_hacks(f"Speed Hacks failed ({e})", _unexpected(e))
            return sigmas
        _latest_sigmas = values
        if _pass is not None and _pass.schedule is not None and _pass.schedule.index is None:
            _pass.schedule.sigmas = values
        return sigmas

    get_sigmas._trt_original = original
    return get_sigmas


def _wrap_vae_decode(original):
    def decode(self, samples_in, *args, **kwargs):
        # Only on the generation's thread: reForge/Neo's progress endpoint decodes "Full" live previews on another
        # thread while this one is sampling on the UNet engine.
        vae_options = args[0] if args else kwargs.get("vae_options")
        if not vae_options and setting() in (AUTOMATIC, SPEED_HACKS) and threading.get_ident() == _generation_thread:
            try:
                pixels = vae_run.decode(self, samples_in)
                if pixels is not None:
                    return pixels
            except Exception as e:
                notify(f"The VAE decoder engines failed ({e}), so this decode runs on PyTorch.", e)
        return original(self, samples_in, *args, **kwargs)

    decode._trt_original = original
    return decode


def _wrap_vae_encode(original):
    def encode(self, pixel_samples):
        if setting() in (AUTOMATIC, SPEED_HACKS) and threading.get_ident() == _generation_thread:
            try:
                latents = vae_run.encode(self, pixel_samples)
                if latents is not None:
                    return latents
            except Exception as e:
                notify(f"The VAE encoder engines failed ({e}), so this encode runs on PyTorch.", e)
        return original(self, pixel_samples)

    encode._trt_original = original
    return encode


def _wrap_vae_reload(original):
    def wrapped(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            vae_run.invalidate_weights()

    wrapped._trt_original = original
    return wrapped


_installed = set()  # (owner, name) hooks already installed, so another extension wrapping on top doesn't cause double-wrapping


def _hook(owner, name, wrap):
    key = (owner, name)
    if key in _installed:
        return
    current = getattr(owner, name)
    if not hasattr(current, "_trt_original"):
        setattr(owner, name, wrap(current))
    _installed.add(key)


def _rebind(owner, name, wrap):  # also replaces references other modules took with `from ... import name`
    key = (owner, name)
    if key in _installed:
        return
    current = getattr(owner, name)
    if not hasattr(current, "_trt_original"):
        replacement = wrap(current)
        setattr(owner, name, replacement)
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if isinstance(namespace, dict) and namespace.get(name) is current:
                setattr(module, name, replacement)
    _installed.add(key)


_hook_failures = set()  # install_hooks runs every generation: log each failure once


# Per CFG denoiser call of a Speed Hacks pass, which scheduled prompt each prompt in the batch uses. For multi-entry
# schedules (e.g. [a|b]), reForge slices each step into a fresh dict/tensor view, so equal conds are canonicalized by
# value to a shared rep id; single-entry schedules take id() directly with no GPU work.
_prompts = []
_prompt_seen = {}  # id(cond) -> (cond, rep id), cleared at pass end, capped at 128
_prompt_reps = []  # [(rep id, cond), ...], capped at 32


def _same_cond(a, b) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_cond(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_cond(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and bool(torch.equal(a, b))
    if hasattr(a, "dtype") and hasattr(a, "shape") and hasattr(a, "tobytes"):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype == object:
            return all(_same_cond(x, y) for x, y in zip(a.flat, b.flat))
        if getattr(a.dtype, "hasobject", False):
            return all(_same_cond(tuple(x), tuple(y)) for x, y in zip(a.flat, b.flat))
        return bool((a == b).all())
    return bool(a == b)


def _cond_key(schedule, step) -> int:
    if len(schedule) == 1:
        return id(schedule[0].cond)
    cond = next((e.cond for e in schedule if step <= e.end_at_step), schedule[0].cond)
    cid = id(cond)
    entry = _prompt_seen.get(cid)
    if entry is not None and entry[0] is cond:
        return entry[1]
    for rep_id, rep in _prompt_reps:
        if _same_cond(cond, rep):
            if len(_prompt_seen) < 128:
                _prompt_seen[cid] = (cond, rep_id)
            return rep_id
    if len(_prompt_reps) < 32:
        _prompt_reps.append((cid, cond))
        if len(_prompt_seen) < 128:
            _prompt_seen[cid] = (cond, cid)
    return cid


def _note_prompts(schedules_of, step):
    # schedules_of() -> one list of ScheduledPromptConditioning per prompt
    if _pass is None or _pass.schedule is None or _pass.schedule.idle or _pass.schedule.done or _pass.fallback is not None:
        return
    try:
        _prompts.append(tuple(_cond_key(s, step) for s in schedules_of()))
    except Exception:
        _prompts.append(None)


def _wrap_reconstruct_cond_batch(original):
    def reconstruct_cond_batch(c, current_step):
        _note_prompts(lambda: c, current_step)
        return original(c, current_step)

    reconstruct_cond_batch._trt_original = original
    return reconstruct_cond_batch


def _wrap_reconstruct_multicond_batch(original):
    def reconstruct_multicond_batch(c, current_step):
        _note_prompts(lambda: [prompt.schedules for prompts in c.batch for prompt in prompts], current_step)
        return original(c, current_step)

    reconstruct_multicond_batch._trt_original = original
    return reconstruct_multicond_batch


def _needs_async_offload_fix(ldm_ops, mm) -> bool:
    # With --async-offload (NUM_STREAMS > 1), reForge's cast_bias_weight allocates cast/offloaded weight and bias
    # tensors on an offload stream and syncs current_stream onto it, without calling record_stream(current_stream).
    # Under --cuda-malloc's shared driver pool, freeing those temporaries on the offload stream lets the next layer
    # overwrite weights still being read by current_stream (corrupting large CLIP batches and alternating prompts).
    if getattr(mm, "NUM_STREAMS", 1) <= 1 or (ldm_ops, "cast_bias_weight") in _installed:
        return False
    cbw = getattr(ldm_ops, "cast_bias_weight", None)
    if not callable(cbw):
        return False
    orig = getattr(cbw, "_trt_original", cbw)
    try:
        cbw_src = inspect.getsource(orig)
    except (OSError, TypeError):
        return callable(getattr(mm, "get_offload_stream", None))
    if "get_offload_stream" not in cbw_src or "record_stream" in cbw_src:
        return False
    cast_to = getattr(mm, "cast_to", None)
    if callable(cast_to):
        try:
            if "record_stream" in inspect.getsource(cast_to):
                return False
        except (OSError, TypeError):
            pass
    return True


def _wrap_cast_bias_weight(original):
    def cast_bias_weight(s, *args, **kwargs):
        result = original(s, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and getattr(common.mm(), "NUM_STREAMS", 1) > 1:
            weight, bias = result
            orig_w = getattr(s, "weight", None)
            if isinstance(weight, torch.Tensor) and weight is not orig_w and weight.is_cuda:
                weight.record_stream(torch.cuda.current_stream(weight.device))
            orig_b = getattr(s, "bias", None)
            if isinstance(bias, torch.Tensor) and bias is not orig_b and bias.is_cuda:
                bias.record_stream(torch.cuda.current_stream(bias.device))
        return result

    cast_bias_weight._trt_original = original
    return cast_bias_weight


def _hook_failed(message: str):
    if message not in _hook_failures:
        _hook_failures.add(message)
        logger.warning(message)


def install_hooks():
    from modules import processing

    sampler_mod = compat.sampling_module()
    _hook(compat.unet_class(), "forward", _wrap_forward)
    _hook(processing.StableDiffusionProcessing, "close", _wrap_close)
    _rebind(sampler_mod, "sampling_prepare", _wrap_prepare)
    _rebind(sampler_mod, "sampling_cleanup", _wrap_cleanup)
    _rebind(common.mm(), "free_memory", _wrap_free_memory)
    _rebind(common.mm(), "get_free_memory", _wrap_get_free_memory)
    _rebind(common.mm(), "unload_all_models", _wrap_unload_all)
    if not compat.is_neo():
        try:
            from ldm_patched.modules import ops as ldm_ops

            if _needs_async_offload_fix(ldm_ops, common.mm()):
                _rebind(ldm_ops, "cast_bias_weight", _wrap_cast_bias_weight)
                logger.debug("Applied record_stream workaround for reForge's --async-offload cast_bias_weight.")
        except Exception as e:
            logger.debug(f"Couldn't check reForge's cast_bias_weight for --async-offload workaround: {e}")
    try:
        vae_cls = compat.vae_class()
        _hook(vae_cls, "decode", _wrap_vae_decode)
        _hook(vae_cls, "encode", _wrap_vae_encode)
    except Exception as e:
        _hook_failed(f"Couldn't hook the WebUI's VAE, so it always runs on PyTorch: {e}")
    try:
        from modules import sd_vae

        for fn_name in ("reload_vae_weights", "restore_vae_weights"):
            if callable(getattr(sd_vae, fn_name, None)):
                _rebind(sd_vae, fn_name, _wrap_vae_reload)
    except Exception as e:
        logger.debug(f"Couldn't hook sd_vae reload helpers: {e}")
    try:
        from modules import sd_samplers_cfg_denoiser

        _hook(sd_samplers_cfg_denoiser.CFGDenoiser, "forward", _wrap_denoiser)
    except Exception as e:
        _hook_failed(f"Couldn't hook the WebUI's CFG denoiser, so Speed Hacks never skip a step: {e}")
    try:
        from modules import sd_samplers_kdiffusion

        # AlterSampler (reForge's "alter" samplers) and LCMSampler subclass KDiffusionSampler; AlterSampler overrides
        # get_sigmas without calling the base class's.
        classes = [sd_samplers_kdiffusion.KDiffusionSampler]
        for cls in classes:
            classes.extend(c for c in cls.__subclasses__() if c not in classes)
            if "get_sigmas" in cls.__dict__:
                _hook(cls, "get_sigmas", _wrap_get_sigmas)
    except Exception as e:
        _hook_failed(f"Couldn't hook the WebUI's sampler schedules, so Speed Hacks place their start and end shares by call count: {e}")
    try:
        from modules import prompt_parser

        _rebind(prompt_parser, "reconstruct_cond_batch", _wrap_reconstruct_cond_batch)
        _rebind(prompt_parser, "reconstruct_multicond_batch", _wrap_reconstruct_multicond_batch)
    except Exception as e:
        _hook_failed(f"Couldn't hook the WebUI's prompt schedules, so Speed Hacks see prompt edits a call late: {e}")


def new_batch():  # each batch starts on the engine that fits it best, not the one the last batch ended on
    _last_used.clear()


_generation = None
_generation_thread = None


def _note_speed_hacks(schedule):
    p = _generation() if _generation is not None else None
    if p is not None:
        p.extra_generation_params["TensorRT"] = SPEED_HACKS
        p.extra_generation_params[INFOTEXT[FULL_START]] = round(schedule.warmup * 100)
        p.extra_generation_params[INFOTEXT[FULL_END]] = round(schedule.tail * 100)
        p.extra_generation_params[INFOTEXT[QUALITY]] = round(schedule.quality, 4)


def new_generation(p=None):
    global _generation, _generation_thread, _latest_sigmas
    end_pass()
    common.new_generation()
    _latest_sigmas = None
    _generation = weakref.ref(p) if p is not None else None
    _generation_thread = threading.get_ident()
    if setting() not in (AUTOMATIC, SPEED_HACKS):
        for model in list(_models.values()):
            if model.opened:
                _unload(model, keep=False)
        vae_run.release_all()
        refits.clear()
    else:
        refits.trim()
        strips.trim_pinned()
    try:
        install_hooks()
    except Exception as e:
        notify(f"Couldn't hook reForge's sampler, so the engines don't run: {e}", e)
