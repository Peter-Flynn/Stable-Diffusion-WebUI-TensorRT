# Runs reForge's UNet on TensorRT engines. A hook on UNetModel.forward sends each UNet call of a sampling pass (a Pass)
# to the engine whose profile fits its shapes; anything an engine can't run falls back to PyTorch for the rest of the
# pass. Each engine is a model in reForge's model manager (TrtModel). Only one is in VRAM at a time, and the manager
# evicts engines only as a last resort, because loading one means refitting every weight (or restoring it from RAM,
# refits.py). Before anything loads, _fits checks the engine and the pass's working memory (working.py) can fit in
# VRAM at all; if not, the pass runs on PyTorch. During a pass the PyTorch UNet stays patched but unloaded, in RAM.
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
from . import common, fresh, gpu, profiles, refits, speed, store, strips, working
from . import vae_run
from .common import inference_mode, logger, notify
from .weights import EffectiveWeights

_models = {}
_last_used = {}
_pass = None


class Fallback(Exception):
    pass


class TensorRTUNet:
    pass


def call_bytes(model, patcher, x) -> int:
    found = _call_shapes(model, x)
    batch = found[1]["sample"][0] if found is not None else 2 * x.shape[0]
    return working.call_bytes(model.record.arch, patcher, batch, x.shape[3] * 8, x.shape[2] * 8)


class DoesNotFit(RuntimeError):
    pass


class TrtModel:
    # Implements the parts of ModelPatcher that reForge's LoadedModel calls, so the model manager can hold an engine
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
            from .engine import Arena, Engine

            arena = Arena()
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

    def model_dtype(self):
        return torch.float16

    def model_patches_to(self, *args):
        pass

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
                    common.make_room(common.padded(need) if not attempt else common.obtainable(self.load_device) - common.reserve(), self.load_device)
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
                made.append(refits.store(loaded[role].serialize))
                blobs[role], signatures[role] = made[-1], dict(self.applied[role])
        except BaseException:
            for blob in made:
                blob.free()
            raise
        kept = refits.Kept(signatures, blobs, label, self._source())
        if not refits.put(key, kept, expected=entry):
            for blob in made:
                blob.free()
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
        need = self.full(True, kept) - self.full(False, kept)
        common.make_room(common.padded(need), self.load_device, _loaded_entries(self))
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
            engine_mod.LOGGER.clear()
            failures = gpu.failures()
            try:
                seconds = engine.refit(changed, weights.tensor)
            except Exception as e:
                if role != "shallow" or not drop_shallow:
                    raise
                ran_out = isinstance(e, torch.cuda.OutOfMemoryError) or gpu.failures() != failures or engine_mod.LOGGER.out_of_memory
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
    _last_used.clear()
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


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, option: SpeedHacksOption):
        super().__init__()
        self.option = option

    def forward(self, *args, **kwargs):
        raise RuntimeError("TensorRT engines run through the UNet's own forward")


_SPEED_HACKS_OPTION = SpeedHacksOption()  # one instance: reForge compares options by identity when re-listing them


def list_unets(options: list):
    options.append(_SPEED_HACKS_OPTION)


def _unsupported(patcher) -> str:
    if getattr(patcher, "controlnet_linked_list", None) is not None:
        return "ControlNet"
    if getattr(patcher, "extra_concat_condition", None) is not None:
        return "extra input channels (inpainting add-ons)"
    module = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if module is not None:
        paddings = {m.padding_mode for m in module.modules() if isinstance(m, torch.nn.Conv2d) and m.padding_mode != "zeros"}
        if paddings:
            return "tiling (circular padding)" if paddings == {"circular"} else f"non-zero conv padding ({', '.join(sorted(paddings))})"
    options = patcher.model_options.get("transformer_options", {})
    kinds = sorted({k for group in ("patches", "patches_replace") for k, v in options.get(group, {}).items() if v})
    if options.get("wrappers", {}).get("diffusion_model"):
        kinds.append("UNet wrappers")
    if any(k.startswith("diffusion_model.") for k in getattr(patcher, "object_patches", {})):
        kinds.append("replaced UNet modules")
    if any(v for k, v in (getattr(patcher, "weight_wrapper_patches", None) or {}).items() if k.startswith("diffusion_model.")):
        kinds.append("weight wrappers")
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
        if not _evicting and not getattr(mm, "DISABLE_SMART_MEMORY", False):
            engines = [lm for lm in mm.current_loaded_models if getattr(lm.model, "evicted_last", False) and lm.device == device and lm not in keep]
        if not engines:
            return original(memory_required, device, keep)
        unloaded = original(memory_required, device, keep + engines)
        short = memory_required - mm.get_free_memory(device)
        if 0 < short <= sum(lm.model_loaded_memory() for lm in engines):
            unloaded += original(memory_required, device, keep)
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
    with _evictable():
        common.mm().load_models_gpu([patcher], memory_required=required)


class Pass:
    def __init__(self, records: list, patcher, x, hacks: bool = False):
        self.patcher = patcher
        model = patcher.model
        self.schedule = self.tracker = None
        if hacks:
            try:
                self.tracker = fresh.Tracker(model.model_sampling, common.device(), model.diffusion_model.out_channels)
                self.schedule = speed.Schedule(*speed_settings())
            except Exception as e:
                notify(f"Speed Hacks failed ({e}), so this pass runs without them.", e)
                self.tracker, hacks = None, False
        self.slots = store.slots(records)
        self.model = None
        self.weights = EffectiveWeights(patcher)
        self.module = patcher.model.diffusion_model
        arch = records[0].arch
        self.arch_id = arch.id
        self.shapes, self.takes_y = [r.profile_shapes[j] for r, j in self.slots], bool(arch.adm_channels)
        self.channels, self.context_dim = arch.in_channels, arch.context_dim
        self.shape = (x.shape[0] * 2, *x.shape[1:])
        self.fallback = None
        self.checked = self.shallow_checked = False
        self._keep = None
        self._hint_batching()
        self.started = self.schedule  # for end()'s report once Speed Hacks turn off mid-pass
        self._watching = False
        if hacks:
            self._watch_apply_model(model)

    def use(self, record):
        model = model_for(record)
        if model is not self.model or not model.loaded:
            for other in list(_models.values()):
                if other is not model and other.opened:
                    _unload(other)
            self._hold()
            model.weights = self.weights
            model.want_shallow = bool(self.schedule is not None and model.hacks is not None and not model.shallow_failed)
            with torch.inference_mode(), _keeping(self.patcher):
                common.register(model)
            self.model = model
            self._hold(model, self.patcher)
        _last_used[self.arch_id] = record.key
        model.sync()
        if self.schedule is not None and model.hacks and not model.shallow_loaded and not model.shallow_failed:
            model.want_shallow = True
            try:
                with torch.inference_mode():
                    model.load_shallow()
            except Exception as e:
                notify(f"Couldn't load the Speed Hacks part of the {model.name}, so it runs without Speed Hacks: {e}")
        return model

    def preferred(self) -> set:
        key = _last_used.get(self.arch_id)
        return {i for i, (r, _) in enumerate(self.slots) if r.key == key}

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
        if self.model is not None and self.model.loaded:
            self.model.engine.release_activations()
        if self.schedule is not None:
            self.schedule.release()
        if self.tracker is not None:
            self.tracker.release()
        self.schedule = self.tracker = None

    def _hint_batching(self):
        # reForge decides how many conds to batch into one UNet call from the UNet's memory_required(). Overridden for
        # the pass: 0 for any batch some profile runs (the engine's memory is already allocated), infinite otherwise,
        # which makes reForge split the batch.
        shapes = self.shapes

        def memory_required(input_shape, cond_shapes={}):
            b, _, h, w = input_shape[:4]
            lengths = [s[1] for s in cond_shapes.get("c_crossattn", ()) if len(s) > 1]
            context = math.lcm(*lengths) if lengths else profiles.CONTEXT_PER_CHUNK
            return 0 if profiles.select(shapes, b, h, w, context) is not None else float("inf")

        self.patcher.model.memory_required = memory_required

    def end(self):
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
        for model in _models.values():
            if model.weights is self.weights:
                model.weights = None  # don't keep this checkpoint alive

    def fall_back(self, reason: str, error: Exception = None):
        self.fallback = reason
        notify(f"{reason}, so the rest of this pass runs on PyTorch.", error)
        self._restore()
        _load_torch_unet(self.patcher, self.shape)
        raise Fallback(reason)

    def run(self, x, timesteps, context, y, cond_or_uncond=None):
        b, c, h, w = x.shape
        if c != self.channels or context.shape[2] != self.context_dim:
            self.fall_back(f"The UNet was given {c} input channels and {context.shape[2]}-wide prompts, but the engine takes {self.channels} and {self.context_dim}")
        choice = profiles.select(self.shapes, b, h, w, context.shape[1], prefer=self.preferred())
        if choice is None:
            tokens = context.shape[1] // profiles.CONTEXT_PER_CHUNK * profiles.TOKENS_PER_CHUNK
            self.fall_back(
                f"No profile runs {w * 8}x{h * 8} with a UNet batch of {b} and {tokens}-token prompts "
                "(reForge pads the prompt and negative prompt to the same length)"
            )
        index, padded, repeats = choice
        record, profile = self.slots[index]
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
        if self.model is None or self.model.record.key != record.key or not self.model.loaded:
            try:
                if self.model is not None and self.model.loaded:
                    self.model.engine.release_activations()
                self.use(record)
            except Exception as e:
                failed = _models.get(record.key)
                if failed is not None and failed.opened and not failed.loaded:
                    _unload(failed, keep=False)
                self.fall_back(f"Couldn't load the {record.arch.name} {record.label} ({e})", _unexpected(e))
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
        out = self._infer_with_room(feed, profile, stream, hack)
        # A shallow call whose Speed Hacks failed ran on the full engine instead
        shallow = bool(hack) and hack[2] is not None and self.schedule is not None
        if not shallow and not self.checked:
            self.checked = True
            if not torch.isfinite(out).all():
                self.fall_back("The engine produced NaN/Inf values")
        if shallow and not self.shallow_checked:
            self.shallow_checked = True
            if not torch.isfinite(out).all():
                if self.started is not None:
                    self.started.ran_full()
                self.drop_speed_hacks("Speed Hacks produced NaN/Inf values")
                out = self._infer_with_room(feed, profile, stream)
        return out[:b].to(x.dtype)

    def _infer_with_room(self, feed, profile, stream, hack=None):
        def clear(error):
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

        try:
            return self._infer(feed, profile, stream, hack)
        except torch.cuda.OutOfMemoryError as e:
            clear(e)
            self.model.engine.release_activations()
            common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, self.model.load_device, keep_loaded=_loaded_entries(self.model, self.patcher))
            common.mm().soft_empty_cache(force=True)
            try:
                return self._infer(feed, profile, stream, hack, retry=True)
            except torch.cuda.OutOfMemoryError as e:
                clear(e)
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
    if not profiles.covers_size([s for r in records for s in r.profile_shapes], h, w, b):
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


def _fits(model: TrtModel, p: Pass, x):
    shallow = bool(model.hacks and p.schedule is not None and not model.shallow_failed)
    calls = max(call_bytes(model, p.patcher, x), model.took.get(tuple(x.shape[i] for i in (0, 2, 3)), 0))
    held = model.loaded_size()
    if model.loaded:
        engine = held + (model._serialized_size("shallow") if shallow and not model.shallow_loaded else 0)
    else:
        engine = model.full(shallow, model._kept(p.weights))
    room = common.obtainable(model.load_device, keep=[model]) - common.reserve() + held
    need = held + common.padded(engine - held + calls)  # margin only on what isn't in VRAM yet
    logger.info(
        f"The {model.name} at {_sizes(x)} needs {common.gib(need)} of VRAM ({common.gib(engine)} for the engine, {common.gib(calls)} "
        f"to run it, {common.gib(need - engine - calls)} margin); {common.gib(room)} can be freed."
    )
    if need > room:
        raise DoesNotFit(f"it needs {common.gib(need)} of VRAM at {_sizes(x)}, and at most {common.gib(room)} can be freed")


def _make_room_for_activations(model: TrtModel, patcher, x):
    try:
        takes = _call_bytes(model, x)
    except Exception as e:
        logger.debug(f"Couldn't ask TensorRT what running the {model.name} takes: {e}")
        return
    if not takes:
        return
    counted = call_bytes(model, patcher, x)
    if takes > counted:
        most = " (the most for any size its profile takes: it gave no figure for this one)" if model.engine.size_is_most else ""
        logger.debug(f"TensorRT takes {common.gib(takes)} to run the {model.name} at {_sizes(x)}{most}, more than the {common.gib(counted)} estimated.")
    need = common.padded(max(takes, counted))
    free = common.make_room(need, model.load_device, _loaded_entries(model, patcher))
    if free < need:
        model.took[tuple(x.shape[i] for i in (0, 2, 3))] = takes  # so _fits rejects these sizes before loading next time
        raise RuntimeError(f"running it at {_sizes(x)} needs {common.gib(need)} of VRAM besides the engine, and only {common.gib(free)} could be freed")


def _call_shapes(model: TrtModel, x):
    b, c, h, w = x.shape
    held = model.record.profile_shapes
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


def _call_bytes(model: TrtModel, x) -> int:
    found = _call_shapes(model, x)
    if found is None:
        return 0
    profile, shapes = found
    act = model.engine.estimate_activation_bytes(profile, shapes, torch.cuda.current_stream(model.load_device).cuda_stream)
    io = model.engine.io_bytes()
    return act + sum(io.values())


def _wrap_forward(original):
    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        p = _pass
        if p is not None and p.fallback is None and self is p.module:
            try:
                if control is not None:
                    p.fall_back("TensorRT engines can't apply ControlNet")
                if any(v for g in ("patches", "patches_replace") for v in transformer_options.get(g, {}).values()):
                    p.fall_back("TensorRT engines can't apply UNet patches")
                if (transformer_options.get("wrappers") or {}).get("diffusion_model"):
                    p.fall_back("TensorRT engines can't apply UNet wrappers")
                return p.run(x, timesteps, context, y, transformer_options.get("cond_or_uncond"))
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
        if _pass is not None and _pass.schedule is not None and _denoiser_state is not None:
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
    # reForge closes the processing object however a generation ended. One that failed mid-pass never reached
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
        current_state = None
        if setting() in (AUTOMATIC, SPEED_HACKS):
            try:
                from modules import shared

                state = shared.state
                current_state = (int(self.step), self.total_steps, state.sampling_steps, state.sampling_step, x, sigma)
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
    def decode(self, samples_in, vae_options={}):
        # Only on the generation's thread: reForge's progress endpoint decodes "Full" live previews on another thread
        # while this one is sampling on the UNet engine.
        if not vae_options and setting() in (AUTOMATIC, SPEED_HACKS) and threading.get_ident() == _generation_thread:
            try:
                pixels = vae_run.decode(self, samples_in)
                if pixels is not None:
                    return pixels
            except Exception as e:
                notify(f"The VAE decoder engines failed ({e}), so this decode runs on PyTorch.", e)
        return original(self, samples_in, vae_options)

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
    from ldm_patched.ldm.modules.diffusionmodules import openaimodel
    from modules import processing
    from modules_forge import forge_sampler

    _hook(openaimodel.UNetModel, "forward", _wrap_forward)
    _hook(processing.StableDiffusionProcessing, "close", _wrap_close)
    _rebind(forge_sampler, "sampling_prepare", _wrap_prepare)
    _rebind(forge_sampler, "sampling_cleanup", _wrap_cleanup)
    _rebind(common.mm(), "free_memory", _wrap_free_memory)
    _rebind(common.mm(), "get_free_memory", _wrap_get_free_memory)
    _rebind(common.mm(), "unload_all_models", _wrap_unload_all)
    try:
        from ldm_patched.modules import ops as ldm_ops

        if _needs_async_offload_fix(ldm_ops, common.mm()):
            _rebind(ldm_ops, "cast_bias_weight", _wrap_cast_bias_weight)
            logger.debug("Applied record_stream workaround for reForge's --async-offload cast_bias_weight.")
    except Exception as e:
        logger.debug(f"Couldn't check reForge's cast_bias_weight for --async-offload workaround: {e}")
    try:
        from ldm_patched.modules import sd as ldm_sd

        _hook(ldm_sd.VAE, "decode", _wrap_vae_decode)
        _hook(ldm_sd.VAE, "encode", _wrap_vae_encode)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's VAE, so it always runs on PyTorch: {e}")
    try:
        from modules import sd_samplers_cfg_denoiser

        _hook(sd_samplers_cfg_denoiser.CFGDenoiser, "forward", _wrap_denoiser)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's CFG denoiser, so Speed Hacks never skip a step: {e}")
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
        _hook_failed(f"Couldn't hook reForge's sampler schedules, so Speed Hacks place their start and end shares by call count: {e}")
    try:
        from modules import prompt_parser

        _rebind(prompt_parser, "reconstruct_cond_batch", _wrap_reconstruct_cond_batch)
        _rebind(prompt_parser, "reconstruct_multicond_batch", _wrap_reconstruct_multicond_batch)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's prompt schedules, so Speed Hacks see prompt edits a call late: {e}")


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
