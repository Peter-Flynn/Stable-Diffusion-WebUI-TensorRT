# Runs reForge's UNet on TensorRT engines. A hook on UNetModel.forward sends each UNet call of a sampling pass (a Pass)
# to the engine whose profile fits its shapes; anything an engine can't run falls back to PyTorch for the rest of the
# pass. Each engine is a model in reForge's model manager (TrtModel). Only one is in VRAM at a time, and the manager
# evicts engines only as a last resort, because loading one means refitting every weight (or restoring it from RAM,
# refits.py). Before anything loads, _fits checks the engine and the pass's working memory (working.py) can fit in
# VRAM at all; if not, the pass runs on PyTorch. During a pass the PyTorch UNet stays patched but unloaded, in RAM.
import contextlib
import math
import os
import sys
import threading
import time
import weakref

import torch

from modules import sd_unet

from . import arch as arch_mod
from . import common, fresh, gpu, profiles, refits, speed, store, working
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
    # (and its Speed Hacks partner) like any model. Its size is what gpu.py measured once loaded, an estimate
    # before.
    parent = None
    evicted_last = True

    def __init__(self, record: store.Record):
        mm = common.mm()
        self.record = record
        self.owner = f"{record.arch.label} {record.label}"  # key into gpu.py's per-owner accounting: must be unique
        self.name = f"{record.arch.name} {record.label}"
        self.hacks = record.speed_hacks
        if self.hacks and self.hacks.get("fresh") not in (None, fresh.SIGNATURE):
            notify(f"The {self.name} was built by an older version of Speed Hacks, so it runs without them until it's rebuilt in the TensorRT tab.")
            self.hacks = None
        self.model = TensorRTUNet()
        self.load_device = mm.get_torch_device()
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

    @property
    def tracks(self) -> bool:
        return self.shallow_loaded and self.shallow_takes_tracker

    @property
    def shallow_takes_tracker(self) -> bool:  # False for shallow engines built before fresh.py
        return bool(self.hacks and self.hacks.get("fresh"))

    def full(self, shallow: bool, kept=None) -> int:
        total = 0
        for role in ("unet", "shallow") if shallow else ("unet",):
            blob = kept.blobs.get(role) if kept is not None else None
            total += memoryview(blob).nbytes if blob is not None else self._serialized_size(role)
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
        if self.weights is None:  # loaded by reForge outside a sampling pass: use the loaded checkpoint's weights
            patcher = arch_mod.unet_patcher()
            if patcher is None:
                raise RuntimeError("no checkpoint to take the engine's weights from")
            self.weights = EffectiveWeights(patcher)
        shallow = bool(self.hacks and self.want_shallow)
        start, need = time.time(), 0
        if self.opened:  # in VRAM without weights, after a failed refit
            self._discard()
        for attempt in (0, 1):
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
                    refit, more, kept = self._load_engines(kept)
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
                if past and engine_mod.own_allocator(failures, self.owner, e):
                    continue
                if gpu.attached() and not load.get("failures") and gpu.failures() == failures:  # didn't run out: more room won't help
                    raise
                logger.debug(f"Loading the {self.name} ran out of VRAM with {common.gib(need)} freed ({e}); freeing all there is and trying again.")
        refits.used(self.record.key)
        hacks = " with Speed Hacks" if self.shallow_loaded else ""
        how = ["from RAM" if kept else f"plan {plan_seconds:.1f}s"] + ([_refit_times(refit)] if refit else [])
        logger.debug(f"Loaded the {self.name}{hacks} in {time.time() - start:.1f}s ({', '.join(how)}); it holds {common.gib(gpu.held(self.owner))} of VRAM.")
        common.memory(f"after loading the {self.name}")
        return self.model_size()

    def _load_engines(self, kept) -> tuple:
        plan_seconds = 0.0
        kept = kept if "unet" in self.applied else None
        if self.want_shallow and self.shallow is not None:
            try:
                plan_seconds += self._load("shallow", self.shallow, kept)
            except Exception as e:
                notify(f"Couldn't load the Speed Hacks part of the {self.name}, so it runs without Speed Hacks: {e}")
                self.shallow.unload()
                self.want_shallow, self.shallow_failed = False, True
        refit = self.sync(quiet=True)
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
            try:
                seconds = engine.load(blob)
                self.applied[role] = dict(kept.signatures[role])
                return seconds
            except Exception as e:
                logger.debug(f"Couldn't load the {self.name} from RAM ({e}); loading its plan and refitting it instead.")
                refits.drop(self.record.key, role)
                engine.unload()
        return engine.load()

    def _kept(self):
        entry = refits.get(self.record.key)
        if entry is None:
            return None
        wanted = self.weights.signature
        if "unet" in entry.blobs and all(wanted(name) == sig for sigs in entry.signatures.values() for name, sig in sigs.items()):
            return entry
        refits.drop(self.record.key)
        return None

    def _keep(self):
        loaded = {role: e for role, e in (("unet", self.engine), ("shallow", self.shallow)) if e is not None and e.loaded and self.applied.get(role)}
        if "unet" not in loaded or not self._current():
            why = "it holds no weights" if "unet" not in loaded else "its weights aren't the loaded checkpoint's"
            logger.debug(f"Not keeping the {self.name} in RAM: {why}.")
            return
        key, label = self.record.key, self.name
        entry = refits.get(key)
        if entry is not None and entry.signatures.get("unet") != self.applied["unet"]:
            refits.drop(key)
            entry = None
        todo = [role for role in loaded if entry is None or entry.signatures.get(role) != self.applied[role]]
        if not todo:
            return
        if not refits.room(sum(self._serialized_size(role) for role in todo), key):
            logger.debug(f"Not keeping the {label} in RAM: {refits.explain()}.")
            return
        start = time.time()
        torch.cuda.current_stream().synchronize()
        blobs = dict(entry.blobs) if entry is not None else {}
        signatures = dict(entry.signatures) if entry is not None else {}
        for role in todo:
            blobs[role], signatures[role] = loaded[role].serialize(), dict(self.applied[role])
        kept = refits.Kept(signatures, blobs, label, self._source())
        refits.put(key, kept)
        count, total = refits.held()
        logger.debug(f"Kept the {label} in RAM ({common.gib(kept.nbytes)}, {time.time() - start:.1f}s); {count} kept in all, {common.gib(total)}.")

    def _serialized_size(self, role: str) -> int:
        plan, weights = {
            "unet": (self.record.plan, self.record.weight_bytes),
            "shallow": (self.record.shallow_plan, (self.hacks or {}).get("weight_bytes", 0)),
        }[role]
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
        with gpu.loading(self.owner):
            try:
                self._load("shallow", self.shallow, kept)
                self.sync(quiet=True)
            except Exception:
                self.shallow.unload()
                self.want_shallow, self.shallow_failed = False, True
                raise
        logger.debug(f"Loaded the Speed Hacks part of the {self.name} in {time.time() - start:.1f}s; the engine holds {common.gib(gpu.held(self.owner))} of VRAM with it.")

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
            self.applied, self.shallow_failed = {}, False
            left = gpu.ledger(self.owner)
            if left:
                logger.debug(f"TensorRT still holds {common.gib(left)} of the {self.name}'s VRAM after unloading it.")
            gpu.forget(self.owner)
            logger.debug(f"Unloaded the {self.name}.")
        return self.model

    def sync(self, quiet=False):
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
            torch.cuda.current_stream().synchronize()  # no inference in flight while the weights change
            seconds = engine.refit(changed, weights.tensor)
            self.applied[role] = wanted
            refit = {"seconds": seconds, "count": len(changed), "patched": sum(weights.is_patched(name) for name in changed)}
            total = refit if total is None else {k: total[k] + refit[k] for k in refit}
        if total is not None and not quiet:
            logger.debug(f"The {self.name} {_refit_times(total)}.")
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


def speed_settings() -> tuple:
    from modules import shared

    def setting(key, default, scale=1.0):
        try:
            return min(max(float(getattr(shared.opts, key)) / scale, 0.0), 1.0)
        except Exception:
            return default

    return setting(FULL_START, speed.WARMUP, 100), setting(FULL_END, speed.TAIL, 100), setting(QUALITY, speed.QUALITY)


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


def _no_free_memory(memory_required, device, keep_loaded=[]):
    return []


@contextlib.contextmanager
def _keeping(*models):
    # Loads inside this block can't evict these models. During a pass the UNet must stay loaded (unloading it undoes its
    # patches, e.g. model sampling, which the pass still uses) and so must the engine (it would reload on the next step).
    mm = common.mm()
    original = mm.free_memory

    def free_memory(memory_required, device, keep_loaded=[]):
        return original(memory_required, device, keep_loaded=list(keep_loaded) + _loaded_entries(*models))

    mm.free_memory = free_memory
    try:
        yield
    finally:
        mm.free_memory = original


_evicting = 0  # > 0 inside _evictable(): loads for PyTorch's UNet and unload_all_models evict engines normally


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
        mm = common.mm()
        engines = []
        if not _evicting and not getattr(mm, "DISABLE_SMART_MEMORY", False):
            engines = [lm for lm in mm.current_loaded_models if getattr(lm.model, "evicted_last", False) and lm.device == device and lm not in keep_loaded]
        if not engines:
            return original(memory_required, device, keep_loaded)
        unloaded = original(memory_required, device, list(keep_loaded) + engines)
        short = memory_required - mm.get_free_memory(device)
        if 0 < short <= sum(lm.model_loaded_memory() for lm in engines):
            unloaded += original(memory_required, device, keep_loaded)
        return unloaded

    free_memory._trt_original = original
    return free_memory


def _wrap_get_free_memory(original):
    # Never report less than the driver's free memory. With --cuda-malloc, PyTorch reads its counts from the driver's
    # pool, whose in-use count has been seen 1.3 GiB above what it holds, and reForge then loaded its UNet partially
    # and gave its VAE 128 MiB.
    def get_free_memory(dev=None, torch_free_too=False):
        result = original(dev, torch_free_too)
        try:
            device = torch.device(dev if dev is not None else common.mm().get_torch_device())
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
        with _evictable():
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
        self.schedule = speed.Schedule(*speed_settings()) if hacks else None
        self.tracker = _tracker(patcher) if hacks else None
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

    def use(self, record):
        model = model_for(record)
        if model is not self.model or not model.loaded:
            for other in list(_models.values()):
                if other is not model and other.opened:
                    _unload(other)
            self._hold(None)
            model.weights = self.weights
            model.want_shallow = model.want_shallow or (self.schedule is not None and model.hacks is not None and not model.shallow_failed)
            with torch.inference_mode(), _keeping(self.patcher):
                common.register(model)
            self.model = model
            self._hold(_keeping(model, self.patcher))
        _last_used[self.arch_id] = record.key
        if self.schedule is not None and model.hacks and not model.shallow_loaded and not model.shallow_failed:
            model.want_shallow = True
            try:
                with torch.inference_mode():
                    model.load_shallow()
            except Exception as e:
                notify(f"Couldn't load the Speed Hacks part of the {model.name}, so it runs without Speed Hacks: {e}")
        model.sync()
        return model

    def preferred(self) -> set:
        key = _last_used.get(self.arch_id)
        return {i for i, (r, _) in enumerate(self.slots) if r.key == key}

    def _hold(self, keep):
        if self._keep is not None:
            self._keep, old = None, self._keep
            old.__exit__(None, None, None)
        if keep is not None:
            keep.__enter__()
            self._keep = keep

    def _restore(self):
        self.patcher.model.__dict__.pop("memory_required", None)
        self._hold(None)
        if self.model is not None and self.model.loaded:
            self.model.engine.release_activations()
        if self.schedule is not None:
            self.schedule.release()
        self.tracker = None

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
        hacks = self.schedule
        if hacks is not None and hacks.shallow_calls:
            logger.debug(f"Speed Hacks ran {hacks.shallow_calls} of {hacks.shallow_calls + hacks.full_calls} UNet calls on the shallow engine.")
            logger.debug(f"Speed Hacks pass: {hacks.summary()}")
            _note_speed_hacks()
        for model in _models.values():
            if model.weights is self.weights:
                model.weights = None  # don't keep this checkpoint alive

    def fall_back(self, reason: str, error: Exception = None):
        self.fallback = reason
        notify(f"{reason}, so the rest of this pass runs on PyTorch.", error)
        self._restore()
        _load_torch_unet(self.patcher, self.shape)
        raise Fallback(reason)

    def run(self, x, timesteps, context, y, chunks: int = 0):
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
        stream = torch.cuda.current_stream().cuda_stream
        slot = deep = origin = stale = tracking = None
        if self.schedule is not None and self.model.shallow_loaded:
            slot, deep, origin, stale = self.schedule.call(feed["sample"].shape)
            if self.model.tracks:
                tracking = self.tracker.call(feed["sample"], chunks, padded > b, slot, self.schedule.index)
        if self.tracker is not None:
            self.tracker.stash = None
        out = self._infer_with_room(feed, profile, stream, slot, deep, origin, tracking, stale)
        if deep is None and not self.checked:
            self.checked = True
            if not torch.isfinite(out).all():
                self.fall_back("The engine produced NaN/Inf values")
        if deep is not None and not self.shallow_checked:
            self.shallow_checked = True
            if not torch.isfinite(out).all():
                notify("Speed Hacks produced NaN/Inf values, so the rest of this pass runs without them.")
                self.schedule = None
                out = self._infer_with_room(feed, profile, stream)
        return out[:b].to(x.dtype)

    def _infer_with_room(self, feed, profile, stream, slot=None, deep=None, origin=None, tracking=None, stale=None):
        try:
            return self._infer(feed, profile, stream, slot, deep, origin, tracking, stale)
        except torch.cuda.OutOfMemoryError:
            self.model.engine.release_activations()
            common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, self.model.load_device, keep_loaded=_loaded_entries(self.model, self.patcher))
            common.mm().soft_empty_cache(force=True)
            try:
                return self._infer(feed, profile, stream, slot, deep, origin, tracking, stale)
            except torch.cuda.OutOfMemoryError:
                self.fall_back("The engine ran out of VRAM")

    def _infer(self, feed, profile, stream, slot=None, deep=None, origin=None, tracking=None, stale=None):
        try:
            if deep is not None:
                extra = {}
                if self.model.shallow_takes_tracker:
                    extra = self.tracker.shallow_inputs(tracking, origin) if tracking is not None else self.tracker.dummy_inputs(feed["sample"])
                outputs = self.model.shallow.infer({**feed, "deep": deep, **extra}, profile, stream)
                if tracking is not None:
                    self.tracker.after_shallow(tracking, outputs, origin)
                    self.schedule.shallow_done(outputs["unknown"], origin)
                return outputs["latent"]
            outputs = self.model.engine.infer(feed, profile, stream)
            out = outputs["latent"]
            if slot is not None and "deep" in outputs:
                self.schedule.keep(slot, feed["sample"].shape, outputs["deep"])
                if self.schedule.check and slot == 0 and self.model.shallow_takes_tracker and (stale is not None or tracking is not None):
                    self._check(feed, profile, stream, tracking, out, outputs["deep"], stale)
                    return out
            if tracking is not None:
                self.tracker.after_full(tracking, out)
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            self.fall_back(f"The engine failed ({e})", _unexpected(e))

    def _check(self, feed, profile, stream, tracking, out, deep, stale):
        # stale: (deep feature, index) this full call replaces, on which the shallow engine runs as the shallow call at
        # this index would have. Without one (no full call before it), on this call's own deep feature, for the noise
        # share alone.
        origin = stale[1] if stale is not None else None
        extra = self.tracker.shallow_inputs(tracking, origin, out) if tracking is not None else self.tracker.dummy_inputs(feed["sample"], out)
        outputs = self.model.shallow.infer({**feed, "deep": stale[0] if stale is not None else deep, **extra}, profile, stream)
        if tracking is not None:
            self.tracker.after_full(tracking, out, outputs)
        if origin is not None:
            self.schedule.checked(outputs["error"], self.schedule.index - origin)


def _tracker(patcher):
    sampling = getattr(patcher.model, "model_sampling", None)
    try:
        pred = fresh.supported(sampling)
    except Exception:
        pred = None
    return fresh.Tracker(sampling.log_sigmas if pred is not None else None, pred, common.mm().get_torch_device())


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
    hacks = chosen == SPEED_HACKS and any(r.speed_hacks for r in records)
    failed = sum(not r.speed_hacks for r in records) if chosen == SPEED_HACKS else 0
    if failed:
        retry = " Building them again in the TensorRT tab retries once." if any(r.retry_speed_hacks for r in records) else ""
        notify(f"Speed Hacks couldn't be built for {failed} of the {arch.name} engines, so they run without them.{retry}")
    return records, hacks


def _begin(records, hacks, patcher, x, prepare):
    mm = common.mm()
    with torch.inference_mode():
        _unload_torch_unet(patcher)
        state = mm.vram_state, mm.free_memory
        # reForge's sampling_prepare applies the UNet's patches; NO_VRAM keeps its weights in RAM while it does
        mm.vram_state, mm.free_memory = mm.VRAMState.NO_VRAM, _no_free_memory
        try:
            prepare(patcher, x)
        finally:
            mm.vram_state, mm.free_memory = state
        p = Pass(records, patcher, x, hacks)
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
    shallow = bool(model.hacks and (model.want_shallow or (p.schedule is not None and not model.shallow_failed)))
    calls = max(call_bytes(model, p.patcher, x), model.took.get(tuple(x.shape[i] for i in (0, 2, 3)), 0))
    held = gpu.held(model.owner)
    if model.loaded:
        engine = held + (model._serialized_size("shallow") if shallow and not model.shallow_loaded else 0)
    else:
        model.weights = p.weights
        engine = model.full(shallow, model._kept())
    room = common.obtainable(model.load_device, keep=[model]) - common.reserve() + held
    need = held + common.padded(engine - held + calls)  # margin only on what isn't in VRAM yet
    logger.debug(
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
    act = model.engine.estimate_activation_bytes(profile, shapes, torch.cuda.current_stream().cuda_stream)
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
                return p.run(x, timesteps, context, y, len(transformer_options.get("cond_or_uncond") or ()))
            except Fallback:
                pass
        return original(self, x, timesteps, context, y, control, transformer_options, **kwargs)

    forward._trt_original = original
    return forward


def _wrap_prepare(original):
    def sampling_prepare(unet, x, *args, **kwargs):
        global _pass
        end_pass()
        records, hacks = _engine_records(unet, x)
        if records is None:
            with _evictable():
                return original(unet, x, *args, **kwargs)
        _pass = _begin(records, hacks, unet, x, lambda u, v: original(u, v, *args, **kwargs))

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
    def forward(self, x, *args, **kwargs):
        p = _pass
        if p is not None and p.schedule is not None and p.fallback is None:
            try:
                from modules import shared

                p.schedule.begin(int(self.step), self.total_steps, shared.state.sampling_steps)
                # Inpainting blends x with the image outside the mask, which isn't in fresh.py's linear chain.
                if p.tracker is not None and getattr(self, "mask", None) is not None:
                    p.tracker.active = False
            except Exception as e:
                notify(f"Speed Hacks failed ({e}), so the rest of this pass runs without them.", e)
                p.schedule = None
        return original(self, x, *args, **kwargs)

    forward._trt_original = original
    return forward


def _wrap_apply_model(original):
    # Stashes the FP32 x and sigma of the UNet call apply_model is about to make, for fresh.py's fit. The UNet's own
    # input is FP16, whose rounding (up to 2^-11 of x) is far above fresh.FLOOR and would count as fresh noise.
    def apply_model(self, x, t, *args, **kwargs):
        p = _pass
        if p is not None and p.tracker is not None and self is p.patcher.model:
            p.tracker.stash = (x, t)
        return original(self, x, t, *args, **kwargs)

    apply_model._trt_original = original
    return apply_model


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


def _rebind(owner, name, wrap):  # also replaces references other modules took with `from ... import name`
    current = getattr(owner, name)
    if hasattr(current, "_trt_original"):
        return
    replacement = wrap(current)
    setattr(owner, name, replacement)
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get(name) is current:
            setattr(module, name, replacement)


_hook_failures = set()  # install_hooks runs every generation: log each failure once


def _hook_failed(message: str):
    if message not in _hook_failures:
        _hook_failures.add(message)
        logger.warning(message)


def install_hooks():
    from ldm_patched.ldm.modules.diffusionmodules import openaimodel
    from modules import processing
    from modules_forge import forge_sampler

    if not hasattr(openaimodel.UNetModel.forward, "_trt_original"):
        openaimodel.UNetModel.forward = _wrap_forward(openaimodel.UNetModel.forward)
    if not hasattr(processing.StableDiffusionProcessing.close, "_trt_original"):
        processing.StableDiffusionProcessing.close = _wrap_close(processing.StableDiffusionProcessing.close)
    _rebind(forge_sampler, "sampling_prepare", _wrap_prepare)
    _rebind(forge_sampler, "sampling_cleanup", _wrap_cleanup)
    _rebind(common.mm(), "free_memory", _wrap_free_memory)
    _rebind(common.mm(), "get_free_memory", _wrap_get_free_memory)
    _rebind(common.mm(), "unload_all_models", _wrap_unload_all)
    try:
        from ldm_patched.modules import sd as ldm_sd

        if not hasattr(ldm_sd.VAE.decode, "_trt_original"):
            ldm_sd.VAE.decode = _wrap_vae_decode(ldm_sd.VAE.decode)
        if not hasattr(ldm_sd.VAE.encode, "_trt_original"):
            ldm_sd.VAE.encode = _wrap_vae_encode(ldm_sd.VAE.encode)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's VAE, so it always runs on PyTorch: {e}")
    try:
        from modules import sd_samplers_cfg_denoiser

        denoiser = sd_samplers_cfg_denoiser.CFGDenoiser
        if not hasattr(denoiser.forward, "_trt_original"):
            denoiser.forward = _wrap_denoiser(denoiser.forward)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's CFG denoiser, so Speed Hacks never skip a step: {e}")
    try:
        from ldm_patched.modules import model_base

        if not hasattr(model_base.BaseModel.apply_model, "_trt_original"):
            model_base.BaseModel.apply_model = _wrap_apply_model(model_base.BaseModel.apply_model)
    except Exception as e:
        _hook_failed(f"Couldn't hook reForge's apply_model, so Speed Hacks don't correct the noise ancestral and SDE samplers add: {e}")


def new_batch():  # each batch starts on the engine that fits it best, not the one the last batch ended on
    _last_used.clear()


_generation = None
_generation_thread = None


def _note_speed_hacks():
    p = _generation() if _generation is not None else None
    if p is not None:
        p.extra_generation_params["TensorRT"] = SPEED_HACKS


def new_generation(p=None):
    global _generation, _generation_thread
    end_pass()
    common.new_generation()
    _generation = weakref.ref(p) if p is not None else None
    _generation_thread = threading.get_ident()
    if setting() not in (AUTOMATIC, SPEED_HACKS):
        for model in list(_models.values()):
            if model.loaded:
                _unload(model, keep=False)
        vae_run.release_all()
        refits.clear()
    else:
        refits.trim()
    try:
        install_hooks()
    except Exception as e:
        notify(f"Couldn't hook reForge's sampler, so the engines don't run: {e}", e)
