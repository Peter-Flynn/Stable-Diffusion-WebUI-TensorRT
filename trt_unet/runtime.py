"""Running reForge's UNet on a TensorRT engine.

An architecture has an engine per profile, or one merged engine holding them all (store.py), and a sampling
pass loads the one for its sizes; only one is in VRAM at a time. Each is a model of reForge's own model
manager, held like any other (it makes exactly its room as it loads), but evicted only as a last resort
(_wrap_free_memory): an engine comes back without its weights. One that's unloaded
is kept in RAM as it is, weights and all, while RAM allows (refits.py), and comes back from there for the
same weights; otherwise every weight is refitted. While a sampling pass runs on an engine, the PyTorch UNet
stays in RAM (its patches still apply, it just isn't loaded).

Whether a pass fits is worked out before anything loads (_fits), from what it will take: the engine, which
is its weights and its plan, and the working memory of its UNet calls, which is the image's size: so many
bytes for each pixel of each image of the batch (CALL_BYTES_PER_PIXEL). One that doesn't fit runs on
PyTorch, with the figures said. Nothing of this is kept on disk."""

import contextlib
import math
import os
import sys
import threading
import time
import traceback
import weakref

import torch

from modules import sd_unet

from . import arch as arch_mod
from . import common, gpu, profiles, refits, speed, store
from . import vae as vae_mod
from .common import inference_mode, log, notify
from .weights import EffectiveWeights

_models = {}  # Record.key -> TrtModel
_last_used = {}  # arch id -> Record.key of the engine the batch being generated last ran on (see new_batch)
_pass = None  # the sampling pass running on an engine


class Fallback(Exception):
    pass


class TensorRTUNet:
    """What the model manager holds for an engine (its name shows in reForge's log)."""


# The working VRAM of a pass's UNet calls, for each pixel of each image of its batch. An SDXL UNet in FP16
# takes this at every size and batch size it was seen at (1024x1472 at 1 and at 2, 1536x2208 at 1: within 1%)
CALL_BYTES_PER_PIXEL = 433


def call_bytes(x) -> int:
    """The working VRAM a pass on latents `x` takes for its UNet calls."""
    return x.shape[0] * x.shape[2] * 8 * x.shape[3] * 8 * CALL_BYTES_PER_PIXEL


class DoesNotFit(RuntimeError):
    """A pass's engine and calls can't be in VRAM together: said with the figures."""


class TrtModel:
    """An engine as reForge's model manager sees a model (the parts of ModelPatcher it uses), with its
    shallow partner for Speed Hacks (speed.py), loaded only for passes that use them. The two share their
    activation memory, since they never run at once. Loaded, its size is what its engines hold in VRAM
    (gpu.py); before that, what they will (full)."""

    parent = None
    evicted_last = True  # by reForge's manager (_wrap_free_memory), and not counted on for room otherwise (common.obtainable)

    def __init__(self, record: store.Record):
        mm = common.mm()
        self.record = record
        self.owner = f"{record.arch.label} {record.label}"  # whose TensorRT memory it is (gpu.py)
        self.hacks = record.speed_hacks  # read once: it checks the disk, and model_size() is called often
        self.model = TensorRTUNet()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.engine = self.shallow = None
        self.want_shallow = False  # load the shallow engine with this one
        self.shallow_failed = False  # its load failed: not tried again until this engine loads again
        self.weights = None  # EffectiveWeights the engines should run with
        self.applied = {}  # "unet" / "shallow" -> {weight name: signature of what's in that engine}
        store.forget_sizes(record)
        self.took = {}  # a pass's sizes (batch, height, width of its latents) -> what TensorRT turned out to take for its calls, where that didn't fit
        self.keep = True  # whether unloading keeps the engines in RAM as they are (refits.py)

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
        """Whether its engine is in VRAM, with or without weights."""
        return self.engine is not None and self.engine.loaded

    @property
    def loaded(self) -> bool:
        """Whether its engine is in VRAM with weights in it."""
        return self.opened and bool(self.applied.get("unet"))

    @property
    def shallow_loaded(self) -> bool:
        return self.shallow is not None and self.shallow.loaded

    def full(self, shallow: bool, kept=None) -> int:
        """What the engines hold in VRAM once loaded: their weights and their plans' size, or, as kept in
        RAM with their weights, that serialized engine's bytes."""
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
        """Loads the engines, with room made for what that takes (full) beyond what's opened. A load that
        fails is tried once more: on TensorRT's own allocator if PyTorch's could be why
        (engine.own_allocator), else with all the room there is, if it ran out (or, without PyTorch's
        allocator, for all that's known)."""
        from . import engine as engine_mod

        if self.loaded:
            return 0
        if self.weights is None:  # a load outside a sampling pass: the loaded checkpoint's weights
            patcher = arch_mod.unet_patcher()
            if patcher is None:
                raise RuntimeError("no checkpoint to take the engine's weights from")
            self.weights = EffectiveWeights(patcher)
        shallow = bool(self.hacks and self.want_shallow)
        start, need, predicted = time.time(), 0, 0
        for attempt in (0, 1):
            load, failures = {}, gpu.failures()
            try:
                kept = self._kept()
                predicted = self.full(shallow, kept)
                need = max(0, predicted - gpu.held(self.owner))
                common.make_room(common.padded(need) if not attempt else common.obtainable(self.load_device) - common.reserve(), self.load_device)
                common.memory(f"loading the {self.record.label}: before", f"room made for {common.gib(need)}")
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
                past = self.opened  # it deserialized, and failed past that
                if not attempt:
                    e.__traceback__ = None  # it holds the failed load's frames, and through them what TensorRT allocated for it
                self._discard()
                if attempt:
                    raise
                if past and engine_mod.own_allocator(failures, self.owner):
                    log(f"Loading the {self.record.label} failed with PyTorch's allocator ({e}); trying again with TensorRT's own.")
                    continue
                if gpu.attached() and not load.get("failures") and gpu.failures() == failures:  # it didn't run out: more room changes nothing
                    raise
                log(f"Could not load the {self.record.label} with {common.gib(need)} of room made ({e}); making all the room there is and trying again.")
        size = gpu.held(self.owner)
        refits.used(self.record.key)
        with_shallow = " with its Speed Hacks part" if self.shallow_loaded else ""
        source = f"as kept in RAM with its weights {plan_seconds:.1f}s" if kept else f"plan {plan_seconds:.1f}s"
        log(
            f"Loaded the {self.owner}{with_shallow} ({common.gib(size)} in VRAM, {common.gib(predicted)} predicted) in {time.time() - start:.1f}s: "
            f"{source}, " + (_refit_times(refit) if refit else "no weights to refit")
        )
        common.memory(f"loading the {self.record.label}: done")
        return self.model_size()

    def _load_engines(self, kept) -> tuple:
        """With the UNet engine deserialized: (what refitting took, None if they came as they were kept; the
        seconds loading the shallow engine took; what was kept in RAM of them, if they came from there.)"""
        plan_seconds = 0.0
        kept = kept if "unet" in self.applied else None
        if self.want_shallow and self.shallow is not None:
            try:
                plan_seconds += self._load("shallow", self.shallow, kept)
            except Exception as e:  # the engine runs without it
                log(f"Could not load the Speed Hacks part of the {self.record.label} ({e}); running without it.")
                self.shallow.unload()
                self.want_shallow, self.shallow_failed = False, True
        refit = self.sync(quiet=True)  # what isn't in them yet: everything, unless they came as they were kept
        return refit, plan_seconds, kept

    def _discard(self):
        """Unloads what a failed load left, keeping nothing of it in RAM."""
        self.keep = False
        try:
            self.detach()
        finally:
            self.keep = True
        gpu.forget(self.owner)

    def _load(self, role: str, engine, kept) -> float:
        """Loads one of the engines: as kept in RAM with its weights, if it is (what it holds then goes to
        `applied`, so nothing is refitted), else its plan, without weights. Returns the seconds it took."""
        blob = kept.blobs.get(role) if kept is not None else None
        self.applied.pop(role, None)
        if blob is not None:
            try:
                seconds = engine.load(blob)
                self.applied[role] = dict(kept.signatures[role])
                return seconds
            except Exception as e:
                log(f"Could not load the {self.record.label} as kept in RAM ({e}); loading its plan and refitting it.")
                refits.drop(self.record.key, role)
                engine.unload()
        return engine.load()

    def _kept(self):
        """This engine as kept in RAM (refits.Kept), if what's kept holds the weights wanted now. One kept
        for other weights is dropped: the engine will be kept again, as it is then, when it's next unloaded."""
        entry = refits.get(self.record.key)
        if entry is None:
            return None
        wanted = self.weights.signature
        if "unet" in entry.blobs and all(wanted(name) == sig for sigs in entry.signatures.values() for name, sig in sigs.items()):
            return entry
        refits.drop(self.record.key)
        return None

    def _keep(self):
        """Before the engines unload: they go to RAM as they are, weights and all (refits.py), unless they're
        kept with these weights already, the weights aren't the loaded checkpoint's anymore, or RAM is short."""
        loaded = {role: e for role, e in (("unet", self.engine), ("shallow", self.shallow)) if e is not None and e.loaded and self.applied.get(role)}
        if "unet" not in loaded or not self._current():
            why = "it holds no weights" if "unet" not in loaded else "its weights aren't the loaded checkpoint's"
            log(f"Not keeping the {self.record.label} in RAM with its weights: {why}.")
            return
        key, label = self.record.key, f"{self.record.arch.label} {self.record.label}"
        entry = refits.get(key)
        if entry is not None and entry.signatures.get("unet") != self.applied["unet"]:
            refits.drop(key)
            entry = None
        todo = [role for role in loaded if entry is None or entry.signatures.get(role) != self.applied[role]]
        if not todo:
            return
        if not refits.room(sum(self._serialized_size(role) for role in todo), key):
            log(f"Not keeping the {label} in RAM with its weights ({refits.explain()}). Loading it again will refit it.")
            return
        start = time.time()
        torch.cuda.current_stream().synchronize()  # no inference in flight
        blobs = dict(entry.blobs) if entry is not None else {}
        signatures = dict(entry.signatures) if entry is not None else {}
        for role in todo:
            blobs[role], signatures[role] = loaded[role].serialize(), dict(self.applied[role])
        kept = refits.Kept(signatures, blobs, label, self._source())
        refits.put(key, kept)
        count, total = refits.held()
        log(f"Kept the {label} in RAM with its weights ({common.gib(kept.nbytes)}) in {time.time() - start:.1f}s; {count} kept, {common.gib(total)}.")

    def _serialized_size(self, role: str) -> int:
        """About what serializing an engine with its weights takes in RAM: its plan (kernels and all, without
        weights) and its weights."""
        plan, weights = (self.record.plan, self.record.weight_bytes) if role == "unet" else (self.record.shallow_plan, self.hacks["weight_bytes"])
        try:
            return weights + os.path.getsize(plan)
        except OSError:
            return weights

    def _source(self):
        """() -> the UNet module the engine's weights are from, or None once it's gone: its signatures' weak
        reference to it."""
        return next(iter(self.applied["unet"].values()))[0][0].ref

    def _current(self) -> bool:
        """Whether the weights in the engine are the loaded checkpoint's UNet's (another checkpoint's
        would never be asked for again)."""
        try:
            module = self._source()()
            return module is not None and module is arch_mod.unet_patcher().model.diffusion_model
        except Exception:
            return False

    def load_shallow(self):
        """The shallow engine, for an engine loaded without it, with room made for what it takes (full)."""
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
        log(f"Loaded the Speed Hacks part of the {self.owner} in {time.time() - start:.1f}s ({common.gib(gpu.held(self.owner))} in VRAM with the engine)")

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all and self.opened:
            common.memory(f"unloading the {self.record.label}: before")
            if self.keep and self.loaded:
                try:
                    self._keep()
                except Exception as e:  # unloads all the same: loading again refits
                    log(f"Could not keep the {self.record.label} in RAM with its weights: {e}")
            if self.shallow is not None:
                self.shallow.unload()
            self.engine.unload()
            self.applied, self.shallow_failed = {}, False
            left = gpu.ledger(self.owner)
            if left:
                log(f"TensorRT still holds {common.gib(left)} of the {self.owner}'s memory after unloading it.")
            gpu.forget(self.owner)
            common.memory(f"unloading the {self.record.label}: done")
        return self.model

    def sync(self, quiet=False):
        """Refits the weights whose source changed, in each loaded engine: all of them for a new engine or
        checkpoint, the patched ones when LoRAs change. Returns what _refit_times reports, or None if
        nothing changed."""
        weights = self.weights
        weights.refresh()
        total, start = None, time.time()
        for role, engine in (("unet", self.engine), ("shallow", self.shallow)):
            if engine is None or not engine.loaded:
                continue
            # The UNet's weights; anything else refittable keeps what the plan holds.
            wanted = {name: weights.signature(name) for name in engine.weight_names if weights.has(name)}
            applied = self.applied.get(role, {})
            changed = [name for name in wanted if applied.get(name) != wanted[name]]
            if not changed:
                continue
            self.applied[role] = {}
            refits.drop(self.record.key, role)  # what's kept in RAM of it holds other weights than it will now
            torch.cuda.current_stream().synchronize()  # no inference in flight while the weights change
            seconds = engine.refit(changed, weights.tensor, label=f"{self.record.label} ({role})")
            self.applied[role] = wanted
            refit = {**seconds, "count": len(changed), "patched": sum(weights.is_patched(name) for name in changed)}
            total = refit if total is None else {k: total[k] + refit[k] for k in refit}
        if total is not None and not quiet:
            log(f"Refitted {total['count']} weights in {time.time() - start:.1f}s: {_refit_times(total)}")
        return total


def _refit_times(refit: dict) -> str:
    patched = f", {refit['patched']} with LoRAs" if refit["patched"] else ""
    return f"{refit['count']} weights {refit['weights']:.1f}s{patched}, TensorRT refit {refit['tensorrt']:.1f}s"


def _loaded_entries(*models):
    """The manager's entries for these models (or clones of them)."""
    return [lm for lm in common.mm().current_loaded_models if lm.model is not None and any(lm.model is m or m.is_clone(lm.model) for m in models)]


@inference_mode
def _unload(model: TrtModel, keep: bool = True):
    """Unloads an engine: kept in RAM as it is, to come back without a refit (refits.py), unless `keep` is
    False (it's being cleared out, not making room)."""
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
    """Ends the sampling pass running on an engine, if any: its hooks on the model manager go."""
    global _pass
    p, _pass = _pass, None
    if p is not None:
        p.end()


def release_all():
    end_pass()  # a generation that failed mid-pass left its hooks in place
    for model in list(_models.values()):
        _unload(model, keep=False)
    _models.clear()
    _last_used.clear()
    vae_mod.release_all()
    refits.clear()
    common.mm().soft_empty_cache(force=True)


def model_for(record: store.Record) -> TrtModel:
    model = _models.get(record.key)
    if model is None or model.record.meta != record.meta:
        if model is not None:  # built again
            _unload(model, keep=False)
            refits.drop(record.key)
        model = _models[record.key] = TrtModel(record)
    return model


AUTOMATIC, SPEED_HACKS = "Automatic", "Speed Hacks"


def setting() -> str:
    """SD Unet: "Automatic" runs the loaded checkpoint's architecture's engines, "Speed Hacks" runs them with
    Speed Hacks (speed.py), anything else (reForge's "None") runs PyTorch."""
    try:
        from modules import shared

        return str(shared.opts.sd_unet)
    except Exception:
        return AUTOMATIC


# Settings (Stable Diffusion, TensorRT; ui.py adds them): the percentage of a pass's steps that Speed Hacks
# run in full at its start and at its end.
FULL_START, FULL_END = "trt_speed_hacks_full_start", "trt_speed_hacks_full_end"


def speed_settings() -> tuple:
    """(share at the start, share at the end), as speed.Schedule takes them."""
    from modules import shared

    def share(key, default):
        try:
            return min(max(float(getattr(shared.opts, key)) / 100, 0.0), 1.0)
        except Exception:  # not registered (e.g. a settings UI that failed to load)
            return default

    return share(FULL_START, speed.WARMUP), share(FULL_END, speed.TAIL)


class SpeedHacksOption(sd_unet.SdUnetOption):
    """SD Unet's "Speed Hacks" entry. "Automatic" needs none: the engines run whenever it's chosen."""

    label = SPEED_HACKS
    model_name = None  # never what "Automatic" picks

    def create_unet(self):
        return TrtUnet(self)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, option: SpeedHacksOption):
        super().__init__()
        self.option = option

    def forward(self, *args, **kwargs):
        raise RuntimeError("TensorRT engines run through the UNet's own forward")


def list_unets(options: list):
    options.append(SpeedHacksOption())
    try:  # an engine per architecture had an entry of its own before ("[TRT] SDXL (...)")
        from modules import shared

        saved = shared.opts.data.get("sd_unet")
        if isinstance(saved, str) and saved.startswith("[TRT] "):
            shared.opts.data["sd_unet"] = AUTOMATIC
            log(f"SD Unet was set to {saved}; engines don't have entries of their own anymore, so it's {AUTOMATIC} now.")
    except Exception:
        pass


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
    """Model loads in here don't unload these models: unloading the UNet mid-pass would undo its patches
    (e.g. its model sampling), which the pass still needs while the engine runs."""
    mm = common.mm()
    original = mm.free_memory

    def free_memory(memory_required, device, keep_loaded=[]):
        return original(memory_required, device, keep_loaded=list(keep_loaded) + _loaded_entries(*models))

    mm.free_memory = free_memory
    try:
        yield
    finally:
        mm.free_memory = original


_evicting = 0  # > 0 while loads may evict engines like any other model (see _evictable)


@contextlib.contextmanager
def _evictable():
    """Loads in here evict engines the way reForge evicts any model: a pass on PyTorch's UNet doesn't use
    them, and unloading every model means them too."""
    global _evicting
    _evicting += 1
    try:
        yield
    finally:
        _evicting -= 1


def _wrap_free_memory(original):
    """reForge makes room for a load by evicting models until its estimate of what the load needs fits. An
    engine is the costliest model to bring back (every weight refitted, where PyTorch's UNet is copied
    back), so it's evicted last, and only if that makes the estimate fit: a worst-case estimate that nothing
    could meet (e.g. the VAE decode's, at large sizes or in FP32) no longer takes it along. The load makes do
    with what's free, as it does anyway when the estimate isn't met (reForge's VAE decode loads the VAE
    partially, and decodes in tiles if it runs out)."""

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
    """reForge counts free VRAM as what the driver has free plus what PyTorch holds unused (its reserved
    less its active). With --cuda-malloc PyTorch reads both off the driver's pool, whose count of memory in
    use has been seen above what it holds (by 1.3 GiB: gpu.py takes TensorRT off that pool's books, which is
    where it came from; this is for whatever else may do it). reForge then counts that much less free than
    there is: it loaded its UNet partially where it fit, and gave its VAE 128 MiB. Never less than the
    driver's own free memory: that much is free whatever PyTorch counts."""

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
    """One sampling pass on an architecture's engines: each UNet call runs in the profile that fits it best,
    on the engine holding it, loaded (in place of another) when it isn't. A merged engine holds every
    profile, so it only ever changes profiles. With Speed Hacks, each evaluation's UNet calls run on the
    engine or its shallow partner as `schedule` says (speed.py). On anything no engine can run, the rest of
    the pass falls back to PyTorch, with the UNet loaded the way reForge would have."""

    def __init__(self, records: list, patcher, x, hacks: bool = False):
        self.records, self.patcher = records, patcher
        self.schedule = speed.Schedule(*speed_settings()) if hacks else None
        self.slots = store.slots(records)  # (record, optimization profile in it), in the profiles' order
        self.model = None
        self.weights = EffectiveWeights(patcher)
        self.module = patcher.model.diffusion_model
        arch = records[0].arch
        self.arch_id = arch.id
        self.shapes, self.takes_y = [r.profile_shapes[j] for r, j in self.slots], bool(arch.adm_channels)
        self.channels, self.context_dim = arch.in_channels, arch.context_dim
        self.shape = (x.shape[0] * 2, *x.shape[1:])
        self.fallback = None
        self.checked = self.shallow_checked = False  # whether the engine's, and its shallow partner's, first output was finite
        self._keep = None
        self._hint_batching()

    def use(self, record):
        """Loads the engine for `record` with the pass's weights, unloading any other engine first: one
        engine in VRAM at a time."""
        model = model_for(record)
        if model is not self.model or not model.loaded:
            for other in list(_models.values()):
                if other is not model and other.loaded:
                    _unload(other)
            self._hold(None)
            model.weights = self.weights
            model.want_shallow = model.want_shallow or (self.schedule is not None and model.hacks is not None and not model.shallow_failed)
            with torch.inference_mode(), _keeping(self.patcher):
                common.register(model)  # it makes exactly its room as it loads
            self.model = model
            # Other loads mid-pass (e.g. a live preview's decoder) unload other models first: unloading
            # the engine would mean reloading it on the next step, the UNet would lose its patches.
            self._hold(_keeping(model, self.patcher))
        _last_used[self.arch_id] = record.key
        if self.schedule is not None and model.hacks and not model.shallow_loaded and not model.shallow_failed:
            model.want_shallow = True
            try:
                with torch.inference_mode():
                    model.load_shallow()
            except Exception as e:  # the pass runs in full
                log(f"Could not load the Speed Hacks part of the {record.label} ({e}); running without it.")
        model.sync()
        return model

    def preferred(self) -> set:
        """The profiles of the engine this batch's earlier passes ran on, which this pass stays on while it
        fits, even if another engine's profile suits the sizes better: loading another means bringing all
        its weights to VRAM (from RAM as it was kept, or by refitting them). A merged engine holds them all,
        so it changes to whichever fits best."""
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
        if self.schedule is not None:  # its deep features' VRAM, for PyTorch or whatever runs next
            self.schedule.release()

    def _hint_batching(self):
        """reForge decides how many conds (cond and uncond) to batch into one UNet call by the PyTorch
        UNet's memory estimate. With the engine's activations already allocated, batch whatever a profile
        runs: the whole batch, and prompts padded to the length reForge would pad them to (their least
        common multiple). Batches no profile runs are split, so each prompt runs at its own length."""
        shapes = self.shapes

        def memory_required(input_shape, cond_shapes={}):
            b, _, h, w = input_shape[:4]
            lengths = [s[1] for s in cond_shapes.get("c_crossattn", ()) if len(s) > 1]
            context = math.lcm(*lengths) if lengths else profiles.CONTEXT_PER_CHUNK
            return 0 if profiles.select(shapes, b, h, w, context) is not None else float("inf")

        self.patcher.model.memory_required = memory_required

    def end(self):
        """Undoes what the pass changed; the engine stays loaded for the next one."""
        common.phase("UNet pass: ended, before its memory is released")
        self._restore()
        common.memory("sampling pass ended")
        from .engine import profile_report

        profile_report("sampling pass")
        hacks = self.schedule
        if hacks is not None and hacks.shallow_calls:
            log(f"Speed Hacks: {hacks.shallow_calls} of {hacks.shallow_calls + hacks.full_calls} UNet calls ran on the shallow engine.")
            _note_speed_hacks()
        for model in _models.values():
            if model.weights is self.weights:
                model.weights = None  # don't keep this checkpoint alive

    def fall_back(self, reason: str):
        self.fallback = reason
        notify(f"{reason}; finishing this pass with PyTorch.")
        self._restore()  # PyTorch's UNet may need the engine's VRAM
        _load_torch_unet(self.patcher, self.shape)
        raise Fallback(reason)

    def run(self, x, timesteps, context, y):
        b, c, h, w = x.shape
        if c != self.channels or context.shape[2] != self.context_dim:
            self.fall_back(f"The UNet got {c} input channels and {context.shape[2]}-wide prompts, the engine takes {self.channels} and {self.context_dim}")
        choice = profiles.select(self.shapes, b, h, w, context.shape[1], prefer=self.preferred())
        if choice is None:
            tokens = context.shape[1] // profiles.CONTEXT_PER_CHUNK * profiles.TOKENS_PER_CHUNK
            self.fall_back(
                f"No profile of the engine runs {w * 8}x{h * 8} with a UNet batch of {b} and {tokens}-token prompts "
                "(reForge pads the prompt and negative prompt to a common length)"
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
        if self.model is None or self.model.record.key != record.key or not self.model.loaded:  # another profile, or unloaded mid-pass (e.g. for a VAE encode)
            try:
                if self.model is not None and self.model.loaded:
                    self.model.engine.release_activations()
                self.use(record)
            except Exception as e:
                if not isinstance(e, RuntimeError):
                    traceback.print_exc()
                self.fall_back(f"Could not load the {record.label} ({e})")
        stream = torch.cuda.current_stream().cuda_stream
        slot = deep = None
        if self.schedule is not None and self.model.shallow_loaded:
            slot, deep = self.schedule.call(feed["sample"].shape)
        out = self._infer_with_room(feed, profile, stream, slot, deep)
        if deep is None and not self.checked:
            self.checked = True
            if not torch.isfinite(out).all():
                self.fall_back("The engine produced NaN/Inf values")
        if deep is not None and not self.shallow_checked:
            self.shallow_checked = True
            if not torch.isfinite(out).all():
                notify("The Speed Hacks part of the engine produced NaN/Inf values; running this pass without Speed Hacks")
                self.schedule = None
                out = self._infer_with_room(feed, profile, stream)
        return out[:b].to(x.dtype)

    def _infer_with_room(self, feed, profile, stream, slot=None, deep=None):
        """_infer, again after unloading other models if it ran out of VRAM."""
        try:
            return self._infer(feed, profile, stream, slot, deep)
        except torch.cuda.OutOfMemoryError:
            self.model.engine.release_activations()
            common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, self.model.load_device, keep_loaded=_loaded_entries(self.model, self.patcher))
            common.mm().soft_empty_cache(force=True)
            try:
                return self._infer(feed, profile, stream, slot, deep)
            except torch.cuda.OutOfMemoryError:
                self.fall_back("Out of VRAM running the engine")

    def _infer(self, feed, profile, stream, slot=None, deep=None):
        """The engine's output, or with a deep feature, its shallow partner's. The engine's deep feature is
        kept for the call's slot."""
        try:
            if deep is not None:
                return self.model.shallow.infer({**feed, "deep": deep}, profile, stream)["latent"]
            outputs = self.model.engine.infer(feed, profile, stream)
            if slot is not None and "deep" in outputs:
                self.schedule.keep(slot, feed["sample"].shape, outputs["deep"])
            return outputs["latent"]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            traceback.print_exc()
            self.fall_back(f"The engine failed ({e})")


def _engine_records(patcher, x):
    """(the engines to run this pass on, whether with Speed Hacks), or (None, False) for PyTorch."""
    chosen = setting()
    if chosen not in (AUTOMATIC, SPEED_HACKS):
        return None, False
    arch = arch_mod.detect(patcher)
    records = store.engines(arch.id) if arch is not None else []
    if not records:  # e.g. a refiner of another architecture: with SD Unet on Automatic, that's expected
        if chosen == SPEED_HACKS:
            notify(f"There are no TensorRT engines for {arch.label if arch else 'this model'}; using PyTorch. Build them in the TensorRT tab.")
        return None, False
    reason = _unsupported(patcher)
    if reason:
        notify(f"TensorRT engines can't apply {reason}; using PyTorch.")
        return None, False
    b, _, h, w = x.shape
    if not profiles.covers_size([s for r in records for s in r.profile_shapes], h, w, b):
        notify(f"No profile of the {arch.label} engines covers {w * 8}x{h * 8} at batch size {b}; using PyTorch for this pass. Add one in the TensorRT tab.")
        return None, False
    hacks = chosen == SPEED_HACKS and any(r.speed_hacks for r in records)
    if chosen == SPEED_HACKS:
        outdated = sum(r.outdated for r in records)
        failed = sum(not r.speed_hacks and not r.outdated for r in records)
        if outdated:
            notify(f"{outdated} of the {arch.label} engines were built before Speed Hacks, and run without them; build them again in the TensorRT tab.")
        if failed:
            retry = " Building them again in the TensorRT tab retries once." if any(r.retry_speed_hacks for r in records) else ""
            notify(f"Speed Hacks couldn't be built for {failed} of the {arch.label} engines (the build's console output says why), so they run without them.{retry}")
    return records, hacks


def _begin(records, hacks, patcher, x, prepare):
    """sampling_prepare for a pass on the engines: the Pass, or None if it runs on PyTorch (said why). The
    engine for the likeliest sizes (cond and uncond batched, the shortest prompts) loads now, the others
    when needed; if it and the pass's calls can't fit (_fits), no weights go to VRAM at all."""

    mm = common.mm()
    with torch.inference_mode():
        _unload_torch_unet(patcher)
        state = mm.vram_state, mm.free_memory
        mm.vram_state, mm.free_memory = mm.VRAMState.NO_VRAM, _no_free_memory  # patches applied, weights stay in RAM
        try:
            prepare(patcher, x)
        finally:
            mm.vram_state, mm.free_memory = state
        p = Pass(records, patcher, x, hacks)
        try:
            b, _, h, w = x.shape
            prefer = p.preferred()
            choice = profiles.select(p.shapes, 2 * b, h, w, profiles.CONTEXT_PER_CHUNK, prefer) or profiles.select(p.shapes, b, h, w, profiles.CONTEXT_PER_CHUNK, prefer)
            index = choice[0] if choice else next(i for i, s in enumerate(p.shapes) if profiles.covers_size([s], h, w, b))
            record = p.slots[index][0]
            _fits(model_for(record), p, x)
            model = p.use(record)
            _make_room_for_activations(model, patcher, x)
            common.phase(f"UNet pass on the {record.label}: loaded, before the first step")
        except Exception as e:
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
            notify(f"Could not run the engine ({e}); using PyTorch.")
            p.end()
            _load_torch_unet(patcher, (x.shape[0] * 2, *x.shape[1:]))
            return None
    return p


def _sizes(x) -> str:
    return f"{x.shape[3] * 8}x{x.shape[2] * 8}, batch size {x.shape[0]}"


def _fits(model: TrtModel, p: Pass, x):
    """Before anything loads: raises DoesNotFit, with the figures, if the engine and the working memory of a
    pass's UNet calls at these sizes can't be in VRAM whatever else is evicted; says them if they can. The
    engine is what TrtModel.full gives, or what it holds if it's loaded; the calls are the image's size
    (call_bytes), or what TensorRT turned out to take at these sizes this session, where that didn't fit."""
    shallow = bool(model.hacks and (model.want_shallow or (p.schedule is not None and not model.shallow_failed)))
    calls = max(call_bytes(x), model.took.get(tuple(x.shape[i] for i in (0, 2, 3)), 0))
    held = gpu.held(model.owner)
    if model.loaded:
        engine = held + (model._serialized_size("shallow") if shallow and not model.shallow_loaded else 0)
    else:
        model.weights = p.weights
        engine = model.full(shallow, model._kept())
    room = common.obtainable(model.load_device, keep=[model]) - common.reserve() + held
    need = held + common.padded(engine - held + calls)  # the buffer on what isn't in VRAM yet
    said = (
        f"the {model.record.label} at {_sizes(x)} needs {common.gib(need)} of VRAM ({common.gib(engine)} for the engine, "
        f"{common.gib(calls)} for its calls, {common.gib(need - engine - calls)} of buffer), and there can be {common.gib(room)}"
    )
    if need > room:
        raise DoesNotFit(said)
    log(f"It fits: {said}.")


def _make_room_for_activations(model: TrtModel, patcher, x):
    """Makes room for what the pass's UNet calls hold beside the engine, which is TensorRT's to say now that
    the engine is loaded (its activation memory is allocated as it asks). Where that's well off what the
    image's size gives (call_bytes), both are said. Raises RuntimeError if there can't be that much: the
    pass runs on PyTorch before it starts, and these sizes are weighed by it from now on (_fits)."""
    try:
        takes = _call_bytes(model, x)
    except Exception as e:
        log(f"Could not size the engine's calls: {e}")
        return
    if not takes:
        return
    counted = call_bytes(x)
    if abs(takes - counted) > 0.15 * counted:
        most = " (the most its profile's sizes take: it gave no figure for these)" if model.engine.size_is_most else ""
        log(f"TensorRT takes {common.gib(takes)} for the {model.record.label}'s calls at {_sizes(x)}{most}; their size gives {common.gib(counted)}.")
    need = common.padded(takes)
    free = common.make_room(need, model.load_device, _loaded_entries(model, patcher))
    if free < need:
        model.took[tuple(x.shape[i] for i in (0, 2, 3))] = takes
        raise RuntimeError(f"a pass at {_sizes(x)} holds {common.gib(need)} of VRAM beside the {model.record.label}, and there can be {common.gib(free)}")


def _call_shapes(model: TrtModel, x):
    """(The profile of the engine a UNet call at the likeliest shapes runs in (cond and uncond batched, or
    one at a time if no profile can batch them), its inputs' shapes there), or None if none runs it."""
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
    """What a UNet call at the likeliest shapes holds beside the loaded engine, as TensorRT has it: its
    activation memory, its inputs as the engine takes them and its outputs; with Speed Hacks, also the deep
    feature kept from the call before."""
    found = _call_shapes(model, x)
    if found is None:
        return 0
    profile, shapes = found
    act = model.engine.estimate_activation_bytes(profile, shapes, torch.cuda.current_stream().cuda_stream)
    io = model.engine.io_bytes()
    return act + sum(io.values()) + io.get("deep", 0)


def _wrap_forward(original):
    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        p = _pass
        if p is not None and p.fallback is None and self is p.module:
            try:
                if control is not None:
                    p.fall_back("TensorRT engines can't apply ControlNet")
                if any(v for g in ("patches", "patches_replace") for v in transformer_options.get(g, {}).values()):
                    p.fall_back("TensorRT engines can't apply UNet patches")
                return p.run(x, timesteps, context, y)
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
        if records is None:  # PyTorch's UNet: engines can make room for it as usual
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


def _wrap_denoiser(original):
    """reForge's CFG denoiser runs one evaluation of the sampler (its UNet calls for every cond): tells the
    pass's Speed Hacks schedule which one, out of how many, at which latent."""

    def forward(self, x, *args, **kwargs):
        p = _pass
        if p is not None and p.schedule is not None and p.fallback is None:
            try:
                from modules import shared

                p.schedule.begin(int(self.step), self.total_steps, shared.state.sampling_steps, x)
            except Exception as e:
                log(f"Speed Hacks stopped for this pass: {e}")
                p.schedule = None
        return original(self, x, *args, **kwargs)

    forward._trt_original = original
    return forward


def _wrap_vae_decode(original):
    """Decodes on the VAE decoder's engines when SD Unet runs the engines (vae.py), with the room it needs
    made by reForge's manager like any model's (the UNet engines are evicted last, and only if that makes
    the room); anything else decodes on PyTorch, as reForge would."""

    def decode(self, samples_in, vae_options={}):
        # Not from another thread: reForge's progress endpoint decodes "Full" live previews on its own,
        # while the generation's thread samples on the UNet engine.
        if not vae_options and setting() in (AUTOMATIC, SPEED_HACKS) and threading.get_ident() == _generation_thread:
            try:
                pixels = vae_mod.decode(self, samples_in)
                if pixels is not None:
                    return pixels
            except Exception as e:
                traceback.print_exc()
                notify(f"The VAE decoder engines failed ({e}); decoding on PyTorch")
        return original(self, samples_in, vae_options)

    decode._trt_original = original
    return decode


def _wrap_vae_encode(original):
    """Encodes on the VAE encoder's engines likewise (an img2img's image, a Hires. fix's upscaled one);
    anything they pass on encodes on PyTorch, as reForge would."""

    def encode(self, pixel_samples):
        if setting() in (AUTOMATIC, SPEED_HACKS) and threading.get_ident() == _generation_thread:
            try:
                latents = vae_mod.encode(self, pixel_samples)
                if latents is not None:
                    return latents
            except Exception as e:
                traceback.print_exc()
                notify(f"The VAE encoder engines failed ({e}); encoding on PyTorch")
        return original(self, pixel_samples)

    encode._trt_original = original
    return encode


def _rebind(owner, name, wrap):
    """Wraps owner.name, also where modules imported it by name."""
    current = getattr(owner, name)
    if hasattr(current, "_trt_original"):
        return
    replacement = wrap(current)
    setattr(owner, name, replacement)
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get(name) is current:
            setattr(module, name, replacement)


def install_hooks():
    """Idempotent."""
    from ldm_patched.ldm.modules.diffusionmodules import openaimodel
    from modules_forge import forge_sampler

    if not hasattr(openaimodel.UNetModel.forward, "_trt_original"):
        openaimodel.UNetModel.forward = _wrap_forward(openaimodel.UNetModel.forward)
    _rebind(forge_sampler, "sampling_prepare", _wrap_prepare)
    _rebind(forge_sampler, "sampling_cleanup", _wrap_cleanup)
    _rebind(common.mm(), "free_memory", _wrap_free_memory)
    _rebind(common.mm(), "get_free_memory", _wrap_get_free_memory)
    _rebind(common.mm(), "unload_all_models", _wrap_unload_all)
    try:  # the VAE's engines: it decodes and encodes on PyTorch without them
        from ldm_patched.modules import sd as ldm_sd

        if not hasattr(ldm_sd.VAE.decode, "_trt_original"):
            ldm_sd.VAE.decode = _wrap_vae_decode(ldm_sd.VAE.decode)
        if not hasattr(ldm_sd.VAE.encode, "_trt_original"):
            ldm_sd.VAE.encode = _wrap_vae_encode(ldm_sd.VAE.encode)
    except Exception as e:
        log(f"Could not hook reForge's VAE, so it decodes and encodes on PyTorch: {e}")
    try:  # Speed Hacks only: the engines run without it
        from modules import sd_samplers_cfg_denoiser

        denoiser = sd_samplers_cfg_denoiser.CFGDenoiser
        if not hasattr(denoiser.forward, "_trt_original"):
            denoiser.forward = _wrap_denoiser(denoiser.forward)
    except Exception as e:
        log(f"Could not hook reForge's CFG denoiser, so Speed Hacks run every evaluation in full: {e}")


def new_batch():
    """A batch's first pass runs on the engine whose profile fits it best, and its later passes (e.g. its
    Hires. fix) stay on that engine while it covers their sizes. The next batch starts afresh, so nothing
    keeps running on an engine picked for other sizes."""
    _last_used.clear()


_generation = None  # weakref to the generation's processing object, for its infotext
_generation_thread = None  # the thread generations run in


def _note_speed_hacks():
    p = _generation() if _generation is not None else None
    if p is not None:
        p.extra_generation_params["TensorRT"] = SPEED_HACKS


def new_generation(p=None):
    global _generation, _generation_thread
    end_pass()  # a generation that failed mid-pass never cleaned up
    common.new_generation()
    _generation = weakref.ref(p) if p is not None else None
    _generation_thread = threading.get_ident()
    if setting() not in (AUTOMATIC, SPEED_HACKS):  # SD Unet is None: give their VRAM back, and the RAM
        for model in list(_models.values()):
            if model.loaded:
                _unload(model, keep=False)
        vae_mod.release_all()
        refits.clear()
    else:
        refits.trim()
    try:
        install_hooks()
    except Exception as e:
        log(f"Could not hook reForge's sampler: {e}")
