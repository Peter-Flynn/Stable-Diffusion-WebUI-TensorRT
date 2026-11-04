"""Running reForge's UNet on a TensorRT engine.

An architecture has an engine per profile, or one merged engine holding them all (store.py), and a sampling
pass loads the one for its sizes; only one is in VRAM at a time. Each is a model of reForge's own model
manager, so the manager makes room for it and loads it back for sampling like any other model, but evicts
it only as a last resort (_wrap_free_memory): bringing an engine back means refitting every weight. While a
sampling pass runs on an engine, the PyTorch UNet stays in RAM (its patches still apply, it just isn't
loaded)."""

import contextlib
import math
import sys
import time
import traceback
import weakref

import torch

from modules import sd_unet

from . import arch as arch_mod
from . import common, profiles, speed, store
from . import vae as vae_mod
from . import weights as weights_mod
from .common import inference_mode, log, notify
from .weights import EffectiveWeights

HEADROOM = 256 * 2**20
_models = {}  # Record.key -> TrtModel
_last_used = {}  # arch id -> Record.key of the engine the batch being generated last ran on (see new_batch)
_pass = None  # the sampling pass running on an engine


class Fallback(Exception):
    pass


class TensorRTUNet:
    """What the model manager holds for an engine (its name shows in reForge's log)."""


class TrtModel:
    """An engine as reForge's model manager sees a model (the parts of ModelPatcher it uses), with its
    shallow partner for Speed Hacks (speed.py), loaded only for passes that use them. The two share their
    activation memory, since they never run at once."""

    parent = None

    def __init__(self, record: store.Record):
        mm = common.mm()
        self.record = record
        self.hacks = record.speed_hacks  # read once: it checks the disk, and model_size() is called often
        self.model = TensorRTUNet()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.engine = self.shallow = None
        self.want_shallow = False  # load the shallow engine with this one
        self.shallow_failed = False  # its load failed: not tried again until this engine loads again
        self.weights = None  # EffectiveWeights the engines should run with
        self.applied = {}  # "unet" / "shallow" -> {weight name: signature of what's in that engine}
        self.measured = 0

    def _engines(self):
        if self.engine is None:
            from .engine import Arena, Engine

            arena = Arena()
            self.engine = Engine(self.record.plan, arena)
            if self.hacks:
                self.shallow = Engine(self.record.shallow_plan, arena)
        return self.engine

    @property
    def loaded(self) -> bool:
        return self.engine is not None and self.engine.loaded

    @property
    def shallow_loaded(self) -> bool:
        return self.shallow is not None and self.shallow.loaded

    def model_size(self):
        return self.measured or self.record.weight_bytes + (self.hacks["weight_bytes"] if self.hacks and self.want_shallow else 0)

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
        if self.loaded:
            return 0
        if self.weights is None:  # a load outside a sampling pass: the loaded checkpoint's weights
            patcher = arch_mod.unet_patcher()
            if patcher is None:
                raise RuntimeError("no checkpoint to take the engine's weights from")
            self.weights = EffectiveWeights(patcher)
        start, free, reserved = time.time(), torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
        engine = self._engines()
        plan_seconds = engine.load()
        if self.want_shallow and self.shallow is not None:
            try:
                plan_seconds += self.shallow.load()
            except Exception as e:  # the engine runs without it
                log(f"Could not load the Speed Hacks part of the {self.record.label} ({e}); running without it.")
                self.shallow.unload()
                self.want_shallow, self.shallow_failed = False, True
        try:
            self.applied = {}
            refit = self.sync(quiet=True)
        except Exception:
            self.detach()
            raise
        # TensorRT's own allocations, not what PyTorch cached meanwhile (e.g. computing LoRA weights)
        used = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
        self.measured = max(self.measured, used)
        with_shallow = " with its Speed Hacks part" if self.shallow_loaded else ""
        log(
            f"Loaded the {self.record.arch.label} {self.record.label}{with_shallow} ({common.gib(self.model_size())}) in {time.time() - start:.1f}s: "
            f"plan {plan_seconds:.1f}s, " + (_refit_times(refit) if refit else "no weights to refit")
        )
        return self.model_size()

    def load_shallow(self):
        """The shallow engine, for an engine loaded without it."""
        if not self.loaded or self.shallow is None or self.shallow.loaded:
            return
        start, free, reserved = time.time(), torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
        try:
            self.shallow.load()
            self.applied.pop("shallow", None)
            self.sync(quiet=True)
        except Exception:
            self.shallow.unload()
            self.want_shallow, self.shallow_failed = False, True
            raise
        used = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
        self.measured += max(used, 0)
        log(f"Loaded the Speed Hacks part of the {self.record.arch.label} {self.record.label} in {time.time() - start:.1f}s")

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all and self.loaded:
            if self.shallow is not None:
                self.shallow.unload()
            self.engine.unload()
            self.applied, self.shallow_failed = {}, False
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
            weights.prepare(changed)
            reused = weights.reused
            torch.cuda.current_stream().synchronize()  # no inference in flight while the weights change
            seconds = engine.refit(changed, weights.tensor)
            self.applied[role] = wanted
            refit = {**seconds, "count": len(changed), "patched": sum(weights.is_patched(name) for name in changed), "reused": weights.reused - reused}
            total = refit if total is None else {k: total[k] + refit[k] for k in refit}
        if total is not None and not quiet:
            log(f"Refitted {total['count']} weights in {time.time() - start:.1f}s: {_refit_times(total)}")
        return total


def _refit_times(refit: dict) -> str:
    patched = f", {refit['patched']} with LoRAs" if refit["patched"] else ""
    reused = f", {refit['reused']} computed before (kept in RAM)" if refit.get("reused") else ""
    staged = f" ({common.gib(refit['staged'])} of the weights from VRAM)" if refit.get("staged") else ""
    return f"{refit['count']} weights {refit['weights']:.1f}s{patched}{reused}, TensorRT refit {refit['tensorrt']:.1f}s{staged}"


def _loaded_entries(*models):
    """The manager's entries for these models (or clones of them)."""
    return [lm for lm in common.mm().current_loaded_models if lm.model is not None and any(lm.model is m or m.is_clone(lm.model) for m in models)]


@inference_mode
def _unload(model: TrtModel):
    mm = common.mm()
    for i in reversed(range(len(mm.current_loaded_models))):
        if mm.current_loaded_models[i].model is model:
            mm.current_loaded_models.pop(i).model_unload()
    model.detach()


def end_pass():
    """Ends the sampling pass running on an engine, if any: its hooks on the model manager go."""
    global _pass
    p, _pass = _pass, None
    if p is not None:
        p.end()


def release_all():
    end_pass()  # a generation that failed mid-pass left its hooks in place
    for model in list(_models.values()):
        _unload(model)
    _models.clear()
    _last_used.clear()
    vae_mod.release_all()
    weights_mod.clear_computed()
    common.mm().soft_empty_cache(force=True)


def model_for(record: store.Record) -> TrtModel:
    model = _models.get(record.key)
    if model is None or model.record.meta != record.meta:
        if model is not None:
            _unload(model)
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
            engines = [lm for lm in mm.current_loaded_models if isinstance(lm.model, TrtModel) and lm.device == device and lm not in keep_loaded]
        if not engines:
            return original(memory_required, device, keep_loaded)
        unloaded = original(memory_required, device, list(keep_loaded) + engines)
        short = memory_required - mm.get_free_memory(device)
        if 0 < short <= sum(lm.model_loaded_memory() for lm in engines):
            unloaded += original(memory_required, device, keep_loaded)
        return unloaded

    free_memory._trt_original = original
    return free_memory


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
                common.mm().load_models_gpu([model])
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
        fits, even if another engine's profile suits the sizes better: loading another means refitting
        every weight. A merged engine holds them all, so it changes to whichever fits best."""
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
        self._restore()
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
    """sampling_prepare for a pass on the engines: the Pass, or None if it fell back to PyTorch. Loads the
    engine for the likeliest sizes (cond and uncond batched, the shortest prompts) now, the others when needed."""

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
            model = p.use(p.slots[index][0])
            _make_room_for_activations(model, patcher, x)
        except Exception as e:
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
            notify(f"Could not run the engine ({e}); using PyTorch.")
            p.end()
            _load_torch_unet(patcher, (x.shape[0] * 2, *x.shape[1:]))
            return None
    return p


def _make_room_for_activations(model: TrtModel, patcher, x):
    try:
        need = _activation_estimate(model, x)
    except Exception as e:
        log(f"Could not estimate the engine's activation memory: {e}")
        return
    if need:
        common.mm().free_memory(need + HEADROOM, model.load_device, keep_loaded=_loaded_entries(model, patcher))


def _activation_estimate(model: TrtModel, x) -> int:
    b, c, h, w = x.shape
    held = model.record.profile_shapes
    # cond and uncond batched, or one at a time if no profile can batch them (e.g. at its largest batch)
    choice = profiles.select(held, 2 * b, h, w, profiles.CONTEXT_PER_CHUNK) or profiles.select(held, b, h, w, profiles.CONTEXT_PER_CHUNK)
    if choice is None:
        return 0
    profile, padded, _ = choice
    s = held[profile]
    shapes = {
        "sample": (padded, c, h, w),
        "timesteps": (padded,),
        "encoder_hidden_states": (padded, s["encoder_hidden_states"][1][1], s["encoder_hidden_states"][0][2]),
    }
    if "y" in s:
        shapes["y"] = (padded, s["y"][0][1])
    return model.engine.estimate_activation_bytes(profile, shapes, torch.cuda.current_stream().cuda_stream)


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
    """Decodes on the VAE decoder's engines when SD Unet runs the engines and they fit (vae.py), keeping
    the UNet engines loaded; anything else decodes on PyTorch, as reForge would."""

    def decode(self, samples_in, vae_options={}):
        if not vae_options and setting() in (AUTOMATIC, SPEED_HACKS):
            try:
                with _keeping(*[m for m in _models.values() if m.loaded]):
                    pixels = vae_mod.decode(self, samples_in)
                if pixels is not None:
                    return pixels
            except Exception as e:
                traceback.print_exc()
                notify(f"The VAE decoder engines failed ({e}); decoding on PyTorch")
        return original(self, samples_in, vae_options)

    decode._trt_original = original
    return decode


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
    _rebind(common.mm(), "unload_all_models", _wrap_unload_all)
    try:  # the VAE decoder's engines: decodes run on PyTorch without them
        from ldm_patched.modules import sd as ldm_sd

        if not hasattr(ldm_sd.VAE.decode, "_trt_original"):
            ldm_sd.VAE.decode = _wrap_vae_decode(ldm_sd.VAE.decode)
    except Exception as e:
        log(f"Could not hook reForge's VAE, so it decodes on PyTorch: {e}")
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


def _note_speed_hacks():
    p = _generation() if _generation is not None else None
    if p is not None:
        p.extra_generation_params["TensorRT"] = SPEED_HACKS


def new_generation(p=None):
    global _generation
    end_pass()  # a generation that failed mid-pass never cleaned up
    common.new_generation()
    _generation = weakref.ref(p) if p is not None else None
    if setting() not in (AUTOMATIC, SPEED_HACKS):  # SD Unet is None: give their VRAM back, and the RAM
        for model in list(_models.values()):
            if model.loaded:
                _unload(model)
        vae_mod.release_all()
        weights_mod.clear_computed()
    try:
        install_hooks()
    except Exception as e:
        log(f"Could not hook reForge's sampler: {e}")
