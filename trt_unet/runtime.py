"""Running reForge's UNet on a TensorRT engine.

An architecture has an engine per profile, and a sampling pass loads the one for its sizes; only one is in
VRAM at a time. Each is a model of reForge's own model manager, so the manager makes room for it, unloads it
when another model needs the VRAM and loads it back for sampling, like any other model. While a sampling
pass runs on an engine, the PyTorch UNet stays in RAM (its patches still apply, it just isn't loaded)."""

import contextlib
import math
import sys
import time
import traceback

import torch

from modules import sd_unet

from . import arch as arch_mod
from . import common, profiles, store
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
    """An engine as reForge's model manager sees a model (the parts of ModelPatcher it uses)."""

    parent = None

    def __init__(self, record: store.Record):
        mm = common.mm()
        self.record = record
        self.model = TensorRTUNet()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.engine = None
        self.weights = None  # EffectiveWeights the engine should run with
        self.applied = {}  # weight name -> signature of what's in the engine
        self.measured = 0

    def _engine(self):
        if self.engine is None:
            from .engine import Engine

            self.engine = Engine(self.record.plan)
        return self.engine

    @property
    def loaded(self) -> bool:
        return self.engine is not None and self.engine.loaded

    def model_size(self):
        return self.measured or self.record.weight_bytes

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
        engine = self._engine()
        plan_seconds = engine.load()
        try:
            self.applied = {}
            refit = self.sync(quiet=True)
        except Exception:
            engine.unload()
            raise
        # TensorRT's own allocations, not what PyTorch cached meanwhile (e.g. computing LoRA weights)
        used = free - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved)
        self.measured = max(self.measured, used)
        log(
            f"Loaded the {self.record.arch.label} engine for profile {self.record.index + 1} ({common.gib(self.model_size())}) in {time.time() - start:.1f}s: "
            f"plan {plan_seconds:.1f}s, " + (_refit_times(refit) if refit else "no weights to refit")
        )
        return self.model_size()

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all and self.loaded:
            self.engine.unload()
            self.applied = {}
        return self.model

    def sync(self, quiet=False):
        """Refits the weights whose source changed: all of them for a new engine or checkpoint, the
        patched ones when LoRAs change. Returns what _refit_times reports, or None if nothing changed."""
        engine, weights = self.engine, self.weights
        weights.refresh()
        # The UNet's weights; anything else refittable keeps what the plan holds.
        wanted = {name: weights.signature(name) for name in engine.weight_names if weights.has(name)}
        changed = [name for name in wanted if self.applied.get(name) != wanted[name]]
        if not changed:
            return None
        start = time.time()
        self.applied = {}
        torch.cuda.current_stream().synchronize()  # no inference in flight while the weights change
        seconds = engine.refit(changed, weights.tensor)
        self.applied = wanted
        refit = {**seconds, "count": len(changed), "patched": sum(weights.is_patched(name) for name in changed)}
        if not quiet:
            log(f"Refitted {len(changed)} weights in {time.time() - start:.1f}s: {_refit_times(refit)}")
        return refit


def _refit_times(refit: dict) -> str:
    patched = f", {refit['patched']} with LoRAs" if refit["patched"] else ""
    staged = f" ({common.gib(refit['staged'])} of the weights from VRAM)" if refit.get("staged") else ""
    return f"{refit['count']} weights {refit['weights']:.1f}s{patched}, TensorRT refit {refit['tensorrt']:.1f}s{staged}"


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
    common.mm().soft_empty_cache(force=True)


def model_for(record: store.Record) -> TrtModel:
    model = _models.get(record.key)
    if model is None or model.record.meta != record.meta:
        if model is not None:
            _unload(model)
        model = _models[record.key] = TrtModel(record)
    return model


class TrtUnetOption(sd_unet.SdUnetOption):
    """An architecture's engines."""

    def __init__(self, arch):
        self.arch = arch
        self.label = f"[TRT] {arch.label}"

    @property
    def model_name(self):
        """The loaded checkpoint if it has this architecture: what "Automatic" in SD Unet looks for."""
        try:
            sd_model = arch_mod.loaded_model()
            arch = arch_mod.detect(arch_mod.unet_patcher(sd_model)) if sd_model is not None else None
            info = getattr(sd_model, "sd_checkpoint_info", None)
            return getattr(info, "model_name", None) if arch is not None and arch.id == self.arch.id else None
        except Exception as e:  # reForge asks every option at the start of each generation
            log(f"Could not check the loaded model's architecture: {e}")
            return None

    def create_unet(self):
        return TrtUnet(self)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, option: TrtUnetOption):
        super().__init__()
        self.option = option

    def forward(self, *args, **kwargs):
        raise RuntimeError("TensorRT engines run through the UNet's own forward")

    def deactivate(self):
        for model in list(_models.values()):
            if model.record.arch.id == self.option.arch.id:
                _unload(model)


def list_unets(options: list):
    ours = [TrtUnetOption(arch) for arch in {r.arch.id: r.arch for r in store.records() if r.loadable}.values()]
    options.extend(ours)
    try:  # SD Unet naming an engine that's gone (deleted, or built for another TensorRT or GPU) would select nothing
        from modules import shared

        saved = shared.opts.data.get("sd_unet")
        if isinstance(saved, str) and saved.startswith("[TRT] ") and saved not in {o.label for o in ours}:
            shared.opts.data["sd_unet"] = "Automatic"
            log(f"SD Unet was set to {saved}, an engine that can't be used anymore; it's Automatic now.")
    except Exception:
        pass


def _explicit_choice() -> bool:
    try:
        from modules import shared

        return str(shared.opts.sd_unet).startswith("[TRT] ")
    except Exception:
        return True


def _selected_arch():
    unet = sd_unet.current_unet
    return unet.option.arch if isinstance(unet, TrtUnet) else None


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


@inference_mode
def _load_torch_unet(patcher, shape):
    estimate = patcher.model_options.get("memory_peak_estimation_modifier", patcher.memory_required)
    required = estimate(list(shape)) + getattr(patcher, "extra_preserved_memory_during_sampling", 0)
    common.mm().load_models_gpu([patcher], memory_required=required)


class Pass:
    """One sampling pass on an architecture's engines: each UNet call runs on the engine of the profile
    that fits it best, loaded (in place of another) when it isn't. On anything no engine can run, the rest
    of the pass falls back to PyTorch, with the UNet loaded the way reForge would have."""

    def __init__(self, records: list, patcher, x):
        self.records, self.patcher = records, patcher
        self.model = None
        self.weights = EffectiveWeights(patcher)
        self.module = patcher.model.diffusion_model
        arch = records[0].arch
        self.arch_id = arch.id
        self.shapes, self.takes_y = [r.shapes for r in records], bool(arch.adm_channels)
        self.channels, self.context_dim = arch.in_channels, arch.context_dim
        self.shape = (x.shape[0] * 2, *x.shape[1:])
        self.fallback = None
        self.checked = False
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
            with torch.inference_mode(), _keeping(self.patcher):
                common.mm().load_models_gpu([model])
            self.model = model
            # Other loads mid-pass (e.g. a live preview's decoder) unload other models first: unloading
            # the engine would mean reloading it on the next step, the UNet would lose its patches.
            self._hold(_keeping(model, self.patcher))
        _last_used[self.arch_id] = record.key
        model.sync()
        return model

    def preferred(self):
        """The index of the engine this batch's earlier passes ran on, which this pass stays on while it
        fits, even if another profile suits the sizes better: loading another means refitting every weight."""
        key = _last_used.get(self.arch_id)
        return next((i for i, r in enumerate(self.records) if r.key == key), None)

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
        record = self.records[index]
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
                self.fall_back(f"Could not load the engine for profile {record.index + 1} ({e})")
        stream = torch.cuda.current_stream().cuda_stream
        try:
            out = self._infer(feed, 0, stream)
        except torch.cuda.OutOfMemoryError:
            self.model.engine.release_activations()
            common.inference_mode(common.mm().free_memory)(sys.maxsize // 2, self.model.load_device, keep_loaded=_loaded_entries(self.model, self.patcher))
            common.mm().soft_empty_cache(force=True)
            try:
                out = self._infer(feed, 0, stream)
            except torch.cuda.OutOfMemoryError:
                self.fall_back("Out of VRAM running the engine")
        if not self.checked:
            self.checked = True
            if not torch.isfinite(out).all():
                self.fall_back("The engine produced NaN/Inf values")
        return out[:b].to(x.dtype)

    def _infer(self, feed, index, stream):
        try:
            return self.model.engine.infer(feed, index, stream)["latent"]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            traceback.print_exc()
            self.fall_back(f"The engine failed ({e})")


def _engine_records(patcher, x):
    """The engines to run this pass on, or None for PyTorch."""
    selected = _selected_arch()
    if selected is None:
        return None
    arch = arch_mod.detect(patcher)
    if arch is None or arch.id != selected.id:  # e.g. a refiner: with SD Unet on Automatic, that's expected
        if _explicit_choice():
            notify(f"The selected engines are for {selected.label}, not {arch.label if arch else 'this model'}; using PyTorch.")
        return None
    reason = _unsupported(patcher)
    if reason:
        notify(f"TensorRT engines can't apply {reason}; using PyTorch.")
        return None
    records = store.engines(arch.id)
    b, _, h, w = x.shape
    if not profiles.covers_size([r.shapes for r in records], h, w, b):
        notify(f"No profile of the {arch.label} engines covers {w * 8}x{h * 8} at batch size {b}; using PyTorch for this pass. Add one in the TensorRT tab.")
        return None
    return records


def _begin(records, patcher, x, prepare):
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
        p = Pass(records, patcher, x)
        try:
            b, _, h, w = x.shape
            prefer = p.preferred()
            choice = profiles.select(p.shapes, 2 * b, h, w, profiles.CONTEXT_PER_CHUNK, prefer) or profiles.select(p.shapes, b, h, w, profiles.CONTEXT_PER_CHUNK, prefer)
            index = choice[0] if choice else next(i for i, s in enumerate(p.shapes) if profiles.covers_size([s], h, w, b))
            model = p.use(records[index])
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
    s = model.record.shapes
    # cond and uncond batched, or one at a time if the profile can't batch them (e.g. at its largest batch)
    choice = profiles.select([s], 2 * b, h, w, profiles.CONTEXT_PER_CHUNK) or profiles.select([s], b, h, w, profiles.CONTEXT_PER_CHUNK)
    if choice is None:
        return 0
    _, padded, _ = choice
    shapes = {
        "sample": (padded, c, h, w),
        "timesteps": (padded,),
        "encoder_hidden_states": (padded, s["encoder_hidden_states"][1][1], s["encoder_hidden_states"][0][2]),
    }
    if "y" in s:
        shapes["y"] = (padded, s["y"][0][1])
    return model.engine.estimate_activation_bytes(0, shapes, torch.cuda.current_stream().cuda_stream)


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
        records = _engine_records(unet, x)
        if records is None:
            return original(unet, x, *args, **kwargs)
        _pass = _begin(records, unet, x, lambda u, v: original(u, v, *args, **kwargs))

    sampling_prepare._trt_original = original
    return sampling_prepare


def _wrap_cleanup(original):
    def sampling_cleanup(unet, *args, **kwargs):
        end_pass()
        return original(unet, *args, **kwargs)

    sampling_cleanup._trt_original = original
    return sampling_cleanup


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


def new_batch():
    """A batch's first pass runs on the engine whose profile fits it best, and its later passes (e.g. its
    Hires. fix) stay on that engine while it covers their sizes. The next batch starts afresh, so nothing
    keeps running on an engine picked for other sizes."""
    _last_used.clear()


def new_generation():
    end_pass()  # a generation that failed mid-pass never cleaned up
    common.new_generation()
    try:
        install_hooks()
    except Exception as e:
        log(f"Could not hook reForge's sampler: {e}")
