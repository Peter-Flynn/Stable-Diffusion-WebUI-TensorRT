"""Decoding and encoding on the VAE's engines (vae.py).

Each decode takes the first way there can be room for, decided before it starts: whole on the pre and
post engines, beside any UNet engine loaded; the post part in strips on engines of its own, all in VRAM
beside it; whole, the UNet engine evicted; in strips, the UNet engine evicted and what doesn't fit in VRAM
held in RAM; else reForge's own decode. (A UNet engine takes seconds to bring back: more than strips
cost over whole, less than holding tensors in RAM does.) Room is made for exactly what the way takes,
the VAE's engines it doesn't run on going first, then other models by reForge's manager. Each engine loads
when it's first used.

An encode (an img2img's or an inpaint's image, a Hires. fix's upscaled one) takes its way as a decode
does; in strips (strips.py's Encode, on engines of its own) it runs up to the mid block's attention, which
with its GroupNorm and its queries, keys and values runs on PyTorch at the latent's size, then the
encode-post engine."""

import contextlib
import gc
import hashlib
import os
import time
import traceback

import torch

from . import common, gpu, store, strips
from .common import logger, notify
from .vae import (
    ENCODE_POST, ENCODE_PRE, ENCODER, EXPONENT, EXPONENT_MAX, EXPONENT_STEP, POST, PRE, Record, dtype_name, encoder_digest,
    encoder_unsupported, engine_dtype, is_decoder_weight, is_encoder_weight, label, precisions, record_for, scaled_bias,
    strip_dtype, unsupported, weight_bytes,
)


class TensorRTVAE:
    """What the model manager holds for the VAE's engines (its name shows in reForge's log)."""


class _Side:
    """The VAE's decoder's engines, or its encoder's: the two parts of the one that runs it whole (PRE and
    POST, or ENCODER's) and the engines of the stages of the one that runs it in strips. Each loads when it's
    first used, with room made for what it takes (its plan's size and its weights'), and holds the weights of `key` (the VAE's decoder's or encoder's), the
    stages' scaled by 2^-exponent; refitted when it's used holding others."""

    def __init__(self, model: "VaeModel", encoding: bool):
        self.model, self.encoding = model, encoding
        self.strips_key = "encode-strips" if encoding else "strips"  # its stages' engines, in VaeModel.using
        self.key = None  # the key of the weights its engines are to hold (decode() or encode() sets it)
        self.engines = {}  # a part, or a Stage.identity -> its Engine
        self.names = {}  # likewise -> {the engine's weight: the VAE's}
        self.lasts = {}  # Stage.identity -> a context on its engine for the stage's last strips
        self.probes = {}  # kind name -> a context for activation sizes, on the engine of a stage of that kind
        self.part_probes = {}  # a part -> a context for activation sizes on its engine (part_probe)
        self.applied = {}  # a part, or a Stage.identity -> (key, exponent) of the weights its engine holds
        self.sizes = {}  # likewise -> what its engine held once loaded
        self.activations = {}  # (kind, rows, width) -> TensorRT's activation memory for a strip (its plan's own figure)
        self.loads = self.refits = 0  # since the last report(), and the seconds they took
        self.load_seconds = self.refit_seconds = 0.0

    @property
    def exponent(self) -> int:
        """What its stages' engines scale this side by, as 2^-exponent (0: they aren't scaled)."""
        return (_exponents.get(self.key) or EXPONENT) if self.model.record.scaled else 0

    @property
    def scale(self) -> float:
        return 2.0 ** -self.exponent

    @property
    def kinds(self) -> dict:
        """{kind name: {"shapes", "weights", "stripped"}}: the engines of the one that runs in strips."""
        return self.model.record.encoder_strips if self.encoding else self.model.record.strips

    def engine(self, part: str):
        """The engine of a part of the one that runs whole, loaded, holding the current weights."""
        record = self.model.record
        listed = (record.encoder if self.encoding else record.meta)["weights"][part]
        return self._get(part, record.plan(part), None, listed)

    def part_probe(self, part: str):
        """A context of its own on a part's engine (loaded, holding the current weights), for TensorRT's
        activation sizes: the context that runs has only the shapes it runs set on it, as the stages' do."""
        e = self.engine(part)
        probe = self.part_probes.get(part)
        if probe is None or probe.engine is not e.engine:
            if probe is not None:
                probe.unload()
            probe = self.part_probes[part] = e.twin()
        return probe

    def stage(self, stage) -> tuple:
        """(The engine of a stage of the one that runs in strips, loaded, holding the current weights; a context
        on it for the stage's last strips, so neither changes shapes from one decode of a size to the next.)"""
        record, kind = self.model.record, stage.kind.name
        plan = record.encoder_strip_plan(kind) if self.encoding else record.strip_plan(kind)
        e = self._get(stage.identity, plan, stage.weights, stage.weights.values())
        if stage.identity not in self.lasts:
            self.lasts[stage.identity] = e.twin()
        return e, self.lasts[stage.identity]

    def probe(self, stage):
        """A context of the stage's kind, for TensorRT's activation sizes."""
        if stage.kind.name not in self.probes:
            self.probes[stage.kind.name] = self.stage(stage)[0].twin()
        return self.probes[stage.kind.name]

    def runner_for(self, stage):
        return strips.EngineRunner(lambda: self.stage(stage), lambda: self.probe(stage), self.kinds[stage.kind.name]["shapes"], stage.kind)

    def _get(self, which, plan: str, weights, listed):
        """`which`'s engine (a part's, whose weights are named as the VAE's, or a stage's, `weights` naming
        them: {the engine's: the VAE's}): loaded from `plan` if it isn't, with room made for what it takes,
        the plan's size and the `listed` weights' (a load that fails anyway is tried once more: with all the
        room there could be if it ran out, or with TensorRT's own allocator if PyTorch's could be why);
        refitted if it holds other weights than the current ones. What it holds is what loading it added
        to the VRAM in use (gpu.loading)."""
        from . import engine as engine_mod
        from .engine import Engine

        model = self.model
        e = self.engines.get(which)
        if e is not None and e.loaded:
            if self.applied.get(which) != self._holds(which):
                start = time.time()
                self._refit(which, e)
                self.refits, self.refit_seconds = self.refits + 1, self.refit_seconds + time.time() - start
            return e
        start = time.time()
        # What it takes: what its plan holds (its file's size) and the weights refitted into it
        room = weight_bytes(model.vae, listed, model.record.strips_dtype if weights is not None else model.record.dtype)
        with contextlib.suppress(OSError):
            room += os.path.getsize(plan)
        e = Engine(plan, model.arena, model.owner)
        for attempt in (0, 1):
            model.make_room(common.padded(room) if not attempt else room)
            load, failures = {}, gpu.failures()
            try:
                with gpu.loading(model.owner) as load:
                    e.load()
                    state = model.vae.first_stage_model.state_dict(keep_vars=True)
                    own = weights if weights is not None else {n: n for n in e.weight_names if n in state}
                    self.names[which] = {n: own[n] for n in e.weight_names if n in own}
                    self._refit(which, e)
                break
            except Exception as error:
                past = e.loaded  # it deserialized, and failed past that
                if not attempt:
                    error.__traceback__ = None  # it holds the failed load's frames, and through them what TensorRT allocated for it
                e.unload()
                gpu.unloaded(model.owner, load.get("grew", 0))  # what it had taken went with it
                self.names.pop(which, None)
                self.applied.pop(which, None)
                if attempt:
                    raise
                alone = not model.decoder.engines and not model.encoder.engines  # nothing of this owner's to lose
                if past and alone and engine_mod.own_allocator(failures, model.owner, error):
                    continue
                if gpu.attached() and not load.get("failures"):  # it didn't run out: more room changes nothing
                    raise
                room = model.obtainable()
                logger.debug(f"Loading {os.path.basename(plan)} ran out of VRAM ({error}); freeing all there is and trying again.")
        self.sizes[which] = load["grew"]
        self.engines[which] = e
        self.loads, self.load_seconds = self.loads + 1, self.load_seconds + time.time() - start
        return e

    def _holds(self, which) -> tuple:
        """(The key of the weights `which`'s engine is to hold, the exponent they're scaled by: a stage's
        this side's, a whole-image part's none.)"""
        return self.key, 0 if isinstance(which, str) else self.exponent

    def _refit(self, which, e):
        key, exponent = self._holds(which)
        get, names = self.model.weights(2.0 ** -exponent), self.names[which]
        torch.cuda.current_stream().synchronize()  # nothing in flight on the weights it holds now
        e.refit(list(names), lambda name, device: get(names[name], device))
        self.applied[which] = (key, exponent)

    def _kept(self, which, keep) -> bool:
        return which in keep if isinstance(which, str) else self.strips_key in keep

    def spare(self, keep) -> int:
        """What its engines that aren't in `keep` hold (VaeModel.using's keys)."""
        return sum(self.sizes.get(w, 0) for w in self.engines if not self._kept(w, keep))

    def drop(self, keep=()):
        """Unloads its engines that aren't in `keep` (VaeModel.using's keys), and the contexts on them."""
        if self.strips_key not in keep:
            for e in list(self.probes.values()) + list(self.lasts.values()):
                e.unload()  # the contexts on an engine before it
            self.probes, self.lasts = {}, {}
        for which in [w for w in self.engines if not self._kept(w, keep)]:
            probe = self.part_probes.pop(which, None)
            if probe is not None:
                probe.unload()  # the context on a part's engine before it
            self.engines.pop(which).unload()
            gpu.unloaded(self.model.owner, self.sizes.get(which, 0))
            for d in (self.names, self.applied, self.sizes):
                d.pop(which, None)

    def report(self) -> str:
        """For the log: the engines it loaded and refitted since the last report."""
        parts = []
        if self.loads:
            parts.append(f"loaded {self.loads} engine{'s' if self.loads > 1 else ''} in {self.load_seconds:.1f}s")
        if self.refits:
            parts.append(f"refitted {self.refits} for other weights in {self.refit_seconds:.1f}s")
        self.loads = self.refits = 0
        self.load_seconds = self.refit_seconds = 0.0
        return ", ".join(parts)


class VaeModel:
    """The VAE's engines as reForge's model manager sees a model: held as loaded while a decode or an encode
    runs on them (common.register), and evicted like any other, which unloads them all (each comes back with
    a small refit). Its size is what its engines hold in VRAM, as counted (gpu.py). They load as they're
    first used (_Side), and those the decode or encode under way doesn't run on (not in `using`) are the
    first to go when it needs room: their weights are copies of those of the ones it runs on."""

    parent = None

    def __init__(self, record: Record):
        from .engine import Arena

        mm = common.mm()
        self.record = record
        self.owner = f"VAE engines ({os.path.basename(record.base)})"  # whose TensorRT memory it is (gpu.py)
        self.model = TensorRTVAE()
        self.load_device = mm.get_torch_device()
        self.offload_device = torch.device("cpu")
        self.arena = Arena()  # every engine's activation memory: one runs at a time
        self.vae = None  # the VAE to take the weights from, while a decode or an encode runs
        self.active = False  # held as loaded by reForge's manager
        self.using = set()  # the engines the decode or encode under way runs on (_Side.drop's keys)
        self.decoder, self.encoder = _Side(self, False), _Side(self, True)
        self.needs = {}  # (encoding, images, height, width, ...) -> what need() or need_encode() found
        self.attention = {}  # (encoding, images, height, width) -> the attention's working memory, as it took
        self.failed = set()  # (encoding, height, width) that TensorRT failed to run whole: in strips from then on
        self.beside = False  # the way under way runs beside the UNet engines: making room doesn't evict them

    @property
    def loaded(self) -> bool:
        return self.active

    def weights(self, scale: float):
        """(name, device) -> the VAE's weight as engines scaled by `scale` take it: the biases added to scaled
        tensors, scaled."""
        state = self.vae.first_stage_model.state_dict(keep_vars=True)

        def get(name, device):
            t = state[name].data.to(device)
            return t.float() * scale if scale != 1.0 and scaled_bias(name) else t

        return get

    def model_size(self):
        return gpu.held(self.owner)

    def loaded_size(self):
        return self.model_size() if self.active else 0

    def current_loaded_device(self):
        return self.load_device if self.active else self.offload_device

    def model_dtype(self):
        return self.record.dtype

    def model_patches_to(self, *args):
        pass

    def lowvram_patch_counter(self):
        return 0

    def is_clone(self, other):
        return other is self

    def partially_unload(self, device, memory_to_free=0):
        return 0

    def partially_load(self, device, extra_memory=0, force_patch_weights=False):
        """Held as loaded: its engines load as they're first used, each with its room made (_Side)."""
        self.active = True
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all:
            self.decoder.drop()
            self.encoder.drop()
            self.arena.release()
            self.active = False
            gpu.forget(self.owner)
        return self.model

    def use(self, side: _Side, parts=(), strips: bool = False, load: bool = True):
        """What's under way runs on these parts of `side`'s whole one, loaded now (each with its room) unless
        not `load`, and with `strips`, on its stages' engines (loaded as they're planned): the room made
        for it is beside them."""
        self.using = set(parts) | ({side.strips_key} if strips else set())
        for part in parts if load else ():
            side.engine(part)

    def obtainable(self, engines: bool = True) -> int:
        """The most VRAM there could be beside the reserve for what's under way: what's free, what every other
        model holds (the UNet engines' too, unless not `engines`), and what this one's engines that it doesn't
        run on hold."""
        spare = self.decoder.spare(self.using) + self.encoder.spare(self.using)
        return common.obtainable(self.load_device, keep=[self], engines=engines) - common.reserve() + spare

    def make_room(self, nbytes: int) -> int:
        """Makes `nbytes` free beside the reserve: this model's engines that what's under way doesn't run on go
        first (their weights are copies, and they're the cheapest to bring back), then other models, by
        reForge's manager (the UNet engines last; not at all for a way that runs beside them, `beside`:
        if what else there is gives back less than it holds, that way has less room, and the next way is
        the one that evicts them). Returns what's free then, beside the reserve."""
        if common.free(self.load_device) < nbytes:
            self.decoder.drop(self.using)
            self.encoder.drop(self.using)
        keep = _own(self)
        if self.beside:
            keep += [lm for lm in common.mm().current_loaded_models if getattr(lm.model, "evicted_last", False)]
        return common.make_room(nbytes, self.load_device, keep)

    def need(self, n: int, h: int, w: int, out_gpu: bool = False, whole: bool = True) -> tuple:
        """(The most VRAM a decode of n latents of h x w on the pre and post engines holds beside what's
        loaded, their activation memory): TensorRT's activation memory for those shapes (one arena, the
        larger engine's), and beside it the tensors held at each step: the latents and what the pre engine
        makes of them; then the mid block's input, the attention's queries, keys and values, its working
        memory (as it took at this size before) and its output; then that input, the attention's output,
        the image, and its FP32 copy if the VAE outputs on the GPU. Not whole: the part of a decode in strips
        before them (the pre engine's activation memory, the first two steps)."""
        key = (False, n, h, w, out_gpu, whole)
        if key not in self.needs:
            cfg, size = self.record.meta["config"], self.record.dtype.itemsize
            stream = torch.cuda.current_stream().cuda_stream
            c, zc = cfg["mid_channels"], cfg["latent_channels"]
            act = self.decoder.part_probe(PRE).estimate_activation_bytes(0, {"latent": (n, zc, h, w)}, stream)
            lat, z = n * c * h * w * size, n * zc * h * w * (size + 4)  # the latents, and their FP32 form on the way
            steps = [z + 4 * lat, 5 * lat + self.attention.get((False, n, h, w), 0)]
            if whole:
                act = max(act, self.decoder.part_probe(POST).estimate_activation_bytes(0, {"x": (n, c, h, w), "attention": (n, c, h, w)}, stream))
                image = n * cfg["out_ch"] * 64 * h * w
                steps.append(2 * lat + image * size + (image * 4 if out_gpu and size != 4 else 0))
            self.needs[key] = (act + max(steps), act)
        return self.needs[key]

    def need_encode(self, n: int, h: int, w: int, pixels_gpu: bool = False) -> tuple:
        """(The most VRAM an encode of n images of h x w on the encoder's engines holds beside what's
        loaded, their activation memory), likewise: the images (and their FP32 form, if they come from the
        GPU in another precision) with what the encode-pre engine makes of them; then the attention's; then
        the mid block's input, the attention's output, the moments and their FP32 copy."""
        c, mid = self.record.encoder["config"]["mid_channels"], (h // 8, w // 8)
        key = (True, n, *mid, pixels_gpu, h, w)  # the attention's size first (see _attention)
        if key not in self.needs:
            cfg, size = self.record.encoder["config"], self.record.dtype.itemsize
            stream = torch.cuda.current_stream().cuda_stream
            act = max(
                self.encoder.part_probe(ENCODE_PRE).estimate_activation_bytes(0, {"image": (n, cfg["in_channels"], h, w)}, stream),
                self.encoder.part_probe(ENCODE_POST).estimate_activation_bytes(0, {"x": (n, c, *mid), "attention": (n, c, *mid)}, stream),
            )
            lat, image = n * c * mid[0] * mid[1] * size, n * cfg["in_channels"] * h * w
            moments = n * cfg["moments"] * mid[0] * mid[1]
            steps = (
                image * size + (image * 4 if pixels_gpu and size != 4 else 0) + 4 * lat,
                5 * lat + self.attention.get((True, n, *mid), 0),
                2 * lat + moments * (size + (4 if size != 4 else 0)),
            )
            self.needs[key] = (act + max(steps), act)
        return self.needs[key]

    def need_after_strips(self, h: int, w: int) -> int:
        """What an encode in strips holds for an image once its strips are done, beyond their output (the mid
        block's input in the strips' precision, of a latent of h x w): the attention's GroupNorm's output in
        FP32, with the queries, keys and values made of it (_encode_strips); then those, the attention's
        working memory (as it took at this size before) and its output; then that output and the mid block's
        input as the encode-post engine takes them, its activation memory, the moments and their FP32 copy."""
        cfg, size = self.record.encoder["config"], self.record.dtype.itemsize
        c, moments = cfg["mid_channels"], cfg["moments"] * h * w
        wide, lat = c * h * w * 4, c * h * w * size
        shapes = {"x": (1, c, h, w), "attention": (1, c, h, w)}
        act = self.encoder.part_probe(ENCODE_POST).estimate_activation_bytes(0, shapes, torch.cuda.current_stream().cuda_stream)
        work = self.attention.get((True, 1, h, w), 0)
        cast = lat if size != 4 else 0  # the attention's output in the engine's precision, where that isn't FP32
        return max(4 * wide + work, 4 * wide + cast, act + max(wide, lat) + lat + moments * (size + (4 if size != 4 else 0)))

    def release(self):
        self.arena.release()


_models = {}  # Record.base -> VaeModel
_told = set()  # what was logged once already: records in another precision, architectures without engines
_exponents = {}  # the key of a VAE's decoder's or encoder's weights -> its exponent, where it outgrew EXPONENT (None: every one)


class _Overflow(Exception):
    """A scaled decode or encode produced NaN/Inf: its scale wasn't small enough for this VAE."""


_weights_keys = {}  # whether the encoder's -> (the VAE's model, the generation, the key of its decoder's or encoder's weights)


def _key(vae, encoder: bool = False) -> str:
    """A key for the decoder's weights (or the encoder's): their names, types, shapes and checksums. Once
    per generation: a VAE's weights can change in place. The checksums are computed where the weights are
    (VRAM, once reForge's own decode or encode has run), so keying them copies nothing to RAM."""
    from .weights import _Same, checksum

    fsm = vae.first_stage_model
    known = _weights_keys.get(encoder)
    if known is not None and known[0] == _Same(fsm) and known[1] == common.generation:
        return known[2]
    state = fsm.state_dict(keep_vars=True)
    h = hashlib.sha1()
    for name in sorted(n for n in state if (is_encoder_weight(n) if encoder else is_decoder_weight(n))):
        t = state[name].data
        h.update(f"{name} {t.dtype} {tuple(t.shape)} {checksum(t)};".encode())
    key = h.hexdigest()
    _weights_keys[encoder] = (_Same(fsm), common.generation, key)
    return key


def model_for(record: Record) -> VaeModel:
    model = _models.get(record.base)
    if model is None or model.record.meta != record.meta:
        if model is not None:
            _unload(model)
        model = _models[record.base] = VaeModel(record)
    return model


@common.inference_mode
def _unload(model: VaeModel):
    mm = common.mm()
    for i in reversed(range(len(mm.current_loaded_models))):
        if mm.current_loaded_models[i].model is model:
            mm.current_loaded_models.pop(i).model_unload()
    model.detach()


def release_all():
    for model in list(_models.values()):
        _unload(model)
    _models.clear()
    strips.release_pinned()  # the RAM that decodes in strips held their tensors in


def _fit(model: VaeModel, top: int, need_for, engines: bool, load) -> tuple:
    """(The most latents a decode on the pre and post engines takes at once, up to `top` (or images, an
    encode on the encoder's), 0 if not even one can; what that holds at its most; the engines' activation
    memory for it; the most room there could be.) `need_for(n)` gives what n hold (it loads the engines for
    a size it hasn't weighed before; `load()` loads them once one fits, and it's weighed again beside them:
    what they hold isn't room). With `engines`, a UNet engine may go for it; never for several at a time,
    which is barely faster than one. Room is made for that with the buffer on top (this model's engines
    that it doesn't run on first, then by reForge's manager), and the activation memory is allocated for
    it, once."""
    model.beside = not engines
    model.release()

    def room_for(n):  # what n hold, with the buffer on top
        return common.padded(need_for(n)[0])

    for loaded in (False, True):
        can = model.obtainable(engines)
        beside = model.obtainable(engines=False)
        n = next((n for n in range(top, 0, -1) if room_for(n) <= beside), 0) or (1 if room_for(1) <= can else 0)
        if not n:
            return 0, *need_for(1), can
        if not loaded:
            load()
    need, act = need_for(n)
    free = model.make_room(room_for(n))
    if free < room_for(n):
        logger.debug(f"Freeing {common.gib(room_for(n))} of VRAM left only {common.gib(free)}: the models loaded held more than reForge counts.")
        return 0, need, act, can
    model.arena.get(act)
    return n, need, act, can


PLANNINGS = 8  # plans made in a row at most while the stage engines they run load (_settled)
_NO_ROOM = object()  # what a way that there can't be room for returns (the next one is tried)


def _settled(planner, h: int, w: int, budget, ram: bool = True):
    """The planner's plan for latents of h x w in `budget()` bytes (all in VRAM, if not `ram`), once the
    stage engines it runs are loaded: planning loads engines of its own (for TensorRT's activation sizes),
    and a plan may run others, so it's planned again until one runs no engine that isn't loaded and none
    loaded while it was made. None if no such plan fits (or, which planning converges well before, PLANNINGS
    plans weren't enough)."""
    for _ in range(PLANNINGS):
        room = budget()  # (what PyTorch freed goes back to the driver first: common.free)
        plan = planner.plan(h, w, room, ram=ram)
        if plan is None:
            return None
        missing = [r for r in plan.runners if r.engines is None]
        for r in missing:
            r.engines = r.load()
        if not missing and budget() >= room:
            return plan
    logger.debug(f"Planning the strips didn't settle in {PLANNINGS} tries while their engines loaded.")
    return None


def _plan_strips(model: VaeModel, planner, h: int, w: int, beside: int, engines: bool):
    """The fastest plan in strips there can be room for, with exactly that room made, or None. Planned for
    the most there could be (VaeModel.obtainable: beside any UNet engine and all in VRAM, or with `engines`, the UNet engines
    evicted and tensors held in RAM if it takes that), room made for what it holds and `beside` (what's
    held beside the strips that isn't allocated yet when they're planned), then planned for what's free
    then, which gives that plan unless reForge's manager couldn't make the room."""
    best = _settled(planner, h, w, lambda: common.usable(model.obtainable(engines)) - beside, engines)
    if best is None:
        return None
    model.make_room(common.padded(best.vram + beside))
    return _settled(planner, h, w, lambda: common.usable(common.free(model.load_device)) - beside, engines)


class _Peak:
    """The most PyTorch's allocator held during a block beyond what it held when the block began (with
    gpu.py's allocator, TensorRT's allocations are PyTorch's), through the resets nested blocks make."""

    carried = 0  # what an enclosing block saw before a nested one reset the allocator's peak

    def __init__(self):
        self.base = self.held = 0

    def __enter__(self):
        _Peak.carried = max(_Peak.carried, torch.cuda.max_memory_allocated())
        self.base = torch.cuda.memory_allocated()
        self.outer = _Peak.carried
        _Peak.carried = 0
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        self.held = max(_Peak.carried, torch.cuda.max_memory_allocated()) - self.base
        _Peak.carried = max(self.outer, _Peak.carried, torch.cuda.max_memory_allocated())
        return False


def _counted(held: int, need: int) -> str:
    """For the log: what a decode or an encode used at most, beside what was estimated for it."""
    return f"Used {common.gib(held)} of VRAM ({common.gib(need)} estimated)"


def _passed(need: int, can: int) -> str:
    """For the log: why a decode or an encode didn't run whole."""
    return f" (whole would need {common.gib(need)}, and at most {common.gib(can)} can be freed)"


def _did(side: _Side) -> str:
    """For the log: "; " and the engines the side loaded and refitted, if any."""
    said = side.report()
    return f"; {said}" if said else ""


def _own(model: VaeModel) -> list:
    mm = common.mm()
    return [lm for lm in mm.current_loaded_models if lm.model is model]


_copy_streams = {}


def _copy_stream(device):
    if device not in _copy_streams:
        _copy_streams[device] = torch.cuda.Stream(device=device)
    return _copy_streams[device]


def _in_strips(record: Record, side: _Side) -> str:
    """For the log: the precision the strips' engines run in, and their scale."""
    return label(dtype_name(record.strips_dtype)) + (f" scaled by 2^-{side.exponent}" if record.scaled else "")


def _beside(engines: bool) -> str:
    return "with any UNet engine evicted" if engines else "beside any UNet engine"


def _outgrew(side: _Side, what: str) -> bool:
    """A decode or encode in strips came out NaN/Inf in FP16 at the side's scale (_Overflow): the next
    exponent for this VAE, from now on. Returns whether there is one; if not, said, and strips are off for
    it."""
    exponent = side.exponent
    if exponent + EXPONENT_STEP > EXPONENT_MAX:
        _exponents[side.key] = None
        notify(f"The VAE {what}'s strip engines produced NaN/Inf values even scaled down by 2^-{exponent}, so this VAE's {what} won't run in strips.")
        return False
    _exponents[side.key] = exponent + EXPONENT_STEP
    logger.info(f"This VAE's {what} overflowed FP16 in strips scaled by 2^-{exponent}, so it's scaled by 2^-{exponent + EXPONENT_STEP} from now on.")
    return True


def decode(vae, samples_in):
    """The decoded images as VAE.decode returns them, or None to decode on PyTorch."""
    from . import arch as arch_mod

    if unsupported(vae) or samples_in.ndim != 4:
        return None
    arch = arch_mod.detect()
    if arch is None:
        return None
    record = record_for(arch.id, vae)
    if record is None:
        if arch.id not in _told and store.engines(arch.id):  # the tab shows it as to build, too
            _told.add(arch.id)
            logger.info("There are no engines for this VAE yet, so it runs on PyTorch. Building in the TensorRT tab adds them.")
        return None
    b, c, h, w = samples_in.shape
    found = None
    if record.strips:
        try:
            found = strips.stages(vae.first_stage_model, record.meta["config"]["padding"])
            if strips.supported(found) or not record.strips_cover(found, c, h, w):
                found = None
        except (ValueError, KeyError, AttributeError, StopIteration):
            found = None
    whole = record.covers(b, c, h, w)
    if not whole and found is None:
        return None
    wanted = (engine_dtype(vae), strip_dtype(vae))
    if (record.dtype, record.strips_dtype) != wanted and record.base not in _told:
        _told.add(record.base)
        logger.info(
            f"The VAE engines were built in {precisions(record.dtype, record.strips_dtype)}, and would now be built in "
            f"{precisions(*wanted)}. Building again in the TensorRT tab matches them."
        )
    model = model_for(record)
    whole = whole and (False, h, w) not in model.failed
    if not whole and found is None:
        return None
    side = model.decoder
    model.vae, side.key = vae, _key(vae)
    again = True
    try:
        while True:
            if found is not None and record.scaled and _exponents.get(side.key, EXPONENT) is None:
                found = None  # no scale kept this VAE's decoder finite in strips: said when found out
            if not whole and found is None:
                return None
            try:
                with torch.inference_mode():
                    return _decode(model, vae, samples_in, whole, found)
            except _Overflow:
                _outgrew(side, "decoder")
            except torch.cuda.OutOfMemoryError as e:
                said = _ran_out(model, e)
                del e
                logger.debug(f"The VAE engines ran out of VRAM decoding ({said}).")
                if not again:
                    logger.warning("The VAE engines ran out of VRAM, so this decode runs on PyTorch.")
                    _make_way(model)
                    return None
                again = False
    finally:
        model.release()
        model.vae, model.using, model.beside = None, set(), False  # don't keep this VAE alive
        _profiled("VAE decode")


def _ran_out(model: VaeModel, error) -> str:
    """After a decode or encode ran out of VRAM though room was made for all it holds: where, and how VRAM
    stood, for the log; then its working memory is released and what PyTorch holds unused goes back to the
    driver. PyTorch's freed memory isn't always usable for the next request (with --cuda-malloc it's the
    driver's pool, in pieces as they were freed), so a run that fits can still run out; it's tried again."""
    frame = traceback.extract_tb(error.__traceback__)[-1]
    said = f"in {frame.name}, {os.path.basename(frame.filename)}:{frame.lineno}"
    error.__traceback__ = None  # it holds the run's frames, and through them its tensors
    with contextlib.suppress(Exception):
        said += f"; {common.gib(torch.cuda.mem_get_info()[0])} was free"
    model.release()
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return said


def _make_way(model: VaeModel):
    """Before PyTorch decodes or encodes instead: the VAE engines' VRAM is its to use."""
    with contextlib.suppress(Exception):
        model.decoder.drop()
        model.encoder.drop()
        model.release()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _profiled(where: str):
    from .engine import profile_report

    profile_report(where)


def _decode(model: VaeModel, vae, samples_in, whole: bool, found):
    """One try at a decode, the first way there can be room for: whole on the pre and post engines beside
    any UNet engine; in strips (`found`, the stages), all in VRAM beside it; whole, the UNet engine
    evicted; in strips, likewise, holding in RAM what it must. None if none. Each gets exactly its room
    (VaeModel.need, _plan_strips). Raises _Overflow if a scaled decode in strips outgrew its scale."""
    b, _, h, w = samples_in.shape
    side, record = model.decoder, model.record
    images = f"{b} image{'' if b == 1 else 's'} of {w * 8}x{h * 8}"
    common.register(model)
    out_gpu = torch.device(vae.output_device).type == "cuda"
    held_out = b * record.meta["config"]["out_ch"] * 64 * h * w * 4 if out_gpu else 0  # the images, all of them, as returned

    def need_for(n):
        need, act = model.need(n, h, w, out_gpu)
        return need + held_out, act

    passed = ""  # why whole was passed over, for the log
    for engines in (False, True):
        if whole:
            model.use(side, (PRE, POST), load=False)
            n, need, act, can = _fit(model, min(b, record.max_batch), need_for, engines, lambda: model.use(side, (PRE, POST)))
            if n:
                start = time.time()
                try:
                    with _Peak() as peak:
                        pixels = _run(model, vae, samples_in, n)  # ends with a copy to the CPU, so the time is the decode's
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:
                    whole, passed = False, _whole_failed(model, False, h, w, f"decode {w * 8}x{h * 8}", e, found is not None)
                else:
                    if pixels is not None:
                        at_once = f", {n} at a time" if n < b else ""
                        logger.debug(
                            f"Decoded {images} on the VAE engines in {time.time() - start:.2f}s: whole{at_once}, in {label(record.meta['dtype'])}, "
                            f"{_beside(engines)}. {_counted(peak.held + act, need)}{_did(side)}."
                        )
                    return pixels
            else:
                passed = _passed(need, can)
        if found is not None:
            model.use(side, (PRE,), strips=True)
            model.release()
            start = time.time()
            result = _run_strips(model, vae, samples_in, found, out_gpu, engines)
            if result is not _NO_ROOM:
                pixels, report = result
                if pixels is not None:
                    logger.debug(
                        f"Decoded {images} on the VAE engines in {time.time() - start:.2f}s: in strips, in {_in_strips(record, side)}, "
                        f"{_beside(engines)}{passed}. {report}{_did(side)}."
                    )
                return pixels
    if (False, h, w) not in model.failed:  # (said, if they failed to run it)
        logger.warning(f"There's too little VRAM to decode {w * 8}x{h * 8} on the VAE engines{passed or ' (not even in strips)'}, so it runs on PyTorch.")
    return None


def _whole_failed(model: VaeModel, encoding: bool, h: int, w: int, what: str, error, strips_too: bool) -> str:
    """TensorRT failed to run the whole-image engines at this size: said, and not tried whole at this size
    again this session. Returns why whole was passed over, for the log."""
    model.release()
    model.failed.add((encoding, h, w))
    then = "in strips" if strips_too else "on PyTorch"
    logger.warning(f"The VAE engines failed to {what} whole, so that size runs {then} from now on: {error}")
    return " (the engines failed to run it whole)"


def _attention(model: VaeModel, vae, q, k, v, encoding: bool = False):
    """The mid block's attention on these queries, keys and values, on PyTorch with reForge's own
    implementation, as PyTorch decodes or encodes; its output in the whole-image engines' precision. The
    working memory it takes is kept, per size, for VaeModel.need, need_encode and need_after_strips."""
    fsm = vae.first_stage_model
    attention = (fsm.encoder if encoding else fsm.decoder).mid.attn_1.optimized_attention
    key = (encoding, q.shape[0], q.shape[2], q.shape[3])
    with _Peak() as peak:
        out = attention(q, k, v)
    work = max(0, peak.held - out.numel() * out.element_size())
    if work > model.attention.get(key, 0):  # what was counted at this size is short
        model.attention[key] = work
        model.needs = {kept: need for kept, need in model.needs.items() if kept[:4] != key}
    return out.to(model.record.dtype)


def _run(model: VaeModel, vae, samples_in, n: int):
    """The images of a decode on the pre and post engines, n latents at a time; None if they came out
    NaN/Inf (said: PyTorch's would, as wrong)."""
    side, record = model.decoder, model.record
    dtype, device = record.dtype, model.load_device
    stream = torch.cuda.current_stream().cuda_stream
    pre, post = side.engine(PRE), side.engine(POST)
    pixels = None
    for start in range(0, samples_in.shape[0], n):
        z = vae._to_vae_latent(samples_in[start:start + n]).to(device=device, dtype=dtype)
        mid = pre.infer({"latent": z}, 0, stream)
        del z
        h = _attention(model, vae, mid.pop("q"), mid.pop("k"), mid.pop("v"))
        image = post.infer({"x": mid.pop("x"), "attention": h}, 0, stream)["image"]
        del h
        if pixels is None and not torch.isfinite(image).all():
            notify(f"The VAE decoder engines produced NaN/Inf values in {label(record.meta['dtype'])}, so this decode runs on PyTorch.")
            return None
        if pixels is None:
            pixels = torch.empty((samples_in.shape[0],) + tuple(image.shape[1:]), device=vae.output_device)
        pixels[start:start + n] = vae.process_output(image.to(vae.output_device).float())
        del image
    return pixels.movedim(1, -1)


def _run_strips(model: VaeModel, vae, samples_in, found: list, out_gpu: bool, engines: bool):
    """(The images, decoded one at a time with the post part in strips, or None; how it went for the first,
    for the log), or _NO_ROOM if there can't be room for it this way (`engines`: whether a UNet engine may
    go for it, and tensors be held in RAM). The part before the strips gets exactly its room (VaeModel.need,
    not whole), the strips theirs (_plan_strips). Raises _Overflow if scaled strips outgrew their scale. A
    plan that runs out of VRAM anyway raises (decode() says so): it was counted wrong."""
    model.beside = not engines
    side, record = model.decoder, model.record
    dtype, sdtype, device = record.dtype, record.strips_dtype, model.load_device
    stream = torch.cuda.current_stream().cuda_stream
    cfg = record.meta["config"]
    decoder = strips.Decode(vae.first_stage_model, cfg["padding"], side.runner_for, sdtype, device, _copy_stream(device), model.arena, side.activations, set(side.kinds), side.scale)
    b, _, h, w = samples_in.shape
    image = cfg["out_ch"] * 64 * h * w  # an image's values
    # Held beside the strips but not yet when they're planned: the image they make, copied out of them; on
    # the GPU, also its FP32 copy and the images decoded so far
    beside = image * sdtype.itemsize + (image * 4 * (b + 1) if out_gpu else 0)
    before = model.need(1, h, w, whole=False)[0]  # the part before the strips
    held = 2 * cfg["mid_channels"] * h * w * dtype.itemsize  # what it leaves held through them: the mid block's input, the attention's output
    before = common.padded(before)
    if before > model.obtainable(engines) or _settled(decoder, h, w, lambda: common.usable(model.obtainable(engines)) - beside - held, engines) is None:
        return _NO_ROOM
    pixels = plan = None
    report = ""
    for i in range(b):
        if i == 0:
            if model.make_room(before) < before and model.beside:  # what else there is gave back less than it holds
                return _NO_ROOM
        else:  # the image before's strips are freed: back to the driver, for this one's part before them
            model.release()
            common.trim()
        z = vae._to_vae_latent(samples_in[i:i + 1]).to(device=device, dtype=dtype)
        mid = side.engine(PRE).infer({"latent": z}, 0, stream)
        del z
        x = mid.pop("x")
        attended = _attention(model, vae, mid.pop("q"), mid.pop("k"), mid.pop("v"))
        del mid
        model.release()  # the pre engine's activations: the strips size their own
        common.trim()
        if plan is None:
            plan = _plan_strips(model, decoder, h, w, beside, engines)
            if plan is None:
                return _NO_ROOM
        if side.scale != 1.0:  # as the strips run: scaled, while still in a precision that holds what a VAE makes
            x, attended = x.float().mul_(side.scale), attended.float().mul_(side.scale)
        with _Peak() as peak:
            image_out = decoder.run(x, attended, plan)
        del x, attended
        if not torch.isfinite(image_out).all():
            if record.scaled:
                raise _Overflow()
            if pixels is None:
                notify(f"The VAE decoder's strip engines produced NaN/Inf values in {label(dtype_name(sdtype))}, so this decode runs on PyTorch.")
                return None, ""
        if pixels is None:
            report = _strips_report(plan, peak.held, plan.vram + beside)
            pixels = torch.empty((b,) + tuple(image_out.shape[1:]), device=vae.output_device)
        pixels[i:i + 1] = vae.process_output(image_out.to(vae.output_device).float())
        del image_out
    return pixels.movedim(1, -1), report


def encode(vae, pixel_samples):
    """The latents as VAE.encode returns them, or None to encode on PyTorch."""
    from . import arch as arch_mod

    if unsupported(vae) or pixel_samples.ndim != 4 or encoder_unsupported(vae):
        return None
    arch = arch_mod.detect()
    record = record_for(arch.id, vae) if arch is not None else None
    if record is None or record.encoder is None:
        return None  # (vae.decode says when an architecture has no engines)
    pixels = vae.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
    b, c, h, w = pixels.shape
    if not record.has_encoder(encoder_digest(vae), c):
        return None
    found = None
    if record.encoder_strips:
        try:
            found = strips.encoder_stages(vae.first_stage_model, record.encoder["config"]["padding"])
            if strips.supported(found) or not record.encoder_strips_cover(found, h, w):
                found = None
        except (ValueError, KeyError, AttributeError, StopIteration):
            found = None
    whole = record.encodes(h, w)
    if not whole and found is None:
        return None
    model = model_for(record)
    whole = whole and (True, h // 8, w // 8) not in model.failed
    if not whole and found is None:
        return None
    side = model.encoder
    model.vae, side.key = vae, _key(vae, encoder=True)
    again = True
    try:
        while True:
            if found is not None and record.scaled and _exponents.get(side.key, EXPONENT) is None:
                found = None  # no scale kept this VAE's encoder finite in strips: said when found out
            if not whole and found is None:
                return None
            try:
                with torch.inference_mode():
                    return _encode(model, vae, pixels, whole, found)
            except _Overflow:
                _outgrew(side, "encoder")
            except torch.cuda.OutOfMemoryError as e:
                said = _ran_out(model, e)
                del e
                logger.debug(f"The VAE engines ran out of VRAM encoding ({said}).")
                if not again:
                    logger.warning("The VAE engines ran out of VRAM, so this encode runs on PyTorch.")
                    _make_way(model)
                    return None
                again = False
    finally:
        model.release()
        model.vae, model.using, model.beside = None, set(), False  # don't keep this VAE alive
        _profiled("VAE encode")


def _encode(model: VaeModel, vae, pixels, whole: bool, found):
    """One try at an encode of images (b, channels, h, w) in 0..1 on the encoder's engines, the first way
    there can be room for, as _decode: whole or in strips (`found`, the stages), beside any UNet engine,
    then with it evicted. None if none. Raises _Overflow if a scaled encode in strips outgrew its scale."""
    b, _, h, w = pixels.shape
    side, record = model.encoder, model.record
    images = f"{b} image{'' if b == 1 else 's'} of {w}x{h}"
    common.register(model)
    pixels_gpu = pixels.is_cuda
    passed = "" if whole else " (larger than the engines take whole)"
    for engines in (False, True):
        if whole:
            model.use(side, ENCODER, load=False)
            n, need, act, can = _fit(model, min(b, record.encode_batch), lambda n: model.need_encode(n, h, w, pixels_gpu), engines, lambda: model.use(side, ENCODER))
            if n:
                start = time.time()
                samples = None
                try:
                    with _Peak() as peak:
                        for at in range(0, b, n):
                            latent = _encode_images(model, vae, vae.process_input(pixels[at:at + n]))
                            if latent is None:
                                return None
                            out = latent.to(vae.output_device).float()
                            out = vae._from_vae_latent(out)
                            if samples is None:
                                samples = torch.empty((b,) + tuple(out.shape[1:]), device=vae.output_device)
                            samples[at:at + n] = out
                            del latent, out
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:
                    whole, passed = False, _whole_failed(model, True, h // 8, w // 8, f"encode {w}x{h}", e, found is not None)
                else:
                    at_once = f", {n} at a time" if n < b else ""
                    logger.debug(
                        f"Encoded {images} on the VAE engines in {time.time() - start:.2f}s: whole{at_once}, in {label(record.meta['dtype'])}, "
                        f"{_beside(engines)}. {_counted(peak.held + act, need)}{_did(side)}."
                    )
                    return samples
            else:
                passed = _passed(need, can)
        if found is not None:
            model.use(side, (ENCODE_POST,), strips=True)
            model.release()
            start = time.time()
            result = _encode_strips(model, vae, pixels, found, engines)
            if result is not _NO_ROOM:
                samples, report = result
                if samples is not None:
                    logger.debug(
                        f"Encoded {images} on the VAE engines in {time.time() - start:.2f}s: in strips, in {_in_strips(record, side)}, "
                        f"{_beside(engines)}{passed}. {report}{_did(side)}."
                    )
                return samples
    if (True, h // 8, w // 8) not in model.failed:  # (said, if they failed to run it)
        logger.warning(f"There's too little VRAM to encode {w}x{h} on the VAE engines{passed or ' (not even in strips)'}, so it runs on PyTorch.")
    return None


def _encode_strips(model: VaeModel, vae, pixels, found: list, engines: bool):
    """(The latents of images (b, channels, h, w) in 0..1, encoded one at a time in strips up to the mid
    block's attention, or None; how it went for the first, for the log), or _NO_ROOM if there can't be
    room for it this way (`engines`: whether a UNet engine may go for it, and tensors be held in RAM). After
    the strips, on PyTorch at the latent's size: the attention's GroupNorm (from the statistics the strips
    gave), its queries, keys and values (in FP32, as a GroupNorm's output is), and the attention; then the
    encode-post engine, as a whole encode. The image gets exactly its room, the strips theirs
    (_plan_strips), and the rest its own (VaeModel.need_after_strips). Raises _Overflow if scaled strips
    outgrew their scale."""
    model.beside = not engines
    side, record, device = model.encoder, model.record, model.load_device
    dtype, sdtype, stream = record.dtype, record.strips_dtype, torch.cuda.current_stream().cuda_stream
    cfg = record.encoder["config"]
    fsm = vae.first_stage_model
    encoder = strips.Encode(fsm, cfg["padding"], side.runner_for, sdtype, device, _copy_stream(device), model.arena, side.activations, set(side.kinds), side.scale)
    b, c, h, w = pixels.shape
    mh, mw = h // 8, w // 8
    image_bytes = c * h * w * (4 + sdtype.itemsize)  # the image, and its FP32 form on the way
    beside = cfg["mid_channels"] * mh * mw * sdtype.itemsize  # the strips' output, copied out of them
    after = model.need_after_strips(mh, mw)
    can = model.obtainable(engines)
    image_room, after = common.padded(image_bytes), common.padded(after)
    if max(image_room, after + beside) > can or _settled(encoder, mh, mw, lambda: common.usable(model.obtainable(engines)) - beside - image_bytes, engines) is None:
        return _NO_ROOM
    norm = strips.norms_of(fsm, [strips.ATTENTION_NORM], device)[strips.ATTENTION_NORM]
    a = fsm.encoder.mid.attn_1
    convs = [(conv.weight.detach().to(device, torch.float32), conv.bias.detach().to(device, torch.float32)) for conv in (a.q, a.k, a.v)]
    samples = plan = None
    report = ""
    for i in range(b):
        if i == 0:
            if model.make_room(image_room) < image_room and model.beside:
                return _NO_ROOM
        else:  # what the image before freed: back to the driver
            model.release()
            common.trim()
        image = vae.process_input(pixels[i:i + 1]).to(device=device, dtype=sdtype)
        if plan is None:
            plan = _plan_strips(model, encoder, mh, mw, beside, engines)
            if plan is None:
                return _NO_ROOM
        with _Peak() as peak:
            x, stats = encoder.run(image, plan)
        del image
        model.release()  # the strips' activations
        if record.scaled and not torch.isfinite(x).all():
            raise _Overflow()
        if i == 0:
            if model.make_room(after) < after and model.beside:
                return _NO_ROOM
        else:
            common.trim()
        center, factor, shift = stats.affine(norm, side.scale)
        normed = (x.float() - center) * factor + shift
        q, k, v = (torch.nn.functional.conv2d(normed, weight, bias) for weight, bias in convs)
        del normed
        attended = _attention(model, vae, q, k, v, encoding=True)
        del q, k, v
        if i == 0:
            report = _strips_report(plan, peak.held, plan.vram + beside)
        x = (x.float().div_(side.scale) if side.scale != 1.0 else x).to(dtype)  # as the encode-post engine takes it
        moments = side.engine(ENCODE_POST).infer({"x": x, "attention": attended}, 0, stream)["moments"]
        del x, attended
        if not torch.isfinite(moments).all():
            notify(f"The VAE encoder engines produced NaN/Inf values in {label(record.meta['dtype'])}, so this encode runs on PyTorch.")
            return None, ""
        out = fsm.regularization(moments.float())[0].to(vae.output_device).float()
        out = vae._from_vae_latent(out)
        if samples is None:
            samples = torch.empty((b,) + tuple(out.shape[1:]), device=vae.output_device)
        samples[i:i + 1] = out
        del moments, out
    return samples, report


def _encode_images(model: VaeModel, vae, image):
    """The latents (in VRAM, FP32) of images as the encoder takes them (-1..1), on the encoder's engines;
    None if they came out NaN/Inf (said)."""
    side, record = model.encoder, model.record
    stream = torch.cuda.current_stream().cuda_stream
    image = image.to(device=model.load_device, dtype=record.dtype)
    mid = side.engine(ENCODE_PRE).infer({"image": image}, 0, stream)
    del image
    attended = _attention(model, vae, mid.pop("q"), mid.pop("k"), mid.pop("v"), encoding=True)
    moments = side.engine(ENCODE_POST).infer({"x": mid.pop("x"), "attention": attended}, 0, stream)["moments"]
    del attended, mid
    if not torch.isfinite(moments).all():
        notify(f"The VAE encoder engines produced NaN/Inf values in {label(record.meta['dtype'])}, so this encode runs on PyTorch.")
        return None
    return vae.first_stage_model.regularization(moments.float())[0]  # reForge's own: the distribution's mode


def _strips_report(plan, held: int, need: int) -> str:
    """For the log: how a plan in strips ran, and what it used of the VRAM it was planned for."""
    parts = [_counted(held, need)]
    if plan.describe():
        parts.append(plan.describe())
    if plan.host:
        parts.append(f"{len(plan.host)} tensors held in RAM")
    return "; ".join(parts)
