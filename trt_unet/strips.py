"""Tensor Stripped: the VAE decoder's "post" part (vae.py) in horizontal strips, for decodes too large for its
post engine in the VRAM free. Same results as the whole decode, up to rounding.

The post part is cut into stages of one convolution each: the projection into the mid block's second
resnet block, each resnet block's two convolutions (the second with the block's shortcut), the upsamplings
and the output convolution. A stage runs on one strip of rows at a time, given its input's rows with a
row more on each side for its 3x3 convolution (the image's padding there: zeros, or reflected, replicated
or wrapped rows, as the convolution pads). The GroupNorm before a convolution needs statistics of its
whole input: each stage outputs, with every strip, its mean and spread over the strip for each group,
which add up (Chan et al.'s parallel variance) to the next GroupNorm's over the whole image. So a stage
finishes before the next starts, and what's between stages is held whole: in VRAM when it fits, else in
pinned RAM, copied in and out while the engine works on the strip before.

Stages with the same layout share an engine plan ("kind"), each loaded with its own weights."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

STRIP_ELEMENTS = 32 * 2**20  # the most elements in a strip of a stage's largest tensor: what kinds are built for
CONV, UP, PROJ, OUT = "conv", "up", "proj", "out"
INPUTS = {
    CONV: ("x", "keep", "center", "scale", "shift"),  # and "res" with a residual
    OUT: ("x", "keep", "center", "scale", "shift"),
    UP: ("x",),
    PROJ: ("x", "res"),
}
PADDINGS = ("zeros", "reflect", "replicate", "circular")
HEADROOM = 256 * 2**20  # VRAM left free during a decode, and a 16th of what's free


@dataclass(frozen=True)
class Kind:
    """What one engine computes. CONV: GroupNorm (as center, scale and shift given), SiLU and a 3x3 conv,
    plus a residual ("res" channels, through a 1x1 conv when that's not cout); UP: nearest 2x upsampling and
    a 3x3 conv; PROJ: a 1x1 conv plus a residual; OUT: as CONV, its output the image. With groups, also the
    output's mean and spread ("m2", the sum of squared deviations) per group."""

    op: str
    cin: int
    cout: int
    res: int = 0
    groups: int = 0
    pad: str = "zeros"  # the 3x3 conv's padding across (its padding down is the strips' rows)

    @property
    def name(self) -> str:
        res = f"r{self.res}" if self.res else ""
        groups = f"g{self.groups}" if self.groups else ""
        pad = "" if self.pad == "zeros" else f"-{self.pad}"
        return f"{self.op}{self.cin}-{self.cout}{res}{groups}{pad}"

    @property
    def shortcut(self) -> bool:
        return bool(self.res) and self.res != self.cout and self.op == CONV

    @property
    def halo(self) -> int:
        """Rows of the input beyond the strip's, on each side (UP: of its input, before upsampling)."""
        return 0 if self.op == PROJ else 1

    def inputs(self) -> tuple:
        return INPUTS[self.op] + (("res",) if self.op == CONV and self.res else ())

    def outputs(self) -> tuple:
        return ("out",) + (("mean", "m2") if self.groups else ())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Stage:
    kind: Kind
    weights: dict  # the engine's weight name -> the decoder's (in the first stage model's state dict)
    scale: int  # the output's size in latents (1 at the mid block, 8 at the image)
    src: int  # the buffer of its input: the index of the stage that makes it (-2: the mid block's input, -1: its attention)
    res: int = None  # the buffer of its residual
    norm: str = None  # the GroupNorm normalizing its input (module name), with the statistics of src
    pad: str = "zeros"  # its conv's padding, which the strips' rows give down the image

    @property
    def in_scale(self) -> int:
        return self.scale // 2 if self.kind.op == UP else self.scale


def stages(fsm, padding: dict) -> list:
    """The post part's stages, for a first stage model (or what vae.fresh makes) and the config's padding."""
    d = fsm.decoder
    pad = lambda name: padding.get(name, "zeros")  # noqa: E731
    found = []

    def add(op, cin, cout, prefix, convs, scale, src, res=None, res_channels=0, norm=None):
        conv_name = prefix + convs["conv"]
        kind_pad = pad(conv_name)
        weights = {}
        for local, name in convs.items():
            weights[f"{local}.weight"] = f"{prefix}{name}.weight"
            weights[f"{local}.bias"] = f"{prefix}{name}.bias"
        found.append(Stage(Kind(op, cin, cout, res_channels, 0, kind_pad), weights, scale, src, res, norm, kind_pad))
        return len(found) - 1

    def block(prefix, blk, src, scale):
        cin, cout = int(blk.in_channels), int(blk.out_channels)
        if getattr(blk, "use_conv_shortcut", False) and cin != cout:
            raise ValueError("a resnet block with a 3x3 shortcut")
        h = add(CONV, cin, cout, prefix, {"conv": "conv1"}, scale, src, norm=prefix + "norm1")
        convs = {"conv": "conv2"} if cin == cout else {"conv": "conv2", "nin": "nin_shortcut"}
        return add(CONV, cout, cout, prefix, convs, scale, h, res=src, res_channels=cin, norm=prefix + "norm2")

    c = int(d.mid.attn_1.in_channels)
    x = add(PROJ, c, c, "decoder.mid.attn_1.", {"conv": "proj_out"}, 1, -1, res=-2, res_channels=c)
    x = block("decoder.mid.block_2.", d.mid.block_2, x, 1)
    scale = 1
    for i in reversed(range(d.num_resolutions)):
        for j, blk in enumerate(d.up[i].block):
            x = block(f"decoder.up.{i}.block.{j}.", blk, x, scale)
        if i != 0:
            c = int(d.up[i].block[-1].out_channels)
            scale *= 2
            x = add(UP, c, c, f"decoder.up.{i}.upsample.", {"conv": "conv"}, scale, x)
    c = int(d.norm_out.num_channels)
    add(OUT, c, int(d.conv_out.out_channels), "decoder.", {"conv": "conv_out"}, scale, x, norm="decoder.norm_out")

    # Each output's statistics are for the GroupNorm that reads it; outputs no GroupNorm reads (before an
    # upsampling) still give them, so stages of the same layout share a kind.
    modules = dict(fsm.named_modules())
    for n, stage in enumerate(found):
        if stage.kind.op == OUT:
            continue
        reader = next((s.norm for s in found[n + 1:] if s.src == n and s.norm), None)
        if reader is None:
            reader = next(s.norm for s in found[n + 1:] if s.norm)  # the next GroupNorm (the same groups in these VAEs)
        stage.kind = Kind(**{**stage.kind.to_dict(), "groups": int(modules[reader].num_groups)})
    return found


def kinds(found: list) -> dict:
    """{kind name: (Kind, [the scales of its stages' outputs])}."""
    out = {}
    for s in found:
        out.setdefault(s.kind.name, (s.kind, set()))[1].add(s.scale)
    return {k: (kind, sorted(scales)) for k, (kind, scales) in out.items()}


def supported(found: list) -> str:
    """Why the stripped decode can't run these stages ("" if it can)."""
    for s in found:
        if s.pad not in PADDINGS:
            return f"{s.pad} padding"
        if s.kind.groups and s.kind.cout % s.kind.groups:
            return "GroupNorm groups that don't divide the channels"
    return ""


# --- The stages' computation, on PyTorch (exported to ONNX for the engines)


def _conv3(x, conv, pad: str):
    """A 3x3 conv over x's rows as given (the strip's own and one more on each side), padded across."""
    if pad == "zeros":
        return F.conv2d(x, conv.weight, conv.bias, padding=(0, 1))
    if pad == "reflect":
        x = torch.cat([x[..., 1:2], x, x[..., -2:-1]], dim=-1)
    elif pad == "replicate":
        x = torch.cat([x[..., :1], x, x[..., -1:]], dim=-1)
    else:  # circular
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
    return F.conv2d(x, conv.weight, conv.bias)


def _stats(out, groups: int):
    o = out.float().reshape(out.shape[0], groups, -1)
    mean = o.mean(dim=-1)
    return mean, (o - mean.unsqueeze(-1)).square().sum(dim=-1)


class StageModule(torch.nn.Module):
    """A stage of a decoder: its convs are the decoder's own modules (named as Stage.weights has them)."""

    def __init__(self, stage: Stage, fsm):
        super().__init__()
        self.kind = stage.kind
        modules = dict(fsm.named_modules())
        for local in sorted({n.split(".")[0] for n in stage.weights}):
            setattr(self, local, modules[stage.weights[f"{local}.weight"][: -len(".weight")]])

    def forward(self, *args):
        k = self.kind
        feed = dict(zip(k.inputs(), args))
        x = feed["x"]
        if k.op == PROJ:
            out = feed["res"] + F.conv2d(x, self.conv.weight, self.conv.bias)
        elif k.op == UP:
            out = _conv3(F.interpolate(x, scale_factor=2.0, mode="nearest")[:, :, 1:-1], self.conv, k.pad)
        else:
            normed = ((x.float() - feed["center"]) * feed["scale"] + feed["shift"]).to(x.dtype)
            out = _conv3(F.silu(normed) * feed["keep"], self.conv, k.pad)
            if "res" in feed:
                out = out + (self.nin(feed["res"]) if k.shortcut else feed["res"])
        return (out,) + (_stats(out, k.groups) if k.groups else ())


def example_inputs(kind: Kind, dtype, device, rows: int = 4, width: int = 8) -> tuple:
    """Inputs for tracing: `rows` of output, `width` of input."""
    n = rows // 2 + 2 * kind.halo if kind.op == UP else rows + 2 * kind.halo
    shapes = feed_shapes(kind, rows, width * (2 if kind.op == UP else 1))
    out = []
    for name in kind.inputs():
        shape = shapes[name]
        if name in ("center", "scale", "shift"):
            out.append(torch.randn(shape, dtype=torch.float32, device=device))
        elif name == "keep":
            out.append(torch.ones(shape, dtype=dtype, device=device))
        else:
            out.append(torch.randn(shape, dtype=dtype, device=device))
    assert out[0].shape[2] == n
    return tuple(out)


def feed_shapes(kind: Kind, rows: int, width: int) -> dict:
    """The inputs' shapes for `rows` rows of output `width` wide."""
    n = rows + 2 * kind.halo
    shapes = {}
    for name in kind.inputs():
        if name == "x":
            shapes[name] = (1, kind.cin, rows // 2 + 2 * kind.halo, width // 2) if kind.op == UP else (1, kind.cin, n, width)
        elif name == "keep":
            shapes[name] = (1, 1, n, 1)
        elif name in ("center", "scale", "shift"):
            shapes[name] = (1, kind.cin, 1, 1)
        else:  # res
            shapes[name] = (1, kind.res, rows, width)
    return shapes


def dynamic_axes(kind: Kind) -> dict:
    axes = {"x": {2: "rows", 3: "width"}, "out": {2: "out_rows", 3: "out_width"}}
    if "keep" in kind.inputs():
        axes["keep"] = {2: "rows"}
    if "res" in kind.inputs():
        axes["res"] = {2: "out_rows", 3: "out_width"}
    return axes


def max_rows(kind: Kind, width: int) -> int:
    """The most output rows a strip of this kind can have at this output width (even for UP), by
    STRIP_ELEMENTS."""
    channels = max(kind.cin, kind.cout, kind.res)
    rows = max(1, STRIP_ELEMENTS // max(1, channels * width))
    return max(2, rows - rows % 2) if kind.op == UP else rows


def kind_shapes(kind: Kind, scales: list, lo_w: int, opt: tuple, hi: tuple) -> dict:
    """The engine's profile ({input: [min, opt, max]}) for strips of its stages' outputs at these scales,
    for latents from lo_w wide to hi = (h, w), optimal at opt = (h, w): any number of rows up to a strip's
    (STRIP_ELEMENTS at the widest), across any width in that range."""
    hi_scale, lo_scale = max(scales), min(scales)
    lo_width, hi_width = lo_w * lo_scale, hi[1] * hi_scale
    hi_rows = max(min(hi[0] * s, max_rows(kind, hi[1] * s)) for s in scales)
    opt_width = opt[1] * hi_scale
    opt_rows = min(opt[0] * hi_scale, max_rows(kind, opt_width))
    lo_rows = 2 if kind.op == UP else 1
    hi_rows, opt_rows = max(hi_rows, lo_rows), max(min(opt_rows, hi_rows), lo_rows)
    if kind.op == UP:
        hi_rows, opt_rows = hi_rows - hi_rows % 2, opt_rows - opt_rows % 2
    lo, mid, top = feed_shapes(kind, lo_rows, lo_width), feed_shapes(kind, opt_rows, opt_width), feed_shapes(kind, hi_rows, hi_width)
    return {name: [list(lo[name]), list(mid[name]), list(top[name])] for name in kind.inputs()}


def covers(shapes: dict, kind: Kind, rows: int, width: int) -> bool:
    """Whether an engine of these profile shapes runs strips of `rows` output rows `width` wide."""
    want = feed_shapes(kind, rows, width)
    return all(all(lo <= w <= hi for lo, w, hi in zip(shapes[n][0], want[n], shapes[n][2])) for n in kind.inputs())


def engine_rows(shapes: dict, kind: Kind, width: int) -> int:
    """The most output rows of a strip `width` wide the engine with these shapes runs (0: none)."""
    x_lo, _, x_hi = shapes["x"]
    w_in = width // 2 if kind.op == UP else width
    if not x_lo[3] <= w_in <= x_hi[3]:
        return 0
    if kind.op == UP:
        rows = (x_hi[2] - 2 * kind.halo) * 2
    else:
        rows = x_hi[2] - 2 * kind.halo
    if "res" in shapes:
        rows = min(rows, shapes["res"][2][2])
    return max(0, rows)


# --- Running


class Stats:
    """A GroupNorm's statistics over an image, added up strip by strip (in FP64)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, count: int, mean, m2):
        mean, m2 = mean.double(), m2.double()
        if not self.n:
            self.n, self.mean, self.m2 = count, mean, m2
            return
        n = self.n + count
        delta = mean - self.mean
        self.mean = self.mean + delta * (count / n)
        self.m2 = self.m2 + m2 + delta.square() * (self.n * count / n)
        self.n = n

    def affine(self, norm) -> tuple:
        """(center, scale, shift), each (1, channels, 1, 1) in FP32, that apply the GroupNorm `norm` (whose
        weight and bias are given in FP64) as (x - center) * scale + shift."""
        weight, bias, eps, groups = norm
        per = weight.numel() // groups
        rstd = torch.rsqrt(self.m2 / self.n + eps)
        center = self.mean.repeat_interleave(per, dim=1)
        scale = weight.view(1, -1) * rstd.repeat_interleave(per, dim=1)
        shift = bias.view(1, -1).expand_as(scale)
        return tuple(t.float().reshape(1, -1, 1, 1).contiguous() for t in (center, scale, shift))


def norms(fsm, found: list, device) -> dict:
    """{GroupNorm module name: (weight, bias in FP64, eps, groups)} for the stages."""
    modules = dict(fsm.named_modules())
    out = {}
    for s in found:
        if s.norm and s.norm not in out:
            m = modules[s.norm]
            out[s.norm] = (m.weight.detach().to(device, torch.float64), m.bias.detach().to(device, torch.float64), float(m.eps), int(m.num_groups))
    return out


def _row(r: int, rows: int, pad: str):
    """Row r of a tensor `rows` high padded with `pad`: the row it takes, or None for zeros."""
    if 0 <= r < rows:
        return r
    if pad == "zeros":
        return None
    if pad == "reflect":
        return -r if r < 0 else 2 * rows - 2 - r
    if pad == "replicate":
        return 0 if r < 0 else rows - 1
    return r % rows  # circular


def source_rows(stage: Stage, a: int, b: int, rows_in: int) -> list:
    """For output rows [a, b) of a stage: the rows of its input to give it (None: zeros), its halo included.
    For UP, the input's rows whose upsampling gives the conv its rows, padded as the upsampled image is."""
    if stage.kind.op == UP:
        up_rows = rows_in * 2
        out = []
        for r in range(a // 2 - 1, b // 2 + 1):
            # The conv reads upsampled rows a-1 .. b; source row r upsamples to rows 2r and 2r+1, of which
            # the halo rows' outer one is cropped: map the inner one (2r+1 above, 2r below) through the padding.
            inner = 2 * r + 1 if r < a // 2 else 2 * r
            mapped = _row(inner, up_rows, stage.pad)
            out.append(None if mapped is None else mapped // 2)
        return out
    halo = stage.kind.halo
    return [_row(r, rows_in, stage.pad) for r in range(a - halo, b + halo)]


def _runs(rows: list):
    """[(dst start, src start, length)] for the runs of consecutive rows, and the indexes of zero rows."""
    runs, zeros, i = [], [], 0
    while i < len(rows):
        if rows[i] is None:
            zeros.append(i)
            i += 1
            continue
        j = i + 1
        while j < len(rows) and rows[j] is not None and rows[j] == rows[j - 1] + 1:
            j += 1
        runs.append((i, rows[i], j - i))
        i = j
    return runs, zeros


ALIGN = 256  # bytes: where each tensor in a slab starts


def _align(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


def _layout(events: list, sizes: dict) -> tuple:
    """First-fit offsets in one slab for tensors allocated and freed in the order of `events` ([(key, what)]:
    True to allocate, False to free, or the key of a tensor whose memory it takes over): ({key: offset},
    the slab's size, the keys live when it reached that size)."""
    live, offsets, peak, at_peak = {}, {}, 0, []
    for key, what in events:
        if what is False:
            live.pop(key, None)
            continue
        if what is not True:
            live[key] = live.pop(what)
            offsets[key] = live[key][0]
            continue
        size, offset = _align(sizes[key]), 0
        for start, length in sorted(live.values()):
            if start - offset >= size:
                break
            offset = max(offset, start + length)
        live[key] = (offset, size)
        offsets[key] = offset
        if offset + size > peak:
            peak, at_peak = offset + size, list(live)
    return offsets, peak, at_peak


@dataclass
class Plan:
    """How a stripped decode runs, decided before it starts so it never runs out of VRAM: each stage's rows
    per strip; which tensors between stages are held in RAM; where each is in its slab (one in VRAM, one in
    pinned RAM); each stage's working tensors' bytes; the engines' activations at most."""

    rows: list
    host: set
    offsets: dict
    device_bytes: int
    host_bytes: int
    io: list
    act: int
    moved: int = 0  # bytes copied between RAM and VRAM, counted as the decode runs

    @property
    def vram(self) -> int:
        return self.device_bytes + max(self.io) + self.act


_pinned = {"tensor": None, "registered": False}


def _host_slab(nbytes: int):
    """Pinned RAM of at least nbytes, kept for the next decode (release_pinned frees it)."""
    t = _pinned["tensor"]
    if t is not None and t.numel() >= nbytes:
        return t
    release_pinned()
    t = torch.empty(max(nbytes, 1), dtype=torch.uint8)
    try:  # registered rather than PyTorch's pinned allocator, which rounds up to a power of two
        if int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)) == 0:
            _pinned.update(tensor=t, registered=True)
            return t
    except Exception:
        pass
    try:
        t = torch.empty(max(nbytes, 1), dtype=torch.uint8, pin_memory=True)
    except RuntimeError:  # not enough RAM to pin: copies wait for the CPU
        pass
    _pinned.update(tensor=t, registered=False)
    return t


def release_pinned():
    t = _pinned["tensor"]
    if t is not None and _pinned["registered"]:
        try:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
        except Exception:
            pass
    _pinned.update(tensor=None, registered=False)


class Held:
    """A tensor held between stages, rows first (rows, channels, width), in a slab in VRAM or in RAM."""

    def __init__(self, t, host: bool):
        self.t, self.host = t, host


class Decode:
    """One image's post part in strips. `runners[i]` runs stage i: runner(feed, outputs, last) -> outputs
    (the strip's output written to outputs["out"]; `last`: the shorter last strip), runner.activation(shapes)
    -> bytes, runner.rows(width) -> its most output rows at that width (0: none). `cache` keeps the
    engines' activation sizes ({(kind, rows, width): bytes}) between decodes."""

    def __init__(self, found: list, runners: list, norm_params: dict, dtype, device, copy_stream=None, arena=None, cache=None):
        self.stages, self.runners, self.norms = found, runners, norm_params
        self.dtype, self.device, self.copy_stream, self.arena = dtype, torch.device(device), copy_stream, arena
        self.size = torch.tensor([], dtype=dtype).element_size()
        self.cache = cache if cache is not None else {}
        self.last_use = {}
        for n, st in enumerate(found):
            for key in (st.src, st.res):
                if key is not None:
                    self.last_use[key] = n

    def geometry(self, stage: Stage, h: int, w: int) -> tuple:
        """(output rows, output width, input rows) for latents of h x w."""
        return h * stage.scale, w * stage.scale, h * stage.in_scale

    def _shapes(self, h: int, w: int) -> dict:
        """{held tensor: its shape, rows first}."""
        c = self.stages[0].kind.cin
        shapes = {-2: (h, c, w), -1: (h, c, w)}
        for n, st in enumerate(self.stages):
            rows, width, _ = self.geometry(st, h, w)
            shapes[n] = (rows, st.kind.cout, width)
        return shapes

    def _in_place(self) -> set:
        """The stages whose output can take over their residual's memory as they go: its last use, read
        without a halo, a row of the output no larger than one of it (so a strip's output only overwrites
        rows of it already read)."""
        out = set()
        for n, st in enumerate(self.stages):
            if st.res is not None and self.last_use[st.res] == n and st.kind.cout <= st.kind.res:
                out.add(n)
        return out

    def _events(self, in_place: set) -> list:
        events = [(-2, True), (-1, True)]
        for n, st in enumerate(self.stages):
            events.append((n, st.res if n in in_place else True))
            events += [(key, False) for key in (st.src, st.res) if key is not None and self.last_use[key] == n and not (n in in_place and key == st.res)]
        return events

    def _activation(self, n: int, rows: int, width: int) -> int:
        kind = self.stages[n].kind
        key = (kind.name, rows, width)
        if key not in self.cache:
            self.cache[key] = int(self.runners[n].activation(feed_shapes(kind, rows, width)))
        return self.cache[key]

    def _io(self, n: int, rows: int, width: int, host: set) -> int:
        """Bytes of a stage's working tensors: its input, residual and output for a strip (and two strips'
        worth of staging for those held in RAM), and the rows to keep."""
        st = self.stages[n]
        shapes = feed_shapes(st.kind, rows, width)
        x, out = math.prod(shapes["x"]) * self.size, st.kind.cout * rows * width * self.size
        res = math.prod(shapes["res"]) * self.size if "res" in shapes else 0
        total = _align(x) + _align(res) + _align(out) + _align(shapes["x"][2] * self.size)
        total += 2 * _align(x) if st.src in host else 0
        total += 2 * _align(res) if st.res is not None and st.res in host else 0
        return total + (2 * _align(out) if n in host else 0)

    def plan(self, h: int, w: int, free: int, force_host: bool = False):
        """The Plan for latents of h x w in `free` bytes of VRAM, or None if not even the smallest strips
        fit: everything in VRAM with strips as large as fit (down to an 8th of what the engines take); else
        the tensors held at the VRAM's peak moved to RAM, the largest first, until it fits."""
        caps, widths = [], []
        for n, st in enumerate(self.stages):
            out_rows, width, _ = self.geometry(st, h, w)
            cap = min(out_rows, self.runners[n].rows(width))
            if st.kind.op == UP:
                cap -= cap % 2
            if cap < (2 if st.kind.op == UP else 1):
                return None
            caps.append(cap)
            widths.append(width)
        budget = free - HEADROOM - free // 32
        shapes = self._shapes(h, w)
        sizes = {k: math.prod(v) * self.size for k, v in shapes.items()}
        in_place = self._in_place()

        def attempt(div: int, host: set):
            rows = []
            for n, cap in enumerate(caps):
                step = 2 if self.stages[n].kind.op == UP else 1
                rows.append(max(step, cap // div // step * step))
            act = max(self._activation(n, r, widths[n]) for n, r in enumerate(rows))
            io = [self._io(n, r, widths[n], host) for n, r in enumerate(rows)]
            events = self._events({n for n in in_place if (n in host) == (self.stages[n].res in host)})
            offsets, peak, at_peak = _layout([e for e in events if e[0] not in host], sizes)
            return rows, act, io, offsets, peak, at_peak

        def done(rows, act, io, offsets, peak, host):
            events = self._events({n for n in in_place if (n in host) == (self.stages[n].res in host)})
            host_offsets, host_bytes, _ = _layout([e for e in events if e[0] in host], sizes)
            return Plan(rows, set(host), {**offsets, **host_offsets}, peak, host_bytes, io, act)

        if force_host:
            host = set(range(len(self.stages)))
            rows, act, io, offsets, peak, _ = attempt(1, host)
            return done(rows, act, io, offsets, peak, host)
        for div in (1, 2, 4, 8):
            rows, act, io, offsets, peak, _ = attempt(div, set())
            if peak + max(io) + act <= budget:
                return done(rows, act, io, offsets, peak, set())
        for div in (1, 2, 4, 8):
            host = set()
            while True:
                rows, act, io, offsets, peak, at_peak = attempt(div, host)
                if peak + max(io) + act <= budget:
                    return done(rows, act, io, offsets, peak, host)
                movable = [k for k in at_peak if k >= 0 and k not in host]
                if not movable:
                    break
                host.add(max(movable, key=sizes.get))
        return None

    def run(self, x, attention, plan: Plan):
        """The image (1, out channels, H, W) for the mid block's input x and its attention's output (each
        (1, channels, h, w), in VRAM)."""
        h, w = x.shape[2], x.shape[3]
        shapes = self._shapes(h, w)
        slab = torch.empty(max(plan.device_bytes, 1), dtype=torch.uint8, device=self.device)
        ram = _host_slab(plan.host_bytes) if plan.host else None
        io = torch.empty(max(max(plan.io), 1), dtype=torch.uint8, device=self.device)
        if self.arena is not None:
            self.arena.get(plan.act)  # once, at its largest: every engine binds it as is
        held = {}
        for key, shape in shapes.items():
            host = key in plan.host
            base, offset = (ram, plan.offsets[key]) if host else (slab, plan.offsets[key])
            nbytes = math.prod(shape) * self.size
            held[key] = Held(base[offset:offset + nbytes].view(self.dtype).view(shape), host)
        held[-2].t.copy_(x[0].permute(1, 0, 2))
        held[-1].t.copy_(attention[0].permute(1, 0, 2))
        plan.moved = 0
        stats = {}
        for n, stage in enumerate(self.stages):
            stats[n] = self._sweep(n, stage, held, stats.get(stage.src), plan, io)
            for key in (stage.src, stage.res):
                if key is not None and self.last_use[key] == n:
                    stats.pop(key, None)
        image = held[len(self.stages) - 1].t
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        return image.to(self.device).permute(1, 0, 2).unsqueeze(0).contiguous()

    def _sweep(self, n: int, stage: Stage, held: dict, src_stats, plan: Plan, io):
        kind, runner = stage.kind, self.runners[n]
        src, out = held[stage.src], held[n]
        res = held[stage.res] if stage.res is not None else None
        out_rows, width = out.t.shape[0], out.t.shape[2]
        rows = plan.rows[n]
        shapes = feed_shapes(kind, rows, width)
        c_in, rows_in, w_in = shapes["x"][1], shapes["x"][2], shapes["x"][3]
        offset = [0]

        def take(elements):  # a flat view of the working slab
            nbytes = elements * self.size
            view = io[offset[0]:offset[0] + nbytes].view(self.dtype)
            offset[0] += _align(nbytes)
            return view

        x_flat = take(c_in * rows_in * w_in)
        res_flat = take(kind.res * rows * width) if res is not None else None
        out_flat = take(kind.cout * rows * width)
        keep_flat = take(rows_in)
        x_stage = [take(c_in * rows_in * w_in) for _ in range(2)] if src.host else None
        res_stage = [take(kind.res * rows * width) for _ in range(2)] if res is not None and res.host else None
        out_stage = [take(kind.cout * rows * width) for _ in range(2)] if out.host else None
        fixed = {}
        if stage.norm:
            center, scale, shift = src_stats.affine(self.norms[stage.norm])
            fixed = {"center": center, "scale": scale, "shift": shift}
        stats = Stats() if kind.groups else None
        per_group = kind.cout // kind.groups * width if kind.groups else 0
        strips = [(a, min(a + rows, out_rows)) for a in range(0, out_rows, rows)]
        copy = self.copy_stream if (x_stage or res_stage or out_stage) else None
        compute = torch.cuda.current_stream(self.device) if copy is not None else None
        if copy is not None:
            copy.wait_stream(compute)  # the working slab's last users (the stage before) are done
        loaded, consumed, stored = {}, [None, None], [None, None]

        def load(k):  # what's in RAM, into staging on the copy stream (a strip ahead)
            a, b = strips[k]
            slot = k % 2
            parts = []
            if x_stage:
                parts.append(("x", src, source_rows(stage, a, b, src.t.shape[0]), x_stage[slot]))
            if res_stage:
                parts.append(("res", res, list(range(a, b)), res_stage[slot]))
            context = torch.cuda.stream(copy) if copy is not None else _nothing()
            with context:
                if copy is not None and consumed[slot] is not None:
                    copy.wait_event(consumed[slot])  # the strip two back is done with this staging
                views = {}
                for name, held_t, want, flat in parts:
                    view = flat[:len(want) * held_t.t[0].numel()].view((len(want),) + tuple(held_t.t.shape[1:]))
                    runs, zeros = _runs(want)
                    for d, start, length in runs:
                        view[d:d + length].copy_(held_t.t[start:start + length], non_blocking=True)
                        plan.moved += length * held_t.t[0].numel() * self.size
                    for i in zeros:
                        view[i].zero_()
                    views[name] = view
                event = None
                if copy is not None:
                    event = torch.cuda.Event()
                    event.record(copy)
            loaded[k] = (views, event)

        def gather(dst, held_t, want):  # rows of a tensor in VRAM, straight into the strip's input
            runs, zeros = _runs(want)
            for d, start, length in runs:
                dst[0, :, d:d + length, :].copy_(held_t.t[start:start + length].permute(1, 0, 2))
            for i in zeros:
                dst[0, :, i, :].zero_()

        if x_stage or res_stage:
            load(0)
        for k, (a, b) in enumerate(strips):
            r = b - a
            sh = feed_shapes(kind, r, width)
            want = source_rows(stage, a, b, src.t.shape[0])
            x = x_flat[:math.prod(sh["x"])].view(sh["x"])
            staged = {}
            if x_stage or res_stage:
                if k + 1 < len(strips):
                    load(k + 1)  # its copies run while this strip computes
                staged, event = loaded.pop(k)
                if event is not None:
                    compute.wait_event(event)
            if "x" in staged:
                x[0].copy_(staged["x"].permute(1, 0, 2))
            else:
                gather(x, src, want)
            feed = {"x": x}
            if res is not None:
                rv = res_flat[:math.prod(sh["res"])].view(sh["res"])
                if "res" in staged:
                    rv[0].copy_(staged["res"].permute(1, 0, 2))
                else:
                    gather(rv, res, list(range(a, b)))
                feed["res"] = rv
            if staged and compute is not None:
                done = torch.cuda.Event()
                done.record(compute)
                consumed[k % 2] = done
            if "keep" in kind.inputs():
                keep = keep_flat[:len(want)].view(1, 1, -1, 1)
                keep.fill_(1)
                if stage.pad == "zeros":
                    for i, row in enumerate(want):
                        if row is None:
                            keep[:, :, i].zero_()
                feed["keep"] = keep
            feed.update(fixed)
            result_out = out_flat[:kind.cout * r * width].view(1, kind.cout, r, width)
            result = runner(feed, {"out": result_out}, r != rows)
            if stats is not None:
                stats.add(per_group * r, result["mean"], result["m2"])
            rows_first = result_out[0].permute(1, 0, 2)
            if out_stage is None:
                out.t[a:b].copy_(rows_first)
                continue
            slot = k % 2
            staging = out_stage[slot][:r * kind.cout * width].view(r, kind.cout, width)
            if compute is not None and stored[slot] is not None:
                compute.wait_event(stored[slot])  # its last copy out is done
            staging.copy_(rows_first)
            plan.moved += staging.numel() * self.size
            if copy is None:
                out.t[a:b].copy_(staging)
                continue
            ready = torch.cuda.Event()
            ready.record(compute)
            with torch.cuda.stream(copy):
                copy.wait_event(ready)
                out.t[a:b].copy_(staging, non_blocking=True)
                stored[slot] = torch.cuda.Event()
                stored[slot].record(copy)
        if copy is not None:
            compute.wait_stream(copy)
        return stats


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TorchRunner:
    """A stage on PyTorch (for checks, and where no engine runs it)."""

    def __init__(self, module: StageModule):
        self.module = module

    def __call__(self, feed: dict, outputs: dict, last: bool = False) -> dict:
        kind = self.module.kind
        with torch.no_grad():
            result = dict(zip(kind.outputs(), self.module(*[feed[n] for n in kind.inputs()])))
        outputs["out"].copy_(result["out"])
        result["out"] = outputs["out"]
        return result

    def activation(self, shapes: dict) -> int:
        return 0

    def rows(self, width: int) -> int:
        return max_rows(self.module.kind, width)


class EngineRunner:
    """A stage on its kind's engine (engine.py's Engine), loaded with the stage's weights: one execution
    context for its strips, another for the shorter last one, so neither changes shapes from one decode of
    a size to the next; and the kind's probe for activation sizes."""

    def __init__(self, engine, last, probe, shapes: dict, kind: Kind):
        self.engine, self.last, self.probe, self.shapes, self.kind = engine, last, probe, shapes, kind

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def __call__(self, feed: dict, outputs: dict, last: bool = False) -> dict:
        return (self.last if last else self.engine).infer(feed, 0, self._stream(), outputs)

    def activation(self, shapes: dict) -> int:
        return self.probe.estimate_activation_bytes(0, {n: tuple(s) for n, s in shapes.items()}, self._stream())

    def rows(self, width: int) -> int:
        return engine_rows(self.shapes, self.kind, width)
