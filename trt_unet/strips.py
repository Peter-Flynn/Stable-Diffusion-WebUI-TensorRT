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
from dataclasses import asdict, dataclass, field

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


def _vram_now(held: int) -> dict:
    try:
        return {"free": torch.cuda.mem_get_info()[0], "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(), "held": held}
    except Exception:
        return {"held": held}


class Held:
    """A tensor held between stages, rows first (rows, channels, width): in VRAM or in (pinned) RAM."""

    def __init__(self, rows: int, channels: int, width: int, dtype, device, host: bool):
        self.host = host
        if host:
            try:
                self.t = torch.empty((rows, channels, width), dtype=dtype, pin_memory=torch.device(device).type == "cuda")
            except RuntimeError:  # not enough RAM to pin: copies wait for the CPU
                self.t = torch.empty((rows, channels, width), dtype=dtype)
        else:
            self.t = torch.empty((rows, channels, width), dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.t.numel() * self.t.element_size()

    @classmethod
    def of(cls, nchw):
        held = cls.__new__(cls)
        held.host, held.t = False, nchw[0].permute(1, 0, 2).contiguous()
        return held


@dataclass
class Plan:
    """How a stripped decode runs: each stage's rows per strip, and how much VRAM the held tensors may take."""

    rows: list
    work: int  # VRAM for the strips' working tensors and the engines' activations, at most
    held: int  # VRAM for held tensors (less once VRAM ran short)
    host: list = field(default_factory=list)  # whether each stage's output went to RAM, decided as it runs
    short: list = field(default_factory=list)  # the stages that ran out of VRAM (and ran again with less)
    when: dict = field(default_factory=dict)  # the VRAM then, the first time: driver's free, PyTorch's allocated and reserved, held


class Decode:
    """One image's post part in strips. `runners[i]` runs stage i: runner(feed) -> outputs,
    runner.activation(shapes) -> bytes, runner.rows(width) -> its most output rows at that width (0: none)."""

    def __init__(self, found: list, runners: list, norm_params: dict, dtype, device, copy_stream=None):
        self.stages, self.runners, self.norms = found, runners, norm_params
        self.dtype, self.device, self.copy_stream = dtype, torch.device(device), copy_stream
        self.size = torch.tensor([], dtype=dtype).element_size()

    def geometry(self, stage: Stage, h: int, w: int) -> tuple:
        """(output rows, output width, input rows) for latents of h x w."""
        return h * stage.scale, w * stage.scale, h * stage.in_scale

    def work(self, n: int, rows: int, h: int, w: int) -> int:
        """VRAM a strip of `rows` output rows of stage n takes: its engine's activations and its tensors
        (input, residual and output, each up to three times: staged, rearranged and in flight)."""
        stage, runner = self.stages[n], self.runners[n]
        out_rows, width, _ = self.geometry(stage, h, w)
        shapes = feed_shapes(stage.kind, rows, width)
        io = sum(math.prod(shapes[name]) for name in ("x", "res") if name in shapes) + stage.kind.cout * rows * width
        return int(runner.activation(shapes)) + 3 * io * self.size

    def plan(self, h: int, w: int, free: int):
        """Each stage's strip rows for latents of h x w with `free` bytes of VRAM (the latent-size tensors
        already in it), or None if a stage can't run: as many rows as its engine takes, fewer if they don't
        fit; the VRAM the strips don't take holds what it can of the tensors between stages."""
        caps = []
        for n, stage in enumerate(self.stages):
            out_rows, width, _ = self.geometry(stage, h, w)
            cap = min(out_rows, self.runners[n].rows(width))
            if stage.kind.op == UP:
                cap -= cap % 2
            if cap < (2 if stage.kind.op == UP else 1):
                return None
            caps.append(cap)
        budget = free - HEADROOM - free // 16
        rows = list(caps)
        worst = max(self.work(n, r, h, w) for n, r in enumerate(rows))
        if worst > budget:
            for n, cap in enumerate(caps):
                lo, hi, step = (2, cap, 2) if self.stages[n].kind.op == UP else (1, cap, 1)
                if self.work(n, lo, h, w) > budget:
                    return None
                while hi - lo > step:  # the most rows that fit
                    mid = (lo + hi) // 2
                    mid -= mid % step
                    lo, hi = (mid, hi) if self.work(n, mid, h, w) <= budget else (lo, mid)
                rows[n] = hi if self.work(n, hi, h, w) <= budget else lo
            worst = max(self.work(n, r, h, w) for n, r in enumerate(rows))
        return Plan(rows, worst, max(0, budget - worst))

    def run(self, x, attention, plan: Plan, force_host: bool = False):
        """The image (1, out channels, H, W) for the mid block's input x and its attention's output (each
        (1, channels, h, w), in VRAM). A stage that runs out of VRAM runs again with less in it: its output
        in RAM, then its inputs, then fewer rows per strip; and the stages after it hold no more in VRAM."""
        h, w = x.shape[2], x.shape[3]
        plan.host, plan.short = [], []
        held = {-2: Held.of(x), -1: Held.of(attention)}
        stats = {}
        self.in_vram = 0  # bytes of held tensors in VRAM, past the latent-size ones (which `free` counted)
        last_use = {}
        for n, s in enumerate(self.stages):
            last_use[s.src] = n
            if s.res is not None:
                last_use[s.res] = n
        image = None
        for n, stage in enumerate(self.stages):
            out_rows, width, _ = self.geometry(stage, h, w)
            nbytes = out_rows * stage.kind.cout * width * self.size
            host = force_host or self.in_vram + nbytes > plan.held
            out = None
            while True:
                short = False
                try:
                    if out is None:
                        out = Held(out_rows, stage.kind.cout, width, self.dtype, self.device, host)
                        self.in_vram += 0 if host else nbytes
                    stats[n] = self._sweep(n, stage, held[stage.src], held.get(stage.res), out, stats.get(stage.src), plan.rows[n])
                    break
                except torch.cuda.OutOfMemoryError:
                    short = True
                if short:  # out of the except block: its traceback no longer holds the strip's tensors
                    self._settle()
                    if not plan.short:
                        plan.when = _vram_now(self.in_vram)
                    plan.short.append(n)
                    if not host:  # its output to RAM
                        if out is not None:
                            self.in_vram -= nbytes
                        out, host = None, True
                    elif not self._spill(held, stage):  # else an input to RAM, else fewer rows
                        step = 2 if stage.kind.op == UP else 1
                        if plan.rows[n] <= step:
                            raise torch.cuda.OutOfMemoryError(f"Not even a strip of {step} rows of the VAE decoder's stage {n} fits in the VRAM free")
                        plan.rows[n] = max(step, plan.rows[n] // 2 // step * step)
                    plan.held = min(plan.held, self.in_vram)
                    torch.cuda.empty_cache()
            plan.host.append(out.host)
            for buf in {stage.src, stage.res} - {None}:
                if last_use.get(buf) == n:
                    gone = held.pop(buf)
                    if not gone.host and buf >= 0:
                        self.in_vram -= gone.nbytes
                    stats.pop(buf, None)
            held[n] = out
            if stage.kind.op == OUT:
                image = out
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        return image.t.to(self.device).permute(1, 0, 2).unsqueeze(0).contiguous()

    def _settle(self):
        """After running out of VRAM mid-sweep: the copies queued finish before anything is freed or moved."""
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def _spill(self, held: dict, stage: Stage) -> bool:
        """Moves the largest of a stage's inputs in VRAM to RAM; whether there was one."""
        keys = [k for k in (stage.src, stage.res) if k is not None and not held[k].host]
        if not keys:
            return False
        key = max(keys, key=lambda k: held[k].nbytes)
        old = held[key]
        new = Held(*old.t.shape, self.dtype, self.device, True)
        new.t.copy_(old.t)
        held[key] = new
        if key >= 0:
            self.in_vram -= old.nbytes
        return True

    def _sweep(self, n: int, stage: Stage, src: Held, res, out: Held, src_stats, rows: int):
        kind, runner = stage.kind, self.runners[n]
        out_rows, width = out.t.shape[0], out.t.shape[2]
        fixed = {}
        if stage.norm:
            center, scale, shift = src_stats.affine(self.norms[stage.norm])
            fixed = {"center": center, "scale": scale, "shift": shift}
        stats = Stats() if kind.groups else None
        per_group = kind.cout // kind.groups * width if kind.groups else 0
        copying = self.copy_stream is not None and (src.host or (res is not None and res.host) or out.host)
        pipe = _Pipe(self, src, res, out, rows, copying)
        strips = [(a, min(a + rows, out_rows)) for a in range(0, out_rows, rows)]
        pipe.load(stage, strips, 0)
        for k, (a, b) in enumerate(strips):
            feed = pipe.inputs(k)
            if k + 1 < len(strips):
                pipe.load(stage, strips, k + 1)  # its copies run while this strip computes
            if "keep" in kind.inputs():
                keep = [1.0 if r is not None or stage.pad != "zeros" else 0.0 for r in source_rows(stage, a, b, src.t.shape[0])]
                feed["keep"] = torch.tensor(keep, dtype=self.dtype).view(1, 1, -1, 1).to(self.device, non_blocking=True)
            feed.update(fixed)
            result = runner(feed)
            del feed
            if stats is not None:
                stats.add(per_group * (b - a), result["mean"], result["m2"])
            pipe.store(k, a, b, result["out"])
            del result
        pipe.finish()
        return stats


class _Pipe:
    """A sweep's copies: inputs gathered (from VRAM, or copied in from RAM on the copy stream, a strip
    ahead), outputs put back (to RAM on the copy stream, while the next strip computes)."""

    def __init__(self, decode: Decode, src: Held, res, out: Held, rows: int, copying: bool):
        self.d, self.src, self.res, self.out = decode, src, res, out
        self.copying = copying
        self.loaded = {}  # strip -> (its inputs: staged in VRAM, or (Held, rows) to gather; the copies' event)
        self.stored = [None, None]  # the events of the last two strips' copies out to RAM

    def _gather(self, held: Held, rows: list):
        """The rows (None: zeros) of a held tensor, rows first, in VRAM."""
        t = held.t
        staged = torch.empty((len(rows),) + tuple(t.shape[1:]), dtype=t.dtype, device=self.d.device)
        runs, zeros = _runs(rows)
        for dst, start, length in runs:
            staged[dst:dst + length].copy_(t[start:start + length], non_blocking=True)
        for i in zeros:
            staged[i].zero_()
        return staged

    def load(self, stage: Stage, strips: list, k: int):
        a, b = strips[k]
        want = {"x": (self.src, source_rows(stage, a, b, self.src.t.shape[0]))}
        if self.res is not None:
            want["res"] = (self.res, list(range(a, b)))
        if not self.copying or not any(held.host for held, _ in want.values()):
            self.loaded[k] = (want, None)
            return
        # From RAM on the copy stream: the caching allocator keeps each staged copy until the compute
        # stream is done with it (record_stream)
        compute, copy = torch.cuda.current_stream(self.d.device), self.d.copy_stream
        parts = dict(want)
        with torch.cuda.stream(copy):
            for name, (held, rows) in want.items():
                if held.host:
                    parts[name] = self._gather(held, rows)
                    parts[name].record_stream(compute)
            event = torch.cuda.Event()
            event.record(copy)
        self.loaded[k] = (parts, event)

    def inputs(self, k: int) -> dict:
        parts, event = self.loaded.pop(k)
        if event is not None:
            torch.cuda.current_stream(self.d.device).wait_event(event)
        feed = {}
        for name, part in parts.items():
            if isinstance(part, tuple):  # in VRAM (or no copy stream): gathered here
                part = self._gather(*part)
            feed[name] = part.permute(1, 0, 2).unsqueeze(0).contiguous()
        return feed

    def store(self, k: int, a: int, b: int, result):
        rows_first = result[0].permute(1, 0, 2)
        if not self.out.host:
            self.out.t[a:b].copy_(rows_first)
            return
        if not self.copying:
            self.out.t[a:b].copy_(rows_first.to("cpu") if self.d.device.type != "cpu" else rows_first)
            return
        compute, copy = torch.cuda.current_stream(self.d.device), self.d.copy_stream
        slot = k % 2
        if self.stored[slot] is not None:
            compute.wait_event(self.stored[slot])  # at most two strips' outputs waiting to go out
        staged = rows_first.contiguous()
        ready = torch.cuda.Event()
        ready.record(compute)
        with torch.cuda.stream(copy):
            copy.wait_event(ready)
            self.out.t[a:b].copy_(staged, non_blocking=True)
            staged.record_stream(copy)
            done = torch.cuda.Event()
            done.record(copy)
        self.stored[slot] = done

    def finish(self):
        if self.copying:
            torch.cuda.current_stream(self.d.device).wait_stream(self.d.copy_stream)
        self.stored = [None, None]
        self.loaded.clear()


class TorchRunner:
    """A stage on PyTorch (for checks, and where no engine runs it)."""

    def __init__(self, module: StageModule):
        self.module = module

    def __call__(self, feed: dict) -> dict:
        kind = self.module.kind
        with torch.no_grad():
            result = self.module(*[feed[n] for n in kind.inputs()])
        return dict(zip(kind.outputs(), result))

    def activation(self, shapes: dict) -> int:
        return 0

    def rows(self, width: int) -> int:
        return max_rows(self.module.kind, width)


class EngineRunner:
    """A stage on its kind's engine (engine.py's Engine), loaded with the stage's weights."""

    def __init__(self, engine, shapes: dict, kind: Kind):
        self.engine, self.shapes, self.kind = engine, shapes, kind

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def __call__(self, feed: dict) -> dict:
        return self.engine.infer(feed, 0, self._stream())

    def activation(self, shapes: dict) -> int:
        return self.engine.estimate_activation_bytes(0, {n: tuple(s) for n, s in shapes.items()}, self._stream())

    def rows(self, width: int) -> int:
        return engine_rows(self.shapes, self.kind, width)
