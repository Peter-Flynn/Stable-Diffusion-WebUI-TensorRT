"""Tensor Stripped: the VAE decoder's "post" part (vae.py) in horizontal strips, for decodes too large for its
post engine in the VRAM there can be; and the encoder up to its mid block's attention likewise, for encodes too
large for its engines. Same results as the whole decode or encode, up to rounding.

The post part is cut into stages at its GroupNorms, which need statistics of their whole input: each stage
outputs, with every strip, its output's mean and spread over the strip for each group, which add up (Chan
et al.'s parallel variance) to the next GroupNorm's over the whole image. So a stage finishes before the
next starts, and what's between stages is held whole. A stage runs on one strip of rows at a time, given
its input's rows with as many more on each side as its 3x3 convolutions reach (the image's padding there:
zeros, or reflected, replicated or wrapped rows, as the convolutions pad).

Each resnet block runs in one of three ways, the plan (Decode.plan) choosing for each the fastest that
fits in the VRAM free:
- STORE: a stage for its first conv, whose output is held, and one for its second (plus its shortcut);
- RECOMPUTE: the first conv's stage only for its statistics, then a BLOCK stage computing both convs from
  the block's input, so nothing but the block's input is held (the first conv runs twice);
- FUSED, for a block right after an upsampling: the upsampling's stage only for its statistics, an UPCONV
  stage for the first conv's, and an UPBLOCK stage computing the upsampling and both convs from what's
  upsampled, so the upsampled tensor (the largest, at the image's size) is never held at all.
What can't be held in VRAM even so goes to pinned RAM, copied in and out while the engine works on the
strip before.

Stages with the same layout share an engine plan ("kind"), each loaded with its own weights.

The encoder (Encode) runs the same way, from the image: its first conv (IN, on the image's rows), its resnet
blocks (as the decoder's, STORE or RECOMPUTE), and its downsamplings (DOWN: a 3x3 conv with stride 2 on its
input padded with a row and a column of zeros at the bottom and the right), through its mid block's first
resnet block, whose output and statistics (for the attention's GroupNorm) it returns.

In FP16 the decoder runs scaled by a power of two (vae.py says why and how): the tensors between its
GroupNorms are that much smaller than the VAE makes them. Here that takes three things: each conv's input
is multiplied by the scale where rows are kept or zeroed anyway ("keep"), the attention's output is
scaled as it comes in, and a GroupNorm's epsilon is scaled with the variance it's added to."""

import math
import time
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn.functional as F

STRIP_BYTES = 128 * 2**20  # the most a strip of a stage's largest tensor takes: what kinds are built for
CONV, UP, PROJ, OUT, BLOCK, UPCONV, UPBLOCK = "conv", "up", "proj", "out", "block", "upconv", "upblock"
IN, DOWN = "in", "down"  # the encoder's
CONVS = {PROJ: 0, CONV: 1, OUT: 1, UP: 1, BLOCK: 2, UPCONV: 2, UPBLOCK: 3, IN: 1, DOWN: 1}  # 3x3 convs in a row: the rows of halo
UPSAMPLING = (UP, UPCONV, UPBLOCK)
TWO_NORMS = (BLOCK, UPBLOCK)
PADDINGS = ("zeros", "reflect", "replicate", "circular")
CHAINED = ("zeros", "circular")  # paddings the kinds with several 3x3 convs in a row take
STORE, RECOMPUTE, FUSED = "store", "recompute", "fused"


@dataclass(frozen=True)
class Kind:
    """What one engine computes. With x normalized by a GroupNorm (given as center, scale and shift), then
    SiLU and a 3x3 conv: CONV, plus a residual ("res" channels, through a 1x1 conv when that's not cout);
    OUT, the image. UP: nearest 2x upsampling and a 3x3 conv. PROJ: a 1x1 conv plus a residual. BLOCK: a
    resnet block (a GroupNorm, SiLU and 3x3 conv twice, plus its shortcut). UPCONV: UP, then the first
    GroupNorm, SiLU and conv of the block after it, for their statistics only. UPBLOCK: UP and that block.
    The encoder's: IN, a 3x3 conv on the image; DOWN, a 3x3 conv with stride 2 on x padded with zeros at the
    bottom and the right (its halo: the row below the strip's last).
    With groups, the output's mean and spread ("m2", the sum of squared deviations) per group."""

    op: str
    cin: int
    cout: int
    res: int = 0
    groups: int = 0
    pad: str = "zeros"  # the 3x3 convs' padding across (their padding down is the strips' rows)

    @property
    def name(self) -> str:
        res = f"r{self.res}" if self.res else ""
        groups = f"g{self.groups}" if self.groups else ""
        pad = "" if self.pad == "zeros" else f"-{self.pad}"
        return f"{self.op}{self.cin}-{self.cout}{res}{groups}{pad}"

    @property
    def shortcut(self) -> bool:
        if self.op == CONV:
            return bool(self.res) and self.res != self.cout
        return self.op in TWO_NORMS and self.cin != self.cout

    @property
    def halo(self) -> int:
        """Rows beyond a strip's on each side that its 3x3 convs reach, at the output's size."""
        return CONVS[self.op]

    @property
    def step(self) -> int:
        """Strips' rows come in multiples of this (an upsampling's, whole source rows)."""
        return 2 if self.op in UPSAMPLING else 1

    def inputs(self) -> tuple:
        if self.op in (UP, IN, DOWN):
            return ("x",)
        if self.op == PROJ:
            return ("x", "res")
        names = ("x", "keep", "center", "scale", "shift")
        if self.op in TWO_NORMS:
            names += ("keep2", "center2", "scale2", "shift2")
        return names + (("res",) if self.op == CONV and self.res else ())

    def outputs(self) -> tuple:
        if self.op == UPCONV:
            return ("mean", "m2")
        return ("out",) + (("mean", "m2") if self.groups else ())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Stage:
    kind: Kind
    weights: dict  # the engine's weight name -> the decoder's (in the first stage model's state dict)
    scale: int  # the output's size in latents (1 at the mid block, 8 at the image)
    src: int  # the buffer of its input: the index of the stage that makes it (-2: the mid block's input, -1: its attention)
    res: int = None  # the buffer of its residual (CONV, PROJ)
    norms: tuple = ()  # ((GroupNorm module name, the stage whose output's statistics it takes), ...)
    pad: str = "zeros"  # its convs' padding, which the strips' rows give down the image
    keep: bool = True  # whether its output is held for the stages after it (else only its statistics are)
    block: int = None  # the resnet block it's part of (an upsampling: of the block after it)

    @property
    def in_scale(self) -> int:
        if self.kind.op == DOWN:
            return self.scale * 2
        return self.scale // 2 if self.kind.op in UPSAMPLING else self.scale

    @property
    def identity(self) -> tuple:
        """What its engine is: its kind's plan with its weights."""
        return (self.kind.name, tuple(sorted(self.weights.items())))


def blocks(fsm) -> list:
    """The decoder's resnet blocks after its attention, in order: [(prefix, module, its output's scale, the
    prefix of the upsampling right before it or None)]."""
    d = fsm.decoder
    found, scale, up = [("decoder.mid.block_2.", d.mid.block_2, 1, None)], 1, None
    for i in reversed(range(d.num_resolutions)):
        for j, blk in enumerate(d.up[i].block):
            found.append((f"decoder.up.{i}.block.{j}.", blk, scale, up))
            up = None
        if i != 0:
            up, scale = f"decoder.up.{i}.upsample.", scale * 2
    return found


def choices(fsm, padding: dict) -> list:
    """For each block, the ways it can run (STORE always; RECOMPUTE and FUSED when its convs pad with zeros or
    wrap the same way, FUSED only right after an upsampling), from fastest to holding the least."""
    pad = lambda name: padding.get(name, "zeros")  # noqa: E731
    out = []
    for prefix, blk, _, up in blocks(fsm):
        pads = {pad(prefix + "conv1"), pad(prefix + "conv2")}
        chained = len(pads) == 1 and pads <= set(CHAINED) and not getattr(blk, "use_conv_shortcut", False)
        ways = [STORE] + ([RECOMPUTE] if chained else [])
        if chained and up is not None and pad(up + "conv") in pads:
            ways.append(FUSED)
        out.append(tuple(ways))
    return out


def stages(fsm, padding: dict, modes: dict = None) -> list:
    """The post part's stages for a first stage model (or what vae.fresh makes), the config's padding, and
    each block's way ({block: STORE, RECOMPUTE or FUSED}; STORE where not given)."""
    d = fsm.decoder
    modes = modes or {}
    pad = lambda name: padding.get(name, "zeros")  # noqa: E731
    found = []

    def add(op, cin, cout, convs, scale, src, block, res=None, res_channels=0, norms=(), keep=True):
        weights = {}
        for local, module in convs.items():
            weights[f"{local}.weight"], weights[f"{local}.bias"] = f"{module}.weight", f"{module}.bias"
        main = convs.get("conv") or convs.get("conv1") or convs["up"]
        found.append(Stage(Kind(op, cin, cout, res_channels, 0, pad(main)), weights, scale, src, res, tuple(norms), pad(main), keep, block))
        return len(found) - 1

    c = int(d.mid.attn_1.in_channels)
    x = add(PROJ, c, c, {"conv": "decoder.mid.attn_1.proj_out"}, 1, -1, 0, res=-2, res_channels=c)
    listed = blocks(fsm)
    for b, (prefix, blk, scale, up) in enumerate(listed):
        cin, cout = int(blk.in_channels), int(blk.out_channels)
        if getattr(blk, "use_conv_shortcut", False) and cin != cout:
            raise ValueError("a resnet block with a 3x3 shortcut")
        mode = modes.get(b, STORE)
        tail = {"nin": prefix + "nin_shortcut"} if cin != cout else {}
        if up is not None and mode == FUSED:
            u = add(UP, cin, cin, {"conv": up + "conv"}, scale, x, b, keep=False)
            h = add(UPCONV, cin, cout, {"up": up + "conv", "conv1": prefix + "conv1"}, scale, x, b, norms=((prefix + "norm1", u),), keep=False)
            convs = {"up": up + "conv", "conv1": prefix + "conv1", "conv2": prefix + "conv2", **tail}
            x = add(UPBLOCK, cin, cout, convs, scale, x, b, norms=((prefix + "norm1", u), (prefix + "norm2", h)))
            continue
        if up is not None:
            x = add(UP, cin, cin, {"conv": up + "conv"}, scale, x, b)
        if mode == STORE:
            h = add(CONV, cin, cout, {"conv": prefix + "conv1"}, scale, x, b, norms=((prefix + "norm1", x),))
            x = add(CONV, cout, cout, {"conv": prefix + "conv2", **tail}, scale, h, b, res=x, res_channels=cin, norms=((prefix + "norm2", h),))
        else:
            h = add(CONV, cin, cout, {"conv": prefix + "conv1"}, scale, x, b, norms=((prefix + "norm1", x),), keep=False)
            convs = {"conv1": prefix + "conv1", "conv2": prefix + "conv2", **tail}
            x = add(BLOCK, cin, cout, convs, scale, x, b, norms=((prefix + "norm1", x), (prefix + "norm2", h)))
    c = int(d.norm_out.num_channels)
    add(OUT, c, int(d.conv_out.out_channels), {"conv": "decoder.conv_out"}, listed[-1][2], x, len(listed), norms=(("decoder.norm_out", x),))
    _groups(found, fsm)
    return found


def _groups(found: list, fsm, last: str = None):
    """Each output's statistics are for the GroupNorm that reads them; outputs no GroupNorm reads (before an
    upsampling) still give them, so stages of the same layout share a kind. The last stage's are for the
    GroupNorm `last` (the encoder's attention's), unless it's the decoder's OUT, which has none."""
    modules = dict(fsm.named_modules())
    for n, stage in enumerate(found):
        if stage.kind.op == OUT:
            continue
        reader = next((name for s in found[n + 1:] for name, key in s.norms if key == n), None)
        if reader is None:
            # the next GroupNorm (the same groups in these VAEs)
            reader = next((name for s in found[n + 1:] for name, _ in s.norms), last)
        stage.kind = replace(stage.kind, groups=int(modules[reader].num_groups))


ATTENTION_NORM = "encoder.mid.attn_1.norm"  # what reads the encoder's last stage's statistics


def encoder_blocks(fsm) -> list:
    """The encoder's resnet blocks up to its attention, in order: [(prefix, module, its output's scale, the
    prefix of the downsampling right after it or None)]. Raises ValueError for an encoder with attention
    before its mid block."""
    e = fsm.encoder
    scale, found = 2 ** (e.num_resolutions - 1), []
    for i in range(e.num_resolutions):
        if len(getattr(e.down[i], "attn", ())):
            raise ValueError("attention in the encoder's downsampling")
        last = len(e.down[i].block) - 1
        down = f"encoder.down.{i}.downsample." if i != e.num_resolutions - 1 else None
        for j, blk in enumerate(e.down[i].block):
            found.append((f"encoder.down.{i}.block.{j}.", blk, scale, down if j == last else None))
        if down is not None:
            scale //= 2
    found.append(("encoder.mid.block_1.", e.mid.block_1, 1, None))
    return found


def encoder_choices(fsm, padding: dict) -> list:
    """For each of the encoder's blocks, the ways it can run: STORE always; RECOMPUTE when its convs pad with
    zeros or wrap the same way."""
    pad = lambda name: padding.get(name, "zeros")  # noqa: E731
    out = []
    for prefix, blk, _, _ in encoder_blocks(fsm):
        pads = {pad(prefix + "conv1"), pad(prefix + "conv2")}
        chained = len(pads) == 1 and pads <= set(CHAINED) and not getattr(blk, "use_conv_shortcut", False)
        out.append((STORE, RECOMPUTE) if chained else (STORE,))
    return out


def encoder_stages(fsm, padding: dict, modes: dict = None) -> list:
    """The encoder's stages up to its attention, from the image (the buffer -1), for a first stage model
    (or what vae.fresh_encoder makes), the config's padding, and each block's way."""
    e = fsm.encoder
    modes = modes or {}
    pad = lambda name: padding.get(name, "zeros")  # noqa: E731
    found = []

    def add(op, cin, cout, convs, scale, src, block, res=None, res_channels=0, norms=(), keep=True, padded=None):
        weights = {}
        for local, module in convs.items():
            weights[f"{local}.weight"], weights[f"{local}.bias"] = f"{module}.weight", f"{module}.bias"
        mode = padded or pad(convs.get("conv") or convs["conv1"])
        found.append(Stage(Kind(op, cin, cout, res_channels, 0, mode), weights, scale, src, res, tuple(norms), mode, keep, block))
        return len(found) - 1

    top = 2 ** (e.num_resolutions - 1)
    x = add(IN, int(e.conv_in.in_channels), int(e.conv_in.out_channels), {"conv": "encoder.conv_in"}, top, -1, None)
    for b, (prefix, blk, scale, down) in enumerate(encoder_blocks(fsm)):
        cin, cout = int(blk.in_channels), int(blk.out_channels)
        if getattr(blk, "use_conv_shortcut", False) and cin != cout:
            raise ValueError("a resnet block with a 3x3 shortcut")
        tail = {"nin": prefix + "nin_shortcut"} if cin != cout else {}
        if modes.get(b, STORE) == STORE:
            h = add(CONV, cin, cout, {"conv": prefix + "conv1"}, scale, x, b, norms=((prefix + "norm1", x),))
            x = add(CONV, cout, cout, {"conv": prefix + "conv2", **tail}, scale, h, b, res=x, res_channels=cin, norms=((prefix + "norm2", h),))
        else:
            h = add(CONV, cin, cout, {"conv": prefix + "conv1"}, scale, x, b, norms=((prefix + "norm1", x),), keep=False)
            x = add(BLOCK, cin, cout, {"conv1": prefix + "conv1", "conv2": prefix + "conv2", **tail}, scale, x, b, norms=((prefix + "norm1", x), (prefix + "norm2", h)))
        if down is not None:  # its padding is its own (zeros, at the bottom and the right), whatever the convs'
            x = add(DOWN, cout, cout, {"conv": down + "conv"}, scale // 2, x, b, padded="zeros")
    _groups(found, fsm, ATTENTION_NORM)
    return found


def encoder_every_way(fsm, padding: dict) -> list:
    """Stage lists that together have every kind the encoder's blocks can run as."""
    lists = [encoder_stages(fsm, padding)]
    ways = encoder_choices(fsm, padding)
    if any(RECOMPUTE in w for w in ways):
        lists.append(encoder_stages(fsm, padding, {b: RECOMPUTE for b, w in enumerate(ways) if RECOMPUTE in w}))
    return lists


def every_way(fsm, padding: dict) -> list:
    """Stage lists that together have every kind the decoder's blocks can run as."""
    ways = choices(fsm, padding)
    lists = [stages(fsm, padding)]
    if any(RECOMPUTE in w for w in ways):
        lists.append(stages(fsm, padding, {b: RECOMPUTE for b, w in enumerate(ways) if RECOMPUTE in w}))
    if any(FUSED in w for w in ways):
        lists.append(stages(fsm, padding, {b: w[-1] for b, w in enumerate(ways)}))
    return lists


def kinds(found) -> dict:
    """{kind name: (Kind, [the scales of its stages' outputs])}, for a stage list or several."""
    lists = found if found and isinstance(found[0], list) else [found]
    out = {}
    for each in lists:
        for s in each:
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


def _normed(x, center, scale, shift, keep):
    """GroupNorm (in FP32, as PyTorch's), SiLU, and the rows outside the image zeroed (as a conv pads)."""
    return F.silu(((x.float() - center) * scale + shift).to(x.dtype)) * keep


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
        for local in sorted({n.rsplit(".", 1)[0] for n in stage.weights}):
            setattr(self, local, modules[stage.weights[f"{local}.weight"][: -len(".weight")]])

    def forward(self, *args):
        k = self.kind
        feed = dict(zip(k.inputs(), args))
        x = feed["x"]
        if k.op == IN:
            out = _conv3(x, self.conv, k.pad)
        elif k.op == DOWN:
            out = F.conv2d(F.pad(x, (0, 1)), self.conv.weight, self.conv.bias, stride=2)  # the bottom row: given, zeros
        elif k.op == PROJ:
            out = feed["res"] + F.conv2d(x, self.conv.weight, self.conv.bias)
        elif k.op in UPSAMPLING:
            up = F.interpolate(x, scale_factor=2.0, mode="nearest")
            if k.halo % 2:
                up = up[:, :, 1:-1]
            if k.op == UP:
                out = _conv3(up, self.conv, k.pad)
            else:
                u = _conv3(up, self.up, k.pad)
                h = _conv3(_normed(u, feed["center"], feed["scale"], feed["shift"], feed["keep"]), self.conv1, k.pad)
                if k.op == UPCONV:
                    return _stats(h, k.groups)
                out = _conv3(_normed(h, feed["center2"], feed["scale2"], feed["shift2"], feed["keep2"]), self.conv2, k.pad)
                core = u[:, :, 2:-2]
                out = out + (self.nin(core) if k.shortcut else core)
        elif k.op == BLOCK:
            h = _conv3(_normed(x, feed["center"], feed["scale"], feed["shift"], feed["keep"]), self.conv1, k.pad)
            out = _conv3(_normed(h, feed["center2"], feed["scale2"], feed["shift2"], feed["keep2"]), self.conv2, k.pad)
            core = x[:, :, 2:-2]
            out = out + (self.nin(core) if k.shortcut else core)
        else:
            out = _conv3(_normed(x, feed["center"], feed["scale"], feed["shift"], feed["keep"]), self.conv, k.pad)
            if "res" in feed:
                out = out + (self.nin(feed["res"]) if k.shortcut else feed["res"])
        return (out,) + (_stats(out, k.groups) if k.groups else ())


def feed_shapes(kind: Kind, rows: int, width: int) -> dict:
    """The inputs' shapes for `rows` rows of output `width` wide."""
    k = kind.halo
    shapes = {}
    for name in kind.inputs():
        if name == "x":
            if kind.op == DOWN:  # twice the rows and one more below, twice as wide
                shapes[name] = (1, kind.cin, 2 * rows + 1, 2 * width)
            elif kind.op in UPSAMPLING:
                shapes[name] = (1, kind.cin, rows // 2 + 2 * ((k + 1) // 2), width // 2)
            else:
                shapes[name] = (1, kind.cin, rows + 2 * k, width)
        elif name == "keep":  # the rows of what the first GroupNorm normalizes: x, or the upsampled conv's output
            shapes[name] = (1, 1, rows + 2 * (k - 1 if kind.op in (UPCONV, UPBLOCK) else k), 1)
        elif name == "keep2":
            shapes[name] = (1, 1, rows + 2, 1)
        elif name in ("center", "scale", "shift"):
            shapes[name] = (1, kind.cin, 1, 1)
        elif name in ("center2", "scale2", "shift2"):
            shapes[name] = (1, kind.cout, 1, 1)
        else:  # res
            shapes[name] = (1, kind.res or kind.cin, rows, width)
    return shapes


def example_inputs(kind: Kind, dtype, device, rows: int = 4, width: int = 8) -> tuple:
    """Inputs for tracing: `rows` of output, `width` of input."""
    shapes = feed_shapes(kind, rows, width * 2 if kind.op in UPSAMPLING else width // 2 if kind.op == DOWN else width)
    out = []
    for name in kind.inputs():
        if name.startswith(("center", "scale", "shift")):
            out.append(torch.randn(shapes[name], dtype=torch.float32, device=device))
        elif name.startswith("keep"):
            out.append(torch.ones(shapes[name], dtype=dtype, device=device))
        else:
            out.append(torch.randn(shapes[name], dtype=dtype, device=device))
    return tuple(out)


def dynamic_axes(kind: Kind) -> dict:
    axes = {"x": {2: "rows", 3: "width"}}
    if "out" in kind.outputs():
        axes["out"] = {2: "out_rows", 3: "out_width"}
    for name in ("keep", "keep2"):
        if name in kind.inputs():
            axes[name] = {2: f"{name}_rows"}
    if "res" in kind.inputs():
        axes["res"] = {2: "out_rows", 3: "out_width"}
    return axes


def min_rows(kind: Kind) -> int:
    """The fewest output rows a strip of this kind has: a step, and two for BLOCK (whose output may take
    over its input's memory a strip behind, so a strip must be at least its halo)."""
    return 2 if kind.op == BLOCK else kind.step


def max_rows(kind: Kind, width: int, itemsize: int = 4) -> int:
    """The most output rows a strip of this kind can have at this output width, by STRIP_BYTES in a
    precision of `itemsize` bytes."""
    channels = max(kind.cin, kind.cout, kind.res) * (4 if kind.op == DOWN else 1)  # DOWN's input: 4 times its output
    rows = max(1, STRIP_BYTES // itemsize // max(1, channels * width))
    return max(min_rows(kind), rows - rows % kind.step)


def kind_shapes(kind: Kind, scales: list, lo_w: int, opt: tuple, hi: tuple, itemsize: int = 4) -> dict:
    """The engine's profile ({input: [min, opt, max]}) for strips of its stages' outputs at these scales,
    for latents from lo_w wide to hi = (h, w), optimal at opt = (h, w): any number of rows up to a strip's
    (STRIP_BYTES at the widest, in a precision of `itemsize` bytes), across any width in that range."""
    hi_scale, lo_scale = max(scales), min(scales)
    lo_width, hi_width = lo_w * lo_scale, hi[1] * hi_scale
    hi_rows = max(min(hi[0] * s, max_rows(kind, hi[1] * s, itemsize)) for s in scales)
    opt_width = opt[1] * hi_scale
    opt_rows = min(opt[0] * hi_scale, max_rows(kind, opt_width, itemsize))
    lo_rows = min_rows(kind)
    hi_rows, opt_rows = max(hi_rows, lo_rows), max(min(opt_rows, hi_rows), lo_rows)
    hi_rows, opt_rows = hi_rows - hi_rows % kind.step, opt_rows - opt_rows % kind.step
    lo, mid, top = feed_shapes(kind, lo_rows, lo_width), feed_shapes(kind, opt_rows, opt_width), feed_shapes(kind, hi_rows, hi_width)
    return {name: [list(lo[name]), list(mid[name]), list(top[name])] for name in kind.inputs()}


def covers(shapes: dict, kind: Kind, rows: int, width: int) -> bool:
    """Whether an engine of these profile shapes runs strips of `rows` output rows `width` wide."""
    want = feed_shapes(kind, rows, width)
    return all(all(lo <= w <= hi for lo, w, hi in zip(shapes[n][0], want[n], shapes[n][2])) for n in kind.inputs())


def engine_rows(shapes: dict, kind: Kind, width: int) -> int:
    """The most output rows of a strip `width` wide the engine with these shapes runs (0: none)."""
    x_lo, _, x_hi = shapes["x"]
    up, down = kind.op in UPSAMPLING, kind.op == DOWN
    if not x_lo[3] <= (width // 2 if up else width * 2 if down else width) <= x_hi[3]:
        return 0
    k = kind.halo
    if down:
        rows = (x_hi[2] - 1) // 2
    else:
        rows = (x_hi[2] - 2 * ((k + 1) // 2)) * 2 if up else x_hi[2] - 2 * k
    if "res" in shapes:
        rows = min(rows, shapes["res"][2][2])
    rows -= rows % kind.step
    return rows if rows >= min_rows(kind) else 0


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

    def affine(self, norm, scaled: float = 1.0) -> tuple:
        """(center, scale, shift), each (1, channels, 1, 1) in FP32, that apply the GroupNorm `norm` (whose
        weight and bias are given in FP64) as (x - center) * scale + shift. `scaled`: what the tensor was
        multiplied by, so the epsilon keeps its size beside the variance."""
        weight, bias, eps, groups = norm
        per = weight.numel() // groups
        rstd = torch.rsqrt(self.m2 / self.n + eps * scaled * scaled)
        center = self.mean.repeat_interleave(per, dim=1)
        scale = weight.view(1, -1) * rstd.repeat_interleave(per, dim=1)
        shift = bias.view(1, -1).expand_as(scale)
        return tuple(t.float().reshape(1, -1, 1, 1).contiguous() for t in (center, scale, shift))


def norms(fsm, found: list, device) -> dict:
    """{GroupNorm module name: (weight, bias in FP64, eps, groups)} for the stages."""
    return norms_of(fsm, [name for s in found for name, _ in s.norms], device)


def norms_of(fsm, names, device) -> dict:
    """{GroupNorm module name: (weight, bias in FP64, eps, groups)} for these GroupNorms."""
    modules = dict(fsm.named_modules())
    out = {}
    for name in names:
        if name not in out:
            m = modules[name]
            out[name] = (m.weight.detach().to(device, torch.float64), m.bias.detach().to(device, torch.float64), float(m.eps), int(m.num_groups))
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
    For an upsampling, the input's rows whose upsampling gives the first conv its rows, padded as the
    upsampled image is."""
    k = stage.kind.halo
    if stage.kind.op == DOWN:  # rows 2a to 2b; past the bottom, the zeros it's padded with
        return [r if r < rows_in else None for r in range(2 * a, 2 * b + 1)]
    if stage.kind.op not in UPSAMPLING:
        return [_row(r, rows_in, stage.pad) for r in range(a - k, b + k)]
    lo, hi = a - k, b + k  # the upsampled rows the first conv reads
    out = []
    for r in range(lo // 2, (hi - 1) // 2 + 1):
        if 0 <= r < rows_in:
            out.append(r)
        elif k == 1:
            # One of r's two upsampled rows is read (the inner one): map it through the upsampled padding
            mapped = _row(2 * r + 1 if r < 0 else 2 * r, rows_in * 2, stage.pad)
            out.append(None if mapped is None else mapped // 2)
        else:  # zeros or wrapped (CHAINED): both of r's upsampled rows map alike
            out.append(None if stage.pad == "zeros" else r % rows_in)
    return out


def _keep_rows(stage: Stage, a: int, b: int, rows_out: int, second: bool = False) -> list:
    """Which rows of what a stage's GroupNorm (or second GroupNorm) normalizes are in the image: as a conv
    pads with zeros, those outside count as zeros (other paddings: as given or computed)."""
    k = stage.kind.halo
    if second:
        reach = 1
    elif stage.kind.op in (UPCONV, UPBLOCK):
        reach = k - 1
    else:
        reach = k
    return [stage.pad != "zeros" or 0 <= r < rows_out for r in range(a - reach, b + reach)]


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
    """Offsets in one slab for tensors allocated and freed in the order of `events` ([(key, what)]: True to
    allocate, False to free, or the key of a tensor that takes over the memory of `what`, which stays
    allocated until freed too): ({key: offset}, the slab's size, the key allocated when the most bytes
    were live). A tensor and those taking over its memory are placed together (at one offset, each as large
    as it is while it lives), largest first, at the lowest offset clear of the others alive at the same
    time: about as small as the most bytes alive at once."""
    members, group_of, groups, t = {}, {}, {}, 0
    for key, what in events:
        t += 1
        if what is False:
            members[key]["end"] = t
            continue
        members[key] = {"start": t, "end": None, "size": _align(sizes[key])}
        group_of[key] = key if what is True else group_of[what]
        groups.setdefault(group_of[key], []).append(key)
    for m in members.values():
        if m["end"] is None:
            m["end"] = t + 1
    # The most bytes alive at once (a group's members sharing memory count as its largest alive)
    most, at = -1, None
    for key, what in events:
        if what is False:
            continue
        when = members[key]["start"]
        live = 0
        for keys in groups.values():
            alive = [members[k]["size"] for k in keys if members[k]["start"] <= when < members[k]["end"]]
            live += max(alive, default=0)
        if live > most:
            most, at = live, key
    placed, offsets, size = [], {}, 0  # placed: (offset, member)
    for first, keys in sorted(groups.items(), key=lambda g: (-max(members[k]["size"] for k in g[1]), members[g[0]]["start"])):
        mine = [members[k] for k in keys]
        candidates = sorted({0} | {o + m["size"] for o, m in placed})
        for offset in candidates:
            if all(not (m["start"] < p["end"] and p["start"] < m["end"] and offset < o + p["size"] and o < offset + m["size"]) for m in mine for o, p in placed):
                break
        for k in keys:
            offsets[k] = offset
            placed.append((offset, members[k]))
            size = max(size, offset + members[k]["size"])
    return offsets, size, at


_pinned = {"tensor": None, "kind": None, "seconds": 0.0}


def _host_slab(nbytes: int):
    """RAM of at least nbytes for held tensors, pinned if it can be (registered with CUDA, not PyTorch's
    pinned allocator, which rounds up to a power of two), kept for the next decode (release_pinned)."""
    t = _pinned["tensor"]
    if t is not None and t.numel() >= nbytes:
        return t
    release_pinned()
    start = time.perf_counter()
    t, kind = torch.empty(max(nbytes, 1), dtype=torch.uint8), "pageable"
    try:
        if int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)) == 0:
            kind = "registered"
    except Exception:
        pass
    if kind == "pageable":
        try:
            t, kind = torch.empty(max(nbytes, 1), dtype=torch.uint8, pin_memory=True), "pinned"
        except RuntimeError:  # not enough RAM to pin: copies wait for the CPU
            pass
    _pinned.update(tensor=t, kind=kind, seconds=time.perf_counter() - start)
    return t


def release_pinned():
    t = _pinned["tensor"]
    if t is not None and _pinned["kind"] == "registered":
        try:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
        except Exception:
            pass
    _pinned.update(tensor=None, kind=None, seconds=0.0)


def host_memory() -> tuple:
    """(How the RAM for held tensors was had: "registered", "pinned" or "pageable"; its bytes; seconds)."""
    t = _pinned["tensor"]
    return _pinned["kind"], 0 if t is None else t.numel(), _pinned["seconds"]


class Held:
    """A tensor held between stages, rows first (rows, channels, width), in a slab in VRAM or in RAM."""

    def __init__(self, t, host: bool):
        self.t, self.host = t, host


@dataclass
class Plan:
    """How a stripped decode runs, decided before it starts so it never runs out of VRAM: each block's way,
    the stages and their runners that makes; each stage's rows per strip; which held tensors are in RAM;
    where each is in its slab (one in VRAM, one in pinned RAM); which stages' outputs take over their
    input's memory and how many strips behind; each stage's working tensors' bytes; the engines'
    activations at most."""

    modes: dict
    stages: list
    runners: list
    rows: list
    host: set
    offsets: dict
    device_bytes: int
    host_bytes: int
    in_place: dict
    io: list
    act: int
    moved: int = 0  # bytes copied between RAM and VRAM, counted as the decode runs
    times: dict = field(default_factory=dict)  # stage -> CUDA events around it, for the log
    acts: list = field(default_factory=list)  # each stage's activation memory as planned, for the diagnostics
    free: int = 0  # the VRAM free it was planned for, and the budget made of that (diagnostics)
    budget: int = 0

    @property
    def vram(self) -> int:
        return self.device_bytes + max(self.io) + self.act

    def describe(self) -> str:
        """For the log: the blocks' ways other than STORE."""
        words = {RECOMPUTE: "recomputing their first conv", FUSED: "fused with their upsampling"}
        counts = {mode: sum(1 for m in self.modes.values() if m == mode) for mode in words}
        return ", ".join(f"{n} block{'s' if n > 1 else ''} {words[mode]}" for mode, n in counts.items() if n)


class Decode:
    """One image's post part in strips, for a first stage model. `runner_for(stage)` gives what runs a
    stage: runner(feed, outputs, last) -> outputs (the strip's output written to outputs["out"]; `last`: the
    shorter last strip), runner.rows(width) -> its most output rows at that width (0: none),
    runner.activation(shapes) -> TensorRT's activation memory for those shapes, and runner.took() -> what
    it was when the stage last ran. `cache` keeps TensorRT's answers between decodes ({(kind, rows, width):
    bytes}): they're the engines' own, for as long as their plans are the same."""

    inputs = (-2, -1)  # the held tensors a run starts with: the mid block's input and its attention's output

    @staticmethod
    def _choices(fsm, padding: dict) -> list:
        return choices(fsm, padding)

    @staticmethod
    def _list(fsm, padding: dict, modes: dict = None) -> list:
        return stages(fsm, padding, modes)

    def __init__(self, fsm, padding: dict, runner_for, dtype, device, copy_stream=None, arena=None, cache=None, available=None, scale: float = 1.0):
        self.fsm, self.padding, self.runner_for = fsm, padding, runner_for
        self.dtype, self.device, self.copy_stream, self.arena = dtype, torch.device(device), copy_stream, arena
        self.scale = scale  # what the tensors between GroupNorms are multiplied by (the engines' biases too)
        self.size = torch.tensor([], dtype=dtype).element_size()
        self.cache = cache if cache is not None else {}
        self.ways = self._choices(fsm, padding)
        if available is not None:  # only the ways whose kinds have engines
            for b, ways in enumerate(self.ways):
                self.ways[b] = tuple(m for m in ways if all(s.kind.name in available for s in self._list(fsm, padding, {b: m}) if s.block == b))
        self._lists = {}
        self.norms = norms(fsm, self._list(fsm, padding), device)

    def _stages(self, modes: dict):
        key = tuple(sorted(modes.items()))
        if key not in self._lists:
            found = self._list(self.fsm, self.padding, modes)
            self._lists[key] = (found, [self.runner_for(s) for s in found])
        return self._lists[key]

    @staticmethod
    def last_use(found: list) -> dict:
        out = {}
        for n, st in enumerate(found):
            for key in (st.src, st.res):
                if key is not None:
                    out[key] = n
        return out

    @staticmethod
    def geometry(stage: Stage, h: int, w: int) -> tuple:
        """(output rows, output width, input rows) for latents of h x w."""
        return h * stage.scale, w * stage.scale, h * stage.in_scale

    def _shapes(self, found: list, h: int, w: int) -> dict:
        """{held tensor: its shape, rows first}."""
        first = found[0]
        if self.inputs == (-1,):  # the encoder's image
            shapes = {-1: (h * first.in_scale, first.kind.cin, w * first.in_scale)}
        else:
            shapes = {-2: (h, first.kind.cin, w), -1: (h, first.kind.cin, w)}
        for n, st in enumerate(found):
            if st.keep:
                rows, width, _ = self.geometry(st, h, w)
                shapes[n] = (rows, st.kind.cout, width)
        return shapes

    def _in_place(self, found: list, host: set) -> dict:
        """{stage: strips behind} for the outputs that take over their input's memory as they go: its last use,
        a row of the output no larger than one of it, held in the same place. Read without a halo (a
        residual), right away; with a halo of 2 (BLOCK's input), a strip behind, so the next strip has read
        the rows it overwrites."""
        last = self.last_use(found)
        out = {}
        for n, st in enumerate(found):
            if not st.keep:
                continue
            if st.res is not None and last[st.res] == n and st.kind.cout <= st.kind.res and (n in host) == (st.res in host):
                out[n] = 0
            elif st.kind.op == BLOCK and st.pad == "zeros" and last[st.src] == n and st.kind.cout <= st.kind.cin and (n in host) == (st.src in host):
                out[n] = 1  # (wrapped, the last strip reads the first rows, which the first strips overwrote)
        return out

    def _events(self, found: list, in_place: dict) -> list:
        last = self.last_use(found)
        events = [(key, True) for key in self.inputs]
        for n, st in enumerate(found):
            took = None
            if n in in_place:
                took = st.res if in_place[n] == 0 else st.src
            if st.keep:
                events.append((n, took if took is not None else True))
            events += [(key, False) for key in (st.src, st.res) if key is not None and last[key] == n]
        return events

    def _activation(self, runner, kind: Kind, rows: int, width: int) -> int:
        """The activation memory a strip of this kind takes: TensorRT's own figure for its shapes (asked once
        per shape, then kept)."""
        key = (kind.name, rows, width)
        if key not in self.cache:
            self.cache[key] = int(runner.activation(feed_shapes(kind, rows, width)))
        return self.cache[key]

    def _io(self, st: Stage, n: int, rows: int, width: int, host: set, lag: int) -> int:
        """Bytes of a stage's working tensors: its input, residual and output for a strip (the output twice
        when it's stored a strip behind; two strips' worth of staging for those held in RAM), and the rows
        to keep."""
        shapes = feed_shapes(st.kind, rows, width)
        x = math.prod(shapes["x"]) * self.size
        res = math.prod(shapes["res"]) * self.size if "res" in shapes else 0
        out = st.kind.cout * rows * width * self.size if "out" in st.kind.outputs() else 0
        keeps = sum(_align(shapes[k][2] * self.size) for k in ("keep", "keep2") if k in shapes)
        total = _align(x) + _align(res) + _align(out) * (2 if lag else 1) + keeps
        total += 2 * _align(x) if st.src in host else 0
        total += 2 * _align(res) if st.res is not None and st.res in host else 0
        return total + (2 * _align(out) if st.keep and n in host else 0)

    def _evaluate(self, modes: dict, div: int, host: set, h: int, w: int):
        found, runners = self._stages(modes)
        rows = []
        for n, st in enumerate(found):
            out_rows, width, _ = self.geometry(st, h, w)
            cap = min(out_rows, runners[n].rows(width))
            cap -= cap % st.kind.step
            if cap < min_rows(st.kind):
                return None
            rows.append(min(cap, max(min_rows(st.kind), cap // div // st.kind.step * st.kind.step)))
        widths = [self.geometry(st, h, w)[1] for st in found]
        acts = [self._activation(runners[n], st.kind, rows[n], widths[n]) for n, st in enumerate(found)]
        act = max(acts)
        in_place = self._in_place(found, host)
        io = [self._io(st, n, rows[n], widths[n], host, in_place.get(n, 0)) for n, st in enumerate(found)]
        shapes = self._shapes(found, h, w)
        sizes = {k: math.prod(v) * self.size for k, v in shapes.items()}
        events = self._events(found, in_place)
        offsets, peak, at = _layout([e for e in events if e[0] not in host], sizes)
        host_offsets, host_bytes, _ = _layout([e for e in events if e[0] in host], sizes)
        plan = Plan(dict(modes), found, runners, rows, set(host), {**offsets, **host_offsets}, peak, host_bytes, in_place, io, act, acts=acts)
        return plan, at, sizes

    def plan(self, h: int, w: int, free: int, force_host: bool = False, force_modes: dict = None):
        """_plan, with what it was planned for noted on it (diagnostics)."""
        plan = self._plan(h, w, free, force_host, force_modes)
        if plan is not None:
            plan.free = plan.budget = free
        return plan

    def memory_report(self, plan: Plan) -> str:
        """For the memory diagnostics: the plan's VRAM by part, and for each kind of stage, the activation
        memory the plan counted against what TensorRT took for it when it ran (the most of its stages)."""
        mib = lambda n: f"{n / 2**20:.0f} MiB"
        parts = [
            f"planned for {mib(plan.free)} free: {mib(plan.vram)} = held {mib(plan.device_bytes)} + working {mib(max(plan.io))} + activations {mib(plan.act)}",
            f"held in RAM {mib(plan.host_bytes)} ({len(plan.host)} tensors)",
        ]
        kinds = {}
        for n, st in enumerate(plan.stages):
            took = plan.runners[n].took() or 0
            counted = plan.acts[n] if n < len(plan.acts) else 0
            k = kinds.setdefault(st.kind.name, [0, 0, set()])
            k[0], k[1] = max(k[0], counted), max(k[1], took)
            k[2].add(plan.rows[n])
        top = sorted(kinds.items(), key=lambda kv: -kv[1][0])
        parts.append("activations counted -> took: " + ", ".join(
            f"{name} ({'/'.join(map(str, sorted(rows)))} rows) {mib(r)} -> {mib(t)}" for name, (r, t, rows) in top
        ))
        return " | ".join(parts)

    def _plan(self, h: int, w: int, free: int, force_host: bool = False, force_modes: dict = None):
        """The Plan for latents of h x w in `free` bytes of VRAM, or None if not even the smallest strips
        fit. The fastest that fits: every block STORE with strips as large as fit (down to an 8th of what the
        engines take); else, of the tensors held at the VRAM's peak, the largest one a block could do
        without (a first conv's output: that block RECOMPUTE; an upsampling's: the block after it FUSED)
        gone, and so on; else the tensors held at the peak moved to RAM, the largest first. Every size it
        weighs is exact (the held tensors', the working ones', TensorRT's activation memory), so `free` is
        all it plans for."""
        budget = free
        if force_host or force_modes is not None:
            modes = force_modes or {}
            found, _ = self._stages(modes)
            host = {n for n, st in enumerate(found) if st.keep} if force_host else set()
            result = self._evaluate(modes, 1, host, h, w)
            return result[0] if result else None
        modes = {b: ways[0] for b, ways in enumerate(self.ways)}
        while True:
            result = None
            for div in (1, 2, 4, 8):
                result = self._evaluate(modes, div, set(), h, w)
                if result is None:
                    return None
                if result[0].vram <= budget:
                    return result[0]
            plan, at, sizes = result
            better = []
            for key in self._live_at(plan, at):
                st = plan.stages[key] if key >= 0 else None
                if st is None or st.block is None or st.block >= len(self.ways):
                    continue
                ways, now = self.ways[st.block], modes[st.block]
                if st.kind.op == CONV and st.res is None and st.keep:
                    want = RECOMPUTE  # a first conv's output
                elif st.kind.op == UP and st.keep:
                    want = FUSED  # an upsampling's output
                else:
                    continue
                if want in ways and ways.index(want) > ways.index(now):
                    better.append((sizes[key], st.block, want))
            if not better:
                break
            _, block, want = max(better)
            modes[block] = want
        for div in (1, 2, 4, 8):
            host = set()
            while True:
                result = self._evaluate(modes, div, host, h, w)
                if result is None:
                    return None
                plan, at, sizes = result
                if plan.vram <= budget:
                    return plan
                movable = [k for k in self._live_at(plan, at) if k >= 0 and k not in host]
                if not movable:
                    break
                host.add(max(movable, key=sizes.get))
        return None

    def _live_at(self, plan: Plan, at) -> list:
        """The held tensors in VRAM alive when `at` was allocated."""
        if at is None:
            return []
        events = self._events(plan.stages, plan.in_place)
        live = set()
        for key, what in events:
            if key in plan.host:
                continue
            if what is False:
                live.discard(key)
            else:
                live.add(key)
            if key == at and what is not False:
                return list(live)
        return list(live)

    def run(self, x, attention, plan: Plan, timed: bool = False):
        """The image (1, out channels, H, W) for the mid block's input x and its attention's output (each
        (1, channels, h, w), in VRAM)."""

        def fill(held):
            held[-2].t.copy_(x[0].permute(1, 0, 2))  # scaled already, by the engine before
            held[-1].t.copy_(attention[0].permute(1, 0, 2))
            if self.scale != 1.0:
                held[-1].t.mul_(self.scale)

        held, _ = self._run(plan, x.shape[2], x.shape[3], fill, timed)
        image = held[len(plan.stages) - 1].t
        return image.to(self.device).permute(1, 0, 2).unsqueeze(0).contiguous()

    def _run(self, plan: Plan, h: int, w: int, fill, timed: bool = False) -> tuple:
        """Runs the plan's stages for latents of h x w, `fill(held)` having put the tensors it starts with in.
        Returns (the held tensors, the last stage's output's statistics)."""
        found = plan.stages
        shapes = self._shapes(found, h, w)
        slab = torch.empty(max(plan.device_bytes, 1), dtype=torch.uint8, device=self.device)
        ram = _host_slab(plan.host_bytes) if plan.host else None
        io = torch.empty(max(max(plan.io), 1), dtype=torch.uint8, device=self.device)
        if self.arena is not None:
            self.arena.get(plan.act)  # once, at its largest: every engine binds it as is
        held = {}
        for key, shape in shapes.items():
            base = ram if key in plan.host else slab
            offset, nbytes = plan.offsets[key], math.prod(shape) * self.size
            held[key] = Held(base[offset:offset + nbytes].view(self.dtype).view(shape), key in plan.host)
        fill(held)
        plan.moved, plan.times = 0, {}
        stats_last = {}
        for n, st in enumerate(found):
            for _, key in st.norms:
                stats_last[key] = n
        stats = {}
        for n, stage in enumerate(found):
            if timed and self.device.type == "cuda":
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
            stats[n] = self._sweep(n, stage, held, stats, plan, io)
            if timed and self.device.type == "cuda":
                events[1].record()
                plan.times[n] = events
            for _, key in stage.norms:
                if stats_last[key] == n:
                    stats.pop(key, None)
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        return held, stats.get(len(found) - 1)

    def _sweep(self, n: int, stage: Stage, held: dict, stats: dict, plan: Plan, io):
        kind, runner = stage.kind, plan.runners[n]
        src = held[stage.src]
        res = held[stage.res] if stage.res is not None else None
        out = held.get(n)
        out_rows, width = self._out_size(stage, src)
        rows = plan.rows[n]
        lag = plan.in_place.get(n, 0)
        shapes_r = feed_shapes(kind, rows, width)
        offset = [0]

        def take(elements):  # a flat view of the working slab
            nbytes = elements * self.size
            view = io[offset[0]:offset[0] + nbytes].view(self.dtype)
            offset[0] += _align(nbytes)
            return view

        x_flat = take(math.prod(shapes_r["x"]))
        res_flat = take(math.prod(shapes_r["res"])) if "res" in shapes_r else None
        has_out = "out" in kind.outputs()
        out_flats = [take(kind.cout * rows * width) for _ in range(2 if lag else 1)] if has_out else []
        keep_flats = {k: take(shapes_r[k][2]) for k in ("keep", "keep2") if k in shapes_r}
        x_stage = [take(math.prod(shapes_r["x"])) for _ in range(2)] if src.host else None
        res_stage = [take(math.prod(shapes_r["res"])) for _ in range(2)] if res is not None and res.host else None
        out_stage = [take(kind.cout * rows * width) for _ in range(2)] if out is not None and out.host else None
        fixed = {}
        for i, (name, key) in enumerate(stage.norms):
            suffix = "2" if i else ""
            center, scale, shift = stats[key].affine(self.norms[name], self.scale)
            fixed.update({f"center{suffix}": center, f"scale{suffix}": scale, f"shift{suffix}": shift})
        result_stats = Stats() if kind.groups else None
        per_group = kind.cout // kind.groups * width if kind.groups else 0
        strips = [(a, min(a + rows, out_rows)) for a in range(0, out_rows, rows)]
        copy = self.copy_stream if (x_stage or res_stage or out_stage) else None
        compute = torch.cuda.current_stream(self.device) if copy is not None else None
        if copy is not None:
            copy.wait_stream(compute)  # the working slab's last users (the stage before) are done
        loaded, consumed, stored = {}, [None, None], [None, None]
        rows_in = src.t.shape[0]

        def load(k):  # what's in RAM, into staging on the copy stream (a strip ahead)
            a, b = strips[k]
            slot = k % 2
            parts = []
            if x_stage:
                parts.append(("x", src, source_rows(stage, a, b, rows_in), x_stage[slot]))
            if res_stage:
                parts.append(("res", res, list(range(a, b)), res_stage[slot]))
            with (torch.cuda.stream(copy) if copy is not None else _nothing()):
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

        def store(k, a, b, result_out):
            rows_first = result_out[0].permute(1, 0, 2)
            if out_stage is None:
                out.t[a:b].copy_(rows_first)
                return
            slot = k % 2
            staging = out_stage[slot][:(b - a) * kind.cout * width].view(b - a, kind.cout, width)
            if compute is not None and stored[slot] is not None:
                compute.wait_event(stored[slot])  # its last copy out is done
            staging.copy_(rows_first)
            plan.moved += staging.numel() * self.size
            if copy is None:
                out.t[a:b].copy_(staging)
                return
            ready = torch.cuda.Event()
            ready.record(compute)
            with torch.cuda.stream(copy):
                copy.wait_event(ready)
                out.t[a:b].copy_(staging, non_blocking=True)
                stored[slot] = torch.cuda.Event()
                stored[slot].record(copy)

        if x_stage or res_stage:
            load(0)
        pending = None  # a strip's output to store once the next strip has read its input (in place, a strip behind)
        for k, (a, b) in enumerate(strips):
            r = b - a
            sh = feed_shapes(kind, r, width)
            want = source_rows(stage, a, b, rows_in)
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
            if pending is not None:
                store(*pending)
                pending = None
            for name, flat in keep_flats.items():
                keep_list = _keep_rows(stage, a, b, out_rows, second=name == "keep2")
                keep = flat[:len(keep_list)].view(1, 1, -1, 1)
                keep.fill_(1.0 if kind.op == OUT else self.scale)  # the image itself isn't scaled
                for i, inside in enumerate(keep_list):
                    if not inside:
                        keep[:, :, i].zero_()
                feed[name] = keep
            feed.update(fixed)
            outputs = {}
            if has_out:
                outputs["out"] = out_flats[k % len(out_flats)][:kind.cout * r * width].view(1, kind.cout, r, width)
            result = runner(feed, outputs, r != rows)
            if result_stats is not None:
                result_stats.add(per_group * r, result["mean"], result["m2"])
            if out is None:
                continue
            if lag:
                pending = (k, a, b, outputs["out"])
            else:
                store(k, a, b, outputs["out"])
        if pending is not None:
            store(*pending)
        if copy is not None:
            compute.wait_stream(copy)
        return result_stats

    def _out_size(self, stage: Stage, src: Held) -> tuple:
        """(output rows, output width) from the input's size."""
        rows, width = src.t.shape[0], src.t.shape[2]
        if stage.kind.op == DOWN:
            return rows // 2, width // 2
        return (rows * 2, width * 2) if stage.kind.op in UPSAMPLING else (rows, width)


class Encode(Decode):
    """The encoder up to its mid block's attention, in strips: as Decode, from an image."""

    inputs = (-1,)  # the image

    @staticmethod
    def _choices(fsm, padding: dict) -> list:
        return encoder_choices(fsm, padding)

    @staticmethod
    def _list(fsm, padding: dict, modes: dict = None) -> list:
        return encoder_stages(fsm, padding, modes)

    def run(self, image, plan: Plan, timed: bool = False) -> tuple:
        """(The mid block's first resnet block's output (1, channels, h, w) in VRAM, scaled as the encoder runs
        scaled; its statistics, for the attention's GroupNorm) for an image (1, channels, H, W) in VRAM, as
        the encoder takes it (-1..1)."""
        first = plan.stages[0]
        h, w = image.shape[2] // first.in_scale, image.shape[3] // first.in_scale

        def fill(held):
            held[-1].t.copy_(image[0].permute(1, 0, 2))
            if self.scale != 1.0:
                held[-1].t.mul_(self.scale)  # as the scaled encoder takes it (vae._EncodePre)

        held, stats = self._run(plan, h, w, fill, timed)
        x = held[len(plan.stages) - 1].t
        return x.to(self.device).permute(1, 0, 2).unsqueeze(0).contiguous(), stats


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TorchRunner:
    """A stage on PyTorch (for checks, and where no engine runs it)."""

    engines = ()  # nothing to load (see EngineRunner)

    def __init__(self, module: StageModule):
        self.module = module

    def __call__(self, feed: dict, outputs: dict, last: bool = False) -> dict:
        kind = self.module.kind
        with torch.no_grad():
            result = dict(zip(kind.outputs(), self.module(*[feed[n] for n in kind.inputs()])))
        if "out" in result and "out" in outputs:
            outputs["out"].copy_(result["out"])
            result["out"] = outputs["out"]
        return result

    def activation(self, shapes: dict) -> int:
        return 0

    def took(self):
        return None

    def rows(self, width: int) -> int:
        return max_rows(self.module.kind, width)


class EngineRunner:
    """A stage on its kind's engine (engine.py's Engine), loaded with the stage's weights when it first
    runs (`load()` -> (its context for strips, its context for the shorter last one), so neither changes
    shapes from one decode of a size to the next); activation sizes from the kind's probe context."""

    def __init__(self, load, probe, shapes: dict, kind: Kind):
        self.load, self.probe, self.shapes, self.kind = load, probe, shapes, kind
        self.engines = None

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def __call__(self, feed: dict, outputs: dict, last: bool = False) -> dict:
        if self.engines is None:
            self.engines = self.load()
        return self.engines[1 if last else 0].infer(feed, 0, self._stream(), outputs)

    def activation(self, shapes: dict) -> int:
        return self.probe().estimate_activation_bytes(0, {n: tuple(s) for n, s in shapes.items()}, self._stream())

    def took(self):
        """The activation memory its strips took when it last ran (None before)."""
        return self.engines[0].size if self.engines is not None else None

    def rows(self, width: int) -> int:
        return engine_rows(self.shapes, self.kind, width)
