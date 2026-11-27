# Traceable ONNX forward wrappers, exact pure-PyTorch FP32 norm/RoPE/SiTU ops, and cached runtime RoPE helpers
# for Forge Neo DiT architectures (Anima and MOSS-Image).
import math

import torch
import torch.nn.functional as F


def _to_dtype(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if y.dtype == dtype:
        return y
    if dtype == torch.float16:
        # Clamp finite values beyond FP16 range without masking NaN (CUDA fminf(NaN, 65504) would hide NaN as 65504).
        y = torch.where(y.abs() > 65504.0, torch.sign(y) * 65504.0, y)
    return y.to(dtype)


def _rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, out_dtype: torch.dtype = None) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return _to_dtype(y, x.dtype if out_dtype is None else out_dtype)


def _adaln_layer_norm_fp32(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    y = x.float()
    mean = y.mean(dim=-1, keepdim=True)
    diff = y - mean
    var = (diff * diff).mean(dim=-1, keepdim=True)
    y = diff * torch.rsqrt(var + eps)
    y = y * (1.0 + scale.float()) + shift.float()
    return _to_dtype(y, x.dtype if out_dtype is None else out_dtype)


def _modulated_rms_norm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    # Matches moss_image.py modulated_rms_norm: ck.rms_adaln(x, weight * (1 + scale) - 1, shift, eps)
    # == rms_norm(x) * (weight * (1 + scale)) + shift.
    y = x.float()
    y = y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + eps)
    y = y * (weight.float() * (1.0 + scale.float())) + shift.float()
    return _to_dtype(y, x.dtype if out_dtype is None else out_dtype)


def _apply_rope_split_half(x: torch.Tensor, rope_cs: torch.Tensor, out_dtype: torch.dtype = None) -> torch.Tensor:
    # Matches comfy_kitchen.backends.eager.rope.apply_rope_split_half1:
    # x: (B, L, H, D), rope_cs: (1, L, 1, D//2, 2) FP32 where [..., 0]=cos, [..., 1]=sin.
    target = x.dtype if out_dtype is None else out_dtype
    xf = x if x.dtype == torch.float32 else x.float()
    half = xf.shape[-1] // 2
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    cos = rope_cs[..., 0]
    sin = rope_cs[..., 1]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    out = torch.cat([out1, out2], dim=-1)
    return _to_dtype(out, target)


def _situ_fp32(hidden: torch.Tensor, gate: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    # Matches operations_mixed_precision._situ_eager and moss_image.situ:
    # (4.0 * tanh(gate / 4.0) * sigmoid(gate)) * (25.0 * tanh(hidden / 25.0)).
    h = hidden.float()
    g = gate.float()
    out = (4.0 * torch.tanh(g * 0.25) * torch.sigmoid(g)) * (25.0 * torch.tanh(h * 0.04))
    return _to_dtype(out, out_dtype)


def _attn_bhsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_fn=None) -> torch.Tensor:
    # q, k, v: (B, H, L, D) -> returns (B, L, H * D).
    b, h, _, d = q.shape
    if attn_fn is not None:
        if v.dtype == torch.float16:
            # TensorRT 11's FP16 FlashAttention (_gemm_mha_v2) overflows its unnormalized FP16 P@V accumulator on long
            # DiT sequences (L>=4096) when |v| is large, and algebraically hoists static V scales past MHA.
            # Per-head dynamic amax scaling keeps |v_s| <= 2.0 inside _gemm_mha_v2 (0 NaN/Inf, fast MHA, ~64 MB ws).
            v_scale = (v.float().abs().amax(dim=(-2, -1), keepdim=True) * 0.5).clamp_min(1.0)
            v_scaled = (v.float() / v_scale).to(v.dtype)
            out = attn_fn(q, k, v_scaled, h, skip_reshape=True, skip_output_reshape=True)
            out = _to_dtype(out.float() * v_scale, v.dtype)
            return out.transpose(1, 2).reshape(b, -1, h * d)
        return attn_fn(q, k, v, h, skip_reshape=True)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, -1, h * d)


def _anima_embed(module, x: torch.Tensor, timesteps: torch.Tensor, rope: torch.Tensor, compute_dtype: torch.dtype):
    b, c, t, h_lat, w_lat = x.shape
    ps, pt = module.patch_spatial, module.patch_temporal
    concat_mask = module.x_embedder.dim > module.in_channels * ps * ps * pt
    if concat_mask:
        x_in = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        c_in = c + 1
    else:
        x_in = x
        c_in = c
    h_p, w_p = rope.shape[1], rope.shape[2]
    # Exact dynamic equivalent of Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)", r=pt, m=ps, n=ps)
    patches = (
        x_in.reshape(b, c_in, t // pt, pt, h_p, ps, w_p, ps)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(b, -1, c_in * pt * ps * ps)
    )
    x_seq = module.x_embedder.proj[1](patches)

    t_flat = timesteps.reshape(-1).float()
    half_dim = module.t_embedder[0].num_channels // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=x.device) / float(half_dim)
    freqs = torch.exp(exponent)
    args = t_flat[:, None] * freqs[None, :]
    t_sin = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).reshape(b, 1, -1).to(compute_dtype)
    t_emb, adaln_lora = module.t_embedder[1](t_sin)
    t_emb = _rms_norm_fp32(t_emb, module.t_embedding_norm.weight, module.t_embedding_norm.eps, out_dtype=compute_dtype)
    rope_seq = rope.reshape(1, -1, 1, rope.shape[-2], 2)
    return x_seq, t_emb, adaln_lora, rope_seq, h_p, w_p


def _anima_block(block, x_res: torch.Tensor, t_emb: torch.Tensor, adaln_lora: torch.Tensor, context: torch.Tensor, rope_seq: torch.Tensor, compute_dtype: torch.dtype, attn_fn=None) -> torch.Tensor:
    b = x_res.shape[0]
    res_dtype = x_res.dtype
    shift_sa, scale_sa, gate_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, dim=-1)
    shift_ca, scale_ca, gate_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, dim=-1)
    shift_ml, scale_ml, gate_ml = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, dim=-1)

    sa = block.self_attn
    h = _adaln_layer_norm_fp32(x_res, scale_sa, shift_sa, 1e-6, compute_dtype)
    q = _apply_rope_split_half(
        _rms_norm_fp32(sa.q_proj(h).reshape(b, -1, sa.n_heads, sa.head_dim), sa.q_norm.weight, sa.q_norm.eps, out_dtype=torch.float32),
        rope_seq,
        out_dtype=compute_dtype,
    ).transpose(1, 2)
    k = _apply_rope_split_half(
        _rms_norm_fp32(sa.k_proj(h).reshape(b, -1, sa.n_heads, sa.head_dim), sa.k_norm.weight, sa.k_norm.eps, out_dtype=torch.float32),
        rope_seq,
        out_dtype=compute_dtype,
    ).transpose(1, 2)
    v = sa.v_proj(h).reshape(b, -1, sa.n_heads, sa.head_dim).transpose(1, 2)
    res_sa = sa.output_proj(_attn_bhsd(q, k, v, attn_fn))
    x_res = x_res + gate_sa.to(res_dtype) * res_sa.to(res_dtype)

    ca = block.cross_attn
    h = _adaln_layer_norm_fp32(x_res, scale_ca, shift_ca, 1e-6, compute_dtype)
    q = _rms_norm_fp32(ca.q_proj(h).reshape(b, -1, ca.n_heads, ca.head_dim), ca.q_norm.weight, ca.q_norm.eps, out_dtype=compute_dtype).transpose(1, 2)
    k = _rms_norm_fp32(ca.k_proj(context).reshape(b, -1, ca.n_heads, ca.head_dim), ca.k_norm.weight, ca.k_norm.eps, out_dtype=compute_dtype).transpose(1, 2)
    v = ca.v_proj(context).reshape(b, -1, ca.n_heads, ca.head_dim).transpose(1, 2)
    res_ca = ca.output_proj(_attn_bhsd(q, k, v, attn_fn))
    x_res = x_res + gate_ca.to(res_dtype) * res_ca.to(res_dtype)

    h = _adaln_layer_norm_fp32(x_res, scale_ml, shift_ml, 1e-6, compute_dtype)
    res_ml = block.mlp(h)
    return x_res + gate_ml.to(res_dtype) * res_ml.to(res_dtype)


def _anima_final(module, x_res: torch.Tensor, t_emb: torch.Tensor, adaln_lora: torch.Tensor, h_p, w_p, compute_dtype: torch.dtype) -> torch.Tensor:
    b = x_res.shape[0]
    fl = module.final_layer
    shift_f, scale_f = (fl.adaln_modulation(t_emb) + adaln_lora[:, :, : 2 * fl.hidden_size]).chunk(2, dim=-1)
    h = _adaln_layer_norm_fp32(x_res, scale_f, shift_f, 1e-6, compute_dtype)
    out = fl.linear(h)
    ps, pt, out_c = module.patch_spatial, module.patch_temporal, module.out_channels
    return (
        out.reshape(b, 1, h_p, w_p, ps, ps, pt, out_c)
        .permute(0, 7, 1, 6, 2, 4, 3, 5)
        .reshape(b, out_c, pt, h_p * ps, w_p * ps)
    )


def anima_traceable_forward(
    module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    rope: torch.Tensor,
    *,
    fp32_residual: bool = False,
    attn_fn=None,
):
    compute_dtype = x.dtype
    x_seq, t_emb, adaln_lora, rope_seq, h_p, w_p = _anima_embed(module, x, timesteps, rope, compute_dtype)
    x_res = x_seq.float() if (fp32_residual or compute_dtype == torch.float16) else x_seq
    for block in module.blocks:
        x_res = _anima_block(block, x_res, t_emb, adaln_lora, context, rope_seq, compute_dtype, attn_fn=attn_fn)
    return _anima_final(module, x_res, t_emb, adaln_lora, h_p, w_p, compute_dtype)


def _moss_embed(module, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, rope_img: torch.Tensor, rope_txt: torch.Tensor, fp32_residual: bool, attn_fn=None):
    b, c, h, w = x.shape
    compute_dtype = x.dtype
    res_dtype = torch.float32 if (fp32_residual or compute_dtype == torch.float16) else compute_dtype

    t_scaled = 1000.0 * timesteps.reshape(-1).float()
    half = 128
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=x.device) / float(half))
    args = t_scaled[:, None] * freqs[None, :]
    t_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(compute_dtype)

    vec = module.time_embedding(t_emb)
    mod = module.shared_block_modulation(vec).reshape(b, 6, 1, -1).unbind(1)
    head_shift, head_scale = module.video_head_modulation(vec).reshape(b, 2, 1, -1).unbind(1)

    head_dim = rope_img.shape[-2] * 2
    txt = module.text_embedder.proj(context)
    txt_res = txt.to(res_dtype)
    for tb in module.text_embedder.blocks:
        h1 = _rms_norm_fp32(txt_res, tb.norm1.weight, tb.norm1.eps, out_dtype=compute_dtype)
        q, k, v = tb.qkv(h1).chunk(3, dim=-1)
        q = _apply_rope_split_half(
            _rms_norm_fp32(q.reshape(b, -1, tb.num_heads, head_dim), tb.q_norm.weight, tb.q_norm.eps, out_dtype=torch.float32),
            rope_txt,
            out_dtype=compute_dtype,
        ).transpose(1, 2)
        k = _apply_rope_split_half(
            _rms_norm_fp32(k.reshape(b, -1, tb.num_heads, head_dim), tb.k_norm.weight, tb.k_norm.eps, out_dtype=torch.float32),
            rope_txt,
            out_dtype=compute_dtype,
        ).transpose(1, 2)
        v = v.reshape(b, -1, tb.num_heads, head_dim).transpose(1, 2)
        txt_res = txt_res + tb.proj(_attn_bhsd(q, k, v, attn_fn)).to(res_dtype)
        h2 = _rms_norm_fp32(txt_res, tb.norm2.weight, tb.norm2.eps, out_dtype=compute_dtype)
        ffn_out = tb.ffn_down(_situ_fp32(tb.ffn_up(h2), tb.ffn_gate(h2), compute_dtype))
        txt_res = txt_res + ffn_out.to(res_dtype)

    vid = module.video_embedder(x.reshape(b, c, -1).transpose(1, 2))
    x_res = torch.cat([vid.to(res_dtype), txt_res], dim=1)
    rope_all = torch.cat([rope_img.reshape(1, -1, 1, rope_img.shape[-2], 2), rope_txt], dim=1)
    return x_res, mod, head_shift, head_scale, rope_all, head_dim, h, w


def _moss_block(block, x_res: torch.Tensor, mod: tuple, rope_all: torch.Tensor, head_dim: int, compute_dtype: torch.dtype, attn_fn=None) -> torch.Tensor:
    b = x_res.shape[0]
    res_dtype = x_res.dtype
    y = _modulated_rms_norm_fp32(x_res, block.norm1.weight, mod[1], mod[0], block.norm1.eps, compute_dtype)
    if block.fused_qkv:
        q, k, v = block.proj["qkv"](y).chunk(3, dim=-1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
    q = _apply_rope_split_half(
        _rms_norm_fp32(q.reshape(b, -1, block.num_heads, head_dim), block.q_norm.weight, block.q_norm.eps, out_dtype=torch.float32),
        rope_all,
        out_dtype=compute_dtype,
    ).transpose(1, 2)
    k = _apply_rope_split_half(
        _rms_norm_fp32(k.reshape(b, -1, block.num_heads, head_dim), block.k_norm.weight, block.k_norm.eps, out_dtype=torch.float32),
        rope_all,
        out_dtype=compute_dtype,
    ).transpose(1, 2)
    v = v.reshape(b, -1, block.num_heads, head_dim).transpose(1, 2)
    attn_out = _attn_bhsd(q, k, v, attn_fn)
    x_res = x_res + block.proj["o"](attn_out).to(res_dtype) * mod[2].to(res_dtype)

    y2 = _modulated_rms_norm_fp32(x_res, block.norm2.weight, mod[4], mod[3], block.norm2.eps, compute_dtype)
    hidden, gate = block.proj["ffn_up_gate"](y2).chunk(2, dim=-1)
    act = _situ_fp32(hidden, gate, compute_dtype)
    x_res = x_res + block.proj["ffn_down"](act).to(res_dtype) * mod[5].to(res_dtype)
    if res_dtype == torch.float16:
        x_res = x_res.clamp(-65504.0, 65504.0)
    return x_res


def _moss_final(module, x_vid: torch.Tensor, head_scale: torch.Tensor, head_shift: torch.Tensor, b, h, w, compute_dtype: torch.dtype) -> torch.Tensor:
    hf = _modulated_rms_norm_fp32(x_vid, module.final_norm_video.weight, head_scale, head_shift, module.final_norm_video.eps, compute_dtype)
    out = module.final_linear_video(hf)
    return out.transpose(1, 2).reshape(b, module.latent_channels, h, w)


def moss_traceable_forward(
    module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    rope_img: torch.Tensor,
    rope_txt: torch.Tensor,
    *,
    fp32_residual: bool = False,
    attn_fn=None,
):
    b = x.shape[0]
    compute_dtype = x.dtype
    x_res, mod, head_shift, head_scale, rope_all, head_dim, h, w = _moss_embed(
        module, x, timesteps, context, rope_img, rope_txt, fp32_residual, attn_fn=attn_fn
    )
    for block in module.layers:
        x_res = _moss_block(block, x_res, mod, rope_all, head_dim, compute_dtype, attn_fn=attn_fn)
    return _moss_final(module, x_res[:, : h * w], head_scale, head_shift, b, h, w, compute_dtype)


def anima_rope(module, h_lat: int, w_lat: int, device, cache: dict = None) -> torch.Tensor:
    ps = module.patch_spatial
    h_p, w_p = h_lat // ps, w_lat // ps
    key = ("anima", h_p, w_p, str(device))
    if cache is not None and key in cache:
        return cache[key]
    dummy = torch.empty((1, 1, h_p, w_p, 1), device=device, dtype=torch.float32)
    pe = module.pos_embedder(dummy, device=device)  # (h_p * w_p, head_dim//2, 2, 2) FP32
    rope = torch.stack([pe[..., 0, 0], pe[..., 1, 0]], dim=-1).reshape(1, h_p, w_p, 1, -1, 2).contiguous()
    if cache is not None:
        if len(cache) >= 32:
            cache.clear()
        cache[key] = rope
    return rope


def moss_rope(module, h: int, w: int, text_len: int, device, cache: dict = None) -> tuple:
    key = ("moss", h, w, text_len, str(device))
    if cache is not None and key in cache:
        return cache[key]
    pe = module.rope(h, w, text_len, device=device)  # (1, h*w + text_len, 1, head_dim//2, 2, 2) FP32
    cs = torch.stack([pe[..., 0, 0], pe[..., 1, 0]], dim=-1)  # (1, h*w + text_len, 1, head_dim//2, 2)
    n_img = h * w
    rope_img = cs[:, :n_img].reshape(1, h, w, 1, -1, 2).contiguous()
    rope_txt = cs[:, n_img:].contiguous()
    out = (rope_img, rope_txt)
    if cache is not None:
        if len(cache) >= 32:
            cache.clear()
        cache[key] = out
    return out
