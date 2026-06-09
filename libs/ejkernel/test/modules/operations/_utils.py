from __future__ import annotations

import importlib.util
from typing import Any

import jax
import jax.numpy as jnp


def device_platform() -> str:
    return jax.devices()[0].platform


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def has_cutlass() -> bool:
    return importlib.util.find_spec("cutlass") is not None


def rand_qkv(
    key: jax.Array,
    *,
    batch: int,
    q_len: int,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    v_head_dim: int | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if v_head_dim is None:
        v_head_dim = head_dim
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (batch, q_len, q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (batch, kv_len, kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (batch, kv_len, kv_heads, v_head_dim), dtype=jnp.float32).astype(dtype)
    return q, k, v


def _normalize_softmax_aux(
    softmax_aux: jax.Array | None,
    *,
    num_heads: int,
    dtype: jnp.dtype,
) -> jax.Array | None:
    if softmax_aux is None:
        return None

    aux = jnp.asarray(softmax_aux, dtype=dtype)
    if aux.ndim == 1:
        # Heuristic: if length matches head count, treat as per-head single sink logit.
        if aux.shape[0] == num_heads:
            return aux[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))
    if aux.ndim == 2:
        if aux.shape[0] == 1:
            return jnp.broadcast_to(aux, (num_heads, aux.shape[1]))
        if aux.shape[0] != num_heads:
            raise ValueError(f"softmax_aux first dim must be 1 or num_heads ({num_heads}); got {aux.shape[0]}")
        return aux
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


def dense_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    bias: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Dense attention reference for [B,Q,H,D] x [B,K,Hkv,D] (+GQA).

    Returns:
      - out: [B,Q,H,Dv]
      - weights: [B,H,Q,K]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 tensors [B, T, H, D]")

    b, q_len, q_heads, head_dim = q.shape
    bk, kv_len, kv_heads, k_dim = k.shape
    bv, vk_len, v_heads, _v_dim = v.shape
    if (bk, bv) != (b, b) or (vk_len, v_heads) != (kv_len, kv_heads) or k_dim != head_dim:
        raise ValueError(f"shape mismatch: q={q.shape} k={k.shape} v={v.shape}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})")

    reps = q_heads // kv_heads
    k_rep = jnp.repeat(k.astype(jnp.float32), reps, axis=2)
    v_rep = jnp.repeat(v.astype(jnp.float32), reps, axis=2)
    q_f32 = q.astype(jnp.float32)

    scale = float(softmax_scale) if softmax_scale is not None else (head_dim**-0.5)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q_f32, k_rep, optimize=True) * jnp.asarray(scale, jnp.float32)
    allow = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)

    if logits_soft_cap is not None:
        cap = jnp.asarray(float(logits_soft_cap), dtype=jnp.float32)
        logits = cap * jnp.tanh(logits / cap)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            left, right = int(sliding_window), int(sliding_window)
        else:
            left, right = int(sliding_window[0]), int(sliding_window[1])
        q_pos = jnp.arange(q_len)[:, None]
        k_pos = jnp.arange(kv_len)[None, :]
        window = (k_pos >= (q_pos - left)) & (k_pos <= (q_pos + right))
        allow = allow & window[None, :, :]
        logits = jnp.where(window[None, None, :, :], logits, jnp.finfo(jnp.float32).min)

    if causal:
        causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=jnp.bool_))
        allow = allow & causal_mask[None, :, :]
        logits = jnp.where(causal_mask[None, None, :, :], logits, jnp.finfo(jnp.float32).min)

    if bias is not None:
        bias_f32 = jnp.asarray(bias, dtype=jnp.float32)
        if bias_f32.shape != (b, q_heads, q_len, kv_len):
            raise ValueError(f"bias must have shape {(b, q_heads, q_len, kv_len)}; got {bias_f32.shape}")
        logits = logits + bias_f32

    if attention_mask is not None:
        m = attention_mask
        if m.dtype != jnp.bool_:
            m = m != 0
        if m.ndim == 4:
            if m.shape[:1] != (b,) or m.shape[2:] != (q_len, kv_len):
                raise ValueError(f"attention_mask must have shape (B, 1/H, Q, K); got {m.shape}")
            m = m[:, 0, :, :]
        elif m.ndim == 3:
            if m.shape != (b, q_len, kv_len):
                raise ValueError(f"attention_mask must have shape (B, Q, K); got {m.shape}")
        else:
            raise ValueError(f"unsupported attention_mask rank {m.ndim} with shape {m.shape}")
        allow = allow & m
        logits = jnp.where(m[:, None, :, :], logits, jnp.finfo(jnp.float32).min)

    aux = _normalize_softmax_aux(softmax_aux, num_heads=q_heads, dtype=jnp.float32)
    if aux is not None:
        sinks = jnp.broadcast_to(aux[None, :, None, :], (b, q_heads, q_len, aux.shape[-1]))
        combined = jnp.concatenate([logits, sinks], axis=-1)
        probs = jax.nn.softmax(combined, axis=-1)
        weights = probs[..., :kv_len]
    else:
        weights = jax.nn.softmax(logits, axis=-1)

    row_has_any = jnp.any(allow, axis=-1)  # [B, Q]
    weights = weights * row_has_any[:, None, :, None].astype(weights.dtype)

    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v_rep, optimize=True)
    return out.astype(q.dtype), weights.astype(jnp.float32)


def assert_allclose(
    a: Any,
    b: Any,
    *,
    atol: float,
    rtol: float = 0.0,
    err_msg: str | None = None,
):
    a32 = jnp.asarray(a, dtype=jnp.float32)
    b32 = jnp.asarray(b, dtype=jnp.float32)
    if not jnp.allclose(a32, b32, atol=atol, rtol=rtol):
        diff = jnp.max(jnp.abs(a32 - b32))
        msg = f"not allclose: max(|a-b|)={float(diff)} atol={atol} rtol={rtol}"
        if err_msg:
            msg = f"{err_msg}: {msg}"
        raise AssertionError(msg)
