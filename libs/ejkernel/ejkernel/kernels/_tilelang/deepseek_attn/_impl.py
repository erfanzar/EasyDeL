# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX glue for native tile-lang DeepSeek Sparse Attention (forward + backward).

The DSA pipeline is: native Lightning-Indexer kernel -> ``top_k`` selection
-> additive mask -> differentiable KV up-projection (native GEMM kernel,
``jax.custom_vjp``) -> FlashAttention with the mask folded in as a bias.

Because the KV reconstruction is a ``custom_vjp`` over native GEMM kernels
and FlashAttention is itself a ``custom_vjp`` over native kernels, the whole
thing is differentiable w.r.t. ``query`` / ``key_value`` / ``w_kc`` /
``w_vc`` end-to-end — the gradient never falls back to ``jnp.einsum``. The
indexer / top-k-bias path is integer selection and is correctly stop-gradient
(matching the XLA reference, which also does not differentiate the indexer).
"""

from __future__ import annotations

import math
import threading
from functools import partial

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ..flash_attention._impl import flash_attention_tilelang
from ._kernel import (
    make_add_prim_func,
    make_cast_prim_func,
    make_crop_lastdim_prim_func,
    make_dsa_indexer_prim_func,
    make_matmul_prim_func,
    make_pack_shared_tail_prim_func,
    make_pad_lastdim_prim_func,
    make_reduce_shared_tail_prim_func,
    make_topk_bias_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_INDEXER_CACHE: dict[tuple, callable] = {}
_MATMUL_CACHE: dict[tuple, callable] = {}
_ADD_CACHE: dict[tuple, callable] = {}
_CAST_CACHE: dict[tuple, callable] = {}
_PAD4_CACHE: dict[tuple, callable] = {}
_CROP4_CACHE: dict[tuple, callable] = {}
_PACK_TAIL_CACHE: dict[tuple, callable] = {}
_REDUCE_TAIL_CACHE: dict[tuple, callable] = {}
_TOPK_BIAS_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


_DEFAULT_GEMM_BLOCK: int = 128
"""Constant fallback for direct-kernel callers (tests / low-level scripts).
The operation layer (``DeepSeekAttention`` op via ``DeepSeekAttentionConfig.gemm_block``)
is the single source of truth for shape-aware tile selection."""


def _get_indexer_ffi(B, S, HI, DI, index_scale, causal, dtype, *, gemm_block: int):
    bt = bs = int(gemm_block)
    key = (B, S, HI, DI, bt, bs, round(float(index_scale), 8), bool(causal), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INDEXER_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_dsa_indexer_prim_func(
            batch=B,
            seq_len=S,
            index_heads=HI,
            index_head_dim=DI,
            block_t=bt,
            block_s=bs,
            index_scale=float(index_scale),
            causal=bool(causal),
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, S), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INDEXER_CACHE[key] = ffi
        return ffi


def _get_matmul_ffi(M, K, N, dtype, out_dtype, *, gemm_block: int):
    """Native ``(M, K) @ (K, N) -> (M, N)`` GEMM FFI."""
    bm = bn = int(gemm_block)
    bk = 64 if K % 64 == 0 else 32
    key = (M, K, N, bm, bn, bk, str(jnp.dtype(dtype)), str(jnp.dtype(out_dtype)))
    with _LOCK:
        cached = _MATMUL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_matmul_prim_func(
            m=M,
            k=K,
            n=N,
            block_m=bm,
            block_n=bn,
            block_k=bk,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((M, N), out_dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _MATMUL_CACHE[key] = ffi
        return ffi


def _get_add_ffi(M, N, dtype, *, gemm_block: int):
    """Native elementwise add FFI."""
    bm = bn = int(gemm_block)
    key = (M, N, bm, bn, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _ADD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_add_prim_func(
            m=M,
            n=N,
            dtype=dtype,
            block_m=bm,
            block_n=bn,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((M, N), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _ADD_CACHE[key] = ffi
        return ffi


def _get_cast_ffi(M, N, in_dtype, out_dtype, *, gemm_block: int):
    """Native 2D dtype cast FFI."""
    bm = bn = int(gemm_block)
    key = (M, N, bm, bn, str(jnp.dtype(in_dtype)), str(jnp.dtype(out_dtype)))
    with _LOCK:
        cached = _CAST_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_cast_prim_func(
            m=M,
            n=N,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            block_m=bm,
            block_n=bn,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((M, N), out_dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _CAST_CACHE[key] = ffi
        return ffi


def _get_pad4_ffi(B, S, H, in_dim, out_dim, dtype):
    """Native 4D last-dimension pad FFI."""
    key = (B, S, H, in_dim, out_dim, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _PAD4_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_pad_lastdim_prim_func(
            batch=B,
            seq_len=S,
            heads=H,
            in_dim=in_dim,
            out_dim=out_dim,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, H, out_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PAD4_CACHE[key] = ffi
        return ffi


def _get_crop4_ffi(B, S, H, in_dim, out_dim, dtype):
    """Native 4D last-dimension crop FFI."""
    key = (B, S, H, in_dim, out_dim, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _CROP4_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_crop_lastdim_prim_func(
            batch=B,
            seq_len=S,
            heads=H,
            in_dim=in_dim,
            out_dim=out_dim,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, H, out_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _CROP4_CACHE[key] = ffi
        return ffi


def _get_pack_tail_ffi(B, S, H, main_dim, tail_dim, out_dim, dtype):
    """Native shared-tail pack FFI."""
    key = (B, S, H, main_dim, tail_dim, out_dim, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _PACK_TAIL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_pack_shared_tail_prim_func(
            batch=B,
            seq_len=S,
            heads=H,
            main_dim=main_dim,
            tail_dim=tail_dim,
            out_dim=out_dim,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, H, out_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACK_TAIL_CACHE[key] = ffi
        return ffi


def _get_reduce_tail_ffi(B, S, H, main_dim, tail_dim, in_dim, dtype):
    """Native shared-tail gradient reduction FFI."""
    key = (B, S, H, main_dim, tail_dim, in_dim, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_TAIL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_shared_tail_prim_func(
            batch=B,
            seq_len=S,
            heads=H,
            main_dim=main_dim,
            tail_dim=tail_dim,
            in_dim=in_dim,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, tail_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_TAIL_CACHE[key] = ffi
        return ffi


def _get_topk_bias_ffi(B, S, index_topk, causal, *, gemm_block: int):
    """Build (or retrieve from cache) the top-k bias mask FFI call.

    The mask is a ``(B, 1, S, S)`` float32 tensor of ``0.0`` / ``-inf``
    values produced by the native top-k bias kernel.

    Args:
        B: batch size.
        S: sequence length.
        index_topk: number of top-k positions to keep (set to ``0.0``).
        causal: apply causal masking on top of the top-k selection.

    Returns:
        Compiled callable ``(index_score[B,S,S]) -> mask[B,1,S,S]``.
    """
    bs = int(gemm_block)
    key = (B, S, int(index_topk), bs, bool(causal))
    with _LOCK:
        cached = _TOPK_BIAS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_topk_bias_prim_func(
            batch=B,
            seq_len=S,
            index_topk=int(index_topk),
            block_s=bs,
            causal=bool(causal),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, 1, S, S), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _TOPK_BIAS_CACHE[key] = ffi
        return ffi


def _matmul(a, b, out_dtype, *, gemm_block: int):
    """``a @ b`` through the native tile-lang GEMM kernel."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, (a.shape, b.shape)
    ffi = _get_matmul_ffi(m, k, n, a.dtype, out_dtype, gemm_block=gemm_block)
    return ffi(a, b)


def _add(a, b, *, gemm_block: int):
    """``a + b`` through a native tile-lang elementwise kernel."""
    m, n = a.shape
    assert a.shape == b.shape, (a.shape, b.shape)
    ffi = _get_add_ffi(m, n, a.dtype, gemm_block=gemm_block)
    return ffi(a, b)


def _cast2d(a, out_dtype, *, gemm_block: int):
    """Cast a 2D tensor through a native tile-lang kernel."""
    if jnp.dtype(a.dtype) == jnp.dtype(out_dtype):
        return a
    m, n = a.shape
    ffi = _get_cast_ffi(m, n, a.dtype, out_dtype, gemm_block=gemm_block)
    return ffi(a)


def _pad_lastdim_impl(x, out_dim):
    """Native 4D last-dimension zero pad."""
    B, S, H, in_dim = x.shape
    if in_dim == out_dim:
        return x
    ffi = _get_pad4_ffi(B, S, H, in_dim, out_dim, x.dtype)
    return ffi(x)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _pad_lastdim(x, out_dim):
    """Differentiable native 4D last-dimension zero pad."""
    return _pad_lastdim_impl(x, out_dim)


def _pad_lastdim_fwd(x, out_dim):
    return _pad_lastdim_impl(x, out_dim), x.shape[-1]


def _pad_lastdim_bwd(out_dim, in_dim, g):
    if in_dim == out_dim:
        return (g,)
    return (_crop_lastdim_impl(g, in_dim),)


_pad_lastdim.defvjp(_pad_lastdim_fwd, _pad_lastdim_bwd)


def _crop_lastdim_impl(x, out_dim):
    """Native 4D last-dimension crop."""
    B, S, H, in_dim = x.shape
    if in_dim == out_dim:
        return x
    ffi = _get_crop4_ffi(B, S, H, in_dim, out_dim, x.dtype)
    return ffi(x)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _crop_lastdim(x, out_dim):
    """Differentiable native 4D last-dimension crop."""
    return _crop_lastdim_impl(x, out_dim)


def _crop_lastdim_fwd(x, out_dim):
    return _crop_lastdim_impl(x, out_dim), x.shape[-1]


def _crop_lastdim_bwd(out_dim, in_dim, g):
    if in_dim == out_dim:
        return (g,)
    return (_pad_lastdim_impl(g, in_dim),)


_crop_lastdim.defvjp(_crop_lastdim_fwd, _crop_lastdim_bwd)


def _pack_shared_tail_impl(main, tail, out_dim):
    """Native pack of per-head main dims and head-shared tail dims."""
    B, S, H, main_dim = main.shape
    tail_dim = tail.shape[-1]
    ffi = _get_pack_tail_ffi(B, S, H, main_dim, tail_dim, out_dim, main.dtype)
    return ffi(main, tail)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _pack_shared_tail(main, tail, out_dim):
    """Differentiable native shared-tail pack."""
    return _pack_shared_tail_impl(main, tail, out_dim)


def _pack_shared_tail_fwd(main, tail, out_dim):
    return _pack_shared_tail_impl(main, tail, out_dim), (main.shape[-1], tail.shape[-1])


def _pack_shared_tail_bwd(out_dim, residual, g):
    main_dim, tail_dim = residual
    B, S, H, _ = g.shape
    d_main = _crop_lastdim_impl(g, main_dim)
    reduce_tail = _get_reduce_tail_ffi(B, S, H, main_dim, tail_dim, out_dim, g.dtype)
    d_tail = reduce_tail(g)
    return d_main, d_tail


_pack_shared_tail.defvjp(_pack_shared_tail_fwd, _pack_shared_tail_bwd)


def _mla_attention_tilelang(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    b_q: jax.Array | None,
    b_k: jax.Array | None,
    softmax_scale: float,
    causal: bool,
    bias: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    softmax_aux: jax.Array | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    dropout_prob: float = 0.0,
    dropout_key=None,
) -> jax.Array:
    """Run MLA attention through native pack/pad kernels plus FlashAttention."""
    _, _, _, q_dim = query.shape
    _, _, _, k_dim = key.shape
    v_dim = value.shape[-1]

    if b_k is None:
        score_dim = q_dim
        if k_dim != q_dim:
            raise ValueError(f"MLA without b_k requires query dim {q_dim} to match key dim {k_dim}.")
        q_full = _pad_lastdim(query, max(score_dim, v_dim))
        k_full = _pad_lastdim(key, max(score_dim, v_dim))
    else:
        rope_dim = b_k.shape[-1]
        if b_q is None:
            score_dim = q_dim
            if q_dim != k_dim + rope_dim:
                raise ValueError(
                    f"MLA with b_k requires query dim {q_dim} to equal key dim {k_dim} plus rope dim {rope_dim}."
                )
            q_full = _pad_lastdim(query, max(score_dim, v_dim))
        else:
            score_dim = k_dim + rope_dim
            if q_dim != k_dim:
                raise ValueError(f"MLA with b_q/b_k requires query dim {q_dim} to match key dim {k_dim}.")
            q_full = _pack_shared_tail(query, b_q, max(score_dim, v_dim))
        k_full = _pack_shared_tail(key, b_k, max(score_dim, v_dim))

    attn_dim = max(score_dim, v_dim)
    v_full = _pad_lastdim(value, attn_dim)
    out_full = flash_attention_tilelang(
        q_full,
        k_full,
        v_full,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        attention_mask=attention_mask,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=dropout_prob,
        dropout_key=dropout_key,
    )
    return _crop_lastdim(out_full, v_dim)


def _kv_recon_impl(key_value, w_kc, w_vc, *, gemm_block: int):
    """Reconstruct ``K`` / ``V`` from the compressed latent via native GEMMs."""
    B, S, L = key_value.shape
    _, H, Dk = w_kc.shape
    _, Hv, Dv = w_vc.shape
    assert H == Hv, (w_kc.shape, w_vc.shape)
    k_hd = H * Dk
    v_hd = H * Dv
    kv_flat = key_value.reshape(B * S, L)
    wkc_flat = w_kc.reshape(L, k_hd)
    wvc_flat = w_vc.reshape(L, v_hd)
    k_r = _matmul(kv_flat, wkc_flat, key_value.dtype, gemm_block=gemm_block).reshape(B, S, H, Dk)
    v_r = _matmul(kv_flat, wvc_flat, key_value.dtype, gemm_block=gemm_block).reshape(B, S, H, Dv)
    return k_r, v_r


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _kv_recon(key_value, w_kc, w_vc, gemm_block=_DEFAULT_GEMM_BLOCK):
    """Differentiable KV up-projection ``(K_r, V_r)`` from the latent."""
    return _kv_recon_impl(key_value, w_kc, w_vc, gemm_block=int(gemm_block))


def _kv_recon_fwd(key_value, w_kc, w_vc, gemm_block=_DEFAULT_GEMM_BLOCK):
    k_r, v_r = _kv_recon_impl(key_value, w_kc, w_vc, gemm_block=int(gemm_block))
    return (k_r, v_r), (key_value, w_kc, w_vc)


def _kv_recon_bwd(gemm_block, residual, g):
    gb = int(gemm_block)
    key_value, w_kc, w_vc = residual
    dk_r, dv_r = g
    B, S, L = key_value.shape
    _, H, Dk = w_kc.shape
    _, Hv, Dv = w_vc.shape
    assert H == Hv, (w_kc.shape, w_vc.shape)
    k_hd = H * Dk
    v_hd = H * Dv
    kv_flat = key_value.reshape(B * S, L)
    wkc_flat = w_kc.reshape(L, k_hd)
    wvc_flat = w_vc.reshape(L, v_hd)
    dkr_flat = _cast2d(dk_r.reshape(B * S, k_hd), key_value.dtype, gemm_block=gb)
    dvr_flat = _cast2d(dv_r.reshape(B * S, v_hd), key_value.dtype, gemm_block=gb)

    dkv_k = _matmul(dkr_flat, wkc_flat.T, key_value.dtype, gemm_block=gb)
    dkv_v = _matmul(dvr_flat, wvc_flat.T, key_value.dtype, gemm_block=gb)
    dkv = _add(dkv_k, dkv_v, gemm_block=gb).reshape(B, S, L)

    dwkc = _matmul(kv_flat.T, dkr_flat, w_kc.dtype, gemm_block=gb).reshape(L, H, Dk)
    dwvc = _matmul(kv_flat.T, dvr_flat, w_vc.dtype, gemm_block=gb).reshape(L, H, Dv)
    return dkv, dwkc, dwvc


_kv_recon.defvjp(_kv_recon_fwd, _kv_recon_bwd)


def deepseek_attn_tilelang(
    query: jax.Array,
    key_value: jax.Array,
    w_kc: jax.Array,
    w_vc: jax.Array,
    query_index: jax.Array,
    key_index: jax.Array,
    index_weights: jax.Array,
    *,
    index_topk: int,
    softmax_scale: float | None,
    index_softmax_scale: float | None,
    causal: bool,
    b_q: jax.Array | None = None,
    b_k: jax.Array | None = None,
    gemm_block: int = _DEFAULT_GEMM_BLOCK,
) -> jax.Array:
    """DeepSeek Sparse Attention — native tile-lang kernels, forward + backward.

    Executes the full DSA pipeline:

    1. Lightning-Indexer: ``index_score = indexer(query_index, key_index,
       index_weights)`` via a native tile-lang GEMM + reduce kernel.
    2. Top-k bias: ``mask = stop_gradient(topk_bias(index_score))`` — a
       ``(B, 1, S, S)`` float32 mask that zeros out the top-k positions
       and fills the rest with ``-inf``.
    3. KV reconstruction: ``k_r, v_r = _kv_recon(key_value, w_kc, w_vc)``
       via native GEMMs with VJP.
    4. MLA attention: ``_mla_attention_tilelang(query, k_r, v_r, bias=mask)``.

    Differentiable w.r.t. ``query``, ``key_value``, ``w_kc`` and ``w_vc``.
    The indexer / top-k path is integer-valued and stop-gradient.

    Args:
        query: ``(batch, seq_len, q_heads, q_head_dim)``.
        key_value: compressed KV latent ``(batch, seq_len, kv_lora_rank)``.
        w_kc: key projection weight ``(kv_lora_rank, kv_heads, qk_nope_head_dim)``.
        w_vc: value projection weight ``(kv_lora_rank, kv_heads, v_head_dim)``.
        query_index: indexer query heads ``(batch, seq_len, index_heads, index_head_dim)``.
        key_index: shared indexer key ``(batch, seq_len, index_head_dim)``.
        index_weights: per-head indexer bias ``(batch, seq_len, index_heads)``.
        index_topk: number of positions to keep in the sparse attention mask.
        softmax_scale: attention softmax scale; defaults to
            ``1/sqrt(score_dim)`` where ``score_dim`` accounts for RoPE.
        index_softmax_scale: indexer softmax scale; defaults to
            ``1/sqrt(index_head_dim)``.
        causal: apply causal masking inside the indexer.
        b_q: optional RoPE query tail ``(batch, seq_len, qk_rope_head_dim)``.
        b_k: optional shared RoPE key tail ``(batch, seq_len, qk_rope_head_dim)``.

    Returns:
        ``(batch, seq_len, q_heads, v_head_dim)`` attention output.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang deepseek_attn requires both `tilelang` and `jax_tvm_ffi`.")

    B, S, _HQ, D = query.shape
    HI, DI = query_index.shape[2], query_index.shape[3]
    if softmax_scale is None:
        effective_dim = w_kc.shape[-1] + (0 if b_k is None else b_k.shape[-1])
        softmax_scale = 1.0 / math.sqrt(effective_dim if b_k is not None else D)
    if index_softmax_scale is None:
        index_softmax_scale = 1.0 / math.sqrt(DI)

    gb = int(gemm_block)
    if gb <= 0:
        raise ValueError(f"deepseek_attn: gemm_block must be a positive int (got {gb}).")

    indexer = _get_indexer_ffi(B, S, HI, DI, index_softmax_scale, causal, query_index.dtype, gemm_block=gb)
    index_score = indexer(query_index, key_index, index_weights)

    topk_bias = _get_topk_bias_ffi(B, S, index_topk, causal, gemm_block=gb)
    mask = jax.lax.stop_gradient(topk_bias(index_score))

    k_r, v_r = _kv_recon(key_value, w_kc, w_vc, gb)

    return _mla_attention_tilelang(
        query,
        k_r,
        v_r,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=softmax_scale,
        causal=False,
        bias=mask,
    )
