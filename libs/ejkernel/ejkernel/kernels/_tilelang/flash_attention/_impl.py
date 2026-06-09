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

"""JAX glue around the tile-lang FlashAttention prim_funcs.

Two execution paths share this module:

* The **lean path** (:func:`_flash_attention_core`) is the autotuned
  FlashAttention-2 used when the caller requests no score-space features.
  It is byte-identical to the benchmarked hot kernel and is never perturbed.
* The **full path** (:func:`_fa_full_core`) is selected the moment any of
  ``bias`` / ``attention_mask`` / ``sliding_window`` / ``q_segment_ids`` /
  ``kv_segment_ids`` / ``logits_soft_cap`` / ``softmax_aux`` / dropout /
  GQA / ``normalize_output=False`` is in play. It runs the feature-complete
  tile-lang kernels: every modifier is applied natively inside the kernel,
  nothing is ignored.

Public layout is ``(B, N, H, D)``; internally we transpose to ``(B, H, N, D)``
so each CTA touches a contiguous head slab.
"""

from __future__ import annotations

import functools
import math
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_bwd_dkdv_prim_func,
    make_bwd_dkdv_prim_func_full,
    make_bwd_dq_prim_func,
    make_bwd_dq_prim_func_full,
    make_bwd_preprocess_prim_func,
    make_fwd_prim_func,
    make_fwd_prim_func_full,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)


_FWD_FFI_CACHE: dict[tuple, callable] = {}
_BWD_PRE_FFI_CACHE: dict[tuple, callable] = {}
_BWD_DKDV_FFI_CACHE: dict[tuple, callable] = {}
_BWD_DQ_FFI_CACHE: dict[tuple, callable] = {}
_FWD_FULL_CACHE: dict[tuple, callable] = {}
_BWD_DKDV_FULL_CACHE: dict[tuple, callable] = {}
_BWD_DQ_FULL_CACHE: dict[tuple, callable] = {}
_CACHE_LOCK = threading.Lock()


_DEFAULT_FWD_BLOCK_M: int = 64
_DEFAULT_FWD_BLOCK_N: int = 64
_DEFAULT_BWD_BLOCK_M: int = 32
_DEFAULT_BWD_BLOCK_N: int = 64


def _threads_from_warps(num_warps) -> int:
    """Convert a warp-count hint to TileLang CTA threads."""
    if num_warps is None:
        return 128
    return max(32, int(num_warps) * 32)


def _get_fwd_ffi(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    dtype: jnp.dtype,
    block_m: int | None = None,
    block_n: int | None = None,
    num_stages: int = 2,
    threads: int = 128,
):
    default_m, default_n = _DEFAULT_FWD_BLOCK_M, _DEFAULT_FWD_BLOCK_N
    block_m = default_m if block_m is None else int(block_m)
    block_n = default_n if block_n is None else int(block_n)
    key = (
        "fwd",
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_m,
        block_n,
        int(num_stages),
        int(threads),
        round(float(softmax_scale), 8),
        bool(causal),
        str(jnp.dtype(dtype)),
    )
    with _CACHE_LOCK:
        cached = _FWD_FFI_CACHE.get(key)
        if cached is not None:
            return cached, block_m, block_n

    out_spec = (
        jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, head_dim), dtype),
        jax.ShapeDtypeStruct((batch, num_heads, seq_len_q), jnp.float32),
    )

    def _builder(*, block_m, block_n, num_stages, threads):
        return make_fwd_prim_func(
            batch=batch,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            num_stages=num_stages,
            threads=threads,
        )

    ffi = build_tilelang_call(
        _builder(block_m=block_m, block_n=block_n, num_stages=int(num_stages), threads=int(threads)),
        output_shape_dtype=out_spec,
        compile_flags=_DEFAULT_COMPILE_FLAGS,
    )

    with _CACHE_LOCK:
        _FWD_FFI_CACHE[key] = ffi
    return ffi, block_m, block_n


def _get_bwd_pre_ffi(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    head_dim: int,
    dtype: jnp.dtype,
    block_m: int | None = None,
    threads: int = 128,
):
    default_m, _ = _DEFAULT_BWD_BLOCK_M, _DEFAULT_BWD_BLOCK_N
    block_m = default_m if block_m is None else int(block_m)
    key = (batch, num_heads, seq_len_q, head_dim, block_m, int(threads), str(jnp.dtype(dtype)))
    with _CACHE_LOCK:
        cached = _BWD_PRE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_preprocess_prim_func(
            batch=batch,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            head_dim=head_dim,
            block_m=block_m,
            dtype=dtype,
            threads=int(threads),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_heads, seq_len_q), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_PRE_FFI_CACHE[key] = ffi
        return ffi


def _get_bwd_dkdv_ffi(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    dtype: jnp.dtype,
    block_m: int | None = None,
    block_n: int | None = None,
    num_stages: int = 2,
    threads: int = 128,
):
    default_m, default_n = _DEFAULT_BWD_BLOCK_M, _DEFAULT_BWD_BLOCK_N
    block_m = default_m if block_m is None else int(block_m)
    block_n = default_n if block_n is None else int(block_n)
    key = (
        "dkdv",
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_m,
        block_n,
        int(num_stages),
        int(threads),
        round(float(softmax_scale), 8),
        bool(causal),
        str(jnp.dtype(dtype)),
    )
    with _CACHE_LOCK:
        cached = _BWD_DKDV_FFI_CACHE.get(key)
        if cached is not None:
            return cached

    out_spec = (
        jax.ShapeDtypeStruct((batch, num_heads, seq_len_k, head_dim), dtype),
        jax.ShapeDtypeStruct((batch, num_heads, seq_len_k, head_dim), dtype),
    )

    def _builder(*, block_m, block_n, num_stages, threads):
        return make_bwd_dkdv_prim_func(
            batch=batch,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            num_stages=num_stages,
            threads=threads,
        )

    ffi = build_tilelang_call(
        _builder(block_m=block_m, block_n=block_n, num_stages=int(num_stages), threads=int(threads)),
        output_shape_dtype=out_spec,
        compile_flags=_DEFAULT_COMPILE_FLAGS,
    )

    with _CACHE_LOCK:
        _BWD_DKDV_FFI_CACHE[key] = ffi
    return ffi


def _get_bwd_dq_ffi(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    dtype: jnp.dtype,
    block_m: int | None = None,
    block_n: int | None = None,
    num_stages: int = 2,
    threads: int = 128,
):
    default_m, default_n = _DEFAULT_BWD_BLOCK_M, _DEFAULT_BWD_BLOCK_N
    block_m = default_m if block_m is None else int(block_m)
    block_n = default_n if block_n is None else int(block_n)
    key = (
        "dq",
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_m,
        block_n,
        int(num_stages),
        int(threads),
        round(float(softmax_scale), 8),
        bool(causal),
        str(jnp.dtype(dtype)),
    )
    with _CACHE_LOCK:
        cached = _BWD_DQ_FFI_CACHE.get(key)
        if cached is not None:
            return cached

    out_spec = jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, head_dim), dtype)

    def _builder(*, block_m, block_n, num_stages, threads):
        return make_bwd_dq_prim_func(
            batch=batch,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            num_stages=num_stages,
            threads=threads,
        )

    ffi = build_tilelang_call(
        _builder(block_m=block_m, block_n=block_n, num_stages=int(num_stages), threads=int(threads)),
        output_shape_dtype=out_spec,
        compile_flags=_DEFAULT_COMPILE_FLAGS,
    )

    with _CACHE_LOCK:
        _BWD_DQ_FFI_CACHE[key] = ffi
    return ffi


def _get_fwd_ffi_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float | None,
    normalize_output: bool,
    block_m: int,
    block_n: int,
    dtype: jnp.dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    softmax_aux_shape: tuple[int, int],
    softmax_aux_dtype,
    use_softmax_aux: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
):
    """Build (and cache) the feature-complete forward FFI call."""
    key = (
        "fwd_full",
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        round(float(softmax_scale), 8),
        bool(causal),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
        bool(normalize_output),
        block_m,
        block_n,
        str(jnp.dtype(dtype)),
        tuple(bias_shape),
        str(jnp.dtype(bias_dtype)),
        bool(use_bias),
        tuple(mask_shape),
        str(jnp.dtype(mask_dtype)),
        bool(use_mask),
        tuple(q_segment_shape),
        str(jnp.dtype(q_segment_dtype)),
        tuple(kv_segment_shape),
        str(jnp.dtype(kv_segment_dtype)),
        bool(use_segments),
        tuple(softmax_aux_shape),
        str(jnp.dtype(softmax_aux_dtype)),
        bool(use_softmax_aux),
        None if window is None else tuple(int(x) for x in window),
        round(float(dropout_prob), 8),
    )
    with _CACHE_LOCK:
        cached = _FWD_FULL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func_full(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            bias_shape=tuple(bias_shape),
            bias_dtype=bias_dtype,
            use_bias=bool(use_bias),
            mask_shape=tuple(mask_shape),
            mask_dtype=mask_dtype,
            use_mask=bool(use_mask),
            q_segment_shape=tuple(q_segment_shape),
            q_segment_dtype=q_segment_dtype,
            kv_segment_shape=tuple(kv_segment_shape),
            kv_segment_dtype=kv_segment_dtype,
            use_segments=bool(use_segments),
            softmax_aux_shape=tuple(softmax_aux_shape),
            softmax_aux_dtype=softmax_aux_dtype,
            use_softmax_aux=bool(use_softmax_aux),
            window=window,
            dropout_prob=float(dropout_prob),
            logits_soft_cap=logits_soft_cap,
            normalize_output=bool(normalize_output),
            num_stages=2,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, head_dim), dtype),
                jax.ShapeDtypeStruct((batch, num_heads, seq_len_q), jnp.float32),
                jax.ShapeDtypeStruct((batch, num_heads, seq_len_q), jnp.float32),
                jax.ShapeDtypeStruct((batch, num_heads, seq_len_q), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_FULL_CACHE[key] = ffi
        return ffi


def _get_bwd_dkdv_ffi_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float | None,
    normalize_output: bool,
    block_m: int,
    block_n: int,
    dtype: jnp.dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
):
    """Build (and cache) the feature-complete dK/dV backward FFI call."""
    key = (
        "dkdv_full",
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        round(float(softmax_scale), 8),
        bool(causal),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
        bool(normalize_output),
        block_m,
        block_n,
        str(jnp.dtype(dtype)),
        tuple(bias_shape),
        str(jnp.dtype(bias_dtype)),
        bool(use_bias),
        tuple(mask_shape),
        str(jnp.dtype(mask_dtype)),
        bool(use_mask),
        tuple(q_segment_shape),
        str(jnp.dtype(q_segment_dtype)),
        tuple(kv_segment_shape),
        str(jnp.dtype(kv_segment_dtype)),
        bool(use_segments),
        None if window is None else tuple(int(x) for x in window),
        round(float(dropout_prob), 8),
    )
    with _CACHE_LOCK:
        cached = _BWD_DKDV_FULL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_dkdv_prim_func_full(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            bias_shape=tuple(bias_shape),
            bias_dtype=bias_dtype,
            use_bias=bool(use_bias),
            mask_shape=tuple(mask_shape),
            mask_dtype=mask_dtype,
            use_mask=bool(use_mask),
            q_segment_shape=tuple(q_segment_shape),
            q_segment_dtype=q_segment_dtype,
            kv_segment_shape=tuple(kv_segment_shape),
            kv_segment_dtype=kv_segment_dtype,
            use_segments=bool(use_segments),
            window=window,
            dropout_prob=float(dropout_prob),
            logits_soft_cap=logits_soft_cap,
            normalize_output=bool(normalize_output),
            num_stages=2,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, num_kv_heads, seq_len_k, head_dim), jnp.float32),
                jax.ShapeDtypeStruct((batch, num_kv_heads, seq_len_k, head_dim), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_DKDV_FULL_CACHE[key] = ffi
        return ffi


def _get_bwd_dq_ffi_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float | None,
    normalize_output: bool,
    block_m: int,
    block_n: int,
    dtype: jnp.dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
):
    """Build (and cache) the feature-complete dQ backward FFI call."""
    key = (
        "dq_full",
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        round(float(softmax_scale), 8),
        bool(causal),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
        bool(normalize_output),
        block_m,
        block_n,
        str(jnp.dtype(dtype)),
        tuple(bias_shape),
        str(jnp.dtype(bias_dtype)),
        bool(use_bias),
        tuple(mask_shape),
        str(jnp.dtype(mask_dtype)),
        bool(use_mask),
        tuple(q_segment_shape),
        str(jnp.dtype(q_segment_dtype)),
        tuple(kv_segment_shape),
        str(jnp.dtype(kv_segment_dtype)),
        bool(use_segments),
        None if window is None else tuple(int(x) for x in window),
        round(float(dropout_prob), 8),
    )
    with _CACHE_LOCK:
        cached = _BWD_DQ_FULL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_dq_prim_func_full(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            bias_shape=tuple(bias_shape),
            bias_dtype=bias_dtype,
            use_bias=bool(use_bias),
            mask_shape=tuple(mask_shape),
            mask_dtype=mask_dtype,
            use_mask=bool(use_mask),
            q_segment_shape=tuple(q_segment_shape),
            q_segment_dtype=q_segment_dtype,
            kv_segment_shape=tuple(kv_segment_shape),
            kv_segment_dtype=kv_segment_dtype,
            use_segments=bool(use_segments),
            window=window,
            dropout_prob=float(dropout_prob),
            logits_soft_cap=logits_soft_cap,
            normalize_output=bool(normalize_output),
            num_stages=2,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_DQ_FULL_CACHE[key] = ffi
        return ffi


def _to_bhnd(x: jax.Array) -> jax.Array:
    """``(B, N, H, D) -> (B, H, N, D)``."""
    return jnp.transpose(x, (0, 2, 1, 3))


def _to_bnhd(x: jax.Array) -> jax.Array:
    """Transpose ``(B, H, N, D)`` to public ``(B, N, H, D)`` layout."""
    return jnp.transpose(x, (0, 2, 1, 3))


def _flash_attention_fwd_only(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    softmax_scale: float | None,
    causal: bool,
    fwd_block_m: int | None = None,
    fwd_block_n: int | None = None,
    fwd_num_stages: int = 2,
    fwd_threads: int = 128,
):
    """Lean forward-only entry point. Returns ``(O, L)`` in BHND layout."""
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    batch, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, num_kv_heads, _ = k.shape
    assert num_kv_heads == num_heads, "lean path is MHA-only; GQA routes to the full path"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qb = _to_bhnd(q)
    kb = _to_bhnd(k)
    vb = _to_bhnd(v)

    ffi, _, _ = _get_fwd_ffi(
        batch=batch,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        dtype=q.dtype,
        block_m=fwd_block_m,
        block_n=fwd_block_n,
        num_stages=fwd_num_stages,
        threads=fwd_threads,
    )
    o_bhnd, lse = ffi(qb, kb, vb)
    return _to_bnhd(o_bhnd), lse


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(3, 13)))
def _flash_attention_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    softmax_scale: float | None,
    causal: bool,
    fwd_block_m: int,
    fwd_block_n: int,
    bwd_block_m: int,
    bwd_block_n: int,
    fwd_num_stages: int,
    bwd_num_stages: int,
    fwd_threads: int,
    bwd_threads: int,
) -> jax.Array:
    """The lean differentiable core. Public ``(B, N, H, D)`` layout."""
    o, _ = _flash_attention_fwd_only(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        fwd_block_m=fwd_block_m,
        fwd_block_n=fwd_block_n,
        fwd_num_stages=fwd_num_stages,
        fwd_threads=fwd_threads,
    )
    return o


def _fwd_for_vjp(
    q,
    k,
    v,
    softmax_scale,
    causal,
    fwd_block_m,
    fwd_block_n,
    bwd_block_m,
    bwd_block_n,
    fwd_num_stages,
    bwd_num_stages,
    fwd_threads,
    bwd_threads,
):
    _ = bwd_block_m, bwd_block_n, bwd_num_stages, bwd_threads
    o, lse = _flash_attention_fwd_only(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        fwd_block_m=fwd_block_m,
        fwd_block_n=fwd_block_n,
        fwd_num_stages=fwd_num_stages,
        fwd_threads=fwd_threads,
    )
    return o, (q, k, v, o, lse)


def _bwd_from_vjp(
    softmax_scale,
    causal,
    fwd_block_m,
    fwd_block_n,
    bwd_block_m,
    bwd_block_n,
    fwd_num_stages,
    bwd_num_stages,
    fwd_threads,
    bwd_threads,
    residual,
    dO,
):
    _ = fwd_block_m, fwd_block_n, fwd_num_stages, fwd_threads
    q, k, v, o, lse = residual
    batch, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qb = _to_bhnd(q)
    kb = _to_bhnd(k)
    vb = _to_bhnd(v)
    ob = _to_bhnd(o)
    dob = _to_bhnd(dO)

    pre = _get_bwd_pre_ffi(
        batch=batch,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        dtype=q.dtype,
        block_m=bwd_block_m,
        threads=bwd_threads,
    )
    delta = pre(ob, dob)

    dkdv = _get_bwd_dkdv_ffi(
        batch=batch,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        dtype=q.dtype,
        block_m=bwd_block_m,
        block_n=bwd_block_n,
        num_stages=bwd_num_stages,
        threads=bwd_threads,
    )
    dk, dv = dkdv(qb, kb, vb, dob, lse, delta)

    dq_kernel = _get_bwd_dq_ffi(
        batch=batch,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        dtype=q.dtype,
        block_m=bwd_block_m,
        block_n=bwd_block_n,
        num_stages=bwd_num_stages,
        threads=bwd_threads,
    )
    dq = dq_kernel(qb, kb, vb, dob, lse, delta)

    return _to_bnhd(dq), _to_bnhd(dk), _to_bnhd(dv)


_flash_attention_core.defvjp(_fwd_for_vjp, _bwd_from_vjp)


def _full_fwd_only(
    q,
    k,
    v,
    bias,
    attention_mask,
    q_segment_ids,
    kv_segment_ids,
    softmax_aux,
    dropout_seed_buf,
    softmax_scale,
    causal,
    logits_soft_cap,
    num_kv_heads,
    normalize_output,
    fwd_block_m,
    fwd_block_n,
    use_bias,
    use_mask,
    use_segments,
    use_softmax_aux,
    window,
    dropout_prob,
):
    """Feature-complete forward-only entry point. Returns ``(O, L)`` in BHND layout.

    ``q/k/v`` are public ``(B, N, H, D)``; all score-space features are
    evaluated inside the tile-lang kernel from compact original inputs.
    """
    batch, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape

    qb = _to_bhnd(q)
    kb = _to_bhnd(k)
    vb = _to_bhnd(v)

    ffi = _get_fwd_ffi_full(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        logits_soft_cap=logits_soft_cap,
        normalize_output=bool(normalize_output),
        block_m=fwd_block_m,
        block_n=fwd_block_n,
        dtype=q.dtype,
        bias_shape=bias.shape,
        bias_dtype=bias.dtype,
        use_bias=bool(use_bias),
        mask_shape=attention_mask.shape,
        mask_dtype=attention_mask.dtype,
        use_mask=bool(use_mask),
        q_segment_shape=q_segment_ids.shape,
        q_segment_dtype=q_segment_ids.dtype,
        kv_segment_shape=kv_segment_ids.shape,
        kv_segment_dtype=kv_segment_ids.dtype,
        use_segments=bool(use_segments),
        softmax_aux_shape=softmax_aux.shape,
        softmax_aux_dtype=softmax_aux.dtype,
        use_softmax_aux=bool(use_softmax_aux),
        window=window,
        dropout_prob=float(dropout_prob),
    )
    o_bhnd, lse, m, am = ffi(
        qb,
        kb,
        vb,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        dropout_seed_buf,
    )
    return _to_bnhd(o_bhnd), lse, m, am


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 24)))
def _fa_full_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array,
    attention_mask: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    softmax_aux: jax.Array,
    dropout_seed_buf: jax.Array,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float | None,
    num_kv_heads: int,
    normalize_output: bool,
    fwd_block_m: int,
    fwd_block_n: int,
    bwd_block_m: int,
    bwd_block_n: int,
    use_bias: bool,
    use_mask: bool,
    use_segments: bool,
    use_softmax_aux: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
) -> jax.Array:
    """Feature-complete differentiable core (differentiates ``q/k/v`` only,
    matching the XLA reference — ``bias`` / mask / sink / dropout are
    constants w.r.t. autodiff)."""
    o, _, _, _ = _full_fwd_only(
        q,
        k,
        v,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        dropout_seed_buf,
        softmax_scale,
        causal,
        logits_soft_cap,
        num_kv_heads,
        normalize_output,
        fwd_block_m,
        fwd_block_n,
        use_bias,
        use_mask,
        use_segments,
        use_softmax_aux,
        window,
        dropout_prob,
    )
    return o


def _fa_full_fwd(
    q,
    k,
    v,
    bias,
    attention_mask,
    q_segment_ids,
    kv_segment_ids,
    softmax_aux,
    dropout_seed_buf,
    softmax_scale,
    causal,
    logits_soft_cap,
    num_kv_heads,
    normalize_output,
    fwd_block_m,
    fwd_block_n,
    bwd_block_m,
    bwd_block_n,
    use_bias,
    use_mask,
    use_segments,
    use_softmax_aux,
    window,
    dropout_prob,
):
    o, lse, m, am = _full_fwd_only(
        q,
        k,
        v,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        dropout_seed_buf,
        softmax_scale,
        causal,
        logits_soft_cap,
        num_kv_heads,
        normalize_output,
        fwd_block_m,
        fwd_block_n,
        use_bias,
        use_mask,
        use_segments,
        use_softmax_aux,
        window,
        dropout_prob,
    )
    return o, (q, k, v, o, lse, m, am, bias, attention_mask, q_segment_ids, kv_segment_ids, dropout_seed_buf)


def _fa_full_bwd(
    softmax_scale,
    causal,
    logits_soft_cap,
    num_kv_heads,
    normalize_output,
    fwd_block_m,
    fwd_block_n,
    bwd_block_m,
    bwd_block_n,
    use_bias,
    use_mask,
    use_segments,
    use_softmax_aux,
    window,
    dropout_prob,
    residual,
    dO,
):
    q, k, v, o, lse, m, am, bias, attention_mask, q_segment_ids, kv_segment_ids, dropout_seed_buf = residual
    batch, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape

    qb = _to_bhnd(q)
    kb = _to_bhnd(k)
    vb = _to_bhnd(v)
    ob = _to_bhnd(o)
    dob = _to_bhnd(dO)

    pre = _get_bwd_pre_ffi(
        batch=batch,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        dtype=q.dtype,
    )
    delta = pre(ob, dob)

    block_m, block_n = bwd_block_m, bwd_block_n

    dkdv = _get_bwd_dkdv_ffi_full(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        logits_soft_cap=logits_soft_cap,
        normalize_output=bool(normalize_output),
        block_m=block_m,
        block_n=block_n,
        dtype=q.dtype,
        bias_shape=bias.shape,
        bias_dtype=bias.dtype,
        use_bias=bool(use_bias),
        mask_shape=attention_mask.shape,
        mask_dtype=attention_mask.dtype,
        use_mask=bool(use_mask),
        q_segment_shape=q_segment_ids.shape,
        q_segment_dtype=q_segment_ids.dtype,
        kv_segment_shape=kv_segment_ids.shape,
        kv_segment_dtype=kv_segment_ids.dtype,
        use_segments=bool(use_segments),
        window=window,
        dropout_prob=float(dropout_prob),
    )
    dk, dv = dkdv(
        qb,
        kb,
        vb,
        dob,
        lse,
        m,
        am,
        delta,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        dropout_seed_buf,
    )

    dq_kernel = _get_bwd_dq_ffi_full(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        logits_soft_cap=logits_soft_cap,
        normalize_output=bool(normalize_output),
        block_m=block_m,
        block_n=block_n,
        dtype=q.dtype,
        bias_shape=bias.shape,
        bias_dtype=bias.dtype,
        use_bias=bool(use_bias),
        mask_shape=attention_mask.shape,
        mask_dtype=attention_mask.dtype,
        use_mask=bool(use_mask),
        q_segment_shape=q_segment_ids.shape,
        q_segment_dtype=q_segment_ids.dtype,
        kv_segment_shape=kv_segment_ids.shape,
        kv_segment_dtype=kv_segment_ids.dtype,
        use_segments=bool(use_segments),
        window=window,
        dropout_prob=float(dropout_prob),
    )
    dq = dq_kernel(
        qb,
        kb,
        vb,
        dob,
        lse,
        m,
        am,
        delta,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        dropout_seed_buf,
    )

    return _to_bnhd(dq), _to_bnhd(dk), _to_bnhd(dv), None, None, None, None, None, None


_fa_full_core.defvjp(_fa_full_fwd, _fa_full_bwd)


def _normalize_window(sliding_window):
    """Normalise a ``sliding_window`` argument to ``(left, right)`` or ``None``.

    Args:
        sliding_window: ``None`` (disabled), a symmetric integer, or a
            ``(left, right)`` tuple.

    Returns:
        ``None`` if disabled, else ``(int, int)`` with non-negative bounds.

    Raises:
        ValueError: if either bound is negative.
    """
    if sliding_window is None:
        return None
    if isinstance(sliding_window, int):
        return (int(sliding_window), int(sliding_window))
    left, right = sliding_window
    if left < 0 or right < 0:
        raise ValueError("sliding_window bounds must be non-negative.")
    return (int(left), int(right))


def _as_4d_feature(x: jax.Array, name: str) -> jax.Array:
    """Reshape a broadcastable score feature to rank four.

    Prepends unit dimensions until rank 4 is reached.  Does not copy data.

    Raises:
        ValueError: if ``x.ndim > 4``.
    """
    if x.ndim > 4:
        raise ValueError(f"{name} must be broadcastable to (B,H,NQ,NK); got rank {x.ndim}.")
    if x.ndim == 4:
        return x
    return jnp.reshape(x, (1,) * (4 - x.ndim) + x.shape)


def _as_2d_feature(x: jax.Array, name: str) -> jax.Array:
    """Reshape a per-head feature to rank two.

    Rank-1 ``(num_sinks,)`` inputs are reshaped to ``(1, num_sinks)``.

    Raises:
        ValueError: if ``x.ndim > 2``.
    """
    if x.ndim > 2:
        raise ValueError(f"{name} must be rank 1 or 2; got rank {x.ndim}.")
    if x.ndim == 2:
        return x
    return jnp.reshape(x, (1, x.shape[0]))


def _dropout_seed_buffer(dropout_seed: int | None, dropout_key) -> jax.Array:
    """Return the compact seed buffer consumed by the native dropout hash."""
    if dropout_key is not None:
        if getattr(dropout_key, "shape", None) == (2,) and jnp.dtype(dropout_key.dtype) == jnp.dtype(jnp.uint32):
            return dropout_key
        raise ValueError(
            "tile-lang flash_attention accepts dropout_seed or a legacy uint32[2] dropout_key; "
            "typed JAX keys would require host-side key unpacking."
        )
    return jnp.array([0 if dropout_seed is None else int(dropout_seed), 0], dtype=jnp.uint32)


def _prepare_full_features(
    query,
    bias,
    attention_mask,
    q_segment_ids,
    kv_segment_ids,
    softmax_aux,
    dropout_seed,
    dropout_key,
):
    """Prepare compact per-feature buffers for the full-path kernel.

    For each optional feature either normalises the caller-supplied array to
    the shape expected by the kernel, or creates a unit-sized placeholder when
    the feature is absent.  No score-space values are computed here; that
    happens inside the tile-lang kernel.

    Returns:
        A 10-tuple ``(bias_buf, mask_buf, qseg_buf, kvseg_buf, aux_buf,
        seed_buf, use_bias, use_mask, use_segments, use_softmax_aux)``.
    """
    use_bias = bias is not None
    use_mask = attention_mask is not None
    use_segments = q_segment_ids is not None
    use_softmax_aux = softmax_aux is not None

    if use_bias:
        bias_buf = _as_4d_feature(bias, "bias")
    else:
        bias_buf = jnp.empty((1, 1, 1, 1), dtype=query.dtype)

    if use_mask:
        mask_buf = _as_4d_feature(attention_mask, "attention_mask")
    else:
        mask_buf = jnp.empty((1, 1, 1, 1), dtype=jnp.bool_)

    if use_segments:
        qseg_buf = q_segment_ids
        kvseg_buf = kv_segment_ids if kv_segment_ids is not None else q_segment_ids
    else:
        qseg_buf = jnp.empty((1, 1), dtype=jnp.int32)
        kvseg_buf = jnp.empty((1, 1), dtype=jnp.int32)

    if use_softmax_aux:
        aux_buf = _as_2d_feature(softmax_aux, "softmax_aux")
    else:
        aux_buf = jnp.empty((1, 1), dtype=query.dtype)

    seed_buf = _dropout_seed_buffer(dropout_seed, dropout_key)
    return (
        bias_buf,
        mask_buf,
        qseg_buf,
        kvseg_buf,
        aux_buf,
        seed_buf,
        use_bias,
        use_mask,
        use_segments,
        use_softmax_aux,
    )


def _resolve_block_sizes(head_dim, seq_len_q, seq_len_k, fwd_params, bwd_params):
    """Pick FA tiles, honouring caller ``FwdParams`` / ``BwdParams`` hints.

    The heuristic picker provides the default; an explicit ``q_blocksize`` /
    ``kv_blocksize`` on the params object overrides it (clamped to ``>= 16``).
    """
    fwd_bm, fwd_bn = _DEFAULT_FWD_BLOCK_M, _DEFAULT_FWD_BLOCK_N
    bwd_bm, bwd_bn = _DEFAULT_BWD_BLOCK_M, _DEFAULT_BWD_BLOCK_N
    if fwd_params is not None:
        qb = getattr(fwd_params, "q_blocksize", None)
        kvb = getattr(fwd_params, "kv_blocksize", None)
        if qb is not None:
            fwd_bm = max(16, int(qb))
        if kvb is not None:
            fwd_bn = max(16, int(kvb))
    if bwd_params is not None:
        qb = getattr(bwd_params, "q_blocksize", None)
        kvb = getattr(bwd_params, "kv_blocksize", None)
        if qb is not None:
            bwd_bm = max(16, int(qb))
        if kvb is not None:
            bwd_bn = max(16, int(kvb))
    return fwd_bm, fwd_bn, bwd_bm, bwd_bn


def flash_attention_tilelang(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    bias: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    q_segment_ids: jax.Array | None = None,
    kv_segment_ids: jax.Array | None = None,
    softmax_aux: jax.Array | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    dropout_prob: float = 0.0,
    dropout_seed: int | None = None,
    dropout_key=None,
    normalize_output: bool = True,
    fwd_params=None,
    bwd_params=None,
) -> jax.Array:
    """Tile-lang FlashAttention-2 entry point (forward + backward).

    Every score-space feature is applied natively inside the tile-lang
    kernel. When none is requested the call routes to the lean autotuned
    FA2 kernel; otherwise it routes to the feature-complete kernel — both
    forward and backward.

    Args:
        query: ``(batch, seq_len_q, num_heads, head_dim)``.
        key:   ``(batch, seq_len_k, num_kv_heads, head_dim)`` — GQA/MQA
            (``num_kv_heads`` dividing ``num_heads``) is supported.
        value: ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        softmax_scale: ``QK^T`` multiplier. Defaults to ``1/sqrt(head_dim)``.
        causal: upper-triangular causal mask aligned to ``seq_len_k - seq_len_q``.
        bias: additive ``(batch, num_heads, seq_len_q, seq_len_k)`` logit bias.
        attention_mask: boolean/int keep-mask broadcastable to
            ``(batch, num_heads, seq_len_q, seq_len_k)``.
        q_segment_ids / kv_segment_ids: packed-sequence segment ids.
        softmax_aux: attention-sink logits ``(num_sinks,)`` or
            ``(num_heads, num_sinks)``.
        sliding_window: local-attention window (symmetric int or
            ``(left, right)``).
        logits_soft_cap: ``cap * tanh(logits / cap)`` soft cap.
        dropout_prob / dropout_seed: attention-weight dropout.
        normalize_output: divide by the softmax denominator (default True).

    Returns:
        ``(batch, seq_len_q, num_heads, head_dim)`` attention output.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError(
            "tile-lang FlashAttention requires both `tilelang` and `jax_tvm_ffi` to be importable in this environment."
        )

    _, seq_len_q, num_heads, head_dim = query.shape
    _, seq_len_k, num_kv_heads, _ = key.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads}).")
    if not 0.0 <= float(dropout_prob) < 1.0:
        raise ValueError("dropout_prob must be in [0, 1).")

    window = _normalize_window(sliding_window)

    needs_full = (
        bias is not None
        or attention_mask is not None
        or q_segment_ids is not None
        or kv_segment_ids is not None
        or softmax_aux is not None
        or window is not None
        or logits_soft_cap is not None
        or dropout_prob > 0.0
        or num_kv_heads != num_heads
        or not normalize_output
    )

    fwd_bm, fwd_bn, bwd_bm, bwd_bn = _resolve_block_sizes(head_dim, seq_len_q, seq_len_k, fwd_params, bwd_params)
    fwd_stages = (
        2 if fwd_params is None or getattr(fwd_params, "num_stages", None) is None else int(fwd_params.num_stages)
    )
    bwd_stages = (
        2 if bwd_params is None or getattr(bwd_params, "num_stages", None) is None else int(bwd_params.num_stages)
    )
    fwd_threads = _threads_from_warps(None if fwd_params is None else getattr(fwd_params, "num_warps", None))
    bwd_threads = _threads_from_warps(None if bwd_params is None else getattr(bwd_params, "num_warps", None))

    if not needs_full:
        return _flash_attention_core(
            query,
            key,
            value,
            float(softmax_scale),
            bool(causal),
            int(fwd_bm),
            int(fwd_bn),
            int(bwd_bm),
            int(bwd_bn),
            int(fwd_stages),
            int(bwd_stages),
            int(fwd_threads),
            int(bwd_threads),
        )

    (
        bias_buf,
        mask_buf,
        qseg_buf,
        kvseg_buf,
        aux_buf,
        seed_buf,
        use_bias,
        use_mask,
        use_segments,
        use_softmax_aux,
    ) = _prepare_full_features(
        query,
        bias=bias,
        attention_mask=attention_mask,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        dropout_seed=dropout_seed,
        dropout_key=dropout_key,
    )
    return _fa_full_core(
        query,
        key,
        value,
        bias_buf,
        mask_buf,
        qseg_buf,
        kvseg_buf,
        aux_buf,
        seed_buf,
        float(softmax_scale),
        bool(causal),
        None if logits_soft_cap is None else float(logits_soft_cap),
        int(num_kv_heads),
        bool(normalize_output),
        int(fwd_bm),
        int(fwd_bn),
        int(bwd_bm),
        int(bwd_bn),
        bool(use_bias),
        bool(use_mask),
        bool(use_segments),
        bool(use_softmax_aux),
        window,
        float(dropout_prob),
    )
