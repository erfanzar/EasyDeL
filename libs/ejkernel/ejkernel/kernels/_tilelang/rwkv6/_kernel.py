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

"""Native tile-lang RWKV-6 forward kernel.

Per-step recurrence (matches the XLA reference):

    kv      = k_t outer v_t                  # (K, V)
    o_t     = r_t @ (h + kv * u)             # (V,)  — u is the per-head bonus
    h_next  = h * exp(w_t) + kv              # (K, V)

Grid: ``(num_heads, batch)`` — one CTA per ``(b, h)`` walks the time axis
sequentially. The ``(K, V)`` state lives in a fragment for the lifetime of
the CTA.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the TileLang dtype string for a supported floating-point dtype.

    Args:
        dtype: any dtype specifier accepted by ``jnp.dtype``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: if ``dtype`` is not one of the three supported types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for rwkv6: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the RWKV-6 forward ``@T.prim_func`` (no state materialisation).

    Grid: ``(num_heads, batch)`` — one CTA per ``(b, h)``. The ``(K, V)``
    state matrix is held in a register fragment for the CTA's lifetime and
    the time axis is walked sequentially (forward or reverse).

    When ``reverse=True`` the index mapping is ``t = S - 1 - step`` so the
    recurrence runs in reverse time, which is used for the non-causal scan.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of attention heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        softmax_scale: scalar multiplied onto ``R`` before the inner product.
        reverse: if ``True`` iterate time in reverse (``t = S-1-step``).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, K, V, W, U, H0, O, Hf)`` where:
        ``R/K/V/W`` are ``(B,S,H,K or V, dtype)``,
        ``U`` is ``(H, K, dtype)``,
        ``H0/Hf`` are fp32 ``(B, H, K, V)``,
        ``O`` is ``(B, S, H, V, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_fwd(
        R: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        H0: T.Tensor((B, H, K_, V_), accum),
        O: T.Tensor((B, S, H, V_), ts),
        Hf: T.Tensor((B, H, K_, V_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            h_state = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for k in T.Parallel(K_):
                u_loc[k] = T.Cast(accum, U[hx, k])
            for i, j in T.Parallel(K_, V_):
                h_state[i, j] = H0[bx, hx, i, j]

            for step in T.serial(S):
                if reverse:
                    t = S - 1 - step
                else:
                    t = step
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[bx, t, hx, k]) * softmax_scale
                    k_loc[k] = T.Cast(accum, K[bx, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[bx, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[bx, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]

                for i, j in T.Parallel(K_, V_):
                    h_bonus[i, j] = h_state[i, j] + kv[i, j] * u_loc[i]

                for i, j in T.Parallel(K_, V_):
                    prod[i, j] = r_loc[i] * h_bonus[i, j]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    O[bx, t, hx, j] = T.Cast(ts, o_loc[j])

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    h_state[i, j] = h_state[i, j] * decay[i] + kv[i, j]

            for i, j in T.Parallel(K_, V_):
                Hf[bx, hx, i, j] = h_state[i, j]

    return rwkv6_fwd


def make_init_state_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-6 zero-state initialiser ``@T.prim_func`` (batched).

    Grid: ``(num_heads, batch)``. Writes a zero fp32 ``(B, H, K, V)``
    initial state.  ``seq_len`` is baked in only for dtype-probe allocation.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S`` (used for dtype-probe only).
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, H0)``; ``R`` is read-only and
        used for type inference only.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_init_state(
        R: T.Tensor((B, S, H, K_), ts),
        H0: T.Tensor((B, H, K_, V_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = R[0, 0, 0, 0]
            _seq_ref[0] = S
            for i, j in T.Parallel(K_, V_):
                H0[bx, hx, i, j] = 0.0

    return rwkv6_init_state


def make_packed_init_state_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-6 zero-state initialiser for packed (ragged) sequences.

    Grid: ``(num_heads, num_seqs)``. Writes a zero fp32 ``(N, H, K, V)``
    initial state, one entry per sequence.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count across all sequences ``TQ``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, CuSeqLens, H0)`` where
        ``CuSeqLens`` is int32 ``(N+1,)`` (cumulative sequence offsets, used
        for dtype-probe only here; actual offsets matter in the forward kernel).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_packed_init_state(
        R: T.Tensor((1, TQ, H, K_), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        H0: T.Tensor((N, H, K_, V_), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), "int32")
            _ts_ref[0] = R[0, 0, 0, 0]
            _seq_ref[0] = CuSeqLens[nx + 1] - CuSeqLens[nx]
            for i, j in T.Parallel(K_, V_):
                H0[nx, hx, i, j] = 0.0

    return rwkv6_packed_init_state


def make_fwd_states_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the RWKV-6 forward ``@T.prim_func`` that materialises all states.

    Same recurrence as :func:`make_fwd_prim_func` but additionally writes
    ``Hscan[b, step+1, h, :, :]`` at each time-step (index 0 = initial state
    from ``H0``). The ``Hscan`` tensor is fp32 ``(B, S+1, H, K, V)`` and is
    used by the backward kernel.

    Grid: ``(num_heads, batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        softmax_scale: scalar multiplied onto ``R``.
        reverse: if ``True`` iterate time in reverse.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, K, V, W, U, H0, O, Hf, Hscan)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_fwd_states(
        R: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        H0: T.Tensor((B, H, K_, V_), accum),
        O: T.Tensor((B, S, H, V_), ts),
        Hf: T.Tensor((B, H, K_, V_), accum),
        Hscan: T.Tensor((B, S + 1, H, K_, V_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            h_state = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for k in T.Parallel(K_):
                u_loc[k] = T.Cast(accum, U[hx, k])
            for i, j in T.Parallel(K_, V_):
                h_state[i, j] = H0[bx, hx, i, j]
                Hscan[bx, 0, hx, i, j] = h_state[i, j]

            for step in T.serial(S):
                if reverse:
                    t = S - 1 - step
                else:
                    t = step
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[bx, t, hx, k]) * softmax_scale
                    k_loc[k] = T.Cast(accum, K[bx, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[bx, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[bx, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]
                for i, j in T.Parallel(K_, V_):
                    h_bonus[i, j] = h_state[i, j] + kv[i, j] * u_loc[i]
                for i, j in T.Parallel(K_, V_):
                    prod[i, j] = r_loc[i] * h_bonus[i, j]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    O[bx, t, hx, j] = T.Cast(ts, o_loc[j])

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    h_state[i, j] = h_state[i, j] * decay[i] + kv[i, j]
                    Hscan[bx, step + 1, hx, i, j] = h_state[i, j]

            for i, j in T.Parallel(K_, V_):
                Hf[bx, hx, i, j] = h_state[i, j]

    return rwkv6_fwd_states


def make_packed_fwd_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the RWKV-6 packed-sequence forward ``@T.prim_func``.

    Handles variable-length sequences packed into a single token dimension
    (no padding). Each CTA corresponds to one ``(sequence, head)`` pair.
    Tokens outside the sequence's valid range (``step >= seq_len``) are
    skipped via predicated writes.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ`` (= ``max_seq_len`` in the kernel
            loop, actual per-seq length is read from ``CuSeqLens``).
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: scalar multiplied onto ``R``.
        reverse: if ``True`` iterate within each sequence in reverse.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, K, V, W, U, CuSeqLens, H0, O, Hf)`` where tensors have a
        leading batch-of-1 dim: ``R/K/V/W/O`` are ``(1, TQ, H, K/V, dtype)``;
        ``H0/Hf`` are ``(N, H, K, V, fp32)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_packed_fwd(
        R: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        H0: T.Tensor((N, H, K_, V_), accum),
        O: T.Tensor((1, TQ, H, V_), ts),
        Hf: T.Tensor((N, H, K_, V_), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            h_state = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            for k in T.Parallel(K_):
                u_loc[k] = T.Cast(accum, U[hx, k])
            for i, j in T.Parallel(K_, V_):
                h_state[i, j] = H0[nx, hx, i, j]

            for step in T.serial(TQ):
                active = step < seq_len
                if reverse:
                    raw_t = end - 1 - step
                else:
                    raw_t = start + step
                t = T.if_then_else(active, raw_t, 0)

                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[0, t, hx, k]) * softmax_scale
                    k_loc[k] = T.Cast(accum, K[0, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[0, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[0, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]
                    h_bonus[i, j] = h_state[i, j] + kv[i, j] * u_loc[i]
                    prod[i, j] = r_loc[i] * h_bonus[i, j]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        O[0, t, hx, j] = T.Cast(ts, o_loc[j])

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    if active:
                        h_state[i, j] = h_state[i, j] * decay[i] + kv[i, j]

            for i, j in T.Parallel(K_, V_):
                Hf[nx, hx, i, j] = h_state[i, j]

    return rwkv6_packed_fwd


def make_packed_fwd_states_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the packed RWKV-6 forward ``@T.prim_func`` with state materialisation.

    Extends :func:`make_packed_fwd_prim_func` by also writing
    ``Hscan[n, step+1, h, :, :]`` at each active step. Steps beyond the
    sequence's ``seq_len`` are skipped via the ``active`` predicate.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: scalar multiplied onto ``R``.
        reverse: if ``True`` iterate in reverse within each sequence.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, K, V, W, U, CuSeqLens, H0, O, Hf, Hscan)`` where
        ``Hscan`` is fp32 ``(N, TQ+1, H, K, V)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_packed_fwd_states(
        R: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        H0: T.Tensor((N, H, K_, V_), accum),
        O: T.Tensor((1, TQ, H, V_), ts),
        Hf: T.Tensor((N, H, K_, V_), accum),
        Hscan: T.Tensor((N, TQ + 1, H, K_, V_), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            h_state = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            for k in T.Parallel(K_):
                u_loc[k] = T.Cast(accum, U[hx, k])
            for i, j in T.Parallel(K_, V_):
                h_state[i, j] = H0[nx, hx, i, j]
                Hscan[nx, 0, hx, i, j] = h_state[i, j]

            for step in T.serial(TQ):
                active = step < seq_len
                if reverse:
                    raw_t = end - 1 - step
                else:
                    raw_t = start + step
                t = T.if_then_else(active, raw_t, 0)

                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[0, t, hx, k]) * softmax_scale
                    k_loc[k] = T.Cast(accum, K[0, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[0, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[0, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]
                    h_bonus[i, j] = h_state[i, j] + kv[i, j] * u_loc[i]
                    prod[i, j] = r_loc[i] * h_bonus[i, j]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        O[0, t, hx, j] = T.Cast(ts, o_loc[j])

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    if active:
                        h_state[i, j] = h_state[i, j] * decay[i] + kv[i, j]
                        Hscan[nx, step + 1, hx, i, j] = h_state[i, j]

            for i, j in T.Parallel(K_, V_):
                Hf[nx, hx, i, j] = h_state[i, j]

    return rwkv6_packed_fwd_states


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the RWKV-6 backward ``@T.prim_func`` (reverse-time adjoint scan).

    Walks time in reverse from ``S-1`` to ``0``, computing adjoint state
    ``dh`` and writing per-timestep gradients ``dR, dK, dV, dW``. The bonus
    gradient ``dU`` is accumulated per-head per-batch into ``dU_p`` and
    reduced over batch in the JAX glue.

    Grid: ``(num_heads, batch)``. No atomics — all output slices are keyed
    by ``(b, h, t)`` or ``(b, h)`` and are strictly disjoint.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: same scale used during the forward pass.
        reverse: must match the ``reverse`` flag used in the forward pass.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, K, V, W, U, Hscan, dO, dHf, dR, dK, dV, dW, dU_p, dH0)``
        where ``dU_p`` is fp32 ``(B, H, K)`` (partial, to be batch-reduced).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_bwd(
        R: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        Hscan: T.Tensor((B, S + 1, H, K_, V_), accum),
        dO: T.Tensor((B, S, H, V_), ts),
        dHf: T.Tensor((B, H, K_, V_), accum),
        dR: T.Tensor((B, S, H, K_), ts),
        dK: T.Tensor((B, S, H, K_), ts),
        dV: T.Tensor((B, S, H, V_), ts),
        dW: T.Tensor((B, S, H, K_), ts),
        dU_p: T.Tensor((B, H, K_), accum),
        dH0: T.Tensor((B, H, K_, V_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            dh = T.alloc_fragment((K_, V_), accum)
            dh_prev = T.alloc_fragment((K_, V_), accum)
            h_prev = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            do_loc = T.alloc_fragment((V_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            dkv = T.alloc_fragment((K_, V_), accum)
            dh_bonus = T.alloc_fragment((K_, V_), accum)
            d_vec_prod = T.alloc_fragment((K_, V_), accum)
            dr_loc = T.alloc_fragment((K_,), accum)
            dk_loc = T.alloc_fragment((K_,), accum)
            dw_loc = T.alloc_fragment((K_,), accum)
            du_loc = T.alloc_fragment((K_,), accum)
            dv_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, j in T.Parallel(K_, V_):
                dh[i, j] = dHf[bx, hx, i, j]
            for i in T.Parallel(K_):
                u_loc[i] = T.Cast(accum, U[hx, i])
                du_loc[i] = 0.0

            for step_rev in T.serial(S):
                step = S - 1 - step_rev
                if reverse:
                    t = S - 1 - step
                else:
                    t = step

                for i, j in T.Parallel(K_, V_):
                    h_prev[i, j] = Hscan[bx, step, hx, i, j]
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[bx, t, hx, k])
                    k_loc[k] = T.Cast(accum, K[bx, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[bx, t, hx, k])
                for j in T.Parallel(V_):
                    v_loc[j] = T.Cast(accum, V[bx, t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[bx, t, hx, j])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]
                    h_bonus[i, j] = h_prev[i, j] + kv[i, j] * u_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_bonus[i, j] * do_loc[j]
                T.reduce_sum(d_vec_prod, dr_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    dR[bx, t, hx, i] = T.Cast(ts, dr_loc[i] * softmax_scale)

                for i, j in T.Parallel(K_, V_):
                    dh_bonus[i, j] = r_loc[i] * softmax_scale * do_loc[j]
                    dkv[i, j] = dh[i, j] + dh_bonus[i, j] * u_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_bonus[i, j] * kv[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    du_loc[i] = du_loc[i] + dw_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dkv[i, j] * v_loc[j]
                T.reduce_sum(d_vec_prod, dk_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    dK[bx, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dkv[i, j] * k_loc[i]
                T.reduce_sum(d_vec_prod, dv_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    dV[bx, t, hx, j] = T.Cast(ts, dv_loc[j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * h_prev[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    decay[i] = T.exp(w_loc[i])
                    dW[bx, t, hx, i] = T.Cast(ts, dw_loc[i] * decay[i])

                for i, j in T.Parallel(K_, V_):
                    dh_prev[i, j] = dh[i, j] * decay[i] + dh_bonus[i, j]
                for i, j in T.Parallel(K_, V_):
                    dh[i, j] = dh_prev[i, j]

            for i in T.Parallel(K_):
                dU_p[bx, hx, i] = du_loc[i]
            for i, j in T.Parallel(K_, V_):
                dH0[bx, hx, i, j] = dh[i, j]

    return rwkv6_bwd


def make_packed_bwd_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    softmax_scale: float,
    reverse: bool,
    threads: int = 128,
):
    """Build the RWKV-6 packed-sequence backward ``@T.prim_func``.

    Mirrors :func:`make_bwd_prim_func` but for packed sequences: iterates
    in reverse over the padded ``TQ`` dimension and applies an ``active``
    predicate to skip tokens beyond the sequence's true length.  Gradient
    writes are predicated on ``active``.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: same scale used in the forward pass.
        reverse: must match the ``reverse`` flag used in the forward pass.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, K, V, W, U, CuSeqLens, Hscan, dO, dHf, dR, dK, dV, dW, dU_p, dH0)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv6_packed_bwd(
        R: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        U: T.Tensor((H, K_), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        Hscan: T.Tensor((N, TQ + 1, H, K_, V_), accum),
        dO: T.Tensor((1, TQ, H, V_), ts),
        dHf: T.Tensor((N, H, K_, V_), accum),
        dR: T.Tensor((1, TQ, H, K_), ts),
        dK: T.Tensor((1, TQ, H, K_), ts),
        dV: T.Tensor((1, TQ, H, V_), ts),
        dW: T.Tensor((1, TQ, H, K_), ts),
        dU_p: T.Tensor((N, H, K_), accum),
        dH0: T.Tensor((N, H, K_, V_), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            dh = T.alloc_fragment((K_, V_), accum)
            dh_prev = T.alloc_fragment((K_, V_), accum)
            h_prev = T.alloc_fragment((K_, V_), accum)
            h_bonus = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            u_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            do_loc = T.alloc_fragment((V_,), accum)
            kv = T.alloc_fragment((K_, V_), accum)
            dkv = T.alloc_fragment((K_, V_), accum)
            dh_bonus = T.alloc_fragment((K_, V_), accum)
            d_vec_prod = T.alloc_fragment((K_, V_), accum)
            dr_loc = T.alloc_fragment((K_,), accum)
            dk_loc = T.alloc_fragment((K_,), accum)
            dw_loc = T.alloc_fragment((K_,), accum)
            du_loc = T.alloc_fragment((K_,), accum)
            dv_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            for i, j in T.Parallel(K_, V_):
                dh[i, j] = dHf[nx, hx, i, j]
            for i in T.Parallel(K_):
                u_loc[i] = T.Cast(accum, U[hx, i])
                du_loc[i] = 0.0

            for step_rev in T.serial(TQ):
                active = step_rev < seq_len
                step = seq_len - 1 - step_rev
                if reverse:
                    raw_t = end - 1 - step
                else:
                    raw_t = start + step
                t = T.if_then_else(active, raw_t, 0)
                scan_pos = T.if_then_else(active, step, 0)

                for i, j in T.Parallel(K_, V_):
                    h_prev[i, j] = Hscan[nx, scan_pos, hx, i, j]
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[0, t, hx, k])
                    k_loc[k] = T.Cast(accum, K[0, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[0, t, hx, k])
                for j in T.Parallel(V_):
                    v_loc[j] = T.Cast(accum, V[0, t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[0, t, hx, j])

                for i, j in T.Parallel(K_, V_):
                    kv[i, j] = k_loc[i] * v_loc[j]
                    h_bonus[i, j] = h_prev[i, j] + kv[i, j] * u_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_bonus[i, j] * do_loc[j]
                T.reduce_sum(d_vec_prod, dr_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    if active:
                        dR[0, t, hx, i] = T.Cast(ts, dr_loc[i] * softmax_scale)

                for i, j in T.Parallel(K_, V_):
                    dh_bonus[i, j] = r_loc[i] * softmax_scale * do_loc[j]
                    dkv[i, j] = dh[i, j] + dh_bonus[i, j] * u_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_bonus[i, j] * kv[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    if active:
                        du_loc[i] = du_loc[i] + dw_loc[i]

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dkv[i, j] * v_loc[j]
                T.reduce_sum(d_vec_prod, dk_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    if active:
                        dK[0, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dkv[i, j] * k_loc[i]
                T.reduce_sum(d_vec_prod, dv_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        dV[0, t, hx, j] = T.Cast(ts, dv_loc[j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * h_prev[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    decay[i] = T.exp(w_loc[i])
                    if active:
                        dW[0, t, hx, i] = T.Cast(ts, dw_loc[i] * decay[i])

                for i, j in T.Parallel(K_, V_):
                    dh_prev[i, j] = dh[i, j] * decay[i] + dh_bonus[i, j]
                for i, j in T.Parallel(K_, V_):
                    if active:
                        dh[i, j] = dh_prev[i, j]

            for i in T.Parallel(K_):
                dU_p[nx, hx, i] = du_loc[i]
            for i, j in T.Parallel(K_, V_):
                dH0[nx, hx, i, j] = dh[i, j]

    return rwkv6_packed_bwd


def make_reduce_du_prim_func(
    *,
    batch: int,
    num_heads: int,
    qk_head_dim: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-6 ``dU`` batch-reduce ``@T.prim_func``.

    Sums ``dU_p`` over the batch dimension:
    ``dU[h, k] = sum_b dU_p[b, h, k]``.

    Grid: ``(num_heads, qk_head_dim)``.

    Args:
        batch: batch size (or number of sequences for packed mode) ``B``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        dtype: output dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(dU_p, dU)`` where
        ``dU_p`` is fp32 ``(B, H, K)`` and ``dU`` is ``(H, K, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, K_ = batch, num_heads, qk_head_dim

    @T.prim_func
    def rwkv6_reduce_du(
        dU_p: T.Tensor((B, H, K_), accum),
        dU: T.Tensor((H, K_), ts),
    ):
        with T.Kernel(H, K_, threads=threads) as (hx, kx):
            total = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            total[0] = 0.0
            for b in T.serial(B):
                total[0] = total[0] + dU_p[b, hx, kx]
            dU[hx, kx] = T.Cast(ts, total[0])

    return rwkv6_reduce_du
