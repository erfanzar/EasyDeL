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

"""Native tile-lang RWKV-7 forward kernel.

RWKV-7 step (DPLR — Diagonal Plus Low-Rank):

    hb       = b @ h                              # (V,)
    h_next   = h * exp(w) + a outer hb + k outer v
    o        = r @ h_next                         # (V,)

Grid: ``(num_heads, batch)``. One CTA per ``(b, h)`` walks the time axis.
The ``(K, V)`` state lives in a fragment for the lifetime of the CTA.
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
        raise TypeError(f"Unsupported dtype for rwkv7: {dtype}")
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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the RWKV-7 forward ``@T.prim_func`` (no state materialisation).

    Grid: ``(num_heads, batch)`` — one CTA per ``(b, h)``.  The ``(K, V)``
    state matrix lives in a register fragment.  The DPLR update is::

        b_t = Bi_t  (standard) or b_t = -A_t, a_t = A_t * Bi_t  (mul_variant)
        hb      = sum_i b[i] * h[:, i]          (row-weighted state contraction)
        h_next  = h * exp(w) + a outer hb + k outer v
        o_t     = r @ h_next                     (after the state update)

    Note: the output ``o_t`` is computed from the **updated** ``h_next``,
    not ``h_prev``.  This matches the canonical RWKV-7 definition.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of attention heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        softmax_scale: scalar multiplied onto ``R``.
        reverse: if ``True`` iterate time in reverse.
        mul_variant: if ``True`` interpret inputs as ``(kk, a)`` and derive
            ``a_loc = kk * a``, ``b_loc = -kk`` (RWKV-7-mul parameterisation).
            If ``False`` interpret inputs directly as ``(a, b)``.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, W, K, V, A, Bi, H0, O, Hf)`` where ``A`` and ``Bi`` are
        ``(B, S, H, K, dtype)`` and their roles depend on ``mul_variant``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_fwd(
        R: T.Tensor((B, S, H, K_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        A: T.Tensor((B, S, H, K_), ts),
        Bi: T.Tensor((B, S, H, K_), ts),
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
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)

            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)

            ro_prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

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
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k]) * T.Cast(accum, Bi[bx, t, hx, k])
                        b_loc[k] = -T.Cast(accum, A[bx, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[bx, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[bx, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_state[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    h_state[i, j] = h_state[i, j] * decay[i] + a_loc[i] * hb[j] + k_loc[i] * v_loc[j]

                for i, j in T.Parallel(K_, V_):
                    ro_prod[i, j] = r_loc[i] * h_state[i, j]
                T.reduce_sum(ro_prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    O[bx, t, hx, j] = T.Cast(ts, o_loc[j])

            for i, j in T.Parallel(K_, V_):
                Hf[bx, hx, i, j] = h_state[i, j]

    return rwkv7_fwd


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
    """Build the RWKV-7 zero-state initialiser ``@T.prim_func`` (batched).

    Grid: ``(num_heads, batch)``. Writes a zero fp32 ``(B, H, K, V)`` state.
    ``seq_len`` is baked in only for dtype-probe allocation.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S`` (dtype-probe only).
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, H0)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_init_state(
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

    return rwkv7_init_state


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
    """Build the RWKV-7 zero-state initialiser for packed sequences.

    Grid: ``(num_heads, num_seqs)``. Writes zero fp32 ``(N, H, K, V)`` states.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total tokens ``TQ`` (dtype-probe only).
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, CuSeqLens, H0)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_packed_init_state(
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

    return rwkv7_packed_init_state


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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the RWKV-7 forward ``@T.prim_func`` that materialises all states.

    Identical recurrence to :func:`make_fwd_prim_func` but also writes
    ``Hscan[b, step+1, h, :, :]`` after each state update (index 0 stores
    the initial ``H0``). The ``Hscan`` tensor is fp32 ``(B, S+1, H, K, V)``
    and consumed by the backward kernel.

    Grid: ``(num_heads, batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: scalar applied to ``R``.
        reverse: if ``True`` iterate in reverse.
        mul_variant: see :func:`make_fwd_prim_func`.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(R, W, K, V, A, Bi, H0, O, Hf, Hscan)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_fwd_states(
        R: T.Tensor((B, S, H, K_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        A: T.Tensor((B, S, H, K_), ts),
        Bi: T.Tensor((B, S, H, K_), ts),
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
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)
            ro_prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

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
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k]) * T.Cast(accum, Bi[bx, t, hx, k])
                        b_loc[k] = -T.Cast(accum, A[bx, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[bx, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[bx, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_state[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    h_state[i, j] = h_state[i, j] * decay[i] + a_loc[i] * hb[j] + k_loc[i] * v_loc[j]
                    Hscan[bx, step + 1, hx, i, j] = h_state[i, j]

                for i, j in T.Parallel(K_, V_):
                    ro_prod[i, j] = r_loc[i] * h_state[i, j]
                T.reduce_sum(ro_prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    O[bx, t, hx, j] = T.Cast(ts, o_loc[j])

            for i, j in T.Parallel(K_, V_):
                Hf[bx, hx, i, j] = h_state[i, j]

    return rwkv7_fwd_states


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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the RWKV-7 packed-sequence forward ``@T.prim_func``.

    Each CTA covers one ``(sequence, head)`` pair.  Tokens beyond the
    sequence's valid length are skipped via an ``active`` predicate on writes.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ`` (loop bound).
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: scalar applied to ``R``.
        reverse: if ``True`` iterate in reverse within each sequence.
        mul_variant: see :func:`make_fwd_prim_func`.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, W, K, V, A, Bi, CuSeqLens, H0, O, Hf)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_packed_fwd(
        R: T.Tensor((1, TQ, H, K_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        A: T.Tensor((1, TQ, H, K_), ts),
        Bi: T.Tensor((1, TQ, H, K_), ts),
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
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)
            ro_prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

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
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k]) * T.Cast(accum, Bi[0, t, hx, k])
                        b_loc[k] = -T.Cast(accum, A[0, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[0, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[0, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_state[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    if active:
                        h_state[i, j] = h_state[i, j] * decay[i] + a_loc[i] * hb[j] + k_loc[i] * v_loc[j]

                for i, j in T.Parallel(K_, V_):
                    ro_prod[i, j] = r_loc[i] * h_state[i, j]
                T.reduce_sum(ro_prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        O[0, t, hx, j] = T.Cast(ts, o_loc[j])

            for i, j in T.Parallel(K_, V_):
                Hf[nx, hx, i, j] = h_state[i, j]

    return rwkv7_packed_fwd


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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the packed RWKV-7 forward ``@T.prim_func`` with state materialisation.

    Extends :func:`make_packed_fwd_prim_func` by also writing
    ``Hscan[n, step+1, h, :, :]`` at each active step.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: scalar applied to ``R``.
        reverse: if ``True`` iterate in reverse.
        mul_variant: see :func:`make_fwd_prim_func`.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, W, K, V, A, Bi, CuSeqLens, H0, O, Hf, Hscan)`` where
        ``Hscan`` is fp32 ``(N, TQ+1, H, K, V)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_packed_fwd_states(
        R: T.Tensor((1, TQ, H, K_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        A: T.Tensor((1, TQ, H, K_), ts),
        Bi: T.Tensor((1, TQ, H, K_), ts),
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
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)
            ro_prod = T.alloc_fragment((K_, V_), accum)
            o_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

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
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k]) * T.Cast(accum, Bi[0, t, hx, k])
                        b_loc[k] = -T.Cast(accum, A[0, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[0, t, hx, k])
                for d in T.Parallel(V_):
                    v_loc[d] = T.Cast(accum, V[0, t, hx, d])

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_state[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for k in T.Parallel(K_):
                    decay[k] = T.exp(w_loc[k])
                for i, j in T.Parallel(K_, V_):
                    if active:
                        h_state[i, j] = h_state[i, j] * decay[i] + a_loc[i] * hb[j] + k_loc[i] * v_loc[j]
                        Hscan[nx, step + 1, hx, i, j] = h_state[i, j]

                for i, j in T.Parallel(K_, V_):
                    ro_prod[i, j] = r_loc[i] * h_state[i, j]
                T.reduce_sum(ro_prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        O[0, t, hx, j] = T.Cast(ts, o_loc[j])

            for i, j in T.Parallel(K_, V_):
                Hf[nx, hx, i, j] = h_state[i, j]

    return rwkv7_packed_fwd_states


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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the RWKV-7 backward ``@T.prim_func`` (reverse-time adjoint scan).

    Walks time from ``S-1`` down to ``0`` using the materialised state
    trajectory ``Hscan``. Computes gradients for all inputs and writes them
    to the output buffers without atomics (all writes are keyed by ``(b, h, t)``
    or ``(b, h)``).

    When ``mul_variant=True`` the chain rule is applied through the
    ``a_loc = A * Bi``, ``b_loc = -A`` transformation so that ``dA`` and
    ``dBi`` are the gradients w.r.t. the original ``(A, Bi)`` inputs.

    Grid: ``(num_heads, batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: same value used in the forward pass.
        reverse: must match the forward-pass ``reverse`` flag.
        mul_variant: must match the forward-pass ``mul_variant`` flag.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, W, K, V, A, Bi, Hscan, dO, dHf, dR, dW, dK, dV, dA, dBi, dH0)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, K_, V_ = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_bwd(
        R: T.Tensor((B, S, H, K_), ts),
        W: T.Tensor((B, S, H, K_), ts),
        K: T.Tensor((B, S, H, K_), ts),
        V: T.Tensor((B, S, H, V_), ts),
        A: T.Tensor((B, S, H, K_), ts),
        Bi: T.Tensor((B, S, H, K_), ts),
        Hscan: T.Tensor((B, S + 1, H, K_, V_), accum),
        dO: T.Tensor((B, S, H, V_), ts),
        dHf: T.Tensor((B, H, K_, V_), accum),
        dR: T.Tensor((B, S, H, K_), ts),
        dW: T.Tensor((B, S, H, K_), ts),
        dK: T.Tensor((B, S, H, K_), ts),
        dV: T.Tensor((B, S, H, V_), ts),
        dA: T.Tensor((B, S, H, K_), ts),
        dBi: T.Tensor((B, S, H, K_), ts),
        dH0: T.Tensor((B, H, K_, V_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            dh = T.alloc_fragment((K_, V_), accum)
            dh_prev = T.alloc_fragment((K_, V_), accum)
            h_cur = T.alloc_fragment((K_, V_), accum)
            h_prev = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            a_raw = T.alloc_fragment((K_,), accum)
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            do_loc = T.alloc_fragment((V_,), accum)
            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)
            dhb_prod = T.alloc_fragment((K_, V_), accum)
            dhb = T.alloc_fragment((V_,), accum)
            d_vec_prod = T.alloc_fragment((K_, V_), accum)
            dk_loc = T.alloc_fragment((K_,), accum)
            da_loc = T.alloc_fragment((K_,), accum)
            db_loc = T.alloc_fragment((K_,), accum)
            dw_loc = T.alloc_fragment((K_,), accum)
            dr_loc = T.alloc_fragment((K_,), accum)
            dv_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, j in T.Parallel(K_, V_):
                dh[i, j] = dHf[bx, hx, i, j]

            for step_rev in T.serial(S):
                step = S - 1 - step_rev
                if reverse:
                    t = S - 1 - step
                else:
                    t = step

                for i, j in T.Parallel(K_, V_):
                    h_prev[i, j] = Hscan[bx, step, hx, i, j]
                    h_cur[i, j] = Hscan[bx, step + 1, hx, i, j]
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[bx, t, hx, k])
                    k_loc[k] = T.Cast(accum, K[bx, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[bx, t, hx, k])
                    a_raw[k] = T.Cast(accum, Bi[bx, t, hx, k])
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k]) * a_raw[k]
                        b_loc[k] = -T.Cast(accum, A[bx, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[bx, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[bx, t, hx, k])
                for j in T.Parallel(V_):
                    v_loc[j] = T.Cast(accum, V[bx, t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[bx, t, hx, j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_cur[i, j] * do_loc[j]
                T.reduce_sum(d_vec_prod, dr_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    dR[bx, t, hx, i] = T.Cast(ts, dr_loc[i] * softmax_scale)

                for i, j in T.Parallel(K_, V_):
                    dh[i, j] = dh[i, j] + r_loc[i] * softmax_scale * do_loc[j]

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_prev[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for i, j in T.Parallel(K_, V_):
                    dhb_prod[i, j] = dh[i, j] * a_loc[i]
                T.reduce_sum(dhb_prod, dhb, dim=0, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * v_loc[j]
                T.reduce_sum(d_vec_prod, dk_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    dK[bx, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * k_loc[i]
                T.reduce_sum(d_vec_prod, dv_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    dV[bx, t, hx, j] = T.Cast(ts, dv_loc[j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * hb[j]
                T.reduce_sum(d_vec_prod, da_loc, dim=1, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_prev[i, j] * dhb[j]
                T.reduce_sum(d_vec_prod, db_loc, dim=1, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh[i, j] * h_prev[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    decay[i] = T.exp(w_loc[i])
                    dW[bx, t, hx, i] = T.Cast(ts, dw_loc[i] * decay[i])
                    if mul_variant:
                        dA[bx, t, hx, i] = T.Cast(ts, da_loc[i] * a_raw[i] - db_loc[i])
                        dBi[bx, t, hx, i] = T.Cast(ts, da_loc[i] * T.Cast(accum, A[bx, t, hx, i]))
                    else:
                        dA[bx, t, hx, i] = T.Cast(ts, da_loc[i])
                        dBi[bx, t, hx, i] = T.Cast(ts, db_loc[i])

                for i, j in T.Parallel(K_, V_):
                    dh_prev[i, j] = dh[i, j] * decay[i] + b_loc[i] * dhb[j]
                for i, j in T.Parallel(K_, V_):
                    dh[i, j] = dh_prev[i, j]

            for i, j in T.Parallel(K_, V_):
                dH0[bx, hx, i, j] = dh[i, j]

    return rwkv7_bwd


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
    mul_variant: bool,
    threads: int = 128,
):
    """Build the RWKV-7 packed-sequence backward ``@T.prim_func``.

    Mirrors :func:`make_bwd_prim_func` for the packed case. All gradient
    writes are predicated on ``active`` to skip steps beyond the sequence's
    true length.

    Grid: ``(num_heads, num_seqs)``.

    Args:
        num_seqs: number of packed sequences ``N``.
        total_tokens: total token count ``TQ``.
        num_heads: number of heads ``H``.
        qk_head_dim: Q/K head dimension ``K``.
        v_head_dim: V head dimension ``V``.
        dtype: tensor dtype.
        softmax_scale: same value used in the forward pass.
        reverse: must match the forward-pass ``reverse`` flag.
        mul_variant: must match the forward-pass ``mul_variant`` flag.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(R, W, K, V, A, Bi, CuSeqLens, Hscan, dO, dHf,
        dR, dW, dK, dV, dA, dBi, dH0)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, K_, V_ = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rwkv7_packed_bwd(
        R: T.Tensor((1, TQ, H, K_), ts),
        W: T.Tensor((1, TQ, H, K_), ts),
        K: T.Tensor((1, TQ, H, K_), ts),
        V: T.Tensor((1, TQ, H, V_), ts),
        A: T.Tensor((1, TQ, H, K_), ts),
        Bi: T.Tensor((1, TQ, H, K_), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        Hscan: T.Tensor((N, TQ + 1, H, K_, V_), accum),
        dO: T.Tensor((1, TQ, H, V_), ts),
        dHf: T.Tensor((N, H, K_, V_), accum),
        dR: T.Tensor((1, TQ, H, K_), ts),
        dW: T.Tensor((1, TQ, H, K_), ts),
        dK: T.Tensor((1, TQ, H, K_), ts),
        dV: T.Tensor((1, TQ, H, V_), ts),
        dA: T.Tensor((1, TQ, H, K_), ts),
        dBi: T.Tensor((1, TQ, H, K_), ts),
        dH0: T.Tensor((N, H, K_, V_), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            dh = T.alloc_fragment((K_, V_), accum)
            dh_work = T.alloc_fragment((K_, V_), accum)
            dh_prev = T.alloc_fragment((K_, V_), accum)
            h_cur = T.alloc_fragment((K_, V_), accum)
            h_prev = T.alloc_fragment((K_, V_), accum)
            r_loc = T.alloc_fragment((K_,), accum)
            k_loc = T.alloc_fragment((K_,), accum)
            v_loc = T.alloc_fragment((V_,), accum)
            w_loc = T.alloc_fragment((K_,), accum)
            a_raw = T.alloc_fragment((K_,), accum)
            a_loc = T.alloc_fragment((K_,), accum)
            b_loc = T.alloc_fragment((K_,), accum)
            decay = T.alloc_fragment((K_,), accum)
            do_loc = T.alloc_fragment((V_,), accum)
            hb_prod = T.alloc_fragment((K_, V_), accum)
            hb = T.alloc_fragment((V_,), accum)
            dhb_prod = T.alloc_fragment((K_, V_), accum)
            dhb = T.alloc_fragment((V_,), accum)
            d_vec_prod = T.alloc_fragment((K_, V_), accum)
            dk_loc = T.alloc_fragment((K_,), accum)
            da_loc = T.alloc_fragment((K_,), accum)
            db_loc = T.alloc_fragment((K_,), accum)
            dw_loc = T.alloc_fragment((K_,), accum)
            dr_loc = T.alloc_fragment((K_,), accum)
            dv_loc = T.alloc_fragment((V_,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            for i, j in T.Parallel(K_, V_):
                dh[i, j] = dHf[nx, hx, i, j]

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
                    h_cur[i, j] = Hscan[nx, scan_pos + 1, hx, i, j]
                for k in T.Parallel(K_):
                    r_loc[k] = T.Cast(accum, R[0, t, hx, k])
                    k_loc[k] = T.Cast(accum, K[0, t, hx, k])
                    w_loc[k] = T.Cast(accum, W[0, t, hx, k])
                    a_raw[k] = T.Cast(accum, Bi[0, t, hx, k])
                    if mul_variant:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k]) * a_raw[k]
                        b_loc[k] = -T.Cast(accum, A[0, t, hx, k])
                    else:
                        a_loc[k] = T.Cast(accum, A[0, t, hx, k])
                        b_loc[k] = T.Cast(accum, Bi[0, t, hx, k])
                for j in T.Parallel(V_):
                    v_loc[j] = T.Cast(accum, V[0, t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[0, t, hx, j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_cur[i, j] * do_loc[j]
                T.reduce_sum(d_vec_prod, dr_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    if active:
                        dR[0, t, hx, i] = T.Cast(ts, dr_loc[i] * softmax_scale)

                for i, j in T.Parallel(K_, V_):
                    dh_work[i, j] = dh[i, j] + r_loc[i] * softmax_scale * do_loc[j]

                for i, j in T.Parallel(K_, V_):
                    hb_prod[i, j] = b_loc[i] * h_prev[i, j]
                T.reduce_sum(hb_prod, hb, dim=0, clear=True)

                for i, j in T.Parallel(K_, V_):
                    dhb_prod[i, j] = dh_work[i, j] * a_loc[i]
                T.reduce_sum(dhb_prod, dhb, dim=0, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_work[i, j] * v_loc[j]
                T.reduce_sum(d_vec_prod, dk_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    if active:
                        dK[0, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_work[i, j] * k_loc[i]
                T.reduce_sum(d_vec_prod, dv_loc, dim=0, clear=True)
                for j in T.Parallel(V_):
                    if active:
                        dV[0, t, hx, j] = T.Cast(ts, dv_loc[j])

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_work[i, j] * hb[j]
                T.reduce_sum(d_vec_prod, da_loc, dim=1, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = h_prev[i, j] * dhb[j]
                T.reduce_sum(d_vec_prod, db_loc, dim=1, clear=True)

                for i, j in T.Parallel(K_, V_):
                    d_vec_prod[i, j] = dh_work[i, j] * h_prev[i, j]
                T.reduce_sum(d_vec_prod, dw_loc, dim=1, clear=True)
                for i in T.Parallel(K_):
                    decay[i] = T.exp(w_loc[i])
                    if active:
                        dW[0, t, hx, i] = T.Cast(ts, dw_loc[i] * decay[i])
                        if mul_variant:
                            dA[0, t, hx, i] = T.Cast(ts, da_loc[i] * a_raw[i] - db_loc[i])
                            dBi[0, t, hx, i] = T.Cast(ts, da_loc[i] * T.Cast(accum, A[0, t, hx, i]))
                        else:
                            dA[0, t, hx, i] = T.Cast(ts, da_loc[i])
                            dBi[0, t, hx, i] = T.Cast(ts, db_loc[i])

                for i, j in T.Parallel(K_, V_):
                    dh_prev[i, j] = dh_work[i, j] * decay[i] + b_loc[i] * dhb[j]
                for i, j in T.Parallel(K_, V_):
                    if active:
                        dh[i, j] = dh_prev[i, j]

            for i, j in T.Parallel(K_, V_):
                dH0[nx, hx, i, j] = dh[i, j]

    return rwkv7_packed_bwd
