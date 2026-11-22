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

"""tile-lang prim_funcs for recurrent linear-attention recurrence (fwd + bwd).

Per-token forward update:
    ``h_{t+1} = h_t + k_t outer v_t``
    ``o_t = (h_{t+1}.T @ q_t) * softmax_scale``     (sums over qk_head_dim)

Backward (running in reverse, reconstructing ``h_t`` by subtraction — valid
only because v0 has no decay / gating):

    ``dh_acc += q_t outer dO_t * scale``  (then propagates backwards)
    ``dq_t = (h_{t+1} @ dO_t) * scale``
    ``dk_t = dh_acc @ v_t``
    ``dv_t = dh_acc.T @ k_t``
    ``h_t = h_{t+1} - k_t outer v_t``

Layout: ``(B, S, H, Dq)`` for ``q``/``k`` and ``(B, S, H, Dv)`` for ``v``,
matching the public XLA signature. Parallelism is across ``(B, H)`` only;
the time axis is sequential inside each CTA.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang recurrent: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    gamma_batch: int,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
    dtype,
    threads: int = 128,
):
    """Build the batched forward ``@T.prim_func`` for recurrent linear-attention.

    Grid: ``T.Kernel(num_heads, batch)``.  Each CTA maintains a ``(Dq, Dv)``
    fp32 hidden-state fragment and walks the sequence time axis sequentially.

    Accumulation is always fp32; the output is cast back to ``dtype`` on store.

    Args:
        batch: ``B``.
        seq_len: ``S``.
        num_heads: ``H`` (total Q heads).
        num_kv_heads: ``HK``; GQA group size ``G = H // HK`` is derived internally.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        gamma_batch: First dimension of ``GGamma``; 1 = broadcast.
        softmax_scale: Output scale multiplier.
        has_g / has_gk / has_gv / has_g_gamma: Gate-enable flags.
        use_static_gamma: If ``True``, use ``exp(slope * head_idx)`` for gamma.
        static_gamma_slope: Slope for static gamma.
        reverse: If ``True``, scan right-to-left.
        dtype: Compute dtype.
        threads: Threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` (``rec_fwd``) with buffers::

            Q, K, V, GDecay, GKey, GValue, H0, GGamma, O, Hf, HStates

        * ``H0``: initial state ``(B, H, Dq, Dv)`` fp32; zero-filled by the
          JAX glue when not supplied.
        * ``Hf``: final state ``(B, H, Dq, Dv)`` fp32.
        * ``HStates``: per-step scan buffer ``(B, S, H, Dq, Dv)`` fp32;
          required by the backward kernel.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, HK, Dq, Dv, GB = batch, seq_len, num_heads, num_kv_heads, qk_head_dim, v_head_dim, gamma_batch
    G = H // HK
    scale = float(softmax_scale)
    gamma_slope = float(static_gamma_slope)

    @T.prim_func
    def rec_fwd(
        Q: T.Tensor((B, S, H, Dq), ts),
        K: T.Tensor((B, S, HK, Dq), ts),
        V: T.Tensor((B, S, HK, Dv), ts),
        GDecay: T.Tensor((B, S, H, Dq), ts),
        GKey: T.Tensor((B, S, H, Dq), ts),
        GValue: T.Tensor((B, S, H, Dv), ts),
        H0: T.Tensor((B, H, Dq, Dv), accum),
        GGamma: T.Tensor((GB, H), accum),
        O: T.Tensor((B, S, H, Dv), ts),
        Hf: T.Tensor((B, H, Dq, Dv), accum),
        HStates: T.Tensor((B, S, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            kvh = hx // G
            h_state = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            g_loc = T.alloc_fragment((Dq,), accum)
            gk_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            gv_loc = T.alloc_fragment((Dv,), accum)
            o_loc = T.alloc_fragment((Dv,), accum)
            prod = T.alloc_fragment((Dq, Dv), accum)
            decay = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gb_ref = T.alloc_fragment((1,), accum)
            _hk_ref = T.alloc_fragment((1,), accum)
            _gb_ref[0] = GB
            _hk_ref[0] = HK
            _ts_ref[0] = Q[0, 0, 0, 0]
            decay[0] = 1.0
            if has_g_gamma:
                if use_static_gamma:
                    decay[0] = T.exp(gamma_slope * T.Cast(accum, hx))
                else:
                    decay[0] = T.exp(GGamma[T.if_then_else(GB == 1, 0, bx), hx])

            for i, j in T.Parallel(Dq, Dv):
                h_state[i, j] = H0[bx, hx, i, j]

            for t_iter in T.serial(S):
                t = T.if_then_else(reverse, S - 1 - t_iter, t_iter)
                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[bx, t, hx, i])
                    k_loc[i] = T.Cast(accum, K[bx, t, kvh, i])
                    g_loc[i] = 1.0
                    gk_loc[i] = 1.0
                    if has_g:
                        g_loc[i] = T.exp(T.Cast(accum, GDecay[bx, t, hx, i]))
                    if has_gk:
                        gk_loc[i] = T.exp(T.Cast(accum, GKey[bx, t, hx, i]))
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[bx, t, kvh, j])
                    gv_loc[j] = 1.0
                    if has_gv:
                        gv_loc[j] = T.exp(T.Cast(accum, GValue[bx, t, hx, j]))

                for i, j in T.Parallel(Dq, Dv):
                    h_state[i, j] = h_state[i, j] * decay[0] * g_loc[i] * gk_loc[i] * gv_loc[j] + (k_loc[i] * v_loc[j])
                    HStates[bx, t, hx, i, j] = h_state[i, j]

                for i, j in T.Parallel(Dq, Dv):
                    prod[i, j] = h_state[i, j] * q_loc[i]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(Dv):
                    O[bx, t, hx, j] = T.Cast(ts, o_loc[j] * scale)

            for i, j in T.Parallel(Dq, Dv):
                Hf[bx, hx, i, j] = h_state[i, j]

    return rec_fwd


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
    """Build a zero-initialisation kernel for the batched recurrent state.

    Grid: ``T.Kernel(num_heads, batch)``.  Writes 0.0 to every element of the
    fp32 state buffer ``H0[batch, num_heads, qk_head_dim, v_head_dim]``.

    The ``Q`` tensor is taken as an input only to satisfy TileLang's dtype-
    inference requirement; its values are not used.

    Returns:
        A TileLang ``@T.prim_func`` (``rec_init_state``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, Dq, Dv = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rec_init_state(
        Q: T.Tensor((B, S, H, Dq), ts),
        H0: T.Tensor((B, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = Q[0, 0, 0, 0]
            _seq_ref[0] = S
            for i, j in T.Parallel(Dq, Dv):
                H0[bx, hx, i, j] = 0.0

    return rec_init_state


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
    """Build a zero-initialisation kernel for the packed (varlen) recurrent state.

    Grid: ``T.Kernel(num_heads, num_seqs)``.  Writes 0.0 to each per-sequence
    state slot in ``H0[num_seqs, num_heads, qk_head_dim, v_head_dim]``.

    Returns:
        A TileLang ``@T.prim_func`` (``rec_packed_init_state``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, Dq, Dv = num_seqs, total_tokens, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def rec_packed_init_state(
        Q: T.Tensor((1, TQ, H, Dq), ts),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        H0: T.Tensor((N, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), "int32")
            _ts_ref[0] = Q[0, 0, 0, 0]
            _seq_ref[0] = CuSeqLens[nx + 1] - CuSeqLens[nx]
            for i, j in T.Parallel(Dq, Dv):
                H0[nx, hx, i, j] = 0.0

    return rec_packed_init_state


def make_packed_fwd_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    gamma_batch: int,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
    dtype,
    threads: int = 128,
):
    """Build the packed (varlen) forward ``@T.prim_func`` for recurrent linear-attention.

    Grid: ``T.Kernel(num_heads, num_seqs)``.  Each CTA processes one sequence
    and iterates up to ``total_tokens`` steps; active steps are gated by the
    ``CuSeqLens`` boundaries.

    Buffer layout::

        Q, K, V:    (1, total_tokens, num_heads/num_kv_heads, Dq/Dv)  compute dtype
        GDecay, GKey, GValue:  gate tensors (same shapes as Q/K/V)
        H0:         (num_seqs, num_heads, Dq, Dv)  float32
        GGamma:     (gamma_batch, num_heads)         float32
        CuSeqLens:  (num_seqs + 1,)                  int32
        O:          (1, total_tokens, num_heads, Dv)  compute dtype
        Hf:         (num_seqs, num_heads, Dq, Dv)    float32
        HStates:    (num_seqs, total_tokens, num_heads, Dq, Dv) float32

    Args:
        num_seqs: Number of packed sequences ``N``.
        total_tokens: Maximum token count ``TQ`` (padded to the longest sequence).
        num_heads: ``H``.
        num_kv_heads: ``HK``; GQA grouping ``G = H // HK`` is computed internally.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        gamma_batch: First dimension of ``GGamma``; 1 means broadcast across seqs.
        softmax_scale: Output multiplier.
        has_g / has_gk / has_gv: Whether the corresponding gate tensors contain
            meaningful values (otherwise defaulted to 1.0).
        has_g_gamma: Whether ``GGamma`` is active.
        use_static_gamma: If ``True``, use ``exp(slope * head_idx)`` instead of
            reading from ``GGamma``.
        static_gamma_slope: Slope for the static head-indexed gamma.
        reverse: If ``True``, scan right-to-left.
        dtype: Compute dtype.
        threads: Threads per CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rec_packed_fwd``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, HK, Dq, Dv, GB = (
        num_seqs,
        total_tokens,
        num_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        gamma_batch,
    )
    G = H // HK
    scale = float(softmax_scale)
    gamma_slope = float(static_gamma_slope)

    @T.prim_func
    def rec_packed_fwd(
        Q: T.Tensor((1, TQ, H, Dq), ts),
        K: T.Tensor((1, TQ, HK, Dq), ts),
        V: T.Tensor((1, TQ, HK, Dv), ts),
        GDecay: T.Tensor((1, TQ, H, Dq), ts),
        GKey: T.Tensor((1, TQ, H, Dq), ts),
        GValue: T.Tensor((1, TQ, H, Dv), ts),
        H0: T.Tensor((N, H, Dq, Dv), accum),
        GGamma: T.Tensor((GB, H), accum),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        O: T.Tensor((1, TQ, H, Dv), ts),
        Hf: T.Tensor((N, H, Dq, Dv), accum),
        HStates: T.Tensor((N, TQ, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            kvh = hx // G
            h_state = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            g_loc = T.alloc_fragment((Dq,), accum)
            gk_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            gv_loc = T.alloc_fragment((Dv,), accum)
            o_loc = T.alloc_fragment((Dv,), accum)
            prod = T.alloc_fragment((Dq, Dv), accum)
            decay = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gb_ref = T.alloc_fragment((1,), accum)
            _hk_ref = T.alloc_fragment((1,), accum)
            _gb_ref[0] = GB
            _hk_ref[0] = HK
            _ts_ref[0] = Q[0, 0, 0, 0]

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            decay[0] = 1.0
            if has_g_gamma:
                if use_static_gamma:
                    decay[0] = T.exp(gamma_slope * T.Cast(accum, hx))
                else:
                    decay[0] = T.exp(GGamma[T.if_then_else(GB == 1, 0, nx), hx])

            for i, j in T.Parallel(Dq, Dv):
                h_state[i, j] = H0[nx, hx, i, j]
                HStates[nx, 0, hx, i, j] = h_state[i, j]

            for step in T.serial(TQ):
                active = step < seq_len
                if reverse:
                    raw_t = end - 1 - step
                else:
                    raw_t = start + step
                t = T.if_then_else(active, raw_t, 0)

                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[0, t, hx, i])
                    k_loc[i] = T.Cast(accum, K[0, t, kvh, i])
                    g_loc[i] = 1.0
                    gk_loc[i] = 1.0
                    if has_g:
                        g_loc[i] = T.exp(T.Cast(accum, GDecay[0, t, hx, i]))
                    if has_gk:
                        gk_loc[i] = T.exp(T.Cast(accum, GKey[0, t, hx, i]))
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[0, t, kvh, j])
                    gv_loc[j] = 1.0
                    if has_gv:
                        gv_loc[j] = T.exp(T.Cast(accum, GValue[0, t, hx, j]))

                for i, j in T.Parallel(Dq, Dv):
                    if active:
                        h_state[i, j] = h_state[i, j] * decay[0] * g_loc[i] * gk_loc[i] * gv_loc[j] + (
                            k_loc[i] * v_loc[j]
                        )
                        HStates[nx, step, hx, i, j] = h_state[i, j]

                for i, j in T.Parallel(Dq, Dv):
                    prod[i, j] = h_state[i, j] * q_loc[i]
                T.reduce_sum(prod, o_loc, dim=0, clear=True)
                for j in T.Parallel(Dv):
                    if active:
                        O[0, t, hx, j] = T.Cast(ts, o_loc[j] * scale)

            for i, j in T.Parallel(Dq, Dv):
                Hf[nx, hx, i, j] = h_state[i, j]

    return rec_packed_fwd


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    gamma_batch: int,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
    dtype,
    threads: int = 128,
):
    """Build the batched backward ``@T.prim_func`` for recurrent linear-attention.

    Grid: ``T.Kernel(num_heads, batch)``.  Each CTA iterates in reverse order
    over the sequence using ``HStates`` (the forward per-step scan buffer) and
    accumulates gradients via:

        dh_acc += q_t ⊗ dO_t * scale
        dq_t = (h_t @ dO_t) * scale
        dk_t = dh_acc @ v_t
        dv_t = dh_acc.T @ k_t

    Note: gradients for ``GKey`` and ``GValue`` (per-dimension gates) are also
    computed when the corresponding ``has_g*`` flags are ``True``.

    Args:
        Same as :func:`make_fwd_prim_func`.

    Returns:
        ``@T.prim_func`` (``rec_bwd``) with buffers::

            Q, K, V, GDecay, GKey, GValue,
            HStates, GGamma,
            dO, dH_final,
            dQ, dK, dV, dG, dGKey, dGValue, dH0

        * ``dH_final``: gradient flowing into the final state; zeros if the
          caller does not differentiate through it.
        * ``dH0``: gradient w.r.t. the initial state ``H0``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, HK, Dq, Dv, GB = batch, seq_len, num_heads, num_kv_heads, qk_head_dim, v_head_dim, gamma_batch
    G = H // HK
    scale = float(softmax_scale)
    gamma_slope = float(static_gamma_slope)

    @T.prim_func
    def rec_bwd(
        Q: T.Tensor((B, S, H, Dq), ts),
        K: T.Tensor((B, S, HK, Dq), ts),
        V: T.Tensor((B, S, HK, Dv), ts),
        GDecay: T.Tensor((B, S, H, Dq), ts),
        GKey: T.Tensor((B, S, H, Dq), ts),
        GValue: T.Tensor((B, S, H, Dv), ts),
        HStates: T.Tensor((B, S, H, Dq, Dv), accum),
        GGamma: T.Tensor((GB, H), accum),
        dO: T.Tensor((B, S, H, Dv), ts),
        dH_final: T.Tensor((B, H, Dq, Dv), accum),
        dQ: T.Tensor((B, S, H, Dq), ts),
        dK: T.Tensor((B, S, H, Dq), ts),
        dV: T.Tensor((B, S, H, Dv), ts),
        dG: T.Tensor((B, S, H, Dq), ts),
        dGKey: T.Tensor((B, S, H, Dq), ts),
        dGValue: T.Tensor((B, S, H, Dv), ts),
        dH0: T.Tensor((B, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            kvh = hx // G
            dh_acc = T.alloc_fragment((Dq, Dv), accum)
            h_t = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            g_loc = T.alloc_fragment((Dq,), accum)
            gk_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            gv_loc = T.alloc_fragment((Dv,), accum)
            do_loc = T.alloc_fragment((Dv,), accum)
            dq_loc = T.alloc_fragment((Dq,), accum)
            dk_loc = T.alloc_fragment((Dq,), accum)
            dv_loc = T.alloc_fragment((Dv,), accum)
            dg_loc = T.alloc_fragment((Dq,), accum)
            dgv_loc = T.alloc_fragment((Dv,), accum)
            prod_h_do = T.alloc_fragment((Dq, Dv), accum)
            prod_dh_v = T.alloc_fragment((Dq, Dv), accum)
            prod_dh_k = T.alloc_fragment((Dq, Dv), accum)
            prod_dg = T.alloc_fragment((Dq, Dv), accum)
            h_mid = T.alloc_fragment((Dq, Dv), accum)
            decay = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gb_ref = T.alloc_fragment((1,), accum)
            _hk_ref = T.alloc_fragment((1,), accum)
            _gb_ref[0] = GB
            _hk_ref[0] = HK
            _ts_ref[0] = Q[0, 0, 0, 0]
            decay[0] = 1.0
            if has_g_gamma:
                if use_static_gamma:
                    decay[0] = T.exp(gamma_slope * T.Cast(accum, hx))
                else:
                    decay[0] = T.exp(GGamma[T.if_then_else(GB == 1, 0, bx), hx])

            for i, j in T.Parallel(Dq, Dv):
                dh_acc[i, j] = dH_final[bx, hx, i, j]

            for t_iter in T.serial(S):
                t = T.if_then_else(reverse, t_iter, S - 1 - t_iter)
                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[bx, t, hx, i])
                    k_loc[i] = T.Cast(accum, K[bx, t, kvh, i])
                    g_loc[i] = 1.0
                    gk_loc[i] = 1.0
                    if has_g:
                        g_loc[i] = T.exp(T.Cast(accum, GDecay[bx, t, hx, i]))
                    if has_gk:
                        gk_loc[i] = T.exp(T.Cast(accum, GKey[bx, t, hx, i]))
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[bx, t, kvh, j])
                    gv_loc[j] = 1.0
                    if has_gv:
                        gv_loc[j] = T.exp(T.Cast(accum, GValue[bx, t, hx, j]))
                    do_loc[j] = T.Cast(accum, dO[bx, t, hx, j])
                for i, j in T.Parallel(Dq, Dv):
                    h_t[i, j] = HStates[bx, t, hx, i, j]

                for i, j in T.Parallel(Dq, Dv):
                    prod_h_do[i, j] = h_t[i, j] * do_loc[j]
                T.reduce_sum(prod_h_do, dq_loc, dim=1, clear=True)
                for i in T.Parallel(Dq):
                    dQ[bx, t, hx, i] = T.Cast(ts, dq_loc[i] * scale)

                for i, j in T.Parallel(Dq, Dv):
                    dh_acc[i, j] = dh_acc[i, j] + q_loc[i] * do_loc[j] * scale

                for i, j in T.Parallel(Dq, Dv):
                    prod_dh_v[i, j] = dh_acc[i, j] * v_loc[j]
                T.reduce_sum(prod_dh_v, dk_loc, dim=1, clear=True)
                for i in T.Parallel(Dq):
                    dK[bx, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(Dq, Dv):
                    prod_dh_k[i, j] = dh_acc[i, j] * k_loc[i]
                T.reduce_sum(prod_dh_k, dv_loc, dim=0, clear=True)
                for j in T.Parallel(Dv):
                    dV[bx, t, hx, j] = T.Cast(ts, dv_loc[j])

                if has_g or has_gk or has_gv:
                    for i, j in T.Parallel(Dq, Dv):
                        h_mid[i, j] = h_t[i, j] - k_loc[i] * v_loc[j]
                        prod_dg[i, j] = dh_acc[i, j] * h_mid[i, j]
                if has_g or has_gk:
                    T.reduce_sum(prod_dg, dg_loc, dim=1, clear=True)
                if has_gv:
                    T.reduce_sum(prod_dg, dgv_loc, dim=0, clear=True)
                if has_g:
                    for i in T.Parallel(Dq):
                        dG[bx, t, hx, i] = T.Cast(ts, dg_loc[i])
                if has_gk:
                    for i in T.Parallel(Dq):
                        dGKey[bx, t, hx, i] = T.Cast(ts, dg_loc[i])
                if has_gv:
                    for j in T.Parallel(Dv):
                        dGValue[bx, t, hx, j] = T.Cast(ts, dgv_loc[j])

                for i, j in T.Parallel(Dq, Dv):
                    dh_acc[i, j] = dh_acc[i, j] * decay[0] * g_loc[i] * gk_loc[i] * gv_loc[j]

            for i, j in T.Parallel(Dq, Dv):
                dH0[bx, hx, i, j] = dh_acc[i, j]

    return rec_bwd


def make_packed_bwd_prim_func(
    *,
    num_seqs: int,
    total_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    gamma_batch: int,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
    dtype,
    threads: int = 128,
):
    """Build the packed (varlen) backward ``@T.prim_func`` for recurrent linear-attention.

    Grid: ``T.Kernel(num_heads, num_seqs)``.  Mirrors
    :func:`make_bwd_prim_func` but operates on packed varlen tensors shaped
    ``(1, total_tokens, ...)`` and uses ``CuSeqLens`` to bound each sequence's
    active range.

    Args:
        Same as :func:`make_packed_fwd_prim_func`; all gate and scale
        parameters control which gradient paths are active.

    Returns:
        A TileLang ``@T.prim_func`` (``rec_packed_bwd``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, TQ, H, HK, Dq, Dv, GB = (
        num_seqs,
        total_tokens,
        num_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        gamma_batch,
    )
    G = H // HK
    scale = float(softmax_scale)
    gamma_slope = float(static_gamma_slope)

    @T.prim_func
    def rec_packed_bwd(
        Q: T.Tensor((1, TQ, H, Dq), ts),
        K: T.Tensor((1, TQ, HK, Dq), ts),
        V: T.Tensor((1, TQ, HK, Dv), ts),
        GDecay: T.Tensor((1, TQ, H, Dq), ts),
        GKey: T.Tensor((1, TQ, H, Dq), ts),
        GValue: T.Tensor((1, TQ, H, Dv), ts),
        HStates: T.Tensor((N, TQ, H, Dq, Dv), accum),
        GGamma: T.Tensor((GB, H), accum),
        CuSeqLens: T.Tensor((N + 1,), "int32"),
        dO: T.Tensor((1, TQ, H, Dv), ts),
        dH_final: T.Tensor((N, H, Dq, Dv), accum),
        dQ: T.Tensor((1, TQ, H, Dq), ts),
        dK: T.Tensor((1, TQ, H, Dq), ts),
        dV: T.Tensor((1, TQ, H, Dv), ts),
        dG: T.Tensor((1, TQ, H, Dq), ts),
        dGKey: T.Tensor((1, TQ, H, Dq), ts),
        dGValue: T.Tensor((1, TQ, H, Dv), ts),
        dH0: T.Tensor((N, H, Dq, Dv), accum),
    ):
        with T.Kernel(H, N, threads=threads) as (hx, nx):
            kvh = hx // G
            dh_acc = T.alloc_fragment((Dq, Dv), accum)
            h_t = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            g_loc = T.alloc_fragment((Dq,), accum)
            gk_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            gv_loc = T.alloc_fragment((Dv,), accum)
            do_loc = T.alloc_fragment((Dv,), accum)
            dq_loc = T.alloc_fragment((Dq,), accum)
            dk_loc = T.alloc_fragment((Dq,), accum)
            dv_loc = T.alloc_fragment((Dv,), accum)
            dg_loc = T.alloc_fragment((Dq,), accum)
            dgv_loc = T.alloc_fragment((Dv,), accum)
            prod_h_do = T.alloc_fragment((Dq, Dv), accum)
            prod_dh_v = T.alloc_fragment((Dq, Dv), accum)
            prod_dh_k = T.alloc_fragment((Dq, Dv), accum)
            prod_dg = T.alloc_fragment((Dq, Dv), accum)
            h_mid = T.alloc_fragment((Dq, Dv), accum)
            decay = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gb_ref = T.alloc_fragment((1,), accum)
            _hk_ref = T.alloc_fragment((1,), accum)
            _gb_ref[0] = GB
            _hk_ref[0] = HK
            _ts_ref[0] = Q[0, 0, 0, 0]

            start = T.Cast("int32", CuSeqLens[nx])
            end = T.Cast("int32", CuSeqLens[nx + 1])
            seq_len = end - start

            decay[0] = 1.0
            if has_g_gamma:
                if use_static_gamma:
                    decay[0] = T.exp(gamma_slope * T.Cast(accum, hx))
                else:
                    decay[0] = T.exp(GGamma[T.if_then_else(GB == 1, 0, nx), hx])

            for i, j in T.Parallel(Dq, Dv):
                dh_acc[i, j] = dH_final[nx, hx, i, j]

            for step_rev in T.serial(TQ):
                active = step_rev < seq_len
                step = seq_len - 1 - step_rev
                if reverse:
                    raw_t = end - 1 - step
                else:
                    raw_t = start + step
                t = T.if_then_else(active, raw_t, 0)
                scan_pos = T.if_then_else(active, step, 0)

                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[0, t, hx, i])
                    k_loc[i] = T.Cast(accum, K[0, t, kvh, i])
                    g_loc[i] = 1.0
                    gk_loc[i] = 1.0
                    if has_g:
                        g_loc[i] = T.exp(T.Cast(accum, GDecay[0, t, hx, i]))
                    if has_gk:
                        gk_loc[i] = T.exp(T.Cast(accum, GKey[0, t, hx, i]))
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[0, t, kvh, j])
                    gv_loc[j] = 1.0
                    if has_gv:
                        gv_loc[j] = T.exp(T.Cast(accum, GValue[0, t, hx, j]))
                    do_loc[j] = T.Cast(accum, dO[0, t, hx, j])
                for i, j in T.Parallel(Dq, Dv):
                    h_t[i, j] = HStates[nx, scan_pos, hx, i, j]

                for i, j in T.Parallel(Dq, Dv):
                    prod_h_do[i, j] = h_t[i, j] * do_loc[j]
                T.reduce_sum(prod_h_do, dq_loc, dim=1, clear=True)
                for i in T.Parallel(Dq):
                    if active:
                        dQ[0, t, hx, i] = T.Cast(ts, dq_loc[i] * scale)

                for i, j in T.Parallel(Dq, Dv):
                    if active:
                        dh_acc[i, j] = dh_acc[i, j] + q_loc[i] * do_loc[j] * scale

                for i, j in T.Parallel(Dq, Dv):
                    prod_dh_v[i, j] = dh_acc[i, j] * v_loc[j]
                T.reduce_sum(prod_dh_v, dk_loc, dim=1, clear=True)
                for i in T.Parallel(Dq):
                    if active:
                        dK[0, t, hx, i] = T.Cast(ts, dk_loc[i])

                for i, j in T.Parallel(Dq, Dv):
                    prod_dh_k[i, j] = dh_acc[i, j] * k_loc[i]
                T.reduce_sum(prod_dh_k, dv_loc, dim=0, clear=True)
                for j in T.Parallel(Dv):
                    if active:
                        dV[0, t, hx, j] = T.Cast(ts, dv_loc[j])

                if has_g or has_gk or has_gv:
                    for i, j in T.Parallel(Dq, Dv):
                        h_mid[i, j] = h_t[i, j] - k_loc[i] * v_loc[j]
                        prod_dg[i, j] = dh_acc[i, j] * h_mid[i, j]
                if has_g or has_gk:
                    T.reduce_sum(prod_dg, dg_loc, dim=1, clear=True)
                if has_gv:
                    T.reduce_sum(prod_dg, dgv_loc, dim=0, clear=True)
                if has_g:
                    for i in T.Parallel(Dq):
                        if active:
                            dG[0, t, hx, i] = T.Cast(ts, dg_loc[i])
                if has_gk:
                    for i in T.Parallel(Dq):
                        if active:
                            dGKey[0, t, hx, i] = T.Cast(ts, dg_loc[i])
                if has_gv:
                    for j in T.Parallel(Dv):
                        if active:
                            dGValue[0, t, hx, j] = T.Cast(ts, dgv_loc[j])

                for i, j in T.Parallel(Dq, Dv):
                    if active:
                        dh_acc[i, j] = dh_acc[i, j] * decay[0] * g_loc[i] * gk_loc[i] * gv_loc[j]

            for i, j in T.Parallel(Dq, Dv):
                dH0[nx, hx, i, j] = dh_acc[i, j]

    return rec_packed_bwd


def make_reduce_kv_heads_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    threads: int = 128,
):
    """Build a GQA gradient reduction kernel ``@T.prim_func``.

    In GQA the backward pass produces per-Q-head gradients ``dKPart`` and
    ``dVPart`` shaped ``(batch, seq_len, num_heads, D)``.  These must be
    summed over the ``G = num_heads // num_kv_heads`` Q-heads that share
    each KV head to produce the true ``dK``/``dV`` of shape
    ``(batch, seq_len, num_kv_heads, D)``.

    Grid: ``T.Kernel(num_kv_heads, seq_len, batch)``.

    Args:
        batch: ``B``.
        seq_len: ``S``.
        num_heads: ``H`` (total Q heads).
        num_kv_heads: ``HK``; ``H`` must be divisible by ``HK``.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        dtype: Compute dtype.
        threads: Threads per CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rec_reduce_kv_heads``) with signature
        ``(dKPart, dVPart, dK, dV)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, HK, Dq, Dv = batch, seq_len, num_heads, num_kv_heads, qk_head_dim, v_head_dim
    G = H // HK

    @T.prim_func
    def rec_reduce_kv_heads(
        dKPart: T.Tensor((B, S, H, Dq), ts),
        dVPart: T.Tensor((B, S, H, Dv), ts),
        dK: T.Tensor((B, S, HK, Dq), ts),
        dV: T.Tensor((B, S, HK, Dv), ts),
    ):
        with T.Kernel(HK, S, B, threads=threads) as (kvh, tx, bx):
            acc_k = T.alloc_fragment((Dq,), accum)
            acc_v = T.alloc_fragment((Dv,), accum)
            _h_ref = T.alloc_fragment((1,), accum)
            _hk_ref = T.alloc_fragment((1,), accum)
            _h_ref[0] = H
            _hk_ref[0] = HK
            for i in T.Parallel(Dq):
                acc_k[i] = 0.0
            for j in T.Parallel(Dv):
                acc_v[j] = 0.0
            for gi in T.serial(G):
                hx = kvh * G + gi
                for i in T.Parallel(Dq):
                    acc_k[i] = acc_k[i] + T.Cast(accum, dKPart[bx, tx, hx, i])
                for j in T.Parallel(Dv):
                    acc_v[j] = acc_v[j] + T.Cast(accum, dVPart[bx, tx, hx, j])
            for i in T.Parallel(Dq):
                dK[bx, tx, kvh, i] = T.Cast(ts, acc_k[i])
            for j in T.Parallel(Dv):
                dV[bx, tx, kvh, j] = T.Cast(ts, acc_v[j])

    return rec_reduce_kv_heads
