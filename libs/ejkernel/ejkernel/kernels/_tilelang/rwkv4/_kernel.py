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

"""tile-lang RWKV-4 time-mix scan kernel (forward).

Channels are independent — each block of channels iterates the recurrence
sequentially over time. The state per channel is ``(alpha, beta, eps)`` in
fp32. The log-sum-exp trick (via ``eps``) keeps the running statistics
numerically stable.

Per-step update at time ``t`` (per channel ``c``):

    ukt = u[c] + k[t, c]
    tau = max(ukt, eps)
    e1a = exp(eps - tau)
    e2a = exp(ukt - tau)
    wkv[t, c] = (e1a * alpha + e2a * v[t, c]) / (e1a * beta + e2a)

    w_eps = w[c] + eps
    eps_next = max(w_eps, k[t, c])
    e1b = exp(w_eps - eps_next)
    e2b = exp(k[t, c] - eps_next)
    alpha_next = e1b * alpha + e2b * v[t, c]
    beta_next  = e1b * beta + e2b
    eps = eps_next
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the TileLang dtype string for a supported floating-point dtype.

    Args:
        dtype: any dtype specifier accepted by ``jnp.dtype`` (e.g. ``jnp.float16``,
            ``"bfloat16"``, a NumPy dtype, etc.).

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
        raise TypeError(f"Unsupported dtype for tile-lang rwkv4: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    channels: int,
    block_c: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-4 forward ``@T.prim_func``.

    Grid: ``(ceildiv(channels, BLOCK_C), batch)``. Each CTA owns
    ``BLOCK_C`` channels and walks the entire time axis sequentially.

    Returns:
        ``@T.prim_func`` with buffers
        ``(W, U, K, V, State0, WKV, StateF)`` where ``State0`` / ``StateF``
        pack ``(alpha, beta, eps)`` along the second axis as
        ``(batch, 3, channels)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, C, BC = batch, seq_len, channels, block_c

    @T.prim_func
    def rwkv4_fwd(
        W: T.Tensor((C,), ts),
        U: T.Tensor((C,), ts),
        K: T.Tensor((B, S, C), ts),
        V: T.Tensor((B, S, C), ts),
        State0: T.Tensor((B, 3, C), accum),
        WKV: T.Tensor((B, S, C), ts),
        StateF: T.Tensor((B, 3, C), accum),
    ):
        with T.Kernel(T.ceildiv(C, BC), B, threads=threads) as (cx, bx):
            w_loc = T.alloc_fragment((BC,), accum)
            u_loc = T.alloc_fragment((BC,), accum)
            alpha = T.alloc_fragment((BC,), accum)
            beta = T.alloc_fragment((BC,), accum)
            eps = T.alloc_fragment((BC,), accum)
            kt = T.alloc_fragment((BC,), accum)
            vt = T.alloc_fragment((BC,), accum)
            wkv_loc = T.alloc_fragment((BC,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                in_range = c_idx < C
                w_loc[c] = T.if_then_else(in_range, -T.exp(T.Cast(accum, W[c_idx])), 0.0)
                u_loc[c] = T.if_then_else(in_range, T.Cast(accum, U[c_idx]), 0.0)
                alpha[c] = T.if_then_else(in_range, State0[bx, 0, c_idx], 0.0)
                beta[c] = T.if_then_else(in_range, State0[bx, 1, c_idx], 0.0)
                eps[c] = T.if_then_else(in_range, State0[bx, 2, c_idx], -1e30)

            for t in T.serial(S):
                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    in_range = c_idx < C
                    kt[c] = T.if_then_else(in_range, T.Cast(accum, K[bx, t, c_idx]), 0.0)
                    vt[c] = T.if_then_else(in_range, T.Cast(accum, V[bx, t, c_idx]), 0.0)

                for c in T.Parallel(BC):
                    ukt = u_loc[c] + kt[c]
                    tau = T.max(ukt, eps[c])
                    e1a = T.exp(eps[c] - tau)
                    e2a = T.exp(ukt - tau)
                    wkv_loc[c] = (e1a * alpha[c] + e2a * vt[c]) / (e1a * beta[c] + e2a)

                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    if c_idx < C:
                        WKV[bx, t, c_idx] = T.Cast(ts, wkv_loc[c])

                for c in T.Parallel(BC):
                    w_eps = w_loc[c] + eps[c]
                    eps_next = T.max(w_eps, kt[c])
                    e1b = T.exp(w_eps - eps_next)
                    e2b = T.exp(kt[c] - eps_next)
                    alpha[c] = e1b * alpha[c] + e2b * vt[c]
                    beta[c] = e1b * beta[c] + e2b
                    eps[c] = eps_next

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                if c_idx < C:
                    StateF[bx, 0, c_idx] = alpha[c]
                    StateF[bx, 1, c_idx] = beta[c]
                    StateF[bx, 2, c_idx] = eps[c]

    return rwkv4_fwd


def make_init_state_prim_func(
    *,
    batch: int,
    seq_len: int,
    channels: int,
    block_c: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-4 zero-state initialisation ``@T.prim_func``.

    Grid: ``(ceildiv(channels, BLOCK_C), batch)``. Writes ``(alpha=0, beta=0,
    eps=-1e30)`` into the output buffer. The ``seq_len`` parameter is baked in
    only to satisfy dtype-inference constraints inside TileLang; it does not
    affect the initialised values.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S`` (used only for dtype-probe allocation).
        channels: number of channels ``C``.
        block_c: number of channels per CTA tile ``BLOCK_C``.
        dtype: input tensor dtype (float16 / bfloat16 / float32).
        threads: number of CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(K, State0)`` where
        ``K`` is ``(B, S, C, dtype)`` (read-only, used for type inference) and
        ``State0`` is ``(B, 3, C, float32)`` zeroed with ``eps=-1e30``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, C, BC = batch, seq_len, channels, block_c

    @T.prim_func
    def rwkv4_init_state(
        K: T.Tensor((B, S, C), ts),
        State0: T.Tensor((B, 3, C), accum),
    ):
        with T.Kernel(T.ceildiv(C, BC), B, threads=threads) as (cx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = K[0, 0, 0]
            _seq_ref[0] = S
            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                if c_idx < C:
                    State0[bx, 0, c_idx] = 0.0
                    State0[bx, 1, c_idx] = 0.0
                    State0[bx, 2, c_idx] = -1e30

    return rwkv4_init_state


def make_fwd_states_prim_func(
    *,
    batch: int,
    seq_len: int,
    channels: int,
    block_c: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-4 forward ``@T.prim_func`` that materialises every
    intermediate state for the backward pass.

    Identical recurrence to :func:`make_fwd_prim_func` but also writes
    ``Hscan[b, t, :, c]`` at every time-step ``t in [0, S]`` (``t=0`` is the
    initial state from ``State0``; ``t=t+1`` is the post-step state). The
    ``Hscan`` tensor is consumed by the backward kernel.

    Grid: ``(ceildiv(channels, BLOCK_C), batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        channels: number of channels ``C``.
        block_c: number of channels per CTA tile ``BLOCK_C``.
        dtype: input/output tensor dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(W, U, K, V, State0, WKV, StateF, Hscan)`` where ``Hscan`` is
        fp32 ``(B, S+1, 3, C)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, C, BC = batch, seq_len, channels, block_c

    @T.prim_func
    def rwkv4_fwd_states(
        W: T.Tensor((C,), ts),
        U: T.Tensor((C,), ts),
        K: T.Tensor((B, S, C), ts),
        V: T.Tensor((B, S, C), ts),
        State0: T.Tensor((B, 3, C), accum),
        WKV: T.Tensor((B, S, C), ts),
        StateF: T.Tensor((B, 3, C), accum),
        Hscan: T.Tensor((B, S + 1, 3, C), accum),
    ):
        with T.Kernel(T.ceildiv(C, BC), B, threads=threads) as (cx, bx):
            w_loc = T.alloc_fragment((BC,), accum)
            u_loc = T.alloc_fragment((BC,), accum)
            alpha = T.alloc_fragment((BC,), accum)
            beta = T.alloc_fragment((BC,), accum)
            eps = T.alloc_fragment((BC,), accum)
            kt = T.alloc_fragment((BC,), accum)
            vt = T.alloc_fragment((BC,), accum)
            wkv_loc = T.alloc_fragment((BC,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                in_range = c_idx < C
                w_loc[c] = T.if_then_else(in_range, -T.exp(T.Cast(accum, W[c_idx])), 0.0)
                u_loc[c] = T.if_then_else(in_range, T.Cast(accum, U[c_idx]), 0.0)
                alpha[c] = T.if_then_else(in_range, State0[bx, 0, c_idx], 0.0)
                beta[c] = T.if_then_else(in_range, State0[bx, 1, c_idx], 0.0)
                eps[c] = T.if_then_else(in_range, State0[bx, 2, c_idx], -1e30)
                if c_idx < C:
                    Hscan[bx, 0, 0, c_idx] = alpha[c]
                    Hscan[bx, 0, 1, c_idx] = beta[c]
                    Hscan[bx, 0, 2, c_idx] = eps[c]

            for t in T.serial(S):
                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    in_range = c_idx < C
                    kt[c] = T.if_then_else(in_range, T.Cast(accum, K[bx, t, c_idx]), 0.0)
                    vt[c] = T.if_then_else(in_range, T.Cast(accum, V[bx, t, c_idx]), 0.0)

                for c in T.Parallel(BC):
                    ukt = u_loc[c] + kt[c]
                    tau = T.max(ukt, eps[c])
                    e1a = T.exp(eps[c] - tau)
                    e2a = T.exp(ukt - tau)
                    wkv_loc[c] = (e1a * alpha[c] + e2a * vt[c]) / (e1a * beta[c] + e2a)

                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    if c_idx < C:
                        WKV[bx, t, c_idx] = T.Cast(ts, wkv_loc[c])

                for c in T.Parallel(BC):
                    w_eps = w_loc[c] + eps[c]
                    eps_next = T.max(w_eps, kt[c])
                    e1b = T.exp(w_eps - eps_next)
                    e2b = T.exp(kt[c] - eps_next)
                    alpha[c] = e1b * alpha[c] + e2b * vt[c]
                    beta[c] = e1b * beta[c] + e2b
                    eps[c] = eps_next

                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    if c_idx < C:
                        Hscan[bx, t + 1, 0, c_idx] = alpha[c]
                        Hscan[bx, t + 1, 1, c_idx] = beta[c]
                        Hscan[bx, t + 1, 2, c_idx] = eps[c]

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                if c_idx < C:
                    StateF[bx, 0, c_idx] = alpha[c]
                    StateF[bx, 1, c_idx] = beta[c]
                    StateF[bx, 2, c_idx] = eps[c]

    return rwkv4_fwd_states


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    channels: int,
    block_c: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-4 backward ``@T.prim_func`` (reverse-time adjoint scan).

    Reads the full-trajectory state ``Hscan`` produced by
    :func:`make_fwd_states_prim_func` and runs in reverse time (``t`` from
    ``S-1`` down to ``0``) computing the adjoint state ``(da, db, de)`` and
    accumulating per-batch parameter gradients ``dW_p`` / ``dU_p`` (which are
    later summed over ``batch`` by the reduce-param kernel).

    Grid: ``(ceildiv(channels, BLOCK_C), batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        channels: number of channels ``C``.
        block_c: channels per CTA tile ``BLOCK_C``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(W, U, K, V, Hscan, dWKV, dStateF, dW_p, dU_p, dK, dV, dState0)``
        where ``dW_p`` and ``dU_p`` are fp32 ``(B, C)`` per-batch partials.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, C, BC = batch, seq_len, channels, block_c

    @T.prim_func
    def rwkv4_bwd(
        W: T.Tensor((C,), ts),
        U: T.Tensor((C,), ts),
        K: T.Tensor((B, S, C), ts),
        V: T.Tensor((B, S, C), ts),
        Hscan: T.Tensor((B, S + 1, 3, C), accum),
        dWKV: T.Tensor((B, S, C), ts),
        dStateF: T.Tensor((B, 3, C), accum),
        dW_p: T.Tensor((B, C), accum),
        dU_p: T.Tensor((B, C), accum),
        dK: T.Tensor((B, S, C), ts),
        dV: T.Tensor((B, S, C), ts),
        dState0: T.Tensor((B, 3, C), accum),
    ):
        with T.Kernel(T.ceildiv(C, BC), B, threads=threads) as (cx, bx):
            w_raw = T.alloc_fragment((BC,), accum)
            w_loc = T.alloc_fragment((BC,), accum)
            u_loc = T.alloc_fragment((BC,), accum)
            da = T.alloc_fragment((BC,), accum)
            db = T.alloc_fragment((BC,), accum)
            de = T.alloc_fragment((BC,), accum)
            dw_acc = T.alloc_fragment((BC,), accum)
            du_acc = T.alloc_fragment((BC,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                in_range = c_idx < C
                w_raw[c] = T.if_then_else(in_range, T.Cast(accum, W[c_idx]), 0.0)
                w_loc[c] = -T.exp(w_raw[c])
                u_loc[c] = T.if_then_else(in_range, T.Cast(accum, U[c_idx]), 0.0)
                da[c] = T.if_then_else(in_range, dStateF[bx, 0, c_idx], 0.0)
                db[c] = T.if_then_else(in_range, dStateF[bx, 1, c_idx], 0.0)
                de[c] = T.if_then_else(in_range, dStateF[bx, 2, c_idx], 0.0)
                dw_acc[c] = 0.0
                du_acc[c] = 0.0

            for t_rev in T.serial(S):
                t = S - 1 - t_rev
                for c in T.Parallel(BC):
                    c_idx = cx * BC + c
                    in_range = c_idx < C
                    alpha = T.if_then_else(in_range, Hscan[bx, t, 0, c_idx], 0.0)
                    beta = T.if_then_else(in_range, Hscan[bx, t, 1, c_idx], 0.0)
                    eps = T.if_then_else(in_range, Hscan[bx, t, 2, c_idx], -1e30)
                    kt = T.if_then_else(in_range, T.Cast(accum, K[bx, t, c_idx]), 0.0)
                    vt = T.if_then_else(in_range, T.Cast(accum, V[bx, t, c_idx]), 0.0)
                    dy = T.if_then_else(in_range, T.Cast(accum, dWKV[bx, t, c_idx]), 0.0)

                    ukt = u_loc[c] + kt
                    tau = T.max(ukt, eps)
                    e1a = T.exp(eps - tau)
                    e2a = T.exp(ukt - tau)
                    num = e1a * alpha + e2a * vt
                    den = e1a * beta + e2a
                    dnum = dy / den
                    dden = -(dy * num) / (den * den)
                    de1a = dnum * alpha + dden * beta
                    de2a = dnum * vt + dden
                    dalpha = dnum * e1a
                    dbeta = dden * e1a
                    dvt = dnum * e2a
                    deps = de1a * e1a
                    dukt = de2a * e2a
                    dtau = -(de1a * e1a + de2a * e2a)
                    dukt_tau = dukt + T.if_then_else(ukt > eps, dtau, 0.0)
                    deps_tau = deps + T.if_then_else(ukt > eps, 0.0, dtau)

                    w_eps = w_loc[c] + eps
                    eps_next = T.max(w_eps, kt)
                    e1b = T.exp(w_eps - eps_next)
                    e2b = T.exp(kt - eps_next)
                    de1b = da[c] * alpha + db[c] * beta
                    de2b = da[c] * vt + db[c]
                    dalpha_total = dalpha + da[c] * e1b
                    dbeta_total = dbeta + db[c] * e1b
                    dvt_total = dvt + da[c] * e2b
                    dw_eps = de1b * e1b
                    dkt = de2b * e2b
                    deps_next = de[c] - de1b * e1b - de2b * e2b
                    dw_eps_total = dw_eps + T.if_then_else(w_eps > kt, deps_next, 0.0)
                    dkt_total = dkt + T.if_then_else(w_eps > kt, 0.0, deps_next) + dukt_tau
                    du_acc[c] = du_acc[c] + dukt_tau
                    dw_acc[c] = dw_acc[c] + dw_eps_total * w_loc[c]
                    deps_total = deps_tau + dw_eps_total
                    da[c] = dalpha_total
                    db[c] = dbeta_total
                    de[c] = deps_total
                    if c_idx < C:
                        dK[bx, t, c_idx] = T.Cast(ts, dkt_total)
                        dV[bx, t, c_idx] = T.Cast(ts, dvt_total)

            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                if c_idx < C:
                    dW_p[bx, c_idx] = dw_acc[c]
                    dU_p[bx, c_idx] = du_acc[c]
                    dState0[bx, 0, c_idx] = da[c]
                    dState0[bx, 1, c_idx] = db[c]
                    dState0[bx, 2, c_idx] = de[c]

    return rwkv4_bwd


def make_reduce_param_prim_func(
    *,
    batch: int,
    channels: int,
    block_c: int,
    dtype,
    threads: int = 128,
):
    """Build the RWKV-4 per-batch parameter-gradient reduce ``@T.prim_func``.

    Sums ``dP`` over the batch axis: ``dOut[c] = sum_b dP[b, c]``.
    Used in the backward pass to collapse the per-batch partial gradients
    for ``W`` and ``U`` into channel-shaped output tensors.

    Grid: ``(ceildiv(channels, BLOCK_C), 1)``.

    Args:
        batch: batch size ``B``.
        channels: number of channels ``C``.
        block_c: channels per CTA tile ``BLOCK_C``.
        dtype: output dtype (float16 / bfloat16 / float32). Partials are fp32.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(dP, dOut)`` where
        ``dP`` is fp32 ``(B, C)`` and ``dOut`` is ``(C, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, C, BC = batch, channels, block_c

    @T.prim_func
    def rwkv4_reduce_param(
        dP: T.Tensor((B, C), accum),
        dOut: T.Tensor((C,), ts),
    ):
        with T.Kernel(T.ceildiv(C, BC), 1, threads=threads) as (cx, _):
            total = T.alloc_fragment((BC,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            for c in T.Parallel(BC):
                c_idx = cx * BC + c
                total[c] = 0.0
                for b in T.serial(B):
                    if c_idx < C:
                        total[c] = total[c] + dP[b, c_idx]
                if c_idx < C:
                    dOut[c_idx] = T.Cast(ts, total[c])

    return rwkv4_reduce_param
