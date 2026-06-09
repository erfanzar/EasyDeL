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

"""Native tile-lang SSM-2 (Mamba2) forward kernel.

Per-step recurrence:

    dA      = exp(dt * A)                       # scalar per head
    dBx[i, n] = (dt * B[n]) * x[i]              # (head_dim, ssm_state_size)
    h[i, n] = dA * h_prev[i, n] + dBx[i, n]
    y[i]    = sum_n h[i, n] * C[n] + D * x[i]

Grid: ``(num_heads, batch)``. Per-head ``(head_dim, ssm_state_size)``
state lives in a fragment for the lifetime of the CTA.
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
        raise TypeError(f"Unsupported dtype for SSM2: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    n_groups: int,
    head_dim: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-2 (Mamba2) forward ``@T.prim_func`` (no state materialisation).

    Grid: ``(repeat, n_groups, batch)`` where ``repeat = num_heads // n_groups``,
    so each CTA handles one head.  The ``A`` and ``D`` scalars are per-head;
    ``Bp`` and ``C`` are shared across the ``repeat`` heads in each group.

    Per-step recurrence::

        dA        = exp(dt * A)              # scalar per head
        dBx[p, n] = dt * Bp[n] * x[p]
        h[p, n]   = dA * h[p, n] + dBx[p, n]
        y[p]      = sum_n h[p, n] * C[n] + D * x[p]

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H`` (must be divisible by ``n_groups``).
        n_groups: number of B/C groups ``G``; each group covers ``H//G`` heads.
        head_dim: per-head feature dimension ``P``.
        ssm_state_size: SSM state size ``N``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(X, A, Bp, C, D, Dt, H0, Y, Hf)``
        where ``Bp/C`` are ``(B, S, G, N, dtype)`` and ``A/D`` are ``(H, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, G, P, N_ = batch, seq_len, num_heads, n_groups, head_dim, ssm_state_size
    repeat = H // G

    @T.prim_func
    def ssm2_fwd(
        X: T.Tensor((B, S, H, P), ts),
        A: T.Tensor((H,), ts),
        Bp: T.Tensor((B, S, G, N_), ts),
        C: T.Tensor((B, S, G, N_), ts),
        D: T.Tensor((H,), ts),
        Dt: T.Tensor((B, S, H), ts),
        H0: T.Tensor((B, H, P, N_), accum),
        Y: T.Tensor((B, S, H, P), ts),
        Hf: T.Tensor((B, H, P, N_), accum),
    ):
        with T.Kernel(repeat, G, B, threads=threads) as (rx, gx, bx):
            hx = gx * repeat + rx
            h_state = T.alloc_fragment((P, N_), accum)
            x_loc = T.alloc_fragment((P,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            dt_loc = T.alloc_fragment((1,), accum)
            A_loc = T.alloc_fragment((1,), accum)
            D_loc = T.alloc_fragment((1,), accum)
            dA = T.alloc_fragment((1,), accum)
            y_prod = T.alloc_fragment((P, N_), accum)
            y_loc = T.alloc_fragment((P,), accum)
            _group_ref = T.alloc_fragment((1,), accum)
            _head_ref = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _group_ref[0] = G
            _head_ref[0] = H

            A_loc[0] = T.Cast(accum, A[hx])
            D_loc[0] = T.Cast(accum, D[hx])

            for i, n in T.Parallel(P, N_):
                h_state[i, n] = H0[bx, hx, i, n]

            for t in T.serial(S):
                for i in T.Parallel(P):
                    x_loc[i] = T.Cast(accum, X[bx, t, hx, i])
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, gx, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, gx, n])
                dt_loc[0] = T.Cast(accum, Dt[bx, t, hx])

                dA[0] = T.exp(dt_loc[0] * A_loc[0])

                for i, n in T.Parallel(P, N_):
                    h_state[i, n] = dA[0] * h_state[i, n] + dt_loc[0] * Bp_loc[n] * x_loc[i]

                for i, n in T.Parallel(P, N_):
                    y_prod[i, n] = h_state[i, n] * C_loc[n]
                T.reduce_sum(y_prod, y_loc, dim=1, clear=True)
                for i in T.Parallel(P):
                    Y[bx, t, hx, i] = T.Cast(ts, y_loc[i] + D_loc[0] * x_loc[i])

            for i, n in T.Parallel(P, N_):
                Hf[bx, hx, i, n] = h_state[i, n]

    return ssm2_fwd


def make_fwd_states_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    n_groups: int,
    head_dim: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """SSM-2 forward that also emits all hidden states ``Hall (B,S,H,P,N)``
    for the backward adjoint scan."""
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, G, P, N_ = batch, seq_len, num_heads, n_groups, head_dim, ssm_state_size
    repeat = H // G

    @T.prim_func
    def ssm2_fwd_states(
        X: T.Tensor((B, S, H, P), ts),
        A: T.Tensor((H,), ts),
        Bp: T.Tensor((B, S, G, N_), ts),
        C: T.Tensor((B, S, G, N_), ts),
        D: T.Tensor((H,), ts),
        Dt: T.Tensor((B, S, H), ts),
        H0: T.Tensor((B, H, P, N_), accum),
        Y: T.Tensor((B, S, H, P), ts),
        Hf: T.Tensor((B, H, P, N_), accum),
        Hall: T.Tensor((B, S, H, P, N_), accum),
    ):
        with T.Kernel(repeat, G, B, threads=threads) as (rx, gx, bx):
            hx = gx * repeat + rx
            h_state = T.alloc_fragment((P, N_), accum)
            x_loc = T.alloc_fragment((P,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            dt_loc = T.alloc_fragment((1,), accum)
            A_loc = T.alloc_fragment((1,), accum)
            D_loc = T.alloc_fragment((1,), accum)
            dA = T.alloc_fragment((1,), accum)
            y_prod = T.alloc_fragment((P, N_), accum)
            y_loc = T.alloc_fragment((P,), accum)
            _group_ref = T.alloc_fragment((1,), accum)
            _head_ref = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _group_ref[0] = G
            _head_ref[0] = H

            A_loc[0] = T.Cast(accum, A[hx])
            D_loc[0] = T.Cast(accum, D[hx])
            for i, n in T.Parallel(P, N_):
                h_state[i, n] = H0[bx, hx, i, n]

            for t in T.serial(S):
                for i in T.Parallel(P):
                    x_loc[i] = T.Cast(accum, X[bx, t, hx, i])
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, gx, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, gx, n])
                dt_loc[0] = T.Cast(accum, Dt[bx, t, hx])
                dA[0] = T.exp(dt_loc[0] * A_loc[0])
                for i, n in T.Parallel(P, N_):
                    h_state[i, n] = dA[0] * h_state[i, n] + dt_loc[0] * Bp_loc[n] * x_loc[i]
                for i, n in T.Parallel(P, N_):
                    y_prod[i, n] = h_state[i, n] * C_loc[n]
                T.reduce_sum(y_prod, y_loc, dim=1, clear=True)
                for i in T.Parallel(P):
                    Y[bx, t, hx, i] = T.Cast(ts, y_loc[i] + D_loc[0] * x_loc[i])
                for i, n in T.Parallel(P, N_):
                    Hall[bx, t, hx, i, n] = h_state[i, n]

            for i, n in T.Parallel(P, N_):
                Hf[bx, hx, i, n] = h_state[i, n]

    return ssm2_fwd_states


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    n_groups: int,
    head_dim: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """SSM-2 reverse-time adjoint scan. Grid ``(num_heads, batch)`` — every
    output slice is keyed by ``(b, h)`` so all writes are disjoint (no
    atomics). ``dA`` / ``dD`` are emitted as per-(b,h) partials and reduced
    over batch in the JAX glue."""
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, G, P, N_ = batch, seq_len, num_heads, n_groups, head_dim, ssm_state_size
    repeat = H // G

    @T.prim_func
    def ssm2_bwd(
        X: T.Tensor((B, S, H, P), ts),
        A: T.Tensor((H,), ts),
        Bp: T.Tensor((B, S, G, N_), ts),
        C: T.Tensor((B, S, G, N_), ts),
        D: T.Tensor((H,), ts),
        Dt: T.Tensor((B, S, H), ts),
        H0: T.Tensor((B, H, P, N_), accum),
        Hall: T.Tensor((B, S, H, P, N_), accum),
        dY: T.Tensor((B, S, H, P), ts),
        dHf: T.Tensor((B, H, P, N_), accum),
        dX: T.Tensor((B, S, H, P), accum),
        dA_p: T.Tensor((B, H), accum),
        dBp_o: T.Tensor((B, S, H, N_), accum),
        dC_o: T.Tensor((B, S, H, N_), accum),
        dD_p: T.Tensor((B, H), accum),
        dDt: T.Tensor((B, S, H), accum),
        dH0: T.Tensor((B, H, P, N_), accum),
    ):
        with T.Kernel(repeat, G, B, threads=threads) as (rx, gx, bx):
            hx = gx * repeat + rx
            dh = T.alloc_fragment((P, N_), accum)
            h_cur = T.alloc_fragment((P, N_), accum)
            h_prev = T.alloc_fragment((P, N_), accum)
            x_loc = T.alloc_fragment((P,), accum)
            dy_loc = T.alloc_fragment((P,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            dt_loc = T.alloc_fragment((1,), accum)
            A_loc = T.alloc_fragment((1,), accum)
            D_loc = T.alloc_fragment((1,), accum)
            dA_t = T.alloc_fragment((1,), accum)
            d_dBx = T.alloc_fragment((P, N_), accum)
            tmp_pn = T.alloc_fragment((P, N_), accum)
            tmp_p = T.alloc_fragment((P,), accum)
            tmp_n = T.alloc_fragment((N_,), accum)
            scal = T.alloc_fragment((1,), accum)
            dA_acc = T.alloc_fragment((1,), accum)
            dD_acc = T.alloc_fragment((1,), accum)
            _group_ref = T.alloc_fragment((1,), accum)
            _head_ref = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _group_ref[0] = G
            _head_ref[0] = H

            A_loc[0] = T.Cast(accum, A[hx])
            D_loc[0] = T.Cast(accum, D[hx])
            dA_acc[0] = 0.0
            dD_acc[0] = 0.0
            for i, n in T.Parallel(P, N_):
                dh[i, n] = dHf[bx, hx, i, n]

            for t_rev in T.serial(S):
                t = S - 1 - t_rev
                for i in T.Parallel(P):
                    x_loc[i] = T.Cast(accum, X[bx, t, hx, i])
                    dy_loc[i] = T.Cast(accum, dY[bx, t, hx, i])
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, gx, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, gx, n])
                dt_loc[0] = T.Cast(accum, Dt[bx, t, hx])
                dA_t[0] = T.exp(dt_loc[0] * A_loc[0])

                for i, n in T.Parallel(P, N_):
                    h_cur[i, n] = Hall[bx, t, hx, i, n]
                    h_prev[i, n] = T.if_then_else(
                        t == 0,
                        H0[bx, hx, i, n],
                        Hall[bx, T.max(t - 1, 0), hx, i, n],
                    )

                for i, n in T.Parallel(P, N_):
                    dh[i, n] = dh[i, n] + C_loc[n] * dy_loc[i]

                d_dA = T.alloc_fragment((1,), accum)
                for i, n in T.Parallel(P, N_):
                    d_dBx[i, n] = dh[i, n]
                    tmp_pn[i, n] = dh[i, n] * h_prev[i, n]
                T.reduce_sum(tmp_pn, tmp_p, dim=1, clear=True)
                T.reduce_sum(tmp_p, d_dA, dim=0, clear=True)
                dA_acc[0] = dA_acc[0] + d_dA[0] * dt_loc[0] * dA_t[0]

                for i, n in T.Parallel(P, N_):
                    tmp_pn[i, n] = h_cur[i, n] * dy_loc[i]
                T.reduce_sum(tmp_pn, tmp_n, dim=0, clear=True)
                for n in T.Parallel(N_):
                    dC_o[bx, t, hx, n] = tmp_n[n]

                for i, n in T.Parallel(P, N_):
                    tmp_pn[i, n] = d_dBx[i, n] * dt_loc[0] * x_loc[i]
                T.reduce_sum(tmp_pn, tmp_n, dim=0, clear=True)
                for n in T.Parallel(N_):
                    dBp_o[bx, t, hx, n] = tmp_n[n]

                for i, n in T.Parallel(P, N_):
                    tmp_pn[i, n] = d_dBx[i, n] * Bp_loc[n] * x_loc[i]
                T.reduce_sum(tmp_pn, tmp_p, dim=1, clear=True)
                T.reduce_sum(tmp_p, scal, dim=0, clear=True)
                dDt[bx, t, hx] = scal[0] + d_dA[0] * A_loc[0] * dA_t[0]

                for i, n in T.Parallel(P, N_):
                    tmp_pn[i, n] = d_dBx[i, n] * dt_loc[0] * Bp_loc[n]
                T.reduce_sum(tmp_pn, tmp_p, dim=1, clear=True)
                for i in T.Parallel(P):
                    dX[bx, t, hx, i] = tmp_p[i] + D_loc[0] * dy_loc[i]

                for i, n in T.Parallel(P, N_):
                    tmp_pn[i, n] = T.if_then_else(n == 0, dy_loc[i] * x_loc[i], 0.0)
                T.reduce_sum(tmp_pn, tmp_p, dim=1, clear=True)
                T.reduce_sum(tmp_p, scal, dim=0, clear=True)
                dD_acc[0] = dD_acc[0] + scal[0]

                for i, n in T.Parallel(P, N_):
                    dh[i, n] = dA_t[0] * dh[i, n]

            dA_p[bx, hx] = dA_acc[0]
            dD_p[bx, hx] = dD_acc[0]
            for i, n in T.Parallel(P, N_):
                dH0[bx, hx, i, n] = dh[i, n]

    return ssm2_bwd


def make_init_state_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-2 zero-state initialiser ``@T.prim_func``.

    Grid: ``(num_heads, batch)``. Writes zero fp32 ``(B, H, P, N)`` state.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S`` (dtype-probe only).
        num_heads: number of heads ``H``.
        head_dim: per-head feature dimension ``P``.
        ssm_state_size: SSM state size ``N``.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(X, H0)``; ``X`` is read-only.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, P, N_ = batch, seq_len, num_heads, head_dim, ssm_state_size

    @T.prim_func
    def ssm2_init_state(
        X: T.Tensor((B, S, H, P), ts),
        H0: T.Tensor((B, H, P, N_), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = X[0, 0, 0, 0]
            _seq_ref[0] = S
            for i, n in T.Parallel(P, N_):
                H0[bx, hx, i, n] = 0.0

    return ssm2_init_state


def make_reduce_bh_prim_func(
    *,
    batch: int,
    num_heads: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-2 batch-reduce ``@T.prim_func`` for ``(B, H)`` partials.

    Sums ``Partials`` over the batch axis: ``Out[h] = sum_b Partials[b, h]``.
    Used to reduce the per-batch partial gradients for ``A`` and ``D``.

    Grid: ``(num_heads, 1)``.

    Args:
        batch: batch size ``B``.
        num_heads: number of heads ``H``.
        dtype: output dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(Partials, Out)`` where
        ``Partials`` is fp32 ``(B, H)`` and ``Out`` is ``(H, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H = batch, num_heads

    @T.prim_func
    def ssm2_reduce_bh(
        Partials: T.Tensor((B, H), accum),
        Out: T.Tensor((H,), ts),
    ):
        with T.Kernel(H, 1, threads=threads) as (hx, _):
            total = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            total[0] = 0.0
            for b in T.serial(B):
                total[0] = total[0] + Partials[b, hx]
            Out[hx] = T.Cast(ts, total[0])

    return ssm2_reduce_bh


def make_reduce_bshn_to_bsgn_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    n_groups: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-2 head-to-group reduce ``@T.prim_func``.

    Folds per-head gradients ``(B, S, H, N)`` back to grouped shape
    ``(B, S, G, N)`` by summing the ``repeat = H//G`` heads within each group::

        Out[b, t, g, n] = sum_{r=0}^{repeat-1} Partials[b, t, g*repeat+r, n]

    Used by the backward pass to produce ``dBp`` and ``dC`` in group shape.

    Grid: ``(n_groups, seq_len, batch)``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: total number of heads ``H``.
        n_groups: number of B/C groups ``G`` (``H`` must be divisible by ``G``).
        ssm_state_size: SSM state size ``N``.
        dtype: output dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(Partials, Out)`` where
        ``Partials`` is ``(B, S, H, N, dtype)`` and ``Out`` is
        ``(B, S, G, N, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, G, N_ = batch, seq_len, num_heads, n_groups, ssm_state_size
    repeat = H // G

    @T.prim_func
    def ssm2_reduce_bshn_to_bsgn(
        Partials: T.Tensor((B, S, H, N_), ts),
        Out: T.Tensor((B, S, G, N_), ts),
    ):
        with T.Kernel(G, S, B, threads=threads) as (gx, tx, bx):
            total = T.alloc_fragment((N_,), accum)
            _head_ref = T.alloc_fragment((1,), accum)
            _head_ref[0] = H
            for n in T.Parallel(N_):
                total[n] = 0.0
            for r in T.serial(repeat):
                hx = gx * repeat + r
                for n in T.Parallel(N_):
                    total[n] = total[n] + T.Cast(accum, Partials[bx, tx, hx, n])
            for n in T.Parallel(N_):
                Out[bx, tx, gx, n] = T.Cast(ts, total[n])

    return ssm2_reduce_bshn_to_bsgn
