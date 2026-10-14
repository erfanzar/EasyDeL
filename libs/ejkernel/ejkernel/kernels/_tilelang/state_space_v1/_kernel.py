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

"""Native tile-lang SSM-1 (Mamba-style selective SSM) forward kernel.

Per-step recurrence (matches the XLA reference):

    dA[d, n]  = exp(A[d, n] * dt[d])
    dBx[d, n] = dt[d] * B[n] * x[d]
    h[d, n]   = dA[d, n] * h_prev[d, n] + dBx[d, n]
    y[d]      = sum_n h[d, n] * C[n] + D_skip[d] * x[d]

Grid: ``(ceildiv(D, BLOCK_D), batch)`` — each CTA owns ``BLOCK_D`` channels
across the full ``N`` state size and walks the time axis sequentially.

``D = intermediate_size`` (the channel axis) and ``N = ssm_state_size``
(the state axis, typically 16). The per-CTA state lives in a fragment
(``BLOCK_D * N`` fp32 ≈ 4 KB for BLOCK_D=64, N=16).
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
        raise TypeError(f"Unsupported dtype for SSM1: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    intermediate_size: int,
    ssm_state_size: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-1 (Mamba) forward ``@T.prim_func`` (no state materialisation).

    Grid: ``(ceildiv(D, BLOCK_D), batch)`` — one CTA per ``(batch, d_block)``.
    The ``(BLOCK_D, N)`` state fragment lives in registers for the lifetime
    of the CTA. Channels outside the block's valid range are masked via
    ``T.if_then_else``.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        intermediate_size: channel dimension ``D``.
        ssm_state_size: state size ``N`` (typically 16).
        block_d: channels per CTA tile ``BLOCK_D``.
        dtype: tensor dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(X, A, Bp, C, D, Dt, H0, Y, Hf)`` where:
        ``X`` is ``(B, S, D, dtype)``; ``A`` is ``(D, N, dtype)``;
        ``Bp, C`` are ``(B, S, N, dtype)``; ``D, Dt`` are channel/time
        scalars/tensors; ``H0/Hf`` are fp32 ``(B, D, N)``; ``Y`` is the
        output ``(B, S, D, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D_, N_ = batch, seq_len, intermediate_size, ssm_state_size
    BD = block_d

    @T.prim_func
    def ssm1_fwd(
        X: T.Tensor((B, S, D_), ts),
        A: T.Tensor((D_, N_), ts),
        Bp: T.Tensor((B, S, N_), ts),
        C: T.Tensor((B, S, N_), ts),
        D: T.Tensor((D_,), ts),
        Dt: T.Tensor((B, S, D_), ts),
        H0: T.Tensor((B, D_, N_), accum),
        Y: T.Tensor((B, S, D_), ts),
        Hf: T.Tensor((B, D_, N_), accum),
    ):
        with T.Kernel(T.ceildiv(D_, BD), B, threads=threads) as (dx, bx):
            h_state = T.alloc_fragment((BD, N_), accum)
            A_loc = T.alloc_fragment((BD, N_), accum)
            D_skip = T.alloc_fragment((BD,), accum)
            x_loc = T.alloc_fragment((BD,), accum)
            dt_loc = T.alloc_fragment((BD,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            dA = T.alloc_fragment((BD, N_), accum)
            dBx = T.alloc_fragment((BD, N_), accum)
            y_prod = T.alloc_fragment((BD, N_), accum)
            y_loc = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                A_loc[i, n] = T.if_then_else(
                    d_idx < D_,
                    T.Cast(accum, A[d_idx, n]),
                    0.0,
                )
            for i in T.Parallel(BD):
                d_idx = dx * BD + i
                D_skip[i] = T.if_then_else(
                    d_idx < D_,
                    T.Cast(accum, D[d_idx]),
                    0.0,
                )

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                h_state[i, n] = T.if_then_else(
                    d_idx < D_,
                    H0[bx, d_idx, n],
                    0.0,
                )

            for t in T.serial(S):
                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    x_loc[i] = T.if_then_else(
                        d_idx < D_,
                        T.Cast(accum, X[bx, t, d_idx]),
                        0.0,
                    )
                    dt_loc[i] = T.if_then_else(
                        d_idx < D_,
                        T.Cast(accum, Dt[bx, t, d_idx]),
                        0.0,
                    )
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, n])

                for i, n in T.Parallel(BD, N_):
                    dA[i, n] = T.exp(A_loc[i, n] * dt_loc[i])
                for i, n in T.Parallel(BD, N_):
                    dBx[i, n] = dt_loc[i] * Bp_loc[n] * x_loc[i]

                for i, n in T.Parallel(BD, N_):
                    h_state[i, n] = dA[i, n] * h_state[i, n] + dBx[i, n]

                for i, n in T.Parallel(BD, N_):
                    y_prod[i, n] = h_state[i, n] * C_loc[n]
                T.reduce_sum(y_prod, y_loc, dim=1, clear=True)
                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    if d_idx < D_:
                        Y[bx, t, d_idx] = T.Cast(ts, y_loc[i] + D_skip[i] * x_loc[i])

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                if d_idx < D_:
                    Hf[bx, d_idx, n] = h_state[i, n]

    return ssm1_fwd


def make_fwd_states_prim_func(
    *,
    batch: int,
    seq_len: int,
    intermediate_size: int,
    ssm_state_size: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Forward that also emits every hidden state ``Hall (B, S, D, N)``.

    Used by the ``custom_vjp`` forward rule: the backward pass needs each
    ``h_t`` and reconstructing it by dividing out ``dA`` is unstable
    (``dA = exp(A·dt)`` with negative ``A`` can be tiny), so we materialise
    the full state trajectory.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D_, N_ = batch, seq_len, intermediate_size, ssm_state_size
    BD = block_d

    @T.prim_func
    def ssm1_fwd_states(
        X: T.Tensor((B, S, D_), ts),
        A: T.Tensor((D_, N_), ts),
        Bp: T.Tensor((B, S, N_), ts),
        C: T.Tensor((B, S, N_), ts),
        D: T.Tensor((D_,), ts),
        Dt: T.Tensor((B, S, D_), ts),
        H0: T.Tensor((B, D_, N_), accum),
        Y: T.Tensor((B, S, D_), ts),
        Hf: T.Tensor((B, D_, N_), accum),
        Hall: T.Tensor((B, S, D_, N_), accum),
    ):
        with T.Kernel(T.ceildiv(D_, BD), B, threads=threads) as (dx, bx):
            h_state = T.alloc_fragment((BD, N_), accum)
            A_loc = T.alloc_fragment((BD, N_), accum)
            D_skip = T.alloc_fragment((BD,), accum)
            x_loc = T.alloc_fragment((BD,), accum)
            dt_loc = T.alloc_fragment((BD,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            dA = T.alloc_fragment((BD, N_), accum)
            y_prod = T.alloc_fragment((BD, N_), accum)
            y_loc = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                A_loc[i, n] = T.if_then_else(d_idx < D_, T.Cast(accum, A[d_idx, n]), 0.0)
            for i in T.Parallel(BD):
                d_idx = dx * BD + i
                D_skip[i] = T.if_then_else(d_idx < D_, T.Cast(accum, D[d_idx]), 0.0)
            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                h_state[i, n] = T.if_then_else(d_idx < D_, H0[bx, d_idx, n], 0.0)

            for t in T.serial(S):
                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    x_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, X[bx, t, d_idx]), 0.0)
                    dt_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, Dt[bx, t, d_idx]), 0.0)
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, n])

                for i, n in T.Parallel(BD, N_):
                    dA[i, n] = T.exp(A_loc[i, n] * dt_loc[i])
                for i, n in T.Parallel(BD, N_):
                    h_state[i, n] = dA[i, n] * h_state[i, n] + dt_loc[i] * Bp_loc[n] * x_loc[i]

                for i, n in T.Parallel(BD, N_):
                    y_prod[i, n] = h_state[i, n] * C_loc[n]
                T.reduce_sum(y_prod, y_loc, dim=1, clear=True)
                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    if d_idx < D_:
                        Y[bx, t, d_idx] = T.Cast(ts, y_loc[i] + D_skip[i] * x_loc[i])
                for i, n in T.Parallel(BD, N_):
                    d_idx = dx * BD + i
                    if d_idx < D_:
                        Hall[bx, t, d_idx, n] = h_state[i, n]

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                if d_idx < D_:
                    Hf[bx, d_idx, n] = h_state[i, n]

    return ssm1_fwd_states


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    intermediate_size: int,
    ssm_state_size: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Reverse-time adjoint scan for SSM-1.

    Given ``dY`` (and ``dHf`` for the final state), walks ``t`` from
    ``S-1`` down to ``0`` accumulating the adjoint state ``dh`` and the
    parameter gradients.

    To avoid atomics entirely, every gradient output carries the axis it
    is reduced over so each CTA — indexed by ``(d_block, batch)`` — writes
    a strictly disjoint slice:

    * ``dX`` / ``dDt`` are ``(B, S, D)`` — per ``(b, d)``;
    * ``dA_p`` / ``dD_p`` are ``(B, D, N)`` / ``(B, D)`` — per-batch partial
      (the JAX glue sums over batch);
    * ``dBp_p`` / ``dC_p`` are ``(NDB, B, S, N)`` — per-D-block partial
      (the JAX glue sums over the D-block axis).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D_, N_ = batch, seq_len, intermediate_size, ssm_state_size
    BD = block_d
    NDB = (D_ + BD - 1) // BD

    @T.prim_func
    def ssm1_bwd(
        X: T.Tensor((B, S, D_), ts),
        A: T.Tensor((D_, N_), ts),
        Bp: T.Tensor((B, S, N_), ts),
        C: T.Tensor((B, S, N_), ts),
        Dsk: T.Tensor((D_,), ts),
        Dt: T.Tensor((B, S, D_), ts),
        H0: T.Tensor((B, D_, N_), accum),
        Hall: T.Tensor((B, S, D_, N_), accum),
        dY: T.Tensor((B, S, D_), ts),
        dHf: T.Tensor((B, D_, N_), accum),
        dX: T.Tensor((B, S, D_), accum),
        dA_p: T.Tensor((B, D_, N_), accum),
        dBp_p: T.Tensor((NDB, B, S, N_), accum),
        dC_p: T.Tensor((NDB, B, S, N_), accum),
        dD_p: T.Tensor((B, D_), accum),
        dDt: T.Tensor((B, S, D_), accum),
        dH0: T.Tensor((B, D_, N_), accum),
    ):
        with T.Kernel(NDB, B, threads=threads) as (dx, bx):
            dh = T.alloc_fragment((BD, N_), accum)
            A_loc = T.alloc_fragment((BD, N_), accum)
            x_loc = T.alloc_fragment((BD,), accum)
            dt_loc = T.alloc_fragment((BD,), accum)
            dy_loc = T.alloc_fragment((BD,), accum)
            Bp_loc = T.alloc_fragment((N_,), accum)
            C_loc = T.alloc_fragment((N_,), accum)
            h_cur = T.alloc_fragment((BD, N_), accum)
            h_prev = T.alloc_fragment((BD, N_), accum)
            dA_t = T.alloc_fragment((BD, N_), accum)
            d_dA_t = T.alloc_fragment((BD, N_), accum)
            d_dBx = T.alloc_fragment((BD, N_), accum)
            dA_acc = T.alloc_fragment((BD, N_), accum)
            dD_acc = T.alloc_fragment((BD,), accum)
            tmp_n = T.alloc_fragment((N_,), accum)
            tmp_dn = T.alloc_fragment((BD, N_), accum)
            ddt_acc = T.alloc_fragment((BD,), accum)
            dx_acc = T.alloc_fragment((BD,), accum)
            Dsk_loc = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                A_loc[i, n] = T.if_then_else(d_idx < D_, T.Cast(accum, A[d_idx, n]), 0.0)
                dh[i, n] = T.if_then_else(d_idx < D_, dHf[bx, d_idx, n], 0.0)
                dA_acc[i, n] = 0.0
            for i in T.Parallel(BD):
                d_idx = dx * BD + i
                dD_acc[i] = 0.0
                Dsk_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, Dsk[d_idx]), 0.0)

            for t_rev in T.serial(S):
                t = S - 1 - t_rev
                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    x_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, X[bx, t, d_idx]), 0.0)
                    dt_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, Dt[bx, t, d_idx]), 0.0)
                    dy_loc[i] = T.if_then_else(d_idx < D_, T.Cast(accum, dY[bx, t, d_idx]), 0.0)
                for n in T.Parallel(N_):
                    Bp_loc[n] = T.Cast(accum, Bp[bx, t, n])
                    C_loc[n] = T.Cast(accum, C[bx, t, n])

                for i, n in T.Parallel(BD, N_):
                    d_idx = dx * BD + i
                    h_cur[i, n] = T.if_then_else(d_idx < D_, Hall[bx, t, d_idx, n], 0.0)
                    h_prev[i, n] = T.if_then_else(
                        t == 0,
                        T.if_then_else(d_idx < D_, H0[bx, d_idx, n], 0.0),
                        T.if_then_else(d_idx < D_, Hall[bx, T.max(t - 1, 0), d_idx, n], 0.0),
                    )
                    dA_t[i, n] = T.exp(A_loc[i, n] * dt_loc[i])

                for i, n in T.Parallel(BD, N_):
                    dh[i, n] = dh[i, n] + C_loc[n] * dy_loc[i]

                for i, n in T.Parallel(BD, N_):
                    d_dA_t[i, n] = dh[i, n] * h_prev[i, n]
                    d_dBx[i, n] = dh[i, n]

                for i, n in T.Parallel(BD, N_):
                    dA_acc[i, n] = dA_acc[i, n] + d_dA_t[i, n] * dt_loc[i] * dA_t[i, n]

                for i, n in T.Parallel(BD, N_):
                    tmp_dn[i, n] = h_cur[i, n] * dy_loc[i]
                T.reduce_sum(tmp_dn, tmp_n, dim=0, clear=True)
                for n in T.Parallel(N_):
                    dC_p[dx, bx, t, n] = tmp_n[n]

                for i, n in T.Parallel(BD, N_):
                    tmp_dn[i, n] = d_dBx[i, n] * dt_loc[i] * x_loc[i]
                T.reduce_sum(tmp_dn, tmp_n, dim=0, clear=True)
                for n in T.Parallel(N_):
                    dBp_p[dx, bx, t, n] = tmp_n[n]

                for i, n in T.Parallel(BD, N_):
                    tmp_dn[i, n] = d_dA_t[i, n] * A_loc[i, n] * dA_t[i, n] + d_dBx[i, n] * Bp_loc[n] * x_loc[i]
                T.reduce_sum(tmp_dn, ddt_acc, dim=1, clear=True)

                for i, n in T.Parallel(BD, N_):
                    tmp_dn[i, n] = d_dBx[i, n] * dt_loc[i] * Bp_loc[n]
                T.reduce_sum(tmp_dn, dx_acc, dim=1, clear=True)

                for i in T.Parallel(BD):
                    d_idx = dx * BD + i
                    if d_idx < D_:
                        dD_acc[i] = dD_acc[i] + dy_loc[i] * x_loc[i]
                        dDt[bx, t, d_idx] = ddt_acc[i]
                        dX[bx, t, d_idx] = dx_acc[i] + Dsk_loc[i] * dy_loc[i]

                for i, n in T.Parallel(BD, N_):
                    dh[i, n] = dA_t[i, n] * dh[i, n]

            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                if d_idx < D_:
                    dA_p[bx, d_idx, n] = dA_acc[i, n]
            for i in T.Parallel(BD):
                d_idx = dx * BD + i
                if d_idx < D_:
                    dD_p[bx, d_idx] = dD_acc[i]
            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                if d_idx < D_:
                    dH0[bx, d_idx, n] = dh[i, n]

    return ssm1_bwd


def make_init_state_prim_func(
    *,
    batch: int,
    seq_len: int,
    intermediate_size: int,
    ssm_state_size: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-1 zero-state initialiser ``@T.prim_func``.

    Grid: ``(ceildiv(D, BLOCK_D), batch)``. Writes zero fp32 ``(B, D, N)``
    state. ``seq_len`` is baked in only for dtype-probe allocation.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S`` (dtype-probe only).
        intermediate_size: channel dimension ``D``.
        ssm_state_size: state size ``N``.
        block_d: channels per CTA tile.
        dtype: input tensor dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(X, H0)``; ``X`` is read-only and
        used for type inference.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D_, N_, BD = batch, seq_len, intermediate_size, ssm_state_size, block_d

    @T.prim_func
    def ssm1_init_state(
        X: T.Tensor((B, S, D_), ts),
        H0: T.Tensor((B, D_, N_), accum),
    ):
        with T.Kernel(T.ceildiv(D_, BD), B, threads=threads) as (dx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = X[0, 0, 0]
            _seq_ref[0] = S
            for i, n in T.Parallel(BD, N_):
                d_idx = dx * BD + i
                if d_idx < D_:
                    H0[bx, d_idx, n] = 0.0

    return ssm1_init_state


def make_reduce_bdn_prim_func(
    *,
    batch: int,
    intermediate_size: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-1 batch-reduce ``@T.prim_func`` for ``(B, D, N)`` partials.

    Sums ``Partials`` over the batch axis: ``Out[d, n] = sum_b Partials[b, d, n]``.
    Used to reduce the per-batch partial gradient for ``A``.

    Grid: ``(D, N)``.

    Args:
        batch: batch size ``B``.
        intermediate_size: channel dimension ``D``.
        ssm_state_size: state size ``N``.
        dtype: output dtype (float16 / bfloat16 / float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(Partials, Out)`` where
        ``Partials`` is fp32 ``(B, D, N)`` and ``Out`` is ``(D, N, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, D_, N_ = batch, intermediate_size, ssm_state_size

    @T.prim_func
    def ssm1_reduce_bdn(
        Partials: T.Tensor((B, D_, N_), accum),
        Out: T.Tensor((D_, N_), ts),
    ):
        with T.Kernel(D_, N_, threads=threads) as (dx, nx):
            total = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            total[0] = 0.0
            for b in T.serial(B):
                total[0] = total[0] + Partials[b, dx, nx]
            Out[dx, nx] = T.Cast(ts, total[0])

    return ssm1_reduce_bdn


def make_reduce_bd_prim_func(
    *,
    batch: int,
    intermediate_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-1 batch-reduce ``@T.prim_func`` for ``(B, D)`` partials.

    Sums ``Partials`` over the batch axis: ``Out[d] = sum_b Partials[b, d]``.
    Used to reduce the per-batch partial gradient for ``D`` (skip connection).

    Grid: ``(D, 1)``.

    Args:
        batch: batch size ``B``.
        intermediate_size: channel dimension ``D``.
        dtype: output dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(Partials, Out)`` where
        ``Partials`` is fp32 ``(B, D)`` and ``Out`` is ``(D, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, D_ = batch, intermediate_size

    @T.prim_func
    def ssm1_reduce_bd(
        Partials: T.Tensor((B, D_), accum),
        Out: T.Tensor((D_,), ts),
    ):
        with T.Kernel(D_, 1, threads=threads) as (dx, _):
            total = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            total[0] = 0.0
            for b in T.serial(B):
                total[0] = total[0] + Partials[b, dx]
            Out[dx] = T.Cast(ts, total[0])

    return ssm1_reduce_bd


def make_reduce_ndb_bsn_prim_func(
    *,
    num_d_blocks: int,
    batch: int,
    seq_len: int,
    ssm_state_size: int,
    dtype,
    threads: int = 128,
):
    """Build the SSM-1 d-block reduce ``@T.prim_func`` for ``(NDB, B, S, N)`` partials.

    Sums ``Partials`` over the ``num_d_blocks`` axis:
    ``Out[b, t, n] = sum_db Partials[db, b, t, n]``.
    Used to reduce the per-D-block partial gradients for ``Bp`` and ``C``.

    Grid: ``(B, S, N)``.

    Args:
        num_d_blocks: number of D-blocks (= ``ceildiv(D, BLOCK_D)``).
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        ssm_state_size: state size ``N``.
        dtype: output dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(Partials, Out)`` where
        ``Partials`` is fp32 ``(NDB, B, S, N)`` and ``Out`` is
        ``(B, S, N, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NDB, B, S, N_ = num_d_blocks, batch, seq_len, ssm_state_size

    @T.prim_func
    def ssm1_reduce_ndb_bsn(
        Partials: T.Tensor((NDB, B, S, N_), accum),
        Out: T.Tensor((B, S, N_), ts),
    ):
        with T.Kernel(B, S, N_, threads=threads) as (bx, sx, nx):
            total = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            total[0] = 0.0
            for db in T.serial(NDB):
                total[0] = total[0] + Partials[db, bx, sx, nx]
            Out[bx, sx, nx] = T.Cast(ts, total[0])

    return ssm1_reduce_ndb_bsn
