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

"""Tile-lang grouped matmul forward.

Computes ``out[start_g:end_g, :] = lhs[start_g:end_g, :] @ rhs[g]``.

Grid: ``(num_groups, ceildiv(m, BM), ceildiv(n, BN))``. Each CTA is
assigned a group ``g`` and an ``(m_tile, n_tile)``. It runs the dense
``lhs_tile @ rhs[g]`` GEMM but only **writes** the rows whose global
index falls inside ``[group_start[g], group_end[g])``. Because distinct
groups own disjoint row ranges, the writes never collide and the result
is exact regardless of how group boundaries align with the tiles.

Cost is ``num_groups ×`` a dense matmul — acceptable for the small group
counts (typically ≤ 16) seen in MoE / expert-routing workloads, and it
keeps the whole computation on a single tile-lang kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the tile-lang dtype string for a JAX/NumPy activation dtype.

    Raises:
        TypeError: if ``dtype`` is not float16, bfloat16 or float32.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for grouped_matmul: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    block_m: int,
    block_n: int,
    block_k: int,
    dtype,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build the grouped-matmul forward ``@T.prim_func``.

    Grid: ``(num_groups, ceildiv(m, block_m), ceildiv(n, block_n))``.
    Each CTA is assigned a group ``g`` and a ``(block_m, block_n)`` output
    tile.  It runs the full dense ``lhs_tile @ rhs[g]`` GEMM, but only
    **writes** rows whose global index falls inside the half-open interval
    ``[GroupStarts[g], GroupEnds[g])``. Rows outside the interval are
    silently discarded.

    Because distinct groups own disjoint row ranges the writes are
    conflict-free.  The cost is ``num_groups × dense_matmul``; this is
    acceptable for the small group counts (typically ≤ 16) seen in MoE.

    K-tiles are pipelined with ``num_stages`` software-pipeline stages.
    Accumulation is always float32; the result is cast to ``dtype`` on store.

    Args:
        m: total rows of ``Lhs`` and ``Y``.
        n: columns of ``Rhs`` and ``Y``.
        k: inner dimension (columns of ``Lhs``, depth of ``Rhs``).
        num_groups: number of groups ``G``.
        block_m: tile height (M dimension).
        block_n: tile width (N dimension).
        block_k: tile depth (K reduction dimension).
        dtype: activation dtype (float16 / bfloat16 / float32).
        threads: threads per CTA (default 128).
        num_stages: K-loop pipeline stages (default 2).

    Returns:
        ``@T.prim_func`` with signature
        ``(Lhs[m,k], Rhs[G,k,n], GroupStarts[G], GroupEnds[G], Y[m,n])``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    BM, BN, BK = block_m, block_n, block_k
    G = num_groups

    @T.prim_func
    def gmm_fwd(
        Lhs: T.Tensor((m, k), ts),
        Rhs: T.Tensor((G, k, n), ts),
        GroupStarts: T.Tensor((G,), "int32"),
        GroupEnds: T.Tensor((G,), "int32"),
        Y: T.Tensor((m, n), ts),
    ):
        with T.Kernel(G, T.ceildiv(m, BM), T.ceildiv(n, BN), threads=threads) as (gx, by, bx):
            Xs = T.alloc_shared((BM, BK), ts)
            Ws = T.alloc_shared((BK, BN), ts)
            C = T.alloc_fragment((BM, BN), accum)

            T.clear(C)
            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                T.copy(
                    Lhs[by * BM : (by + 1) * BM, k_iter * BK : (k_iter + 1) * BK],
                    Xs,
                )
                for ki, j in T.Parallel(BK, BN):
                    Ws[ki, j] = Rhs[gx, k_iter * BK + ki, bx * BN + j]
                T.gemm(Xs, Ws, C)

            for i, j in T.Parallel(BM, BN):
                m_idx = by * BM + i
                n_idx = bx * BN + j
                in_group = (m_idx >= GroupStarts[gx]) & (m_idx < GroupEnds[gx])
                if in_group & (m_idx < m) & (n_idx < n):
                    Y[m_idx, n_idx] = T.Cast(ts, C[i, j])

    return gmm_fwd


def make_rhs_bwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    block_k: int,
    block_n: int,
    transpose_rhs: bool,
    dtype,
    threads: int = 128,
):
    """Build the RHS grouped-matmul backward ``@T.prim_func``.

    Computes ``dRhs[g] = Lhs[group_rows].T @ dY[group_rows]`` for each
    group ``g``, accumulating over all rows that belong to ``g`` in a
    sequential loop over ``m``.

    Grid: ``(num_groups, ceildiv(k, block_k), ceildiv(n, block_n))``.  Each
    CTA owns one ``(group, k_tile, n_tile)`` and accumulates ``dRhs`` for
    that tile in a float32 fragment register.

    Group boundaries are reconstructed by scanning ``GroupSizes`` rather
    than storing explicit ``GroupStarts`` / ``GroupEnds`` buffers.

    The output layout honours ``transpose_rhs``:

    * ``transpose_rhs=False``: ``dRhs[G, k, n]``
    * ``transpose_rhs=True``:  ``dRhs[G, n, k]``

    Args:
        m: total rows of ``Lhs`` and ``dY``.
        n: columns of ``dY`` and the N dimension of ``dRhs``.
        k: inner dimension of ``Lhs``.
        num_groups: number of groups ``G``.
        block_k: tile depth (K dimension).
        block_n: tile width (N dimension).
        transpose_rhs: if True, write ``dRhs`` in transposed ``(G, n, k)`` layout.
        dtype: activation dtype.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Lhs[m,k], dY[m,n], GroupSizes[G], dRhs[G, R1, R2])``
        where ``(R1, R2) = (n, k)`` when ``transpose_rhs`` else ``(k, n)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    BK, BN = block_k, block_n
    G = num_groups
    R1 = n if transpose_rhs else k
    R2 = k if transpose_rhs else n

    @T.prim_func
    def gmm_rhs_bwd(
        Lhs: T.Tensor((m, k), ts),
        dY: T.Tensor((m, n), ts),
        GroupSizes: T.Tensor((G,), "int32"),
        dRhs: T.Tensor((G, R1, R2), ts),
    ):
        with T.Kernel(G, T.ceildiv(k, BK), T.ceildiv(n, BN), threads=threads) as (gx, kx, nx):
            acc = T.alloc_fragment((BK, BN), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            _r1_ref = T.alloc_fragment((1,), accum)
            _r2_ref = T.alloc_fragment((1,), accum)
            _r1_ref[0] = R1
            _r2_ref[0] = R2

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[gi]
            group_end[0] = group_start[0] + GroupSizes[gx]

            for ki, ni in T.Parallel(BK, BN):
                acc[ki, ni] = 0.0

            for mi in T.serial(m):
                if (mi >= group_start[0]) & (mi < group_end[0]):
                    for ki, ni in T.Parallel(BK, BN):
                        k_idx = kx * BK + ki
                        n_idx = nx * BN + ni
                        if (k_idx < k) & (n_idx < n):
                            acc[ki, ni] = acc[ki, ni] + T.Cast(accum, Lhs[mi, k_idx]) * T.Cast(accum, dY[mi, n_idx])

            for ki, ni in T.Parallel(BK, BN):
                k_idx = kx * BK + ki
                n_idx = nx * BN + ni
                if (k_idx < k) & (n_idx < n):
                    if transpose_rhs:
                        dRhs[gx, n_idx, k_idx] = T.Cast(ts, acc[ki, ni])
                    else:
                        dRhs[gx, k_idx, n_idx] = T.Cast(ts, acc[ki, ni])

    return gmm_rhs_bwd
