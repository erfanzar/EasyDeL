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

"""TileLang prim_funcs for grouped matmul v3 — forward and four backward kernels.

The five factory functions correspond to distinct compilation units:

* :func:`make_fwd_prim_func` — forward GEMM with optional scale, bias,
  and existing-output accumulation.
* :func:`make_lhs_bwd_prim_func` — gradient w.r.t. ``lhs``
  (``dLhs = dY @ rhs^T * scale``).
* :func:`make_rhs_bwd_prim_func` — gradient w.r.t. ``rhs``
  (``dRhs[g] = lhs[g_rows]^T @ dY[g_rows] * scale[g]``).
* :func:`make_scale_bwd_prim_func` — gradient w.r.t. ``rhs_scale``.
* :func:`make_bias_bwd_prim_func` — gradient w.r.t. ``rhs_bias``
  (column-wise row-sum of ``dY`` per group).

Grid layout (forward kernel):
    ``Kernel(G, ceil(m/BM), ceil(n/BN))`` — one block per
    (group, m-tile, n-tile) triplet.  Pipelining: 2-stage software pipeline
    on the k-reduction loop.

Shared memory per block (forward):
    ``Xs (BM, BK)`` and ``Ws (BK, BN)`` tiles in the input dtype ``ts``.
    Fragment accumulators ``C (BM, BN)`` in float32.

All kernels accumulate in float32 (``accum = "float32"``); inputs and
outputs are in the dtype supplied to the factory function.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Convert a NumPy/JAX dtype to a TileLang type-string.

    Args:
        dtype: A dtype understood by ``jnp.dtype``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: For any dtype not in the supported set.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for grouped_matmulv3: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    group_sizes_len: int,
    group_offset_size: int,
    num_scale_blocks: int,
    block_m: int,
    block_n: int,
    block_k: int,
    transpose_rhs: bool,
    has_scale: bool,
    has_bias: bool,
    has_existing_out: bool,
    use_group_offset: bool,
    dtype,
    scale_dtype,
    bias_dtype,
    existing_dtype,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build the TileLang forward prim_func for grouped matmul v3.

    Performs ``Y[g_rows] = Lhs[g_rows] @ Rhs[g] * Scale[g] + Bias[g]
    + ExistingOut[g_rows]`` for each group ``g``.

    Group membership is determined at runtime from ``GroupSizes`` (and
    optionally ``GroupOffset``): each block computes a ``(BM, BN)`` tile of
    the output, checks whether its rows belong to the current group ``gx``,
    and writes the result only for in-group rows.

    Grid layout:
        ``Kernel(G, ceil(m/BM), ceil(n/BN), threads=threads)``
        with axis names ``(gx, by, bx)``.

    Shared memory per block:
        ``Xs (BM, BK)`` — lhs tile in ``ts``
        ``Ws (BK, BN)`` — rhs tile in ``ts``

    Fragment registers per block:
        ``C (BM, BN)`` float32 accumulator.

    Software pipelining:
        ``num_stages``-stage pipeline on the k-reduction loop.

    Args:
        m: Number of rows in ``lhs`` and ``output``.
        n: Number of columns in ``rhs`` and ``output``.
        k: Contraction dimension (inner dimension of both ``lhs`` and ``rhs``).
        num_groups: Number of expert/weight groups ``G``.
        group_sizes_len: Length of the ``GroupSizes`` tensor ``GS``.
        group_offset_size: Length of the ``GroupOffset`` tensor ``GO``
            (1 when ``use_group_offset=False``).
        num_scale_blocks: Number of blocks along ``k`` for ``RhsScale``
            (``NB``).  ``k`` must be divisible by ``NB``.
        block_m: Tile height ``BM``.
        block_n: Tile width ``BN``.
        block_k: Tile depth ``BK`` (k-reduction tile).
        transpose_rhs: If ``True``, ``Rhs`` is stored as ``(G, n, k)``.
        has_scale: Whether to apply ``RhsScale`` element-wise to ``Rhs``.
        has_bias: Whether to add ``RhsBias`` to the output.
        has_existing_out: Whether to accumulate into ``ExistingOut``.
        use_group_offset: Whether to read group boundaries from
            ``GroupOffset``.
        dtype: Element dtype of ``Lhs``, ``Rhs``, and ``Y``.
        scale_dtype: Element dtype of ``RhsScale``.
        bias_dtype: Element dtype of ``RhsBias``.
        existing_dtype: Element dtype of ``ExistingOut``.
        threads: CUDA threads per block (default 128).
        num_stages: Software-pipeline depth for the k-loop (default 2).

    Returns:
        A ``T.prim_func`` with signature::

            gmmv3_fwd(
                Lhs       : [m, k]           <ts>
                Rhs       : [G, R1, R2]      <ts>   (R1/R2 depend on transpose)
                GroupSizes: [GS]             int32
                GroupOffset:[GO]             int32
                RhsScale  : [G, NB, 1, n]   <scale_ts>
                RhsBias   : [G, 1, n]        <bias_ts>
                ExistingOut: [m, n]          <existing_ts>
                Y         : [m, n]           <ts>   (output)
            )
    """
    ts = _dtype_str(dtype)
    scale_ts = _dtype_str(scale_dtype)
    bias_ts = _dtype_str(bias_dtype)
    existing_ts = _dtype_str(existing_dtype)
    accum = "float32"
    BM, BN, BK = block_m, block_n, block_k
    G, GS, GO, NB = num_groups, group_sizes_len, group_offset_size, num_scale_blocks
    R1 = n if transpose_rhs else k
    R2 = k if transpose_rhs else n
    scale_block = k // NB

    @T.prim_func
    def gmmv3_fwd(
        Lhs: T.Tensor((m, k), ts),
        Rhs: T.Tensor((G, R1, R2), ts),
        GroupSizes: T.Tensor((GS,), "int32"),
        GroupOffset: T.Tensor((GO,), "int32"),
        RhsScale: T.Tensor((G, NB, 1, n), scale_ts),
        RhsBias: T.Tensor((G, 1, n), bias_ts),
        ExistingOut: T.Tensor((m, n), existing_ts),
        Y: T.Tensor((m, n), ts),
    ):
        """TileLang kernel: grouped matmul v3 forward.

        See :func:`make_fwd_prim_func` for the full algorithm and grid layout.
        """
        with T.Kernel(G, T.ceildiv(m, BM), T.ceildiv(n, BN), threads=threads) as (gx, by, bx):
            Xs = T.alloc_shared((BM, BK), ts)
            Ws = T.alloc_shared((BK, BN), ts)
            C = T.alloc_fragment((BM, BN), accum)
            Wacc = T.alloc_fragment((BK, BN), accum)
            Out = T.alloc_fragment((BM, BN), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            offset = T.alloc_fragment((1,), "int32")
            _nb_ref = T.alloc_fragment((1,), accum)
            _r1_ref = T.alloc_fragment((1,), accum)
            _r2_ref = T.alloc_fragment((1,), accum)
            _gs_ref = T.alloc_fragment((1,), accum)
            _go_ref = T.alloc_fragment((1,), accum)
            _scale_ts_ref = T.alloc_fragment((1,), scale_ts)
            _bias_ts_ref = T.alloc_fragment((1,), bias_ts)
            _existing_ts_ref = T.alloc_fragment((1,), existing_ts)
            _nb_ref[0] = NB
            _r1_ref[0] = R1
            _r2_ref[0] = R2
            _gs_ref[0] = GS
            _go_ref[0] = GO
            _scale_ts_ref[0] = RhsScale[0, 0, 0, 0]
            _bias_ts_ref[0] = RhsBias[0, 0, 0]
            _existing_ts_ref[0] = ExistingOut[0, 0]
            offset[0] = 0
            if use_group_offset:
                offset[0] = GroupOffset[0]

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[offset[0] + gi]
            group_end[0] = group_start[0] + GroupSizes[offset[0] + gx]

            T.clear(C)
            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                for i, kk in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = k_iter * BK + kk
                    if (m_idx < m) & (k_idx < k):
                        Xs[i, kk] = Lhs[m_idx, k_idx]
                    else:
                        Xs[i, kk] = T.Cast(ts, 0.0)
                for kk, j in T.Parallel(BK, BN):
                    k_idx = k_iter * BK + kk
                    n_idx = bx * BN + j
                    Wacc[kk, j] = 0.0
                    if (k_idx < k) & (n_idx < n):
                        if transpose_rhs:
                            Wacc[kk, j] = T.Cast(accum, Rhs[gx, n_idx, k_idx])
                        else:
                            Wacc[kk, j] = T.Cast(accum, Rhs[gx, k_idx, n_idx])
                        if has_scale:
                            scale_idx = k_idx // scale_block
                            Wacc[kk, j] = Wacc[kk, j] * T.Cast(accum, RhsScale[gx, scale_idx, 0, n_idx])
                    Ws[kk, j] = T.Cast(ts, Wacc[kk, j])
                T.gemm(Xs, Ws, C)

            for i, j in T.Parallel(BM, BN):
                m_idx = by * BM + i
                n_idx = bx * BN + j
                in_group = (m_idx >= group_start[0]) & (m_idx < group_end[0])
                if in_group & (m_idx < m) & (n_idx < n):
                    Out[i, j] = C[i, j]
                    if has_bias:
                        Out[i, j] = Out[i, j] + T.Cast(accum, RhsBias[gx, 0, n_idx])
                    if has_existing_out:
                        Out[i, j] = Out[i, j] + T.Cast(accum, ExistingOut[m_idx, n_idx])
                    Y[m_idx, n_idx] = T.Cast(ts, Out[i, j])

    return gmmv3_fwd


def make_lhs_bwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    group_sizes_len: int,
    group_offset_size: int,
    num_scale_blocks: int,
    block_m: int,
    block_k: int,
    transpose_rhs: bool,
    has_scale: bool,
    use_group_offset: bool,
    dtype,
    scale_dtype,
    threads: int = 128,
):
    """Build the TileLang lhs-gradient prim_func for grouped matmul v3.

    Computes ``dLhs[g_rows, k_idx] = sum_j dY[g_rows, j] * Rhs[g, k_idx, j]
    * Scale[g, k_idx//scale_block, 0, j]`` for each group ``g``.

    Grid layout:
        ``Kernel(G, ceil(m/BM), ceil(k/BK), threads=threads)``

    Fragment registers per block:
        ``acc (BM, BK)`` float32 accumulator over the n-dimension.

    Note:
        The n-reduction is done in a serial loop over individual n-indices,
        which may be slow for large ``n``.  This is acceptable because
        backward kernels are called less frequently than the forward pass
        in a typical training step.

    Args:
        m: Number of rows in ``lhs``.
        n: Number of columns in the output dimension.
        k: Contraction dimension (inner dimension).
        num_groups: Number of expert/weight groups ``G``.
        group_sizes_len: Length of ``GroupSizes`` tensor.
        group_offset_size: Length of ``GroupOffset`` tensor.
        num_scale_blocks: Number of k-blocks in ``RhsScale``.
        block_m: Tile height ``BM`` for the m-dimension.
        block_k: Tile depth ``BK`` for the k-dimension.
        transpose_rhs: Must match the forward kernel's flag.
        has_scale: Whether ``RhsScale`` is applied to ``Rhs``.
        use_group_offset: Whether group boundaries use ``GroupOffset``.
        dtype: Element dtype of input/output tensors.
        scale_dtype: Element dtype of ``RhsScale``.
        threads: CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gmmv3_lhs_bwd(
                dY        : [m, n]           <ts>
                Rhs       : [G, R1, R2]      <ts>
                GroupSizes: [GS]             int32
                GroupOffset:[GO]             int32
                RhsScale  : [G, NB, 1, n]   <scale_ts>
                dLhs      : [m, k]           <ts>   (output)
            )
    """
    ts = _dtype_str(dtype)
    scale_ts = _dtype_str(scale_dtype)
    accum = "float32"
    BM, BK = block_m, block_k
    G, GS, GO, NB = num_groups, group_sizes_len, group_offset_size, num_scale_blocks
    R1 = n if transpose_rhs else k
    R2 = k if transpose_rhs else n
    scale_block = k // NB

    @T.prim_func
    def gmmv3_lhs_bwd(
        dY: T.Tensor((m, n), ts),
        Rhs: T.Tensor((G, R1, R2), ts),
        GroupSizes: T.Tensor((GS,), "int32"),
        GroupOffset: T.Tensor((GO,), "int32"),
        RhsScale: T.Tensor((G, NB, 1, n), scale_ts),
        dLhs: T.Tensor((m, k), ts),
    ):
        """TileLang kernel: gradient of the matmul w.r.t. ``Lhs``.

        See :func:`make_lhs_bwd_prim_func` for the full algorithm.
        """
        with T.Kernel(G, T.ceildiv(m, BM), T.ceildiv(k, BK), threads=threads) as (gx, by, kx):
            acc = T.alloc_fragment((BM, BK), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            offset = T.alloc_fragment((1,), "int32")
            _scale_ref = T.alloc_fragment((1,), scale_ts)
            _r1_ref = T.alloc_fragment((1,), accum)
            _r2_ref = T.alloc_fragment((1,), accum)
            _nb_ref = T.alloc_fragment((1,), accum)
            _gs_ref = T.alloc_fragment((1,), accum)
            _go_ref = T.alloc_fragment((1,), accum)
            _scale_ref[0] = RhsScale[0, 0, 0, 0]
            _r1_ref[0] = R1
            _r2_ref[0] = R2
            _nb_ref[0] = NB
            _gs_ref[0] = GS
            _go_ref[0] = GO
            offset[0] = 0
            if use_group_offset:
                offset[0] = GroupOffset[0]

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[offset[0] + gi]
            group_end[0] = group_start[0] + GroupSizes[offset[0] + gx]

            for i, kk in T.Parallel(BM, BK):
                acc[i, kk] = 0.0

            for n_idx in T.serial(n):
                for i, kk in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = kx * BK + kk
                    in_group = (m_idx >= group_start[0]) & (m_idx < group_end[0])
                    if in_group & (m_idx < m) & (k_idx < k):
                        w = T.alloc_fragment((1,), accum)
                        if transpose_rhs:
                            w[0] = T.Cast(accum, Rhs[gx, n_idx, k_idx])
                        else:
                            w[0] = T.Cast(accum, Rhs[gx, k_idx, n_idx])
                        if has_scale:
                            w[0] = w[0] * T.Cast(accum, RhsScale[gx, k_idx // scale_block, 0, n_idx])
                        acc[i, kk] = acc[i, kk] + T.Cast(accum, dY[m_idx, n_idx]) * w[0]

            for i, kk in T.Parallel(BM, BK):
                m_idx = by * BM + i
                k_idx = kx * BK + kk
                in_group = (m_idx >= group_start[0]) & (m_idx < group_end[0])
                if in_group & (m_idx < m) & (k_idx < k):
                    dLhs[m_idx, k_idx] = T.Cast(ts, acc[i, kk])

    return gmmv3_lhs_bwd


def make_rhs_bwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    group_sizes_len: int,
    group_offset_size: int,
    num_scale_blocks: int,
    block_k: int,
    block_n: int,
    transpose_rhs: bool,
    has_scale: bool,
    use_group_offset: bool,
    dtype,
    scale_dtype,
    threads: int = 128,
):
    """Build the TileLang rhs-gradient prim_func for grouped matmul v3.

    Computes ``dRhs[g, k_idx, n_idx] = sum_{m in g_rows} Lhs[m, k_idx]
    * dY[m, n_idx] * Scale[g, k_idx//scale_block, 0, n_idx]``.

    Output shape: ``(G, k, n)`` when ``transpose_rhs=False``,
    or ``(G, n, k)`` when ``transpose_rhs=True``.

    Grid layout:
        ``Kernel(G, ceil(k/BK), ceil(n/BN), threads=threads)``

    Fragment registers per block:
        ``acc (BK, BN)`` float32 accumulator over the m-dimension.

    Args:
        m: Number of rows in ``lhs``.
        n: Number of columns in the output dimension.
        k: Contraction dimension.
        num_groups: Number of groups ``G``.
        group_sizes_len: Length of ``GroupSizes`` tensor.
        group_offset_size: Length of ``GroupOffset`` tensor.
        num_scale_blocks: Number of k-blocks in ``RhsScale``.
        block_k: Tile depth for the k-dimension.
        block_n: Tile width for the n-dimension.
        transpose_rhs: Must match the forward kernel's flag.
        has_scale: Whether ``RhsScale`` is applied.
        use_group_offset: Whether group boundaries use ``GroupOffset``.
        dtype: Element dtype of input/output tensors.
        scale_dtype: Element dtype of ``RhsScale``.
        threads: CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gmmv3_rhs_bwd(
                Lhs       : [m, k]           <ts>
                dY        : [m, n]           <ts>
                GroupSizes: [GS]             int32
                GroupOffset:[GO]             int32
                RhsScale  : [G, NB, 1, n]   <scale_ts>
                dRhs      : [G, R1, R2]      <ts>   (output)
            )
    """
    ts = _dtype_str(dtype)
    scale_ts = _dtype_str(scale_dtype)
    accum = "float32"
    BK, BN = block_k, block_n
    G, GS, GO, NB = num_groups, group_sizes_len, group_offset_size, num_scale_blocks
    R1 = n if transpose_rhs else k
    R2 = k if transpose_rhs else n
    scale_block = k // NB

    @T.prim_func
    def gmmv3_rhs_bwd(
        Lhs: T.Tensor((m, k), ts),
        dY: T.Tensor((m, n), ts),
        GroupSizes: T.Tensor((GS,), "int32"),
        GroupOffset: T.Tensor((GO,), "int32"),
        RhsScale: T.Tensor((G, NB, 1, n), scale_ts),
        dRhs: T.Tensor((G, R1, R2), ts),
    ):
        """TileLang kernel: gradient of the matmul w.r.t. ``Rhs``.

        See :func:`make_rhs_bwd_prim_func` for the full algorithm.
        """
        with T.Kernel(G, T.ceildiv(k, BK), T.ceildiv(n, BN), threads=threads) as (gx, kx, nx):
            acc = T.alloc_fragment((BK, BN), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            offset = T.alloc_fragment((1,), "int32")
            _scale_ref = T.alloc_fragment((1,), scale_ts)
            _r1_ref = T.alloc_fragment((1,), accum)
            _r2_ref = T.alloc_fragment((1,), accum)
            _nb_ref = T.alloc_fragment((1,), accum)
            _gs_ref = T.alloc_fragment((1,), accum)
            _go_ref = T.alloc_fragment((1,), accum)
            _scale_ref[0] = RhsScale[0, 0, 0, 0]
            _r1_ref[0] = R1
            _r2_ref[0] = R2
            _nb_ref[0] = NB
            _gs_ref[0] = GS
            _go_ref[0] = GO
            offset[0] = 0
            if use_group_offset:
                offset[0] = GroupOffset[0]

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[offset[0] + gi]
            group_end[0] = group_start[0] + GroupSizes[offset[0] + gx]

            for kk, j in T.Parallel(BK, BN):
                acc[kk, j] = 0.0

            for mi in T.serial(m):
                if (mi >= group_start[0]) & (mi < group_end[0]):
                    for kk, j in T.Parallel(BK, BN):
                        k_idx = kx * BK + kk
                        n_idx = nx * BN + j
                        if (k_idx < k) & (n_idx < n):
                            scale = T.alloc_fragment((1,), accum)
                            scale[0] = 1.0
                            if has_scale:
                                scale[0] = T.Cast(accum, RhsScale[gx, k_idx // scale_block, 0, n_idx])
                            acc[kk, j] = (
                                acc[kk, j] + T.Cast(accum, Lhs[mi, k_idx]) * T.Cast(accum, dY[mi, n_idx]) * scale[0]
                            )

            for kk, j in T.Parallel(BK, BN):
                k_idx = kx * BK + kk
                n_idx = nx * BN + j
                if (k_idx < k) & (n_idx < n):
                    if transpose_rhs:
                        dRhs[gx, n_idx, k_idx] = T.Cast(ts, acc[kk, j])
                    else:
                        dRhs[gx, k_idx, n_idx] = T.Cast(ts, acc[kk, j])

    return gmmv3_rhs_bwd


def make_scale_bwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    group_sizes_len: int,
    group_offset_size: int,
    num_scale_blocks: int,
    block_n: int,
    transpose_rhs: bool,
    use_group_offset: bool,
    dtype,
    scale_dtype,
    threads: int = 128,
):
    """Build the TileLang scale-gradient prim_func for grouped matmul v3.

    For each group ``g`` and scale block ``sb``, computes::

        dScale[g, sb, 0, n_idx] = sum_{m in g_rows} sum_{k in block_sb}
            Lhs[m, k] * Rhs[g, k, n_idx] * dY[m, n_idx]

    Grid layout:
        ``Kernel(G, NB, ceil(n/BN), threads=threads)``

    Fragment registers per block:
        ``acc (BN,)`` float32 accumulator over m and k.

    Args:
        m: Number of rows in ``lhs``.
        n: Number of columns.
        k: Contraction dimension.
        num_groups: Number of groups ``G``.
        group_sizes_len: Length of ``GroupSizes`` tensor.
        group_offset_size: Length of ``GroupOffset`` tensor.
        num_scale_blocks: Number of k-blocks in ``RhsScale`` (``NB``).
        block_n: Tile width for the n-dimension.
        transpose_rhs: Must match the forward kernel's flag.
        use_group_offset: Whether group boundaries use ``GroupOffset``.
        dtype: Element dtype of ``Lhs``, ``Rhs``, and ``dY``.
        scale_dtype: Element dtype of ``dScale``.
        threads: CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gmmv3_scale_bwd(
                Lhs       : [m, k]           <ts>
                Rhs       : [G, R1, R2]      <ts>
                dY        : [m, n]           <ts>
                GroupSizes: [GS]             int32
                GroupOffset:[GO]             int32
                dScale    : [G, NB, 1, n]   <scale_ts>  (output)
            )
    """
    ts = _dtype_str(dtype)
    scale_ts = _dtype_str(scale_dtype)
    accum = "float32"
    BN = block_n
    G, GS, GO, NB = num_groups, group_sizes_len, group_offset_size, num_scale_blocks
    R1 = n if transpose_rhs else k
    R2 = k if transpose_rhs else n
    scale_block = k // NB

    @T.prim_func
    def gmmv3_scale_bwd(
        Lhs: T.Tensor((m, k), ts),
        Rhs: T.Tensor((G, R1, R2), ts),
        dY: T.Tensor((m, n), ts),
        GroupSizes: T.Tensor((GS,), "int32"),
        GroupOffset: T.Tensor((GO,), "int32"),
        dScale: T.Tensor((G, NB, 1, n), scale_ts),
    ):
        """TileLang kernel: gradient of the matmul w.r.t. ``RhsScale``.

        See :func:`make_scale_bwd_prim_func` for the full algorithm.
        """
        with T.Kernel(G, NB, T.ceildiv(n, BN), threads=threads) as (gx, sb, nx):
            acc = T.alloc_fragment((BN,), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            offset = T.alloc_fragment((1,), "int32")
            _r1_ref = T.alloc_fragment((1,), accum)
            _r2_ref = T.alloc_fragment((1,), accum)
            _scale_ref = T.alloc_fragment((1,), scale_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gs_ref = T.alloc_fragment((1,), accum)
            _go_ref = T.alloc_fragment((1,), accum)
            _r1_ref[0] = R1
            _r2_ref[0] = R2
            _scale_ref[0] = dScale[0, 0, 0, 0]
            _ts_ref[0] = Lhs[0, 0]
            _gs_ref[0] = GS
            _go_ref[0] = GO
            offset[0] = 0
            if use_group_offset:
                offset[0] = GroupOffset[0]

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[offset[0] + gi]
            group_end[0] = group_start[0] + GroupSizes[offset[0] + gx]

            for j in T.Parallel(BN):
                acc[j] = 0.0

            for k_local in T.serial(scale_block):
                k_idx = sb * scale_block + k_local
                for mi in T.serial(m):
                    if (mi >= group_start[0]) & (mi < group_end[0]) & (k_idx < k):
                        for j in T.Parallel(BN):
                            n_idx = nx * BN + j
                            if n_idx < n:
                                w = T.alloc_fragment((1,), accum)
                                if transpose_rhs:
                                    w[0] = T.Cast(accum, Rhs[gx, n_idx, k_idx])
                                else:
                                    w[0] = T.Cast(accum, Rhs[gx, k_idx, n_idx])
                                acc[j] = acc[j] + T.Cast(accum, Lhs[mi, k_idx]) * T.Cast(accum, dY[mi, n_idx]) * w[0]

            for j in T.Parallel(BN):
                n_idx = nx * BN + j
                if n_idx < n:
                    dScale[gx, sb, 0, n_idx] = T.Cast(scale_ts, acc[j])

    return gmmv3_scale_bwd


def make_bias_bwd_prim_func(
    *,
    m: int,
    n: int,
    num_groups: int,
    group_sizes_len: int,
    group_offset_size: int,
    block_n: int,
    use_group_offset: bool,
    dtype,
    bias_dtype,
    threads: int = 128,
):
    """Build the TileLang bias-gradient prim_func for grouped matmul v3.

    Computes ``dBias[g, 0, n_idx] = sum_{m in g_rows} dY[m, n_idx]``.

    Grid layout:
        ``Kernel(G, ceil(n/BN), threads=threads)``

    Fragment registers per block:
        ``acc (BN,)`` float32 accumulator over the m-dimension.

    Args:
        m: Number of rows in ``dY``.
        n: Number of columns.
        num_groups: Number of groups ``G``.
        group_sizes_len: Length of ``GroupSizes`` tensor.
        group_offset_size: Length of ``GroupOffset`` tensor.
        block_n: Tile width for the n-dimension.
        use_group_offset: Whether group boundaries use ``GroupOffset``.
        dtype: Element dtype of ``dY``.
        bias_dtype: Element dtype of ``dBias``.
        threads: CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gmmv3_bias_bwd(
                dY        : [m, n]       <ts>
                GroupSizes: [GS]         int32
                GroupOffset:[GO]         int32
                dBias     : [G, 1, n]    <bias_ts>  (output)
            )
    """
    ts = _dtype_str(dtype)
    bias_ts = _dtype_str(bias_dtype)
    accum = "float32"
    BN = block_n
    G, GS, GO = num_groups, group_sizes_len, group_offset_size

    @T.prim_func
    def gmmv3_bias_bwd(
        dY: T.Tensor((m, n), ts),
        GroupSizes: T.Tensor((GS,), "int32"),
        GroupOffset: T.Tensor((GO,), "int32"),
        dBias: T.Tensor((G, 1, n), bias_ts),
    ):
        """TileLang kernel: gradient of the matmul w.r.t. ``RhsBias``.

        See :func:`make_bias_bwd_prim_func` for the full algorithm.
        """
        with T.Kernel(G, T.ceildiv(n, BN), threads=threads) as (gx, nx):
            acc = T.alloc_fragment((BN,), accum)
            group_start = T.alloc_fragment((1,), "int32")
            group_end = T.alloc_fragment((1,), "int32")
            offset = T.alloc_fragment((1,), "int32")
            _bias_ref = T.alloc_fragment((1,), bias_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _gs_ref = T.alloc_fragment((1,), accum)
            _go_ref = T.alloc_fragment((1,), accum)
            _bias_ref[0] = dBias[0, 0, 0]
            _ts_ref[0] = dY[0, 0]
            _gs_ref[0] = GS
            _go_ref[0] = GO
            offset[0] = 0
            if use_group_offset:
                offset[0] = GroupOffset[0]

            group_start[0] = 0
            for gi in T.serial(G):
                if gi < gx:
                    group_start[0] = group_start[0] + GroupSizes[offset[0] + gi]
            group_end[0] = group_start[0] + GroupSizes[offset[0] + gx]

            for j in T.Parallel(BN):
                acc[j] = 0.0

            for mi in T.serial(m):
                if (mi >= group_start[0]) & (mi < group_end[0]):
                    for j in T.Parallel(BN):
                        n_idx = nx * BN + j
                        if n_idx < n:
                            acc[j] = acc[j] + T.Cast(accum, dY[mi, n_idx])

            for j in T.Parallel(BN):
                n_idx = nx * BN + j
                if n_idx < n:
                    dBias[gx, 0, n_idx] = T.Cast(bias_ts, acc[j])

    return gmmv3_bias_bwd
