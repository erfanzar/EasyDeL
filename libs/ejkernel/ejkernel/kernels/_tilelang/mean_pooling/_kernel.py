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

"""tile-lang prim_func factories for mean-pooling (forward + backward).

Forward:
    ``y[b, d] = mean_s(x[b, s, d])`` — one CTA per ``(b, d_block)``, looping
    over the sequence axis with software pipelining.

Backward (broadcast scatter):
    ``dx[b, s, d] = dy[b, d] / seq_len`` — one CTA per ``(b, s_block, d_block)``.

The kernels are written for the padded layout ``(B, S, D)``. The ragged
``cu_seqlens`` mode uses separate packed-sequence kernels below.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Map a NumPy/JAX dtype to the TileLang dtype string for mean_pooling.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()`` — float16, bfloat16, float32.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Raises:
        TypeError: If *dtype* is not one of the three supported floating-point types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang mean_pooling: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    hidden_dim: int,
    block_s: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build the mean-pool forward ``@T.prim_func``.

    Grid: ``(ceildiv(hidden_dim, BLOCK_D), batch)``.

    Each CTA handles one ``(batch_idx, hidden_slab)`` pair.  It streams the
    full sequence in chunks of ``BLOCK_S`` using a 2-stage software pipeline
    (``T.Pipelined``), accumulates ``sum_s x[b, s, d]`` into a float32
    fragment, and writes ``Y[b, d] = cast(sum * (1/seq_len), dtype)``.
    Out-of-bounds positions (``s >= S`` or ``d >= D``) are zeroed before
    accumulation.

    Args:
        batch: Batch dimension ``B``.
        seq_len: Sequence length ``S``.
        hidden_dim: Feature dimension ``D``.
        block_s: Tile size along ``S``; determines pipeline chunk size.
        block_d: Tile size along ``D``; determines CTA width along the
            feature axis.
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: Number of CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(X: [B, S, D], Y: [B, D])``.
        ``Y`` is written in *dtype*; internal accumulation uses float32.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D = batch, seq_len, hidden_dim
    BS, BD = block_s, block_d
    inv_seq_len = 1.0 / float(seq_len)

    @T.prim_func
    def mean_pool_fwd(
        X: T.Tensor((B, S, D), ts),
        Y: T.Tensor((B, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), B, threads=threads) as (dx, bx):
            x_chunk = T.alloc_fragment((BS, BD), accum)
            chunk_sum = T.alloc_fragment((BD,), accum)
            sum_local = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            T.fill(sum_local, 0)

            for s_iter in T.Pipelined(T.ceildiv(S, BS), num_stages=2):
                for i, j in T.Parallel(BS, BD):
                    s_idx = s_iter * BS + i
                    d_idx = dx * BD + j
                    x_chunk[i, j] = T.if_then_else(
                        (s_idx < S) & (d_idx < D),
                        T.Cast(accum, X[bx, s_idx, d_idx]),
                        0.0,
                    )
                T.reduce_sum(x_chunk, chunk_sum, dim=0, clear=True)
                for j in T.Parallel(BD):
                    sum_local[j] = sum_local[j] + chunk_sum[j]

            for j in T.Parallel(BD):
                d_idx = dx * BD + j
                if d_idx < D:
                    Y[bx, d_idx] = T.Cast(ts, sum_local[j] * inv_seq_len)

    return mean_pool_fwd


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    hidden_dim: int,
    block_s: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build the mean-pool backward ``@T.prim_func``.

    Grid: ``(ceildiv(hidden_dim, BLOCK_D), ceildiv(seq_len, BLOCK_S), batch)``.

    Each CTA writes a ``(BLOCK_S, BLOCK_D)`` slab of ``dX[b]`` by broadcasting
    ``dY[b, d] * (1/seq_len)`` to every valid token position
    ``s in [sx*BLOCK_S, (sx+1)*BLOCK_S)``.  Out-of-bounds positions
    (``s >= S`` or ``d >= D``) are skipped via a compile-time if-guard.

    Args:
        batch: Batch dimension ``B``.
        seq_len: Sequence length ``S``.
        hidden_dim: Feature dimension ``D``.
        block_s: Tile size along ``S``.
        block_d: Tile size along ``D``.
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: Number of CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(dY: [B, D] fp32, dX: [B, S, D])``.
        ``dX`` is written in *dtype*.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, D = batch, seq_len, hidden_dim
    BS, BD = block_s, block_d
    inv_seq_len = 1.0 / float(seq_len)

    @T.prim_func
    def mean_pool_bwd(
        dY: T.Tensor((B, D), accum),
        dX: T.Tensor((B, S, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), T.ceildiv(S, BS), B, threads=threads) as (dx, sx, bx):
            dy_row = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for j in T.Parallel(BD):
                d_idx = dx * BD + j
                dy_row[j] = T.if_then_else(d_idx < D, dY[bx, d_idx] * inv_seq_len, 0.0)

            for i, j in T.Parallel(BS, BD):
                s_idx = sx * BS + i
                d_idx = dx * BD + j
                if (s_idx < S) & (d_idx < D):
                    dX[bx, s_idx, d_idx] = T.Cast(ts, dy_row[j])

    return mean_pool_bwd


def make_varlen_fwd_prim_func(
    *,
    total_tokens: int,
    num_seqs: int,
    hidden_dim: int,
    block_s: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build packed mean-pooling forward for variable-length (packed) sequences.

    Grid: ``(ceildiv(hidden_dim, BLOCK_D), num_seqs)``.

    Each CTA handles one sequence identified by ``bx`` (the batch/sequence
    index into ``CuSeqLens``).  It computes the maximum per-batch sequence
    length once (needed to safely clamp the slice start), then streams
    ``seq_len`` tokens in chunks of ``BLOCK_S``, accumulating a running
    ``sum_local`` fragment.  The final result is normalised by ``1/seq_len``
    and cast back to the input dtype.

    Unlike the padded kernel, the sequence loop uses ``T.serial`` instead of
    ``T.Pipelined`` because TileLang's pipeline requires a statically
    known tile count, which is not possible when each sequence has an
    independent length.

    Args:
        total_tokens: Total token count across all packed sequences (``TQ``).
        num_seqs: Number of sequences in the batch (``B``).
        hidden_dim: Feature dimension (``D``).
        block_s: Tile size along the sequence axis (``BLOCK_S``).
        block_d: Tile size along the hidden dimension (``BLOCK_D``).
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: Number of CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(X: [TQ, D], CuSeqLens: [B+1], Y: [B, D])``.
        ``Y`` is written in the input dtype.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    TQ, B, D = total_tokens, num_seqs, hidden_dim
    BS, BD = block_s, block_d

    @T.prim_func
    def mean_pool_varlen_fwd(
        X: T.Tensor((TQ, D), ts),
        CuSeqLens: T.Tensor((B + 1,), "int32"),
        Y: T.Tensor((B, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), B, threads=threads) as (dx, bx):
            x_chunk = T.alloc_fragment((BS, BD), accum)
            chunk_sum = T.alloc_fragment((BD,), accum)
            sum_local = T.alloc_fragment((BD,), accum)
            max_len_buf = T.alloc_fragment((1,), "int32")
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[bx])
            end = T.Cast("int32", CuSeqLens[bx + 1])
            seq_len = end - start
            max_len_buf[0] = 0
            for b in T.serial(B):
                b_start = T.Cast("int32", CuSeqLens[b])
                b_end = T.Cast("int32", CuSeqLens[b + 1])
                max_len_buf[0] = T.max(max_len_buf[0], b_end - b_start)
            slice_start = T.min(start, TQ - max_len_buf[0])
            inv_seq_len = T.if_then_else(seq_len > 0, 1.0 / T.Cast(accum, seq_len), 0.0)
            T.fill(sum_local, 0)

            for s_iter in T.serial(T.ceildiv(TQ, BS)):
                for i, j in T.Parallel(BS, BD):
                    local_idx = s_iter * BS + i
                    t_idx = slice_start + local_idx
                    d_idx = dx * BD + j
                    safe_t = T.min(t_idx, TQ - 1)
                    safe_d = T.min(d_idx, D - 1)
                    x_chunk[i, j] = T.if_then_else(
                        (local_idx < seq_len) & (d_idx < D),
                        T.Cast(accum, X[safe_t, safe_d]),
                        0.0,
                    )
                T.reduce_sum(x_chunk, chunk_sum, dim=0, clear=True)
                for j in T.Parallel(BD):
                    sum_local[j] = sum_local[j] + chunk_sum[j]

            for j in T.Parallel(BD):
                d_idx = dx * BD + j
                if d_idx < D:
                    Y[bx, d_idx] = T.Cast(ts, sum_local[j] * inv_seq_len)

    return mean_pool_varlen_fwd


def make_varlen_bwd_prim_func(
    *,
    total_tokens: int,
    num_seqs: int,
    hidden_dim: int,
    block_s: int,
    block_d: int,
    dtype,
    threads: int = 128,
):
    """Build packed mean-pooling backward for variable-length (packed) sequences.

    Grid: ``(ceildiv(hidden_dim, BLOCK_D), num_seqs)``.

    Each CTA broadcasts ``dY[bx, :] / seq_len`` back to all token positions
    belonging to sequence ``bx`` in the packed token buffer.  ``seq_len`` is
    derived from ``CuSeqLens[bx+1] - CuSeqLens[bx]`` and the start offset
    of the sequence in the packed buffer is ``CuSeqLens[bx]``.

    Args:
        total_tokens: Total token count across all packed sequences (``TQ``).
        num_seqs: Number of sequences in the batch (``B``).
        hidden_dim: Feature dimension (``D``).
        block_s: Tile size along the sequence axis (``BLOCK_S``).
        block_d: Tile size along the hidden dimension (``BLOCK_D``).
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: Number of CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(dY: [B, D] fp32, CuSeqLens: [B+1], dX: [TQ, D])``.
        ``dX`` is written in *dtype*.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    TQ, B, D = total_tokens, num_seqs, hidden_dim
    BS, BD = block_s, block_d

    @T.prim_func
    def mean_pool_varlen_bwd(
        dY: T.Tensor((B, D), accum),
        CuSeqLens: T.Tensor((B + 1,), "int32"),
        dX: T.Tensor((TQ, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), B, threads=threads) as (dx, bx):
            dy_row = T.alloc_fragment((BD,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            start = T.Cast("int32", CuSeqLens[bx])
            end = T.Cast("int32", CuSeqLens[bx + 1])
            seq_len = end - start
            inv_len = T.if_then_else(seq_len > 0, 1.0 / T.Cast(accum, seq_len), 0.0)

            for j in T.Parallel(BD):
                d_idx = dx * BD + j
                dy_row[j] = T.if_then_else(d_idx < D, dY[bx, d_idx] * inv_len, 0.0)

            for s_iter in T.serial(T.ceildiv(TQ, BS)):
                for i, j in T.Parallel(BS, BD):
                    local_idx = s_iter * BS + i
                    t_idx = start + local_idx
                    d_idx = dx * BD + j
                    if (local_idx < seq_len) & (t_idx < TQ) & (d_idx < D):
                        dX[t_idx, d_idx] = T.Cast(ts, dy_row[j])

    return mean_pool_varlen_bwd
