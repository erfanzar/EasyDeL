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

"""Native Sparse Attention (NSA) backward pass Triton kernels.

This module implements gradient computation for Native Sparse Attention,
enabling end-to-end training of models using learned sparse attention
patterns.

Backward Pass Algorithm:
-----------------------
1. Preprocess (bwd_kernel_preprocess):
   - Compute delta = sum(output * d_output) per position
   - Required for stable gradient computation

2. Query Gradients (nsa_bwd_kernel_dq):
   - Iterate over selected KV blocks for each query
   - Accumulate dQ contributions using attention weights
   - Scale by softmax_scale for proper gradient flow

3. Key/Value Gradients (nsa_bwd_kernel_dkv):
   - Use block mask to identify which queries attend to each KV block
   - Compute dK and dV by iterating over attending queries
   - Accumulate gradients across the value dimension splits

Key Kernels:
-----------
nsa_bwd_kernel_dq:
    Computes query gradients by recomputing attention weights and
    propagating gradients through the softmax operation.

nsa_bwd_kernel_dkv:
    Computes key and value gradients using a transposed attention
    pattern defined by the block mask.

bwd_kernel_preprocess:
    Preprocessing step computing element-wise product sum for
    numerically stable softmax gradients.

Memory Layout:
-------------
Same as forward pass, with additional:
- Output: [batch, seq_len, num_q_heads, head_dim_v]
- LSE: [batch, seq_len, num_q_heads] log-sum-exp values
- Delta: [batch, seq_len, num_q_heads] precomputed for gradients
- Block Mask: Boolean mask indicating valid (query, block) pairs

Functions:
- bwd_triton_impl: Main backward pass orchestrator
"""

import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import cdiv, next_power_of_2, triton_call

from ....xla_utils.utils import prepare_chunk_indices
from ._utilities import nsa_block_mask


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] != 1,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], jax.Array),
    }
)
@triton.jit
def nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    softmax_scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    dq,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    IndicesSize: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    """Compute dQ gradients for one query token over its selected KV blocks.

    Grid: ``(SEQUENCE, NumVBlocks, Z * KV_HEADS)``

    - Axis 0 (``i_t``): query token position within the sequence.
    - Axis 1 (``i_v``): V-dimension tile index (0 .. NumVBlocks-1).
    - Axis 2 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    Each CTA accumulates the dQ contribution from all KV blocks listed in
    ``block_indices`` for query ``i_t``.  The causal constraint is enforced
    by skipping blocks whose start position ``i_s * BLOCKSIZE > i_t``.

    When ``NumVBlocks > 1``, partial dQ tiles are written into a leading
    ``NumVBlocks`` accumulation dimension and summed by the caller.

    Args:
        q: Query tensor pointer, layout (batch_or_varlen, Q_HEADS, BLOCK_DIMK).
        k: Key tensor pointer, layout (batch_or_varlen, KV_HEADS, BLOCK_DIMK).
        v: Value tensor pointer, layout (batch_or_varlen, KV_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp from forward pass, shape (batch_or_varlen, Q_HEADS).
        delta: Precomputed dot(o, do) per query row, same shape as ``lse``.
        do: Upstream output gradient, layout matching ``v`` output.
        softmax_scale: Scalar attention scale factor (typically ``1/sqrt(head_dim)``).
        block_indices: Sparse KV block indices, shape
            (batch_or_varlen, KV_HEADS, IndicesSize).
        block_counts: Valid block count per query; ``jax.Array`` enables
            ``USE_BLOCK_COUNTS``, int scalar disables per-position count.
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` (int) when ``IS_VARLEN=False``.
        token_indices: Token-to-(sequence, position) lookup table for
            variable-length mode; pass ``1`` (int) when ``IS_VARLEN=False``.
        dq: Output gradient buffer, shape (NumVBlocks, batch_or_varlen, Q_HEADS, BLOCK_DIMK).
        SEQUENCE: Uniform sequence length (constexpr).
        Z: Batch size (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        Q_HEADS: Number of query attention heads (constexpr).
        QK_GROUPS: ``Q_HEADS // KV_HEADS``, GQA group size (constexpr).
        BLOCK_DIMK: Key/query head dimension (constexpr).
        BLOCK_DIMV: Value head dimension (constexpr).
        IndicesSize: Number of selected KV blocks per query (constexpr).
        BLOCKSIZE: Tokens per KV block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        BLOCKSIZE_V: Tile size along the value dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
        USE_BLOCK_COUNTS: Heuristic flag; True when ``block_counts`` is a
            ``jax.Array`` (per-position valid block count).
    """
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    q += (bos + i_t) * Q_HEADS * BLOCK_DIMK
    do += (bos + i_t) * Q_HEADS * BLOCK_DIMV
    lse += (bos + i_t) * Q_HEADS
    delta += (bos + i_t) * Q_HEADS
    dq += (i_v * scope + bos + i_t) * Q_HEADS * BLOCK_DIMK
    block_indices += (bos + i_t) * KV_HEADS * IndicesSize + i_h * IndicesSize

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * KV_HEADS + i_h)
    else:
        NS = IndicesSize

    k += (bos * KV_HEADS + i_h) * BLOCK_DIMK
    v += (bos * KV_HEADS + i_h) * BLOCK_DIMV

    p_q = tl.make_block_ptr(
        q, (Q_HEADS, BLOCK_DIMK), (BLOCK_DIMK, 1), (i_h * QK_GROUPS, 0), (QK_GROUPS, BLOCKSIZE_K), (1, 0)
    )
    p_dq = tl.make_block_ptr(
        dq, (Q_HEADS, BLOCK_DIMK), (BLOCK_DIMK, 1), (i_h * QK_GROUPS, 0), (QK_GROUPS, BLOCKSIZE_K), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * softmax_scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(
        do,
        (Q_HEADS, BLOCK_DIMV),
        (BLOCK_DIMV, 1),
        (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
        (QK_GROUPS, BLOCKSIZE_V),
        (1, 0),
    )
    p_lse = lse + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
    p_delta = delta + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)
    b_dq = tl.zeros([QK_GROUPS, BLOCKSIZE_K], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BLOCKSIZE
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(
                k, (BLOCK_DIMK, SEQUENCE), (1, KV_HEADS * BLOCK_DIMK), (0, i_s), (BLOCKSIZE_K, BLOCKSIZE), (0, 1)
            )
            p_v = tl.make_block_ptr(
                v,
                (BLOCK_DIMV, SEQUENCE),
                (1, KV_HEADS * BLOCK_DIMV),
                (i_v * BLOCKSIZE_V, i_s),
                (BLOCKSIZE_V, BLOCKSIZE),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))

            b_s = tl.dot(b_q, b_k)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BLOCKSIZE)))[None, :], b_p, 0)

            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= softmax_scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    block_mask,
    cu_seqlens,
    chunk_indices,
    softmax_scale,
    dk,
    dv,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    MaskSize: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute dK and dV gradients for one KV block over all attending queries.

    Grid: ``(NumVBlocks, NS, Z * KV_HEADS)``

    - Axis 0 (``i_v``): V-dimension tile index (0 .. NumVBlocks-1).
    - Axis 1 (``i_s``): KV block index (0 .. NS-1), where ``NS = cdiv(SEQUENCE, BLOCKSIZE)``
      for uniform batches or ``len(chunk_indices)`` for variable-length mode.
    - Axis 2 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    Each CTA iterates from ``i_s * BLOCKSIZE`` to ``SEQUENCE``, checking
    ``block_mask[b, t, h, i_s]`` to skip query positions that do not attend
    to this KV block, then accumulates dK and dV from those that do.

    When ``NumVBlocks > 1``, partial dK tiles are written into a leading
    ``NumVBlocks`` accumulation dimension and summed by the caller.
    dV does not require the extra dimension because each CTA writes a disjoint
    V-slice.

    Args:
        q: Query tensor pointer, layout (batch_or_varlen, Q_HEADS, BLOCK_DIMK).
        k: Key tensor pointer, layout (batch_or_varlen, KV_HEADS, BLOCK_DIMK).
        v: Value tensor pointer, layout (batch_or_varlen, KV_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp from forward pass, shape (batch_or_varlen, Q_HEADS).
        delta: Precomputed dot(o, do) per query row, same shape as ``lse``.
        do: Upstream output gradient, layout matching the forward output.
        block_mask: Dense boolean mask, shape (batch_or_varlen, KV_HEADS, MaskSize);
            entry ``[b, t, h, i_s]`` is True iff query ``t`` attended to block ``i_s``.
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` (int) when ``IS_VARLEN=False``.
        chunk_indices: KV-block-to-(sequence, block) lookup table for
            variable-length mode; pass ``1`` (int) when ``IS_VARLEN=False``.
        softmax_scale: Scalar attention scale factor (typically ``1/sqrt(head_dim)``).
        dk: Output gradient buffer for keys, shape
            (NumVBlocks, batch_or_varlen, KV_HEADS, BLOCK_DIMK).
        dv: Output gradient buffer for values, shape
            (batch_or_varlen, KV_HEADS, BLOCK_DIMV).
        SEQUENCE: Uniform sequence length (constexpr).
        Z: Batch size (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        Q_HEADS: Number of query attention heads (constexpr).
        QK_GROUPS: ``Q_HEADS // KV_HEADS``, GQA group size (constexpr).
        BLOCK_DIMK: Key/query head dimension (constexpr).
        BLOCK_DIMV: Value head dimension (constexpr).
        MaskSize: Width of the block mask's last dimension (constexpr).
        BLOCKSIZE: Tokens per KV block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        BLOCKSIZE_V: Tile size along the value dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
    """
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    p_k = tl.make_block_ptr(
        k + (bos * KV_HEADS + i_h) * BLOCK_DIMK,
        (SEQUENCE, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_s * BLOCKSIZE, 0),
        (BLOCKSIZE, BLOCKSIZE_K),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * KV_HEADS + i_h) * BLOCK_DIMV,
        (SEQUENCE, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_s * BLOCKSIZE, i_v * BLOCKSIZE_V),
        (BLOCKSIZE, BLOCKSIZE_V),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * scope * KV_HEADS + bos * KV_HEADS + i_h) * BLOCK_DIMK,
        (SEQUENCE, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_s * BLOCKSIZE, 0),
        (BLOCKSIZE, BLOCKSIZE_K),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * KV_HEADS + i_h) * BLOCK_DIMV,
        (SEQUENCE, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_s * BLOCKSIZE, i_v * BLOCKSIZE_V),
        (BLOCKSIZE, BLOCKSIZE_V),
        (1, 0),
    )

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BLOCKSIZE, BLOCKSIZE_K], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BLOCKSIZE, BLOCKSIZE_V], dtype=tl.float32)

    for i in range(i_s * BLOCKSIZE, SEQUENCE):
        b_m = tl.load(block_mask + (bos + i) * KV_HEADS * MaskSize + i_h * MaskSize + i_s)
        if b_m:
            p_q = tl.make_block_ptr(
                q + (bos + i) * Q_HEADS * BLOCK_DIMK,
                (Q_HEADS, BLOCK_DIMK),
                (BLOCK_DIMK, 1),
                (i_h * QK_GROUPS, 0),
                (QK_GROUPS, BLOCKSIZE_K),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * softmax_scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(
                do + (bos + i) * Q_HEADS * BLOCK_DIMV,
                (Q_HEADS, BLOCK_DIMV),
                (BLOCK_DIMV, 1),
                (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
                (QK_GROUPS, BLOCKSIZE_V),
                (1, 0),
            )
            p_lse = lse + (bos + i) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
            p_delta = delta + (bos + i) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BLOCKSIZE + tl.arange(0, BLOCKSIZE)))[:, None], b_p, 0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bwd_kernel_preprocess(o, do, delta, Z: tl.constexpr, BLOCK_DIMV: tl.constexpr):
    """Compute per-row delta = dot(o, do) for numerically stable softmax gradients.

    Grid: ``(num_rows,)`` where num_rows = batch * seq_len * num_q_heads.

    Args:
        o: Forward output pointer, flattened with last dim = ``BLOCK_DIMV``.
        do: Upstream gradient pointer, same layout as ``o``.
        delta: Output scalar pointer, shape (num_rows,).
        Z: Power-of-2 padded BLOCK_DIMV for aligned vector loads.
        BLOCK_DIMV: Actual (unpadded) value head dimension.
    """
    i_n = tl.program_id(0)
    o_d = tl.arange(0, Z)
    m_d = o_d < BLOCK_DIMV

    b_o = tl.load(o + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def bwd_triton_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """Compute (dq, dk, dv) for NSA sparse attention backward pass.

    Orchestrates three kernel launches:

    1. ``bwd_kernel_preprocess``: compute per-row delta = dot(o, do).
    2. ``nsa_bwd_kernel_dq``: accumulate dQ for each query position using
       the sparse block pattern.
    3. ``nsa_bwd_kernel_dkv``: accumulate dK and dV for each KV block by
       iterating over all queries that attended to it (determined by
       ``nsa_block_mask``).

    When ``NumVBlocks > 1`` (head_dim_v > 128), per-tile partial gradients
    are reduced by summing along axis 0 after the dQ and dK kernel launches.

    Args:
        q: Query tensor, shape (batch, seq_len, num_q_heads, head_dim).
        k: Key tensor, shape (batch, seq_len, num_kv_heads, head_dim).
        v: Value tensor, shape (batch, seq_len, num_kv_heads, head_dim_v).
        o: Forward attention output, same shape as the expected output.
        lse: Log-sum-exp from the forward pass, shape
            (batch, seq_len, num_q_heads).
        do: Upstream gradient for the output, same shape as ``o``.
        block_indices: Sparse block index tensor used in the forward pass.
        block_counts: Valid block count per query position (int or array).
        block_size: Tokens per KV block.
        softmax_scale: Attention scale used in the forward pass.
        cu_seqlens: Optional cumulative sequence lengths (variable-length).
        token_indices: Optional per-token index pairs (variable-length).
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple (dq, dk, dv) with the same shapes as q, k, and v respectively.
    """
    Z, SEQUENCE, KV_HEADS, BLOCK_DIMK, BLOCK_DIMV, IndicesSize = (
        *k.shape,
        v.shape[-1],
        block_indices.shape[-1],
    )
    Q_HEADS = q.shape[2]
    QK_GROUPS = Q_HEADS // KV_HEADS
    BLOCKSIZE = block_size
    BLOCKSIZE_K = min(128, max(int(block_k), next_power_of_2(BLOCK_DIMK)))
    BLOCKSIZE_V = min(128, int(block_v), next_power_of_2(v.shape[-1]))
    NumVBlocks = cdiv(BLOCK_DIMV, BLOCKSIZE_V)

    delta_shape = o.shape[:-1]
    delta_numel = jnp.prod(jnp.array(delta_shape))
    (delta,) = triton_call(
        o,
        do,
        kernel=bwd_kernel_preprocess,
        grid=lambda META: (delta_numel,),
        out_shape=[jax.ShapeDtypeStruct(delta_shape, dtype="f4")],
        name="ejkernel::triton::sparse_attn_compression_bwd_preprocess",
        Z=next_power_of_2(o.shape[-1]),
        BLOCK_DIMV=o.shape[-1],
    )

    outputs = [jax.ShapeDtypeStruct((NumVBlocks, *q.shape), dtype=q.dtype if NumVBlocks == 1 else "f4")]
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        Z=Z,
        KV_HEADS=KV_HEADS,
        Q_HEADS=Q_HEADS,
        QK_GROUPS=QK_GROUPS,
        BLOCK_DIMK=BLOCK_DIMK,
        BLOCK_DIMV=BLOCK_DIMV,
        IndicesSize=IndicesSize,
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
    )
    (dq,) = triton_call(
        q,
        k,
        v,
        lse,
        delta,
        do,
        softmax_scale,
        block_indices if block_indices is not None else 1,
        block_counts if block_counts is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        kernel=nsa_bwd_kernel_dq,
        grid=lambda META: (SEQUENCE, NumVBlocks, Z * KV_HEADS),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_bwd_dq",
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    dq = jnp.sum(dq, 0)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BLOCKSIZE)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = cdiv(SEQUENCE, BLOCKSIZE)

    block_mask = nsa_block_mask(block_indices, block_counts, cu_seqlens, block_size)
    outputs = [
        jax.ShapeDtypeStruct((NumVBlocks, *k.shape), dtype=k.dtype if NumVBlocks == 1 else "f4"),
        jax.ShapeDtypeStruct(v.shape, dtype=v.dtype),
    ]
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        Z=Z,
        KV_HEADS=KV_HEADS,
        Q_HEADS=Q_HEADS,
        QK_GROUPS=QK_GROUPS,
        BLOCK_DIMK=BLOCK_DIMK,
        BLOCK_DIMV=BLOCK_DIMV,
        MaskSize=block_mask.shape[-1],
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
    )
    dk, dv = triton_call(
        q,
        k,
        v,
        lse,
        delta,
        do,
        block_mask if block_mask is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        softmax_scale,
        kernel=nsa_bwd_kernel_dkv,
        grid=lambda META: (NumVBlocks, NS, Z * KV_HEADS),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_bwd_dkdv",
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    dk = jnp.sum(dk, 0)
    return dq, dk, dv
