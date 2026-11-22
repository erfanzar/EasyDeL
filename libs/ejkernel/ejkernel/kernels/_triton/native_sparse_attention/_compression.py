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


"""Compressed attention computation for Native Sparse Attention (NSA).

This module implements the compressed attention component of Native Sparse
Attention, where queries attend to mean-pooled (compressed) key-value blocks
rather than individual tokens. This provides a coarse-grained global context
that guides the selection of important blocks for fine-grained attention.

Compressed Attention Process:
-----------------------------
1. Keys and values are mean-pooled into blocks (e.g., 64 tokens -> 1 block)
2. Each query computes attention over these compressed representations
3. The resulting attention scores indicate block importance
4. Scores are used to select top-K blocks for detailed attention
5. The compressed attention output can also be used directly (gated)

The implementation uses custom Triton kernels for efficient GPU computation
and supports both forward and backward passes with full autodifferentiation.

Key benefits:
- O(N²/B) complexity for block size B (much faster than O(N²))
- Provides global context across entire sequence
- Enables learned sparse attention patterns
- Integrates with variable-length sequence processing

The compressed attention output has two uses in NSA:
1. As attention scores for block selection (via top-K)
2. As a direct output pathway (gated with g_cmp)

Functions:
- nsa_compression: Main user-facing function with custom VJP
- nsa_compression_fwd: Forward pass kernel wrapper
- nsa_compression_bwd: Backward pass kernel wrapper
- nsa_compression_fwd_kernel: Triton kernel for forward pass
- nsa_compression_bwd_kernel_dq: Triton kernel for query gradients
- nsa_compression_bwd_kernel_dkv: Triton kernel for key/value gradients

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.native_sparse_attention._compression import nsa_compression
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> block_size = 64
    >>>
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> k_compressed = jnp.ones((batch, 16, num_heads, head_dim))
    >>> v_compressed = jnp.ones((batch, 16, num_heads, head_dim))
    >>>
    >>> output, lse = nsa_compression(
    ...     q, k_compressed, v_compressed,
    ...     block_size=block_size,
    ...     softmax_scale=head_dim ** -0.5
    ... )
"""

from functools import partial

import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import cdiv, next_power_of_2, triton_call
from ejkernel.xla_utils.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_compression_fwd_kernel(
    q,
    k,
    v,
    softmax_scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    o,
    lse,
    SEQUENCE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Forward attention kernel over compressed (mean-pooled) KV blocks.

    Grid: ``(SEQUENCE, cdiv(BLOCK_DIMV, BLOCKSIZE_V), batch * KV_HEADS)``

    Each program instance computes the compressed attention output for a
    single query token (program axis 0), a value-head tile (axis 1), and a
    (batch, KV-head) pair (axis 2).  The query attends to all compressed
    blocks that precede it causally (``NC = (i_t + 1) // BLOCKSIZE`` blocks).

    The compressed keys/values are stored at a base offset ``boc``
    (chunk offset for variable-length mode, or ``batch * TC`` otherwise) with
    the KV-head interleaved layout ``(TC, KV_HEADS, BLOCK_DIM)``.

    For each query token, the kernel:

    1. Loads the query vector and scales by ``softmax_scale``.
    2. Iterates over compressed KV blocks in tiles of ``BLOCK_SEQ``.
    3. Accumulates attention output using online softmax (running max ``b_m``
       and sum ``b_acc``).
    4. Writes the normalised output to ``o`` and the LSE to ``lse`` (only from
       the first value tile ``i_v == 0`` to avoid double-writes).

    Args:
        q: Query pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMK).
        k: Compressed key pointer, shape (TC_total, KV_HEADS, BLOCK_DIMK),
            where ``TC_total`` is the total number of compressed blocks.
        v: Compressed value pointer, shape (TC_total, KV_HEADS, BLOCK_DIMV).
        softmax_scale: Scalar attention scale (passed at launch time).
        cu_seqlens: Cumulative sequence lengths pointer (variable-length mode).
        token_indices: Per-token (sequence, offset) index pairs (varlen mode).
        chunk_offsets: Per-sequence compressed-block start offsets (varlen).
        o: Output pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp output pointer, shape (batch * SEQUENCE, Q_HEADS).
        SEQUENCE: Sequence length (compile-time constant; overridden per
            sequence in variable-length mode).
        KV_HEADS: Number of KV attention heads.
        Q_HEADS: Total number of query heads (= KV_HEADS * QK_GROUPS).
        QK_GROUPS: GQA group size (Q_HEADS // KV_HEADS).
        BLOCK_DIMK: Key head dimension.
        BLOCK_DIMV: Value head dimension.
        BLOCK_SEQ: Number of compressed KV blocks processed per inner
            iteration.
        BLOCKSIZE: Original block size used for mean-pooling (tokens per
            compressed block).
        BLOCKSIZE_K: Tile width for the key dimension (<= 128).
        BLOCKSIZE_V: Tile width for the value dimension (<= 128).
        IS_VARLEN: Heuristic flag enabling variable-length mode.
    """
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE
        boc = i_b * tl.cdiv(SEQUENCE, BLOCKSIZE)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * Q_HEADS * BLOCK_DIMK,
        (Q_HEADS, BLOCK_DIMK),
        (BLOCK_DIMK, 1),
        (i_h * QK_GROUPS, 0),
        (QK_GROUPS, BLOCKSIZE_K),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * softmax_scale).to(b_q.dtype)

    TC = tl.cdiv(SEQUENCE, BLOCKSIZE)
    NC = (i_t + 1) // BLOCKSIZE

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * Q_HEADS * BLOCK_DIMV,
        (Q_HEADS, BLOCK_DIMV),
        (BLOCK_DIMV, 1),
        (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
        (QK_GROUPS, BLOCKSIZE_V),
        (1, 0),
    )
    b_o = tl.zeros([QK_GROUPS, BLOCKSIZE_V], dtype=tl.float32)
    b_m = tl.full([QK_GROUPS], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([QK_GROUPS], dtype=tl.float32)

    for i_c in range(0, NC, BLOCK_SEQ):
        o_c = i_c + tl.arange(0, BLOCK_SEQ)

        p_k = tl.make_block_ptr(
            k + (boc * KV_HEADS + i_h) * BLOCK_DIMK,
            (BLOCK_DIMK, TC),
            (1, KV_HEADS * BLOCK_DIMK),
            (0, i_c),
            (BLOCKSIZE_K, BLOCK_SEQ),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (boc * KV_HEADS + i_h) * BLOCK_DIMV,
            (TC, BLOCK_DIMV),
            (KV_HEADS * BLOCK_DIMV, 1),
            (i_c, i_v * BLOCKSIZE_V),
            (BLOCK_SEQ, BLOCKSIZE_V),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        b_p = tl.exp(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m
    if NC == 0:
        b_lse = tl.zeros([QK_GROUPS], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + tl.log(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(
            lse + (bos + i_t) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS),
            b_lse.to(lse.dtype.element_ty),
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    softmax_scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    dq,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute dQ for one query token over its causally valid compressed KV blocks.

    Grid: ``(SEQUENCE, NumVBlocks, Z * KV_HEADS)``

    - Axis 0 (``i_t``): query token position within the sequence.
    - Axis 1 (``i_v``): V-dimension tile index (0 .. NumVBlocks-1).
    - Axis 2 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    Each CTA iterates over all compressed KV blocks in tiles of ``BLOCK_SEQ``
    up to ``NC = (i_t + 1) // BLOCKSIZE``, accumulating the dQ gradient:
    ``dQ = scale * sum_c( dS_c @ K_c^T )``, where ``dS_c = P_c * (dP_c - delta)``.

    When ``NumVBlocks > 1`` (value dimension > 128), partial dQ tiles are
    written into a leading NumVBlocks accumulation dimension and reduced by
    the caller.

    Args:
        q: Query pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMK).
        k: Compressed key pointer, shape (TC_total, KV_HEADS, BLOCK_DIMK).
        v: Compressed value pointer, shape (TC_total, KV_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp from the forward pass, shape (batch * SEQUENCE, Q_HEADS).
        delta: Precomputed dot(o, do), shape (batch * SEQUENCE, Q_HEADS).
        do: Upstream gradient, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMV).
        softmax_scale: Scalar attention scale factor.
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` when ``IS_VARLEN=False``.
        token_indices: Per-token (seq, offset) lookup table (varlen mode);
            pass ``1`` when ``IS_VARLEN=False``.
        chunk_offsets: Per-sequence compressed-block start offsets (varlen);
            pass ``1`` when ``IS_VARLEN=False``.
        dq: Output gradient buffer, shape
            (NumVBlocks, batch * SEQUENCE, Q_HEADS, BLOCK_DIMK).
        SEQUENCE: Uniform sequence length (constexpr).
        Z: Batch size (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        Q_HEADS: Total number of query heads (constexpr).
        QK_GROUPS: GQA group size ``Q_HEADS // KV_HEADS`` (constexpr).
        BLOCK_DIMK: Key head dimension (constexpr).
        BLOCK_DIMV: Value head dimension (constexpr).
        BLOCK_SEQ: Number of compressed KV blocks per inner iteration (constexpr).
        BLOCKSIZE: Tokens per compressed block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        BLOCKSIZE_V: Tile size along the value dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
    """
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE
        boc = i_b * tl.cdiv(SEQUENCE, BLOCKSIZE)

    q += (bos + i_t) * Q_HEADS * BLOCK_DIMK
    do += (bos + i_t) * Q_HEADS * BLOCK_DIMV
    lse += (bos + i_t) * Q_HEADS
    delta += (bos + i_t) * Q_HEADS
    dq += (i_v * scope + bos + i_t) * Q_HEADS * BLOCK_DIMK

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

    TC = tl.cdiv(SEQUENCE, BLOCKSIZE)
    NC = (i_t + 1) // BLOCKSIZE

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    b_dq = tl.zeros([QK_GROUPS, BLOCKSIZE_K], dtype=tl.float32)
    for i_c in range(0, NC, BLOCK_SEQ):
        o_c = i_c + tl.arange(0, BLOCK_SEQ)
        p_k = tl.make_block_ptr(
            k + (boc * KV_HEADS + i_h) * BLOCK_DIMK,
            (BLOCK_DIMK, TC),
            (1, KV_HEADS * BLOCK_DIMK),
            (0, i_c),
            (BLOCKSIZE_K, BLOCK_SEQ),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (boc * KV_HEADS + i_h) * BLOCK_DIMV,
            (BLOCK_DIMV, TC),
            (1, KV_HEADS * BLOCK_DIMV),
            (i_v * BLOCKSIZE_V, i_c),
            (BLOCKSIZE_V, BLOCK_SEQ),
            (0, 1),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k)
        b_p = tl.exp(b_s - b_lse[:, None])
        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= softmax_scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bwd_kernel_preprocess(
    o,
    do,
    delta,
    Z: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
):
    """Compute per-token delta = sum(o * do) needed for stable softmax gradients.

    Grid: ``(numel(o[..., 0]),)`` - one program per (batch * seq * head) row.

    ``delta[i] = sum(o[i] * do[i])`` is the dot-product of the forward output
    with the upstream gradient, used to stabilise the softmax backward formula:
    ``dS = P * (dP - delta)``.

    Args:
        o: Forward attention output pointer, last dimension = ``BLOCK_DIMV``.
        do: Upstream gradient pointer, same shape as ``o``.
        delta: Output pointer for the per-row scalar delta values.
        Z: Power-of-2 padded value dimension used to load aligned tiles.
        BLOCK_DIMV: Actual (unpadded) value dimension.
    """
    i_n = tl.program_id(0)
    o_d = tl.arange(0, Z)
    m_d = o_d < BLOCK_DIMV

    b_o = tl.load(o + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
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
    BLOCK_SEQ: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute dK and dV for one compressed KV tile over all attending queries.

    Grid: ``(NumVBlocks, NC_total, Z * KV_HEADS)``

    - Axis 0 (``i_v``): V-dimension tile index (0 .. NumVBlocks-1).
    - Axis 1 (``i_c``): compressed KV block tile index; one CTA handles a
      slice of ``BLOCK_SEQ`` consecutive compressed blocks.
    - Axis 2 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    Each CTA iterates over all query tokens ``i`` that could attend to the
    compressed blocks ``[i_c * BLOCK_SEQ, (i_c+1) * BLOCK_SEQ)``, i.e. tokens
    with ``i >= i_c * BLOCK_SEQ * BLOCKSIZE``.  The causal constraint is
    enforced inside the loop via the masking of ``b_p``.

    Accumulates:
    - ``dV += P^T @ dO``
    - ``dK += dS @ Q`` where ``dS = P * (dP - delta)``

    Both partial dK and dV tiles are written into per-tile accumulation buffers
    with a leading NumVBlocks dimension; the caller sums along axis 0.

    Args:
        q: Query pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMK).
        k: Compressed key pointer, shape (TC_total, KV_HEADS, BLOCK_DIMK).
        v: Compressed value pointer, shape (TC_total, KV_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp from the forward pass, shape (batch * SEQUENCE, Q_HEADS).
        delta: Precomputed dot(o, do), shape (batch * SEQUENCE, Q_HEADS).
        do: Upstream gradient, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMV).
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` when ``IS_VARLEN=False``.
        chunk_indices: KV-block-to-(seq, tile) lookup table for varlen mode;
            pass ``1`` when ``IS_VARLEN=False``.
        chunk_offsets: Per-sequence compressed-block start offsets (varlen);
            pass ``1`` when ``IS_VARLEN=False``.
        softmax_scale: Scalar attention scale factor.
        dk: Output gradient buffer for keys, shape
            (NumVBlocks, TC_total, KV_HEADS, BLOCK_DIMK).
        dv: Output gradient buffer for values, shape
            (NumVBlocks, TC_total, KV_HEADS, BLOCK_DIMV).
        SEQUENCE: Uniform sequence length (constexpr).
        Z: Batch size (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        Q_HEADS: Total number of query heads (constexpr).
        QK_GROUPS: GQA group size ``Q_HEADS // KV_HEADS`` (constexpr).
        BLOCK_DIMK: Key head dimension (constexpr).
        BLOCK_DIMV: Value head dimension (constexpr).
        BLOCK_SEQ: Number of compressed KV blocks per CTA tile (constexpr).
        BLOCKSIZE: Tokens per compressed block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        BLOCKSIZE_V: Tile size along the value dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
    """
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE
        boc = i_b * tl.cdiv(SEQUENCE, BLOCKSIZE)

    TC = tl.cdiv(SEQUENCE, BLOCKSIZE)

    p_k = tl.make_block_ptr(
        k + (boc * KV_HEADS + i_h) * BLOCK_DIMK,
        (TC, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_c * BLOCK_SEQ, 0),
        (BLOCK_SEQ, BLOCKSIZE_K),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (boc * KV_HEADS + i_h) * BLOCK_DIMV,
        (TC, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_c * BLOCK_SEQ, i_v * BLOCKSIZE_V),
        (BLOCK_SEQ, BLOCKSIZE_V),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * scope * KV_HEADS + boc * KV_HEADS + i_h) * BLOCK_DIMK,
        (TC, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_c * BLOCK_SEQ, 0),
        (BLOCK_SEQ, BLOCKSIZE_K),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (i_v * scope * KV_HEADS + boc * KV_HEADS + i_h) * BLOCK_DIMV,
        (TC, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_c * BLOCK_SEQ, i_v * BLOCKSIZE_V),
        (BLOCK_SEQ, BLOCKSIZE_V),
        (1, 0),
    )

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BLOCK_SEQ, BLOCKSIZE_K], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BLOCK_SEQ, BLOCKSIZE_V], dtype=tl.float32)

    for i in range(i_c * BLOCK_SEQ * BLOCKSIZE, SEQUENCE):
        o_c = i_c * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

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
        b_p = tl.where((i >= max(0, (o_c + 1) * BLOCKSIZE - 1))[:, None], b_p, 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def nsa_compression_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Launch the NSA compressed-attention forward kernel.

    Args:
        q: Query tensor, shape (batch, seq_len, Q_HEADS, BLOCK_DIMK).
        k: Compressed key tensor, shape (batch, TC, KV_HEADS, BLOCK_DIMK),
            where ``TC = seq_len // block_size``.
        v: Compressed value tensor, same shape as ``k`` except last dim may
            differ (BLOCK_DIMV).
        block_size: Compression block size (tokens per compressed KV entry).
        softmax_scale: Attention score scaling factor.
        cu_seqlens: Optional cumulative sequence lengths for variable-length
            mode, shape (num_seqs + 1,).
        token_indices: Optional per-token (seq, offset) index pairs for
            variable-length mode.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple of (o, lse):
            - o: Compressed attention output, shape
              (batch, seq_len, Q_HEADS, BLOCK_DIMV).
            - lse: Log-sum-exp values, shape (batch, seq_len, Q_HEADS).
    """
    Z, SEQUENCE, Q_HEADS, BLOCK_DIMK, BLOCK_DIMV = *q.shape, v.shape[-1]
    KV_HEADS = k.shape[2]
    QK_GROUPS = Q_HEADS // KV_HEADS
    BLOCK_SEQ = BLOCKSIZE = block_size
    BLOCKSIZE_K = min(128, max(int(block_k), next_power_of_2(BLOCK_DIMK)))
    BLOCKSIZE_V = min(128, int(block_v), next_power_of_2(BLOCK_DIMV))
    NumKBlocks = cdiv(BLOCK_DIMK, BLOCKSIZE_K)
    NumVBlocks = cdiv(BLOCK_DIMV, BLOCKSIZE_V)
    assert NumKBlocks == 1, "The key dimension can not be larger than 256"
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BLOCKSIZE) if cu_seqlens is not None else None
    outputs = [
        jax.ShapeDtypeStruct((Z, SEQUENCE, Q_HEADS, BLOCK_DIMV), dtype=v.dtype),
        jax.ShapeDtypeStruct((Z, SEQUENCE, Q_HEADS), dtype="f4"),
    ]
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        KV_HEADS=KV_HEADS,
        Q_HEADS=Q_HEADS,
        QK_GROUPS=QK_GROUPS,
        BLOCK_DIMK=BLOCK_DIMK,
        BLOCK_DIMV=BLOCK_DIMV,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
    )
    o, lse = triton_call(
        q,
        k,
        v,
        softmax_scale,
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        chunk_offsets if chunk_offsets is not None else 1,
        kernel=nsa_compression_fwd_kernel,
        grid=lambda META: (SEQUENCE, NumVBlocks, Z * KV_HEADS),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_compression_fwd",
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    return o, lse


def nsa_compression_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """Compute gradients (dq, dk, dv) for the compressed-attention forward pass.

    The backward pass consists of three Triton kernel launches:

    1. ``bwd_kernel_preprocess``: compute per-row delta = dot(o, do).
    2. ``nsa_compression_bwd_kernel_dq``: accumulate dQ for each query token.
    3. ``nsa_compression_bwd_kernel_dkv``: accumulate dK and dV for each
       compressed KV block.

    When the value dimension requires multiple tiles (``NumVBlocks > 1``),
    per-tile partial gradients are summed along axis 0 after the kernel.

    Args:
        q: Forward query tensor, shape (batch, seq_len, Q_HEADS, BLOCK_DIMK).
        k: Compressed key tensor, shape (batch, TC, KV_HEADS, BLOCK_DIMK).
        v: Compressed value tensor, shape (batch, TC, KV_HEADS, BLOCK_DIMV).
        o: Forward attention output, shape (batch, seq_len, Q_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp from forward pass, shape (batch, seq_len, Q_HEADS).
        do: Upstream gradient for the output, same shape as ``o``.
        block_size: Compression block size used in the forward pass.
        softmax_scale: Attention scale used in the forward pass.
        cu_seqlens: Optional cumulative sequence lengths for variable-length mode.
        token_indices: Optional per-token index pairs for variable-length mode.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple (dq, dk, dv) with the same shapes as the corresponding inputs.
    """
    Z, SEQUENCE, Q_HEADS, BLOCK_DIMK, BLOCK_DIMV = *q.shape, v.shape[-1]
    KV_HEADS = k.shape[2]
    QK_GROUPS = Q_HEADS // KV_HEADS
    BLOCK_SEQ = BLOCKSIZE = block_size
    BLOCKSIZE_K = min(128, max(int(block_k), next_power_of_2(BLOCK_DIMK)))
    BLOCKSIZE_V = min(128, int(block_v), next_power_of_2(v.shape[-1]))
    NumVBlocks = cdiv(BLOCK_DIMV, BLOCKSIZE_V)
    if cu_seqlens is not None:
        chunk_indices, chunk_offsets = (
            prepare_chunk_indices(cu_seqlens, BLOCKSIZE),
            prepare_chunk_offsets(cu_seqlens, BLOCKSIZE),
        )
        NC = len(chunk_indices)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = cdiv(cdiv(SEQUENCE, BLOCKSIZE), BLOCK_SEQ)

    (delta,) = triton_call(
        o,
        do,
        kernel=bwd_kernel_preprocess,
        grid=lambda META: (delta.numel(),),
        out_shape=[jax.ShapeDtypeStruct(o.shape[:-1], dtype="f4")],
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
        BLOCK_SEQ=BLOCK_SEQ,
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
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        chunk_offsets if chunk_offsets is not None else 1,
        kernel=nsa_compression_bwd_kernel_dq,
        grid=lambda META: (SEQUENCE, NumVBlocks, Z * KV_HEADS),
        name="ejkernel::triton::sparse_attn_compression_bwd_dq",
        out_shape=outputs,
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    dq = jnp.sum(dq, 0)

    outputs = [
        jax.ShapeDtypeStruct((NumVBlocks, *k.shape), dtype=k.dtype if NumVBlocks == 1 else "f4"),
        jax.ShapeDtypeStruct(v.shape, dtype=v.dtype),
    ]
    dk, dv = triton_call(
        q,
        k,
        v,
        lse,
        delta,
        do,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        chunk_offsets if chunk_offsets is not None else 1,
        softmax_scale,
        kernel=nsa_compression_bwd_kernel_dkv,
        grid=lambda META: (NumVBlocks, NC, Z * KV_HEADS),
        name="ejkernel::triton::sparse_attn_compression_bwd_dkdv",
        out_shape=outputs,
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    dk = jnp.sum(dk, 0)
    return dq, dk, dv


def _fwd_call(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """VJP forward pass: runs compressed attention and saves residuals."""
    o, lse = nsa_compression_fwd(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    residual = q, k, v, o, lse
    return (o, lse), residual


def _bwd_call(
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None,
    token_indices: jax.Array | None,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    residual: tuple[jax.Array],
    do: jax.Array,
):
    """VJP backward pass: delegates to ``nsa_compression_bwd``."""
    q, k, v, o, lse = residual
    dq, dk, dv = nsa_compression_bwd(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        do=do,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
@partial(jax.jit, static_argnums=(3, 4, 7, 8, 9, 10))
def _nsa_compression(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled inner compressed attention with custom VJP."""
    return _fwd_call(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )[0]


_nsa_compression.defvjp(_fwd_call, _bwd_call)


def nsa_compression(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute compressed attention over mean-pooled key-value blocks.

    This function implements the compressed attention pathway of Native Sparse
    Attention, where each query token attends to compressed (mean-pooled)
    representations of key-value blocks. This provides O(N²/B) complexity while
    maintaining global context across the sequence.

    The compressed attention serves two purposes:
    1. Block selection: Attention scores indicate which blocks are important
    2. Direct output: Can be used as a coarse-grained attention output (gated)

    Args:
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim).
            Each query attends to all compressed KV blocks.
        key: Compressed key tensor of shape (batch, num_blocks, num_heads, head_dim).
            Keys have been mean-pooled from blocks of size `block_size`.
        value: Compressed value tensor of shape (batch, num_blocks, num_heads, head_dim).
            Values have been mean-pooled from blocks of size `block_size`.
        block_size: Size of each block in tokens. Keys/values should already be
            compressed at this granularity.
        softmax_scale: Attention score scaling factor, typically 1/sqrt(head_dim).
        cu_seqlens: Optional cumulative sequence lengths for variable-length
            sequences, shape (num_seqs + 1,). If provided, enables packed
            variable-length processing.
        token_indices: Optional token indices for variable-length sequences,
            shape (total_tokens, 2). Each row contains [sequence_id, token_offset].
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing:
            - output: Compressed attention output, shape (batch, seq_len, num_heads, head_dim).
            - lse: Log-sum-exp of attention scores, shape (batch, seq_len, num_heads).
              Used for numerical stability and block selection.

    Note:
        The key and value tensors should already be mean-pooled to the block level.
        Use mean_pooling(k, block_size) and mean_pooling(v, block_size) to prepare them.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.mean_pooling import mean_pooling
        >>> from ejkernel.kernels._triton.native_sparse_attention._compression import nsa_compression
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> block_size = 64
        >>>
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> k_compressed = mean_pooling(k, block_size)
        >>> v_compressed = mean_pooling(v, block_size)
        >>>
        >>>
        >>> output, lse = nsa_compression(
        ...     q, k_compressed, v_compressed,
        ...     block_size=block_size,
        ...     softmax_scale=head_dim ** -0.5
        ... )
        >>> print(output.shape)
        >>> print(lse.shape)
    """
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    return _nsa_compression(
        q=query,
        k=key,
        v=value,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
