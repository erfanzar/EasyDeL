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

"""Native Sparse Attention (NSA) forward pass Triton kernels.

This module implements the forward pass for Native Sparse Attention,
a learned sparse attention mechanism that dynamically selects which
KV blocks to attend to based on compressed attention scores.

Algorithm Overview:
------------------
1. Top-K Selection (nsa_kernel_topk):
   - Compute attention scores against compressed (mean-pooled) keys
   - Use bitonic sort to find top-K highest-scoring blocks
   - Store block indices for selective attention

2. Sparse Attention (nsa_fwd_kernel):
   - For each query, attend only to selected KV blocks
   - Accumulate results using online softmax
   - Produces attention output and LSE for backward pass

Sorting Utilities:
-----------------
- _compare_and_swap: Bitonic sort primitive
- _bitonic_merge: Merge step of bitonic sort
- argsort: GPU-friendly sorting returning indices

Key Features:
- Dynamic sparse attention patterns (not fixed masks)
- Efficient top-K selection using bitonic sort
- Variable-length sequence support (cu_seqlens)
- Optional per-token block counts for adaptive sparsity
- Grouped-query attention (GQA) support

Memory Layout:
-------------
- Queries: [batch, seq_len, num_q_heads, head_dim]
- Keys: [batch, seq_len, num_kv_heads, head_dim]
- Values: [batch, seq_len, num_kv_heads, head_dim_v]
- Block Indices: [batch, seq_len, num_kv_heads, max_blocks]

Functions:
- nsa_topk: Select top-K blocks per query position
- fwd_triton_impl: Compute sparse attention output
"""

import jax
import triton
import triton.language as tl

from ejkernel.callib import cdiv, next_power_of_2, triton_call

from ....xla_utils.utils import prepare_chunk_offsets, prepare_token_indices


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    """Bitonic sort compare-and-swap primitive for GPU-friendly sorting."""
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    """Merge step of bitonic sort, combining sorted subsequences."""
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(
    x,
    ids,
    dim: tl.constexpr = None,
    descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    """GPU-friendly argsort using bitonic sort, returning sorted values and indices."""
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    n_dims: tl.constexpr = tl.log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_kernel_topk(
    q,
    k,
    lse,
    softmax_scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    block_indices,
    SEQUENCE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QHeads: tl.constexpr,
    NumGroups: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    IndSize: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Select the top-``IndSize`` KV blocks for one query token.

    Grid: ``(SEQUENCE, Z * KV_HEADS)``

    - Axis 0 (``i_t``): query token position within the sequence.
    - Axis 1 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    **Algorithm:**

    1. Load the query vector for token ``i_t`` and scale it.
    2. If ``lse`` is provided, load the pre-computed log-sum-exp values to
       convert raw dot-product scores to normalised probabilities.
       Otherwise, compute the LSE inline with an online softmax pass.
    3. Iterate over compressed KV blocks in tiles of ``BLOCKSIZE``, computing
       a per-block importance score (mean of per-group softmax probabilities).
    4. Maintain a running top-``IndSize`` buffer using the ``_bitonic_merge``
       primitive.  The buffer is merged after every tile and sorted at the end.
    5. Write the resulting top-``IndSize`` block indices (1-indexed, then
       offset by -1 on load) to ``block_indices``.

    The causal constraint is applied by masking scores for blocks whose index
    equals the current block (``i_t // BLOCKSIZE``).  Blocks strictly beyond
    ``i_t // BLOCKSIZE`` are also excluded.

    Args:
        q: Query pointer, shape (batch * SEQUENCE, QHeads, BLOCK_DIMK).
        k: Compressed key pointer, shape (TC_total, KV_HEADS, BLOCK_DIMK).
        lse: Optional log-sum-exp pointer, shape (batch * SEQUENCE, QHeads).
            Pass a JAX ``None``-equivalent sentinel to skip (will be recomputed).
        softmax_scale: Scalar attention scale factor.
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` when ``IS_VARLEN=False``.
        token_indices: Per-token (seq, offset) lookup table (varlen mode);
            pass ``1`` when ``IS_VARLEN=False``.
        chunk_offsets: Per-sequence compressed-block start offsets (varlen);
            pass ``1`` when ``IS_VARLEN=False``.
        block_indices: Output tensor for top-K block indices, shape
            (batch * SEQUENCE, KV_HEADS, IndSize), dtype int32.
        SEQUENCE: Uniform sequence length (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        QHeads: Total number of query heads (constexpr).
        NumGroups: GQA group size ``QHeads // KV_HEADS`` (constexpr).
        BLOCK_DIMK: Key head dimension (constexpr).
        IndSize: Power-of-2 number of blocks to select per query (constexpr).
        BLOCKSIZE: Tokens per compressed block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
        q + (bos + i_t) * QHeads * BLOCK_DIMK,
        (QHeads, BLOCK_DIMK),
        (BLOCK_DIMK, 1),
        (i_h * NumGroups, 0),
        (NumGroups, BLOCKSIZE_K),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * softmax_scale).to(b_q.dtype)
    TC = tl.cdiv(SEQUENCE, BLOCKSIZE)
    NC = (i_t + 1) // BLOCKSIZE
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * QHeads + i_h * NumGroups + tl.arange(0, NumGroups))
    else:
        b_m = tl.full([NumGroups], float("-inf"), dtype=tl.float32)
        b_acc = tl.zeros([NumGroups], dtype=tl.float32)
        for i_c in range(0, NC, BLOCKSIZE):
            o_c = i_c + tl.arange(0, BLOCKSIZE)
            p_k = tl.make_block_ptr(
                k + (boc * KV_HEADS + i_h) * BLOCK_DIMK,
                (BLOCK_DIMK, TC),
                (1, KV_HEADS * BLOCK_DIMK),
                (0, i_c),
                (BLOCKSIZE_K, BLOCKSIZE),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            b_p = tl.exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([NumGroups], dtype=tl.float32)
        else:
            b_lse = b_m + tl.log(b_acc)
    b_i = tl.full([BLOCKSIZE], -1, dtype=tl.float32)
    o_i = tl.zeros([BLOCKSIZE], dtype=tl.int32)
    m_i = tl.arange(0, BLOCKSIZE) < BLOCKSIZE // 2
    for i_c in range(0, i_t // BLOCKSIZE + 1, BLOCKSIZE):
        o_c = i_c + tl.arange(0, BLOCKSIZE)
        p_k = tl.make_block_ptr(
            k + (boc * KV_HEADS + i_h) * BLOCK_DIMK,
            (BLOCK_DIMK, TC),
            (1, KV_HEADS * BLOCK_DIMK),
            (0, i_c),
            (BLOCKSIZE_K, BLOCKSIZE),
            (0, 1),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t // BLOCKSIZE > o_c)[None, :], b_s, float("-inf"))
        b_p = tl.where((i_t // BLOCKSIZE == o_c)[None, :], 1.0, tl.exp(b_s - b_lse[:, None]))
        b_i, b_ip = tl.sum(b_p, 0), b_i
        o_i, o_ip = tl.where(o_c <= i_t // BLOCKSIZE, o_c + 1, 0), o_i
        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)
        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
    m_top = tl.arange(0, BLOCKSIZE // IndSize) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i - 1, [BLOCKSIZE // IndSize, IndSize]), 0)
    p_b = tl.make_block_ptr(
        block_indices + (bos + i_t) * KV_HEADS * IndSize, (KV_HEADS * IndSize,), (1,), (i_h * IndSize,), (IndSize,), (0,)
    )
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] != 1,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], jax.Array),
    }
)
@triton.jit
def nsa_fwd_kernel(
    q,
    k,
    v,
    softmax_scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    o,
    lse,
    SEQUENCE: tl.constexpr,
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
    """Compute NSA sparse attention output for one query token.

    Grid: ``(SEQUENCE, NumVBlocks, Z * KV_HEADS)``

    - Axis 0 (``i_t``): query token position within the sequence.
    - Axis 1 (``i_v``): V-dimension tile index (0 .. NumVBlocks-1).
    - Axis 2 (``i_bh``): packed batch-head index; decoded as
      ``i_b = i_bh // KV_HEADS``, ``i_h = i_bh % KV_HEADS``.

    Each CTA iterates over the ``NS`` pre-selected KV blocks listed in
    ``block_indices`` for query ``i_t``, applying the causal constraint
    ``block_start <= i_t``.  Attention is accumulated with an online softmax
    (running max ``b_m``, sum ``b_acc``) for numerical stability.

    Queries with no valid attending blocks produce zero output and zero LSE.

    The KV layout uses a GQA-interleaved format: consecutive tokens share
    KV heads, i.e. stride along the sequence dimension is ``KV_HEADS * dim``.

    Args:
        q: Query pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMK).
        k: Key pointer, layout (batch * SEQUENCE, KV_HEADS, BLOCK_DIMK)
            with row stride ``KV_HEADS * BLOCK_DIMK``.
        v: Value pointer, layout (batch * SEQUENCE, KV_HEADS, BLOCK_DIMV)
            with row stride ``KV_HEADS * BLOCK_DIMV``.
        softmax_scale: Scalar attention scale factor (typically ``1/sqrt(head_dim)``).
        block_indices: Sparse KV block indices, shape
            (batch * SEQUENCE, KV_HEADS, IndicesSize), dtype int32.
        block_counts: Valid block count per query position; a ``jax.Array``
            enables ``USE_BLOCK_COUNTS``, an int disables per-position count.
        cu_seqlens: Cumulative sequence lengths for variable-length mode;
            pass ``1`` when ``IS_VARLEN=False``.
        token_indices: Per-token (seq, offset) lookup table (varlen mode);
            pass ``1`` when ``IS_VARLEN=False``.
        o: Output pointer, shape (batch * SEQUENCE, Q_HEADS, BLOCK_DIMV).
        lse: Log-sum-exp output pointer, shape (batch * SEQUENCE, Q_HEADS);
            written only from the first V-tile (``i_v == 0`` is implicit via
            a shared pointer offset, so all tiles write to the same location).
        SEQUENCE: Uniform sequence length (constexpr).
        KV_HEADS: Number of KV attention heads (constexpr).
        Q_HEADS: Total number of query heads (constexpr).
        QK_GROUPS: GQA group size ``Q_HEADS // KV_HEADS`` (constexpr).
        BLOCK_DIMK: Key head dimension (constexpr).
        BLOCK_DIMV: Value head dimension (constexpr).
        IndicesSize: Maximum number of selected KV blocks per query (constexpr).
        BLOCKSIZE: Tokens per KV block (constexpr).
        BLOCKSIZE_K: Tile size along the key dimension (constexpr).
        BLOCKSIZE_V: Tile size along the value dimension (constexpr).
        IS_VARLEN: Heuristic flag; True when ``cu_seqlens`` is a real array.
        USE_BLOCK_COUNTS: Heuristic flag; True when ``block_counts`` is a
            ``jax.Array`` (per-position valid block count).
    """
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    k += (bos * KV_HEADS + i_h) * BLOCK_DIMK
    v += (bos * KV_HEADS + i_h) * BLOCK_DIMV
    block_indices += (bos + i_t) * KV_HEADS * IndicesSize + i_h * IndicesSize

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * KV_HEADS + i_h)
    else:
        NS = IndicesSize

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

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * Q_HEADS * BLOCK_DIMV,
        (Q_HEADS, BLOCK_DIMV),
        (BLOCK_DIMV, 1),
        (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
        (QK_GROUPS, BLOCKSIZE_V),
        (1, 0),
    )
    p_lse = lse + (bos + i_t) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
    b_o = tl.zeros([QK_GROUPS, BLOCKSIZE_V], dtype=tl.float32)

    b_m = tl.full([QK_GROUPS], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([QK_GROUPS], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BLOCKSIZE
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(
                k, (BLOCK_DIMK, SEQUENCE), (1, KV_HEADS * BLOCK_DIMK), (0, i_s), (BLOCKSIZE_K, BLOCKSIZE), (0, 1)
            )
            p_v = tl.make_block_ptr(
                v,
                (SEQUENCE, BLOCK_DIMV),
                (KV_HEADS * BLOCK_DIMV, 1),
                (i_s, i_v * BLOCKSIZE_V),
                (BLOCKSIZE, BLOCKSIZE_V),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((i_t >= (i_s + tl.arange(0, BLOCKSIZE)))[None, :], b_s, float("-inf"))

            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            b_p = tl.exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m

    b_acc_safe = tl.where(b_acc == 0, 1.0, b_acc)
    b_o = b_o / b_acc_safe[:, None]

    b_o = tl.where(b_acc[:, None] == 0, 0.0, b_o)

    b_m_update = tl.log(b_acc)
    b_m = tl.where(b_acc == 0, 0.0, b_m + b_m_update)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


def nsa_topk(
    q: jax.Array,
    k: jax.Array,
    lse: jax.Array,
    block_counts: jax.Array | int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
) -> jax.Array:
    """Select the top-K KV blocks for each query using compressed attention scores.

    Launches ``nsa_kernel_topk`` to compute per-query block importance scores
    and return the indices of the ``block_counts`` most-attended blocks.

    Args:
        q: Query tensor, shape (batch, seq_len, Q_HEADS, BLOCK_DIMK).
        k: Compressed key tensor, shape (batch, TC, KV_HEADS, BLOCK_DIMK),
           where ``TC = seq_len // block_size``.
        lse: Log-sum-exp from the compressed attention forward pass, shape
             (batch, seq_len, Q_HEADS).  Used to convert raw scores to
             normalised probabilities for ranking.  If ``lse`` were recomputed
             inside this kernel the results would differ from the forward pass
             LSE used in gradient computation.
        block_counts: Number of blocks to select per query.  Can be:
            - An integer (uniform for all positions).
            - A ``jax.Array`` of shape (batch, seq_len, KV_HEADS): per-token
              counts; the maximum is used to size the output tensor.
        block_size: Compression block size (tokens per compressed KV entry).
        softmax_scale: Attention score scaling factor.  Defaults to
            ``1 / sqrt(BLOCK_DIMK)``.
        cu_seqlens: Optional cumulative sequence lengths for variable-length
            mode.

    Returns:
        Block index tensor of shape (batch, seq_len, KV_HEADS, IndSize) in
        int32, where ``IndSize = next_power_of_2(max(block_counts))``.
    """
    Z, SEQUENCE, QHeads, BLOCK_DIMK = q.shape
    KV_HEADS = k.shape[2]
    NumGroups = QHeads // KV_HEADS
    IndSize = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    IndSize = next_power_of_2(IndSize)
    BLOCKSIZE = block_size
    BLOCKSIZE_K = min(128, next_power_of_2(BLOCK_DIMK))
    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, block_size) if cu_seqlens is not None else None
    output_shape_and_dtype = [jax.ShapeDtypeStruct((Z, SEQUENCE, KV_HEADS, IndSize), dtype="i4")]
    kernel_metaparams = dict(
        SEQUENCE=SEQUENCE,
        KV_HEADS=KV_HEADS,
        QHeads=QHeads,
        NumGroups=NumGroups,
        BLOCK_DIMK=BLOCK_DIMK,
        IndSize=IndSize,
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
    )
    (block_indices_output,) = triton_call(
        q,
        k,
        lse,
        float(softmax_scale),
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        chunk_offsets if chunk_offsets is not None else 1,
        grid=lambda META: (SEQUENCE, Z * KV_HEADS),
        kernel=nsa_kernel_topk,
        name="ejkernel::triton::sparse_attn_fwd_topk",
        out_shape=output_shape_and_dtype,
        **kernel_metaparams,
    )

    return block_indices_output


def fwd_triton_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """Launch the NSA sparse-attention forward kernel.

    Args:
        q: Query tensor, shape (batch, seq_len, num_q_heads, head_dim).
        k: Key tensor, shape (batch, seq_len, num_kv_heads, head_dim).
        v: Value tensor, shape (batch, seq_len, num_kv_heads, head_dim_v).
        block_indices: Pre-selected KV block indices, shape
            (batch, seq_len, num_kv_heads, max_attn_blocks).
        block_counts: Number of valid blocks per position.  Integer or
            per-position tensor of shape (batch, seq_len, num_kv_heads).
        block_size: Tokens per KV block.
        softmax_scale: Attention score scaling factor.
        cu_seqlens: Optional cumulative sequence lengths (variable-length mode).
        token_indices: Optional per-token (seq, offset) index pairs.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple (attn_output, lse):
            - attn_output: shape (batch, seq_len, num_q_heads, head_dim_v).
            - lse: Log-sum-exp, shape (batch, seq_len, num_q_heads).

    Raises:
        AssertionError: If ``head_dim > 128`` (``num_k_blocks > 1``).
    """
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    head_dim_v = v.shape[-1]
    max_attn_blocks = block_indices.shape[-1]
    num_gqa_groups = num_q_heads // num_kv_heads

    triton_block_size_k = min(128, max(int(block_k), next_power_of_2(head_dim)))
    triton_block_size_v = min(128, int(block_v), next_power_of_2(head_dim_v))

    num_k_blocks = cdiv(head_dim, triton_block_size_k)
    num_v_blocks = cdiv(head_dim_v, triton_block_size_v)
    assert num_k_blocks == 1, (
        "The key dimension cannot be split into more than one block (head_dim > 128 not supported)."
    )

    output_shapes = [
        jax.ShapeDtypeStruct((batch_size, seq_len, num_q_heads, head_dim_v), dtype=v.dtype),
        jax.ShapeDtypeStruct((batch_size, seq_len, num_q_heads), dtype="f4"),
    ]

    kernel_metaparams = dict(
        SEQUENCE=seq_len,
        KV_HEADS=num_kv_heads,
        Q_HEADS=num_q_heads,
        QK_GROUPS=num_gqa_groups,
        BLOCK_DIMK=head_dim,
        BLOCK_DIMV=head_dim_v,
        IndicesSize=max_attn_blocks,
        BLOCKSIZE=block_size,
        BLOCKSIZE_K=triton_block_size_k,
        BLOCKSIZE_V=triton_block_size_v,
    )

    attn_output, lse = triton_call(
        q,
        k,
        v,
        softmax_scale,
        block_indices if block_indices is not None else 1,
        block_counts if block_counts is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        kernel=nsa_fwd_kernel,
        grid=lambda META: (META["SEQUENCE"], num_v_blocks, batch_size * META["KV_HEADS"]),
        out_shape=output_shapes,
        name="ejkernel::triton::sparse_attn_fwd",
        num_warps=num_warps,
        num_stages=num_stages,
        **kernel_metaparams,
    )

    return attn_output, lse
