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

"""Forward pass Triton kernels for mean pooling operations.

This module provides GPU-accelerated mean pooling over sequence dimensions,
supporting both fixed-length and variable-length sequences.

Algorithm Overview:
------------------
Mean pooling computes the average of hidden states across the sequence dimension:
    output = sum(x[seq_start:seq_end], dim=0) / seq_length

For variable-length sequences, each sequence in the batch may have a different
length, and the pooling is done independently for each sequence.

Memory Layout:
-------------
- Input: [batch, sequence, heads, dim] or packed [total_tokens, heads, dim]
- Output: [batch, heads, dim] - one vector per sequence
- cu_seqlens: Cumulative sequence lengths for variable-length mode

Key Features:
------------
- Chunked processing for efficient memory access patterns
- Explicit block sizes supplied by the launcher
- Variable-length sequence support via cumulative indices
- Fused kernel implementation for reduced memory bandwidth

Functions:
---------
- fwd_kernel: Triton kernel for forward mean pooling computation
- fwd_kernel_call: Python wrapper that handles kernel launch configuration
"""

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call
from ejkernel.xla_utils.utils import prepare_chunk_indices


@triton.jit
def fwd_kernel(
    x,
    cu_seqlens,
    chunk_indices,
    o,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for forward pass of 4-D block mean pooling.

    Computes mean pooling over chunks of size ``CHUNK_SIZE`` along the sequence
    dimension. Supports both fixed-length and variable-length sequences and
    non-power-of-2 ``CHUNK_SIZE`` values via power-of-2 ``BLOCK_SEQ`` padding.

    Args:
        x: Input tensor pointer.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        chunk_indices: Pre-computed chunk indices for variable-length mode.
        o: Output tensor pointer.
        SEQUENCE: Sequence length (compile-time constant for fixed mode).
        HEAD: Number of attention heads.
        DIM: Hidden dimension size.
        CHUNK_SIZE: Semantic pooling window size along the sequence axis.
        BLOCK_SEQ: Power-of-2 tile width covering ``CHUNK_SIZE``.
        BLOCK_DIM: Power-of-2 tile width along ``DIM``.
        IS_VARLEN: Whether using variable-length sequences.
    """
    i_d = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HEAD, i_bh % HEAD
    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        seq_len_i32 = seq_len.to(tl.int32)
    else:
        num_seq_block = tl.cdiv(SEQUENCE, CHUNK_SIZE)
        i_tg = i_b * num_seq_block + i_t
        bos = i_b * SEQUENCE
        seq_len = SEQUENCE
        seq_len_i32 = SEQUENCE

    valid_len = tl.minimum(CHUNK_SIZE, seq_len - i_t * CHUNK_SIZE)
    denom = tl.maximum(valid_len, 1).to(tl.float32)

    if CHUNK_SIZE == BLOCK_SEQ:
        p_x = tl.make_block_ptr(
            x + (bos * HEAD + i_h) * DIM,
            (seq_len_i32, DIM),
            (HEAD * DIM, 1),
            ((i_t * CHUNK_SIZE).to(tl.int32), (i_d * BLOCK_DIM).to(tl.int32)),
            (BLOCK_SEQ, BLOCK_DIM),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (i_tg * HEAD + i_h) * DIM,
            (DIM,),
            (1,),
            ((i_d * BLOCK_DIM).to(tl.int32),),
            (BLOCK_DIM,),
            (0,),
        )
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.sum(b_x, axis=0) / denom
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
    else:
        o_t = i_t * CHUNK_SIZE + tl.arange(0, BLOCK_SEQ)
        o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        chunk_end = tl.minimum((i_t + 1) * CHUNK_SIZE, seq_len)
        mask = (o_t[:, None] < chunk_end) & (o_d[None, :] < DIM)
        b_x = tl.load(
            x + ((bos + o_t[:, None]) * HEAD + i_h) * DIM + o_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b_o = tl.sum(b_x, axis=0) / denom
        tl.store(
            o + (i_tg * HEAD + i_h) * DIM + o_d,
            b_o.to(o.dtype.element_ty),
            mask=o_d < DIM,
        )


@triton.jit
def fwd_global_kernel(
    x,
    o,
    SEQUENCE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Triton kernel for fixed-length global mean pooling over 3-D inputs."""
    i_d = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1).to(tl.int64)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    for i_t in range(0, SEQUENCE, BLOCK_SEQ):
        o_t = i_t + tl.arange(0, BLOCK_SEQ)
        b_x = tl.load(
            x + (i_b * SEQUENCE + o_t[:, None]) * DIM + o_d[None, :],
            mask=(o_t[:, None] < SEQUENCE) & (o_d[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(b_x, axis=0)

    denom = tl.maximum(SEQUENCE, 1).to(tl.float32)
    b_o = acc / denom
    tl.store(o + i_b * DIM + o_d, b_o.to(o.dtype.element_ty), mask=o_d < DIM)


@triton.jit
def fwd_global_varlen_kernel(
    x,
    cu_seqlens,
    o,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Triton kernel for packed variable-length global mean pooling.

    Each program processes a single sequence ``[bos, eos)`` for one feature
    tile, giving O(total_tokens) work across all sequences.
    """
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    length = eos - bos
    acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    for i_t in range(0, length, BLOCK_SEQ):
        o_t = i_t + tl.arange(0, BLOCK_SEQ)
        b_x = tl.load(
            x + (bos + o_t[:, None]) * DIM + o_d[None, :],
            mask=(o_t[:, None] < length) & (o_d[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(b_x, axis=0)

    denom = tl.maximum(length, 1).to(tl.float32)
    b_o = acc / denom
    tl.store(o + i_n * DIM + o_d, b_o.to(o.dtype.element_ty), mask=o_d < DIM)


def fwd_triton_impl(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_dim: int = 64,
    block_size: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "... hidden_dim"]:
    """Execute mean pooling forward pass using Triton kernel.

    Launches the Triton kernel for efficient mean pooling computation,
    handling 3-D fixed global, 2-D packed varlen global, and 4-D block pooling.

    Args:
        x: Input tensor of shape ``[batch, seq_len, dim]``,
            ``[total_tokens, dim]`` (with ``cu_seqlens``), or
            ``[batch, seq_len, heads, dim]`` (4-D block pooling).
        chunk_size: Semantic pooling window size for 4-D block pooling.
        cu_seqlens: Optional cumulative sequence lengths for variable-length mode.
        block_dim: Hidden-dimension tile width.
        block_size: Sequence-axis tile size for 2-D/3-D global pooling.
        num_warps: Number of Triton warps per block.
        num_stages: Number of Triton pipeline stages.

    Returns:
        jax.Array: Mean-pooled output tensor.
    """
    BLOCK_DIM = triton.next_power_of_2(max(1, int(block_dim)))

    if cu_seqlens is None and x.ndim == 3:
        Z, SEQUENCE, DIM = x.shape
        BLOCK_SEQ = triton.next_power_of_2(min(max(1, SEQUENCE), max(1, int(block_size))))
        metaparams = dict(SEQUENCE=SEQUENCE, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=BLOCK_DIM)
        (o,) = triton_call(
            x,
            kernel=fwd_global_kernel,
            out_shape=[jax.ShapeDtypeStruct((Z, DIM), x.dtype)],
            name="ejkernel::triton::mean_pooling_global_fwd",
            grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), Z),
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )
        return o

    if cu_seqlens is not None and x.ndim == 2:
        TOTAL_TOKENS, DIM = x.shape
        NUM_SEQS = int(cu_seqlens.shape[0]) - 1
        if NUM_SEQS == 0:
            return jnp.zeros((0, DIM), dtype=x.dtype)
        BLOCK_SEQ = triton.next_power_of_2(min(max(1, TOTAL_TOKENS), max(1, int(block_size))))
        metaparams = dict(DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=BLOCK_DIM)
        (o,) = triton_call(
            x,
            cu_seqlens,
            kernel=fwd_global_varlen_kernel,
            out_shape=[jax.ShapeDtypeStruct((NUM_SEQS, DIM), x.dtype)],
            name="ejkernel::triton::mean_pooling_global_varlen_fwd",
            grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NUM_SEQS),
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )
        return o

    if x.ndim != 4:
        raise ValueError("Triton mean_pooling expects 3-D fixed, 2-D packed varlen, or 4-D block-pooling inputs.")

    Z, SEQUENCE, HEAD, DIM = x.shape
    CHUNK_SIZE = max(1, int(chunk_size))
    BLOCK_SEQ = triton.next_power_of_2(CHUNK_SIZE)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NUM_SEQ_BLOCK = cdiv(SEQUENCE, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        DIM=DIM,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        IS_VARLEN=cu_seqlens is not None,
    )
    (o,) = triton_call(
        x,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        kernel=fwd_kernel,
        out_shape=[jax.ShapeDtypeStruct((Z, NUM_SEQ_BLOCK, HEAD, DIM), x.dtype)],
        name="ejkernel::triton::mean_pooling_fwd",
        grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NUM_SEQ_BLOCK, Z * HEAD),
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    return o
