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

"""Backward pass Triton kernels for mean pooling operations.

This module provides gradient computation for mean pooling, enabling
end-to-end training with GPU-accelerated pooling layers.

Algorithm Overview:
------------------
The backward pass distributes the output gradient evenly across the sequence:
    dx[i] = do / seq_length  for i in range(seq_start, seq_end)

Since mean pooling averages over the sequence dimension, each input position
receives an equal fraction of the output gradient.

Memory Layout:
-------------
- do (output gradient): [batch, heads, dim] - gradient per sequence
- dx (input gradient): [batch, sequence, heads, dim] or packed format
- cu_seqlens: Cumulative sequence lengths for variable-length mode

Key Features:
------------
- Efficient gradient broadcasting with chunked writes
- Variable-length sequence support matching forward pass
- Explicit launch configuration supplied by the caller
- Memory-coalesced access patterns

Functions:
---------
- bwd_kernel: Triton kernel for backward gradient computation
- bwd_kernel_call: Python wrapper that handles kernel launch configuration
"""

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call
from ejkernel.xla_utils.utils import prepare_chunk_indices


@triton.jit
def bwd_kernel(
    do,
    cu_seqlens,
    chunk_indices,
    dx,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for backward pass of 4-D block mean pooling.

    Distributes the output gradient evenly across the valid tokens in each
    pooling chunk. Uses Triton-legal scalar division and explicit 2-D
    broadcasting so both fixed and variable-length modes compile cleanly.

    Args:
        do: Gradient of output tensor.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        chunk_indices: Pre-computed chunk indices for variable-length mode.
        dx: Gradient of input tensor (output).
        SEQUENCE: Sequence length (fixed mode).
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
        p_dx = tl.make_block_ptr(
            dx + (bos * HEAD + i_h) * DIM,
            (seq_len_i32, DIM),
            (HEAD * DIM, 1),
            ((i_t * CHUNK_SIZE).to(tl.int32), (i_d * BLOCK_DIM).to(tl.int32)),
            (BLOCK_SEQ, BLOCK_DIM),
            (1, 0),
        )
        p_do = tl.make_block_ptr(
            do + (i_tg * HEAD + i_h) * DIM,
            (DIM,),
            (1,),
            ((i_d * BLOCK_DIM).to(tl.int32),),
            (BLOCK_DIM,),
            (0,),
        )
        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32) / denom
        b_dx = tl.broadcast_to(b_do[None, :], (BLOCK_SEQ, BLOCK_DIM))
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
    else:
        o_t = i_t * CHUNK_SIZE + tl.arange(0, BLOCK_SEQ)
        o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        b_do = tl.load(do + (i_tg * HEAD + i_h) * DIM + o_d, mask=o_d < DIM, other=0.0).to(tl.float32) / denom
        b_dx = tl.broadcast_to(b_do[None, :], (BLOCK_SEQ, BLOCK_DIM))
        chunk_end = tl.minimum((i_t + 1) * CHUNK_SIZE, seq_len)
        mask = (o_t[:, None] < chunk_end) & (o_d[None, :] < DIM)
        tl.store(
            dx + ((bos + o_t[:, None]) * HEAD + i_h) * DIM + o_d[None, :],
            b_dx.to(dx.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def bwd_global_kernel(
    do,
    dx,
    SEQUENCE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Triton kernel for fixed-length global mean-pooling gradients."""
    i_d = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    o_t = i_t * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    denom = tl.maximum(SEQUENCE, 1).to(tl.float32)
    b_do = tl.load(do + i_b * DIM + o_d, mask=o_d < DIM, other=0.0).to(tl.float32) / denom
    b_dx = tl.broadcast_to(b_do[None, :], (BLOCK_SEQ, BLOCK_DIM))
    tl.store(
        dx + (i_b * SEQUENCE + o_t[:, None]) * DIM + o_d[None, :],
        b_dx.to(dx.dtype.element_ty),
        mask=(o_t[:, None] < SEQUENCE) & (o_d[None, :] < DIM),
    )


@triton.jit
def bwd_global_varlen_kernel(
    do,
    cu_seqlens,
    dx,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Triton kernel for packed variable-length global mean-pooling gradients.

    Each program handles a single sequence ``[bos, eos)`` for one feature tile,
    achieving O(total_tokens) work across all sequences.
    """
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    length = eos - bos
    denom = tl.maximum(length, 1).to(tl.float32)
    b_do = tl.load(do + i_n * DIM + o_d, mask=o_d < DIM, other=0.0).to(tl.float32) / denom
    b_dx = tl.broadcast_to(b_do[None, :], (BLOCK_SEQ, BLOCK_DIM)).to(dx.dtype.element_ty)

    for i_t in range(0, length, BLOCK_SEQ):
        o_t = i_t + tl.arange(0, BLOCK_SEQ)
        tl.store(
            dx + (bos + o_t[:, None]) * DIM + o_d[None, :],
            b_dx,
            mask=(o_t[:, None] < length) & (o_d[None, :] < DIM),
        )


def bwd_triton_impl(
    do: Float[Array, "... hidden_dim"],
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_dim: int = 64,
    block_size: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "... hidden_dim"]:
    """Execute mean pooling backward pass using Triton kernel.

    Computes gradients with respect to the input by distributing the
    output gradient across the sequence dimension.

    Args:
        do: Gradient tensor from downstream layers.
        batch_size: Leading input dimension (batch size or total_tokens).
        seq_len: Second input dimension (seq_len or hidden_dim for 2-D varlen).
        chunk_size: Semantic pooling window size for 4-D block pooling.
        cu_seqlens: Optional cumulative sequence lengths for variable-length mode.
        block_dim: Hidden-dimension tile width.
        block_size: Sequence-axis tile size for 2-D/3-D global pooling.
        num_warps: Number of Triton warps per block.
        num_stages: Number of Triton pipeline stages.

    Returns:
        jax.Array: Gradient with respect to input tensor.
    """
    BLOCK_DIM = triton.next_power_of_2(max(1, int(block_dim)))

    if cu_seqlens is None and do.ndim == 2:
        Z, SEQUENCE, DIM = int(batch_size), int(seq_len), do.shape[-1]
        BLOCK_SEQ = triton.next_power_of_2(min(max(1, SEQUENCE), max(1, int(block_size))))
        metaparams = dict(SEQUENCE=SEQUENCE, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=BLOCK_DIM)
        NT = cdiv(SEQUENCE, BLOCK_SEQ)
        (dx,) = triton_call(
            do,
            kernel=bwd_global_kernel,
            grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NT, Z),
            out_shape=[jax.ShapeDtypeStruct((Z, SEQUENCE, DIM), dtype=do.dtype)],
            name="ejkernel::triton::mean_pooling_global_bwd",
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )
        return dx

    if cu_seqlens is not None and do.ndim == 2:
        TOTAL_TOKENS, DIM = int(batch_size), int(seq_len)
        NUM_SEQS = int(cu_seqlens.shape[0]) - 1
        if NUM_SEQS == 0:
            return jnp.zeros((TOTAL_TOKENS, DIM), dtype=do.dtype)
        BLOCK_SEQ = triton.next_power_of_2(min(max(1, TOTAL_TOKENS), max(1, int(block_size))))
        metaparams = dict(DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=BLOCK_DIM)
        (dx,) = triton_call(
            do,
            cu_seqlens,
            kernel=bwd_global_varlen_kernel,
            grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NUM_SEQS),
            out_shape=[jax.ShapeDtypeStruct((TOTAL_TOKENS, DIM), dtype=do.dtype)],
            name="ejkernel::triton::mean_pooling_global_varlen_bwd",
            zeroed_outputs=(0,),
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )
        return dx

    Z, SEQUENCE, HEAD, DIM = int(batch_size), int(seq_len), *do.shape[-2:]
    CHUNK_SIZE = max(1, int(chunk_size))
    BLOCK_SEQ = triton.next_power_of_2(CHUNK_SIZE)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = cdiv(SEQUENCE, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    metaparams = dict(
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        DIM=DIM,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        IS_VARLEN=cu_seqlens is not None,
    )
    (dx,) = triton_call(
        do,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        kernel=bwd_kernel,
        grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NT, Z * HEAD),
        out_shape=[jax.ShapeDtypeStruct((Z, SEQUENCE, HEAD, DIM), dtype=do.dtype)],
        name="ejkernel::triton::mean_pooling_bwd",
        zeroed_outputs=(0,) if cu_seqlens is not None else (),
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )

    return dx
