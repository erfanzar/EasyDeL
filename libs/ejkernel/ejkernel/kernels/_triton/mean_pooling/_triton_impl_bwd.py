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
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call
from ejkernel.xla_utils.utils import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def bwd_kernel(
    do,
    cu_seqlens,
    chunk_indices,
    dx,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for backward pass of mean pooling operation.

    Computes gradients for the input tensor by distributing the output
    gradient evenly across the sequence dimension.

    Args:
        do: Gradient of output tensor
        cu_seqlens: Cumulative sequence lengths for variable-length mode
        chunk_indices: Pre-computed chunk indices for variable-length mode
        dx: Gradient of input tensor (output)
        SEQUENCE: Sequence length
        HEAD: Number of attention heads
        DIM: Hidden dimension size
        BLOCK_SEQ: Block size for sequence dimension
        BLOCK_DIM: Block size for hidden dimension
        IS_VARLEN: Whether using variable-length sequences
    """
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // HEAD, i_bh % HEAD
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
        NT = tl.cdiv(SEQUENCE, BLOCK_SEQ)
    else:
        NT = tl.cdiv(SEQUENCE, BLOCK_SEQ)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    p_dx = tl.make_block_ptr(
        dx + (bos * HEAD + i_h) * DIM,
        (SEQUENCE, DIM),
        (HEAD * DIM, 1),
        (i_t * BLOCK_SEQ, i_d * BLOCK_DIM),
        (BLOCK_SEQ, BLOCK_DIM),
        (1, 0),
    )
    p_do = tl.make_block_ptr(do + (i_tg * HEAD + i_h) * DIM, (DIM,), (1,), (i_d * BLOCK_DIM,), (BLOCK_DIM,), (0,))
    b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
    b_dx = b_do / tl.full((BLOCK_SEQ,), min(BLOCK_SEQ, SEQUENCE - i_t * BLOCK_SEQ), dtype=tl.float32)[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


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
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    o_t = i_t * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    b_do = tl.load(do + i_b * DIM + o_d, mask=o_d < DIM, other=0.0).to(tl.float32) / SEQUENCE
    tl.store(
        dx + (i_b * SEQUENCE + o_t[:, None]) * DIM + o_d[None, :],
        b_do[None, :].to(dx.dtype.element_ty),
        mask=(o_t[:, None] < SEQUENCE) & (o_d[None, :] < DIM),
    )


@triton.jit
def bwd_global_varlen_kernel(
    do,
    cu_seqlens,
    dx,
    TOTAL_TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Triton kernel for packed variable-length global mean-pooling gradients."""
    i_d, i_t, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    o_t = i_t * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    length = eos - bos
    denom = tl.maximum(length, 1).to(tl.float32)
    b_do = tl.load(do + i_n * DIM + o_d, mask=o_d < DIM, other=0.0).to(tl.float32) / denom
    tl.store(
        dx + (bos + o_t[:, None]) * DIM + o_d[None, :],
        b_do[None, :].to(dx.dtype.element_ty),
        mask=(o_t[:, None] < length) & (o_d[None, :] < DIM),
    )


def bwd_triton_impl(
    do: Float[Array, "batch hidden_dim"],
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_dim: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len hidden_dim"]:
    """Execute mean pooling backward pass using Triton kernel.

    Computes gradients with respect to the input by distributing the
    output gradient across the sequence dimension.

    Args:
        do: Gradient tensor from downstream layers
        batch_size: Batch size dimension
        seq_len: Sequence length dimension
        chunk_size: Size of chunks for processing
        cu_seqlens: Optional cumulative sequence lengths for variable-length mode

    Returns:
        jax.Array: Gradient with respect to input tensor
    """
    if cu_seqlens is None and do.ndim == 2:
        Z, SEQUENCE, DIM = batch_size, seq_len, do.shape[-1]
        BLOCK_SEQ = chunk_size
        metaparams = dict(SEQUENCE=SEQUENCE, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=block_dim)
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
        TOTAL_TOKENS, DIM = batch_size, seq_len
        NUM_SEQS = len(cu_seqlens) - 1
        BLOCK_SEQ = chunk_size
        metaparams = dict(TOTAL_TOKENS=TOTAL_TOKENS, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=block_dim)
        NT = cdiv(TOTAL_TOKENS, BLOCK_SEQ)
        (dx,) = triton_call(
            do,
            cu_seqlens,
            kernel=bwd_global_varlen_kernel,
            grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NT, NUM_SEQS),
            out_shape=[jax.ShapeDtypeStruct((TOTAL_TOKENS, DIM), dtype=do.dtype)],
            name="ejkernel::triton::mean_pooling_global_varlen_bwd",
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )
        return dx

    Z, SEQUENCE, HEAD, DIM = batch_size, seq_len, *do.shape[-2:]
    BLOCK_SEQ = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = cdiv(SEQUENCE, BLOCK_SEQ) if cu_seqlens is None else len(chunk_indices)

    metaparams = dict(SEQUENCE=SEQUENCE, HEAD=HEAD, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=block_dim)
    (dx,) = triton_call(
        do,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        kernel=bwd_kernel,
        grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NT, Z * HEAD),
        out_shape=[jax.ShapeDtypeStruct((Z, SEQUENCE, HEAD, DIM), dtype=do.dtype)],
        name="ejkernel::triton::mean_pooling_bwd",
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )

    return dx
