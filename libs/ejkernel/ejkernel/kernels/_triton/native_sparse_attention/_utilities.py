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

"""Utility functions for Native Sparse Attention.

This module provides helper functions for Native Sparse Attention,
primarily for generating block masks used in the backward pass.

Block Mask Generation:
---------------------
The block mask is a boolean tensor indicating which (query, KV block)
pairs are valid attention targets. This is derived from the sparse
block indices selected during the forward pass.

For each query position:
- Look up its selected block indices
- Check if the block index is valid (within bounds and causal)
- Optionally check against block_counts for adaptive sparsity

Key Components:
--------------
nsa_kernel_mask:
    Triton kernel that converts block indices to a dense mask format.
    For each (query, selected_block) pair, marks the corresponding
    position in the output mask as True.

nsa_block_mask:
    Python function that launches the mask generation kernel.
    Handles variable-length sequences via cu_seqlens.

Memory Layout:
-------------
- Block Indices: [batch, seq_len, num_kv_heads, num_selected_blocks]
  Selected block indices per query position
- Block Counts: [batch, seq_len, num_kv_heads] or int
  Number of valid blocks per query (optional)
- Block Mask: [batch, seq_len, num_kv_heads, num_blocks]
  Output boolean mask for backward pass
"""

import jax
import triton
import triton.language as tl

from ejkernel.callib import cdiv, triton_call

from ....xla_utils.utils import prepare_lens


@triton.jit
def nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    """Convert sparse block indices to a dense boolean block mask.

    Grid: ``(SEQUENCE, batch, HEAD * SIZE)``

    For each (query-token, batch, kv_head, selected-block-slot) tuple,
    the kernel writes ``True`` into the output mask at position
    ``block_mask[b, t, h, block_index]`` if:

    * ``block_index * BLOCKSIZE <= i_t`` (causal constraint), **and**
    * ``i_s < block_counts[b, t, h]`` when ``USE_BLOCK_COUNTS=True``.

    Out-of-bounds block indices (< 0 or >= NUM_SEQS) are silently skipped.

    Args:
        block_indices: Sparse block index tensor, shape
            (batch, SEQUENCE, HEAD, SIZE).
        block_counts: Number of valid blocks per query position, shape
            (batch, SEQUENCE, HEAD) or scalar int.  Only read when
            ``USE_BLOCK_COUNTS=True``.
        block_mask: Boolean output mask, shape
            (batch, SEQUENCE, HEAD, NUM_SEQS).
        SEQUENCE: Sequence length.
        HEAD: Number of KV attention heads.
        SIZE: Number of selected blocks per query position (``IndicesSize``).
        BLOCKSIZE: Number of tokens per KV block.
        NUM_SEQS: Total number of compressed KV blocks (mask width).
        USE_BLOCK_COUNTS: Whether to use ``block_counts`` for masking.
    """
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // SIZE, i_hs % SIZE

    b_i = tl.load(block_indices + i_b * SEQUENCE * HEAD * SIZE + i_t * HEAD * SIZE + i_h * SIZE + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BLOCKSIZE <= i_t and i_s < tl.load(block_counts + i_b * SEQUENCE * HEAD + i_t * HEAD + i_h)
    else:
        b_m = b_i * BLOCKSIZE <= i_t

    if b_i < NUM_SEQS and b_i >= 0:
        tl.store(
            block_mask + i_b * SEQUENCE * HEAD * NUM_SEQS + i_t * HEAD * NUM_SEQS + i_h * NUM_SEQS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


def nsa_block_mask(
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    cu_seqlens: jax.Array,
    block_size: int,
):
    """Build a dense boolean block mask from sparse NSA block indices.

    The mask is required by the backward pass kernels to determine which
    query tokens attend to each KV block, enabling efficient dK/dV gradient
    accumulation.

    Args:
        block_indices: Sparse block index tensor, shape
            (batch, seq_len, num_kv_heads, num_selected_blocks).
        block_counts: Number of valid blocks per query position.  Can be:
            - A ``jax.Array`` of shape (batch, seq_len, num_kv_heads):
              enables ``USE_BLOCK_COUNTS`` in the kernel.
            - An integer: all positions use the same number of blocks;
              ``USE_BLOCK_COUNTS`` is disabled.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
            When not ``None``, ``NUM_SEQS`` is computed from the maximum
            sequence length rather than the uniform ``SEQUENCE``.
        block_size: Number of tokens per KV block (``BLOCKSIZE``).

    Returns:
        Boolean mask of shape (batch, seq_len, num_kv_heads, NUM_SEQS),
        where ``NUM_SEQS = ceil(max_seq_len / block_size)``.
    """
    B, SEQUENCE, HEAD, SIZE = block_indices.shape
    BLOCKSIZE = block_size
    if cu_seqlens is not None:
        NUM_SEQS = cdiv(prepare_lens(cu_seqlens).max(), BLOCKSIZE)
    else:
        NUM_SEQS = cdiv(SEQUENCE, BLOCKSIZE)

    outputs = [jax.ShapeDtypeStruct((B, SEQUENCE, HEAD, NUM_SEQS), dtype="b1")]

    metaparams = dict(
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        SIZE=SIZE,
        BLOCKSIZE=BLOCKSIZE,
        NUM_SEQS=NUM_SEQS,
        USE_BLOCK_COUNTS=isinstance(block_counts, jax.Array),
    )

    (block_mask,) = triton_call(
        block_indices,
        block_counts,
        kernel=nsa_kernel_mask,
        grid=lambda META: (SEQUENCE, B, HEAD * SIZE),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_mask",
        **metaparams,
    )

    return block_mask
