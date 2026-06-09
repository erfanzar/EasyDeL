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

"""Paged Attention forward pass implementation using XLA/JAX.

This module provides the forward pass for paged attention, enabling efficient
inference with non-contiguous key-value caches organized into fixed-size pages.

Key Components:
    - _page_attention_fwd: Main forward function with page-based KV lookup

Algorithm:
    Paged attention for inference with non-contiguous KV cache:
    1. For each sequence, look up which pages contain its KV cache
    2. Iterate over pages using block_tables for indirect addressing
    3. Apply attention only over valid tokens (up to context_lens)
    4. Accumulate output using online softmax for numerical stability

Features:
    - Page-based KV cache with configurable block_size
    - Efficient memory utilization via non-contiguous storage
    - GQA/MQA support with head broadcasting
    - Variable context lengths per sequence

Memory Layout:
    - key_cache: [num_blocks, num_kv_heads, block_size, head_dim]
    - value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
    - block_tables: [num_seqs, max_blocks] mapping sequence to page indices

Note:
    This is a correctness-focused XLA implementation. For TPU-optimized
    paged attention, see the Pallas implementations.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit


@ejit(static_argnums=(6,))
def _page_attention_fwd(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float,
    block_size: int,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Forward pass for paged attention using JAX/XLA.

    Implements paged attention where the KV cache is stored in non-contiguous
    blocks (pages). Each sequence has a block table that maps logical block
    indices to physical block indices in the cache. The function iterates
    over all blocks for each sequence, computes attention scores, applies
    validity masking based on context length, and produces the final output.

    Supports GQA/MQA by reshaping queries to group query heads that share
    the same KV head.

    Args:
        query: Query tensor of shape [num_seqs, num_heads, head_dim].
            Each sequence has a single query token (decode phase).
        key_cache: Paged key cache of shape
            [num_blocks, num_kv_heads, block_size, head_dim].
        value_cache: Paged value cache of shape
            [num_blocks, num_kv_heads, block_size, head_dim].
        context_lens: Context length per sequence of shape [num_seqs].
            Determines how many tokens are valid in each sequence's cache.
        block_tables: Block table mapping of shape [num_seqs, max_blocks].
            Maps logical block positions to physical block indices.
        attn_scale: Attention scaling factor applied to QK^T scores.
        block_size: Number of tokens per cache block/page (static argument).

    Returns:
        Attention output of shape [num_seqs, num_heads, head_dim].
    """
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    max_blocks = block_tables.shape[1]

    q_heads_per_kv_head = num_heads // num_kv_heads

    query = query.reshape(num_seqs, num_kv_heads, q_heads_per_kv_head, head_dim)

    query = query * attn_scale

    def attend_sequence(seq_idx):
        """Compute paged attention output for a single sequence.

        Gathers key/value blocks from the paged cache using the block
        table, computes attention scores across all blocks, applies
        softmax, and returns the weighted sum of values.

        Args:
            seq_idx: Index of the sequence in the batch.

        Returns:
            Attention output of shape [num_heads, head_dim] for this sequence.
        """
        q = query[seq_idx]
        context_len = context_lens[seq_idx]
        blocks = block_tables[seq_idx]

        def attend_block(block_idx):
            """Compute attention scores and gather values for a single cache block.

            Looks up the physical block from the block table, computes
            QK^T scores for all tokens in the block, and applies a
            validity mask based on the sequence's context length.

            Args:
                block_idx: Logical block index within this sequence's block table.

            Returns:
                Tuple of (scores, v_block) where scores has shape
                [num_kv_heads, q_heads_per_kv_head, block_size] and v_block
                has shape [num_kv_heads, block_size, head_dim].
            """
            physical_block = blocks[block_idx]

            k_block = key_cache[physical_block]
            v_block = value_cache[physical_block]

            scores = jnp.einsum("ihd,ikd->ihk", q, k_block)

            block_start = block_idx * block_size
            token_indices = jnp.arange(block_size) + block_start
            valid_mask = token_indices < context_len
            scores = jnp.where(valid_mask[None, None, :], scores, -1e9)

            return scores, v_block

        all_scores, all_values = jax.vmap(attend_block)(jnp.arange(max_blocks))

        all_scores = all_scores.transpose(1, 2, 0, 3).reshape(num_kv_heads, q_heads_per_kv_head, max_blocks * block_size)

        all_values = all_values.transpose(1, 0, 2, 3).reshape(num_kv_heads, max_blocks * block_size, head_dim)

        attn_weights = jax.nn.softmax(all_scores, axis=-1)

        output = jnp.einsum("ihk,ikd->ihd", attn_weights, all_values)

        return output.reshape(num_heads, head_dim)

    output = jax.vmap(attend_sequence)(jnp.arange(num_seqs))

    return output
