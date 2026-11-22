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

"""Chunked Prefill with Paged Decode forward pass using XLA/JAX.

This module provides the forward pass for combined chunked prefill and
paged decode attention, supporting mixed batches with both prefill and
decode sequences.

Key Components:
    - _update_block_tabled_kv_cache: Update paged KV cache with new tokens
    - _chunked_prefill_paged_decode_fwd: Main forward function

Algorithm:
    Combined prefill and decode in a single batch:
    1. Separate batch into prefill and decode sequences
    2. For prefill: Process multiple query tokens against cached KV
    3. For decode: Process single query token per sequence
    4. Update KV cache with new tokens using block tables

Features:
    - Mixed prefill/decode batching for throughput optimization
    - Block-based KV cache with configurable page size
    - Efficient cache updates using scatter operations
    - Supports variable prefill chunk sizes

Memory Layout:
    - key_cache/value_cache: [num_blocks, page_size, num_kv_heads, head_dim]
    - block_tables: [num_seqs, max_blocks] for page index lookup
    - kv_lens: [num_seqs] current sequence lengths

Note:
    This kernel enables efficient batching of prefill and decode
    operations, reducing scheduling overhead in serving scenarios.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.kernels._xla.unified_attention import unified_attention as xla_unified_attention


def _update_block_tabled_kv_cache(
    *,
    keys: jax.Array,
    values: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Update block-tabled KV cache with new key/value pairs (XLA implementation).

    Uses JAX's lax.fori_loop to iterate over sequences and tokens, inserting
    keys/values into their corresponding cache positions via block table mapping.

    Args:
        keys: Packed key vectors [total_tokens, num_kv_heads, head_dim]
        values: Packed value vectors [total_tokens, num_kv_heads, head_dim]
        key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
        kv_lens: Current KV lengths per sequence [num_seqs]
        block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
        query_start_loc: Cumulative query positions [num_seqs + 1]

    Returns:
        Tuple of (updated_key_cache, updated_value_cache)
    """
    num_blocks, block_size, num_kv_heads, head_dim = map(int, key_cache.shape)
    key_cache_flat = key_cache.reshape(num_blocks * block_size, num_kv_heads, head_dim)
    value_cache_flat = value_cache.reshape(num_blocks * block_size, num_kv_heads, head_dim)

    num_seqs = int(block_tables.shape[0])

    def _seq_body(seq_idx, carry):
        """Process a single sequence, inserting its tokens into the KV cache.

        Args:
            seq_idx: Index of the current sequence being processed.
            carry: Tuple of (key_cache_flat, value_cache_flat) accumulators.

        Returns:
            Updated (key_cache_flat, value_cache_flat) with tokens from
            this sequence inserted at the correct cache positions.
        """
        k_flat, v_flat = carry
        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        ctx_len = kv_lens[seq_idx] - q_len

        def _tok_body(t, carry_tok):
            """Insert a single token's key/value into the flattened cache.

            Computes the physical cache position from the block table
            and inserts the token's key and value at that position.

            Args:
                t: Token offset within this sequence's query chunk.
                carry_tok: Tuple of (key_cache_flat, value_cache_flat).

            Returns:
                Updated (key_cache_flat, value_cache_flat) with this
                token's key/value inserted.
            """
            k_flat_tok, v_flat_tok = carry_tok
            pos = ctx_len + t
            block_idx = pos // jnp.int32(block_size)
            within = pos - block_idx * jnp.int32(block_size)
            phys_block = block_tables[seq_idx, block_idx]
            linear = phys_block * jnp.int32(block_size) + within
            tok = q_start + t
            k_flat_tok = k_flat_tok.at[linear].set(keys[tok])
            v_flat_tok = v_flat_tok.at[linear].set(values[tok])
            return k_flat_tok, v_flat_tok

        return lax.fori_loop(0, q_len, _tok_body, (k_flat, v_flat))

    key_cache_flat, value_cache_flat = lax.fori_loop(0, num_seqs, _seq_body, (key_cache_flat, value_cache_flat))
    return (
        key_cache_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim),
        value_cache_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim),
    )


def _chunked_prefill_paged_decode_fwd(
    *,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    alibi_slopes: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """XLA forward pass for chunked prefill + paged decode attention.

    Updates KV cache and computes attention using pure JAX operations compatible
    with all XLA backends.

    Args:
        queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
        keys: Packed keys to insert [total_tokens, num_kv_heads, head_dim]
        values: Packed values to insert [total_tokens, num_kv_heads, head_dim]
        key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
        kv_lens: KV lengths per sequence [num_seqs]
        block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
        query_start_loc: Cumulative query positions [num_seqs + 1]
        alibi_slopes: Optional ALiBi slopes [num_q_heads]
        softmax_aux: Optional auxiliary softmax parameters [num_q_heads]
        softmax_scale: Attention scaling factor
        causal: Whether to apply causal masking
        sliding_window: Optional local attention window size
        logits_soft_cap: Optional logit capping value

    Returns:
        Tuple of (attention_output, updated_key_cache, updated_value_cache)

    Raises:
        NotImplementedError: If causal=False
        ValueError: If input shapes are invalid or inconsistent
    """
    if not causal:
        raise NotImplementedError("chunked_prefill_paged_decode only supports causal attention.")
    if queries.ndim != 3:
        raise ValueError("queries must be [total_tokens, num_q_heads, head_dim]")
    if keys.ndim != 3 or values.ndim != 3:
        raise ValueError("keys/values must be [total_tokens, num_kv_heads, head_dim]")
    if keys.shape != values.shape:
        raise ValueError("keys/values shape mismatch")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache shape mismatch")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    if keys.shape[0] != total_tokens:
        raise ValueError("keys/values first dim must match total_tokens")
    num_seqs = int(kv_lens.shape[0])
    if query_start_loc.shape != (num_seqs + 1,):
        raise ValueError("query_start_loc must have shape (num_seqs + 1,)")

    new_key_cache, new_value_cache = _update_block_tabled_kv_cache(
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
    )

    out = xla_unified_attention(
        queries=queries,
        key_cache=new_key_cache,
        value_cache=new_value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        qq_bias=None,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=None,
        num_par_softmax_segments=None,
        num_warps=None,
        num_stages=None,
    )

    assert out.shape == (total_tokens, num_q_heads, head_dim)
    return out, new_key_cache, new_value_cache
