# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas.triton import TritonCompilerParams

from ._forward_pallas import _ragged_paged_attention_pallas_kernel  # type:ignore


def ragged_paged_attention(
    queries: jnp.ndarray,
    key_pages: jnp.ndarray,
    value_pages: jnp.ndarray,
    sequence_lengths: jnp.ndarray,
    sequence_page_indices: jnp.ndarray,
    cumulative_query_lengths: jnp.ndarray,
    num_sequences: int,
    softmax_scale: float,
    soft_cap: float | None = None,
) -> jnp.ndarray:
    """
    Performs ragged paged attention using a Pallas kernel.

    Args are identical to the original lax-based function.
    """
    total_query_tokens, num_q_heads, head_size = queries.shape
    num_pages, page_size, num_kv_heads, _ = value_pages.shape
    _, max_pages_per_sequence = sequence_page_indices.shape
    out_shape = (total_query_tokens, num_q_heads, head_size)
    q_heads_per_group = num_q_heads // num_kv_heads
    row_indices = jnp.arange(total_query_tokens)
    q_sequence_indices = jnp.searchsorted(cumulative_query_lengths, row_indices, side="right") - 1
    num_queries_per_seq = jnp.diff(cumulative_query_lengths)
    num_queries_for_my_seq = num_queries_per_seq[q_sequence_indices]
    kv_len_for_my_seq = sequence_lengths[q_sequence_indices]
    relative_q_indices = row_indices - cumulative_query_lengths[q_sequence_indices]
    q_token_positions = kv_len_for_my_seq - num_queries_for_my_seq + relative_q_indices
    KV_PAGE_BLOCK_SIZE = min(16, max_pages_per_sequence if max_pages_per_sequence > 0 else 16)

    grid = (total_query_tokens, num_q_heads)
    attention_output = pl.pallas_call(
        partial(
            _ragged_paged_attention_pallas_kernel,
            softmax_scale=softmax_scale,
            soft_cap=soft_cap,
            q_heads_per_group=q_heads_per_group,
            KV_PAGE_BLOCK_SIZE=KV_PAGE_BLOCK_SIZE,
        ),
        out_shape=jax.ShapeDtypeStruct(out_shape, queries.dtype),
        in_specs=[
            pl.BlockSpec((1, 1, head_size), lambda q_idx, q_head_idx: (q_idx, q_head_idx, 0)),
            pl.BlockSpec(key_pages.shape, lambda q_idx, q_head_idx: (0,) * key_pages.ndim),
            pl.BlockSpec(value_pages.shape, lambda q_idx, q_head_idx: (0,) * value_pages.ndim),
            pl.BlockSpec(sequence_page_indices.shape, lambda q_idx, q_head_idx: (0,) * sequence_page_indices.ndim),
            pl.BlockSpec(sequence_lengths.shape, lambda q_idx, q_head_idx: (0,) * sequence_lengths.ndim),
            pl.BlockSpec((1,), lambda q_idx, q_head_idx: (q_idx,)),
            pl.BlockSpec((1,), lambda q_idx, q_head_idx: (q_idx,)),
        ],
        out_specs=pl.BlockSpec((1, 1, head_size), lambda q_idx, q_head_idx: (q_idx, q_head_idx, 0)),
        grid=grid,
        compiler_params=TritonCompilerParams(4, 2),
    )(queries, key_pages, value_pages, sequence_page_indices, sequence_lengths, q_sequence_indices, q_token_positions)

    return attention_output
