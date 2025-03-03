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

import typing as tp

import chex
import jax
import jax.lax as lax
from jax import numpy as jnp


def _chunk_attention_bias(
	blocksize_q: int,
	blocksize_k: int,
	bias: tp.Optional[chex.Array],
	segment_ids: tp.Optional[chex.Array],
	deterministic: bool,
	attn_dropout: tp.Optional[chex.Array],
	pdrop: float,
	blocksize_c: tp.Optional[int],
	dtype: jnp.dtype,
	query_chunk_idx: int,
	key_chunk_idx: int,
):
	"""Computes the attention bias for a chunk of the input.

	Args:
		blocksize_q: Size of query chunks.
		blocksize_k: Size of key chunks.
		bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
		segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
		deterministic: Whether to apply dropout.
		attn_dropout: Dropout mask.
		pdrop: Dropout probability.
		blocksize_c: Size of causal blocks.
		dtype: dtype of the computation.
		query_chunk_idx: Index of the query chunk.
		key_chunk_idx: Index of the key chunk.

	Returns:
		Attention bias for the chunk.
	"""
	query_offset = query_chunk_idx * blocksize_q
	key_offset = key_chunk_idx * blocksize_k
	chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
	if bias is not None:
		chunk_bias = lax.dynamic_slice(
			bias,
			start_indices=(0, 0, query_offset, key_offset),
			slice_sizes=(
				*bias.shape[:2],
				min(bias.shape[-2], blocksize_q),
				min(bias.shape[-1], blocksize_k),
			),
		)

	if segment_ids is not None:
		q_segment_ids = lax.dynamic_slice(
			segment_ids,
			start_indices=(0, query_offset),
			slice_sizes=(segment_ids.shape[0], blocksize_q),
		)
		k_segment_ids = lax.dynamic_slice(
			segment_ids,
			start_indices=(0, key_offset),
			slice_sizes=(segment_ids.shape[0], blocksize_k),
		)
		segment_ids_mask = ~jnp.equal(q_segment_ids[:, :, None], k_segment_ids[:, None, :])
		segment_ids_mask = segment_ids_mask[:, None]  # B1QK
		segment_ids_bias = segment_ids_mask * jnp.finfo(dtype).min
		chunk_bias = jnp.minimum(chunk_bias, segment_ids_bias)

	if blocksize_c is not None:
		query_idx = lax.broadcasted_iota(
			dtype=jnp.int32, shape=(blocksize_q, 1), dimension=0
		)
		query_idx += query_offset
		key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, blocksize_k), dimension=1)
		key_idx += key_offset
		query_idx //= blocksize_c
		key_idx //= blocksize_c
		causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
		chunk_bias = jnp.minimum(
			chunk_bias, causal_mask_value.reshape(1, 1, *causal_mask_value.shape)
		)

	if not deterministic and pdrop > 0.0:
		attn_dropout_slice = lax.dynamic_slice(
			attn_dropout,
			start_indices=(0, 0, query_offset, key_offset),
			slice_sizes=(
				*attn_dropout.shape[:2],
				min(attn_dropout.shape[-2], blocksize_q),
				min(attn_dropout.shape[-1], blocksize_k),
			),
		)
		chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
	return chunk_bias.astype(dtype)


def below_or_on_diag(
	r: int, r_blk_size: int, c: int, c_blk_size: int, blocksize_c: int
):
	"""Checks if the element at (r, c) is below or on the diagonal.

	Args:
		r: Row index.
		r_blk_size: Block size of the row.
		c: Column index.
		c_blk_size: Block size of the column.
		blocksize_c: Size of causal blocks.

	Returns:
		True if the element is below or on the diagonal, False otherwise.
	"""
	causal_block_size_q = max(blocksize_c, r_blk_size)
	causal_block_size_k = max(blocksize_c, c_blk_size)
	r = jax.lax.div(r, causal_block_size_q // r_blk_size)
	c = jax.lax.div(c, causal_block_size_k // c_blk_size)
	return ((r + 1) * causal_block_size_q - 1) > (c * causal_block_size_k)
