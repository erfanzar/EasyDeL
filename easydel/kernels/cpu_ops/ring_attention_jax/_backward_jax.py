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
from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp

from ._utils import _chunk_attention_bias, below_or_on_diag


def _blockwise_attention_bwd(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	g: chex.Array,
	carry,
	q_chunk_idx_start: int,
	k_chunk_idx_start: int,
	bias: tp.Optional[chex.Array],
	segment_ids: tp.Optional[chex.Array],
	softmax_scale: tp.Optional[float],
	blocksize_c: tp.Optional[int],
	blocksize_q: int,
	blocksize_k: int,
	deterministic: bool,
	dropout_rng: tp.Optional[chex.PRNGKey],
	pdrop: float,
	dtype: jnp.dtype,
	policy,
	precision: lax.PrecisionLike,
	prevent_cse: bool,
):
	"""Backward pass for blockwise attention.

	Args:
		query: Query array of shape (batch, q_len, num_heads, dim_per_head).
		key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
		value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
		g: Gradient of the output.
		carry: Tuple of intermediate values from the forward pass.
		q_chunk_idx_start: Start index of the query chunk.
		k_chunk_idx_start: Start index of the key chunk.
		bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
		segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
		softmax_scale: scale for softmax or depth ** -0.5.
		blocksize_c: Size of causal blocks.
		blocksize_q: Size of query chunks.
		blocksize_k: Size of key chunks.
		deterministic: Whether to apply dropout.
		dropout_rng: PRNG key for dropout.
		pdrop: Dropout probability.
		dtype: dtype of the computation.
		policy: Checkpoint policy.
		precision: Precision of the computation.
		prevent_cse: Whether to prevent common subexpression elimination.

	Returns:
		A tuple containing the gradients of the query, key, and value arrays.
	"""
	batch, q_len, num_heads, dim_per_head = query.shape
	batch, kv_len, num_heads, dim_per_head = key.shape
	batch, kv_len, num_heads, dim_per_head = value.shape
	num_q = q_len // blocksize_q
	num_kv = kv_len // blocksize_k
	dq, dk, dv, output, denominator, max_score = carry
	g = g.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	dq = dq.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	dk = dk.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	dv = dv.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	output = output.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

	denominator = denominator.reshape((batch, num_heads, num_q, blocksize_q))
	max_score = max_score.reshape((batch, num_heads, num_q, blocksize_q))
	denominator, max_score = map(
		lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score)
	)

	query = query.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	key = key.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	value = value.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

	scale = jnp.sqrt(query.shape[-1]) if softmax_scale is None else 1 / softmax_scale
	if not deterministic and pdrop > 0.0:
		attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
		attn_dropout = jax.random.bernoulli(
			attn_dropout_rng, pdrop, (batch, num_heads, q_len, kv_len)
		)
	else:
		attn_dropout = None
	_chunk_bias_fn = partial(
		_chunk_attention_bias,
		blocksize_q,
		blocksize_k,
		bias,
		segment_ids,
		deterministic,
		attn_dropout,
		pdrop,
		blocksize_c,
		dtype,
	)

	def scan_attention(carry, scan):
		dk, dv = carry
		(
			q_chunk,
			dq_chunk,
			g_chunk,
			output_chunk,
			denominator_chunk,
			max_score_chunk,
			q_chunk_idx,
		) = scan
		dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]

		@partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
		def scan_kv_block(carry, scan):
			k_chunk, value_chunk, k_chunk_idx = scan
			dq_chunk = carry
			attn_weights = (
				jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / scale
			)
			bias_chunk = _chunk_bias_fn(
				q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx
			)
			attn_weights = attn_weights + bias_chunk
			exp_weights = (
				jnp.exp(attn_weights - max_score_chunk[..., None])
				/ denominator_chunk[..., None]
			)

			ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
			dl = (ds - dl_part) * exp_weights
			dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
			dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
			dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)

			return dq_chunk, (
				dk_chunk,
				dv_chunk,
			)

		def skip_upper_half(carry, args):
			key_chunk, value_chunk, k_chunk_idx = args
			should_run = jnp.array(True)
			if blocksize_c is not None:
				should_run = below_or_on_diag(
					q_chunk_idx_start + q_chunk_idx,
					blocksize_q,
					k_chunk_idx_start + k_chunk_idx,
					blocksize_k,
					blocksize_c,
				)
			return lax.cond(
				should_run,
				scan_kv_block,
				lambda carry, args: (
					carry,
					(
						jnp.zeros(
							(batch, blocksize_k, num_heads, dim_per_head),
							dtype=jnp.float32,
						),
						jnp.zeros(
							(batch, blocksize_k, num_heads, dim_per_head),
							dtype=jnp.float32,
						),
					),
				),
				carry,
				args,
			)

		dq_chunk, (dk_part, dv_part) = lax.scan(
			skip_upper_half,
			init=dq_chunk,
			xs=(key, value, jnp.arange(0, num_kv)),
		)
		return (dk + dk_part, dv + dv_part), dq_chunk

	(dk, dv), dq = lax.scan(
		scan_attention,
		init=(dk, dv),
		xs=(
			query,
			dq,
			g,
			output,
			denominator,
			max_score,
			jnp.arange(0, num_q),
		),
	)

	dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
	dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
	dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
	dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

	return dq, dk, dv


def _ring_attention_bwd(
	axis_name: tp.Optional[str],
	float32_logits: bool,
	softmax_scale: tp.Optional[float],
	blocksize_q: int,
	blocksize_k: int,
	blocksize_c: tp.Optional[int],
	deterministic: bool,
	dropout_rng: tp.Optional[chex.PRNGKey],
	pdrop: float,
	dtype: jnp.dtype,
	policy,
	precision: lax.PrecisionLike,
	prevent_cse: bool,
	res,
	g: chex.Array,
):
	"""Backward pass for ring attention.

	Args:
		axis_name: Name of the axis to ppermute over.
		float32_logits: Whether to compute logits in float32.
		softmax_scale: scale for softmax or depth ** -0.5.
		blocksize_q: Size of query chunks.
		blocksize_k: Size of key chunks.
		blocksize_c: Size of causal blocks.
		deterministic: Whether to apply dropout.
		dropout_rng: PRNG key for dropout.
		pdrop: Dropout probability.
		dtype: dtype of the computation.
		policy: Checkpoint policy.
		precision: Precision of the computation.
		prevent_cse: Whether to prevent common subexpression elimination.
		res: Tuple of intermediate values from the forward pass.
		g: Gradient of the output.

	Returns:
		A tuple containing the gradients of the inputs.
	"""
	del float32_logits
	output, query, key, value, bias, segment_ids, denominator, max_score = res
	_, q_len, _, _ = query.shape
	_, kv_len, _, _ = key.shape
	axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
	dq = jnp.zeros_like(query, dtype=jnp.float32)
	dk = jnp.zeros_like(key, dtype=jnp.float32)
	dv = jnp.zeros_like(value, dtype=jnp.float32)
	q_block_size, kv_block_size = (
		q_len,
		kv_len,
	)  # assumes this function is pre-sharded inside shard_map

	def scan_kv_block(carry, idx):
		dq, dk, dv, key, value = carry
		axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
		q_block_idx = axis_idx
		q_chunk_idx_start = q_block_idx * (q_block_size // blocksize_q)
		k_block_idx = (axis_idx - idx) % axis_size
		k_chunk_idx_start = k_block_idx * (kv_block_size // blocksize_k)
		dq, dk, dv = _blockwise_attention_bwd(
			query,
			key,
			value,
			g,
			(dq, dk, dv, output, denominator, max_score),
			q_chunk_idx_start,
			k_chunk_idx_start,
			bias=bias,
			segment_ids=segment_ids,
			softmax_scale=softmax_scale,
			blocksize_q=blocksize_q,
			blocksize_k=blocksize_k,
			blocksize_c=blocksize_c,
			deterministic=deterministic,
			dropout_rng=dropout_rng,
			pdrop=pdrop,
			dtype=dtype,
			policy=policy,
			precision=precision,
			prevent_cse=prevent_cse,
		)
		key, value, dk, dv = map(
			lambda x: lax.ppermute(
				x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
			)
			if axis_name is not None
			else x,
			(key, value, dk, dv),
		)
		return (dq, dk, dv, key, value), None

	(dq, dk, dv, key, value), _ = lax.scan(
		scan_kv_block, init=(dq, dk, dv, key, value), xs=jnp.arange(0, axis_size)
	)
	dq, dk, dv = dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(key.dtype)
	return dq, dk, dv, None, None
