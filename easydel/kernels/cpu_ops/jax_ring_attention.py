# @article{liu2023ring,
#     title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
#     author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
#     journal={arXiv preprint arXiv:2310.01889},
#     year={2023}
# }
"""
Efficient Ring Attention Implementation for Single-Device Execution

This module provides an optimized implementation of ring attention,
originally inspired by the work of Liu et al. (2023)
([https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
It incorporates the following enhancements:

- Single-Device Focus: Adapted for efficient execution on a single device,
  removing the need for parallel communication primitives.
- Enhanced JIT Compatibility: Streamlined for smoother integration with
  JAX's Just-In-Time (JIT) compilation.
- Performance Optimizations:  Includes code optimizations for improved speed
  and memory usage.

Note: While based on existing implementations, this version offers significant
modifications to enhance its usability and performance in single-device and multi-host
settings.
- also adding softmax scale option to support custom scales
"""

import typing as tp
from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp
from jax import random as jrnd


def _ring_attention_fwd(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array],
	segment_ids: tp.Optional[chex.Array],
	axis_name: str,
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
):
	"""Forward pass for ring attention.

	Args:
		query: Query array of shape (batch, q_len, num_heads, dim_per_head).
		key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
		value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
		bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
		segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
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

	Returns:
		A tuple containing the output array and a tuple of intermediate values.
	"""
	if float32_logits:
		query, key = query.astype(jnp.float32), key.astype(jnp.float32)
	batch, q_len, num_heads, dim_per_head = query.shape
	batch, kv_len, num_heads, dim_per_head = key.shape
	numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(jnp.float32)
	denominator = jnp.zeros((batch, num_heads, q_len)).astype(jnp.float32)
	axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
	q_block_size, kv_block_size = (q_len, kv_len)

	def scan_kv_block(carry, idx):
		prev_max_score, numerator, denominator, key, value = carry
		axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
		q_block_idx = axis_idx
		q_chunk_idx_start = q_block_idx * (q_block_size // blocksize_q)
		k_block_idx = (axis_idx - idx) % axis_size
		k_chunk_idx_start = k_block_idx * (kv_block_size // blocksize_k)
		numerator, denominator, max_score = _blockwise_attention_fwd(
			query,
			key,
			value,
			(numerator, denominator, prev_max_score),
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
		key, value = map(
			lambda x: lax.ppermute(
				x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
			)
			if axis_name is not None
			else x,
			(key, value),
		)
		return (max_score, numerator, denominator, key, value), None

	prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(jnp.float32)
	(max_score, numerator, denominator, _, _), _ = lax.scan(
		scan_kv_block,
		init=(prev_max_score, numerator, denominator, key, value),
		xs=jnp.arange(0, axis_size),
	)
	output = numerator / rearrange(denominator, "b h query -> b query h")[..., None]
	return output.astype(value.dtype), (
		output,
		query,
		key,
		value,
		bias,
		segment_ids,
		denominator,
		max_score,
	)


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


@partial(
	jax.custom_vjp,
	nondiff_argnums=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
)
def ring_attention(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	segment_ids: tp.Optional[chex.Array] = None,
	axis_name: tp.Optional[str] = None,
	float32_logits: bool = True,
	softmax_scale: tp.Optional[float] = None,
	blocksize_q: int = 512,
	blocksize_k: int = 512,
	blocksize_c: tp.Optional[int] = None,
	deterministic: bool = True,
	dropout_rng: tp.Optional[chex.PRNGKey] = None,
	pdrop: float = 0.0,
	dtype: jnp.dtype = jnp.float32,
	policy=jax.checkpoint_policies.nothing_saveable,
	precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
	prevent_cse: bool = True,
):
	"""
	Computes ring attention with blockwise transformers.

	Args:
		query: Query array of shape (batch, q_len, num_heads, dim_per_head).
		key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
		value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
		bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
		segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
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

	Returns:
		Output array of shape (batch, q_len, num_heads, dim_per_head).
	"""
	y, _ = _ring_attention_fwd(
		query,
		key,
		value,
		bias,
		segment_ids,
		axis_name,
		float32_logits,
		softmax_scale,
		blocksize_q,
		blocksize_k,
		blocksize_c,
		deterministic,
		dropout_rng,
		pdrop,
		dtype,
		policy,
		precision,
		prevent_cse,
	)
	return y


ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
ring_attention = jax.jit(
	ring_attention,
	static_argnums=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
)


def _blockwise_attention_fwd(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
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
	"""Forward pass for blockwise attention.

	Args:
		query: Query array of shape (batch, q_len, num_heads, dim_per_head).
		key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
		value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
		carry: Tuple of intermediate values from the previous iteration.
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
		A tuple containing the numerator, denominator, and max score arrays.
	"""
	batch, q_len, num_heads, dim_per_head = query.shape
	batch, kv_len, num_heads, dim_per_head = key.shape
	batch, kv_len, num_heads, dim_per_head = value.shape
	num_q = q_len // blocksize_q
	num_kv = kv_len // blocksize_k
	query = query.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	key = key.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	value = value.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
	query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

	numerator, denominator, max_score = carry
	numerator = numerator.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
	numerator = jnp.moveaxis(numerator, 1, 0)
	denominator = denominator.reshape((batch, num_heads, num_q, blocksize_q))
	max_score = max_score.reshape((batch, num_heads, num_q, blocksize_q))

	denominator, max_score = map(
		lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score)
	)

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

	def scan_attention(_, scan):
		q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

		@partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
		def scan_kv_block(carry, scan):
			k_chunk, value_chunk, k_chunk_idx = scan

			numerator_chunk, denominator_chunk, prev_max_score_chunk = carry

			attn_weights = (
				jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / scale
			)
			bias_chunk = _chunk_bias_fn(
				q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx
			)
			attn_weights = attn_weights + bias_chunk

			max_score_chunk = jnp.maximum(
				prev_max_score_chunk, jnp.max(attn_weights, axis=-1)
			)
			max_score_chunk = lax.stop_gradient(max_score_chunk)
			exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
			exp_values = jnp.einsum(
				"bhqk,bkhd->bqhd", exp_weights, value_chunk, precision=precision
			)
			correction = rearrange(
				jnp.exp(prev_max_score_chunk - max_score_chunk),
				"b h query -> b query h",
			)[..., None]
			numerator_chunk = numerator_chunk * correction + exp_values
			denominator_chunk = denominator_chunk * jnp.exp(
				prev_max_score_chunk - max_score_chunk
			) + exp_weights.sum(axis=-1)

			return (
				numerator_chunk,
				denominator_chunk,
				max_score_chunk,
			), None

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
			return jax.lax.cond(
				should_run,
				scan_kv_block,
				lambda carry, args: (carry, None),
				carry,
				args,
			)

		(numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
			skip_upper_half,
			init=(numerator_chunk, denominator_chunk, max_score_chunk),
			xs=(key, value, jnp.arange(0, num_kv)),
		)
		output_chunk = numerator_chunk / rearrange(
			denominator_chunk, "b h query -> b query h"
		)[..., None].astype(dtype)
		return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

	_, (_, numerator, denominator, max_score) = lax.scan(
		scan_attention,
		init=(),
		xs=(query, numerator, denominator, max_score, jnp.arange(0, num_q)),
	)

	numerator = jnp.moveaxis(numerator, 1, 0)
	numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
	denominator, max_score = map(
		lambda x: rearrange(x, "n b h c -> b h n c"), (denominator, max_score)
	)
	denominator = denominator.reshape((batch, num_heads, q_len))
	max_score = max_score.reshape((batch, num_heads, q_len))

	return numerator, denominator, max_score


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


jax_ring_attention_mu = ring_attention
__all__ = ["jax_ring_attention_mu"]

if __name__ == "__main__":
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, QS, KS, D = 1, 32, 2048, 2048, 128
	blocksize_k = 256
	blocksize_q = 256
	query = jax.nn.initializers.normal(2)(q_key, (B, QS, H, D), dtype=jnp.float16)
	key = jax.nn.initializers.normal(2)(k_key, (B, KS, H, D), dtype=jnp.float16)
	value = jax.nn.initializers.normal(2)(v_key, (B, KS, H, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if False
		else None
	)
	print("QKV Allocated")
	try:
		co = ring_attention(
			query,
			key,
			value,
			b,
			None,
			None,
			blocksize_q=blocksize_q,
			blocksize_k=blocksize_k,
			float32_logits=False,
		)
		print(co[-1, -1, -1, :5])
	except Exception as er:
		print("Ring OOM", er)
		co = None
	try:
		import flax

		fo = flax.nnx.dot_product_attention(query, key, value, b)
		print(fo[-1, -1, -1, :5])
	except Exception as er:
		print("Flax OOM", er)
		fo = None
	if fo is not None and co is not None:
		print(jnp.allclose(co, fo, 0, 0.125))
