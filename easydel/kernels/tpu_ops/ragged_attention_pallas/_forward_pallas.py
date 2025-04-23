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

# Update/ Added more features from Google version

import functools

import numpy as np

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.experimental import shard_map


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
shard_map = shard_map.shard_map


def get_mha_cost_estimate(shape_dtype):
	"""Provides a Pallas CostEstimate for Multi-Head Attention.

	Calculates an approximate cost estimate (FLOPs, transcendentals, bytes accessed)
	for a Multi-Head Attention operation based on the input shapes and dtypes.
	This is used to inform the Pallas compiler.

	Args:
	    shape_dtype: A tuple containing JAX ShapeDtypeStruct objects for the
	        query, key, value, and lengths tensors. Expected shapes are:
	        - query: [batch_size, 1, num_heads, head_dim]
	        - key: [batch_size, seq_len, num_heads, head_dim]
	        - value: [batch_size, seq_len, num_heads, head_dim]
	        - lengths: [batch_size]

	Returns:
	    A pl.CostEstimate object containing the estimated costs.
	"""
	batch_size, _, num_heads, head_dim = shape_dtype[0].shape
	seq_len = shape_dtype[1].shape[1]

	# Approximate flops calculation for attention
	# fmt: off
	flops = batch_size * num_heads * seq_len * (
    2 * head_dim +  # QK multiplication
    seq_len +       # softmax
    2 * head_dim    # V multiplication
  )
	# fmt: on

	return pl.CostEstimate(
		flops=flops,
		transcendentals=batch_size * num_heads * seq_len,
		bytes_accessed=int(sum(np.prod(s.shape) * s.dtype.itemsize for s in shape_dtype)),
	)


@functools.partial(jax.jit, static_argnames=["mask_value"])
def reference_mqa(
	q: jax.Array,  # Input shape from vmap(MHA): (B, 1, D)
	k: jax.Array,  # Input shape from vmap(MHA): (B, T, D)
	v: jax.Array,  # Input shape from vmap(MHA): (B, T, D)
	lengths: jax.Array,  # (B,)
	starts: jax.Array,  # Input: (B,)
	*,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Multi query attention reference (Called by vmap in reference_mha)."""
	seq_len = k.shape[1]
	logits = jnp.einsum("bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32))
	rages = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
	mask = (starts.reshape(-1, 1) <= rages) < (rages < lengths.reshape(-1, 1))

	logits = logits + jnp.where(mask[:, None, :], 0.0, mask_value)

	logits_max = logits.max(axis=-1)
	logits_shifted = logits - logits_max[..., None]
	unnormalized = jnp.exp(logits_shifted)
	denominator = unnormalized.sum(axis=-1)
	safe_denominator = jnp.maximum(denominator, jnp.finfo(denominator.dtype).tiny)

	o = (
		jnp.einsum("bht,btd->bhd", unnormalized.astype(v.dtype), v)
		/ safe_denominator[..., None]
	)
	return o, logits_max[..., None], denominator[..., None]


@jax.jit  # Keep JIT here
def reference_mha(
	q: jax.Array,  # Input: (B, 1, H, D)
	k: jax.Array,  # Input: (B, T, H, D)
	v: jax.Array,  # Input: (B, T, H, D)
	lengths: jax.Array,  # Input: (B,)
	starts: jax.Array,  # Input: (B,)
	*,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Multi head attention reference.

	Returns:
	  o: [batch_size, 1, num_heads, head_dim]
	  m: [batch_size, 1, num_heads, 1]
	  l: [batch_size, 1, num_heads, 1]
	"""

	q = jnp.swapaxes(q, 1, 2)
	k = jnp.swapaxes(k, 1, 2)
	v = jnp.swapaxes(v, 1, 2)

	o, m, l = jax.vmap(  # noqa
		functools.partial(reference_mqa, mask_value=mask_value),
		in_axes=(1, 1, 1, None, None),
		out_axes=2,
	)(q, k, v, lengths, starts)

	return o, m, l


@functools.partial(jax.jit, static_argnames=["mask_value"])
def reference_gqa(
	q: jax.Array,  # Input shape expected by this ref impl: (B, Hq, D)
	k: jax.Array,  # Input shape expected by this ref impl: (B, T, Hkv, D)
	v: jax.Array,  # Input shape expected by this ref impl: (B, T, Hkv, D)
	lengths: jax.Array,  # (B,)
	starts: jax.Array,  # Input: (B,)
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Vanilla attention GQA implementation for reference.

	Returns:
	  o: [batch_size, 1, num_q_heads, head_dim]
	  m: [batch_size, 1, num_q_heads, 1]
	  l: [batch_size, 1, num_q_heads, 1]
	"""
	batch_size, _, num_heads_q, head_dim = q.shape
	_, seq_len, num_heads_kv, k_head_dim = k.shape
	assert k.shape == v.shape
	assert head_dim == k_head_dim, f"Head dims mismatch: q={head_dim}, k={k_head_dim}"
	assert num_heads_q % num_heads_kv == 0, (
		f"num_heads_q={num_heads_q} not divisible by num_heads_kv={num_heads_kv}"
	)
	num_groups = num_heads_q // num_heads_kv
	q = q.reshape(batch_size, num_heads_kv, num_groups, head_dim)
	k = jnp.swapaxes(k, 1, 2)
	v = jnp.swapaxes(v, 1, 2)

	logits = jnp.einsum("bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32))

	rages = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
	mask = (starts.reshape(-1, 1) <= rages) < (rages < lengths.reshape(-1, 1))
	logits = logits + jnp.where(mask[:, None, None, :], 0.0, mask_value)
	logits_max = logits.max(axis=-1)
	logits_shifted = logits - logits_max[..., None]
	unnormalized = jnp.exp(logits_shifted)
	denominator = unnormalized.sum(axis=-1)
	safe_denominator = jnp.maximum(denominator, jnp.finfo(denominator.dtype).tiny)
	o_unscaled = jnp.einsum("bhgt,bhtd->bhgd", unnormalized.astype(v.dtype), v)
	o = o_unscaled / safe_denominator[..., None]
	o = o.reshape(batch_size, num_heads_q, head_dim)
	o = jnp.expand_dims(o, axis=1)

	logits_max = logits_max.reshape(batch_size, num_heads_q)
	logits_max = jnp.expand_dims(logits_max, axis=(1, 3))

	denominator = denominator.reshape(batch_size, num_heads_q)
	denominator = jnp.expand_dims(denominator, axis=(1, 3))

	return o, logits_max, denominator


def ragged_flash_attention_kernel(
	lengths_ref,
	starts_ref,
	q_ref,
	k_ref,
	v_ref,
	o_ref,
	m_ref,
	l_ref,
	*,
	block_size: int,
	mask_value: float,
):
	"""Pallas kernel implementing Flash Attention for ragged sequences.

	This kernel performs the core Flash Attention computation block by block,
	handling variable sequence lengths specified by `lengths_ref`. It updates
	the output (`o_ref`), maximum logit (`m_ref`), and softmax denominator (`l_ref`)
	incrementally.

	Args:
	    lengths_ref: Reference to the lengths tensor [batch_size].
	    q_ref: Reference to the query tensor block.
	    k_ref: Reference to the key tensor block.
	    v_ref: Reference to the value tensor block.
	    o_ref: Reference to the output tensor block (accumulator).
	    m_ref: Reference to the maximum logit tensor block (accumulator).
	    l_ref: Reference to the softmax denominator tensor block (accumulator).
	    block_size: The size of the blocks to process along the sequence length dimension.
	    mask_value: The value used for masking padding tokens.
	"""
	b, i = pl.program_id(0), pl.program_id(1)

	@pl.when(i == 0)
	def init():
		m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
		l_ref[...] = jnp.zeros_like(l_ref)
		o_ref[...] = jnp.zeros_like(o_ref)

	length = lengths_ref[b]
	start = starts_ref[b]
	idex = i * block_size

	@pl.when((start <= idex) & (idex < length))
	def run():
		q = q_ref[...].astype(jnp.float32)
		k = k_ref[...].astype(jnp.float32)
		v = v_ref[...].astype(jnp.float32)
		m_prev, l_prev = m_ref[...], l_ref[...]

		qk = lax.dot_general(
			q,
			k,
			(((1,), (1,)), ((), ())),
			preferred_element_type=jnp.float32,
		)

		offset = i * block_size + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
		
		mask = (start <= offset) & (offset < length)
		qk = qk + jnp.where(mask, 0.0, mask_value)
		m_curr = qk.max(axis=-1)

		s_curr = jnp.exp(qk - m_curr[..., None])
		l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
		o_curr_times_l_curr = jnp.dot(s_curr, v)

		m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
		m_next = jnp.maximum(m_prev, m_curr)
		alpha = jnp.exp(m_prev - m_next)
		beta = jnp.exp(m_curr - m_next)
		l_next = alpha * l_prev + beta * l_curr
		l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

		m_ref[...], l_ref[...] = m_next, l_next_safe
		o_ref[...] = (
			(l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
		).astype(o_ref.dtype)


def ragged_mqa(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	lengths: jax.Array,
	starts: jax.Array,
	*,
	block_size: int = 256,
	mask_value: float = DEFAULT_MASK_VALUE,
	cost_estimate: pl.CostEstimate | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Ragged multi query attention.

	Args:
	  q: A [batch_size, 1, head_dim] jax.Array.
	  k: A [batch_size, seq_len, head_dim] jax.Array.
	  v: A [batch_size, seq_len, head_dim] jax.Array.
	  lengths: A i32[batch_size] jax.Array.
	  starts: A i32[batch_size] jax.Array.
	  mask_value: The value used for padding in attention. By default it is a very
	    negative floating point number.
	  cost_estimate: A Pallas TPU cost estimate based on a reference implementation

	Returns:
	  The output of attention([batch_size, num_heads, head_dim]), along with the
	  max logit ([batch_size, num_heads, 1]) and softmax denominator ([batch_size,
	  num_heads, 1]).
	"""
	batch_size, num_heads, head_dim = q.shape
	assert lengths.shape == (batch_size,)
	assert starts.shape == (batch_size,)
	assert lengths.dtype == jnp.int32
	assert starts.dtype == jnp.int32
	seq_len = k.shape[1]

	def compute_ragged_block_indices(
		b,
		i,
		lengths_ref,
		starts_ref,
	):
		length = lengths_ref[b]
		start = starts_ref[b]
		idex = i * block_size
		not_done = (start <= idex) & (idex < length)
		am_last_batch = b == batch_size - 1
		last_good_block = lax.div(length, block_size) - 1
		b_next = jnp.where(not_done, b, jnp.where(am_last_batch, b, b + 1))
		i_next = jnp.where(not_done, i, jnp.where(am_last_batch, last_good_block, 0))
		return b_next, i_next, 0

	out, m, l = pl.pallas_call(  # noqa
		functools.partial(
			ragged_flash_attention_kernel,
			block_size=block_size,
			mask_value=mask_value,
		),
		grid_spec=pltpu.PrefetchScalarGridSpec(
			num_scalar_prefetch=2,
			in_specs=[
				pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _, __: (b, 0, 0)),
				pl.BlockSpec((None, block_size, head_dim), compute_ragged_block_indices),
				pl.BlockSpec((None, block_size, head_dim), compute_ragged_block_indices),
			],
			out_specs=[
				pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _, __: (b, 0, 0)),
				pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _, __: (b, 0, 0)),
				pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _, __: (b, 0, 0)),
			],
			grid=(batch_size, seq_len // block_size),
		),
		compiler_params={"mosaic": {"dimension_semantics": ("parallel", "arbitrary")}},
		out_shape=[
			jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
			jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
			jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
		],
		cost_estimate=cost_estimate,
	)(lengths, starts, q, k, v)
	return out, m[..., 0], l[..., 0]


@functools.partial(
	jax.jit,
	static_argnames=["block_size", "mask_value"],
)
def ragged_mha(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	lengths: jax.Array,
	starts: jax.Array,
	*,
	block_size: int = 256,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Ragged multi head attention.

	Args:
	  q: A [batch_size, 1, num_heads, head_dim] jax.Array.
	  k: A [batch_size, seq_len, num_heads, head_dim] jax.Array.
	  v: A [batch_size, seq_len, num_heads, head_dim] jax.Array.
	  lengths: A i32[batch_size] jax.Array.
	  block_size: Value defining the Pallas block length in the seq_len dimension
	  mask_value: The value used for padding in attention. By default it is a very
	    negative floating point number.

	Returns:
	  The output of attention([batch_size, num_heads, head_dim]), along with the
	  max logit ([batch_size, num_heads, 1]) and softmax denominator ([batch_size,
	  num_heads, 1]).
	"""
	shape_dtype = (query, key, value, lengths)
	cost_estimate = get_mha_cost_estimate(shape_dtype)

	query = jnp.swapaxes(query, 1, 2)
	key = jnp.swapaxes(key, 1, 2)
	value = jnp.swapaxes(value, 1, 2)
	o, m, l = jax.vmap(  # noqa
		functools.partial(
			ragged_mqa,
			block_size=block_size,
			mask_value=mask_value,
			cost_estimate=cost_estimate,
		),
		in_axes=(1, 1, 1, None, None),
		out_axes=2,
	)(query, key, value, lengths, starts)
	m = jnp.expand_dims(m, axis=-1)
	l = jnp.expand_dims(l, axis=-1)  # noqa
	o = o * l
	return o, m, l


@functools.partial(
	jax.jit,
	static_argnames=["block_size", "mask_value"],
)
def ragged_gqa(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	lengths: jax.Array,
	starts: jax.Array,
	*,
	block_size: int = 256,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Ragged group query attention.

	Args:
	  q: A [batch_size, num_heads_q, head_dim] jax.Array.
	  k: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
	  v: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
	  lengths: A i32[batch_size] jax.Array.
	  block_size: Value defining the Pallas block length in the seq_len dimension
	  mask_value: The value used for padding in attention. By default it is a very
	    negative floating point number.

	Returns:
	  The output of attention([batch_size, num_heads, head_dim]), along with the
	  max logit ([batch_size, num_heads, 1]) and softmax denominator ([batch_size,
	  num_heads, 1]).
	"""
	shape_dtype = (query, key, value, lengths)
	cost_estimate = get_mha_cost_estimate(shape_dtype)

	batch_size, _, num_heads_q, head_dim = query.shape
	_, _, num_heads_kv, _ = key.shape

	query = query.reshape(batch_size, num_heads_kv, num_heads_q // num_heads_kv, head_dim)
	key = jnp.swapaxes(key, 1, 2)
	value = jnp.swapaxes(value, 1, 2)

	o, m, l = jax.vmap(  # noqa
		functools.partial(
			ragged_mqa,
			block_size=block_size,
			mask_value=mask_value,
			cost_estimate=cost_estimate,
		),
		in_axes=(1, 1, 1, None, None),
		out_axes=1,
	)(query, key, value, lengths, starts)

	m = jnp.reshape(m, (batch_size, 1, num_heads_q, 1))
	l = jnp.reshape(l, (batch_size, 1, num_heads_q, 1))  # noqa
	o = jnp.reshape(o, (batch_size, 1, num_heads_q, head_dim))
	o = o * l
	return o, m, l
