# Copyright 2024 The JAX Authors.
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

# This is a copied-edited version of
# https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention

from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128


@partial(jax.named_call, name="ragged_decode_kernel")
def ragged_decode_kernel_fwd(
	# prefetch scalars:
	start_ref,  # [batch_size] - Start indices for each sequence in the batch
	length_ref,  # [batch_size] - Length of each sequence in the batch
	chunked_start_ref,  # [batch_size // block_batch_size] - Chunked start indices
	chunked_length_ref,  # [batch_size // block_batch_size] - Chunked sequence lengths
	# inputs:
	q_ref,  # [batch_size // block_batch_size, num_heads, head_dim] - Query tensor
	k_ref,  # [batch_size // block_batch_size, block_kv_len, head_dim] - Key tensor
	v_ref,  # [batch_size // block_batch_size, block_kv_len, head_dim] - Value tensor
	qk_prev_ref,  # optional [batch_size // block_batch_size, num_heads, block_kv_len] - Precomputed query-key attention scores
	# outputs:
	o_ref,  # [batch_size // block_batch_size, num_heads, head_dim] - Output attention tensor
	# scratch memory:
	o_scratch_ref,  # [batch_size // block_batch_size, num_heads, head_dim] - Output scratch memory
	l_scratch_ref,  # [batch_size // block_batch_size, num_heads, NUM_LANES] - Log-sum-exp scratch memory
	m_scratch_ref,  # [batch_size // block_batch_size, num_heads, NUM_LANES] - Max value scratch memory
	# parameters:
	kv_seq_len: int,  # Total key/value sequence length
	block_kv: int,  # Block size for key/value sequence dimension
	block_bs: int,  # Block size for batch dimension
	scale: float,  # Scaling factor for attention scores
):
	"""
	Forward pass kernel for ragged batched attention decoding on TPU.

	This kernel implements the core attention mechanism for processing sequences of varying lengths
	in a batched manner. It handles the computation of attention scores, softmax, and weighted
	sum of values with numerical stability considerations.

	The kernel processes data in blocks for efficient TPU utilization.
	"""
	del chunked_start_ref, chunked_length_ref
	mask_value = jnp.finfo(o_scratch_ref.dtype).min
	batch_idx, block_idx = pl.program_id(0), pl.program_id(1)

	@pl.when(block_idx == 0)
	def init():
		"""Initialize scratch memory with appropriate values."""
		m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
		l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
		o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

	def resize(x, new_size_in_dim, axis=-1):
		"""
		Resize the shape of array x to the target size along axis `axis`.

		Args:
			x: Input array to resize
			new_size_in_dim: Target size for the specified dimension
			axis: Axis along which to resize (default: -1)

		Returns:
			Resized array with the specified dimension adjusted to new_size_in_dim
		"""
		if x.shape[axis] > new_size_in_dim:
			assert axis in (-1, x.ndim - 1)
			return x[..., :new_size_in_dim]
		return pltpu.repeat(x, new_size_in_dim // x.shape[axis], axis=axis % x.ndim)

	def loop_fn(batch_in_block, _):
		"""Process a single batch element within the current block."""
		global_batch_idx = block_bs * batch_idx + batch_in_block
		seq_start, seq_length = start_ref[global_batch_idx], length_ref[global_batch_idx]
		block_start, block_end = block_idx * block_kv, (block_idx + 1) * block_kv
		should_compute = (seq_start < seq_length) & (
			(block_start < seq_length) & (block_end >= seq_start)
		)

		@pl.when(should_compute)
		def compute():
			"""Compute attention scores and update output for the current batch element."""
			# compute query-key attention scores
			query, key = q_ref[batch_in_block, ...], k_ref[batch_in_block, ...]
			query_key_scores = jax.lax.dot_general(
				query, key, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
			)
			if qk_prev_ref is not None:
				query_key_scores += qk_prev_ref[batch_in_block, ...]
			query_key_scores *= scale
			position_indices = block_idx * block_kv + jax.lax.broadcasted_iota(
				jnp.int32, query_key_scores.shape, dimension=1
			)
			attention_mask = (position_indices >= seq_start) & (position_indices < seq_length)
			query_key_scores += jnp.where(attention_mask, 0, mask_value)

			# adjust maximum shift value, shift and softmax
			max_prev, log_sum_prev = (
				m_scratch_ref[batch_in_block, ...],
				l_scratch_ref[batch_in_block, ...],
			)
			max_curr = resize(jnp.max(query_key_scores, axis=-1)[:, None], max_prev.shape[-1])
			max_next = jnp.maximum(max_prev, max_curr)
			softmax_scores = jnp.exp(
				query_key_scores - resize(max_next, query_key_scores.shape[-1])
			)
			log_sum_curr = jax.lax.broadcast_in_dim(
				jnp.sum(softmax_scores, axis=-1), log_sum_prev.shape, (0,)
			)

			# compute the weighted sum of values
			value = v_ref[batch_in_block, ...]
			output_curr = jax.lax.dot_general(
				softmax_scores,
				value,
				(((1,), (0,)), ((), ())),
				preferred_element_type=jnp.float32,
			)

			# accumulate the results
			output_prev = o_scratch_ref[batch_in_block, ...]
			max_next = jnp.maximum(max_prev, max_curr)
			alpha = jnp.exp(max_prev - max_next)
			log_sum_next = log_sum_prev * alpha + log_sum_curr
			log_sum_next_safe = log_sum_next
			output_next = resize(alpha, output_prev.shape[-1]) * output_prev + output_curr

			# store scratch values
			m_scratch_ref[batch_in_block, ...] = max_next
			l_scratch_ref[batch_in_block, ...] = log_sum_next_safe
			o_scratch_ref[batch_in_block, ...] = output_next

	jax.lax.fori_loop(0, block_bs, loop_fn, init_val=None)

	@pl.when(block_idx == (kv_seq_len // block_kv) - 1)
	def done():
		"""Finalize the output by normalizing with the log-sum-exp values."""
		log_sum = l_scratch_ref[...]  # noqa
		log_sum_inv = jnp.where(log_sum == 0.0, 1.0, 1.0 / log_sum)
		o_ref[...] = (
			o_scratch_ref[...] * resize(log_sum_inv, o_scratch_ref.shape[-1])
		).astype(o_ref.dtype)
