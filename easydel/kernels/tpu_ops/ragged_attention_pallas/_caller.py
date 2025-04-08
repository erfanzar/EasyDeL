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

import math
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from ._forward_pallas import ragged_decode_kernel_fwd, NUM_LANES


def ragged_decode(
	query: jax.Array,  # [batch_size, num_heads, head_dim]
	key: jax.Array,  # [batch_size, kv_seq_len, head_dim]
	value: jax.Array,  # [batch_size, kv_seq_len, head_dim]
	starts: jax.Array | None = None,  # [batch_size] - Start indices for each sequence
	lengths: jax.Array | None = None,  # [batch_size] - Length of each sequence
	qk_prev: jax.Array
	| None = None,  # [batch_size, num_heads, kv_seq_len] - Precomputed attention scores
	block_bs: int = 4,  # Block size for batch dimension
	block_kv: int = 256,  # Block size for key/value sequence dimension
	scale: float | None = None,  # Scaling factor for attention scores
	interpret: bool = False,  # Whether to run in interpret mode for debugging
):
	"""
	Pallas kernel for ragged batched attention decoding.

	This function implements attention decoding for sequences of varying lengths in a batched manner.
	It can incorporate precomputed attention scores for efficiency.

	Args:
	    query: Query tensor of shape [batch_size, num_heads, head_dim].
	    key: Key tensor of shape [batch_size, kv_seq_len, head_dim].
	    value: Value tensor of shape [batch_size, kv_seq_len, head_dim].
	    starts: Optional start indices for each sequence in the batch. Shape [batch_size].
	    lengths: Optional lengths of each sequence in the batch. Shape [batch_size].
	    qk_prev: Optional previous query-key attention scores. Shape [batch_size, num_heads, kv_seq_len].
	    block_bs: Block size for batch dimension. Defaults to 4.
	    block_kv: Block size for key/value sequence length dimension. Defaults to 256.
	    scale: Optional scaling factor for attention scores. If None, defaults to sqrt(head_dim).
	    interpret: Whether to run the kernel in interpret mode for debugging. Defaults to False.

	Returns:
	    Attention output tensor of shape [batch_size, num_heads, head_dim].
	"""
	scale = math.sqrt(query.shape[-1]) if scale is None else scale
	batch_size_q, num_heads, head_dim_q = query.shape
	batch_size_k, kv_seq_len_k, head_dim_k = key.shape
	assert batch_size_q == batch_size_k and head_dim_q == head_dim_k
	batch_size, kv_seq_len = batch_size_q, kv_seq_len_k
	batch_size_v, kv_seq_len_v, head_dim_v = value.shape
	assert batch_size == batch_size_v and kv_seq_len == kv_seq_len_v

	block_bs = min(batch_size, block_bs)
	assert batch_size % block_bs == 0

	if starts is None:
		starts = jnp.zeros((batch_size,), dtype=jnp.int32)
	if lengths is None:
		lengths = kv_seq_len * jnp.ones((batch_size,), dtype=jnp.int32)

	assert starts.ndim == 1 and starts.size == batch_size
	assert lengths.ndim == 1 and lengths.size == batch_size
	block_kv = min(kv_seq_len, block_kv)
	assert kv_seq_len % block_kv == 0

	chunked_starts = jnp.min(starts.reshape((-1, block_bs)), axis=-1)
	chunked_lengths = jnp.max(lengths.reshape((-1, block_bs)), axis=-1)

	def kv_prefetch_map(
		batch_idx,
		block_idx,
		starts_ref,
		lengths_ref,
		chunked_starts_ref,
		chunked_lengths_ref,
	):
		"""
		Map function for prefetching key and value tensors.

		Determines the appropriate batch and block indices for prefetching based on
		sequence start and length information.
		"""
		del starts_ref, lengths_ref
		seq_start, seq_length = (
			chunked_starts_ref[batch_idx],
			chunked_lengths_ref[batch_idx],
		)
		seq_idx = block_idx * block_kv
		last_batch, seq_done = (
			batch_idx == (batch_size // block_bs) - 1,
			seq_idx > seq_length,
		)
		start_next = chunked_starts_ref[batch_idx + (~last_batch)]
		first_start_i, next_start_i = seq_start // block_kv, start_next // block_kv
		batch_idx = jnp.where(seq_done & (~last_batch), batch_idx + 1, batch_idx)
		block_idx = jnp.where(
			seq_done,
			jnp.where(last_batch, block_idx, next_start_i),
			jnp.maximum(first_start_i, block_idx),
		)
		block_idx = jnp.where(
			last_batch & seq_done, pl.cdiv(seq_length, block_kv) - 1, block_idx
		)
		return batch_idx, block_idx, 0

	in_specs = []
	in_specs += [
		pl.BlockSpec((block_bs, num_heads, query.shape[-1]), lambda b, i, *_: (b, 0, 0))
	]  # query
	in_specs += [
		pl.BlockSpec((block_bs, block_kv, key.shape[-1]), kv_prefetch_map)
	]  # key
	in_specs += [pl.BlockSpec((block_bs, block_kv, head_dim_v), kv_prefetch_map)]  # value

	if qk_prev is not None:
		qk_prev_prefetch_map = kv_prefetch_map
		in_specs += [pl.BlockSpec((block_bs, num_heads, block_kv), qk_prev_prefetch_map)]
	else:
		in_specs += [None]

	out_shape = jax.ShapeDtypeStruct((batch_size, num_heads, head_dim_v), query.dtype)
	grid_spec = pltpu.PrefetchScalarGridSpec(
		num_scalar_prefetch=4,
		grid=(batch_size // block_bs, kv_seq_len // block_kv),
		in_specs=in_specs,
		out_specs=pl.BlockSpec(
			(block_bs, num_heads, head_dim_v), lambda b, i, *_: (b, 0, 0)
		),
		scratch_shapes=[
			pltpu.VMEM((block_bs, num_heads, head_dim_v), dtype=jnp.float32),
			pltpu.VMEM((block_bs, num_heads, NUM_LANES), dtype=jnp.float32),
			pltpu.VMEM((block_bs, num_heads, NUM_LANES), dtype=jnp.float32),
		],
	)
	kernel = partial(
		ragged_decode_kernel_fwd,
		kv_seq_len=kv_seq_len,
		block_kv=block_kv,
		block_bs=block_bs,
		scale=scale,
	)
	attention_output = pl.pallas_call(
		kernel, grid_spec=grid_spec, out_shape=out_shape, interpret=interpret
	)(starts, lengths, chunked_starts, chunked_lengths, query, key, value, qk_prev)
	return attention_output
