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


import jax
from ._forward_pallas import (
	ragged_mha,
	ragged_gqa,
	DEFAULT_MASK_VALUE,
	reference_mha,
	reference_gqa,
)


def ragged_attention(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	lengths: jax.Array,
	starts: jax.Array,
	*,
	block_size: int = 256,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Dispatches to ragged Multi-Head or Grouped-Query Attention.

	This function acts as a caller for either ragged Multi-Head Attention (MHA)
	or ragged Grouped-Query Attention (GQA) based on the number of query heads
	and key/value heads. It utilizes Pallas kernels optimized for TPUs to handle
	ragged inputs efficiently.

	Args:
	    query: Query tensor with shape [batch_size, num_q_heads, head_dim].
	    key: Key tensor with shape [batch_size, seq_len, num_kv_heads, head_dim].
	    value: Value tensor with shape [batch_size, seq_len, num_kv_heads, head_dim].
	    lengths: Integer tensor with shape [batch_size] indicating the true sequence
	        length for each item in the batch.
	    starts: Integer tensor indicating start indices.
	    block_size: The block size to use for Pallas kernels. Defaults to 256.
	    mask_value: The value to use for masking attention logits. Defaults to
	        a large negative number.

	Returns:
	    A tuple containing:
	        - The attention output tensor.
	        - The maximum logit values.
	        - The softmax denominator values.
	    The exact shapes depend on whether MHA or GQA is called.
	"""
	q_head = query.shape[1]
	kv_head = key.shape[2]
	if q_head == kv_head:
		return ragged_mha(
			query=query,
			key=key,
			value=value,
			lengths=lengths,
			starts=starts,
			block_size=block_size,
			mask_value=mask_value,
		)

	else:
		return ragged_gqa(
			query=query,
			key=key,
			value=value,
			lengths=lengths,
			starts=starts,
			block_size=block_size,
			mask_value=mask_value,
		)


def reference_ragged_attention(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	lengths: jax.Array,
	starts: jax.Array,
	*,
	mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Dispatches to ragged Multi-Head or Grouped-Query Attention.

	This function acts as a caller for either ragged Multi-Head Attention (MHA)
	or ragged Grouped-Query Attention (GQA) based on the number of query heads
	and key/value heads. It utilizes Pallas kernels optimized for TPUs to handle
	ragged inputs efficiently.

	Args:
	    query: Query tensor with shape [batch_size, num_q_heads, head_dim].
	    key: Key tensor with shape [batch_size, seq_len, num_kv_heads, head_dim].
	    value: Value tensor with shape [batch_size, seq_len, num_kv_heads, head_dim].
	    lengths: Integer tensor with shape [batch_size] indicating the true sequence
	        length for each item in the batch.
	    starts: Integer tensor indicating start indices.
	    mask_value: The value to use for masking attention logits. Defaults to
	        a large negative number.

	Returns:
	    A tuple containing:
	        - The attention output tensor.
	        - The maximum logit values.
	        - The softmax denominator values.
	    The exact shapes depend on whether MHA or GQA is called.
	"""
	q_head = query.shape[1]
	kv_head = key.shape[2]
	if q_head == kv_head:
		return reference_mha(
			q=query,
			k=key,
			v=value,
			lengths=lengths,
			starts=starts,
			mask_value=mask_value,
		)

	else:
		return reference_gqa(
			q=query,
			k=key,
			v=value,
			lengths=lengths,
			starts=starts,
			mask_value=mask_value,
		)
