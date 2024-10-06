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

import math
from functools import partial
from typing import Optional, Union

import jax
from chex import Array
from fjformer import with_sharding_constraint
from flax.linen.dtypes import promote_dtype
from jax import lax, random
from jax import numpy as jnp
from jax.sharding import PartitionSpec


@partial(
	jax.jit,
	static_argnames=[
		"deterministic",
		"dropout_rng",
		"shard_attention_computation",
		"dtype",
		"precision",
		"attention_dropout",
	],
)
def vanilla_attention(
	query_states: Array,
	key_states: Array,
	value_states: Array,
	bias: Optional[Array] = None,
	deterministic: bool = False,
	dropout_rng: Optional[random.PRNGKey] = None,
	shard_attention_computation: bool = True,
	dtype: jnp.dtype = jnp.float32,
	precision: Optional[jax.lax.Precision] = None,
	attention_dropout: float = 0.0,
):
	query_states, key_states, value_states = promote_dtype(
		query_states, key_states, value_states, dtype=dtype
	)
	sp, tp = "sp", "tp"
	if query_states.shape[1] == 1:
		sp, tp = None, "sp"
	elif query_states.shape[1] != key_states.shape[1]:
		sp, tp = [None] * 2
	if shard_attention_computation:
		query_states = with_sharding_constraint(
			query_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)
		key_states = with_sharding_constraint(
			key_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)
		value_states = with_sharding_constraint(
			value_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)

	depth = query_states.shape[-1]
	query_states = query_states / jnp.sqrt(depth).astype(dtype)
	attention_weight = jnp.einsum(
		"...qhd,...khd->...hqk", query_states, key_states, precision=precision
	)
	if shard_attention_computation:
		attention_weight = with_sharding_constraint(
			attention_weight, PartitionSpec(("dp", "fsdp"), None, sp, None)
		)
		if bias is not None:
			bias = with_sharding_constraint(
				bias, PartitionSpec(("dp", "fsdp"), None, sp, None)
			)
	if bias is not None:
		attention_weight = attention_weight + bias
	attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
	if not deterministic and attention_dropout > 0.0:
		keep_prob = 1.0 - attention_dropout
		dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
		keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

		multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
		attention_weight = attention_weight * multiplier
	attention = jnp.einsum(
		"...hqk,...khd->...qhd", attention_weight, value_states, precision=precision
	)
	if shard_attention_computation:
		attention = with_sharding_constraint(
			attention, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)
	return attention, attention_weight


def _chunk_bias(bias, axis_name, query_length, key_length):
	index = lax.axis_index(axis_name)
	return lax.dynamic_slice(
		bias,
		(0, 0, index * query_length, index * key_length),
		(bias.shape[0], 1, query_length, key_length),
	)


def softmax_grad(X, output):  # output is the result of the softmax function
	softmax_output = output  # Assuming output is provided from the softmax function
	diag_softmax = jnp.diag(softmax_output)
	outer_softmax = softmax_output[:, None] * softmax_output[None, :]
	grad = diag_softmax - outer_softmax
	return grad


def shard_vanilla_attention(
	query_states: Array,
	key_states: Array,
	value_states: Array,
	bias: Optional[Array] = None,
	deterministic: bool = True,
	dropout_rng: Optional[random.PRNGKey] = None,
	dtype: jnp.dtype = jnp.float32,
	precision: Optional[jax.lax.Precision] = None,
	attention_dropout: float = 0.0,
):
	batch_size, query_length, num_heads, dim = query_states.shape
	key_length = key_states.shape[1]
	query_states, key_states, value_states = promote_dtype(
		query_states, key_states, value_states, dtype=dtype
	)

	depth = query_states.shape[-1]
	query_states = query_states / jnp.sqrt(depth).astype(dtype)
	attention_weight = jnp.einsum(
		"...qhd,...khd->...hqk", query_states, key_states, precision=precision
	)
	axis_size = lax.psum(1, "sp")
	if bias is not None:
		bias = lax.dynamic_slice_in_dim(
			lax.dynamic_slice_in_dim(
				bias,
				lax.axis_index("sp") % axis_size * query_length,
				query_length,
				axis=-2,
			),
			lax.axis_index("sp") % axis_size * key_length,
			key_length,
			axis=-1,
		)
		attention_weight = jnp.add(attention_weight, bias)

	attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
	if not deterministic and attention_dropout > 0.0:
		keep_prob = 1.0 - attention_dropout
		dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
		keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

		multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
		attention_weight = attention_weight * multiplier
	attention = jnp.einsum(
		"...hqk,...khd->...qhd", attention_weight, value_states, precision=precision
	)

	return attention


def attention_production(
	query_states: jax.Array,
	key_states: jax.Array,
	value_states: jax.Array,
	attention_bias: jax.Array | None = None,
	deterministic: bool = True,
	dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
	dropout_rate: float = 0.0,
):
	batch, q_sequence_length, q_num_head, head_dim = query_states.shape
	_, kv_sequence_length, kv_num_head, _ = key_states.shape
	assert q_num_head % kv_num_head == 0, (
		f"`query_states` {q_num_head} must be a multiple of `key_states` "
		f"and `value_states` heads {kv_num_head}"
	)
	query_states = jnp.reshape(
		query_states,
		(batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim),
	)
	attention_score = jnp.einsum(
		"...thHd,...Thd->...hHtT", query_states, key_states
	).astype(jnp.float32)
	attention_score *= 1 / math.sqrt(head_dim)
	max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
	attention_score = max_attention_value * jnp.tanh(
		attention_score / max_attention_value
	)
	attention_score = attention_score + attention_bias[:, :, None, :, :]
	attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
	if not deterministic and dropout_rate > 0.0:
		keep_prob = 1.0 - dropout_rate
		dropout_shape = (
			tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
		)
		keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
		multiplier = keep.astype(query_states.dtype) / jnp.asarray(
			keep_prob, dtype=query_states.dtype
		)
		attention_weights = attention_weights * multiplier

	attention = jnp.einsum(
		"...hHtT,...Thd->...thHd", attention_weights, value_states
	).reshape(batch, q_sequence_length, q_num_head, head_dim)
	return attention


def static_sharded_attention_production(
	query_states: jax.Array,
	key_states: jax.Array,
	value_states: jax.Array,
	attention_bias: jax.Array | None = None,
	deterministic: bool = True,
	dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
	dropout_rate: float = 0.0,
):
	assert (
		key_states.shape[1] == value_states.shape[1]
	), "miss match on key_states and value_states sequence length"
	is_generating = (
		query_states.shape[1] == 1 or query_states.shape[1] != key_states.shape[1]
	)
	sp = None if is_generating else "sp"
	tp = "sp" if is_generating else "tp"
	query_states = with_sharding_constraint(
		query_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
	)
	key_states = with_sharding_constraint(
		key_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
	)
	value_states = with_sharding_constraint(
		value_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
	)

	batch, q_sequence_length, q_num_head, head_dim = query_states.shape
	_, kv_sequence_length, kv_num_head, _ = key_states.shape

	assert q_num_head % kv_num_head == 0, (
		f"`query_states` {q_num_head} must be a multiple of"
		f" `key_states` and `value_states` heads {kv_num_head}"
	)

	query_states = jnp.reshape(
		query_states,
		(batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim),
	)

	query_states = with_sharding_constraint(
		query_states, PartitionSpec(("dp", "fsdp"), sp, tp, None, None)
	)

	attention_score = jnp.einsum(
		"...thHd,...Thd->...hHtT", query_states, key_states
	).astype(jnp.float32)

	attention_score *= 1 / math.sqrt(head_dim)

	max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
	attention_score = max_attention_value * jnp.tanh(
		attention_score / max_attention_value
	)
	attention_score = attention_score + attention_bias[:, :, None, :, :]

	attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
	if not deterministic and dropout_rate > 0.0:
		keep_prob = 1.0 - dropout_rate
		dropout_shape = (
			tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
		)
		keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
		multiplier = keep.astype(query_states.dtype) / jnp.asarray(
			keep_prob, dtype=query_states.dtype
		)
		attention_weights = attention_weights * multiplier

	attention = jnp.einsum(
		"...hHtT,...Thd->...thHd", attention_weights, value_states
	).reshape(batch, q_sequence_length, q_num_head, head_dim)

	attention = with_sharding_constraint(
		attention,
		PartitionSpec(
			("dp", "fsdp"),
			sp,
			tp,
			None,
		),
	)

	return attention


def static_sharded_dot_product_attention(
	query_states: Array,
	key_states: Array,
	value_states: Array,
	bias: Optional[Array] = None,
	mask: Optional[Array] = None,
	broadcast_dropout: bool = True,
	dropout_rng: Optional[jax.random.PRNGKey] = None,
	dropout_rate: float = 0.0,
	deterministic: bool = False,
	dtype: Optional[jnp.dtype] = jnp.float32,
	precision: Optional[Union[str, lax.Precision]] = None,
	shard_attention_computation: bool = True,
):
	query_states, key_states, value_states = promote_dtype(
		query_states, key_states, value_states, dtype=dtype
	)

	if query_states.shape[1] == 1:
		sp = None
		tp = "sp"
	elif query_states.shape[1] != key_states.shape[1]:
		sp = None
		tp = None
	else:
		sp = "sp"
		tp = "tp"

	if shard_attention_computation:
		query_states = with_sharding_constraint(
			query_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)

		key_states = with_sharding_constraint(
			key_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)

		value_states = with_sharding_constraint(
			value_states, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)

	depth = query_states.shape[-1]
	query_states = query_states / jnp.sqrt(depth).astype(dtype)
	attention_weight = jnp.einsum(
		"...qhd,...khd->...hqk", query_states, key_states, precision=precision
	)
	if shard_attention_computation:
		attention_weight = with_sharding_constraint(
			attention_weight, PartitionSpec(("dp", "fsdp"), None, sp, None)
		)
		if bias is not None:
			bias = with_sharding_constraint(
				bias, PartitionSpec(("dp", "fsdp"), None, sp, None)
			)
	if bias is not None:
		attention_weight = attention_weight + bias
	if mask is not None:
		big_neg = jnp.finfo(dtype).min
		attention_weight = jnp.where(mask, attention_weight, big_neg)
	attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
	if not deterministic and dropout_rate > 0.0:
		keep_prob = 1.0 - dropout_rate
		if broadcast_dropout:
			dropout_shape = (
				tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
			)
			keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
		else:
			keep = random.bernoulli(dropout_rng, keep_prob, attention_weight.shape)  # type: ignore
		multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
		attention_weight = attention_weight * multiplier
	attention = jnp.einsum(
		"...hqk,...khd->...qhd", attention_weight, value_states, precision=precision
	)
	if shard_attention_computation:
		attention = with_sharding_constraint(
			attention, PartitionSpec(("dp", "fsdp"), sp, tp, None)
		)
	return attention
