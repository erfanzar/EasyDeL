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

import jax
from eformer.escale import PartitionManager, with_sharding_constraint
from eformer.jaximus import implicit
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.layers.caching.transformer.transformer_cache import TransformerCache
from easydel.layers.quantization.quantizers import EasyQuantizer

from .._utils import ResultTokens, sample_top_p_efficient

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any


@auto_pytree
class GenerationState:
	"""Holds the mutable state required for iterative token generation.

	This state is passed between consecutive `prefill` and `decode` steps,
	carrying information like the KV cache, the last generated tokens, and
	current sequence positions. It's decorated with `@auto_pytree` to allow
	it to be seamlessly used within JAX transformations like `jax.jit`.

	Attributes:
	    logits: The logits output from the model for the last generated token(s)
	            in the batch. Shape: (batch_size, vocab_size).
	    cache: The key-value cache (e.g., TransformerCache) holding past attention
	           states. This is typically updated in-place during generation.
	    index: The current generation index (position) within the sequence for
	           each item in the batch. Shape: (batch_size, 1).
	    tokens: The last generated token IDs for each sequence in the batch.
	            Shape: (batch_size, 1).
	    valids: A boolean array indicating valid positions in the input sequences,
	            used for attention masking. Shape: (batch_size, max_length).
	    next_position_ids: The position IDs to be used for the *next* generation
	                       step for each sequence. Shape: (batch_size, 1).
	    generated_tokens: A counter for the number of tokens generated so far for
	                      each sequence in the batch. Shape: (batch_size, 1).
	"""

	logits: jax.Array
	cache: TransformerCache
	index: jax.Array
	tokens: jax.Array
	valids: jax.Array
	next_position_ids: jax.Array
	temperature: jax.Array
	top_p: jax.Array
	generated_tokens: jax.Array


@partial(jax.jit, donate_argnums=(0, 1), static_argnums=(3,))
def continuous_bulk_insert(
	prefix: GenerationState,
	decode_state: GenerationState,
	slots: list[int],
	quantizer: EasyQuantizer,
	partition_manager: PartitionManager,
) -> GenerationState:
	def update_idx1d(x, y, s):
		return jax.lax.dynamic_update_slice(x, y, (s,))

	def update_idx2d(x, y, s):
		sharding = getattr(x, "sharding", PartitionSpec())
		return with_sharding_constraint(
			jax.lax.dynamic_update_slice(x, y, (s, 0)),
			sharding,
		)

	@implicit
	def _cache(mx_cache, prefix, slot):
		return mx_cache.cache.insert(
			prefix.cache,
			quantizer=quantizer,
			slot=slot,
			partition_manager=partition_manager,
		)

	for slot in slots:
		logits = update_idx2d(decode_state.logits, prefix.logits, slot)
		cache = _cache(decode_state, prefix, slot)
		index = update_idx2d(decode_state.index, prefix.index, slot)
		tokens = update_idx2d(decode_state.tokens, prefix.tokens, slot)
		valids = update_idx2d(decode_state.valids, prefix.valids, slot)
		pos = update_idx2d(decode_state.next_position_ids, prefix.next_position_ids, slot)
		gent = update_idx2d(decode_state.generated_tokens, prefix.generated_tokens, slot)
		top_p = update_idx1d(decode_state.top_p, prefix.top_p, slot)
		temperature = update_idx1d(decode_state.temperature, prefix.temperature, slot)

	return GenerationState(
		logits=logits,
		cache=cache,
		index=index,
		tokens=tokens,
		valids=valids,
		temperature=temperature,
		top_p=top_p,
		next_position_ids=pos,
		generated_tokens=gent,
	)


def continuous_insert(
	prefix: GenerationState,
	decode_state: GenerationState,
	slot: int,
	quantizer: EasyQuantizer,
	partition_manager: PartitionManager,
) -> GenerationState:
	def update_idx1d(x, y):
		sharding = getattr(x, "sharding", PartitionSpec())
		return with_sharding_constraint(
			jax.lax.dynamic_update_slice(x, y, (slot,)),
			sharding,
		)

	def update_idx2d(x, y):
		sharding = getattr(x, "sharding", PartitionSpec())
		return with_sharding_constraint(
			jax.lax.dynamic_update_slice(x, y, (slot, 0)),
			sharding,
		)

	@implicit
	def _cache(mx_cache, prefix):
		return mx_cache.cache.insert(
			prefix.cache,
			quantizer=quantizer,
			slot=slot,
			partition_manager=partition_manager,
		)

	return GenerationState(
		logits=update_idx2d(decode_state.logits, prefix.logits),
		cache=_cache(decode_state, prefix),
		index=update_idx2d(decode_state.index, prefix.index),
		tokens=update_idx2d(decode_state.tokens, prefix.tokens),
		valids=update_idx2d(decode_state.valids, prefix.valids),
		top_p=update_idx1d(decode_state.top_p, prefix.top_p),
		temperature=update_idx1d(decode_state.temperature, prefix.temperature),
		next_position_ids=update_idx2d(
			decode_state.next_position_ids,
			prefix.next_position_ids,
		),
		generated_tokens=update_idx2d(
			decode_state.generated_tokens,
			prefix.generated_tokens,
		),
	)


def continuous_prefill(
	graphdef: nn.GraphDef,
	graphstate: nn.GraphState,
	graphothers: nn.GraphState,
	tokens: jax.Array,
	valids: jax.Array,
	true_length: int,
	pad_token_id: int,
	temperature: jax.Array,
	top_p: jax.Array,
	max_length: int,
	samples_per_slot: int,
	rngs: jax.random.PRNGKey,
) -> tuple[GenerationState, ResultTokens]:
	batch_size, sequence_length = tokens.shape
	if valids.shape[-1] != max_length:
		valids = jax.lax.dynamic_update_slice(
			jnp.ones((batch_size, max_length), "b1"),
			valids.astype("b1"),
			(0, 0),
		).astype("b1")
	positions = (valids.cumsum(axis=-1) - 1)[:, :sequence_length]

	@implicit
	def _forward(
		gdef,
		gstate,
		gother,
		input_ids,
		attention_mask,
		position_ids,
	):
		model: EasyDeLBaseModule = nn.merge(gdef, gstate, gother)
		starts = jnp.array([input_ids.shape[-1] - true_length])
		past_key_values = model.init_cache(
			batch_size=batch_size,
			max_length=max_length,
			starts=starts,
		)
		with model.mesh:
			return model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
			)

	outputs = _forward(
		graphdef,
		graphstate,
		graphothers,
		tokens,
		valids,
		positions,
	)
	kv_cache = outputs.past_key_values
	logits = outputs.logits[:, -1]
	next_token = sample_top_p_efficient(
		logits,
		top_p.astype(logits.dtype),
		temperature.astype(logits.dtype),
		rngs,
	)[:, None]
	validity = jnp.ones_like(next_token, dtype="b1")
	lengths = jnp.full((batch_size, 1), 0, dtype="i4")

	result = ResultTokens(
		data=jnp.concatenate([next_token, validity, lengths], axis=1),
		tokens_idx=(0, 1),
		valid_idx=(1, 2),
		length_idx=(2, 3),
		samples_per_slot=samples_per_slot,
	)
	generation_state = GenerationState(
		logits=logits,
		cache=kv_cache,
		index=jnp.array((sequence_length,)).reshape(1, 1) + 1,
		tokens=next_token,
		valids=valids,
		temperature=temperature,
		top_p=top_p,
		next_position_ids=positions[:, -1:] + 1,
		generated_tokens=jnp.zeros((batch_size, 1), dtype=jnp.int32),
	)
	return generation_state, result


def continuous_decode(
	graphdef: nn.GraphDef,
	graphstate: nn.GraphState,
	graphothers: nn.GraphState,
	state: GenerationState,
	samples_per_slot: int,
	rngs: jax.random.PRNGKey,
):
	@implicit
	def _forward(gdef, gstate, gothers, state):
		model: EasyDeLBaseModule = nn.merge(gdef, gstate, gothers)
		with model.mesh:
			return model(
				input_ids=state.tokens,
				attention_mask=state.valids,
				position_ids=state.next_position_ids,
				past_key_values=state.cache,
			)

	outputs = _forward(
		graphdef,
		graphstate,
		graphothers,
		state,
	)
	batch_size = state.tokens.shape[0]
	kv_cache = outputs.past_key_values
	logits = outputs.logits[:, -1]

	next_token = sample_top_p_efficient(
		logits,
		state.top_p.astype(logits.dtype),
		state.temperature.astype(logits.dtype),
		rngs,
	)[:, None]
	lengths = jnp.full(
		(batch_size, 1),
		state.generated_tokens[:, -1:] + 1,
		dtype="i4",
	)
	validity = jnp.ones_like(next_token, dtype="b1")
	result = ResultTokens(
		data=jnp.concatenate([next_token, validity, lengths], axis=1),
		tokens_idx=(0, 1),
		valid_idx=(1, 2),
		length_idx=(2, 3),
		samples_per_slot=samples_per_slot,
	)
	next_generation_state = GenerationState(
		logits=logits,
		cache=kv_cache,
		index=state.index + 1,
		tokens=next_token,
		valids=state.valids,
		temperature=state.temperature,
		top_p=state.top_p,
		next_position_ids=state.next_position_ids + 1,
		generated_tokens=state.generated_tokens + 1,
	)
	return next_generation_state, result
