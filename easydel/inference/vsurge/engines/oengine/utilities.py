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

import jax
from flax import nnx as nn
from jax import numpy as jnp

from easydel.layers.caching.paged_attention import (
	ModelInputBatch,
	ModelOutputBatch,
	PagedAttentionCache,
)
from easydel.layers.caching.paged_attention.managers import ModelIOProcessor


def sample_top_p_efficient(
	logits: jax.Array,
	top_p: jax.Array,
	temperature: jax.Array,
	rng: jax.random.PRNGKey,
	top_k_for_computation: int = 50,
) -> jax.Array:
	if logits.ndim == 1:
		logits = jnp.expand_dims(logits, axis=0)
		batch_size = 1
		input_was_1d = True
	else:
		batch_size = logits.shape[0]
		input_was_1d = False

	vocab_size = logits.shape[-1]
	effective_k = min(top_k_for_computation, vocab_size)
	if top_p.ndim == 0:
		top_p = jnp.repeat(top_p, batch_size)
	if temperature.ndim == 0:
		temperature = jnp.repeat(temperature, batch_size)

	safe_temperature = jnp.where(temperature > 1e-6, temperature, 1.0)
	scaled_logits = logits / jnp.expand_dims(safe_temperature, axis=-1)
	top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, k=effective_k)
	top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)
	cumulative_probs_k = jnp.cumsum(top_k_probs, axis=-1)
	keep_mask_k = cumulative_probs_k <= jnp.expand_dims(top_p, axis=-1)
	keep_mask_k = keep_mask_k.at[..., 0].set(True)
	filtered_top_k_logits = jnp.where(keep_mask_k, top_k_logits, -jnp.inf)
	sampled_k_index = jax.random.categorical(rng, filtered_top_k_logits)
	next_token_index = jnp.take_along_axis(
		top_k_indices,
		jnp.expand_dims(sampled_k_index, axis=-1),
		axis=-1,
	).squeeze(-1)

	if input_was_1d:
		next_token_index = next_token_index.squeeze(0)

	return next_token_index


def execute_forward(
	graphdef: nn.GraphDef,
	graphgstate: nn.GraphState,
	graphother: nn.GraphState,
	model_inputs: ModelInputBatch,
	eos_token_ids: jax.Array,
	cache: PagedAttentionCache,
	rngs: jax.random.PRNGKey,
) -> tp.Union[ModelOutputBatch, PagedAttentionCache, jax.random.PRNGKey]:
	"""Executes a single forward pass of the model using paged attention.

	This function handles both the prefill and decode phases of generation.
	It merges the model graph definition and states, performs the forward pass,
	applies sampling filters to the logits, samples the next token, and
	determines if the sequence generation is complete for each sequence in the batch.

	Args:
	    graphdef: The static definition of the model graph (nn.GraphDef).
	    graphgstate: The trainable state of the model graph (nn.GraphState).
	    graphother: Other non-trainable state/variables of the model graph (nn.GraphState).
	    model_inputs: A `ModelInputBatch` containing token IDs, positions,
	        sampling parameters, and attention metadata for the current batch.
	    eos_token_ids: A JAX array containing the end-of-sequence token IDs.
	    cache: The `PagedAttentionCache` holding the current Key-Value cache state.
	    rngs: A JAX PRNG key used for sampling the next token.

	Returns:
	    A tuple containing:
	        - output: A `ModelOutputBatch` with the generated next tokens and
	          their completion status.
	        - cache: The updated `PagedAttentionCache` after the forward pass.
	        - new_rng: A new JAX PRNG key split from the input `rngs` for future use.
	"""
	model = nn.merge(graphdef, graphgstate, graphother)
	attn_meta = model_inputs.attn_meta
	input_ids = model_inputs.input_ids
	positions = model_inputs.positions

	sampling_params = model_inputs.sampling_params

	decode_mode = attn_meta.is_decode_mode()
	prefill_mode = attn_meta.is_prefill_mode()
	mixin_length = None
	if prefill_mode:
		expand_dim = 0 
	else:
		expand_dim = 1
	if not prefill_mode and not decode_mode:
		mixin_length = attn_meta.prefill_position.shape[0]

	input_ids = jnp.expand_dims(input_ids, expand_dim)
	positions = jnp.expand_dims(positions, expand_dim)
	with model.mesh:
		outputs = model(
			input_ids=input_ids,
			position_ids=positions,
			past_key_values=cache,
			cache_metadata=attn_meta,
		)

	logits = outputs.logits.squeeze(expand_dim)
	cache = outputs.past_key_values
	if mixin_length is not None:
		next_token_decode = sample_top_p_efficient(
			logits=logits[mixin_length:],
			top_p=sampling_params.top_p,
			temperature=sampling_params.temperature,
			rng=rngs,
		)
		complete_decode = jnp.isin(next_token_decode, eos_token_ids)
		complete_decode = jnp.logical_or(
			complete_decode,
			jnp.greater_equal(
				model_inputs.positions[mixin_length:],
				sampling_params.max_tokens - 1,
			),
		)
		next_token_prefill = sample_top_p_efficient(
			logits=logits[:mixin_length],
			top_p=jnp.asarray([0.95]),
			temperature=jnp.asarray([0.4]),
			rng=rngs,
		)
		complete_prefill = jnp.isin(next_token_prefill, eos_token_ids)

		next_token = jnp.concatenate([next_token_prefill, next_token_decode])
		complete = jnp.concatenate([complete_prefill, complete_decode])

	else:
		next_token = sample_top_p_efficient(
			logits=logits,
			top_p=sampling_params.top_p,
			temperature=sampling_params.temperature,
			rng=rngs,
		)
		complete = jnp.isin(next_token, eos_token_ids)

		complete = jnp.logical_or(
			complete,
			jnp.greater_equal(model_inputs.positions, sampling_params.max_tokens - 1),
		)
	padded_length = 0 if decode_mode else attn_meta.prefill_position.shape[0]

	if len(attn_meta.decodes_position.shape) != 0 and padded_length != 0:
		next_token = jnp.concatenate(
			[
				next_token.at[attn_meta.prefill_length - 1].get()[None],
				next_token.at[padded_length:].get(),
			]
		)
		complete = jnp.concatenate(
			[
				complete.at[attn_meta.prefill_length - 1].get()[None],
				complete.at[padded_length:].get(),
			]
		)
	elif padded_length != 0:
		next_token = next_token.at[attn_meta.prefill_length - 1].get()[None]
		complete = complete.at[attn_meta.prefill_length - 1].get()[None]

	output = ModelIOProcessor.prepare_model_output(
		next_token=next_token,
		complete=complete,
		attn_meta=attn_meta,
		sampling_params=sampling_params,
	)

	return output, cache, jax.random.split(rngs, 2)[0]


__all__ = ("execute_forward",)
