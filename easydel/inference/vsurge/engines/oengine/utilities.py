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
from .._utils import sample_top_p_efficient


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
