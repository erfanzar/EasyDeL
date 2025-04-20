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
import numpy as np
from eformer.escale import with_sharding_constraint
from eformer.jaximus import implicit
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.layers.caching.transformer.transformer_cache import TransformerCache
from easydel.layers.quantization.quantizers import EasyQuantizer


if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any


class SlotData(tp.NamedTuple):
	"""Represents the output data for a single inference slot.

	This structure holds the generated tokens, their validity flags, and the
	current sequence length for one specific slot within a batch processed
	by the engine.

	Attributes:
	    tokens: The generated token IDs for the slot (JAX or NumPy array).
	            Shape typically (samples_per_slot, num_speculative_tokens).
	    valid: A boolean array indicating the validity of each generated token
	           (JAX or NumPy array). Shape matches `tokens`.
	    lengths: An array containing the current length(s) of the generated
	             sequence(s) for the slot (JAX or NumPy array). Shape
	             typically (samples_per_slot,).
	"""

	tokens: tp.Union[jax.Array, np.ndarray]
	valid: tp.Union[jax.Array, np.ndarray]
	lengths: tp.Union[jax.Array, np.ndarray]


class ResultTokens(tp.NamedTuple):
	"""Stores the results of a generation step (prefill or decode).

	This structure holds token data, validity flags, and sequence lengths
	concatenated into a single array (`data`) for efficient host transfer.
	Index tuples (`tokens_idx`, `valid_idx`, `length_idx`) specify the slices
	within `data` corresponding to each type of information. This is designed
	to minimize the number of device-to-host transfers.

	Attributes:
	    data: A single JAX or NumPy array containing concatenated token IDs,
	        validity flags, and lengths for the entire batch. Shape typically
	        (batch_size * samples_per_slot, concatenated_data_width).
	    tokens_idx: A tuple (start, end) indicating the column slice for token IDs
	                within the `data` array.
	    valid_idx: A tuple (start, end) indicating the column slice for validity flags
	               within the `data` array.
	    length_idx: A tuple (start, end) indicating the column slice for sequence lengths
	                within the `data` array.
	    samples_per_slot: The number of samples generated per inference slot (e.g., 1).
	                      Used by `get_result_at_slot` to extract data correctly.
	"""

	data: tp.Union[jax.Array, np.ndarray]
	tokens_idx: tp.Tuple[int, int]
	valid_idx: tp.Tuple[int, int]
	length_idx: tp.Tuple[int, int]
	samples_per_slot: int

	def copy_to_host_async(self: "ResultTokens") -> None:
		"""Initiates an asynchronous copy of the `data` array to the host CPU.

		If the data is already a NumPy array, this is a no-op.
		"""
		if isinstance(self.data, np.ndarray):
			return
		self.data.copy_to_host_async()

	def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
		"""Converts the internal `data` array to a NumPy array synchronously.

		Returns:
		    A new ResultTokens instance with the data as a NumPy array.
		"""
		return ResultTokens(
			np.array(self.data),
			self.tokens_idx,
			self.valid_idx,
			self.length_idx,
			self.samples_per_slot,
		)

	def get_result_at_slot(self, slot: int) -> SlotData:
		"""Extracts the generation results for a specific inference slot.

		Args:
		    slot: The index of the inference slot (0-based) for which to retrieve data.

		Returns:
		    A SlotData object containing the tokens, validity, and lengths for the
		    requested slot.

		Note:
		    This method correctly handles potential microbatching by using
		    `samples_per_slot` to calculate the correct indices within the `data` array.
		"""
		start_idx = slot * self.samples_per_slot
		end_idx = (slot + 1) * self.samples_per_slot
		return SlotData(
			tokens=self.data[
				start_idx:end_idx,
				self.tokens_idx[0] : self.tokens_idx[1],
			],
			valid=self.data[
				start_idx:end_idx,
				self.valid_idx[0] : self.valid_idx[1],
			],
			lengths=self.data[
				start_idx:end_idx,
				self.length_idx[0] : self.length_idx[1],
			][:, 0],
		)

	def __str__(self):
		return f"ResultTokens(data={self.data})"


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
	top_k: jax.Array
	generated_tokens: jax.Array


def apply_temprature(logits, temperature):
	return jax.lax.cond(
		temperature != 0.0,
		lambda x, temp: x / temp,
		lambda *x: x[0],
		logits,
		temperature,
	)


def apply_top_k(logits, top_k):
	vocab_size = logits.shape[-1]
	effective_k = jnp.maximum(top_k, 1)
	effective_k = jnp.minimum(effective_k, vocab_size).astype(jnp.int32)

	def _filter_scores(s: jnp.ndarray) -> jnp.ndarray:
		"""Applies the dynamic filtering logic."""
		sorted_scores = jnp.sort(s, axis=-1)[:, ::-1]
		k_index = effective_k - 1
		k_index = jnp.maximum(0, k_index)
		threshold = sorted_scores[:, k_index]
		threshold = threshold[:, None]
		mask = s >= threshold
		return jnp.where(mask, s, -float("inf"))

	def _identity(s: jnp.ndarray) -> jnp.ndarray:
		"""Returns scores unchanged."""
		return s

	return jax.lax.cond(
		(top_k > 0) & (effective_k < vocab_size),
		_filter_scores,
		_identity,
		logits,
	)


def apply_top_p(logits, top_p):
	def _apply(x):
		topk_scores, topk_indices = jax.lax.top_k(x, x.shape[-1])

		mask_scores = jnp.full_like(x, -float("inf"))
		cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
		score_mask = cumulative_probs < top_p
		score_mask = jnp.roll(score_mask, 1)
		score_mask |= score_mask.at[:, 0].set(True)
		score_mask = score_mask.at[:, :1].set(True)
		topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
		x = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]

		return x

	return jax.lax.cond(
		(top_p > 0) & (top_p < 1),
		_apply,
		lambda x: x,
		logits,
	)


@partial(jax.jit, donate_argnums=(0, 1), static_argnums=(3,))
def continuous_bulk_insert(
	prefix: GenerationState,
	decode_state: GenerationState,
	slots: list[int],
	quantizer: EasyQuantizer,
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
		return mx_cache.cache.insert(prefix.cache, quantizer=quantizer, slot=slot)

	for slot in slots:
		logits = update_idx2d(decode_state.logits, prefix.logits, slot)
		cache = _cache(decode_state, prefix, slot)
		index = update_idx2d(decode_state.index, prefix.index, slot)
		tokens = update_idx2d(decode_state.tokens, prefix.tokens, slot)
		valids = update_idx2d(decode_state.valids, prefix.valids, slot)
		pos = update_idx2d(decode_state.next_position_ids, prefix.next_position_ids, slot)
		gent = update_idx2d(decode_state.generated_tokens, prefix.generated_tokens, slot)
		top_p = update_idx1d(decode_state.top_p, prefix.top_p, slot)
		top_k = update_idx1d(decode_state.top_k, prefix.top_k, slot)
		temperature = update_idx1d(decode_state.temperature, prefix.temperature, slot)

	return GenerationState(
		logits=logits,
		cache=cache,
		index=index,
		tokens=tokens,
		valids=valids,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
		next_position_ids=pos,
		generated_tokens=gent,
	)

 
def continuous_insert(
	prefix: GenerationState,
	decode_state: GenerationState,
	slot: int,
	quantizer: EasyQuantizer,
) -> GenerationState:
	def update_idx1d(x, y):
		return jax.lax.dynamic_update_slice(x, y, (slot,))

	def update_idx2d(x, y):
		sharding = getattr(x, "sharding", PartitionSpec())
		return with_sharding_constraint(
			jax.lax.dynamic_update_slice(x, y, (slot, 0)),
			sharding,
		)

	@implicit
	def _cache(mx_cache, prefix):
		return mx_cache.cache.insert(prefix.cache, quantizer=quantizer, slot=slot)

	return GenerationState(
		logits=update_idx2d(decode_state.logits, prefix.logits),
		cache=_cache(decode_state, prefix),
		index=update_idx2d(decode_state.index, prefix.index),
		tokens=update_idx2d(decode_state.tokens, prefix.tokens),
		valids=update_idx2d(decode_state.valids, prefix.valids),
		top_p=update_idx1d(decode_state.top_p, prefix.top_p),
		top_k=update_idx1d(decode_state.top_k, prefix.top_k),
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
	temperature: jax.Array,
	top_p: jax.Array,
	top_k: jax.Array,
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
		past_key_values = model.init_cache(batch_size=batch_size, max_length=max_length)
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
	logits = apply_temprature(logits, temperature[0].astype(logits.dtype))
	logits = apply_top_k(logits, top_k[0])
	logits = apply_top_p(logits, top_p[0].astype(logits.dtype))
	next_token = jax.random.categorical(rngs, logits, axis=-1)[:, None]
	validity = jnp.ones_like(next_token, dtype="b1")
	lengths = jnp.full((batch_size, 1), sequence_length + 1, dtype="i4")

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
		top_k=top_k,
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

	@partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=(0))
	def _apply_filters(logits, top_p, top_k, temperature):
		logits = jnp.expand_dims(logits, 0)
		logits = apply_temprature(logits, temperature.astype(logits.dtype))
		logits = apply_top_k(logits, top_k)
		logits = apply_top_p(logits, top_p.astype(logits.dtype))
		return logits[0]

	logits = _apply_filters(logits, state.top_p, state.top_k, state.temperature)
	next_token = jax.random.categorical(rngs, logits, axis=-1)[:, None]
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
		top_k=state.top_k,
		next_position_ids=state.next_position_ids + 1,
		generated_tokens=state.generated_tokens + 1,
	)
	return next_generation_state, result
