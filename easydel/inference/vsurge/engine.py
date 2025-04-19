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
from eformer.pytree import auto_pytree
import jax
import numpy as np
from flax import nnx as nn
from jax import numpy as jnp
from easydel.layers.caching.transformer.transformer_cache import TransformerCache

from ..utilities import SamplingParams


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
	    sampling_params: An optional `SamplingParams` object containing parameters
	                     controlling the sampling process (e.g., temperature, top_k).
	                     This might be None if sampling parameters are fixed or
	                     handled externally.
	    generated_tokens: A counter for the number of tokens generated so far for
	                      each sequence in the batch. Shape: (batch_size, 1).
	"""

	logits: jax.Array
	cache: TransformerCache
	index: jax.Array
	tokens: jax.Array
	valids: jax.Array
	next_position_ids: jax.Array
	sampling_params: SamplingParams | None
	generated_tokens: jax.Array


class vEngine:
	"""
	Core inference engine for EasyDeL models using NNX graphs.

	This engine manages the model state (split into graph definition, state, and
	other parameters) and provides JIT-compiled functions for the prefill and
	decode steps of autoregressive generation. It handles KV caching and
	sampling.
	"""

	def __init__(
		self,
		model: EasyDeLBaseModule,
		processor,
		max_concurrent_decodes: int | None = None,
		max_concurrent_prefill: int | None = None,
		max_prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		batch_size: int | None = None,
		seed: int = 894,
	):
		"""Initializes the vEngine.

		Args:
		    model: The EasyDeLBaseModule (NNX) to use for inference.
		    processor: The tokenizer/processor associated with the model.
		    max_concurrent_decodes: The maximum number of sequences that can be
		        decoded concurrently. Defaults to the number of available JAX devices.
		    max_concurrent_prefill: The maximum number of prefill requests that can be
		        processed concurrently. Defaults to the number of available JAX devices.
		    max_prefill_lengths: A list of integer bucket lengths to choose from for prefill.
		        Defaults to `DEFAULT_PREFILL_BUCKETS`.
		    max_prefill_length: The maximum allowed length for the initial prompt
		        (prefill phase). Defaults to 4096.
		    max_length: The maximum total sequence length (prompt + generation).
		        Defaults to 8192.
		    batch_size: The batch size for the engine. Defaults to None.
		    seed: The random seed for initializing the PRNG key used in sampling.
		        Defaults to 894.
		"""
		from .utils import DEFAULT_PREFILL_BUCKETS

		self.model = model
		self.graphdef, self.graphstate, self.graphothers = model.split_module()
		self.processor = processor
		self._max_prefill_lengths = max_prefill_lengths or DEFAULT_PREFILL_BUCKETS
		self._max_concurrent_decodes = max_concurrent_decodes or jax.device_count()
		self._max_concurrent_prefill = max_concurrent_prefill or jax.device_count()
		self._max_prefill_length = max_prefill_length or 4096
		self._max_length = max_length or 8192
		self._batch_size = batch_size
		self._prng_key = jax.random.PRNGKey(seed)

	@property
	def samples_per_slot(self) -> int:
		"""Number of samples generated per inference slot.

		This determines how many independent generation results are produced
		for each logical slot managed by the engine. It's often 1, but could
		be higher for techniques like parallel sampling.
		"""
		return 1  # self._max_concurrent_decodes // self._max_concurrent_prefill

	@property
	def prng_key(self) -> jax.random.PRNGKey:
		"""Provides a new PRNG key split from the internal state for sampling.

		Each call to this property consumes the current key and returns a new,
		unique key, ensuring that subsequent sampling operations use different
		randomness.

		Returns:
		    A new JAX PRNGKey.
		"""
		self._prng_key, new_key = jax.random.split(self._prng_key, 2)
		return new_key

	@property
	def max_prefill_lengths(self) -> list[int]:
		"""Returns the configured list of max prefill length buckets for the engine."""
		return self._max_prefill_lengths

	@property
	def batch_size(self) -> int | None:
		"""Returns the configured batch size for the engine, if specified."""
		return self._batch_size

	@property
	def max_concurrent_decodes(self) -> int:
		"""Maximum number of sequences that can be decoded concurrently.

		This determines the batch size used during the decode phase.
		"""
		return self._max_concurrent_decodes

	@property
	def max_prefill_length(self) -> int:
		"""Maximum allowed length for the initial prompt (prefill phase).

		Prompts longer than this will be truncated or handled according to
		the padding/truncation logic.
		"""
		return self._max_prefill_length

	@property
	def max_length(self) -> int:
		"""Maximum total sequence length (prompt + generation).

		This defines the size of the KV cache allocated.
		"""
		return self._max_length

	def get_prefix_destination_sharding(self) -> tp.Any:
		"""Returns the shardings necessary to transfer KV cache data between engines.

		Currently returns None, indicating default or no specific sharding.
		"""
		return None

	def init_decode_state(self, *args, **kwargs) -> GenerationState | None:
		"""Initializes the GenerationState for a new sequence.

		Currently returns None, suggesting state might be initialized elsewhere
		or implicitly within prefill.
		"""
		with self.model.mesh:
			return GenerationState(
				logits=jnp.zeros(
					(self.max_concurrent_decodes, self.model.config.vocab_size), self.model.dtype
				),
				cache=self.model.init_cache(self.max_concurrent_decodes, self.max_length),
				index=jnp.zeros((self.max_concurrent_decodes, 1), "i4"),
				tokens=jnp.zeros((self.max_concurrent_decodes, 1), "i4"),
				valids=jnp.zeros((self.max_concurrent_decodes, self.max_length), "b1"),
				next_position_ids=jnp.zeros((self.max_concurrent_decodes, 1), "i4"),
				sampling_params=None,
				generated_tokens=jnp.zeros((self.max_concurrent_decodes, 1), "i4"),
			)

	def free_resource(self, slot: int) -> bool:
		"""Frees resources associated with a specific inference slot. (Not Implemented)

		Args:
		    slot: The index of the slot to free.

		Returns:
		    Always returns False as it's not implemented.
		"""
		return False  # Placeholder: Implementation needed

	@property
	def colocated_cpus(self) -> tp.Union[list[jax.Device], None]:
		"""Returns CPU devices colocated with the engine's accelerator devices.

		This information can be useful for optimizing data transfers between
		host (CPU) and accelerator (GPU/TPU) memory. Currently returns None
		as the implementation is pending.

		Returns:
		    A list of colocated JAX CPU devices, or None if not implemented or available.
		"""
		return None  # Placeholder: Implementation needed

	@staticmethod
	# @implicit
	@partial(jax.jit, static_argnums=(0, 5))
	def prefill(
		graphdef: nn.GraphDef,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		tokens: jax.Array,
		valids: jax.Array,
		max_length: int,
		sampling_params: SamplingParams,
		samples_per_slot: int,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""Performs the prefill step for initializing the generation process.

		Processes the initial prompt tokens, initializes the KV cache, and generates
		the *first* token of the sequence. This function is JIT-compiled.

		Args:
		    graphdef: The NNX GraphDef of the model.
		    graphstate: The NNX GraphState (parameters) of the model.
		    graphothers: Other NNX state variables of the model.
		    tokens: The input prompt token IDs (batch_size, sequence_length).
		    valids: A boolean array indicating valid token positions in the input
		        (batch_size, sequence_length or batch_size, max_length).
		    max_length: The maximum sequence length for the KV cache (static argument).
		    sampling_params: Parameters controlling the sampling process.
		    rngs: A JAX PRNG key for sampling the first token.

		Returns:
		    A tuple containing:
		        - generation_state: The initial GenerationState for the decode loop.
		        - result: A ResultTokens object containing the *first* generated token.
		"""
		model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphothers)
		batch_size, sequence_length = tokens.shape
		if valids.shape[-1] != max_length:
			valids = jax.lax.dynamic_update_slice(
				jnp.ones((batch_size, max_length), "b1"),
				valids.astype("b1"),
				(0, 0),
			).astype("b1")
		position_ids = (valids.cumsum(axis=-1) - 1)[:, :sequence_length]
		kv_cache = model.init_cache(batch_size=batch_size, max_length=max_length)
		with model.mesh:
			outputs = model(
				input_ids=tokens,
				attention_mask=valids,
				position_ids=position_ids,
				past_key_values=kv_cache,
			)

		kv_cache = outputs.past_key_values
		logits = outputs.logits[:, -1]

		# Apply logits processors/warpers (currently commented out)
		# logits = sampling_params.logits_processor(tokens, logits, sequence_length)
		# logits = sampling_params.logits_warper(tokens, logits, sequence_length)
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
			next_position_ids=position_ids[:, -1:] + 1,
			sampling_params=sampling_params,
			generated_tokens=jnp.zeros((batch_size, 1), dtype=jnp.int32),
		)
		return generation_state, result

	@staticmethod
	# @implicit
	@partial(jax.jit, static_argnums=(0, 4), donate_argnums=(3,))
	def decode(
		graphdef: nn.GraphDef,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: GenerationState,
		samples_per_slot: int,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""Performs a single decode step in the autoregressive generation loop.

		Takes the previous GenerationState, generates the next token using the model
		and KV cache, and updates the state. This function is JIT-compiled and
		allows the input state's cache to be modified in-place (donated).

		Args:
		    graphdef: The NNX GraphDef of the model (static argument).
		    graphstate: The NNX GraphState (parameters) of the model.
		    graphothers: Other NNX state variables of the model.
		    state: The current GenerationState from the previous step. `state.cache`
		        is marked for donation.
		    rngs: A JAX PRNG key for sampling the next token.

		Returns:
		    A tuple containing:
		        - next_generation_state: The updated GenerationState for the next iteration.
		        - result: A ResultTokens object containing the newly generated token.
		"""
		model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphothers)
		batch_size = state.tokens.shape[0]
		with model.mesh:
			outputs = model(
				input_ids=state.tokens,
				attention_mask=state.valids,
				position_ids=state.next_position_ids,
				past_key_values=state.cache,
			)

		kv_cache = outputs.past_key_values
		logits = outputs.logits[:, -1]

		# Apply logits processors/warpers (currently commented out)
		# logits = state.sampling_params.logits_processor(None, logits, state.index)
		# logits = state.sampling_params.logits_warper(None, logits, state.index)

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
			tokens=next_token,  # The token just generated
			valids=state.valids,
			next_position_ids=state.next_position_ids + 1,
			sampling_params=state.sampling_params,
			generated_tokens=state.generated_tokens + 1,
		)
		return next_generation_state, result

	@staticmethod
	@partial(jax.jit, donate_argnums=(0, 1))
	def insert(
		prefix: GenerationState,
		decode_state: GenerationState,
		slot: int,
	) -> GenerationState:
		"""Inserts or updates a generation state, potentially for managing batches. (JIT-compiled)

		This function seems designed to merge or update parts of the generation state,
		possibly inserting a 'prefix' state (e.g., from a completed prefill) into
		a larger batch state ('decode_state') at a specific 'slot'. The exact
		mechanism for insertion isn't fully clear from the current implementation,
		as it primarily focuses on broadcasting the prefix cache and returning the
		prefix state. Both input states' caches are donated.

		Args:
		    prefix: The GenerationState to potentially insert (e.g., from prefill).
		        Its cache is marked for donation.
		    decode_state: The target GenerationState to update (e.g., the main decode loop state).
		        Its cache is marked for donation.
		    slot: The index within the batch where the insertion/update should occur.

		Returns:
		    An updated GenerationState. In the current implementation, it returns the
		    prefix state with its cache potentially broadcasted. Needs clarification
		    on the intended merging logic with `decode_state` and `slot`.
		"""

		def update_idx2d(x, y):
			return jax.lax.dynamic_update_slice(x, y, (slot, 0))

		return GenerationState(
			logits=update_idx2d(decode_state.logits, prefix.logits),
			cache=decode_state.cache.insert(prefix.cache, slot=slot),
			index=update_idx2d(decode_state.index, prefix.index),
			tokens=update_idx2d(decode_state.tokens, prefix.tokens),
			valids=update_idx2d(decode_state.valids, prefix.valids),
			next_position_ids=update_idx2d(
				decode_state.next_position_ids,
				prefix.next_position_ids,
			),
			sampling_params=prefix.sampling_params,
			generated_tokens=update_idx2d(
				decode_state.generated_tokens,
				prefix.generated_tokens,
			),
		)

	@staticmethod
	@partial(jax.jit, donate_argnums=(0, 1))
	def bulk_insert(
		prefix: GenerationState,
		decode_state: GenerationState,
		slots: list[int],
	) -> GenerationState:
		"""Efficiently inserts multiple prefill results into the decode state.

		This function takes a `GenerationState` (`prefix`) typically resulting
		from a batch prefill operation and inserts its relevant components
		(logits, cache, index, tokens, valids, position IDs, generated tokens)
		into the main `decode_state` at multiple specified `slots`. This is
		useful for initializing the decode state after processing a batch of
		prompts simultaneously. Both input states' caches are donated.

		Args:
		    prefix: The `GenerationState` containing the results from a prefill
		        operation (or similar initialization). Its cache is marked for
		        donation.
		    decode_state: The target `GenerationState` (e.g., the main decode
		        loop state) to be updated. Its cache is marked for donation.
		    slots: A list of integer indices indicating the slots within the
		        `decode_state`'s batch dimension where the corresponding data
		        from the `prefix` state should be inserted.

		Returns:
		    An updated `GenerationState` (`decode_state`) with the prefill
		    results inserted at the specified slots.
		"""

		def update_idx2d(x, y, s):
			return jax.lax.dynamic_update_slice(x, y, (s, 0))

		for slot in slots:
			logits = update_idx2d(
				decode_state.logits,
				prefix.logits,
				slot,
			)
			cache = decode_state.cache.insert(
				prefix.cache,
				slot=slot,
			)
			index = update_idx2d(
				decode_state.index,
				prefix.index,
				slot,
			)
			tokens = update_idx2d(
				decode_state.tokens,
				prefix.tokens,
				slot,
			)
			valids = update_idx2d(
				decode_state.valids,
				prefix.valids,
				slot,
			)
			next_position_ids = update_idx2d(
				decode_state.next_position_ids,
				prefix.next_position_ids,
				slot,
			)
			generated_tokens = update_idx2d(
				decode_state.generated_tokens,
				prefix.generated_tokens,
				slot,
			)
		return GenerationState(
			logits=logits,
			cache=cache,
			index=index,
			tokens=tokens,
			valids=valids,
			next_position_ids=next_position_ids,
			generated_tokens=generated_tokens,
		)
