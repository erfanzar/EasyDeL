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
from functools import cached_property

import jax
from eformer import common_types
from eformer import escale as es
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps
from transformers import ProcessorMixin

from easydel.layers.caching.paged_attention import (
	ActiveSequenceBatch,
	HBMPageManager,
	InferenceScheduler,
	ModelIOProcessor,
	ModelOutputBatch,
	PagedAttentionCache,
)
from easydel.layers.caching.paged_attention.types import NextIterationPlan
from easydel.utils.compiling_utils import cjit

from .._abstract_engine import AbstractInferenceEngine
from .utilities import execute_forward

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
	from easydel.infra.utils import ProcessingClassType
else:
	EasyDeLBaseModule = tp.Any
	ProcessingClassType = tp.Any

GenerationState = tp.Any
ResultTokens = tp.Any

NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ
MODE_PREFILL = common_types.MODE_PREFILL


class oEngine(AbstractInferenceEngine):
	"""
	Optimized inference engine for EasyDeL models using Paged Attention.

	This engine manages the inference process, including KV cache management
	with Paged Attention, scheduling, and execution of model forward passes
	for both prefill and decode steps.
	"""

	def __init__(
		self,
		model: EasyDeLBaseModule,
		processor: ProcessingClassType,
		storage: PagedAttentionCache,
		manager: HBMPageManager,
		max_concurrent_decodes: int | None = None,
		max_concurrent_prefill: int | None = None,
		prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		batch_size: int | None = None,
		seed: int = 894,
	):
		"""
		Initializes the oEngine.

		Args:
		    model: The EasyDeL model to use for inference.
		    processor: The processor (tokenizer/feature extractor) for the model.
		    storage: The PagedAttentionCache instance for KV cache management.
		    manager: The HBMPageManager instance for managing memory pages.
		    max_concurrent_decodes: Maximum number of sequences to decode concurrently.
		    max_concurrent_prefill: Maximum number of sequences to prefill concurrently.
		    prefill_lengths: List of maximum prefill lengths to bucket by.
		    max_prefill_length: Maximum allowed length for the initial prompt.
		    max_length: Maximum total sequence length (prompt + generation).
		    batch_size: The batch size for inference.
		    seed: The random seed for PRNG key initialization.
		"""
		self.model = model
		self.storage = storage
		self.manager = manager
		self.scheduler = InferenceScheduler(manager=manager)
		self.metadata = manager.metadata
		# Split the model into graph definition, graph state, and other components
		self.graphdef, self.graphstate, self.graphothers = model.split_module()
		self.processor = processor
		# Initialize the quantizer based on model configuration
		self._quantizer = model._quant_class(
			quantization_method=model.config.kv_cache_quantization_method,
			block_size=model.config.kv_cache_quantization_blocksize,
			quantization_platform=model.config.platform,
		)
		self._partition_manager = model.config.partition_manager
		self._max_prefill_lengths = prefill_lengths or [2**s for s in range(9, 24)]
		self._max_concurrent_decodes = max_concurrent_decodes or jax.device_count()
		self._max_concurrent_prefill = max_concurrent_prefill or jax.device_count()
		self._max_prefill_length = max_prefill_length or 4096
		self._max_length = max_length or 8192
		self._batch_size = batch_size
		self._prng_key = jax.random.PRNGKey(seed)
		self._empty_sharding = Ns(model.mesh, Ps())
		self._setup_functions()

	@property
	def max_concurrent_prefill(self):
		return self._max_concurrent_prefill

	@property
	def max_concurrent_decodes(self):
		return self._max_concurrent_decodes

	def get_state_shardings(self, is_decode: bool = False):
		"""
		Returns the sharding specifications for the engine's state.

		Args:
		    is_decode: A boolean indicating if the sharding is for the decode state.

		Returns:
		    A tuple representing the sharding specification.
		"""
		return (".*", Ps())

	def _setup_functions(self):
		"""
		Sets up JIT-compiled functions for the inference engine.

		This method defines the sharding for model components and the KV cache
		and compiles the `execute_forward` function using JAX's `jit` and
		EasyDeL's `cjit` for efficient execution within the model's mesh.
		"""
		with self.model.mesh:
			# Extract sharding specifications for model state, other components, and storage
			self.graphstate_sharding = es.extract_shardings(self.graphstate)
			self.graphother_sharding = es.extract_shardings(self.graphothers)
			self.storage_sharding = es.extract_shardings(self.storage)

			# Compile the execute_forward function with specified shardings
			self.continuous_forward = cjit(
				jax.jit(
					execute_forward,
					static_argnums=(0,),  # graphdef is static
					in_shardings=(
						self.graphstate_sharding,  # Sharding for graphstate
						self.graphother_sharding,  # Sharding for graphothers
						None,  # Sharding for ModelIOProcessor input (handled internally)
						None,  # Sharding for eos_token_ids (broadcast)
						self.storage_sharding,  # Sharding for storage (KV cache)
						None,  # Sharding for prng_key (handled internally)
					),
					out_shardings=(
						None,
						self.storage_sharding,
						None,
					),  # Output shardings for (output, updated_storage, _)
					donate_argnums=(5,),  # Donate the storage argument to avoid copying
				),
				static_argnums=(0,),  # graphdef is static for cjit as well
			)

	@cached_property
	def pad_token_id(self):
		"""
		Returns the pad token ID from the processor.
		"""
		if isinstance(self.processor, ProcessorMixin):
			pad_token_id = self.processor.tokenizer.pad_token_id
		else:
			pad_token_id = self.processor.pad_token_id
		return pad_token_id

	@cached_property
	def eos_token_ids(self) -> list[int]:
		"""
		Returns a list of end-of-sequence token IDs from the processor and model config.
		"""
		eos_ids = []
		if isinstance(self.processor, ProcessorMixin):
			proc_eos_token_id = self.processor.tokenizer.eos_token_id
		else:
			proc_eos_token_id = self.processor.eos_token_id
		if isinstance(proc_eos_token_id, int):
			proc_eos_token_id = [proc_eos_token_id]
		eos_ids = eos_ids + proc_eos_token_id
		if hasattr(self.model, "generation_config"):
			conf_eos = self.model.generation_config.eos_token_id
			if isinstance(conf_eos, int):
				conf_eos = [conf_eos]
			eos_ids = eos_ids + conf_eos
		return list(set(eos_ids))

	@property
	def samples_per_slot(self) -> int:
		"""Number of samples generated per inference slot.

		This determines how many independent generation results are produced
		for each logical slot managed by the engine. It's often 1, but could
		be higher for techniques like parallel sampling.
		"""
		return 1

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
	def prefill_lengths(self) -> list[int]:
		"""Returns the configured list of max prefill length buckets for the engine."""
		return self._max_prefill_lengths

	@property
	def batch_size(self) -> int | None:
		"""Returns the configured batch size for the engine, if specified."""
		return self._batch_size

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
		"""
		Returns the shardings necessary to transfer KV cache data between engines.

		This method is intended for scenarios involving multiple engines or devices
		where KV cache data needs to be moved.

		Returns:
		    The sharding specification for prefix destinations.
		"""
		pass  # Implementation needed

	def init_decode_state(self, *args, **kwargs) -> ActiveSequenceBatch:
		"""
		Initializes the decode state for active sequences.

		Args:
		    *args: Variable length argument list.
		    **kwargs: Arbitrary keyword arguments.

		Returns:
		    An initialized ActiveSequenceBatch instance.
		"""
		return ActiveSequenceBatch.create(self.metadata, self.model.config.mesh)

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

	def forward(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: ActiveSequenceBatch,
		iteration_plan: NextIterationPlan,
	) -> ModelOutputBatch:
		"""
		Performs a forward pass of the model.

		This method executes the compiled `continuous_forward` function,
		processing the input state and iteration plan to produce model outputs
		and update the KV cache storage.

		Args:
		    graphstate: The graph state of the model.
		    graphothers: Other graph components of the model.
		    state: The current active sequence batch state.
		    iteration_plan: The plan for the current inference iteration.

		Returns:
		    The output batch from the model.
		"""
		with self.model.mesh:
			# Execute the compiled forward pass
			output, self.storage, _ = self.continuous_forward(
				self.graphdef,
				graphstate,
				graphothers,
				ModelIOProcessor.build_input(  # Build the input for the forward pass
					iteration_plan=iteration_plan,
					metadata=self.metadata,
					decodes_state=state,
				),
				jnp.array(self.eos_token_ids).ravel(),  # Provide EOS token IDs
				self.storage,  # Pass the current KV cache storage
				self.prng_key,  # Pass the PRNG key for sampling
			)
		return output

	def prefill(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		tokens: jax.Array,
		valids: jax.Array,
		true_length: int,
		temperature: jax.Array,
		top_p: jax.Array,
		top_k: jax.Array,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""
		Performs the prefill step for a batch of prompts.

		This involves processing the initial prompt tokens and populating
		the KV cache.

		Args:
		    graphstate: The graph state of the model.
		    graphothers: Other graph components of the model.
		    tokens: The input tokens for the prompts.
		    valids: A boolean array indicating valid tokens.
		    true_length: The true length of the sequences.
		    temperature: The temperature for sampling.
		    top_p: The top-p value for sampling.
		    top_k: The top-k value for sampling.
		    rngs: The PRNG key for sampling.

		Returns:
		    A tuple containing the generation state after prefill and the result tokens.
		"""
		pass

	def decode(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: GenerationState,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""
		Performs a single decode step for active sequences.

		This involves generating the next token for each sequence based on
		the current state and KV cache.

		Args:
		    graphstate: The graph state of the model.
		    graphothers: Other graph components of the model.
		    state: The current generation state.
		    rngs: The PRNG key for sampling.

		Returns:
		    A tuple containing the updated generation state and the generated result tokens.
		"""
		pass

	def insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slot: int,
	) -> GenerationState:
		"""
		Inserts or updates a generation state for a specific slot.

		This is typically used to integrate the results of a prefill step
		into the ongoing decode process for a particular sequence slot.

		Args:
		    prefix: The generation state from the prefill step.
		    decode_state: The current decode state.
		    slot: The slot index to insert into.

		Returns:
		    The updated decode state.
		"""
		pass

	def bulk_insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slots: list[int],
	) -> GenerationState:
		"""
		Efficiently inserts multiple prefill results into the decode state.

		This method is optimized for integrating the results of a batch
		prefill operation into the decode state for multiple sequence slots.

		Args:
		    prefix: The generation state from the bulk prefill step.
		    decode_state: The current decode state.
		    slots: A list of slot indices to insert into.

		Returns:
		    The updated decode state.
		"""
		pass
