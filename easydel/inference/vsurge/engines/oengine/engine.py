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
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps
from transformers import ProcessorMixin

from easydel.layers.caching.paged_attention.managers import HBMPageManager
from easydel.layers.caching.paged_attention.paged_attention_cache import (
	PagedAttentionCache,
)
from easydel.utils.compiling_utils import cjit

from .._abstract_engine import AbstractInferenceEngine
from .utilities import execute_forward

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any


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
	def __init__(
		self,
		model: EasyDeLBaseModule,
		processor,
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
		from ...utils import DEFAULT_PREFILL_BUCKETS

		self.model = model
		self.storage = storage
		self.manager = manager
		self.graphdef, self.graphstate, self.graphothers = model.split_module()
		self.processor = processor
		self._quantizer = model._quant_class(
			quantization_method=model.config.kv_cache_quantization_method,
			block_size=model.config.kv_cache_quantization_blocksize,
			quantization_platform=model.config.platform,
		)
		self._partition_manager = model.config.partition_manager
		self._max_prefill_lengths = prefill_lengths or DEFAULT_PREFILL_BUCKETS
		self._max_concurrent_decodes = max_concurrent_decodes or jax.device_count()
		self._max_concurrent_prefill = max_concurrent_prefill or jax.device_count()
		self._max_prefill_length = max_prefill_length or 4096
		self._max_length = max_length or 8192
		self._batch_size = batch_size
		self._prng_key = jax.random.PRNGKey(seed)
		self._empty_sharding = Ns(model.mesh, Ps())
		self._setup_functions()

	def get_state_shardings(self, is_decode: bool = False):
		return (".*", Ps())

	def _setup_functions(self):
		with self.model.mesh:
			self.graphstate_sharding = es.extract_shardings(self.graphstate)
			self.graphother_sharding = es.extract_shardings(self.graphother)
			self.storage_sharding = es.extract_shardings(self.storage)
 
			self.continuous_forward = cjit(
				jax.jit(
					execute_forward,
					static_argnums=(0,),
					in_shardings=(
						self.graphstate_sharding,
						self.graphother_sharding,
						None,
						None,
						self.storage_sharding,
						None,
					),
					out_shardings=(None, self.storage_sharding, None),
					donate_argnums=(5,),
				),
				static_argnums=(0,),
			)

	@cached_property
	def pad_token_id(self):
		if isinstance(self.processor, ProcessorMixin):
			pad_token_id = self.processor.tokenizer.pad_token_id
		else:
			pad_token_id = self.processor.pad_token_id
		return pad_token_id

	@cached_property
	def eos_token_ids(self) -> list[int]:
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
	def prefill_lengths(self) -> list[int]:
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

	def init_decode_state(self, *args, **kwargs) -> GenerationState: ...

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
	) -> tuple[GenerationState, ResultTokens]: ...

	def decode(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: GenerationState,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]: ...

	def insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slot: int,
	) -> GenerationState: ...

	def bulk_insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slots: list[int],
	) -> GenerationState: ...
