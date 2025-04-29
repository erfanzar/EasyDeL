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

import abc
import typing as tp

import jax
from eformer import common_types
from flax import nnx as nn


if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
	from .vengine.utilities import GenerationState, ResultTokens
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


class AbstractInferenceEngine(abc.ABC):
	"""
	Abstract Base Class for vEngine implementations.

	Defines the core interface for inference engines managing model state,
	KV caching, and providing JIT-compiled functions for prefill and decode steps.
	"""

	@property
	@abc.abstractmethod
	def pad_token_id(self) -> int | None:
		"""The ID of the padding token."""
		pass

	@property
	@abc.abstractmethod
	def eos_token_ids(self) -> list[int]:
		"""A list of End-of-Sequence token IDs."""
		pass

	@property
	@abc.abstractmethod
	def samples_per_slot(self) -> int:
		"""Number of samples generated per inference slot."""
		pass

	@property
	@abc.abstractmethod
	def prng_key(self) -> jax.random.PRNGKey:
		"""Provides a new PRNG key split from the internal state."""
		pass

	@property
	@abc.abstractmethod
	def prefill_lengths(self) -> list[int]:
		"""Configured list of max prefill length buckets."""
		pass

	@property
	@abc.abstractmethod
	def batch_size(self) -> int | None:
		"""Configured batch size for the engine."""
		pass

	@property
	@abc.abstractmethod
	def max_concurrent_decodes(self) -> int:
		"""Maximum number of sequences decoded concurrently."""
		pass

	@property
	@abc.abstractmethod
	def max_prefill_length(self) -> int:
		"""Maximum allowed length for the initial prompt."""
		pass

	@property
	@abc.abstractmethod
	def max_length(self) -> int:
		"""Maximum total sequence length (prompt + generation)."""
		pass

	@abc.abstractmethod
	def get_prefix_destination_sharding(self) -> tp.Any:
		"""Returns shardings for transferring KV cache data."""
		pass

	@abc.abstractmethod
	def init_decode_state(self, *args, **kwargs) -> GenerationState:
		"""Initializes the GenerationState for a new sequence."""
		pass

	@abc.abstractmethod
	def free_resource(self, slot: int) -> bool:
		"""Frees resources associated with a specific inference slot."""
		pass

	@property
	@abc.abstractmethod
	def colocated_cpus(self) -> tp.Union[list[jax.Device], None]:
		"""Returns CPU devices colocated with the engine's accelerator devices."""
		pass

	@abc.abstractmethod
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
		"""Performs the prefill step."""
		pass

	@abc.abstractmethod
	def decode(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: GenerationState,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""Performs a single decode step."""
		pass

	@abc.abstractmethod
	def insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slot: int,
	) -> GenerationState:
		"""Inserts or updates a generation state."""
		pass

	@abc.abstractmethod
	def bulk_insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slots: list[int],
	) -> GenerationState:
		"""Efficiently inserts multiple prefill results into the decode state."""
		pass
