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

from __future__ import annotations

import queue
import threading
import typing as tp
import uuid

import jax
import numpy as np
from eformer import common_types
from eformer import escale as es
from eformer.pytree import auto_pytree, xTree
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps

from .paged_attention_cache import PagedAttentionCacheMetaData, PagedAttentionMetadata


@auto_pytree
class SamplingParams:
	"""
	Data class for sampling configuration parameters for text generation.

	Attributes:
	    top_p (jax.Array): Nucleus sampling probability threshold.
	    top_k (jax.Array): Top-k sampling token limit.
	    max_tokens (jax.Array): Maximum number of tokens to generate.
	    temperature (jax.Array): Sampling temperature.
	"""

	top_p: jax.Array
	top_k: jax.Array
	max_tokens: jax.Array
	temperature: jax.Array

	def __post_init__(self):
		acceptin = (float, int)

		if isinstance(self.top_p, acceptin):
			self.top_p = jnp.array(float(self.top_p)).reshape(-1)
		if isinstance(self.top_k, acceptin):
			self.top_k = jnp.array(int(self.top_k)).reshape(-1)
		if isinstance(self.max_tokens, acceptin):
			self.max_tokens = jnp.array(int(self.max_tokens)).reshape(-1)
		if isinstance(self.temperature, acceptin):
			self.temperature = jnp.array(float(self.temperature)).reshape(-1)

	def insert_from_task(self, slot: int, task: GenerationStepTask):
		self.top_p[slot] = task.top_p
		self.top_k[slot] = task.top_k
		self.max_tokens[slot] = task.max_tokens
		self.temperature[slot] = task.temperature

	def insert_from_decode_state(
		self,
		insert_slots: jax.Array,
		update: ActiveSequenceBatch,
	):
		self.top_p = self.top_p.at[insert_slots].set(update.sampling_params.top_p)
		self.top_k = self.top_k.at[insert_slots].set(update.sampling_params.top_k)

		self.max_tokens = self.max_tokens.at[insert_slots].set(
			update.sampling_params.max_tokens
		)
		self.temperature = self.temperature.at[insert_slots].set(
			update.sampling_params.temperature
		)

	def move_to_device(self, sharding: Ns):
		assert self.is_active
		self.top_p = jax.device_put(self.top_p, sharding)
		self.top_k = jax.device_put(self.top_k, sharding)
		self.max_tokens = jax.device_put(self.max_tokens, sharding)
		self.temperature = jax.device_put(self.temperature, sharding)

	@classmethod
	def init_numpy(cls, metadata: PagedAttentionMetadata):
		return cls(
			top_p=jnp.full((metadata.batch_size,), 1, jnp.float32),
			top_k=jnp.full((metadata.batch_size,), 0, jnp.int32),
			max_tokens=jnp.full((metadata.batch_size,), 16, jnp.int32),
			temperature=jnp.full((metadata.batch_size,), 1, jnp.float32),
		)

	@classmethod
	def init_empty(cls):
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		return cls(
			top_p=scalar,
			top_k=scalar,
			max_tokens=scalar,
			temperature=scalar,
		)

	@classmethod
	def create(
		cls,
		metadata: PagedAttentionCacheMetaData,
		mesh: common_types.Mesh,
	):
		d1replicate = Ns(mesh, Ps(None))
		return cls(
			top_p=jnp.full((metadata.batch_size,), 1, jnp.float32, device=d1replicate),
			top_k=jnp.full((metadata.batch_size,), 0, jnp.int32, device=d1replicate),
			max_tokens=jnp.full((metadata.batch_size,), 16, jnp.int32, device=d1replicate),
			temperature=jnp.full((metadata.batch_size,), 1, jnp.float32, device=d1replicate),
		)

	def is_active(self):
		return len(self.top_p.shape) > 0


@auto_pytree
class InitialSequenceRequest:
	"""
	Represents a new request that needs to be processed during the prefill phase
	of paged attention. It holds information necessary to initialize the attention
	state for a given prompt.

	Attributes:
	    id (str | int): A unique identifier for the request.
	    chunk_idx (int): The index of the current chunk being processed for this request.
	        Used when the prompt is processed in multiple chunks.
	    token_ids (jax.Array): The JAX array containing the token IDs for the prefill phase.
	        This might be padded to a fixed length.
	    positions (jax.Array): The JAX array containing the corresponding positions for the token IDs.
	    chunk_size (int): The size of the chunk being processed in this prefill step.
	    page_indices (list[int]): A list to hold the indices of the HBM pages allocated
	        for this request's KV cache. Initialized with zeros.
	    prompt_token_ids (list[int]): The original list of token IDs for the complete prompt.
	"""

	token_ids: jax.Array
	positions: jax.Array
	page_indices: jax.Array

	# sampling_params: SamplingParams

	id: tp.Optional[str | int]
	chunk_idx: tp.Optional[int]
	chunk_size: tp.Optional[int]
	prompt_token_ids: tp.Optional[list[int]]
	length: tp.Optional[jax.Array] = None

	@property
	def is_active(self):
		"""Whether the sequence request is active (token_ids non-empty)."""
		return len(self.token_ids.shape) > 0

	def copy_prefill(self, prefill: InitialSequenceRequest):
		"""
		Copies relevant state from another InitialSequenceRequest.

		Args:
		    prefill (InitialSequenceRequest): Source request to copy state from.
		"""
		length = (prefill.chunk_idx + 1) * prefill.chunk_size
		total_length = len(prefill.prompt_token_ids)
		if length > total_length:
			length = total_length

		self.token_ids = prefill.token_ids
		self.positions = prefill.positions
		self.page_indices = np.array(prefill.page_indices)
		self.length = length
		# self.sampling_params = prefill.sampling_params

	@classmethod
	def init_empty(cls):
		"""
		Creates an empty InitialSequenceRequest placeholder with default values.

		Returns:
		    InitialSequenceRequest: A placeholder request object.
		"""
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		return cls(
			id=None,
			length=scalar,
			chunk_idx=None,
			token_ids=scalar,
			positions=scalar,
			chunk_size=512,
			page_indices=scalar,
			prompt_token_ids=None,
			# sampling_params=SamplingParams.init_empty(),
		)

	@classmethod
	def create(
		cls,
		mesh: common_types.Mesh,
		metadata: PagedAttentionCacheMetaData,
		chunk_size: int,
		prompt_token_ids: list[int],
		prefill_length: tp.Optional[int] = None,
		# sampling_params: tp.Optional[SamplingParams] = None,
	):
		"""
		Factory method to create a InitialSequenceRequest from a list of prompt token IDs.

		Args:
		    mesh (common_types.Mesh): The JAX device mesh for distributing arrays.
		    metadata (PagedAttentionCacheMetaData): Metadata about the paged attention cache configuration.
		    chunk_size (int): The size of chunks for processing the prefill.
		    prompt_token_ids (list[int]): The list of token IDs representing the input prompt.
		    prefill_length (tp.Optional[int]): The target length for padding the prefill tokens.
		        Defaults to `metadata.max_sequences`.

		Returns:
		    InitialSequenceRequest: An initialized InitialSequenceRequest object.
		"""
		if prefill_length is None:
			prefill_length = metadata.max_sequences

		sequence_length = len(prompt_token_ids)
		paddlen = prefill_length - sequence_length
		array = np.array(prompt_token_ids)

		if paddlen < 0:
			padded_token_ids = array[-paddlen:]
		else:
			padded_token_ids = np.pad(array, (0, paddlen), constant_values=(0,))
		sharding = Ns(mesh, Ps(None))
		page_indices = [0] * metadata.num_pages_per_sequence
		token_ids = jax.device_put(padded_token_ids, sharding)
		positions = np.arange(0, token_ids.shape[0])
		positions = jax.device_put(positions, sharding)
		# if sampling_params is None:
		# 	sampling_params = SamplingParams(
		# 		top_k=1,
		# 		top_p=1,
		# 		max_tokens=16,
		# 		temperature=1,
		# 	)
		# sampling_params.move_to_device(sharding)
		return InitialSequenceRequest(
			id=uuid.uuid4(),
			prompt_token_ids=prompt_token_ids,
			chunk_idx=0,
			# sampling_params=sampling_params,
			chunk_size=chunk_size,
			page_indices=page_indices,
			token_ids=token_ids,
			positions=positions,
		)


@auto_pytree
class AllocatedPrefillPages:
	"""
	Represents an update containing the indices of newly allocated HBM pages
	during a prefill step.

	Attributes:
	    page_indices (list[int]): A list of HBM page indices allocated for a prefill chunk.
	"""

	page_indices: list[int]


@auto_pytree
class SlotPageAssignment:
	"""
	Represents an update to a specific page mapping within the decode state's page table.
	This is used when a new page needs to be assigned to a sequence during the decoding phase.

	Attributes:
	    slot (int): The slot index in the batch dimension corresponding to the sequence being updated.
	    page_idx (int): The logical page index within the sequence's page table that needs updating.
	    mapped_idx (int): The physical HBM page index that the logical page index should now point to.
	"""

	slot: int
	page_idx: int
	mapped_idx: int


@auto_pytree
class GenerationStepTask:
	"""
	Represents a request currently in the decoding (generation) phase.

	Attributes:
	    id (str): The unique identifier of the original request.
	    slot (int): The assigned slot index within the batch dimension for this request.
	    position (int): The current sequence position (length) for this request during decoding.
	    page_indices (list[int]): The list of HBM page indices currently allocated to this request.
	    prefill_token_id (jax.Array): The token ID generated during the last step (prefill or decode)
	        which is the input for the current decode step.
	"""

	id: str
	slot: int
	position: int
	page_indices: list[int]
	prefill_token_id: jax.Array


@auto_pytree
class NextIterationPlan:
	"""
	Holds the results of a scheduling decision, indicating which operations (prefill, decode)
	should be performed in the next model iteration and the necessary data updates.

	Attributes:
	    prefill_request (InitialSequenceRequest | None): The prefill request selected for the next iteration,
	        or None if no prefill is scheduled.
	    schedule_prefill (bool): Flag indicating whether a prefill operation should be run.
	    schedule_decodes (bool): Flag indicating whether a decode operation should be run.
	    prefill_pages_update (AllocatedPrefillPages | None): Contains the newly allocated pages if a
	        prefill operation is scheduled and required page allocation.
	    new_decodes_requests (list[GenerationStepTask]): A list of decode requests that are newly added
	        to the batch in this scheduling step.
	    decodes_state_page_updates (list[SlotPageAssignment]): A list of updates to the decode state's
	        page table required for ongoing decode requests.
	"""

	prefill_request: InitialSequenceRequest

	schedule_prefill: bool
	schedule_decodes: bool

	prefill_pages_update: AllocatedPrefillPages
	new_decodes_requests: list[GenerationStepTask]

	decodes_state_page_updates: list[SlotPageAssignment]


@auto_pytree
class ActiveSequenceBatch:
	"""
	Maintains the state required for the decoding phase of paged attention for a batch of requests.
	This includes token IDs, positions, page tables, sampling parameters, and management of active slots.

	Attributes:
	    token_ids (jax.Array): JAX array holding the most recently generated token ID for each active slot.
	        Shape: (batch_size,).
	    positions (jax.Array): JAX array holding the current sequence position (length) for each active slot.
	        Initialized with -1 for inactive slots. Shape: (batch_size,).
	    page_table (jax.Array): JAX array representing the mapping from logical page indices to physical
	        HBM page indices for each active slot. Shape: (batch_size, num_pages_per_sequence).
	    available_slots (queue.SimpleQueue): A queue holding the indices of batch slots that are currently free.
	    active_slot_requests_map (dict[int, GenerationStepTask]): A dictionary mapping active slot indices
	        to their corresponding GenerationStepTask objects.
	    context_lock (threading.Lock): A lock to ensure thread-safe access and modification of the decode state,
	        particularly `available_slots` and `active_slot_requests_map`.
	"""

	token_ids: jax.Array | tp.List[int]
	positions: jax.Array
	page_table: jax.Array

	# sampling_params: SamplingParams

	available_slots: tp.Optional[queue.SimpleQueue]
	active_slot_requests_map: tp.Optional[tp.Dict[int, GenerationStepTask]]
	context_lock: tp.Optional[threading.Lock] = threading.Lock()

	page_update_slots: tp.Optional[jax.Array] = None
	page_update_page_idxs: tp.Optional[jax.Array] = None
	page_update_mapped_idxs: tp.Optional[jax.Array] = None

	@property
	def is_active(self):
		"""Whether the sequence request is active (token_ids non-empty)."""
		return len(self.token_ids.shape) > 0

	def insert_decode_state(self, insert_slots: jax.Array, update: ActiveSequenceBatch):
		update.token_ids = jnp.asarray(update.token_ids)

		self.token_ids = self.token_ids.at[insert_slots].set(update.token_ids)
		self.positions = self.positions.at[insert_slots].set(update.positions)
		# self.sampling_params.insert_from_decode_state(insert_slots, update)
		self.page_table = self.page_table.at[insert_slots, :].set(update.page_table)

		self.page_table = self.page_table.at[
			update.page_update_slots,
			update.page_update_page_idxs,
		].set(update.page_update_mapped_idxs)

	def apply_assignment(self, assignment: SlotPageAssignment):
		assert self.page_update_slots is not None
		assert self.page_update_page_idxs is not None
		assert self.page_update_mapped_idxs is not None

		for i, update in enumerate(assignment):
			self.page_update_slots[i] = update.slot
			self.page_update_page_idxs[i] = update.page_idx
			self.page_update_mapped_idxs[i] = update.mapped_idx

	def pad_tokens(self, pad_length):
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		assert isinstance(self.token_ids, list)
		for i in range(pad_length):
			self.token_ids.append(scalar)

	def copy_decode(self, decode: ActiveSequenceBatch):
		self.token_ids = decode.token_ids
		self.positions = decode.positions
		# self.sampling_params = decode.sampling_params
		self.page_table = decode.page_table

	def insert_from_task(self, slot, task: GenerationStepTask):
		assert isinstance(self.token_ids, list)
		self.token_ids.append(task.prefill_token_id)
		self.positions[slot] = task.position
		# self.sampling_params.insert_from_task(slot, task)
		self.page_table[slot] = np.array(task.page_indices)

	@classmethod
	def init_numpy(cls, metadata: PagedAttentionCacheMetaData):
		return cls(
			token_ids=[],
			positions=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_table=np.full(
				(metadata.batch_size, metadata.num_pages_per_sequence), 1e6, dtype=np.int32
			),
			available_slots=None,
			active_slot_requests_map=None,
			context_lock=None,
			page_update_slots=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_update_page_idxs=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_update_mapped_idxs=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			# sampling_params=SamplingParams.init_numpy(metadata),
		)

	@classmethod
	def init_empty(cls):
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		return cls(
			token_ids=scalar,
			positions=scalar,
			page_table=scalar,
			available_slots=None,
			active_slot_requests_map=None,
			context_lock=None,
			page_update_slots=scalar,
			page_update_page_idxs=scalar,
			page_update_mapped_idxs=scalar,
			# sampling_params=SamplingParams.init_empty(),
		)

	@classmethod
	def create(
		cls,
		metadata: PagedAttentionCacheMetaData,
		mesh: common_types.Mesh,
	):
		"""
		Factory method to create and initialize a ActiveSequenceBatch object.

		Args:
		    metadata (PagedAttentionCacheMetaData): Configuration metadata for the paged attention cache.
		    mesh (common_types.Mesh): The JAX device mesh for array distribution.

		Returns:
		    ActiveSequenceBatch: An initialized ActiveSequenceBatch object with JAX arrays placed on the specified mesh.
		"""
		slots = queue.SimpleQueue()
		for i in range(metadata.batch_size):
			slots.put(i, block=True)
		d1replicate = Ns(mesh, Ps(None))
		d2replicate = Ns(mesh, Ps(None, None))
		return cls(
			token_ids=jnp.zeros(
				shape=(metadata.batch_size,),
				dtype=jnp.int32,
				device=d1replicate,
			),
			positions=jnp.full(
				shape=(metadata.batch_size,),
				fill_value=-1,
				dtype=jnp.int32,
				device=d1replicate,
			),
			page_table=jnp.full(
				shape=(metadata.batch_size, metadata.num_pages_per_sequence),
				fill_value=0,
				dtype=jnp.int32,
				device=d2replicate,
			),
			available_slots=slots,
			active_slot_requests_map={},
			# sampling_params=SamplingParams.create(metadata, mesh),
		)


class ModelInputBatch(xTree):
	"""
	Represents the consolidated input data structure passed to the model for a single
	paged attention forward pass, potentially combining prefill and decode steps.

	Attributes:
	    input_ids (jax.Array): Combined token IDs for both prefill (if any) and decode (if any) requests.
	    positions (jax.Array): Combined position IDs corresponding to `input_ids`.
	    attn_meta (PagedAttentionMetadata): Metadata required by the paged attention mechanism,
	        containing lengths, positions, and page tables for prefill and decode parts.
	"""

	input_ids: jax.Array
	positions: jax.Array
	attn_meta: PagedAttentionMetadata


@auto_pytree
class ModelOutputBatch:
	"""
	Represents the output generated by the model after a paged attention forward pass,
	separating results for prefill and decode phases.

	Attributes:
	    prefill_complete (jax.Array): A scalar boolean JAX array indicating if the prefill operation
	        (if scheduled) resulted in a completed sequence (e.g., EOS token).
	    decodes_completes (jax.Array): A boolean JAX array indicating for each decode slot whether
	        the generated token completed the sequence. Shape: (num_decode_requests,).
	    prefill_token_id (jax.Array): The token ID generated by the prefill operation (if scheduled). Scalar.
	    decodes_token_ids (jax.Array): The token IDs generated for each decode slot (if scheduled).
	        Shape: (num_decode_requests,).
	    prefill_next_position (jax.Array): The next position index for the prefill request after this step. Scalar.
	    decodes_next_position (jax.Array): The next position indices for each decode request after this step.
	        Shape: (num_decode_requests,). Contains -1 for completed sequences.
	"""

	prefill_complete: jax.Array
	decodes_completes: jax.Array
	prefill_token_id: jax.Array
	decodes_token_ids: jax.Array
	prefill_next_position: jax.Array
	decodes_next_position: jax.Array

	@classmethod
	def init_empty(cls):
		"""
		Factory method to create an empty ModelOutputBatch instance with placeholder values.

		Returns:
		    ModelOutputBatch: An instance with default scalar/vector values indicating no output.
		"""
		scalar = jnp.asarray(-1, dtype=jnp.int32)
		vector = jnp.asarray([-1], dtype=jnp.int32)
		return cls(
			prefill_complete=scalar,
			decodes_completes=vector,
			prefill_token_id=scalar,
			decodes_token_ids=vector,
			prefill_next_position=vector,
			decodes_next_position=scalar,
		)


@auto_pytree
class ModelOutputSummary:
	"""
	Contains the information needed to post-process the results of a model's forward pass
	(ModelOutputBatch) and update the scheduler and decode state.

	Attributes:
	    prefill_request_id (str | None): The ID of the prefill request that was processed, or None.
	    prefill_token_id (jax.Array): The token generated by the prefill step, or a placeholder.
	    prefill_complete (jax.Array): Flag indicating if the prefill step completed the sequence.
	    decodes_active_slots (list[int]): List of slot indices that were active during the decode phase.
	    decodes_active_request_ids (list[str]): List of request IDs corresponding to the active decode slots.
	    decodes_token_ids (jax.Array): Tokens generated for the active decode slots.
	    decodes_completes (jax.Array): Flags indicating completion for each active decode slot.
	"""

	prefill_request_id: str | None
	prefill_token_id: jax.Array
	prefill_complete: jax.Array

	decodes_active_slots: list[int]
	decodes_active_request_ids: list[str]
	decodes_token_ids: jax.Array
	decodes_completes: jax.Array

	@classmethod
	def from_output(cls, output: ModelOutputBatch):
		return cls(
			prefill_request_id=None,
			prefill_token_id=output.prefill_token_id,
			prefill_complete=output.prefill_complete,
			decodes_active_slots=[],
			decodes_active_request_ids=[],
			decodes_token_ids=output.decodes_token_ids,
			decodes_completes=output.decodes_completes,
		)


__all__ = (
	"SlotPageAssignment",
	"GenerationStepTask",
	"ActiveSequenceBatch",
	"InitialSequenceRequest",
	"AllocatedPrefillPages",
	"ModelInputBatch",
	"ModelOutputBatch",
	"ModelOutputSummary",
	"NextIterationPlan",
)
