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
from bisect import bisect_left

import jax
import numpy as np
from eformer import common_types
from eformer import escale as es
from eformer.pytree import auto_pytree, xTree, field
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps

from .paged_attention_cache import PagedAttentionCacheMetaData, PagedAttentionMetadata

DEFAULT_PREFILL_BUCKETS = [2**s for s in range(9, 24)]


@auto_pytree
class SamplingParams:
	"""Configuration parameters for controlling text generation sampling.

	This class holds parameters that influence the sampling process during
	text generation, such as top-p (nucleus) sampling, top-k sampling,
	maximum token generation, and temperature scaling.

	Attributes:
	    top_p (jax.Array | float): The probability threshold for nucleus sampling.
	        Defaults to 1.0 (no nucleus sampling).
	    max_tokens (jax.Array | int): The maximum number of tokens to generate
	        for a sequence. Defaults to 32.
	    temperature (jax.Array | float): The temperature for scaling logits before
	        sampling. Defaults to 0.0 (deterministic).
	"""

	top_p: jax.Array | float = field(default_factory=lambda: np.array([1.0]))
	max_tokens: jax.Array | int = field(default_factory=lambda: np.array([32]))
	temperature: jax.Array | float = field(default_factory=lambda: np.array([0.0]))

	def insert_from_task(self, slot: int, task: GenerationStepTask):
		"""Inserts sampling parameters from a GenerationStepTask into a specific slot.

		Args:
		    slot (int): The batch slot index to insert the parameters into.
		    task (GenerationStepTask): The task containing the sampling parameters
		        to insert.
		"""
		assert task.sampling_params.top_p.size == 1
		self.top_p[slot] = task.sampling_params.top_p[0]
		self.max_tokens[slot] = task.sampling_params.max_tokens[0]
		self.temperature[slot] = task.sampling_params.temperature[0]

	def insert_decode_state(self, insert_slots: jax.Array, update: ActiveSequenceBatch):
		"""Updates sampling parameters in specified slots from an ActiveSequenceBatch.

		Args:
		    insert_slots (jax.Array): An array of slot indices to update.
		    update (ActiveSequenceBatch): The batch containing the new sampling
		        parameters.
		"""
		smp = update.sampling_params
		self.top_p = self.top_p.at[insert_slots].set(smp.top_p)
		self.max_tokens = self.max_tokens.at[insert_slots].set(smp.max_tokens)
		self.temperature = self.temperature.at[insert_slots].set(smp.temperature)

	@classmethod
	def init_jax(
		cls, metadata: PagedAttentionCacheMetaData, sharding: jax.sharding.NamedSharding
	) -> "SamplingParams":
		"""Initializes SamplingParams with JAX arrays on the specified device/sharding.

		Args:
		    metadata (PagedAttentionCacheMetaData): Metadata containing batch size.
		    sharding (jax.sharding.NamedSharding): The JAX sharding configuration.

		Returns:
		    SamplingParams: An initialized SamplingParams object with JAX arrays.
		"""
		return cls(
			top_p=jnp.full(
				shape=(metadata.batch_size,),
				fill_value=1,
				dtype=jnp.float32,
				device=sharding,
			),
			max_tokens=jnp.full(
				shape=(metadata.batch_size,),
				fill_value=1e6,
				dtype=jnp.int32,
				device=sharding,
			),
			temperature=jnp.full(
				shape=(metadata.batch_size,),
				fill_value=1e6,
				dtype=jnp.float32,
				device=sharding,
			),
		)

	@classmethod
	def init_numpy(cls, metadata: PagedAttentionCacheMetaData) -> SamplingParams:
		"""Initializes SamplingParams with NumPy arrays.

		Args:
		    metadata (PagedAttentionCacheMetaData): Metadata containing batch size.

		Returns:
		    SamplingParams: An initialized SamplingParams object with NumPy arrays.
		"""
		return cls(
			top_p=np.full((metadata.batch_size,), 1e6, dtype=np.float32),
			max_tokens=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			temperature=np.full((metadata.batch_size,), 1e6, dtype=np.float32),
		)

	@classmethod
	def init_empty(cls) -> "SamplingParams":
		"""Creates an empty SamplingParams placeholder with scalar JAX arrays.

		Returns:
		    SamplingParams: A placeholder SamplingParams object.
		"""
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		return cls(top_p=scalar, max_tokens=scalar, temperature=scalar)


def take_nearest_length(lengths: list[int], length: int) -> int:
	"""Gets the nearest length to the right in a set of lengths.

	Uses binary search to find the smallest length in the `lengths` list that is
	greater than or equal to the input `length`.

	Args:
	    lengths: A sorted list of integer lengths (e.g., prefill buckets).
	    length: The target length to find the nearest value for.

	Returns:
	    The nearest length in `lengths` that is greater than or equal to `length`.
	    If `length` is greater than all lengths in the list, returns the largest length.
	"""
	pos = bisect_left(lengths, length)
	if pos == len(lengths):
		return lengths[-1]
	return lengths[pos]


@auto_pytree
class InitialSequenceRequest:
	"""Represents a request for processing a new sequence during the prefill phase.

	This class encapsulates the information needed to process a new input sequence
	(prompt) in the paged attention mechanism. It includes the token IDs,
	positions, allocated page indices, and associated metadata.

	Attributes:
	    token_ids (jax.Array): JAX array of token IDs for the prefill sequence,
	        potentially padded. (Runtime Argument)
	    positions (jax.Array): JAX array of position IDs corresponding to `token_ids`.
	        (Runtime Argument)
	    page_indices (jax.Array): JAX array holding the indices of HBM pages
	        allocated for this request's KV cache. (Runtime Argument)
	    sampling_params (SamplingParams): Sampling parameters for this specific
	        request. (Runtime Argument)
	    id (tp.Optional[str | int]): A unique identifier for the request.
	        (Scheduler Argument)
	    chunk_idx (tp.Optional[int]): The index of the current chunk being processed
	        if the prompt is chunked. (Scheduler Argument)
	    chunk_size (tp.Optional[int]): The size of the token chunk being processed
	        in this prefill step. (Scheduler Argument)
	    prompt_token_ids (tp.Optional[list[int]]): The original list of token IDs
	        for the complete prompt. (Scheduler Argument)
	    length (tp.Optional[jax.Array]): The actual length of the sequence processed
	        so far (relevant for chunked prefill). Defaults to None.
	        (Scheduler Argument)
	"""

	# Runtime Arguments
	token_ids: jax.Array
	positions: jax.Array
	page_indices: jax.Array

	sampling_params: SamplingParams

	# Scheduler Arguments
	id: tp.Optional[str | int]
	chunk_idx: tp.Optional[int]
	chunk_size: tp.Optional[int]
	prompt_token_ids: tp.Optional[list[int]]
	length: tp.Optional[jax.Array] = None

	@property
	def is_active(self):
		"""Checks if the request is active (i.e., has associated token IDs).

		Returns:
		    bool: True if `token_ids` is a non-empty array, False otherwise.
		"""
		return len(self.token_ids.shape) > 0

	def copy_prefill(self, prefill: InitialSequenceRequest):
		"""Copies runtime state from another InitialSequenceRequest (prefill source).

		This method updates the current request's runtime attributes (`token_ids`,
		`positions`, `page_indices`, `sampling_params`, `length`) based on the
		state of a source `prefill` request, typically used when advancing
		through chunks of a long prompt.

		Args:
		    prefill (InitialSequenceRequest): The source request from which to copy
		        runtime state.
		"""
		length = (prefill.chunk_idx + 1) * prefill.chunk_size
		total_length = len(prefill.prompt_token_ids)
		if length > total_length:
			length = total_length

		self.token_ids = prefill.token_ids
		self.positions = prefill.positions
		self.page_indices = np.array(prefill.page_indices)

		self.sampling_params = prefill.sampling_params

		self.length = length

	@classmethod
	def init_empty(cls):
		"""Creates an empty InitialSequenceRequest placeholder.

		Initializes attributes with placeholder scalar JAX arrays suitable for
		use in contexts where a valid request might not be present (e.g., padding).

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
			sampling_params=SamplingParams.init_empty(),
			chunk_size=0,
			page_indices=scalar,
			prompt_token_ids=None,
		)

	@classmethod
	def create(
		cls,
		id: str,
		mesh: common_types.Mesh,
		metadata: PagedAttentionCacheMetaData,
		chunk_size: int,
		prompt_token_ids: list[int],
		max_prefill_length: tp.Optional[int] = None,
		prefill_lengths: tp.Optional[tp.List[int]] = None,
		sampling_params: tp.Optional[SamplingParams] = None,
	):
		"""Creates an InitialSequenceRequest from prompt token IDs.

		This factory method takes a list of token IDs and prepares them for
		the prefill phase, including padding, creating position IDs, initializing
		page indices, and setting up sampling parameters. Arrays are placed
		on the specified JAX mesh.

		Args:
		    mesh (common_types.Mesh): The JAX device mesh for array distribution.
		    metadata (PagedAttentionCacheMetaData): Paged attention cache configuration.
		    chunk_size (int): The size for potential chunking during prefill.
		    prompt_token_ids (list[int]): The input prompt token IDs.
		    prefill_length (tp.Optional[int]): Target length for padding token IDs.
		        Defaults to `metadata.max_sequences`.
		    sampling_params (tp.Optional[SamplingParams]): Custom sampling parameters.
		        If None, default parameters are used.

		Returns:
		    InitialSequenceRequest: An initialized request object ready for prefill.
		"""
		if max_prefill_length is None:
			max_prefill_length = metadata.max_sequences
		if prefill_lengths is None:
			prefill_lengths = DEFAULT_PREFILL_BUCKETS

		prefill_lengths = prefill_lengths[: prefill_lengths.index(max_prefill_length)]
		prefill_lengths = prefill_lengths + [max_prefill_length]

		sequence_length = len(prompt_token_ids)
		near_length = take_nearest_length(prefill_lengths, sequence_length)

		paddlen = near_length - sequence_length

		array = np.array(prompt_token_ids)

		if paddlen < 0:
			padded_token_ids = array[-near_length:]
		else:
			padded_token_ids = np.pad(array, (0, paddlen), constant_values=(0,))

		sharding = Ns(mesh, Ps(None))
		page_indices = [0] * metadata.num_pages_per_sequence
		token_ids = jax.device_put(padded_token_ids, sharding)
		positions = np.arange(0, token_ids.shape[0])
		positions = jax.device_put(positions, sharding)
		if sampling_params is None:
			sampling_params = SamplingParams()

		top_p = jax.device_put(
			np.array([sampling_params.top_p]).reshape(-1),
			sharding,
		)
		max_tokens = jax.device_put(
			np.array([sampling_params.max_tokens + sequence_length]).reshape(-1),
			sharding,
		)
		temperature = jax.device_put(
			np.array([sampling_params.temperature]).reshape(-1),
			sharding,
		)

		return InitialSequenceRequest(
			id=id,
			prompt_token_ids=prompt_token_ids,
			chunk_idx=0,
			chunk_size=chunk_size,
			page_indices=page_indices,
			token_ids=token_ids,
			positions=positions,
			sampling_params=SamplingParams(
				top_p=top_p,
				max_tokens=max_tokens,
				temperature=temperature,
			),
		)


@auto_pytree
class AllocatedPrefillPages:
	"""Holds the indices of HBM pages allocated during a prefill step.

	This simple structure is used to communicate which physical memory pages
	have been assigned to a sequence during its prefill processing.

	Attributes:
	    page_indices (list[int]): A list containing the indices of the HBM
	        (High Bandwidth Memory) pages allocated for a specific prefill chunk.
	"""

	page_indices: list[int]


@auto_pytree
class SlotPageAssignment:
	"""Represents the assignment of a physical page to a logical page slot.

	During decoding, as sequences grow, new physical memory pages might be
	allocated. This class represents the update instruction to map a specific
	logical page index within a sequence's page table (identified by its `slot`)
	to a newly allocated physical HBM page index (`mapped_idx`).

	Attributes:
	    slot (int): The batch slot index of the sequence whose page table is updated.
	    page_idx (int): The logical page index within the sequence's page table.
	    mapped_idx (int): The physical HBM page index to map the logical page to.
	"""

	slot: int
	page_idx: int
	mapped_idx: int


@auto_pytree
class GenerationStepTask:
	"""Represents a sequence actively undergoing token generation (decoding).

	This class holds the necessary information for a single sequence that is
	currently in the decoding phase within the paged attention batch.

	Attributes:
	    id (str): The unique identifier tracing back to the original request.
	    slot (int): The assigned batch slot index for this sequence.
	    position (int): The current sequence length (position) for the next token.
	    page_indices (list[int]): The list of physical HBM page indices allocated
	        to this sequence's KV cache.
	    prefill_token_id (jax.Array): The token ID generated in the *previous* step
	        (either prefill or the last decode step), which serves as input for
	        the *current* decode step.
	    sampling_params (SamplingParams): The sampling parameters associated with
	        this specific generation task.
	"""

	id: str
	slot: int
	position: int
	page_indices: list[int]
	prefill_token_id: jax.Array
	sampling_params: SamplingParams


@auto_pytree
class NextIterationPlan:
	"""Encapsulates the scheduling decisions for the next model iteration.

	Based on available resources (like HBM pages and batch slots) and pending
	requests, the scheduler produces this plan, detailing which prefill and
	decode operations to execute next, along with necessary state updates.

	Attributes:
	    prefill_request (InitialSequenceRequest): The prefill request scheduled for
	        the next iteration. Can be an empty/placeholder request if no prefill
	        is scheduled.
	    schedule_prefill (bool): True if a prefill operation should be executed.
	    schedule_decodes (bool): True if decode operations should be executed.
	    prefill_pages_update (AllocatedPrefillPages): Contains the indices of pages
	        newly allocated for the scheduled prefill request. Can be empty if no
	        new pages were needed or no prefill is scheduled.
	    new_decodes_requests (list[GenerationStepTask]): A list of sequences that
	        are newly transitioning into the decode phase in this iteration (e.g.,
	        after completing prefill).
	    decodes_state_page_updates (list[SlotPageAssignment]): A list of updates
	        to the page tables of sequences already in the decode phase, typically
	        due to new page allocations for them.
	"""

	prefill_request: InitialSequenceRequest

	schedule_prefill: bool
	schedule_decodes: bool

	prefill_pages_update: AllocatedPrefillPages
	new_decodes_requests: list[GenerationStepTask]

	decodes_state_page_updates: list[SlotPageAssignment]


@auto_pytree
class ActiveSequenceBatch:
	"""Manages the batch state for sequences in the decoding phase.

	This class holds the dynamic state for all sequences currently undergoing
	token generation (decoding) within the paged attention batch. It includes
	the input token IDs for the next step, current positions, page table mappings,
	sampling parameters, and structures for managing available batch slots.

	Attributes:
	    token_ids (jax.Array | tp.List[int]): JAX array (during model execution) or
	        list (during host-side updates) holding the input token ID for the
	        *next* decode step for each active slot. Shape: (batch_size,).
	    positions (jax.Array): JAX array holding the current sequence position
	        (length) for each active slot. Inactive slots might have a placeholder
	        value (e.g., -1 or 1e6). Shape: (batch_size,).
	    page_table (jax.Array): JAX array mapping logical page indices to physical
	        HBM page indices for each sequence slot.
	        Shape: (batch_size, num_pages_per_sequence).
	    sampling_params (SamplingParams): Sampling parameters for all sequences
	        in the batch. (Inference Argument)
	    available_slots (tp.Optional[queue.SimpleQueue]): A queue managing indices
	        of free batch slots. Used by the host-side scheduler. (Scheduler Argument)
	    active_slot_requests_map (tp.Optional[tp.Dict[int, GenerationStepTask]]):
	        A dictionary mapping active slot indices to their corresponding
	        GenerationStepTask. Used by the host-side scheduler. (Scheduler Argument)
	    context_lock (tp.Optional[threading.Lock]): A lock for thread-safe updates
	        to scheduler-related attributes (`available_slots`,
	        `active_slot_requests_map`). (Scheduler Argument)
	    page_update_slots (tp.Optional[jax.Array]): Array storing slot indices for
	        pending page table updates. (Internal State)
	    page_update_page_idxs (tp.Optional[jax.Array]): Array storing logical page
	        indices for pending page table updates. (Internal State)
	    page_update_mapped_idxs (tp.Optional[jax.Array]): Array storing physical
	        mapped indices for pending page table updates. (Internal State)
	"""

	token_ids: jax.Array | tp.List[int]
	positions: jax.Array
	page_table: jax.Array

	# Inference Arguments
	sampling_params: SamplingParams

	available_slots: tp.Optional[queue.SimpleQueue]
	active_slot_requests_map: tp.Optional[tp.Dict[int, GenerationStepTask]]
	context_lock: tp.Optional[threading.Lock] = threading.Lock()

	page_update_slots: tp.Optional[jax.Array] = None
	page_update_page_idxs: tp.Optional[jax.Array] = None
	page_update_mapped_idxs: tp.Optional[jax.Array] = None

	@property
	def is_active(self):
		"""Checks if the batch state is active (i.e., has associated token IDs).

		Returns:
		    bool: True if `token_ids` is a non-empty array, False otherwise.
		"""
		return len(self.token_ids.shape) > 0

	def insert_decode_state(self, insert_slots: jax.Array, update: ActiveSequenceBatch):
		"""Updates the decode state in specified slots from another batch state.

		This is typically used during JAX computation to incorporate updates from
		newly scheduled decode tasks or page table modifications.

		Args:
		    insert_slots (jax.Array): An array of slot indices to update.
		    update (ActiveSequenceBatch): The batch containing the new state to insert.
		"""
		update.token_ids = jnp.asarray(update.token_ids)

		self.token_ids = self.token_ids.at[insert_slots].set(update.token_ids)
		self.positions = self.positions.at[insert_slots].set(update.positions)
		self.page_table = self.page_table.at[insert_slots, :].set(update.page_table)
		self.sampling_params.insert_decode_state(insert_slots, update)

		self.page_table = self.page_table.at[
			update.page_update_slots,
			update.page_update_page_idxs,
		].set(update.page_update_mapped_idxs)

	def apply_assignment(self, assignment: list[SlotPageAssignment]):
		"""Applies page table assignments to the internal update arrays (host-side).

		This method populates the `page_update_*` arrays based on a list of
		`SlotPageAssignment` objects, preparing them for later use in JAX
		computations (`insert_decode_state`).

		Args:
		    assignment (list[SlotPageAssignment]): A list of page assignments to apply.
		"""
		assert self.page_update_slots is not None
		assert self.page_update_page_idxs is not None
		assert self.page_update_mapped_idxs is not None

		for i, update in enumerate(assignment):
			self.page_update_slots[i] = update.slot
			self.page_update_page_idxs[i] = update.page_idx
			self.page_update_mapped_idxs[i] = update.mapped_idx

	def pad_tokens(self, pad_length: int):
		"""Pads the host-side token list with placeholder scalars.

		Ensures the `token_ids` list (when used host-side) reaches the required
		`pad_length` by appending placeholder scalar JAX arrays.

		Args:
		    pad_length (int): The target length for the `token_ids` list.
		"""
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		assert isinstance(self.token_ids, list)
		for i in range(pad_length):
			self.token_ids.append(scalar)

	def copy_decode(self, decode: ActiveSequenceBatch):
		"""Copies essential decode state from another ActiveSequenceBatch (host-side).

		Updates the current object's `token_ids`, `positions`, `page_table`, and
		`sampling_params` based on the source `decode` object. Assumes host-side
		operation.

		Args:
		    decode (ActiveSequenceBatch): The source batch state to copy from.
		"""
		self.token_ids = decode.token_ids
		self.positions = decode.positions
		self.page_table = decode.page_table
		self.sampling_params = decode.sampling_params

	def insert_from_task(self, slot: int, task: GenerationStepTask):
		"""Inserts state from a GenerationStepTask into a specific slot (host-side).

		Updates the host-side representations (`token_ids` list, `positions` array,
		`page_table` array, `sampling_params`) for the given `slot` based on the
		provided `task`.

		Args:
		    slot (int): The batch slot index to insert the task state into.
		    task (GenerationStepTask): The task containing the state to insert.
		"""
		assert isinstance(self.token_ids, list)

		self.token_ids.append(task.prefill_token_id)
		self.positions[slot] = task.position
		self.page_table[slot] = np.array(task.page_indices)
		self.sampling_params.insert_from_task(slot, task)

	@classmethod
	def init_numpy(cls, metadata: PagedAttentionCacheMetaData) -> ActiveSequenceBatch:
		"""Initializes ActiveSequenceBatch with NumPy arrays for host-side use.

		Creates the necessary arrays (`positions`, `page_table`, `page_update_*`)
		using NumPy, suitable for manipulation outside JAX computations.
		`token_ids` is initialized as an empty list.

		Args:
		    metadata (PagedAttentionCacheMetaData): Configuration metadata.

		Returns:
		    ActiveSequenceBatch: An initialized batch state object with NumPy arrays.
		"""
		return cls(
			token_ids=[],
			positions=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_table=np.full(
				(metadata.batch_size, metadata.num_pages_per_sequence),
				1e6,
				dtype=np.int32,
			),
			available_slots=None,
			active_slot_requests_map=None,
			context_lock=None,
			sampling_params=SamplingParams.init_numpy(metadata),
			page_update_slots=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_update_page_idxs=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
			page_update_mapped_idxs=np.full((metadata.batch_size,), 1e6, dtype=np.int32),
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
			sampling_params=SamplingParams.init_empty(),
			available_slots=None,
			active_slot_requests_map=None,
			context_lock=None,
			page_update_slots=scalar,
			page_update_page_idxs=scalar,
			page_update_mapped_idxs=scalar,
		)

	@classmethod
	def create(
		cls,
		metadata: PagedAttentionCacheMetaData,
		mesh: common_types.Mesh,
	):
		"""Creates and initializes an ActiveSequenceBatch for JAX execution.

		This factory method sets up the `ActiveSequenceBatch` with JAX arrays
		(`token_ids`, `positions`, `page_table`, `sampling_params`) appropriately
		sharded across the provided `mesh`. It also initializes the host-side
		scheduler components (`available_slots`, `active_slot_requests_map`).

		Args:
		    metadata (PagedAttentionCacheMetaData): Paged attention cache configuration.
		    mesh (common_types.Mesh): The JAX device mesh for array distribution.

		Returns:
		    ActiveSequenceBatch: An initialized batch state ready for JAX operations.
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
			sampling_params=SamplingParams.init_jax(metadata, d1replicate),
			available_slots=slots,
			active_slot_requests_map={},
		)


class ModelInputBatch(xTree):
	"""Consolidated input data for a single model forward pass in paged attention.

	This structure gathers all necessary inputs for the model, potentially
	combining data for both a prefill step and multiple decode steps into
	a single batch structure suitable for the paged attention kernel.

	Attributes:
	    input_ids (jax.Array): Combined token IDs for prefill and decode sequences.
	    positions (jax.Array): Combined position IDs corresponding to `input_ids`.
	    attn_meta (PagedAttentionMetadata): Metadata required by the paged
	        attention kernel, including sequence lengths, block tables, etc.,
	        for both prefill and decode parts of the batch.
	    sampling_params (SamplingParams): Combined sampling parameters for all
	        sequences included in the batch.
	"""

	input_ids: jax.Array
	positions: jax.Array
	attn_meta: PagedAttentionMetadata
	sampling_params: SamplingParams


@auto_pytree
class ModelOutputBatch:
	"""Output generated by the model after a paged attention forward pass.

	Contains the results from the model, separating outputs corresponding to the
	prefill phase (if one was run) and the decode phase.

	Attributes:
	    prefill_complete (jax.Array): Scalar boolean JAX array. True if the prefill
	        operation (if run) generated a completion token (e.g., EOS).
	    decodes_completes (jax.Array): Boolean JAX array. Indicates for each sequence
	        in the decode part of the batch whether a completion token was generated.
	        Shape: (num_decode_sequences,).
	    prefill_token_id (jax.Array): Scalar JAX array. The token ID generated by
	        the prefill step (if run).
	    decodes_token_ids (jax.Array): JAX array. The token IDs generated for each
	        sequence in the decode part of the batch. Shape: (num_decode_sequences,).
	    prefill_next_position (jax.Array): Scalar JAX array. The next position index
	        for the sequence processed in the prefill step.
	    decodes_next_position (jax.Array): JAX array. The next position indices for
	        each sequence in the decode part of the batch. Completed sequences might
	        have a special value (e.g., -1). Shape: (num_decode_sequences,).
	    next_sampling_params (SamplingParams): Updated sampling parameters after the
	        forward pass (e.g., potentially modified max_tokens).
	"""

	prefill_complete: jax.Array
	decodes_completes: jax.Array

	prefill_token_id: jax.Array
	decodes_token_ids: jax.Array

	prefill_next_position: jax.Array
	decodes_next_position: jax.Array

	next_sampling_params: SamplingParams

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
			next_sampling_params=SamplingParams.init_empty(),
		)


@auto_pytree
class ModelOutputSummary:
	"""Summarizes model output for scheduler and state updates (host-side).

	This structure extracts and organizes key information from the `ModelOutputBatch`
	(which contains JAX arrays) into a format suitable for the host-side scheduler
	to process and update its internal state (like `ActiveSequenceBatch`'s
	scheduler attributes).

	Attributes:
	    prefill_request_id (str | None): The ID of the prefill request processed in
	        the last step, if any.
	    prefill_token_id (jax.Array): The token ID generated by the prefill step.
	    prefill_complete (jax.Array): Boolean flag indicating if the prefill step
	        completed the sequence.
	    decodes_active_slots (list[int]): List of batch slot indices that were active
	        during the decode phase of the last step.
	    decodes_active_request_ids (list[str]): List of request IDs corresponding to
	        the `decodes_active_slots`.
	    decodes_token_ids (jax.Array): Token IDs generated for the active decode slots.
	    decodes_completes (jax.Array): Boolean flags indicating completion for each
	        active decode slot.
	"""

	prefill_request_id: str | None
	prefill_token_id: jax.Array
	prefill_complete: jax.Array

	decodes_active_slots: list[int]
	decodes_active_request_ids: list[str]
	decodes_token_ids: jax.Array
	decodes_completes: jax.Array

	@classmethod
	def from_output(cls, output: ModelOutputBatch) -> ModelOutputSummary:
		"""Creates a ModelOutputSummary from a ModelOutputBatch.

		Initializes the summary, copying relevant fields from the model output.
		Scheduler-specific fields (`prefill_request_id`, `decodes_active_slots`,
		`decodes_active_request_ids`) are initialized as empty/None and need to
		be populated separately by the scheduler based on its knowledge of the
		batch composition.

		Args:
		    output (ModelOutputBatch): The model output batch containing JAX arrays.

		Returns:
		    ModelOutputSummary: An initialized summary object ready for scheduler processing.
		"""
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
