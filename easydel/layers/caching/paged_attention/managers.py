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

import math
import queue
import threading
import typing as tp
import uuid
from functools import partial

import jax
import numpy as np
from eformer import common_types
from eformer import escale as es
from eformer.pytree import auto_pytree, xTree
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps

from easydel.utils.compiling_utils import cjit

from .paged_attention_cache import PagedAttentionCacheMetaData, PagedAttentionMetadata


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
	    temperature (jax.Array): The temperature parameter for sampling, as a JAX array.
	    top_p (jax.Array): The top-p (nucleus) sampling parameter, as a JAX array.
	    top_k (jax.Array): The top-k sampling parameter, as a JAX array.
	    chunk_size (int): The size of the chunk being processed in this prefill step.
	    page_indices (list[int]): A list to hold the indices of the HBM pages allocated
	        for this request's KV cache. Initialized with zeros.
	    prompt_token_ids (list[int]): The original list of token IDs for the complete prompt.
	"""

	id: str | int

	chunk_idx: int
	token_ids: jax.Array
	positions: jax.Array

	temperature: jax.Array
	top_p: jax.Array
	top_k: jax.Array

	chunk_size: int

	page_indices: list[int]
	prompt_token_ids: list[int]

	@classmethod
	def create_from_prompt_ids(
		cls,
		mesh: common_types.Mesh,
		metadata: PagedAttentionCacheMetaData,
		chunk_size: int,
		prompt_token_ids: list[int],
		prefill_length: tp.Optional[int] = None,
		temperature: float = 0.7,
		top_p: float = 0.95,
		top_k: int = 30,
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
		    temperature (float): Sampling temperature. Defaults to 0.7.
		    top_p (float): Nucleus sampling p-value. Defaults to 0.95.
		    top_k (int): Top-k sampling k-value. Defaults to 30.

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

		page_indices = [0] * metadata.num_pages_per_sequence
		token_ids = jax.device_put(padded_token_ids, Ns(mesh, Ps(None)))
		positions = np.arange(0, token_ids.shape[0])
		positions = jax.device_put(positions, Ns(mesh, Ps(None)))

		temperature = jax.device_put(jnp.asarray([temperature]), Ns(mesh, Ps(None)))

		top_p = jax.device_put(jnp.asarray([top_p]), Ns(mesh, Ps(None)))
		top_k = jax.device_put(jnp.asarray([top_k]), Ns(mesh, Ps(None)))

		return InitialSequenceRequest(
			id=uuid.uuid4(),
			prompt_token_ids=prompt_token_ids,
			chunk_idx=0,
			chunk_size=chunk_size,
			page_indices=page_indices,
			token_ids=token_ids,
			positions=positions,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
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
	    temperature (float): The sampling temperature for this request.
	    top_p (float): The nucleus sampling p-value for this request.
	    top_k (int): The top-k sampling k-value for this request.
	"""

	id: str
	slot: int
	position: int
	page_indices: list[int]
	prefill_token_id: jax.Array
	temperature: float
	top_p: float
	top_k: int


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
	    temperature (jax.Array): JAX array of sampling temperatures for each active slot. Shape: (batch_size,).
	    top_p (jax.Array): JAX array of top-p sampling parameters for each active slot. Shape: (batch_size,).
	    top_k (jax.Array): JAX array of top-k sampling parameters for each active slot. Shape: (batch_size,).
	    available_slots (queue.SimpleQueue): A queue holding the indices of batch slots that are currently free.
	    active_slot_requests_map (dict[int, GenerationStepTask]): A dictionary mapping active slot indices
	        to their corresponding GenerationStepTask objects.
	    context_lock (threading.Lock): A lock to ensure thread-safe access and modification of the decode state,
	        particularly `available_slots` and `active_slot_requests_map`.
	"""

	token_ids: jax.Array
	positions: jax.Array
	page_table: jax.Array

	temperature: jax.Array
	top_p: jax.Array
	top_k: jax.Array

	available_slots: queue.SimpleQueue
	active_slot_requests_map: dict[int, GenerationStepTask]
	context_lock: threading.Lock = threading.Lock()

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
			temperature=jnp.zeros(
				shape=(metadata.batch_size,),
				dtype=jnp.float32,
				device=d1replicate,
			),
			top_p=jnp.zeros(
				shape=(metadata.batch_size,),
				dtype=jnp.float32,
				device=d1replicate,
			),
			top_k=jnp.zeros(
				shape=(metadata.batch_size,),
				dtype=jnp.int32,
				device=d1replicate,
			),
			available_slots=slots,
			active_slot_requests_map={},
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
	    temperature (jax.Array): Combined sampling temperatures.
	    top_p (jax.Array): Combined top-p sampling parameters.
	    top_k (jax.Array): Combined top-k sampling parameters.
	"""

	input_ids: jax.Array
	positions: jax.Array
	attn_meta: PagedAttentionMetadata
	temperature: jax.Array
	top_p: jax.Array
	top_k: jax.Array


@auto_pytree
class ModelOutputBatch:
	"""
	Represents the output generated by the model after a paged attention forward pass,
	separating results for prefill and decode phases.

	Attributes:
	    prefill_done (jax.Array): A scalar boolean JAX array indicating if the prefill operation
	        (if scheduled) resulted in a completed sequence (e.g., EOS token).
	    decodes_done (jax.Array): A boolean JAX array indicating for each decode slot whether
	        the generated token completed the sequence. Shape: (num_decode_requests,).
	    prefill_tokens (jax.Array): The token ID generated by the prefill operation (if scheduled). Scalar.
	    decodes_tokens (jax.Array): The token IDs generated for each decode slot (if scheduled).
	        Shape: (num_decode_requests,).
	    prefill_next_position (jax.Array): The next position index for the prefill request after this step. Scalar.
	    decodes_next_position (jax.Array): The next position indices for each decode request after this step.
	        Shape: (num_decode_requests,). Contains -1 for completed sequences.
	"""

	prefill_done: jax.Array
	decodes_done: jax.Array
	prefill_tokens: jax.Array
	decodes_tokens: jax.Array
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
			prefill_done=scalar,
			decodes_done=vector,
			prefill_tokens=scalar,
			decodes_tokens=vector,
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
	    prefill_done (jax.Array): Flag indicating if the prefill step completed the sequence.
	    decodes_active_slots (list[int]): List of slot indices that were active during the decode phase.
	    decodes_active_request_ids (list[str]): List of request IDs corresponding to the active decode slots.
	    decodes_token_ids (jax.Array): Tokens generated for the active decode slots.
	    decodes_done (jax.Array): Flags indicating completion for each active decode slot.
	"""

	prefill_request_id: str | None
	prefill_token_id: jax.Array
	prefill_done: jax.Array

	decodes_active_slots: list[int]
	decodes_active_request_ids: list[str]
	decodes_token_ids: jax.Array
	decodes_done: jax.Array


class HBMPageManager:
	"""
	Manages the allocation and deallocation of physical HBM pages for the KV cache.
	It keeps track of available pages.

	Attributes:
	    _metadata (PagedAttentionCacheMetaData): Configuration for the paged cache.
	    _current_page_index (int): Index representing the "dummy" or initial page (usually 0).
	    _available_hbm_pages (queue.SimpleQueue): Queue holding the indices of free HBM pages.
	"""

	def __init__(self, metadata: PagedAttentionCacheMetaData):
		"""
		Initializes the HBMPageManager.

		Args:
		    metadata (PagedAttentionCacheMetaData): Configuration metadata for the paged cache,
		        including total number of pages, page size, etc.
		"""
		self._metadata = metadata
		self._current_page_index = 0
		self._available_hbm_pages = queue.SimpleQueue()
		for p in range(1, metadata.num_pages_per_layer):
			self._available_hbm_pages.put_nowait(p)

	@property
	def metadata(self) -> PagedAttentionCacheMetaData:
		"""Returns the cache metadata."""
		return self._metadata

	@property
	def page_size(self):
		"""Returns the page size in the number of per-token kv cache items."""
		return self._metadata.page_size

	@property
	def current_page_index(self):
		"""Returns the dummy page index (usually 0)."""
		return self._current_page_index

	def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
		"""
		Allocates the required number of HBM pages for a prompt prefill based on its length.

		Args:
		    prompt_len (int): The length of the prompt (or prompt chunk).

		Returns:
		    list[int]: A list of allocated HBM page indices, or an empty list if
		        not enough pages are available.
		"""
		n = math.ceil(prompt_len / self._metadata.page_size)
		return self.alloc_hbm_pages(n)

	def alloc_hbm_pages(self, n: int) -> list[int]:
		"""
		Allocates a specific number of HBM pages.

		Args:
		    n (int): The number of pages to allocate.

		Returns:
		    list[int]: A list of allocated HBM page indices, or an empty list if
		        fewer than `n` pages are available.
		"""
		if 0 < n <= self._available_hbm_pages.qsize():
			return [self._available_hbm_pages.get(block=True) for _ in range(n)]
		else:
			return []

	def free_hbm_pages(self, pages: list[int]):
		"""
		Returns a list of HBM pages to the available pool.

		Args:
		    pages (list[int]): A list of HBM page indices to free. Pages matching
		        `current_page_index` are ignored.
		"""
		for p in pages:
			if p != self._current_page_index:
				self._available_hbm_pages.put_nowait(p)


class InferenceScheduler:
	"""
	Schedules incoming prefill and decode requests based on available resources (HBM pages, batch slots).
	It interacts with the HBMPageManager and ActiveSequenceBatch.

	Attributes:
	    prefill_queue (queue.Queue[InitialSequenceRequest]): Queue for incoming prefill requests.
	    decodes_queue (queue.Queue[GenerationStepTask]): Queue for requests transitioning from prefill
	        to decode.
	    manager (HBMPageManager): The manager responsible for HBM page allocation.
	    batch_size (int): The maximum number of requests processed concurrently in the decode phase.
	    max_seq_len (int): The maximum sequence length supported.
	"""

	def __init__(self, manager: HBMPageManager):
		"""
		Initializes the InferenceScheduler.

		Args:
		    manager (HBMPageManager): The cache manager instance.
		"""
		self.prefill_queue: queue.Queue[InitialSequenceRequest] = queue.Queue()
		self.decodes_queue: queue.Queue[GenerationStepTask] = queue.Queue()
		self.manager = manager
		self.batch_size = manager.metadata.batch_size
		self.max_seq_len = manager.metadata.max_sequences

	def enqueue_prefill_req(self, req: InitialSequenceRequest):
		"""Adds a new prefill request to the queue."""
		self.prefill_queue.put(req)

	def enqueue_decodes_req(self, req: GenerationStepTask):
		"""Adds a request (that finished prefill) to the decode queue."""
		self.decodes_queue.put(req)

	def schedule(
		self,
		active_prefill: InitialSequenceRequest | None,
		decodes_state: ActiveSequenceBatch,
	) -> NextIterationPlan | None:
		"""
		Determines the workload for the next model iteration.

		This function decides:
		1. If a new prefill request can be started or an existing one continued.
		2. If HBM pages need to be allocated for the prefill request.
		3. Which decode requests from the `decodes_queue` can be added to the active batch.
		4. If HBM pages need to be allocated for ongoing decode requests.

		It updates the host-side state (like `decodes_state.active_slot_requests_map`)
		but prepares updates for the device-side state (JAX arrays) within the returned
		`NextIterationPlan`.

		Args:
		    active_prefill (InitialSequenceRequest | None): The prefill request currently being processed,
		        if any.
		    decodes_state (ActiveSequenceBatch): The current state of the decode batch.

		Returns:
		    NextIterationPlan | None: An object containing the scheduling decisions and necessary updates
		        for the next iteration, or None if no work can be scheduled (e.g., all queues empty).

		Raises:
		    NotImplementedError: If page allocation fails and eviction is required (currently not supported).
		"""
		avail_slot_size = decodes_state.available_slots.qsize()
		next_prefill_req = active_prefill
		prefill_pages_update = None
		next_decodes_reqs = []
		decodes_state_page_updates = []

		schedule_prefill = False
		schedule_decodes = False

		if not next_prefill_req:
			if avail_slot_size > 0:
				try:
					next_prefill_req = self.prefill_queue.get_nowait()
					if not next_prefill_req:
						return None
				except queue.Empty:
					pass

		if next_prefill_req:
			cur_prompt_chunk_len = next_prefill_req.chunk_size
			total_len = len(next_prefill_req.prompt_token_ids)
			if total_len <= (next_prefill_req.chunk_idx + 1) * next_prefill_req.chunk_size:
				cur_prompt_chunk_len = (
					total_len - next_prefill_req.chunk_idx * next_prefill_req.chunk_size
				)
			alloced_pages = self.manager.alloc_prefill_hbm_pages(cur_prompt_chunk_len)
			if len(alloced_pages) == 0:
				raise NotImplementedError("Eviction is not supported yet")
			else:
				start_idx = (
					next_prefill_req.chunk_idx * next_prefill_req.chunk_size
				) // self.manager.page_size
				for i, page in enumerate(alloced_pages):
					next_prefill_req.page_indices[start_idx + i] = page
				prefill_pages_update = AllocatedPrefillPages(alloced_pages)

		with decodes_state.context_lock:
			if not next_prefill_req or (
				len(decodes_state.active_slot_requests_map) + self.decodes_queue.qsize()
				> 0.95 * self.batch_size
			):
				# Add new generate request to the slots.
				while (
					decodes_state.available_slots.qsize() > 0 and self.decodes_queue.qsize() > 0
				):
					gr = self.decodes_queue.get_nowait()
					if not gr:
						return None
					slot = decodes_state.available_slots.get_nowait()
					gr.slot = slot
					decodes_state.active_slot_requests_map[slot] = gr
					next_decodes_reqs.append(gr)

				alloced_pages = self.manager.alloc_hbm_pages(
					len(decodes_state.active_slot_requests_map)
				)
				if len(decodes_state.active_slot_requests_map) != 0 and len(alloced_pages) == 0:
					raise NotImplementedError(
						"Eviction isn't supported yet, please set a lower value for batch_size"
					)

				page_to_use = 0
				for slot, req in decodes_state.active_slot_requests_map.items():
					idx = req.position // self.manager.page_size
					if req.position % self.manager.page_size != 0:
						continue
					if idx >= len(req.page_indices):
						continue

					req.page_indices[idx] = alloced_pages[page_to_use]
					decodes_state_page_updates.append(
						SlotPageAssignment(
							slot=slot,
							page_idx=idx,
							mapped_idx=alloced_pages[page_to_use],
						)
					)
					page_to_use += 1

				self.manager.free_hbm_pages(alloced_pages[page_to_use:])

				if len(decodes_state.active_slot_requests_map) == 0:
					schedule_decodes = False
				else:
					schedule_decodes = True

		if next_prefill_req:
			schedule_prefill = True
		else:
			schedule_prefill = False

		if not schedule_prefill and not schedule_decodes:
			while True:
				if self.prefill_queue.qsize() > 0 or self.decodes_queue.qsize() > 0:
					return self.schedule(active_prefill, decodes_state)

		req = NextIterationPlan(
			schedule_prefill=schedule_prefill,
			schedule_decodes=schedule_decodes,
			prefill_request=next_prefill_req,
			prefill_pages_update=prefill_pages_update,
			new_decodes_requests=next_decodes_reqs,
			decodes_state_page_updates=decodes_state_page_updates,
		)
		return req


class ModelIOProcessor:
	@staticmethod
	def prepare_model_output(
		token_ids: jax.Array,
		complete: jax.Array,
		attn_meta: PagedAttentionMetadata,
	) -> ModelOutputBatch:
		"""
		Processes the raw model output (logits converted to tokens) and completion flags
		to create a structured ModelOutputBatch object.

		It separates the outputs corresponding to the prefill step (if any) and the
		decode steps (if any) based on the structure defined in `attn_meta`.

		Args:
		    token_ids (jax.Array): The generated token IDs from the model.
		        Concatenated prefill token (if any) and decode tokens (if any).
		    complete (jax.Array): Boolean flags indicating sequence completion (e.g., EOS token)
		        for the corresponding generated tokens.
		    attn_meta (PagedAttentionMetadata): Metadata describing the structure of the input batch
		        (prefill vs. decode parts).

		Returns:
		    ModelOutputBatch: A structured object containing separated prefill/decode outputs
		        and next positions.
		"""
		output = ModelOutputBatch.init_empty()

		has_prefill = False
		has_generate = False
		if len(attn_meta.prefill_position.shape) != 0:
			has_prefill = True
		if len(attn_meta.decodes_position.shape) != 0:
			has_generate = True

		if has_prefill and not has_generate:
			output.prefill_tokens = token_ids[0]
			output.prefill_done = complete[0]
			output.prefill_next_position = attn_meta.prefill_length

		if not has_prefill and has_generate:
			output.decodes_tokens = token_ids
			output.decodes_done = complete
			output.decodes_next_position = jnp.where(
				output.decodes_done,
				-1,
				attn_meta.decodes_position + 1,
			)
			output.decodes_next_position = jnp.where(
				output.decodes_next_position,
				output.decodes_next_position,
				-1,
			)

		if has_prefill and has_generate:
			output.prefill_done = complete[0]
			output.prefill_tokens = token_ids[0]
			output.prefill_next_position = attn_meta.prefill_length

			output.decodes_tokens = token_ids[1:]
			output.decodes_done = complete[1:]
			output.decodes_next_position = jnp.where(
				output.decodes_done,
				-1,
				attn_meta.decodes_position + 1,
			)
			output.decodes_next_position = jnp.where(
				output.decodes_next_position,
				output.decodes_next_position,
				-1,
			)

		return output

	@staticmethod
	@partial(cjit, static_argnums=(7,))
	@partial(jax.jit, static_argnums=(7,))
	def prepare_model_input(
		attn_meta: PagedAttentionMetadata,
		prefill_tokens: jax.Array,
		prefill_positions: jax.Array,
		prefill_page_indices: jax.Array,
		prefill_sample_params: jax.Array,
		prefill_length: jax.Array,
		chunk_id: jax.Array,
		chunk_size: jax.Array,
		decodes_tokens: list[jax.Array],
		decodes_positions: jax.Array,
		decodes_page_table: jax.Array,
		decodes_sample_params: jax.Array,
		update_decodes_tokens: jax.Array,
		update_decodes_positions: jax.Array,
		update_decodes_page_table: jax.Array,
		update_decodes_sample_params: jax.Array,
		insert_slots: jax.Array,
		decodes_page_updates: jax.Array,
		decodes_pt_update_slots: jax.Array,
		decodes_pt_update_page_idxs: jax.Array,
	) -> tp.Tuple[
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		jax.Array,
		PagedAttentionMetadata,
	]:
		"""
		(JIT-compiled) Takes potentially scheduled prefill and decode data, applies updates
		(like adding new decode requests or updating page tables), and combines them into
		a single set of inputs suitable for the model's forward pass.

		Handles dynamic slicing for prefill chunks and updates decode state arrays based on
		scheduling decisions.

		Args:
		    attn_meta: Initial PagedAttentionMetadata (will be updated).
		    prefill_tokens: Full token sequence for the prefill request.
		    prefill_positions: Corresponding positions for prefill_tokens.
		    prefill_page_indices: Page table for the prefill request.
		    prefill_sample_params: Tuple of (temperature, top_p, top_k) for prefill.
		    prefill_length: Current length of the prefill sequence being processed.
		    chunk_id: Index of the current prefill chunk.
		    chunk_size: Size of the prefill chunk (static for JIT).
		    decodes_tokens: Current tokens for active decode slots.
		    decodes_positions: Current positions for active decode slots.
		    decodes_page_table: Current page table for active decode slots.
		    decodes_sample_params: Tuple of (temperature, top_p, top_k) for active decode slots.
		    update_decodes_tokens: New tokens for newly added decode requests.
		    update_decodes_positions: Positions for newly added decode requests.
		    update_decodes_page_table: Page tables for newly added decode requests.
		    update_decodes_sample_params: Sampling params for newly added decode requests.
		    insert_slots: Indices where new decode requests should be inserted in the batch.
		    decodes_page_updates: New HBM page indices to update in the decode page table.
		    decodes_pt_update_slots: Batch slots corresponding to page table updates.
		    decodes_pt_update_page_idxs: Logical page indices within slots to update.

		Returns:
		    tuple: Contains the combined inputs for the model:
		        - input_ids (jax.Array): Combined token IDs.
		        - decodes_tokens (jax.Array): Updated decode tokens array.
		        - positions (jax.Array): Combined position IDs.
		        - temperature (jax.Array): Combined temperature values.
		        - top_p (jax.Array): Combined top_p values.
		        - top_k (jax.Array): Combined top_k values.
		        - decodes_temperature (jax.Array): Updated decode temperature array.
		        - decodes_top_p (jax.Array): Updated decode top_p array.
		        - decodes_top_k (jax.Array): Updated decode top_k array.
		        - attn_meta (PagedAttentionMetadata): Updated metadata reflecting the combined input.
		"""
		(
			prefill_temperature,
			prefill_top_p,
			prefill_top_k,
		) = prefill_sample_params
		(
			decodes_temperature,
			decodes_top_p,
			decodes_top_k,
		) = decodes_sample_params
		(
			update_decodes_temperature,
			update_decodes_top_p,
			update_decodes_top_k,
		) = update_decodes_sample_params
		if len(prefill_tokens.shape) > 0:
			idx = chunk_id * chunk_size
			prefill_tokens = jax.lax.dynamic_slice_in_dim(
				prefill_tokens,
				idx,
				chunk_size,
			)
			prefill_positions = jax.lax.dynamic_slice_in_dim(
				prefill_positions,
				idx,
				chunk_size,
			)

		if len(decodes_tokens.shape) > 0:
			update_decodes_tokens = jnp.asarray(update_decodes_tokens)

			decodes_tokens = decodes_tokens.at[insert_slots].set(update_decodes_tokens)
			decodes_positions = decodes_positions.at[insert_slots].set(
				update_decodes_positions
			)
			decodes_temperature = decodes_temperature.at[insert_slots].set(
				update_decodes_temperature
			)
			decodes_top_k = decodes_top_k.at[insert_slots].set(update_decodes_top_k)
			decodes_top_p = decodes_top_p.at[insert_slots].set(update_decodes_top_p)
			decodes_page_table = decodes_page_table.at[insert_slots, :].set(
				update_decodes_page_table
			)
			decodes_page_table = decodes_page_table.at[
				decodes_pt_update_slots,
				decodes_pt_update_page_idxs,
			].set(decodes_page_updates)

		if len(prefill_tokens.shape) > 0 and len(decodes_tokens.shape) > 0:
			input_ids = jnp.concatenate((prefill_tokens, decodes_tokens))
			positions = jnp.concatenate((prefill_positions, decodes_positions))

			attn_meta.prefill_length = prefill_length
			attn_meta.prefill_position = prefill_positions
			attn_meta.prefill_page_table = prefill_page_indices
			attn_meta.decodes_position = decodes_positions
			attn_meta.decodes_page_table = decodes_page_table

		elif len(prefill_tokens.shape) > 0:
			input_ids = prefill_tokens
			positions = prefill_positions
			temperature = prefill_temperature
			top_p = prefill_top_p
			top_k = prefill_top_k

			attn_meta.prefill_length = prefill_length
			attn_meta.prefill_position = prefill_positions
			attn_meta.prefill_page_table = prefill_page_indices

		elif len(decodes_tokens.shape) > 0:
			input_ids = decodes_tokens
			positions = decodes_positions
			temperature = decodes_temperature
			top_p = decodes_top_p
			top_k = decodes_top_k

			attn_meta.decodes_position = decodes_positions
			attn_meta.decodes_page_table = decodes_page_table

		else:
			raise ValueError(
				"Failed to build the input as no prefill or generate gets scheduled"
			)

		return (
			input_ids,
			decodes_tokens,
			positions,
			temperature,
			top_p,
			top_k,
			decodes_temperature,
			decodes_top_p,
			decodes_top_k,
			attn_meta,
		)

	@classmethod
	def build_input(
		cls,
		schedule: NextIterationPlan,
		metadata: PagedAttentionCacheMetaData,
		decodes_state: ActiveSequenceBatch,
	):
		"""
		Orchestrates the preparation of model inputs based on a scheduling decision.

		It retrieves the necessary data from the `schedule` and `decodes_state`,
		formats it (converting lists/numpy arrays to JAX arrays as needed), and
		calls `prepare_model_input` to perform the JIT-compiled merging and updating.
		It also updates the `decodes_state` (device arrays) based on the results
		from `prepare_model_input`.

		Args:
		    cls: The class itself.
		    schedule (NextIterationPlan): The output of the scheduler, indicating what to run.
		    metadata (PagedAttentionCacheMetaData): Cache configuration.
		    decodes_state (ActiveSequenceBatch): The current state of the decode batch.

		Returns:
		    ModelInputBatch: The structured input ready for the model's forward pass.
		"""
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)

		attn_meta = PagedAttentionMetadata(
			prefill_length=scalar,
			prefill_position=scalar,
			prefill_page_table=scalar,
			decodes_position=scalar,
			decodes_page_table=scalar,
		)

		chunk_id = scalar
		chunk_size = 512
		prefill_length = scalar

		(
			prefill_tokens,
			prefill_positions,
			prefill_page_indices,
			prefill_temperature,
			prefill_top_p,
			prefill_top_k,
		) = (scalar, scalar, scalar, scalar, scalar, scalar)

		(
			decodes_tokens,
			decodes_positions,
			decodes_page_table,
			decodes_temperature,
			decodes_top_p,
			decodes_top_k,
		) = (scalar, scalar, scalar, scalar, scalar, scalar)

		(
			update_decodes_tokens,
			update_decodes_positions,
			update_decodes_page_table,
			update_decodes_temperature,
			update_decodes_top_p,
			update_decodes_top_k,
		) = (scalar, scalar, scalar, scalar, scalar, scalar)

		insert_slots = scalar
		decodes_page_updates = scalar
		decodes_pt_update_slots = scalar
		decodes_pt_update_page_idxs = scalar

		if schedule.schedule_prefill:
			prefill = schedule.prefill_request
			(
				prefill_tokens,
				prefill_positions,
				prefill_page_indices,
				prefill_temperature,
				prefill_top_p,
				prefill_top_k,
			) = (
				prefill.token_ids,
				prefill.positions,
				np.array(prefill.page_indices),
				prefill.temperature,
				prefill.top_p,
				prefill.top_k,
			)
			prefill_length = (prefill.chunk_idx + 1) * prefill.chunk_size
			prefill_total_len = len(prefill.prompt_token_ids)
			if prefill_length > prefill_total_len:
				prefill_length = prefill_total_len

			chunk_id = prefill.chunk_idx
			chunk_size = prefill.chunk_size

		if schedule.schedule_decodes:
			(
				decodes_tokens,
				decodes_positions,
				decodes_page_table,
				decodes_temperature,
				decodes_top_p,
				decodes_top_k,
			) = (
				decodes_state.token_ids,
				decodes_state.positions,
				decodes_state.page_table,
				decodes_state.temperature,
				decodes_state.top_p,
				decodes_state.top_k,
			)
			update_decodes_tokens = []
			update_decodes_positions = np.full((metadata.batch_size,), 1e6, dtype=np.int32)
			update_decodes_page_table = np.full(
				(metadata.batch_size, metadata.num_pages_per_sequence),
				1e6,
				dtype=np.int32,
			)

			update_decodes_temperature = np.full(
				(metadata.batch_size,),
				1e6,
				dtype=np.float32,
			)
			update_decodes_top_p = np.full((metadata.batch_size,), 1e6, dtype=np.float32)
			update_decodes_top_k = np.full((metadata.batch_size,), 1e6, dtype=np.int32)
			slots = np.full((metadata.batch_size,), 1e6, dtype=np.int32)

			for i, gr in enumerate(schedule.new_decodes_requests):
				update_decodes_tokens.append(gr.prefill_token_id)
				update_decodes_positions[i] = gr.position
				update_decodes_page_table[i] = np.array(gr.page_indices)
				update_decodes_temperature[i] = gr.temperature[0]
				update_decodes_top_p[i] = gr.top_p[0]
				update_decodes_top_k[i] = gr.top_k[0]
				slots[i] = gr.slot

			for i in range(metadata.batch_size - len(schedule.new_decodes_requests)):
				update_decodes_tokens.append(scalar)

			insert_slots = slots

			page_update_slots = np.full((metadata.batch_size,), 1e6, dtype=np.int32)
			page_update_page_idxs = np.full((metadata.batch_size,), 1e6, dtype=np.int32)
			page_update_mapped_idxs = np.full((metadata.batch_size,), 1e6, dtype=np.int32)

			for i, update in enumerate(schedule.decodes_state_page_updates):
				page_update_slots[i] = update.slot
				page_update_page_idxs[i] = update.page_idx
				page_update_mapped_idxs[i] = update.mapped_idx

			decodes_page_updates = page_update_mapped_idxs
			decodes_pt_update_slots = page_update_slots
			decodes_pt_update_page_idxs = page_update_page_idxs

		prefill_sample_params = (
			prefill_temperature,
			prefill_top_p,
			prefill_top_k,
		)
		decodes_sample_params = (
			decodes_temperature,
			decodes_top_p,
			decodes_top_k,
		)
		update_decodes_sample_params = (
			update_decodes_temperature,
			update_decodes_top_p,
			update_decodes_top_k,
		)
		(
			input_ids,
			decodes_tokens,
			positions,
			temperature,
			top_p,
			top_k,
			decodes_temperature,
			decodes_top_p,
			decodes_top_k,
			attn_meta,
		) = cls.prepare_model_input(
			attn_meta,
			prefill_tokens,
			prefill_positions,
			prefill_page_indices,
			prefill_sample_params,
			prefill_length,
			chunk_id,
			chunk_size,
			decodes_tokens,
			decodes_positions,
			decodes_page_table,
			decodes_sample_params,
			update_decodes_tokens,
			update_decodes_positions,
			update_decodes_page_table,
			update_decodes_sample_params,
			insert_slots,
			decodes_page_updates,
			decodes_pt_update_slots,
			decodes_pt_update_page_idxs,
		)

		if schedule.schedule_decodes:
			decodes_state.token_ids = decodes_tokens
			decodes_state.positions = attn_meta.decodes_position
			decodes_state.page_table = attn_meta.decodes_page_table

			decodes_state.temperature = decodes_temperature
			decodes_state.top_p = decodes_top_p
			decodes_state.top_k = decodes_top_k

		return ModelInputBatch(
			input_ids=input_ids,
			positions=positions,
			attn_meta=attn_meta,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
		)


__all__ = (
	"InferenceScheduler",
	"SlotPageAssignment",
	"HBMPageManager",
	"GenerationStepTask",
	"ActiveSequenceBatch",
	"InitialSequenceRequest",
	"ModelIOProcessor",
	"AllocatedPrefillPages",
	"ModelInputBatch",
	"ModelOutputBatch",
	"ModelOutputSummary",
	"NextIterationPlan",
)
