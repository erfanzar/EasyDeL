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
from functools import partial

import jax
import numpy as np
from eformer import escale as es
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps

from easydel.utils.compiling_utils import cjit

from .paged_attention_cache import PagedAttentionCacheMetaData, PagedAttentionMetadata
from .types import (
	ActiveSequenceBatch,
	AllocatedPrefillPages,
	GenerationStepTask,
	InitialSequenceRequest,
	ModelInputBatch,
	ModelOutputBatch,
	NextIterationPlan,
	SamplingParams,
	SlotPageAssignment,
)


class HBMPageManager:
	"""
	Manages the allocation and deallocation of physical HBM pages for the KV cache.
	It keeps track of available pages.

	Attributes:
	    _metadata (PagedAttentionCacheMetaData): Configuration for the paged cache.
	    _current_page_index (int): Index representing the initial dummy page.
	    _available_hbm_pages (queue.SimpleQueue): Queue of free HBM page indices.
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
		"""Number of per-token KV cache items per page."""
		return self._metadata.page_size

	@property
	def current_page_index(self):
		"""Returns the dummy page index (usually 0)."""
		return self._current_page_index

	def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
		"""
		Allocates the required number of HBM pages for a prompt prefill based on its length.

		Args:
		    prompt_len (int): The length of the prompt (or chunk).

		Returns:
		    list[int]: List of allocated HBM page indices (empty if insufficient pages).
		"""
		n = math.ceil(prompt_len / self._metadata.page_size)
		return self.alloc_hbm_pages(n)

	def alloc_hbm_pages(self, n: int) -> list[int]:
		"""
		Allocates a specific number of HBM pages.

		Args:
		    n (int): Number of pages to allocate.

		Returns:
		    list[int]: Allocated HBM page indices (empty if insufficient pages).
		"""
		if 0 < n <= self._available_hbm_pages.qsize():
			return [self._available_hbm_pages.get(block=True) for _ in range(n)]
		else:
			return []

	def free_hbm_pages(self, pages: list[int]):
		"""
		Returns a list of HBM pages back to the available pool.

		Args:
		    pages (list[int]): HBM page indices to free (ignores dummy page).
		"""
		for p in pages:
			if p != self._current_page_index:
				self._available_hbm_pages.put_nowait(p)


class InferenceScheduler:
	"""
	Schedules incoming prefill and decode requests based on HBM page and slot availability.
	It coordinates with HBMPageManager and ActiveSequenceBatch.

	Attributes:
	    prefill_queue (queue.Queue[InitialSequenceRequest]): Incoming prefill requests queue.
	    decodes_queue (queue.Queue[GenerationStepTask]): Prefill-to-decode transition queue.
	    manager (HBMPageManager): Manager for HBM page allocation.
	    batch_size (int): Max concurrent decode requests.
	    max_seq_len (int): Max sequence length supported.
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

	def enqueue_prefill_request(self, request: InitialSequenceRequest):
		"""Adds a prefill request to the prefill queue."""
		self.prefill_queue.put(request)

	def enqueue_decodes_request(self, request: GenerationStepTask):
		"""Adds a completed prefill request to the decode queue."""
		self.decodes_queue.put(request)

	def create_plan(
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
				while (
					decodes_state.available_slots.qsize() > 0 and self.decodes_queue.qsize() > 0
				):
					decode_request = self.decodes_queue.get_nowait()
					if not decode_request:
						return None
					slot = decodes_state.available_slots.get_nowait()
					decode_request.slot = slot
					decodes_state.active_slot_requests_map[slot] = decode_request
					next_decodes_reqs.append(decode_request)

				alloced_pages = self.manager.alloc_hbm_pages(
					len(decodes_state.active_slot_requests_map)
				)
				if len(decodes_state.active_slot_requests_map) != 0 and len(alloced_pages) == 0:
					raise NotImplementedError(
						"Eviction isn't supported yet, please set a lower value for batch_size"
					)

				page_to_use = 0
				for slot, request in decodes_state.active_slot_requests_map.items():
					idx = request.position // self.manager.page_size
					if request.position % self.manager.page_size != 0:
						continue
					if idx >= len(request.page_indices):
						continue

					request.page_indices[idx] = alloced_pages[page_to_use]
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
					return self.create_plan(active_prefill, decodes_state)

		request = NextIterationPlan(
			schedule_prefill=schedule_prefill,
			schedule_decodes=schedule_decodes,
			prefill_request=next_prefill_req,
			prefill_pages_update=prefill_pages_update,
			new_decodes_requests=next_decodes_reqs,
			decodes_state_page_updates=decodes_state_page_updates,
		)
		return request


class ModelIOProcessor:
	"""
	Processes and transforms model inputs and outputs for paged attention.

	This class handles the construction of model input batches from prefill and decode
	states, and organizes raw model outputs into structured data for further processing.
	"""

	@staticmethod
	def prepare_model_output(
		next_token: jax.Array,
		complete: jax.Array,
		attn_meta: PagedAttentionMetadata,
		sampling_params: SamplingParams,
	) -> ModelOutputBatch:
		"""
		Processes the raw model output (logits converted to tokens) and completion flags
		to create a structured ModelOutputBatch object.

		It separates the outputs corresponding to the prefill step (if any) and the
		decode steps (if any) based on the structure defined in `attn_meta`.

		Args:
		    next_token (jax.Array): The generated token IDs from the model.
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
			output.prefill_token_id = next_token[0]
			output.prefill_complete = complete[0]
			output.prefill_next_position = attn_meta.prefill_length

		if not has_prefill and has_generate:
			output.decodes_token_ids = next_token
			output.decodes_completes = complete
			output.decodes_next_position = jnp.where(
				output.decodes_completes,
				-1,
				attn_meta.decodes_position + 1,
			)
			output.decodes_next_position = jnp.where(
				output.decodes_next_position,
				output.decodes_next_position,
				-1,
			)

		if has_prefill and has_generate:
			output.prefill_complete = complete[0]
			output.prefill_token_id = next_token[0]
			output.prefill_next_position = attn_meta.prefill_length

			output.decodes_token_ids = next_token[1:]
			output.decodes_completes = complete[1:]
			output.decodes_next_position = jnp.where(
				output.decodes_completes,
				-1,
				attn_meta.decodes_position + 1,
			)
			output.decodes_next_position = jnp.where(
				output.decodes_next_position,
				output.decodes_next_position,
				-1,
			)
		output.next_sampling_params = sampling_params
		return output

	@staticmethod
	@partial(cjit, static_argnums=(3,))
	@partial(jax.jit, static_argnums=(3,))
	def prepare_model_input(
		attn_meta: PagedAttentionMetadata,
		ongoing_prefill: InitialSequenceRequest,
		chunk_id: jax.Array,
		chunk_size: jax.Array,
		ongoing_decodes: ActiveSequenceBatch,
		ongoing_updates: ActiveSequenceBatch,
		insert_slots: jax.Array,
	):
		"""
		Prepares model input by slicing and concatenating prefill and decode tokens, updating attention metadata.

		Args:
		    attn_meta (PagedAttentionMetadata): Metadata to update with positions and page tables.
		    ongoing_prefill (InitialSequenceRequest): Current prefill request state.
		    chunk_id (jax.Array): Index of the current prefill chunk.
		    chunk_size (jax.Array): Size of each prefill chunk.
		    ongoing_decodes (ActiveSequenceBatch): Current decode batch state.
		    ongoing_updates (ActiveSequenceBatch): Batch updates for ongoing decodes.
		    insert_slots (jax.Array): Slots indices indicating where to insert new decodes.

		Returns:
		    tuple: (input_ids, decodes_token_ids, positions, attn_meta) where:
		        input_ids (jnp.ndarray): Combined token IDs for model input.
		        decodes_token_ids (jnp.ndarray): Token IDs array for decode phase.
		        positions (jnp.ndarray): Position indices matching input_ids.
		        attn_meta (PagedAttentionMetadata): Updated attention metadata.
		"""
		if ongoing_prefill.is_active:
			idx = chunk_id * chunk_size
			ongoing_prefill.token_ids = jax.lax.dynamic_slice_in_dim(
				ongoing_prefill.token_ids, idx, chunk_size
			)
			ongoing_prefill.positions = jax.lax.dynamic_slice_in_dim(
				ongoing_prefill.positions, idx, chunk_size
			)

		if ongoing_decodes.is_active:
			ongoing_decodes.insert_decode_state(insert_slots, ongoing_updates)

		if ongoing_prefill.is_active and ongoing_decodes.is_active:
			input_ids = jnp.concatenate(
				(ongoing_prefill.token_ids, ongoing_decodes.token_ids)
			)
			positions = jnp.concatenate(
				(ongoing_prefill.positions, ongoing_decodes.positions)
			)

			sampling_params = ongoing_decodes.sampling_params

			attn_meta.prefill_length = ongoing_prefill.length
			attn_meta.prefill_position = ongoing_prefill.positions
			attn_meta.prefill_page_table = ongoing_prefill.page_indices
			attn_meta.decodes_position = ongoing_decodes.positions
			attn_meta.decodes_page_table = ongoing_decodes.page_table

		elif ongoing_prefill.is_active:
			input_ids = ongoing_prefill.token_ids
			positions = ongoing_prefill.positions

			sampling_params = ongoing_prefill.sampling_params

			attn_meta.prefill_length = ongoing_prefill.length
			attn_meta.prefill_position = ongoing_prefill.positions
			attn_meta.prefill_page_table = ongoing_prefill.page_indices

		elif ongoing_decodes.is_active:
			input_ids = ongoing_decodes.token_ids
			positions = ongoing_decodes.positions

			sampling_params = ongoing_decodes.sampling_params

			attn_meta.decodes_position = ongoing_decodes.positions
			attn_meta.decodes_page_table = ongoing_decodes.page_table

		else:
			raise ValueError(
				"Failed to build the input as no prefill or generate gets scheduled"
			)

		return (
			input_ids,
			ongoing_decodes.token_ids,
			positions,
			sampling_params,
			attn_meta,
		)

	@classmethod
	def build_input(
		cls,
		iteration_plan: NextIterationPlan,
		metadata: PagedAttentionCacheMetaData,
		decodes_state: ActiveSequenceBatch,
	):
		"""
		Orchestrates the preparation of model inputs based on a scheduling decision.

		It retrieves the necessary data from the `iteration_plan` and `decodes_state`,
		formats it (converting lists/numpy arrays to JAX arrays as needed), and
		calls `prepare_model_input` to perform the JIT-compiled merging and updating.
		It also updates the `decodes_state` (device arrays) based on the results
		from `prepare_model_input`.

		Args:
		    cls: The class itself.
		    iteration_plan (NextIterationPlan): The output of the scheduler, indicating what to run.
		    metadata (PagedAttentionCacheMetaData): Cache configuration.
		    decodes_state (ActiveSequenceBatch): The current state of the decode batch.

		Returns:
		    ModelInputBatch: The structured input ready for the model's forward pass.
		"""
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)

		attn_meta = PagedAttentionMetadata.init_empty()
		ongoing_prefill = InitialSequenceRequest.init_empty()
		ongoing_decodes = ActiveSequenceBatch.init_empty()
		ongoing_updates = ActiveSequenceBatch.init_empty()

		chunk_id = scalar
		chunk_size = 512
		insert_slots = scalar

		if iteration_plan.schedule_prefill:
			prefill = iteration_plan.prefill_request
			ongoing_prefill.copy_prefill(prefill=prefill)
			chunk_id = prefill.chunk_idx
			chunk_size = prefill.chunk_size

		if iteration_plan.schedule_decodes:
			ongoing_decodes.copy_decode(decodes_state)
			ongoing_updates = ongoing_updates.init_numpy(metadata)
			insert_slots = np.full((metadata.batch_size,), 1e6, dtype=np.int32)
			for i, decode_request in enumerate(iteration_plan.new_decodes_requests):
				ongoing_updates.insert_from_task(i, decode_request)
				insert_slots[i] = decode_request.slot
			ongoing_updates.pad_tokens(
				metadata.batch_size - len(iteration_plan.new_decodes_requests)
			)
			ongoing_updates.apply_assignment(iteration_plan.decodes_state_page_updates)

		(
			input_ids,
			decodes_token_ids,
			positions,
			sampling_params,
			attn_meta,
		) = cls.prepare_model_input(
			attn_meta,
			ongoing_prefill,
			chunk_id,
			chunk_size,
			ongoing_decodes,
			ongoing_updates,
			insert_slots,
		)

		if iteration_plan.schedule_decodes:
			decodes_state.token_ids = decodes_token_ids
			decodes_state.positions = attn_meta.decodes_position
			decodes_state.page_table = attn_meta.decodes_page_table
			decodes_state.sampling_params = sampling_params

		return ModelInputBatch(
			input_ids=input_ids,
			positions=positions,
			attn_meta=attn_meta,
			sampling_params=sampling_params,
		)


__all__ = (
	"InferenceScheduler",
	"HBMPageManager",
	"ModelIOProcessor",
)
