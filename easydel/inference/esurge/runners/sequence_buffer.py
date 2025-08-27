# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Sequence buffer for managing token sequences during generation.

Provides efficient storage and management of token sequences, page tables,
and generation metadata for batch processing.

Classes:
    SequenceBuffer: Main buffer for managing sequences
    DecodeRowInfo: Information about sequences in decode phase
    ModelRunBatch: Batch data for model execution

Example:
    >>> buffer = SequenceBuffer(
    ...     max_num_reqs=32,
    ...     max_model_len=2048,
    ...     max_num_batched_tokens=4096,
    ...     vocab_size=50000,
    ...     page_sizes=[16, 32]
    ... )
    >>> buffer.begin_sequence("req_1", [1, 2, 3])
    >>> batch = buffer.prepare_batch()
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import jax
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp

from easydel.inference.esurge.request import EngineRequest
from easydel.utils.compiling_utils import ejit

from ...sampling_params import SamplingParams, SamplingType
from ..outputs import LogprobsTensors, swap_dict_values
from ..page_table import MultiGroupPageTable

logger = get_logger(__name__)


@ejit(static_argnames=("padded_num_reqs", "padded_prompt_len"))
def pack_prompts(token_ids, num_prompt_tokens, padded_num_reqs, padded_prompt_len, pad_id):
    """Pack prompt tokens into a padded tensor.

    Creates a padded tensor of prompt tokens with consistent shape for
    batch processing. Tokens beyond the prompt length are replaced with
    the padding ID.

    Args:
        token_ids: Token IDs array of shape [max_num_reqs, max_model_len].
        num_prompt_tokens: Number of prompt tokens per request [max_num_reqs].
        padded_num_reqs: Target number of requests after padding.
        padded_prompt_len: Target prompt length after padding.
        pad_id: Token ID to use for padding.

    Returns:
        Packed prompts array of shape [padded_num_reqs, padded_prompt_len]
        with valid tokens and padding.

    Note:
        This function is JIT-compiled with static arguments for padded dimensions
        to enable efficient compilation caching.
    """
    slice_tokens = token_ids[:padded_num_reqs, :padded_prompt_len]
    lengths = num_prompt_tokens[:padded_num_reqs, None]  # [B,1]
    arange = jnp.arange(padded_prompt_len, dtype=lengths.dtype)[None, :]  # [1,T]
    mask = arange < lengths  # [B,T]
    pad_mat = jnp.full_like(slice_tokens, pad_id)
    return jnp.where(mask, slice_tokens, pad_mat)


@ejit(static_argnames=("padded_num_reqs",))
def build_sampling_arrays(temperature, min_p, top_p, top_k, num_reqs, padded_num_reqs):
    """Build padded sampling parameter arrays.

    Pads sampling parameters to a consistent size for batch processing,
    filling unused slots with default values.

    Args:
        temperature: Temperature values for sampling.
        min_p: Minimum probability threshold values.
        top_p: Top-p (nucleus) sampling values.
        top_k: Top-k sampling values.
        num_reqs: Actual number of requests.
        padded_num_reqs: Target padded number of requests.

    Returns:
        A tuple of padded arrays:
            - temperature: Padded with -1.0 (float32)
            - min_p: Padded with 0.0 (float32)
            - top_p: Padded with 1.0 (float32)
            - top_k: Padded with 0 (int32)

    Note:
        Default padding values are chosen to be neutral for sampling operations.
    """

    def fill(arr, fill_val):
        out = jnp.full((padded_num_reqs,), fill_val, dtype=arr.dtype)
        return out.at[:num_reqs].set(arr[:num_reqs])

    return (
        fill(temperature, -1.0).astype(jnp.float32),
        fill(min_p, 0.0).astype(jnp.float32),
        fill(top_p, 1.0).astype(jnp.float32),
        fill(top_k, 0).astype(jnp.int32),
    )


@ejit
def swap_rows(arr, i1, i2):
    """Swap two rows in an array.

    Args:
        arr: Input array to swap rows in.
        i1: Index of first row.
        i2: Index of second row.

    Returns:
        Array with rows i1 and i2 swapped.

    Note:
        This function is JIT-compiled for efficient execution.
    """
    idx = jnp.arange(arr.shape[0])
    idx = idx.at[i1].set(i2)
    idx = idx.at[i2].set(i1)
    return arr[idx]


def swap_rows_pytree(arrs, i1, i2):
    """Swap rows across all arrays in a pytree.

    Args:
        arrs: PyTree containing arrays.
        i1: Index of first row to swap.
        i2: Index of second row to swap.

    Returns:
        PyTree with same structure but rows swapped in all arrays.
    """
    return jax.tree_map(lambda a: swap_rows(a, i1, i2), arrs)


@ejit
def move_row(arr, from_idx, to_idx):
    """Move a row from one index to another.

    Args:
        arr: Input array.
        from_idx: Source row index.
        to_idx: Destination row index.

    Returns:
        Array with row moved from from_idx to to_idx.

    Note:
        This overwrites the destination row without preserving it.
    """
    return arr.at[to_idx].set(arr[from_idx])


@ejit(static_argnames=("vocab_size", "max_allowed"))
def build_allowed_mask(allowed_ids_padded, allowed_lens, vocab_size, max_allowed):
    """Build a mask for allowed token IDs.

    Creates a boolean mask indicating which tokens are allowed for each request.
    The mask uses inverted logic where True means disallowed and False means allowed.

    Args:
        allowed_ids_padded: Padded array of allowed token IDs [B, max_allowed].
        allowed_lens: Number of valid allowed IDs per request [B].
        vocab_size: Total vocabulary size.
        max_allowed: Maximum number of allowed tokens per request.

    Returns:
        Boolean mask of shape [B, vocab_size] where True indicates
        the token is disallowed and False indicates it's allowed.

    Note:
        The inverted logic (True=disallowed) is used for compatibility
        with masking operations that zero out disallowed values.
    """
    B = allowed_ids_padded.shape[0]
    mask = jnp.ones((B, vocab_size), dtype=bool)

    batch_idx = jnp.repeat(jnp.arange(B)[:, None], max_allowed, axis=1)  # [B, max_allowed]
    flat_batch = batch_idx.reshape(-1)
    flat_token = allowed_ids_padded.reshape(-1)

    ar = jnp.arange(max_allowed)[None, :]
    valid = ar < allowed_lens[:, None]
    flat_valid = valid.reshape(-1)

    flat_batch = flat_batch[flat_valid]
    flat_token = flat_token[flat_valid]

    mask = mask.at[flat_batch, flat_token].set(False)
    return mask


@auto_pytree(frozen=True)
class SequenceBuffer:
    """Buffer for managing token sequences during generation.

    Functional, dataclass-PyTree version:
      - Arrays and page_table are leaves
      - Python containers (lists/sets/dicts) are static (non-leaves)
      - Mutating methods return a new SequenceBuffer instance

    Use SequenceBuffer.create(...) to construct an instance.
    """

    # Leaves
    token_ids: jax.Array
    num_tokens: jax.Array
    num_tokens_no_spec: jax.Array
    num_prompt_tokens: jax.Array
    num_computed_tokens: jax.Array

    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    min_p: jax.Array
    frequency_penalties: jax.Array
    presence_penalties: jax.Array
    repetition_penalties: jax.Array

    page_table: MultiGroupPageTable

    # Optional leaf
    # Static configuration (non-leaves)
    max_num_reqs: int = field(pytree_node=False)
    max_model_len: int = field(pytree_node=False)
    max_num_batched_tokens: int = field(pytree_node=False)
    vocab_size: int = field(pytree_node=False)

    # Python bookkeeping (non-leaves)
    _req_ids: list[str | None] = field(default_factory=list, pytree_node=False)
    req_id_to_index: dict[str, int] = field(default_factory=dict, pytree_node=False)
    req_output_token_ids: list[list[int] | None] = field(default_factory=list, pytree_node=False)

    greedy_reqs: set[str] = field(default_factory=set, pytree_node=False)
    random_reqs: set[str] = field(default_factory=set, pytree_node=False)
    top_p_reqs: set[str] = field(default_factory=set, pytree_node=False)
    top_k_reqs: set[str] = field(default_factory=set, pytree_node=False)
    min_p_reqs: set[str] = field(default_factory=set, pytree_node=False)
    frequency_penalties_reqs: set[str] = field(default_factory=set, pytree_node=False)
    presence_penalties_reqs: set[str] = field(default_factory=set, pytree_node=False)
    repetition_penalties_reqs: set[str] = field(default_factory=set, pytree_node=False)
    has_allowed_token_ids: set[str] = field(default_factory=set, pytree_node=False)

    min_tokens: dict[int, tuple[int, set[int]]] = field(default_factory=dict, pytree_node=False)
    generator_seeds: dict[int, int] = field(default_factory=dict, pytree_node=False)
    num_logprobs: dict[str, int] = field(default_factory=dict, pytree_node=False)
    num_prompt_logprobs: dict[str, int] = field(default_factory=dict, pytree_node=False)
    in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = field(default_factory=dict, pytree_node=False)
    logit_bias: list[dict[int, float] | None] = field(default_factory=list, pytree_node=False)
    bad_words_token_ids: dict[int, list[list[int]]] = field(default_factory=dict, pytree_node=False)
    allowed_token_ids_mask: Any = None  # jax.Array | None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        page_sizes: list[int],
    ) -> SequenceBuffer:
        """Create a new SequenceBuffer with initialized arrays.

        Factory method that creates a SequenceBuffer with all arrays
        properly initialized to their default values.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_model_len: Maximum sequence length per request.
            max_num_batched_tokens: Maximum tokens in a batch.
            vocab_size: Size of the model vocabulary.
            page_sizes: List of page sizes for the page table.

        Returns:
            A new SequenceBuffer instance with initialized arrays and page table.

        Example:
            >>> buffer = SequenceBuffer.create(
            ...     max_num_reqs=32,
            ...     max_model_len=2048,
            ...     max_num_batched_tokens=4096,
            ...     vocab_size=50000,
            ...     page_sizes=[16, 32]
            ... )
        """
        token_ids = jnp.zeros((max_num_reqs, max_model_len), dtype=jnp.int32)
        num_tokens = jnp.zeros((max_num_reqs,), dtype=jnp.int32)
        num_tokens_no_spec = jnp.zeros((max_num_reqs,), dtype=jnp.int32)
        num_prompt_tokens = jnp.zeros((max_num_reqs,), dtype=jnp.int32)
        num_computed_tokens = jnp.zeros((max_num_reqs,), dtype=jnp.int32)

        temperature = jnp.full((max_num_reqs,), -1.0, dtype=jnp.float32)
        top_p = jnp.ones((max_num_reqs,), dtype=jnp.float32)
        top_k = jnp.full((max_num_reqs,), vocab_size, dtype=jnp.int32)
        min_p = jnp.zeros((max_num_reqs,), dtype=jnp.float32)
        frequency_penalties = jnp.zeros((max_num_reqs,), dtype=jnp.float32)
        presence_penalties = jnp.zeros((max_num_reqs,), dtype=jnp.float32)
        repetition_penalties = jnp.ones((max_num_reqs,), dtype=jnp.float32)

        page_table = MultiGroupPageTable.create(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            page_sizes=page_sizes,
        )

        return cls(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            vocab_size=vocab_size,
            token_ids=token_ids,
            num_tokens=num_tokens,
            num_tokens_no_spec=num_tokens_no_spec,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            page_table=page_table,
            logit_bias=[None] * max_num_reqs,
        )

    @property
    def req_ids(self) -> list[str]:
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_min_p(self) -> bool:
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0

    def _ensure_logit_bias_capacity(self, upto_idx: int) -> None:
        """Ensure logit_bias list has sufficient capacity.

        Args:
            upto_idx: Index that needs to be accessible.

        Note:
            Extends the list with None values if needed.
        """
        if len(self.logit_bias) <= upto_idx:
            self.logit_bias.extend([None] * (upto_idx + 1 - len(self.logit_bias)))

    def add_request(self, request: EngineRequest, req_index: int | None = None) -> SequenceBuffer:
        """Add a new request to the buffer.

        Adds a request with its tokens, sampling parameters, and metadata.
        Handles prompt truncation if it exceeds maximum length.

        Args:
            request: The engine request to add containing:
                - req_id: Unique request identifier
                - prompt_token_ids: Input prompt tokens
                - sampling_params: Sampling configuration
                - page_ids: Page allocation for KV cache
            req_index: Optional specific index to place the request.
                If None, finds the next available slot.

        Returns:
            A new SequenceBuffer instance with the request added.

        Raises:
            ValueError: If the request ID already exists in the buffer.
            IndexError: If req_index is out of bounds.
            RuntimeError: If the buffer is full.

        Note:
            This method is functional and returns a new buffer instance
            rather than modifying in place.
        """
        req_id = request.req_id
        if req_id in self.req_id_to_index:
            raise ValueError(f"Request ID {req_id} is already present at index {self.req_id_to_index[req_id]}.")

        req_index = self._allocate_index(req_index)

        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids
        self.req_id_to_index[req_id] = req_index

        # Copy tokens into arrays (functional)
        num_prompt_tokens = min(len(request.prompt_token_ids), self.max_model_len)
        new_num_prompt_tokens = self.num_prompt_tokens.at[req_index].set(num_prompt_tokens)
        new_token_ids = self.token_ids.at[req_index, :num_prompt_tokens].set(
            jnp.array(request.prompt_token_ids[:num_prompt_tokens], dtype=jnp.int32)
        )

        if request.output_token_ids:
            start_idx = num_prompt_tokens
            max_output_tokens = self.max_model_len - num_prompt_tokens
            output_tokens_to_copy = request.output_token_ids[:max_output_tokens]
            if output_tokens_to_copy:
                end_idx = min(start_idx + len(output_tokens_to_copy), self.max_model_len)
                new_token_ids = new_token_ids.at[req_index, start_idx:end_idx].set(
                    jnp.array(output_tokens_to_copy, dtype=jnp.int32)
                )

        capped_num_tokens = min(int(request.num_tokens), self.max_model_len)
        new_num_tokens = self.num_tokens.at[req_index].set(capped_num_tokens)
        new_num_tokens_no_spec = self.num_tokens_no_spec.at[req_index].set(capped_num_tokens)
        new_num_computed_tokens = self.num_computed_tokens.at[req_index].set(
            min(int(request.num_computed_tokens), self.max_model_len)
        )

        buf = replace(
            self,
            token_ids=new_token_ids,
            num_prompt_tokens=new_num_prompt_tokens,
            num_tokens=new_num_tokens,
            num_tokens_no_spec=new_num_tokens_no_spec,
            num_computed_tokens=new_num_computed_tokens,
        )

        # Page table
        buf = replace(buf, page_table=buf.page_table.add_row(request.page_ids, req_index))

        # Sampling params
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"
        buf = buf._process_sampling_params(sampling_params, req_id, req_index)
        buf = buf._process_optional_params(request, sampling_params, req_id, req_index)
        return buf

    def remove_request(self, req_id: str) -> tuple[SequenceBuffer, int | None]:
        """Remove a request from the buffer.

        Removes all data associated with a request ID and cleans up
        related bookkeeping structures.

        Args:
            req_id: The request ID to remove.

        Returns:
            A tuple containing:
                - new_buffer: Updated SequenceBuffer with request removed
                - removed_index: Index where the request was removed, or None if not found

        Note:
            This method should typically be followed by condense() to remove
            gaps in the buffer and maintain efficiency.
        """
        req_index = self.req_id_to_index.pop(req_id, None)

        if req_index is None:
            return self, None

        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        for req_set in [
            self.greedy_reqs,
            self.random_reqs,
            self.top_p_reqs,
            self.top_k_reqs,
            self.min_p_reqs,
            self.frequency_penalties_reqs,
            self.presence_penalties_reqs,
            self.repetition_penalties_reqs,
            self.has_allowed_token_ids,
        ]:
            req_set.discard(req_id)

        self.min_tokens.pop(req_index, None)
        self.generator_seeds.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)
        self.bad_words_token_ids.pop(req_index, None)

        # Guarded indexing
        self._ensure_logit_bias_capacity(req_index)
        self.logit_bias[req_index] = None

        new_mask = self.allowed_token_ids_mask
        if new_mask is not None:
            new_mask = new_mask.at[req_index].set(False)

        return replace(self, allowed_token_ids_mask=new_mask), req_index

    def swap_states(self, i1: int, i2: int) -> SequenceBuffer:
        """Swap the states of two requests at given indices.

        Exchanges all data (tokens, parameters, metadata) between two
        request positions in the buffer.

        Args:
            i1: Index of the first request.
            i2: Index of the second request.

        Returns:
            A new SequenceBuffer with the two requests swapped.

        Raises:
            AssertionError: If either index doesn't contain a valid request.

        Note:
            This is useful for buffer reorganization and optimization.
        """
        old_id_i1, old_id_i2 = self._req_ids[i1], self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = old_id_i2, old_id_i1
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )

        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1] = i2
        self.req_id_to_index[old_id_i2] = i1

        # Swap arrays
        new_num_tokens = swap_rows(self.num_tokens, i1, i2)
        new_num_tokens_no_spec = swap_rows(self.num_tokens_no_spec, i1, i2)
        new_num_prompt_tokens = swap_rows(self.num_prompt_tokens, i1, i2)
        new_num_computed_tokens = swap_rows(self.num_computed_tokens, i1, i2)
        new_temperature = swap_rows(self.temperature, i1, i2)
        new_top_p = swap_rows(self.top_p, i1, i2)
        new_top_k = swap_rows(self.top_k, i1, i2)
        new_frequency_penalties = swap_rows(self.frequency_penalties, i1, i2)
        new_presence_penalties = swap_rows(self.presence_penalties, i1, i2)
        new_repetition_penalties = swap_rows(self.repetition_penalties, i1, i2)
        new_min_p = swap_rows(self.min_p, i1, i2)
        new_token_ids = swap_rows(self.token_ids, i1, i2)

        new_mask = self.allowed_token_ids_mask
        if new_mask is not None:
            new_mask = swap_rows(new_mask, i1, i2)

        swap_dict_values(self.generator_seeds, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)
        self.logit_bias[i1], self.logit_bias[i2] = self.logit_bias[i2], self.logit_bias[i1]

        return replace(
            self,
            num_tokens=new_num_tokens,
            num_tokens_no_spec=new_num_tokens_no_spec,
            num_prompt_tokens=new_num_prompt_tokens,
            num_computed_tokens=new_num_computed_tokens,
            temperature=new_temperature,
            top_p=new_top_p,
            top_k=new_top_k,
            frequency_penalties=new_frequency_penalties,
            presence_penalties=new_presence_penalties,
            repetition_penalties=new_repetition_penalties,
            min_p=new_min_p,
            token_ids=new_token_ids,
            page_table=self.page_table.swap_row(i1, i2),
            allowed_token_ids_mask=new_mask,
        )

    def condense(self, empty_req_indices: list[int]) -> SequenceBuffer:
        """Condense the buffer by removing gaps.

        Moves requests from the end of the buffer to fill empty slots,
        maintaining a contiguous block of active requests at the beginning.

        Args:
            empty_req_indices: List of indices that are now empty and need filling.

        Returns:
            A new SequenceBuffer with gaps removed and requests condensed.

        Note:
            This operation is important for maintaining buffer efficiency
            after removing requests. It ensures active requests are packed
            at the beginning of the buffer.
        """
        buf = self
        num_reqs = buf.num_reqs
        if num_reqs == 0:
            buf._req_ids.clear()
            buf.req_output_token_ids.clear()
            return buf

        last_req_index = num_reqs + len(empty_req_indices) - 1

        for empty_index in reversed(empty_req_indices):
            while last_req_index in empty_req_indices and last_req_index > empty_index:
                last_req_index -= 1
            if empty_index >= last_req_index:
                continue
            buf = buf._move_request(last_req_index, empty_index)
            last_req_index -= 1

        del buf._req_ids[buf.num_reqs :]
        del buf.req_output_token_ids[buf.num_reqs :]
        return buf

    def _move_request(self, from_idx: int, to_idx: int) -> SequenceBuffer:
        """Move a request from one index to another.

        Internal method for relocating a request within the buffer.

        Args:
            from_idx: Source index of the request.
            to_idx: Destination index for the request.

        Returns:
            A new SequenceBuffer with the request moved.

        Raises:
            AssertionError: If from_idx doesn't contain a valid request.

        Note:
            This is an internal method used by condense() and other
            buffer reorganization operations.
        """
        req_id = self._req_ids[from_idx]
        assert req_id is not None

        # Static bookkeeping
        self._req_ids[to_idx] = req_id
        self._req_ids[from_idx] = None
        self.req_output_token_ids[to_idx] = self.req_output_token_ids[from_idx]
        self.req_output_token_ids[from_idx] = None
        self.req_id_to_index[req_id] = to_idx

        # Arrays
        new_token_ids = move_row(self.token_ids, from_idx, to_idx)
        new_num_tokens = move_row(self.num_tokens, from_idx, to_idx)
        new_num_tokens_no_spec = move_row(self.num_tokens_no_spec, from_idx, to_idx)
        new_num_prompt_tokens = move_row(self.num_prompt_tokens, from_idx, to_idx)
        new_num_computed_tokens = move_row(self.num_computed_tokens, from_idx, to_idx)
        new_temperature = move_row(self.temperature, from_idx, to_idx)
        new_top_p = move_row(self.top_p, from_idx, to_idx)
        new_top_k = move_row(self.top_k, from_idx, to_idx)
        new_frequency_penalties = move_row(self.frequency_penalties, from_idx, to_idx)
        new_presence_penalties = move_row(self.presence_penalties, from_idx, to_idx)
        new_repetition_penalties = move_row(self.repetition_penalties, from_idx, to_idx)
        new_min_p = move_row(self.min_p, from_idx, to_idx)

        # Page table
        new_page_table = self.page_table.move_row(from_idx, to_idx)

        # Sparse/optional
        buf = replace(
            self,
            token_ids=new_token_ids,
            num_tokens=new_num_tokens,
            num_tokens_no_spec=new_num_tokens_no_spec,
            num_prompt_tokens=new_num_prompt_tokens,
            num_computed_tokens=new_num_computed_tokens,
            temperature=new_temperature,
            top_p=new_top_p,
            top_k=new_top_k,
            frequency_penalties=new_frequency_penalties,
            presence_penalties=new_presence_penalties,
            repetition_penalties=new_repetition_penalties,
            min_p=new_min_p,
            page_table=new_page_table,
        )
        return buf._move_sparse_data(from_idx, to_idx)

    def _move_sparse_data(self, from_idx: int, to_idx: int) -> SequenceBuffer:
        """Move sparse and optional data between indices.

        Handles the movement of data that may not exist for all requests,
        such as generator seeds, min_tokens, bad words, etc.

        Args:
            from_idx: Source index.
            to_idx: Destination index.

        Returns:
            A new SequenceBuffer with sparse data moved.

        Note:
            This method complements _move_request() by handling
            optional parameters that aren't stored in the main arrays.
        """
        if from_idx in self.generator_seeds:
            self.generator_seeds[to_idx] = self.generator_seeds.pop(from_idx)

        if from_idx in self.min_tokens:
            self.min_tokens[to_idx] = self.min_tokens.pop(from_idx)

        if from_idx in self.bad_words_token_ids:
            self.bad_words_token_ids[to_idx] = self.bad_words_token_ids.pop(from_idx)

        self.logit_bias[to_idx] = self.logit_bias[from_idx]
        self.logit_bias[from_idx] = None

        new_mask = self.allowed_token_ids_mask
        if new_mask is not None:
            new_mask = new_mask.at[to_idx].set(new_mask[from_idx])
            new_mask = new_mask.at[from_idx].set(False)

        return replace(self, allowed_token_ids_mask=new_mask)

    def _process_sampling_params(self, sampling_params: SamplingParams, req_id: str, req_index: int) -> SequenceBuffer:
        """Process and store core sampling parameters.

        Updates arrays with sampling configuration like temperature, top_p, top_k, etc.
        Also maintains sets tracking which requests use which sampling strategies.

        Args:
            sampling_params: Sampling configuration containing temperature, top_p, etc.
            req_id: Request identifier for bookkeeping.
            req_index: Index where parameters should be stored.

        Returns:
            A new SequenceBuffer with updated sampling parameters.

        Note:
            Maintains separate tracking sets for different sampling strategies
            to enable optimized execution paths.
        """
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        frequency_penalties = self.frequency_penalties
        presence_penalties = self.presence_penalties
        repetition_penalties = self.repetition_penalties

        if sampling_params.sampling_type == SamplingType.GREEDY:
            temperature = temperature.at[req_index].set(-1.0)
            self.greedy_reqs.add(req_id)
        else:
            temperature = temperature.at[req_index].set(sampling_params.temperature)
            self.random_reqs.add(req_id)

        top_p = top_p.at[req_index].set(sampling_params.top_p)
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)

        tk = sampling_params.top_k
        if 0 < tk < self.vocab_size:
            self.top_k_reqs.add(req_id)
            top_k = top_k.at[req_index].set(tk)
        else:
            top_k = top_k.at[req_index].set(self.vocab_size)

        min_p = min_p.at[req_index].set(sampling_params.min_p)
        if sampling_params.min_p > 1e-5:
            self.min_p_reqs.add(req_id)

        if sampling_params.frequency_penalty != 0.0:
            frequency_penalties = frequency_penalties.at[req_index].set(sampling_params.frequency_penalty)
            self.frequency_penalties_reqs.add(req_id)

        if sampling_params.presence_penalty != 0.0:
            presence_penalties = presence_penalties.at[req_index].set(sampling_params.presence_penalty)
            self.presence_penalties_reqs.add(req_id)

        if sampling_params.repetition_penalty != 1.0:
            repetition_penalties = repetition_penalties.at[req_index].set(sampling_params.repetition_penalty)
            self.repetition_penalties_reqs.add(req_id)

        return replace(
            self,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
        )

    def _process_optional_params(
        self,
        request: EngineRequest,
        sampling_params: SamplingParams,
        req_id: str,
        req_index: int,
    ) -> SequenceBuffer:
        """Process optional and sparse sampling parameters.

        Handles parameters that may not be present for all requests,
        such as logit bias, allowed tokens, bad words, etc.

        Args:
            request: The engine request containing optional metadata.
            sampling_params: Sampling parameters with optional fields.
            req_id: Request identifier.
            req_index: Index for parameter storage.

        Returns:
            A new SequenceBuffer with optional parameters processed.

        Note:
            These parameters are stored in sparse data structures
            to avoid memory overhead for unused features.
        """
        buf = self
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens, sampling_params.all_stop_token_ids)

        if hasattr(request, "generator_seed") and request.generator_seed is not None:
            self.generator_seeds[req_index] = request.generator_seed

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs

        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs

        if sampling_params.logit_bias is not None:
            if len(self.logit_bias) < self.max_num_reqs:
                # Ensure list length
                self.logit_bias.extend([None] * (self.max_num_reqs - len(self.logit_bias)))
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            buf = buf._set_allowed_token_ids(req_id, req_index, sampling_params.allowed_token_ids)

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

        return buf

    def _set_allowed_token_ids(self, req_id: str, req_index: int, allowed_token_ids: list[int]) -> SequenceBuffer:
        """Set the allowed token IDs for a request.

        Creates or updates a mask indicating which tokens are allowed for generation.
        Uses inverted logic where True means disallowed.

        Args:
            req_id: Request identifier.
            req_index: Index of the request.
            allowed_token_ids: List of token IDs that are allowed.

        Returns:
            A new SequenceBuffer with updated allowed token mask.

        Raises:
            ValueError: If any token ID is outside the valid vocabulary range.

        Note:
            The mask uses inverted logic (True=disallowed) for compatibility
            with JAX masking operations.
        """
        if any((t < 0 or t >= self.vocab_size) for t in allowed_token_ids):
            raise ValueError(f"allowed_token_ids must be within [0, {self.vocab_size})")

        self.has_allowed_token_ids.add(req_id)
        mask = self.allowed_token_ids_mask
        if mask is None:
            mask = jnp.zeros((self.max_num_reqs, self.vocab_size), dtype=bool)

        # Start with all True (disallowed) for this row, then set allowed ones to False
        mask = mask.at[req_index].set(True)
        if allowed_token_ids:
            mask = mask.at[req_index, allowed_token_ids].set(False)

        return replace(self, allowed_token_ids_mask=mask)

    def _allocate_index(self, req_index: int | None) -> int:
        """Allocate an index for a new request.

        Finds or validates an index position for placing a new request.

        Args:
            req_index: Optional preferred index. If None, finds next available.

        Returns:
            The allocated index.

        Raises:
            IndexError: If req_index exceeds maximum capacity.
            ValueError: If req_index is already occupied.
            RuntimeError: If buffer is full and no index is available.

        Note:
            This method may extend internal bookkeeping lists as needed.
        """
        if req_index is not None:
            if req_index >= self.max_num_reqs:
                raise IndexError(f"req_index {req_index} >= max_num_reqs {self.max_num_reqs}")
            while len(self._req_ids) < req_index:
                self._req_ids.append(None)
                self.req_output_token_ids.append(None)
            if req_index < len(self._req_ids) and self._req_ids[req_index] is not None:
                raise ValueError(f"req_index {req_index} is already occupied by {self._req_ids[req_index]}")
            return req_index

        for i, rid in enumerate(self._req_ids):
            if rid is None:
                return i

        if len(self._req_ids) < self.max_num_reqs:
            return len(self._req_ids)

        raise RuntimeError("SequenceBuffer is full; cannot allocate a new request index.")

    def _make_prompt_token_ids_tensor(self) -> jax.Array:
        """Create a padded tensor of prompt token IDs.

        Returns:
            A padded array of prompt tokens suitable for batch processing.
            Shape is [num_reqs, max_prompt_len].

        Note:
            Uses the JIT-compiled pack_prompts function for efficiency.
        """
        if self.num_reqs == 0:
            return jnp.empty((0, 0), dtype=jnp.int32)

        max_prompt_len = int(jnp.max(self.num_prompt_tokens[: self.num_reqs]))
        return pack_prompts(self.token_ids, self.num_prompt_tokens, self.num_reqs, max_prompt_len, self.vocab_size)

    def get_request_indices_with_penalty(self) -> jax.Array:
        """Get indices of requests with penalties.

        Returns:
            Array of indices for requests that have frequency, presence,
            or repetition penalties applied.

        Note:
            Used to optimize penalty application by only processing
            requests that actually need it.
        """
        penalty_req_ids = self.frequency_penalties_reqs | self.presence_penalties_reqs | self.repetition_penalties_reqs
        if not penalty_req_ids:
            return jnp.array([], dtype=jnp.int32)

        indices = [self.req_id_to_index[req_id] for req_id in penalty_req_ids]
        return jnp.array(indices, dtype=jnp.int32)

    def get_active_sampling_params(self, req_index: int) -> dict[str, Any]:
        """Get active sampling parameters for a request.

        Args:
            req_index: Index of the request.

        Returns:
            Dictionary containing active sampling parameters for the request.
            Only includes parameters that are actually in use.

        Note:
            Returns empty dict if the index doesn't contain a valid request.
        """
        req_id = self._req_ids[req_index]
        if req_id is None:
            return {}

        params = {
            "temperature": self.temperature[req_index],
            "top_p": self.top_p[req_index],
            "top_k": self.top_k[req_index],
        }

        if req_id in self.min_p_reqs:
            params["min_p"] = self.min_p[req_index]
        if req_id in self.frequency_penalties_reqs:
            params["frequency_penalty"] = self.frequency_penalties[req_index]
        if req_id in self.presence_penalties_reqs:
            params["presence_penalty"] = self.presence_penalties[req_index]
        if req_id in self.repetition_penalties_reqs:
            params["repetition_penalty"] = self.repetition_penalties[req_index]

        return params

    def clear(self) -> SequenceBuffer:
        """Clear all data in the buffer.

        Resets all arrays to their initial values and clears all bookkeeping.

        Returns:
            A new SequenceBuffer with all data cleared.

        Note:
            This maintains the buffer structure and capacity but removes
            all request data.
        """
        self._req_ids.clear()
        self.req_id_to_index.clear()
        self.req_output_token_ids.clear()

        token_ids = jnp.zeros_like(self.token_ids)
        num_tokens = jnp.zeros_like(self.num_tokens)
        num_tokens_no_spec = jnp.zeros_like(self.num_tokens_no_spec)
        num_prompt_tokens = jnp.zeros_like(self.num_prompt_tokens)
        num_computed_tokens = jnp.zeros_like(self.num_computed_tokens)

        temperature = jnp.full_like(self.temperature, -1.0)
        top_p = jnp.ones_like(self.top_p)
        top_k = jnp.full_like(self.top_k, self.vocab_size)
        min_p = jnp.zeros_like(self.min_p)
        frequency_penalties = jnp.zeros_like(self.frequency_penalties)
        presence_penalties = jnp.zeros_like(self.presence_penalties)
        repetition_penalties = jnp.ones_like(self.repetition_penalties)

        for req_set in [
            self.greedy_reqs,
            self.random_reqs,
            self.top_p_reqs,
            self.top_k_reqs,
            self.min_p_reqs,
            self.frequency_penalties_reqs,
            self.presence_penalties_reqs,
            self.repetition_penalties_reqs,
            self.has_allowed_token_ids,
        ]:
            req_set.clear()

        self.min_tokens.clear()
        self.generator_seeds.clear()
        self.num_logprobs.clear()
        self.num_prompt_logprobs.clear()
        self.in_progress_prompt_logprobs_cpu.clear()
        self.bad_words_token_ids.clear()
        self.logit_bias = [None] * self.max_num_reqs

        return replace(
            self,
            token_ids=token_ids,
            num_tokens=num_tokens,
            num_tokens_no_spec=num_tokens_no_spec,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            page_table=self.page_table.clear(),
            allowed_token_ids_mask=jnp.zeros_like(self.allowed_token_ids_mask, dtype=bool)
            if self.allowed_token_ids_mask is not None
            else None,
        )

    def to_device_state(self) -> DeviceSequenceState:
        """Create a device-compatible view of the buffer.

        Returns:
            DeviceSequenceState containing only array/tensor data suitable
            for device operations. No Python containers are included.

        Note:
            This creates a view without copying data, making it efficient
            for passing to device-compiled functions.
        """
        return DeviceSequenceState(
            token_ids=self.token_ids,
            num_tokens=self.num_tokens,
            num_tokens_no_spec=self.num_tokens_no_spec,
            num_prompt_tokens=self.num_prompt_tokens,
            num_computed_tokens=self.num_computed_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            frequency_penalties=self.frequency_penalties,
            presence_penalties=self.presence_penalties,
            repetition_penalties=self.repetition_penalties,
            page_table=self.page_table,
            allowed_token_ids_mask=self.allowed_token_ids_mask,
        )

    def from_device_state(self, dev: DeviceSequenceState) -> SequenceBuffer:
        """Update buffer with data from device state.

        Args:
            dev: DeviceSequenceState containing updated arrays from device execution.

        Returns:
            A new SequenceBuffer with arrays updated from the device state
            while preserving Python bookkeeping structures.

        Note:
            This is used to incorporate results from device computation
            back into the buffer.
        """
        return replace(
            self,
            token_ids=dev.token_ids,
            num_tokens=dev.num_tokens,
            num_tokens_no_spec=dev.num_tokens_no_spec,
            num_prompt_tokens=dev.num_prompt_tokens,
            num_computed_tokens=dev.num_computed_tokens,
            temperature=dev.temperature,
            top_p=dev.top_p,
            top_k=dev.top_k,
            min_p=dev.min_p,
            frequency_penalties=dev.frequency_penalties,
            presence_penalties=dev.presence_penalties,
            repetition_penalties=dev.repetition_penalties,
            page_table=dev.page_table,
            allowed_token_ids_mask=dev.allowed_token_ids_mask,
        )


@auto_pytree(frozen=True)
class DeviceSequenceState:
    """Device-compatible state for sequence processing.

    A PyTree containing only array data suitable for device operations.
    This class excludes Python containers to enable efficient device execution.

    Attributes:
        token_ids: Token IDs for all requests [max_num_reqs, max_model_len].
        num_tokens: Total tokens per request [max_num_reqs].
        num_tokens_no_spec: Tokens excluding speculative decoding [max_num_reqs].
        num_prompt_tokens: Number of prompt tokens [max_num_reqs].
        num_computed_tokens: Tokens already processed [max_num_reqs].
        temperature: Sampling temperature [max_num_reqs].
        top_p: Top-p sampling values [max_num_reqs].
        top_k: Top-k sampling values [max_num_reqs].
        min_p: Minimum probability threshold [max_num_reqs].
        frequency_penalties: Frequency penalty values [max_num_reqs].
        presence_penalties: Presence penalty values [max_num_reqs].
        repetition_penalties: Repetition penalty values [max_num_reqs].
        page_table: Page table for KV cache management.
        allowed_token_ids_mask: Optional mask for allowed tokens.

    Note:
        Use SequenceBuffer.to_device_state()/from_device_state() to convert
        between this representation and the full SequenceBuffer.
    """

    token_ids: jax.Array
    num_tokens: jax.Array
    num_tokens_no_spec: jax.Array
    num_prompt_tokens: jax.Array
    num_computed_tokens: jax.Array

    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    min_p: jax.Array
    frequency_penalties: jax.Array
    presence_penalties: jax.Array
    repetition_penalties: jax.Array

    page_table: MultiGroupPageTable

    allowed_token_ids_mask: jax.Array | None = None

    def with_updates(self, **kwargs) -> DeviceSequenceState:
        """Create a new DeviceSequenceState with updated fields.

        Args:
            **kwargs: Fields to update with new values.

        Returns:
            A new DeviceSequenceState with the specified updates.
        """
        return replace(self, **kwargs)


@auto_pytree
class ModelRunnerSamplingMetadata:
    """Metadata for sampling operations during model execution.

    Contains sampling parameters and optional penalty/constraint data
    for batch processing during inference.

    Attributes:
        temperature: Temperature values for sampling.
        min_p: Minimum probability thresholds.
        top_k: Top-k sampling parameters.
        top_p: Top-p (nucleus) sampling parameters.
        all_greedy: Whether all requests use greedy sampling.
        logprobs: Whether to compute log probabilities.
        no_penalties: Whether penalties are disabled.
        prompt_token_ids: Optional prompt tokens for context.
        frequency_penalties: Optional frequency penalties.
        presence_penalties: Optional presence penalties.
        repetition_penalties: Optional repetition penalties.
        output_token_ids: Generated output tokens.
        min_tokens: Minimum tokens to generate.
        logit_bias: Per-token logit adjustments.
        allowed_token_ids_mask: Mask for allowed tokens.
        bad_words_token_ids: Tokens to avoid generating.
    """

    temperature: jax.Array
    min_p: jax.Array
    top_k: jax.Array
    top_p: jax.Array

    all_greedy: bool = True
    logprobs: bool = False
    no_penalties: bool = True

    prompt_token_ids: Any = None
    frequency_penalties: Any = None
    presence_penalties: Any = None
    repetition_penalties: Any = None

    output_token_ids: list[list[int]] = field(default_factory=list)
    min_tokens: Any = None
    logit_bias: list[dict[int, float]] = field(default_factory=list)
    allowed_token_ids_mask: Any = None
    bad_words_token_ids: Any = None

    @classmethod
    def from_sequence_buffer(
        cls,
        sequence_buffer: SequenceBuffer,
        padded_num_reqs: int,
        generate_params_if_all_greedy: bool = False,
    ):
        """Create sampling metadata from a sequence buffer.

        Args:
            sequence_buffer: Source buffer containing sampling parameters.
            padded_num_reqs: Target padded number of requests.
            generate_params_if_all_greedy: Whether to generate parameters
                even when all requests use greedy sampling.

        Returns:
            ModelRunnerSamplingMetadata with padded sampling arrays.

        Note:
            If all requests use greedy sampling and generate_params_if_all_greedy
            is False, returns zero-filled arrays for efficiency.
        """
        if sequence_buffer.all_greedy is True and not generate_params_if_all_greedy:
            return cls(
                temperature=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                min_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                top_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                top_k=jnp.zeros((padded_num_reqs,), dtype=jnp.int32),
            )

        num_reqs = sequence_buffer.num_reqs

        return cls(
            temperature=fill_slice(sequence_buffer.temperature, -1.0, num_reqs, padded_num_reqs).astype(jnp.float32),
            min_p=fill_slice(sequence_buffer.min_p, 0.0, num_reqs, padded_num_reqs).astype(jnp.float32),
            top_p=fill_slice(sequence_buffer.top_p, 1.0, num_reqs, padded_num_reqs).astype(jnp.float32),
            top_k=fill_slice(sequence_buffer.top_k, 0, num_reqs, padded_num_reqs).astype(jnp.int32),
        )


@ejit(static_argnums=(2, 3))
def fill_slice(arr, fill_val, num_reqs, padded_num_reqs):
    """Fill array slice with padding value.

    Args:
        arr: Input array to pad.
        fill_val: Value to use for padding.
        num_reqs: Number of valid requests.
        padded_num_reqs: Target padded size.

    Returns:
        Array with padding applied from num_reqs to padded_num_reqs.
    """
    return arr.at[num_reqs:padded_num_reqs].set(fill_val)[:padded_num_reqs]
