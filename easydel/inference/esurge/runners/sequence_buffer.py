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

from dataclasses import field
from typing import Any, cast

import jax
from eformer.pytree import auto_pytree
from jax import numpy as jnp

from easydel.inference.esurge.request import EngineRequest
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import get_logger

from ...sampling_params import SamplingParams, SamplingType
from ..outputs import LogprobsTensors, swap_dict_values
from ..page_table import MultiGroupPageTable

logger = get_logger(__name__)


@ejit(static_argnames=("padded_num_reqs", "padded_prompt_len"))
def pack_prompts(token_ids, num_prompt_tokens, padded_num_reqs, padded_prompt_len, pad_id):
    """
    token_ids: [max_num_reqs, max_model_len] int32
    num_prompt_tokens: [max_num_reqs] int32
    returns: [padded_num_reqs, padded_prompt_len]
    """
    slice_tokens = token_ids[:padded_num_reqs, :padded_prompt_len]
    lengths = num_prompt_tokens[:padded_num_reqs, None]  # [B,1]
    arange = jnp.arange(padded_prompt_len, dtype=lengths.dtype)[None, :]  # [1,T]
    mask = arange < lengths  # [B,T]
    pad_mat = jnp.full_like(slice_tokens, pad_id)
    return jnp.where(mask, slice_tokens, pad_mat)


@ejit(static_argnames=("padded_num_reqs",))
def build_sampling_arrays(temperature, min_p, top_p, top_k, num_reqs, padded_num_reqs):
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
    idx = jnp.arange(arr.shape[0])
    idx = idx.at[i1].set(i2)
    idx = idx.at[i2].set(i1)
    return arr[idx]


def swap_rows_pytree(arrs, i1, i2):
    return jax.tree_map(lambda a: swap_rows(a, i1, i2), arrs)


@ejit
def move_row(arr, from_idx, to_idx):
    return arr.at[to_idx].set(arr[from_idx])


@ejit(static_argnames=("vocab_size", "max_allowed"))
def build_allowed_mask(allowed_ids_padded, allowed_lens, vocab_size, max_allowed):
    """
    allowed_ids_padded: [B, max_allowed] int32
    allowed_lens: [B] int32
    returns: [B, vocab_size] bool (True=disallowed, False=allowed)
    """
    B = allowed_ids_padded.shape[0]
    mask = jnp.ones((B, vocab_size), dtype=bool)

    # Flatten indices of (batch, token_id) to scatter False (allowed)
    batch_idx = jnp.repeat(jnp.arange(B)[:, None], max_allowed, axis=1)  # [B, max_allowed]
    flat_batch = batch_idx.reshape(-1)
    flat_token = allowed_ids_padded.reshape(-1)

    # Build a mask to ignore padded entries
    ar = jnp.arange(max_allowed)[None, :]
    valid = ar < allowed_lens[:, None]
    flat_valid = valid.reshape(-1)

    # Only scatter for valid entries
    flat_batch = flat_batch[flat_valid]
    flat_token = flat_token[flat_valid]

    mask = mask.at[flat_batch, flat_token].set(False)
    return mask


class SequenceBuffer:
    """Buffer for managing token sequences during generation.

    Maintains token sequences, page tables, and metadata for efficient
    batch processing. Handles both prefill and decode phases.

    Attributes:
        max_num_reqs: Maximum concurrent requests.
        max_model_len: Maximum sequence length.
        max_num_batched_tokens: Maximum tokens per batch.
        vocab_size: Size of vocabulary.
        token_ids: 2D array of token IDs per request.
        num_tokens: Number of tokens per request.
        page_table: Multi-group page table for KV-cache.

    Example:
        >>> buffer = SequenceBuffer(
        ...     max_num_reqs=32,
        ...     max_model_len=2048,
        ...     max_num_batched_tokens=4096,
        ...     vocab_size=50000,
        ...     page_sizes=[16, 32]
        ... )
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        page_sizes: list[int],
    ):
        """Initialize SequenceBuffer.

        Args:
            max_num_reqs: Maximum concurrent requests.
            max_model_len: Maximum sequence length.
            max_num_batched_tokens: Maximum tokens per batch.
            vocab_size: Size of vocabulary.
            page_sizes: List of page sizes for multi-group caching.
        """
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}
        self.req_output_token_ids: list[list[int] | None] = []

        self.token_ids = jnp.zeros((max_num_reqs, max_model_len), dtype=jnp.int32)
        self.num_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_tokens_no_spec = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_prompt_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_computed_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)

        self.page_table = MultiGroupPageTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            page_sizes=page_sizes,
        )

        self._init_sampling_arrays()
        self._init_request_sets()
        self._init_sparse_storage()

    def _init_sampling_arrays(self):
        """Initialize sampling parameter arrays with appropriate dtypes."""
        self.temperature = jnp.full(self.max_num_reqs, -1.0, dtype=jnp.float32)
        self.top_p = jnp.ones(self.max_num_reqs, dtype=jnp.float32)
        self.top_k = jnp.full(self.max_num_reqs, self.vocab_size, dtype=jnp.int32)
        self.min_p = jnp.zeros(self.max_num_reqs, dtype=jnp.float32)
        self.frequency_penalties = jnp.zeros(self.max_num_reqs, dtype=jnp.float32)
        self.presence_penalties = jnp.zeros(self.max_num_reqs, dtype=jnp.float32)
        self.repetition_penalties = jnp.ones(self.max_num_reqs, dtype=jnp.float32)

    def _init_request_sets(self):
        """Initialize request tracking sets."""
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()
        self.top_p_reqs: set[str] = set()
        self.top_k_reqs: set[str] = set()
        self.min_p_reqs: set[str] = set()
        self.frequency_penalties_reqs: set[str] = set()
        self.presence_penalties_reqs: set[str] = set()
        self.repetition_penalties_reqs: set[str] = set()
        self.has_allowed_token_ids: set[str] = set()

    def _init_sparse_storage(self):
        """Initialize sparse storage for less common parameters."""
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}
        self.generator_seeds: dict[int, int] = {}
        self.num_logprobs: dict[str, int] = {}
        self.num_prompt_logprobs: dict[str, int] = {}
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}
        self.logit_bias: list[dict[int, float] | None] = [None] * self.max_num_reqs
        self.allowed_token_ids_mask: jax.Array | None = None
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

    @property
    def req_ids(self) -> list[str]:
        return cast(list[str], self._req_ids)

    def _copy_tokens(self, request: EngineRequest, req_index: int) -> None:
        """Efficiently copy prompt and output tokens with bounds checking."""
        num_prompt_tokens = min(len(request.prompt_token_ids), self.max_model_len)
        self.num_prompt_tokens = self.num_prompt_tokens.at[req_index].set(num_prompt_tokens)

        prompt_tokens_to_copy = request.prompt_token_ids[:num_prompt_tokens]
        self.token_ids = self.token_ids.at[req_index, :num_prompt_tokens].set(
            jnp.array(prompt_tokens_to_copy, dtype=jnp.int32)
        )

        if request.output_token_ids:
            start_idx = num_prompt_tokens
            max_output_tokens = self.max_model_len - num_prompt_tokens
            output_tokens_to_copy = request.output_token_ids[:max_output_tokens]

            if output_tokens_to_copy:
                end_idx = min(start_idx + len(output_tokens_to_copy), self.max_model_len)
                self.token_ids = self.token_ids.at[req_index, start_idx:end_idx].set(
                    jnp.array(output_tokens_to_copy, dtype=jnp.int32)
                )

    def _process_sampling_params(self, sampling_params: SamplingParams, req_id: str, req_index: int) -> None:
        """Process core sampling parameters."""

        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.temperature = self.temperature.at[req_index].set(-1.0)
            self.greedy_reqs.add(req_id)
        else:
            self.temperature = self.temperature.at[req_index].set(sampling_params.temperature)
            self.random_reqs.add(req_id)

        self.top_p = self.top_p.at[req_index].set(sampling_params.top_p)
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)

        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
            self.top_k = self.top_k.at[req_index].set(top_k)
        else:
            self.top_k = self.top_k.at[req_index].set(self.vocab_size)

        self.min_p = self.min_p.at[req_index].set(sampling_params.min_p)
        if sampling_params.min_p > 1e-5:
            self.min_p_reqs.add(req_id)

        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties = self.frequency_penalties.at[req_index].set(sampling_params.frequency_penalty)
            self.frequency_penalties_reqs.add(req_id)

        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties = self.presence_penalties.at[req_index].set(sampling_params.presence_penalty)
            self.presence_penalties_reqs.add(req_id)

        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties = self.repetition_penalties.at[req_index].set(sampling_params.repetition_penalty)
            self.repetition_penalties_reqs.add(req_id)

    def _allocate_index(self, req_index: int | None) -> int:
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

    def add_request(self, request: EngineRequest, req_index: int | None = None) -> None:
        req_id = request.req_id
        if req_id in self.req_id_to_index:
            raise ValueError(f"Request ID {req_id} is already present at index {self.req_id_to_index[req_id]}.")

        req_index = self._allocate_index(req_index)
        prompt_length = len(request.prompt_token_ids)
        safe_num_tokens = self.max_model_len // 2 if request.num_tokens > self.max_model_len else request.num_tokens
        safe_num_tokens = max(0, min(int(safe_num_tokens), self.max_model_len))

        if prompt_length > self.max_model_len:
            safe_length = self.max_model_len - safe_num_tokens
            if safe_length <= 0:
                logger.warning(
                    f"Request {req_id} has {prompt_length} prompt tokens; "
                    f"no room left for prompt with max_model_len={self.max_model_len} and "
                    f"num_tokens={request.num_tokens}. Dropping the prompt."
                )
                truncated_prompt = []
            else:
                truncated_prompt = request.prompt_token_ids[-safe_length:]
        else:
            truncated_prompt = request.prompt_token_ids

        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index
        original_prompt = request.prompt_token_ids
        try:
            request.prompt_token_ids = truncated_prompt
            self._copy_tokens(request, req_index)
        finally:
            request.prompt_token_ids = original_prompt

        capped_num_tokens = min(int(request.num_tokens), self.max_model_len)
        self.num_tokens = self.num_tokens.at[req_index].set(capped_num_tokens)
        self.num_tokens_no_spec = self.num_tokens_no_spec.at[req_index].set(capped_num_tokens)
        self.num_computed_tokens = self.num_computed_tokens.at[req_index].set(
            min(int(request.num_computed_tokens), self.max_model_len)
        )

        self.page_table.add_row(request.page_ids, req_index)

        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"
        self._process_sampling_params(sampling_params, req_id, req_index)
        self._process_optional_params(request, sampling_params, req_id, req_index)

    def _process_optional_params(
        self,
        request: EngineRequest,
        sampling_params: SamplingParams,
        req_id: str,
        req_index: int,
    ) -> None:
        """Process optional/sparse parameters."""
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens, sampling_params.all_stop_token_ids)

        if hasattr(request, "generator_seed") and request.generator_seed is not None:
            self.generator_seeds[req_index] = request.generator_seed

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs

        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs

        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self._set_allowed_token_ids(req_id, req_index, sampling_params.allowed_token_ids)

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

    def _set_allowed_token_ids(self, req_id: str, req_index: int, allowed_token_ids: list[int]) -> None:
        """Efficiently set allowed token IDs mask (True = disallowed, False = allowed)."""
        if any((t < 0 or t >= self.vocab_size) for t in allowed_token_ids):
            raise ValueError(f"allowed_token_ids must be within [0, {self.vocab_size})")

        self.has_allowed_token_ids.add(req_id)
        if self.allowed_token_ids_mask is None:
            self.allowed_token_ids_mask = jnp.zeros((self.max_num_reqs, self.vocab_size), dtype=bool)

        self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[req_index].set(True)
        if allowed_token_ids:
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[req_index, allowed_token_ids].set(False)

    def remove_request(self, req_id: str) -> int | None:
        """Remove a request. Must be followed by condense()."""
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

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
        self.logit_bias[req_index] = None
        self.bad_words_token_ids.pop(req_index, None)

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[req_index].set(False)

        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        """Swap states between two indices."""

        old_id_i1, old_id_i2 = self._req_ids[i1], self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = old_id_i2, old_id_i1
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )

        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1] = i2
        self.req_id_to_index[old_id_i2] = i1

        self._swap_array_values(i1, i2)

        swap_dict_values(self.generator_seeds, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        self.logit_bias[i1], self.logit_bias[i2] = self.logit_bias[i2], self.logit_bias[i1]
        self.page_table.swap_row(i1, i2)

    def _swap_array_values(self, i1: int, i2: int) -> None:
        """Efficiently swap array values between two indices using JIT-compiled swap_rows."""

        # Use JIT-compiled swap_rows for all arrays
        self.num_tokens = swap_rows(self.num_tokens, i1, i2)
        self.num_tokens_no_spec = swap_rows(self.num_tokens_no_spec, i1, i2)
        self.num_prompt_tokens = swap_rows(self.num_prompt_tokens, i1, i2)
        self.num_computed_tokens = swap_rows(self.num_computed_tokens, i1, i2)
        self.temperature = swap_rows(self.temperature, i1, i2)
        self.top_p = swap_rows(self.top_p, i1, i2)
        self.top_k = swap_rows(self.top_k, i1, i2)
        self.frequency_penalties = swap_rows(self.frequency_penalties, i1, i2)
        self.presence_penalties = swap_rows(self.presence_penalties, i1, i2)
        self.repetition_penalties = swap_rows(self.repetition_penalties, i1, i2)
        self.min_p = swap_rows(self.min_p, i1, i2)
        self.token_ids = swap_rows(self.token_ids, i1, i2)

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = swap_rows(self.allowed_token_ids_mask, i1, i2)

    def condense(self, empty_req_indices: list[int]) -> None:
        """Efficiently condense the buffer by moving requests to fill empty slots."""
        num_reqs = self.num_reqs
        if num_reqs == 0:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        last_req_index = num_reqs + len(empty_req_indices) - 1

        for empty_index in reversed(empty_req_indices):
            while last_req_index in empty_req_indices and last_req_index > empty_index:
                last_req_index -= 1

            if empty_index >= last_req_index:
                continue

            self._move_request(last_req_index, empty_index)
            last_req_index -= 1

        del self._req_ids[self.num_reqs :]
        del self.req_output_token_ids[self.num_reqs :]

    def _move_request(self, from_idx: int, to_idx: int) -> None:
        """Move a request from one index to another using JIT-compiled move_row."""
        req_id = self._req_ids[from_idx]
        assert req_id is not None

        self._req_ids[to_idx] = req_id
        self._req_ids[from_idx] = None
        self.req_output_token_ids[to_idx] = self.req_output_token_ids[from_idx]
        self.req_output_token_ids[from_idx] = None
        self.req_id_to_index[req_id] = to_idx
        self.token_ids = move_row(self.token_ids, from_idx, to_idx)
        self.num_tokens = move_row(self.num_tokens, from_idx, to_idx)
        self.num_tokens_no_spec = move_row(self.num_tokens_no_spec, from_idx, to_idx)
        self.num_prompt_tokens = move_row(self.num_prompt_tokens, from_idx, to_idx)
        self.num_computed_tokens = move_row(self.num_computed_tokens, from_idx, to_idx)
        self.temperature = move_row(self.temperature, from_idx, to_idx)
        self.top_p = move_row(self.top_p, from_idx, to_idx)
        self.top_k = move_row(self.top_k, from_idx, to_idx)
        self.frequency_penalties = move_row(self.frequency_penalties, from_idx, to_idx)
        self.presence_penalties = move_row(self.presence_penalties, from_idx, to_idx)
        self.repetition_penalties = move_row(self.repetition_penalties, from_idx, to_idx)
        self.min_p = move_row(self.min_p, from_idx, to_idx)
        self.page_table.move_row(from_idx, to_idx)
        self._move_sparse_data(from_idx, to_idx)

    def _move_sparse_data(self, from_idx: int, to_idx: int) -> None:
        """Move sparse/optional data from one index to another."""

        if from_idx in self.generator_seeds:
            self.generator_seeds[to_idx] = self.generator_seeds.pop(from_idx)

        if from_idx in self.min_tokens:
            self.min_tokens[to_idx] = self.min_tokens.pop(from_idx)

        if from_idx in self.bad_words_token_ids:
            self.bad_words_token_ids[to_idx] = self.bad_words_token_ids.pop(from_idx)

        self.logit_bias[to_idx] = self.logit_bias[from_idx]
        self.logit_bias[from_idx] = None

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[to_idx].set(
                self.allowed_token_ids_mask[from_idx]
            )
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[from_idx].set(False)

    def _make_prompt_token_ids_tensor(self) -> jax.Array:
        """Create a padded tensor of prompt token IDs using JIT-compiled pack_prompts."""
        if self.num_reqs == 0:
            return jnp.empty((0, 0), dtype=jnp.int32)

        max_prompt_len = int(jnp.max(self.num_prompt_tokens[: self.num_reqs]))

        # Use the JIT-compiled pack_prompts function
        return pack_prompts(self.token_ids, self.num_prompt_tokens, self.num_reqs, max_prompt_len, self.vocab_size)

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

    def get_request_indices_with_penalty(self) -> jax.Array:
        """Get indices of requests that have any penalty applied."""
        penalty_req_ids = self.frequency_penalties_reqs | self.presence_penalties_reqs | self.repetition_penalties_reqs
        if not penalty_req_ids:
            return jnp.array([], dtype=jnp.int32)

        indices = [self.req_id_to_index[req_id] for req_id in penalty_req_ids]
        return jnp.array(indices, dtype=jnp.int32)

    def get_active_sampling_params(self, req_index: int) -> dict[str, Any]:
        """Get active sampling parameters for a specific request."""
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

    def clear(self) -> None:
        """Clear all data in the buffer."""
        self._req_ids.clear()
        self.req_id_to_index.clear()
        self.req_output_token_ids.clear()

        self.token_ids = jnp.zeros_like(self.token_ids)
        self.num_tokens = jnp.zeros_like(self.num_tokens)
        self.num_tokens_no_spec = jnp.zeros_like(self.num_tokens_no_spec)
        self.num_prompt_tokens = jnp.zeros_like(self.num_prompt_tokens)
        self.num_computed_tokens = jnp.zeros_like(self.num_computed_tokens)

        self.temperature = jnp.full_like(self.temperature, -1.0)
        self.top_p = jnp.ones_like(self.top_p)
        self.top_k = jnp.full_like(self.top_k, self.vocab_size)
        self.min_p = jnp.zeros_like(self.min_p)
        self.frequency_penalties = jnp.zeros_like(self.frequency_penalties)
        self.presence_penalties = jnp.zeros_like(self.presence_penalties)
        self.repetition_penalties = jnp.ones_like(self.repetition_penalties)

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
        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = jnp.zeros_like(self.allowed_token_ids_mask, dtype=bool)


@auto_pytree
class ModelRunnerSamplingMetadata:
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
        """
        Copy sampling tensors slices from `sequence_buffer` to numpy arrays.

        Args:
            sequence_buffer: The input batch containing sampling parameters.
            padded_num_reqs: The padded number of requests.
            generate_params_if_all_greedy: If True, generate sampling parameters
                even if all requests are greedy.
        """

        if sequence_buffer.all_greedy is True and not generate_params_if_all_greedy:
            return cls(
                temperature=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                min_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                top_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                top_k=jnp.zeros((padded_num_reqs,), dtype=jnp.int32),
            )

        num_reqs = sequence_buffer.num_reqs

        # For now, fall back to the original implementation since num_reqs is dynamic
        # and JIT has issues with dynamic slicing
        def fill_slice(arr, fill_val):
            return arr.at[num_reqs:padded_num_reqs].set(fill_val)[:padded_num_reqs]

        return cls(
            temperature=fill_slice(sequence_buffer.temperature, -1.0).astype(jnp.float32),
            min_p=fill_slice(sequence_buffer.min_p, 0.0).astype(jnp.float32),
            top_p=fill_slice(sequence_buffer.top_p, 1.0).astype(jnp.float32),
            top_k=fill_slice(sequence_buffer.top_k, 0).astype(jnp.int32),
        )
