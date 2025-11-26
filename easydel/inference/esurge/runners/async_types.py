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

"""Async scheduling types for overlapping token sampling with forward pass.

This module provides data structures for async scheduling, which allows
token sampling to overlap with the next iteration's forward pass.
"""

from dataclasses import dataclass

from jax import Array

from .states import CachedRequestState


@dataclass
class AsyncPreResults:
    """Stores previous iteration's results for async scheduling.

    When async scheduling is enabled, token sampling happens asynchronously
    while the next forward pass begins. This class stores the results from
    the previous iteration that need to be applied to the sequence buffer.

    Attributes:
        req_ids: List of request IDs from the previous iteration.
        next_tokens: JAX array of sampled tokens (on device, async copy to host).
        request_seq_lens: List of (req_idx, req_state, seq_len) tuples for
            requests that generated tokens.
        discard_sampled_tokens_req_indices: Indices of requests whose tokens
            should be discarded (e.g., partial prefill requests).
        placeholder_req_id_to_index: Mapping from request ID to index for
            placeholder token replacement.

    Example:
        >>> # After async sampling
        >>> async_results = AsyncPreResults(
        ...     req_ids=["req1", "req2"],
        ...     next_tokens=jax_array_on_device,
        ...     request_seq_lens=[(0, req_state1, 10), (1, req_state2, 15)],
        ...     discard_sampled_tokens_req_indices=[],
        ...     placeholder_req_id_to_index={"req1": 0, "req2": 1},
        ... )
        >>> # In next iteration, apply these results
        >>> runner._modify_prev_results()
    """

    req_ids: list[str]
    next_tokens: Array
    request_seq_lens: list[tuple[int, CachedRequestState, int]]
    discard_sampled_tokens_req_indices: list[int]
    placeholder_req_id_to_index: dict[str, int]
