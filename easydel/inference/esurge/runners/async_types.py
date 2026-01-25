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
token sampling to overlap with the next iteration's forward pass. This
pipelining technique can improve throughput by hiding sampling latency
behind the compute-bound forward pass.

The async scheduling workflow is:
    1. Forward pass N produces logits
    2. Sampling for iteration N begins asynchronously
    3. Forward pass N+1 starts immediately (without waiting for sampling)
    4. At the start of iteration N+2, apply results from sampling N
    5. Repeat

This overlapping execution reduces the effective per-step latency by
allowing the host to prepare the next batch while the device is still
executing the previous sampling operation.

Classes:
    AsyncPreResults: Container for storing async sampling results that
        need to be applied in the next iteration.

Note:
    Async scheduling is optional and controlled by the scheduler's
    async_scheduling flag. When disabled, sampling is synchronous and
    results are applied immediately.
"""

from dataclasses import dataclass

from jax import Array

from .states import CachedRequestState


@dataclass
class AsyncPreResults:
    """Stores previous iteration's results for async scheduling.

    When async scheduling is enabled, token sampling happens asynchronously
    while the next forward pass begins. This dataclass stores the results
    from the previous iteration that need to be applied to the sequence
    buffer at the start of the next iteration.

    The class is intentionally a simple dataclass (not a PyTree) since it
    is only used for host-side bookkeeping and never passes through JAX
    transformations.

    Attributes:
        req_ids (list[str]): List of request IDs from the previous iteration,
            in the same order as the batch. Used to map sampled tokens back
            to their corresponding requests.
        next_tokens (Array): JAX array of sampled tokens with shape [batch_size].
            Initially device-resident; async copy to host is initiated when
            this object is created. The actual host transfer completes when
            the array is accessed via device_get().
        request_seq_lens (list[tuple[int, CachedRequestState, int]]): List of
            tuples containing (req_idx, req_state, seq_len) for each request
            that generated tokens. req_idx is the index in the batch,
            req_state is the CachedRequestState object, and seq_len is the
            sequence length after generation.
        discard_sampled_tokens_req_indices (list[int]): Indices of requests
            whose sampled tokens should be discarded. This includes requests
            that were in partial prefill (still processing prompt) and thus
            should not append generated tokens.
        placeholder_req_id_to_index (dict[str, int]): Mapping from request ID
            to buffer index for placeholder token replacement. When async
            scheduling is used, placeholder tokens (0) are inserted immediately
            and replaced with actual sampled tokens in the next iteration.

    Example:
        >>> # After async sampling in the runner
        >>> async_results = AsyncPreResults(
        ...     req_ids=["req1", "req2", "req3"],
        ...     next_tokens=jax_array_on_device,  # Shape [3]
        ...     request_seq_lens=[
        ...         (0, req_state1, 10),  # req1 at length 10
        ...         (1, req_state2, 15),  # req2 at length 15
        ...         (2, req_state3, 8),   # req3 at length 8
        ...     ],
        ...     discard_sampled_tokens_req_indices=[2],  # Discard req3's token
        ...     placeholder_req_id_to_index={"req1": 0, "req2": 1},
        ... )
        >>>
        >>> # In next iteration, apply these results
        >>> runner._modify_prev_results()
    """

    req_ids: list[str]
    next_tokens: Array
    request_seq_lens: list[tuple[int, CachedRequestState, int]]
    discard_sampled_tokens_req_indices: list[int]
    placeholder_req_id_to_index: dict[str, int]
