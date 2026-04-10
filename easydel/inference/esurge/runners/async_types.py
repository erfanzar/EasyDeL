# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Async scheduling payloads for overlapped TPU/JAX execution.

The overlap path keeps model execution on the scheduler thread, but returns
device-backed sampled-token arrays before the host blocks on copying them.
These payloads let the lifecycle loop prefetch the next scheduler step while
the device is still producing the current batch's sampled tokens.
"""

from dataclasses import dataclass

from jax import Array

from .states import CachedRequestState


@dataclass
class AsyncWindowResult:
    """Host-copy payload for one runner window.

    Attributes:
        req_ids: Active request ids for this window, in scheduler output order.
        row_positions: Indices into ``sampled_token_ids`` / ``token_logprobs`` that
            correspond to each request id.
        sampled_token_ids: Sampled token tensor with host-copy already initiated.
        valid_mask: Whether each request should materialize a sampled token.
        token_logprobs: Optional per-request sampler metric payload with host-copy
            already initiated.
    """

    req_ids: list[str]
    row_positions: list[int]
    sampled_token_ids: Array
    valid_mask: list[bool]
    token_logprobs: Array | None = None


@dataclass
class AsyncPreResults:
    """Stores previous iteration's sampled-token payloads for async scheduling."""

    windows: list[AsyncWindowResult]
    request_seq_lens: list[tuple[int, int, CachedRequestState, int]]
