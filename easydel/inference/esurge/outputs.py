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

"""Output data structures for the eSurge model runner.

This module defines data structures for representing log probabilities and
model runner outputs in both tensor and list formats for efficient processing.

Classes:
    LogprobsLists: Log probability data in Python list format.
    LogprobsTensors: Log probability data in JAX array format.
    ModelRunnerOutput: Complete output from a model runner step.

Functions:
    swap_dict_values: Helper function to swap dictionary values.

Example:
    >>> from easydel.inference.esurge.outputs import (
    ...     LogprobsLists,
    ...     LogprobsTensors,
    ...     ModelRunnerOutput
    ... )
    >>>
    >>> # Create logprobs in list format
    >>> logprobs = LogprobsLists(
    ...     logprob_token_ids=[[1, 2, 3], [4, 5, 6]],
    ...     logprobs=[[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]],
    ...     sampled_token_ranks=[0, 1]
    ... )
    >>>
    >>> # Slice to get a subset
    >>> sliced = logprobs.slice(0, 1)
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

from jax import Array
from jax import numpy as jnp

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class LogprobsLists(NamedTuple):
    """Log probability data in Python list format.

    Stores log probability information for sampled tokens in a format
    suitable for serialization and Python-level processing.

    Attributes:
        logprob_token_ids: List of lists containing top-k token IDs for each
            position. Shape: [num_positions][num_top_k].
        logprobs: List of lists containing log probabilities for the top-k
            tokens at each position. Shape: [num_positions][num_top_k].
        sampled_token_ranks: List of ranks of the sampled tokens within the
            top-k at each position. Shape: [num_positions].

    Example:
        >>> logprobs = LogprobsLists(
        ...     logprob_token_ids=[[100, 200, 300], [150, 250, 350]],
        ...     logprobs=[[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.2]],
        ...     sampled_token_ranks=[0, 1]
        ... )
        >>> # Access rank of first sampled token
        >>> print(logprobs.sampled_token_ranks[0])  # 0
    """

    logprob_token_ids: list[list[int]]
    """Top-k token IDs for each position."""

    logprobs: list[list[float]]
    """Log probabilities for top-k tokens."""

    sampled_token_ranks: list[int]
    """Ranks of sampled tokens within top-k."""

    def slice(self, start: int, end: int) -> "LogprobsLists":
        """Extract a slice of logprobs data.

        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive).

        Returns:
            New LogprobsLists containing data from positions [start:end].

        Example:
            >>> full = LogprobsLists(
            ...     logprob_token_ids=[[1], [2], [3]],
            ...     logprobs=[[-0.1], [-0.2], [-0.3]],
            ...     sampled_token_ranks=[0, 0, 0]
            ... )
            >>> sliced = full.slice(1, 3)
            >>> len(sliced.sampled_token_ranks)  # 2
        """
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):
    """Log probability data in JAX array format.

    Stores log probability information in JAX arrays for efficient
    GPU processing and computation.

    Attributes:
        logprob_token_ids: JAX array of top-k token IDs.
            Shape: [num_positions, num_top_k], dtype: int32.
        logprobs: JAX array of log probabilities for top-k tokens.
            Shape: [num_positions, num_top_k], dtype: float32.
        selected_token_ranks: JAX array of ranks of selected tokens.
            Shape: [num_positions], dtype: int32.

    Example:
        >>> import jax.numpy as jnp
        >>> tensors = LogprobsTensors(
        ...     logprob_token_ids=jnp.array([[100, 200], [150, 250]]),
        ...     logprobs=jnp.array([[-0.5, -1.0], [-0.3, -0.8]]),
        ...     selected_token_ranks=jnp.array([0, 1])
        ... )
        >>> # Convert to lists for serialization
        >>> lists = tensors.tolists()
    """

    logprob_token_ids: Array
    """Top-k token IDs as JAX array [num_positions, num_top_k]."""

    logprobs: Array
    """Log probabilities as JAX array [num_positions, num_top_k]."""

    selected_token_ranks: Array
    """Ranks of selected tokens as JAX array [num_positions]."""

    def tolists(self) -> LogprobsLists:
        """Convert tensor format to list format.

        Converts JAX arrays to Python lists for serialization or
        Python-level processing.

        Returns:
            LogprobsLists with the same data in list format.

        Note:
            This operation transfers data from device to host and
            converts to Python lists, which may be slow for large data.
        """
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create an empty LogprobsTensors with the specified shape.

        Useful for pre-allocating output buffers before computation.

        Args:
            num_positions: Number of sequence positions.
            num_tokens_per_position: Number of top-k tokens per position.

        Returns:
            LogprobsTensors with uninitialized arrays of the correct shape.

        Example:
            >>> tensors = LogprobsTensors.empty(10, 5)
            >>> tensors.logprob_token_ids.shape  # (10, 5)
        """
        logprob_token_ids = jnp.empty((num_positions, num_tokens_per_position), dtype=jnp.int32)
        logprobs = jnp.empty_like(logprob_token_ids, dtype=jnp.float32)
        selected_token_ranks = jnp.empty(num_positions, dtype=jnp.int32)
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass
class ModelRunnerOutput:
    """Complete output from a model runner step.

    Contains all outputs from a single forward pass of the model runner,
    including sampled tokens, log probabilities, and request tracking.

    Attributes:
        req_ids: List of request IDs processed in this batch.
        req_id_to_index: Mapping from request ID to batch index.
        sampled_token_ids: List of lists of sampled token IDs for each request.
            Multiple tokens per request for beam search or speculative decoding.
        spec_token_ids: Optional speculative token IDs for each request.
            None if speculative decoding is not enabled.
        logprobs: Optional log probabilities for sampled tokens in list format.
        prompt_logprobs_dict: Dictionary mapping request IDs to their prompt
            log probabilities in tensor format. Used for scoring prompts.
        finished_sending: Set of request IDs that finished sending KV cache
            in a KV transfer scenario.
        finished_recving: Set of request IDs that finished receiving KV cache.
        num_nans_in_logits: Dictionary mapping request IDs to the count of
            NaN values detected in their logits (indicates model issues).
        token_logprobs: Dictionary mapping request IDs to their token-level
            log probabilities for the sampled tokens.

    Example:
        >>> output = ModelRunnerOutput(
        ...     req_ids=["req_1", "req_2"],
        ...     req_id_to_index={"req_1": 0, "req_2": 1},
        ...     sampled_token_ids=[[100], [200]],
        ...     spec_token_ids=None,
        ...     logprobs=None,
        ...     prompt_logprobs_dict={}
        ... )
        >>> # Get index for a request
        >>> idx = output.req_id_to_index["req_1"]  # 0
    """

    req_ids: list[str]
    """Request IDs processed in this batch."""

    req_id_to_index: dict[str, int]
    """Mapping from request ID to output batch index."""

    sampled_token_ids: list[list[int]]
    """Sampled token IDs for each request."""

    spec_token_ids: list[list[int]] | None
    """Speculative token IDs if enabled."""

    logprobs: LogprobsLists | None
    """Log probabilities for sampled tokens."""

    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    """Prompt log probabilities by request ID."""

    req_id_to_row_index: dict[str, int] | None = None
    """Optional mapping from request ID to sequence-buffer row index."""

    finished_sending: set[str] | None = None
    """Request IDs that finished sending KV cache."""

    finished_recving: set[str] | None = None
    """Request IDs that finished receiving KV cache."""

    num_nans_in_logits: dict[str, int] | None = None
    """Count of NaN values in logits by request ID."""

    token_logprobs: dict[str, float] | None = None
    """Token-level log probabilities by request ID."""


def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """Swap values for two keys in a dictionary.

    Exchanges the values associated with key1 and key2. If a key is missing,
    it will be removed from the dictionary after the swap.

    Args:
        obj: Dictionary to modify in place.
        key1: First key to swap.
        key2: Second key to swap.

    Example:
        >>> d = {"a": 1, "b": 2}
        >>> swap_dict_values(d, "a", "b")
        >>> d  # {"a": 2, "b": 1}
        >>>
        >>> # Handles missing keys
        >>> d = {"a": 1}
        >>> swap_dict_values(d, "a", "b")
        >>> d  # {"b": 1}

    Note:
        This function modifies the dictionary in place and does not
        return a value.
    """
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)
