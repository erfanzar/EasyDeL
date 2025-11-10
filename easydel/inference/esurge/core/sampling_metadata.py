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

"""Sampling metadata structures for token sampling operations.

This module defines the SamplingMetadata PyTree structure that consolidates
all sampling parameters (temperature, top-k, top-p, min-p, penalties) into
a single JAX-compatible structure for efficient JIT compilation.

Key structures:
    - SamplingMetadata: Frozen PyTree containing all sampling configuration
"""

from __future__ import annotations

import jax
from eformer.pytree import auto_pytree


@auto_pytree(frozen=True)
class SamplingMetadata:
    """Consolidated sampling parameters for JIT-compiled token sampling.

    This frozen PyTree structure packages all sampling configuration for a batch
    of requests, enabling efficient JIT compilation and device transfer. Created
    on the host and passed to device-side sampling functions.

    Attributes:
        temperatures: Temperature values for controlling randomness [batch_size, 1].
            Higher values (>1.0) increase diversity, lower values (<1.0) make
            sampling more deterministic.
        top_ps: Top-p (nucleus) sampling thresholds [batch_size]. Only tokens
            whose cumulative probability is ≥ top_p are sampled from.
        top_ks: Top-k sampling thresholds [batch_size]. Only the k highest
            probability tokens are considered for sampling.
        min_ps: Min-p sampling thresholds [batch_size]. Filters tokens where
            p(token) < min_p * max(p(token)).
        sampling_seeds: Optional random seeds for reproducible sampling [batch_size].
            If provided, enables deterministic sampling for debugging.
        is_all_greedy: Static flag indicating all requests use greedy decoding.
            When True, skips random sampling entirely for efficiency.
        need_min_p_sampling: Static flag indicating min-p filtering is needed
            for at least one request in the batch.
        do_penalties: Static flag indicating linear penalties should be applied.
        linear_penalty: Optional penalty matrix [batch_size, vocab_size] for
            frequency/presence penalties or token biases.

    Note:
        Arrays are device-side tensors. Flags are Python bools used as static
        arguments for JIT tracing to enable compile-time optimizations.

    Example:
        >>> metadata = SamplingMetadata(
        ...     temperatures=jnp.array([[1.0], [0.8]]),
        ...     top_ps=jnp.array([0.9, 0.95]),
        ...     top_ks=jnp.array([50, 40]),
        ...     min_ps=jnp.array([0.0, 0.0]),
        ...     sampling_seeds=None,
        ...     is_all_greedy=False,
        ...     need_min_p_sampling=False,
        ...     do_penalties=False,
        ...     linear_penalty=None,
        ... )
    """

    # Sampling parameters (JAX arrays on device)
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array
    sampling_seeds: jax.Array | None

    # Flags (static for JIT tracing)
    is_all_greedy: bool
    need_min_p_sampling: bool
    do_penalties: bool

    # Penalties (optional, on device)
    linear_penalty: jax.Array | None
