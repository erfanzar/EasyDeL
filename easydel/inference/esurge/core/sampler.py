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

"""Token sampling functions for eSurge inference.

This module provides efficient token sampling implementations using binary search
for top-k, top-p, and min-p filtering. The sampling functions are pure JAX functions
(no stateful modules) and support dtype-aware optimizations for bfloat16.

Functions:
    - sample_tokens: Main entry point for token sampling
    - _regular_sample: Stochastic sampling with filtering
    - _greedy_sample: Deterministic argmax sampling
    - _apply_min_p_mask: Min-p probability filtering
    - _apply_penalties: Linear penalty application
"""

import jax
from jax import lax
from jax import numpy as jnp

from .binary_search import (
    apply_min_p_mask,
    apply_penalties,
    apply_topk_mask,
    apply_topp_mask,
)
from .sampling_metadata import SamplingMetadata


def _greedy_sample(logits: jax.Array) -> jax.Array:
    """Greedy sampling: select token with highest probability.

    Args:
        logits: Input logits [batch, vocab_size].

    Returns:
        Token IDs with maximum logit value [batch].
    """
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _regular_sample(logits: jax.Array, sampling_metadata: SamplingMetadata, rng: jax.Array) -> jax.Array:
    """Stochastic sampling with top-k, top-p, and min-p filtering.

    Applies filtering in order: top-k → top-p → temperature → min-p, then samples
    from the resulting distribution. Uses binary search for efficient filtering
    without sorting. Automatically selects bfloat16-optimized variants when applicable.

    Args:
        logits: Input logits [batch, vocab_size].
        sampling_metadata: Sampling configuration (temperatures, top_k, top_p, etc.).
        rng: JAX random key for sampling.

    Returns:
        Sampled token IDs [batch].

    Performance:
        - Dtype-aware: Uses bfloat16 variants when input is bfloat16
        - Binary search: O(32 reductions) for top-k/p vs O(V log V) for sorting
        - Vectorized: Per-sample parameters via vmap
    """
    # Convert to float32 for numerical stability in sampling operations
    logits = logits.astype(jnp.float32)

    min_val = -1e10

    # Top-k filtering
    need_top_k = jnp.any(sampling_metadata.top_ks > 0)

    def apply_topk(legi):
        def topk_per_sample(logits_i, k_i):
            return lax.cond(k_i > 0, lambda: apply_topk_mask(logits_i[None, :], k_i, min_val)[0], lambda: logits_i)

        return jax.vmap(topk_per_sample)(legi, sampling_metadata.top_ks)

    logits = lax.cond(need_top_k, apply_topk, lambda legi: legi, logits)

    # Top-p filtering
    need_top_p = jnp.any(sampling_metadata.top_ps < 1.0)

    def apply_topp(legi):
        def topp_per_sample(logits_i, p_i):
            return lax.cond(p_i < 1.0, lambda: apply_topp_mask(logits_i[None, :], p_i, min_val)[0], lambda: logits_i)

        return jax.vmap(topp_per_sample)(legi, sampling_metadata.top_ps)

    logits = lax.cond(need_top_p, apply_topp, lambda legi: legi, logits)

    # Temperature scaling (apply before min-p to match sglang-jax behavior)
    need_temp = jnp.any(sampling_metadata.temperatures != 1.0)

    def apply_temp(legi):
        return legi / sampling_metadata.temperatures

    logits = lax.cond(need_temp, apply_temp, lambda legi: legi, logits)

    # Min-p filtering (applied after temperature scaling)
    logits = lax.cond(
        sampling_metadata.need_min_p_sampling,
        lambda legi: apply_min_p_mask(legi, sampling_metadata),
        lambda legi: legi,
        logits,
    )

    # Sample from filtered distribution
    batch_size = logits.shape[0]

    def sample_one(i):
        per_sample_rng = jax.random.fold_in(rng, i)
        return jax.random.categorical(per_sample_rng, logits[i]).astype(jnp.int32)

    samples = jax.vmap(sample_one)(jnp.arange(batch_size))
    return samples


def sample_tokens(
    logits: jax.Array,
    sampling_metadata: SamplingMetadata,
    rng: jax.Array,
) -> jax.Array:
    """Sample next tokens from logits with advanced filtering.

    Main entry point for token sampling in eSurge. Supports greedy and stochastic
    sampling with top-k, top-p, min-p filtering, temperature scaling, and penalties.
    Automatically optimizes for bfloat16 inputs.

    Args:
        logits: Model output logits [batch, vocab_size].
        sampling_metadata: Sampling configuration including:
            - temperatures: Temperature values for scaling
            - top_ks: Top-k filtering thresholds
            - top_ps: Top-p (nucleus) filtering thresholds
            - min_ps: Min-p filtering thresholds
            - is_all_greedy: Whether to use greedy sampling
            - do_penalties: Whether to apply penalties
            - linear_penalty: Optional penalty matrix
        rng: JAX random key for stochastic sampling.

    Returns:
        Sampled token IDs [batch].

    Example:
        >>> logits = jnp.ones((4, 32000))  # [batch=4, vocab=32000]
        >>> metadata = SamplingMetadata(
        ...     temperatures=jnp.array([[0.8], [0.8], [0.8], [0.8]]),
        ...     top_ks=jnp.array([50, 50, 50, 50]),
        ...     top_ps=jnp.array([0.9, 0.9, 0.9, 0.9]),
        ...     min_ps=jnp.array([0.0, 0.0, 0.0, 0.0]),
        ...     is_all_greedy=False,
        ...     need_min_p_sampling=False,
        ...     do_penalties=False,
        ...     linear_penalty=None,
        ... )
        >>> tokens = sample_tokens(logits, metadata, rng_key)
        >>> tokens.shape
        (4,)
    """
    logits = lax.cond(
        sampling_metadata.do_penalties,
        lambda legi: apply_penalties(legi, sampling_metadata),
        lambda legi: legi,
        logits,
    )

    return lax.cond(
        sampling_metadata.is_all_greedy,
        lambda: _greedy_sample(logits),
        lambda: _regular_sample(logits, sampling_metadata, rng),
    )
