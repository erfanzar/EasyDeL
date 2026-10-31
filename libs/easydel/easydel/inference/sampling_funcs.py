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

"""Sampling functions for token generation in language models.

This module provides efficient JAX-based sampling functions for selecting
next tokens, including nucleus sampling and dynamic sampling with penalties.

Functions:
    sample_top_p_efficient: Efficient top-p sampling with temperature scaling.
    vmaped_sample_top_p_efficient: Vectorized version of top-p sampling.
    dynamic_sample_tokens: Dynamic token sampling with configurable penalties.

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from easydel.inference.sampling_funcs import sample_top_p_efficient
    >>> logits = jnp.array([[1.0, 2.0, 0.5, 3.0]])
    >>> top_p = jnp.array([0.9])
    >>> temperature = jnp.array([0.7])
    >>> rng = jax.random.PRNGKey(42)
    >>> token = sample_top_p_efficient(logits, top_p, temperature, rng)
"""

from __future__ import annotations

import os
from functools import partial

import jax
from jax import numpy as jnp

from .logits_process import (
    FrequencyPenaltyLogitsProcessor,
    LogitsProcessorList,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

TOPK_FOR_COMPUTE = int(os.getenv("EASYDEL_TOPK_FOR_COMPUTE", "64"))


def sample_top_p_efficient(
    logits: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    rng: jax.random.PRNGKey,
    top_k_for_computation: int = TOPK_FOR_COMPUTE,
) -> jax.Array:
    """Perform efficient top-p (nucleus) sampling on logits.

    This function implements nucleus sampling by first selecting the top-k
    tokens (for computational efficiency), then filtering based on cumulative
    probability to achieve the top-p cutoff.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size) containing
            unnormalized log probabilities for each token.
        top_p: Top-p threshold tensor of shape (batch_size,) or scalar.
            The smallest set of top tokens whose cumulative probability
            reaches this threshold is kept (the token that crosses the
            threshold is included); the rest are filtered out.
        temperature: Temperature tensor of shape (batch_size,) or scalar.
            Higher values increase randomness, lower values make sampling
            more deterministic.
        rng: JAX random key for sampling.
        top_k_for_computation: Number of top tokens to consider for
            efficiency. Defaults to EASYDEL_TOPK_FOR_COMPUTE env var or 64.

    Returns:
        Sampled token indices of shape (batch_size,).

    Note:
        The function uses a two-stage approach for efficiency:
        1. Select top-k tokens based on logits
        2. Apply top-p filtering within the top-k set
        This reduces memory and computation while maintaining quality.
    """
    vocab_size = logits.shape[-1]
    effective_k = min(top_k_for_computation, vocab_size)
    safe_temperature = jnp.where(temperature > 1e-6, temperature, 1.0)
    scaled_logits = logits / jnp.expand_dims(safe_temperature, axis=-1)
    top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, k=effective_k)
    top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)
    cumulative_probs_k = jnp.cumsum(top_k_probs, axis=-1)
    # Keep every token whose *preceding* cumulative mass is still below ``top_p`` so the token that crosses the
    # threshold is retained (standard nucleus sampling, identical to ``TopPLogitsWarper``). Comparing the inclusive
    # cumulative sum instead would drop that token and shrink the nucleus below the requested mass.
    preceding_mass_k = cumulative_probs_k - top_k_probs
    keep_mask_k = preceding_mass_k < jnp.expand_dims(top_p, axis=-1)
    keep_mask_k = keep_mask_k.at[..., 0].set(True)
    filtered_top_k_logits = jnp.where(keep_mask_k, top_k_logits, -jnp.inf)
    sampled_k_index = jax.random.categorical(rng, filtered_top_k_logits)
    return jnp.take_along_axis(top_k_indices, jnp.expand_dims(sampled_k_index, axis=-1), axis=-1).squeeze(-1)


def vmaped_sample_top_p_efficient(
    logits: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    rng: jax.random.PRNGKey,
    top_k_for_computation: int = TOPK_FOR_COMPUTE,
) -> jax.Array:
    """Vectorize nucleus sampling with per-row controls and a shared random key.

    Args:
        logits: Batched logits of shape ``(batch_size, vocab_size)``.
        top_p: Per-row nucleus thresholds of shape ``(batch_size,)``.
        temperature: Per-row temperatures of shape ``(batch_size,)``.
        rng: Shared JAX random key.
        top_k_for_computation: Maximum number of candidates per row.

    Returns:
        Sampled token indices of shape ``(batch_size,)``.
    """
    # Supply the optional argument positionally to vmap as well: its in_axes
    # describes five inputs even when callers rely on the public default.
    return jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, None, None), out_axes=0)(
        logits, top_p, temperature, rng, top_k_for_computation
    )


def dynamic_sample_tokens(
    tokens: jax.Array,
    length: jax.Array,
    logits: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    random_sampling: jax.Array,
    presence_penalty: jax.Array,
    frequency_penalty: jax.Array,
    repetition_penalty: jax.Array,
    rngs: jax.Array,
) -> jax.Array:
    """Dynamically sample a batch with penalties and greedy or random decoding.

    Args:
        tokens: Batched previous tokens, shape ``(batch_size, sequence_length)``.
        length: Current sequence length per entry.
        logits: Batched next-token logits, shape ``(batch_size, vocab_size)``.
        top_p: Per-entry nucleus thresholds.
        temperature: Per-entry sampling temperatures.
        random_sampling: Per-entry boolean selecting random or greedy decoding.
        presence_penalty: Per-entry fixed penalty for previously seen tokens.
        frequency_penalty: Per-entry count-proportional penalty.
        repetition_penalty: Per-entry multiplicative repetition penalty.
        rngs: Shared JAX random key, not batched.

    Returns:
        Token indices of shape ``(batch_size, 1, 1)``, retaining the original
        vectorized API's singleton dimensions.
    """
    # Keep the key unmapped for positional and keyword callers alike. JAX maps
    # keyword arguments on axis zero regardless of the explicit in_axes tuple.
    return _dynamic_sample_tokens(
        tokens,
        length,
        logits,
        top_p,
        temperature,
        random_sampling,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        rngs,
    )


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None), out_axes=0)
def _dynamic_sample_tokens(
    tokens: jax.Array,
    length: jax.Array,
    logits: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    random_sampling: jax.Array,
    presence_penalty: jax.Array,
    frequency_penalty: jax.Array,
    repetition_penalty: jax.Array,
    rngs: jax.Array,
) -> jax.Array:
    """Sample tokens with configurable penalties and greedy or random decoding.

    Args:
        tokens: Previous tokens, batched as ``(batch_size, sequence_length)``.
        length: Current valid sequence length for each batch entry.
        logits: Current logits, batched as ``(batch_size, vocab_size)``.
        top_p: Per-entry nucleus thresholds.
        temperature: Per-entry temperature scaling.
        random_sampling: Per-entry boolean; false selects greedy argmax.
        presence_penalty: Per-entry fixed penalty for previously seen tokens.
        frequency_penalty: Per-entry count-proportional penalty.
        repetition_penalty: Per-entry multiplicative repetition penalty.
        rngs: Shared JAX random key, not mapped over the batch.

    Returns:
        Token indices of shape ``(batch_size, 1, 1)``, preserving the original
        vectorized API's singleton dimensions.
    """

    def _random_sampling_fn(tokens, length, logits, top_p, temperature, presence, frequency, repetition):
        """Apply the shared logits processors before nucleus sampling."""
        logits = logits.reshape(1, -1)
        processors = LogitsProcessorList(
            [
                PresencePenaltyLogitsProcessor(presence),
                FrequencyPenaltyLogitsProcessor(frequency),
                RepetitionPenaltyLogitsProcessor(repetition),
            ]
        )
        logits = processors(tokens.reshape(1, -1), logits, length)
        return sample_top_p_efficient(logits, top_p, temperature, rngs)[:, None]

    def _greedy_sampling_fn(tokens, length, logits, top_p, temperature, presence, frequency, repetition):
        """Keep the original greedy argmax contract."""
        return jnp.argmax(logits.reshape(1, -1), axis=-1)[:, None]

    return jax.lax.cond(
        random_sampling,
        _random_sampling_fn,
        _greedy_sampling_fn,
        tokens,
        length,
        logits,
        top_p.astype(logits.dtype),
        temperature.astype(logits.dtype),
        presence_penalty.astype(logits.dtype),
        frequency_penalty.astype(logits.dtype),
        repetition_penalty.astype(logits.dtype),
    )
