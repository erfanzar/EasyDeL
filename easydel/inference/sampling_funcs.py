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

"""Sampling functions for token generation in language models.

This module provides efficient JAX-based sampling functions for selecting
the next token during text generation. It includes implementations of
top-p (nucleus) sampling and dynamic sampling with various penalty mechanisms.

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
            Tokens with cumulative probability exceeding this threshold
            are filtered out.
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
    keep_mask_k = cumulative_probs_k <= jnp.expand_dims(top_p, axis=-1)
    keep_mask_k = keep_mask_k.at[..., 0].set(True)
    filtered_top_k_logits = jnp.where(keep_mask_k, top_k_logits, -jnp.inf)
    sampled_k_index = jax.random.categorical(rng, filtered_top_k_logits)
    return jnp.take_along_axis(top_k_indices, jnp.expand_dims(sampled_k_index, axis=-1), axis=-1).squeeze(-1)


vmaped_sample_top_p_efficient = jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, None, None), out_axes=0)
"""Vectorized version of sample_top_p_efficient for batch processing.

Maps the sampling function over batches with independent top_p and
temperature values per sample while sharing the random key.

Args:
    logits: Batched logits of shape (batch_size, vocab_size).
    top_p: Per-sample top_p values of shape (batch_size,).
    temperature: Per-sample temperatures of shape (batch_size,).
    rng: Shared random key.
    top_k_for_computation: Number of top tokens to consider.

Returns:
    Sampled tokens of shape (batch_size,).
"""


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None), out_axes=(0))
def dynamic_sample_tokens(
    tokens: jax.Array,  # [vocab / 1] i4
    length: jax.Array,  # [vocab / 1] i4
    logits: jax.Array,  # [vocab] f4/f2
    top_p: jax.Array,  # [1] f4/f2
    temperature: jax.Array,  # [1] f4/f2
    random_sampling: jax.Array,  # [1] b1
    presence_penalty: jax.Array,  # [1] f4/f2
    frequency_penalty: jax.Array,  # [1] f4/f2
    repetition_penalty: jax.Array,  # [1] f4/f2
    rngs: jax.Array,  # [*] rng
) -> jax.Array:
    """Dynamically sample tokens with configurable penalties and sampling modes.

    This function provides flexible token sampling that can switch between
    greedy and random (top-p) sampling modes, while applying various penalty
    mechanisms to prevent repetition.

    Args:
        tokens: Previously generated tokens of shape (sequence_length,) used
            for computing penalties.
        length: Current sequence length (integer array).
        logits: Logits tensor of shape (vocab_size,) for the current position.
        top_p: Top-p threshold for nucleus sampling.
        temperature: Temperature for scaling logits.
        random_sampling: Boolean flag - True for random sampling, False for greedy.
        presence_penalty: Penalty applied to tokens that appear in the sequence.
        frequency_penalty: Penalty proportional to token frequency in the sequence.
        repetition_penalty: Multiplicative penalty for repeated tokens.
        rngs: Random key for sampling.

    Returns:
        Sampled token index of shape (1,).

    Note:
        This function is vectorized with jax.vmap and expects batched inputs
        along axis 0 for all arguments except rngs.
    """

    def _random_sampling_fn(
        tokens,
        length,
        logits,
        top_p,
        temperature,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
    ):
        """Apply random sampling with penalties."""
        logits = logits.reshape(1, -1)
        processors = LogitsProcessorList()
        processors.append(PresencePenaltyLogitsProcessor(presence_penalty))
        processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        logits = processors(tokens.reshape(1, -1), logits, length)
        return sample_top_p_efficient(logits, top_p, temperature, rngs)[:, None]

    def _gready_sampling_fn(
        tokens,
        length,
        logits,
        top_p,
        temperature,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
    ):
        """Apply greedy sampling (argmax)."""
        return jnp.argmax(logits.reshape(1, -1), axis=-1)[:, None]

    return jax.lax.cond(
        random_sampling,
        _random_sampling_fn,
        _gready_sampling_fn,
        tokens,
        length,
        logits,
        top_p.astype(logits.dtype),
        temperature.astype(logits.dtype),
        presence_penalty.astype(logits.dtype),
        frequency_penalty.astype(logits.dtype),
        repetition_penalty.astype(logits.dtype),
    )
