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
