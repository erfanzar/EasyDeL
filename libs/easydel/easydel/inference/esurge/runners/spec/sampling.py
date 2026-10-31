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

"""Sampled-distribution scores shared by synchronous and deferred runner output.

Only compact top-token summaries cross to the host. Scoring does not draw RNGs
or alter the sampler's penalty counters.
"""

from functools import partial

import jax
from jax import numpy as jnp

from ...core.binary_search import apply_topk_mask, apply_topk_mask_bf16, apply_topp_mask, apply_topp_mask_bf16
from ...core.sampler import apply_history_penalties


@partial(jax.jit, static_argnames=("num_logprobs",))
def summarize_logprobs(log_probs, sampled_tokens, *, num_logprobs: int):
    """Return sampled-first candidate IDs, scores and vocabulary ranks per position.

    ``log_probs`` is the actual behavior distribution, not raw model logits.
    The sampled token is always included, including when ``num_logprobs=0``.
    """
    sampled_tokens = sampled_tokens.reshape(-1).astype(jnp.int32)
    sampled_scores = jnp.take_along_axis(log_probs, sampled_tokens[:, None], axis=-1)
    ranks = jnp.sum(log_probs > sampled_scores, axis=-1).astype(jnp.int32)
    k = min(max(0, int(num_logprobs)), int(log_probs.shape[-1]))
    top_scores, top_ids = jax.lax.top_k(log_probs, k)
    return (
        jnp.concatenate((sampled_tokens[:, None], top_ids), axis=-1),
        jnp.concatenate((sampled_scores, top_scores), axis=-1),
        ranks,
    )


@partial(jax.jit, static_argnames=("num_logprobs",))
def score_sampled_tokens(
    logits,
    sampled_tokens,
    token_history,
    seq_lens,
    temperature,
    top_p,
    top_k,
    min_p,
    frequency_penalty,
    presence_penalty,
    repetition_penalty,
    *,
    num_logprobs: int,
):
    """Score normal sampler output with its exact dtype, penalties and filters.

    Parameter casts and filter order mirror SamplerExecutor and core.sampler:
    history penalties, top-k, top-p, temperature, min-p. Greedy rows are a
    point mass, so their emitted token has behavior log probability zero.
    All probability reductions accumulate in float32.
    """
    active = jnp.ones((logits.shape[0],), dtype=jnp.bool_)
    logits = apply_history_penalties(
        logits,
        token_history=token_history,
        seq_lens=seq_lens,
        active_mask=active,
        presence_penalties=presence_penalty,
        frequency_penalties=frequency_penalty,
        repetition_penalties=repetition_penalty,
    )
    temperature = temperature.astype(logits.dtype).reshape(-1, 1)
    top_p = top_p.astype(logits.dtype)
    min_p = min_p.astype(logits.dtype).reshape(-1, 1)
    topk_mask = apply_topk_mask_bf16 if logits.dtype == jnp.bfloat16 else apply_topk_mask
    topp_mask = apply_topp_mask_bf16 if logits.dtype == jnp.bfloat16 else apply_topp_mask
    if logits.dtype != jnp.bfloat16:
        logits = logits.astype(jnp.float32)
    min_val = -1e10
    logits = jax.vmap(lambda row, k: jax.lax.cond(k > 0, lambda: topk_mask(row[None, :], k, min_val)[0], lambda: row))(
        logits, top_k
    )
    logits = jax.vmap(lambda row, p: jax.lax.cond(p < 1.0, lambda: topp_mask(row[None, :], p, min_val)[0], lambda: row))(
        logits, top_p
    )
    logits = logits.astype(jnp.float32) / jnp.where(temperature <= 0, 1.0, temperature)
    probs = jax.nn.softmax(logits, axis=-1)
    logits = jnp.where(probs >= min_p * jnp.max(probs, axis=-1, keepdims=True), logits, min_val)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    greedy = temperature <= 0
    point_mass = jnp.where(jnp.arange(logits.shape[-1])[None, :] == sampled_tokens.reshape(-1, 1), 0.0, -jnp.inf)
    log_probs = jnp.where(greedy, point_mass, log_probs)
    return summarize_logprobs(log_probs, sampled_tokens, num_logprobs=num_logprobs)
