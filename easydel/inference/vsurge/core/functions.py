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
import typing as tp
from functools import partial

import jax
from eformer.escale import PartitionManager, with_sharding_constraint
from eformer.jaximus import implicit
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.layers.caching import PagesCache, PagesMetadata, TransformerCache, TransformerMetadata
from easydel.layers.quantization.quantizers import EasyQuantizer
from easydel.utils.compiling_utils import ejit

from ...logits_process import (
    FrequencyPenaltyLogitsProcessor,
    LogitsProcessorList,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from ...sampling_params import JitableSamplingParams
from ..utils import GenerationState, ResultTokens

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


TOPK_FOR_COMPUTE = int(os.getenv("EASYDEL_VSURGE_TOPK_FOR_COMPUTE", "64"))


@ejit(donate_argnums=(0, 1), static_argnums=(3,))
def continuous_bulk_insert(
    prefix: GenerationState,
    decode_state: GenerationState,
    slots: list[int],
    quantizer: EasyQuantizer,
    partition_manager: PartitionManager,
) -> GenerationState:
    def update_idx1d(x, y, s):
        return jax.lax.dynamic_update_slice(x, y, (s,))

    def update_idx2d(x, y, s):
        sharding = getattr(x, "sharding", PartitionSpec())
        return with_sharding_constraint(jax.lax.dynamic_update_slice(x, y, (s, 0)), sharding)

    @implicit
    def _cache(mx_cache, prefix, slot):
        return mx_cache.cache.insert(prefix.cache, quantizer=quantizer, slot=slot, partition_manager=partition_manager)

    logits = decode_state.logits
    index = decode_state.index
    tokens = decode_state.tokens
    valids = decode_state.valids
    next_position_ids = decode_state.next_position_ids
    generated_tokens = decode_state.generated_tokens
    sampling_params = decode_state.sampling_params

    for slot in slots:
        logits = update_idx2d(logits, prefix.logits, slot)
        cache = _cache(decode_state, prefix, slot)
        index = update_idx2d(index, prefix.index, slot)
        tokens = update_idx2d(tokens, prefix.tokens, slot)
        valids = update_idx2d(valids, prefix.valids, slot)
        next_position_ids = update_idx2d(next_position_ids, prefix.next_position_ids, slot)
        generated_tokens = update_idx2d(generated_tokens, prefix.generated_tokens, slot)
        sampling_params = sampling_params.insert(prefix.sampling_params, slot)

    return GenerationState(
        logits=logits,
        cache=cache,
        index=index,
        tokens=tokens,
        valids=valids,
        sampling_params=sampling_params,
        next_position_ids=next_position_ids,
        generated_tokens=generated_tokens,
    )


def continuous_insert(
    prefix: GenerationState,
    decode_state: GenerationState,
    slot: int,
    quantizer: EasyQuantizer,
    partition_manager: PartitionManager,
) -> GenerationState:
    def update_idx1d(x, y):
        sharding = getattr(x, "sharding", PartitionSpec())
        return with_sharding_constraint(jax.lax.dynamic_update_slice(x, y, (slot,)), sharding)

    def update_idx2d(x, y):
        sharding = getattr(x, "sharding", PartitionSpec())
        return with_sharding_constraint(jax.lax.dynamic_update_slice(x, y, (slot, 0)), sharding)

    @implicit
    def _cache(cache: TransformerCache, other: TransformerCache):
        return cache.insert(other, quantizer=quantizer, slot=slot, partition_manager=partition_manager)

    return GenerationState(
        logits=update_idx2d(decode_state.logits, prefix.logits),
        cache=_cache(decode_state.cache, prefix.cache),
        index=update_idx2d(decode_state.index, prefix.index),
        tokens=update_idx2d(decode_state.tokens, prefix.tokens),
        valids=update_idx2d(decode_state.valids, prefix.valids),
        sampling_params=decode_state.sampling_params.insert(prefix.sampling_params, slot),
        next_position_ids=update_idx2d(decode_state.next_position_ids, prefix.next_position_ids),
        generated_tokens=update_idx2d(decode_state.generated_tokens, prefix.generated_tokens),
    )


def continuous_bulk_free_state_slots(slots: list[int], decode_state: GenerationState) -> GenerationState:
    for slot in slots:
        for i in range(len(decode_state.cache)):
            decode_state.cache[i].indexs = decode_state.cache[i].indexs.at[slot].set(0)
            decode_state.cache[i].starts = decode_state.cache[i].starts.at[slot].set(0)
    return decode_state


def continuous_prefill(
    graphdef: nn.GraphDef,
    graphstate: nn.GraphState,
    graphothers: nn.GraphState,
    tokens: jax.Array,
    valids: jax.Array,
    true_length: int,
    pad_token_id: int,
    sampling_params: JitableSamplingParams,
    max_length: int,
    samples_per_slot: int,
    cache: TransformerCache | PagesCache | None,
    attn_metadata: TransformerMetadata | PagesMetadata | None,
    rngs: jax.random.PRNGKey,
) -> tuple[GenerationState, ResultTokens]:
    batch_size, sequence_length = tokens.shape
    if valids.shape[-1] != max_length:
        valids = jax.lax.dynamic_update_slice(
            jnp.ones((batch_size, max_length), "b1"),
            valids.astype("b1"),
            (0, 0),
        ).astype("b1")
    positions = (valids.cumsum(axis=-1) - 1)[:, :sequence_length]

    @implicit
    def _forward(gdef, gstate, gother, input_ids, attention_mask, position_ids, cache, metadata):
        model: EasyDeLBaseModule = nn.merge(gdef, gstate, gother)

        if cache is None:
            starts = jnp.array([input_ids.shape[-1] - true_length])
            past_key_values = model.init_cache(batch_size=batch_size, max_length=max_length, starts=starts)
        else:
            past_key_values = cache

        with model.mesh:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_metadata=metadata,
                apply_lm_head=False,
            )
            if getattr(outputs, "last_hidden_state", None) is not None:
                hidden_states = outputs.last_hidden_state
            elif getattr(outputs, "hidden_states", None) is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                raise ValueError(
                    "The model output does not contain 'last_hidden_state' or 'hidden_states'. "
                    "Please ensure the model is configured to return these outputs or open an issue."
                )
            return outputs.past_key_values, model.apply_lm_head(hidden_states[:, -1])

    kv_cache, logits = _forward(graphdef, graphstate, graphothers, tokens, valids, positions, cache, attn_metadata)

    next_token = dynamic_sample_tokens(
        tokens,
        jnp.array([1], "i4"),
        logits,
        sampling_params.top_p,
        sampling_params.temperature,
        sampling_params.random_sampling,
        sampling_params.presence_penalty,
        sampling_params.frequency_penalty,
        sampling_params.repetition_penalty,
        rngs,
    ).reshape(logits.shape[0], -1)
    validity = jnp.ones_like(next_token, dtype="b1")
    lengths = jnp.full((batch_size, 1), 0, dtype="i4")

    result = ResultTokens(
        data=jnp.concatenate([next_token, validity, lengths], axis=1),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        samples_per_slot=samples_per_slot,
    )
    generation_state = GenerationState(
        logits=logits,
        cache=kv_cache,
        index=jnp.array((sequence_length,)).reshape(1, 1) + 1,
        tokens=next_token,
        valids=valids,
        next_position_ids=positions[:, -1:] + 1,
        generated_tokens=jnp.zeros((batch_size, 1), dtype=jnp.int32),
        sampling_params=sampling_params,
    )
    return generation_state, result


def continuous_decode(
    graphdef: nn.GraphDef,
    graphstate: nn.GraphState,
    graphothers: nn.GraphState,
    state: GenerationState,
    cache_metadata: TransformerMetadata | PagesMetadata | None,
    samples_per_slot: int,
    rngs: jax.random.PRNGKey,
):
    @implicit
    def _forward(gdef, gstate, gothers, state, cache_metadata):
        model: EasyDeLBaseModule = nn.merge(gdef, gstate, gothers)
        with model.mesh:
            return model(
                input_ids=state.tokens,
                attention_mask=state.valids,
                position_ids=state.next_position_ids,
                past_key_values=state.cache,
                cache_metadata=cache_metadata,
            )

    outputs = _forward(graphdef, graphstate, graphothers, state, cache_metadata)
    batch_size = state.tokens.shape[0]
    kv_cache = outputs.past_key_values
    logits = outputs.logits[:, -1]

    next_token = dynamic_sample_tokens(
        state.tokens,
        state.generated_tokens,
        logits,
        state.sampling_params.top_p,
        state.sampling_params.temperature,
        state.sampling_params.random_sampling,
        state.sampling_params.presence_penalty,
        state.sampling_params.frequency_penalty,
        state.sampling_params.repetition_penalty,
        rngs,
    ).reshape(logits.shape[0], -1)

    lengths = jnp.full(
        (batch_size, 1),
        state.generated_tokens[:, -1:] + 1,
        dtype="i4",
    )
    validity = jnp.ones_like(next_token, dtype="b1")
    result = ResultTokens(
        data=jnp.concatenate([next_token, validity, lengths], axis=1),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        samples_per_slot=samples_per_slot,
    )
    next_generation_state = GenerationState(
        logits=logits,
        cache=kv_cache,
        index=state.index + 1,
        tokens=next_token,
        valids=state.valids,
        next_position_ids=state.next_position_ids + 1,
        generated_tokens=state.generated_tokens + 1,
        sampling_params=state.sampling_params,
    )
    return next_generation_state, result


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
