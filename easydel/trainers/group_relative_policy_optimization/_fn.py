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

"""Internal functions for Group Relative Policy Optimization training.

This module contains the core computational functions used by the GRPO trainer,
implementing group-based relative policy optimization for RLHF. GRPO improves
training stability by normalizing rewards within groups of samples rather than
across the entire batch, reducing variance in gradient estimates.

The module provides functions for:
- Computing per-token log probabilities from model outputs
- Calculating KL divergence penalties between policy and reference models
- Group-based reward normalization and advantage estimation
- Policy gradient loss computation with various clipping strategies

All functions are JAX-compatible and support distributed training through sharding.
"""

import collections.abc
import math
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)

from ..training_utils import (
    ScheduledLossAdapter,
    compact_generation_model_kwargs,
    extract_generation_model_kwargs,
    make_assertions_and_get_sizes,
    minibatch_call,
    normalize_generation_model_kwargs,
    prepare_generation_model_kwargs_for_call,
    register_scheduled_loss_adapter,
    repeat_prompt_aligned_model_kwargs,
    scheduled_loss_cache_key,
    slice_prompt_aligned_model_kwargs,
    update_metrics,
    update_state_respectfully,
)

RewardFunc = EasyDeLState | tp.Callable[[list, list], list[float]]


def _masked_sum_and_count(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the (numerator, denominator) used by the chunked masked mean.

    Mirrors the ``masked_mean`` helper defined inside :func:`grpo_step`:
    when ``x`` has a trailing-singleton sequence axis (the sequence-level
    importance-sampling path) the per-row mask is ignored and the numerator
    is the plain sum; otherwise both numerator and denominator are taken
    against the completion mask.

    Args:
        x: Per-token tensor of shape ``[batch, seq]`` (or ``[batch, 1]``
            for sequence-level diagnostics).
        mask: Completion mask of shape ``[batch, seq]``. Ignored when the
            sequence axis is singleton.

    Returns:
        ``(numerator, denominator)`` where ``denominator`` is always
        clamped to ``>= 1`` to avoid divide-by-zero downstream.
    """

    if x.shape[1] == 1:
        return jnp.sum(x), jnp.array(x.shape[0], dtype=jnp.float32)
    return jnp.sum(x * mask), jnp.maximum(jnp.sum(mask), 1.0).astype(jnp.float32)


def _compute_log_importance_weights(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    completion_mask: jax.Array,
    importance_sampling_level: str,
) -> jax.Array:
    """Compute token- or sequence-level GRPO log importance weights."""
    log_ratio = per_token_logps - old_per_token_logps
    if importance_sampling_level == "token":
        return log_ratio
    if importance_sampling_level == "sequence":
        return ((log_ratio * completion_mask).sum(axis=-1) / jnp.maximum(completion_mask.sum(axis=-1), 1.0))[:, None]
    if importance_sampling_level == "sequence_token":
        sequence_log_weight = (
            (log_ratio * completion_mask).sum(axis=-1) / jnp.maximum(completion_mask.sum(axis=-1), 1.0)
        )[:, None]
        return per_token_logps - jax.lax.stop_gradient(per_token_logps) + jax.lax.stop_gradient(sequence_log_weight)
    raise ValueError(
        f"Unknown importance sampling level: {importance_sampling_level}. "
        "Possible values are 'token', 'sequence', and 'sequence_token'."
    )


def _compute_importance_weights(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    completion_mask: jax.Array,
    importance_sampling_level: str,
) -> jax.Array:
    """Compute GRPO importance weights without loss-specific clipping/capping."""
    return jnp.exp(
        _compute_log_importance_weights(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            completion_mask=completion_mask,
            importance_sampling_level=importance_sampling_level,
        )
    )


def _compute_off_policy_sequence_mask(
    *,
    per_token_logps: jax.Array,
    sampling_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    threshold: float,
) -> jax.Array:
    """Keep positive-advantage rows and low-drift negative-advantage rows.

    The forward-KL estimate is sequence-level: sampling log-probs minus current
    policy log-probs, averaged over completion tokens. Negative-advantage rows
    whose mean drift exceeds ``threshold`` are removed from the policy objective.
    """
    forward_kl = jax.lax.stop_gradient(sampling_per_token_logps - per_token_logps)
    sequence_kl = jnp.sum(forward_kl * completion_mask, axis=-1, keepdims=True) / jnp.maximum(
        jnp.sum(completion_mask, axis=-1, keepdims=True),
        1.0,
    )
    return ((advantages >= 0) | (sequence_kl <= threshold)).astype(completion_mask.dtype)


def _compute_dppo_divergence_mask(
    *,
    per_token_logps: jax.Array,
    sampling_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    divergence_type: str,
    epsilon_low: float,
    epsilon_high: float,
    current_topk_logps: jax.Array | None = None,
    sampling_topk_logps: jax.Array | None = None,
) -> jax.Array:
    """Compute DPPO trust-region masks for binary or top-k TV/KL.

    Binary variants compare only the generated token probability. Top-k
    variants compare normalized distributions over the generation-time top-k
    support and still use the sampled-token probability to decide whether the
    current policy moved in the positive- or negative-advantage direction.
    """
    prob = jnp.exp(per_token_logps)
    sampling_prob = jnp.exp(sampling_per_token_logps)

    if divergence_type == "binary_tv":
        divergence = jnp.abs(prob - sampling_prob)
    elif divergence_type == "binary_kl":
        safe_prob = jnp.clip(prob, 1e-7, 1 - 1e-7)
        safe_sampling_prob = jnp.clip(sampling_prob, 1e-7, 1 - 1e-7)
        divergence = safe_sampling_prob * (jnp.log(safe_sampling_prob) - jnp.log(safe_prob)) + (
            1 - safe_sampling_prob
        ) * (jnp.log1p(-safe_sampling_prob) - jnp.log1p(-safe_prob))
    elif divergence_type in {"topk_tv", "topk_kl"}:
        if current_topk_logps is None or sampling_topk_logps is None:
            raise ValueError("Top-k DPPO divergence requires current and sampling top-k log-prob tensors.")
        current_logq = jax.nn.log_softmax(current_topk_logps.astype(jnp.float32), axis=-1)
        sampling_logq = jax.nn.log_softmax(jax.lax.stop_gradient(sampling_topk_logps).astype(jnp.float32), axis=-1)
        current_q = jnp.exp(current_logq)
        sampling_q = jnp.exp(sampling_logq)
        if divergence_type == "topk_tv":
            divergence = 0.5 * jnp.sum(jnp.abs(current_q - sampling_q), axis=-1)
        else:
            divergence = jnp.sum(sampling_q * (sampling_logq - current_logq), axis=-1)
    else:
        raise ValueError(f"Unknown DPPO divergence_type: {divergence_type}")

    invalid_pos = (divergence > epsilon_high) & (prob > sampling_prob)
    invalid_neg = (divergence > epsilon_low) & (prob < sampling_prob)
    keep_mask = jnp.where(advantages > 0, ~invalid_pos, ~invalid_neg)
    return keep_mask.astype(completion_mask.dtype) * completion_mask


def _compute_grpo_policy_loss_terms(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    loss_type: str,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    importance_sampling_level: str,
    sapo_temperature_pos: float,
    sapo_temperature_neg: float,
    vespo_k_pos: float,
    vespo_lambda_pos: float,
    vespo_k_neg: float,
    vespo_lambda_neg: float,
    importance_sampling_ratio: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute TRL-compatible GRPO/CISPO per-token surrogate terms."""
    log_ratio = per_token_logps - old_per_token_logps
    log_importance_weights = _compute_log_importance_weights(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        completion_mask=completion_mask,
        importance_sampling_level=importance_sampling_level,
    )

    coef_1 = jnp.exp(log_importance_weights)
    if loss_type == "cispo":
        clamped_ratios = jax.lax.stop_gradient(jnp.minimum(coef_1, epsilon_high))
        return -clamped_ratios * advantages * per_token_logps, coef_1
    if loss_type == "sapo":
        sapo_temperature = jnp.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        sapo_multiplier = jax.nn.sigmoid(sapo_temperature * (coef_1 - 1.0)) * 4.0 / sapo_temperature
        return -sapo_multiplier * advantages, coef_1
    if loss_type == "vespo":
        phi_seq = _vespo_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=completion_mask,
            importance_sampling_ratio=importance_sampling_ratio,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        return -jax.lax.stop_gradient(phi_seq) * advantages * per_token_logps, coef_1
    if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)
        if delta is not None:
            coef_1 = jnp.minimum(coef_1, delta)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        return -jnp.minimum(per_token_loss1, per_token_loss2), coef_1
    raise ValueError(f"Unknown loss type: {loss_type}")


def _vespo_gamma_weights(
    *,
    advantages: jax.Array,
    log_ratio_per_token: jax.Array,
    mask: jax.Array,
    importance_sampling_ratio: jax.Array | None,
    k_pos: float = 2.0,
    lambda_pos: float = 3.0,
    k_neg: float = 3.0,
    lambda_neg: float = 2.0,
) -> jax.Array:
    """Compute VESPO sequence weights from clipped policy/reference ratios.

    Positive and negative advantages use separate ``k`` and ``lambda`` shaping
    parameters. The result is finite by construction and can be multiplied into
    the per-token GRPO-style objective under JIT.
    """
    lower_clamp = math.log(1e-8)
    log_ratio_clamped = jnp.clip(log_ratio_per_token, -20.0, 20.0)
    seq_log_ratio = jnp.sum(log_ratio_clamped * mask, axis=-1, keepdims=True)
    if importance_sampling_ratio is not None:
        log_is_ratio = jnp.clip(jnp.log(jnp.maximum(importance_sampling_ratio, 1e-8)), lower_clamp, 20.0)
        seq_log_ratio = seq_log_ratio + jnp.sum(log_is_ratio, axis=-1, keepdims=True)

    log_w_seq = jnp.clip(seq_log_ratio, lower_clamp, 20.0)
    w_seq = jnp.exp(log_w_seq)
    is_nonnegative_advantage = advantages >= 0
    k_seq = jnp.where(is_nonnegative_advantage, k_pos, k_neg)
    lambda_seq = jnp.maximum(jnp.where(is_nonnegative_advantage, lambda_pos, lambda_neg), 1e-4)
    log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
    return jnp.nan_to_num(jnp.exp(log_phi), nan=0.0, posinf=0.0, neginf=0.0)


def get_per_token_logps(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
):
    """Compute per-token log probabilities for generated sequences.

    This function extracts log probabilities for each token in the completion
    portion of the sequence (after the prompt). It's used to compute likelihood
    ratios between policy and reference models for GRPO training.

    Args:
        model: The language model (EasyDeLBaseModule) to compute log probabilities.
        input_ids: Input token IDs including prompt and completion.
            Shape: [batch_size, seq_len]
        attention_mask: Binary mask indicating valid tokens (1) vs padding (0).
            Shape: [batch_size, seq_len]
        prompt_length: Number of tokens in the prompt portion. Log probabilities
            are only computed for tokens after this position.
        model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
            tensors like ``pixel_values`` or ``inputs_embeds``). Defaults to None.
        logprob_vocab_chunk_size: When set to a positive value, the log-softmax over
            the vocabulary is computed in chunks of this size to reduce peak
            memory. ``None`` disables vocabulary chunking and computes the
            full softmax in one pass.

    Returns:
        Array: Per-token log probabilities for the completion portion.
            Shape: [batch_size, seq_len - prompt_length]

    Note:
        The function shifts logits by one position to align with the autoregressive
        nature of language models, where each position predicts the next token.
        When the model's ``lmhead_chunksize`` is configured, the forward pass
        is run with ``apply_lm_head=False`` and log probabilities are computed
        directly from hidden states in a chunked fashion, avoiding
        materialization of the full logit tensor.
    """

    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]
    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states,
            targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=False,
            vocab_shard_stage=vocab_shard_stage,
        )
        return token_log_probs
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits.")
    logits = logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs


def get_per_token_logps_and_selected_logps(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    selected_indices,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
):
    """Return generated-token log-probs plus log-probs at selected vocab indices.

    ``selected_indices`` is shaped ``[batch, completion_len, k]`` and is
    interpreted as the fixed support from the sampling policy. The returned
    selected log-probs are current-policy log-probs gathered at exactly those
    indices, so top-k trust-region losses compare like-for-like supports.
    """
    del logprob_vocab_chunk_size
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    outputs = model(**call_kwargs)
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits; top-k scoring requires full logits.")
    logits = logits[:, prompt_length - 1 : -1, :]
    targets = input_ids[:, prompt_length:]
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    selected_logps = jnp.take_along_axis(log_probs, selected_indices, axis=-1)
    return token_log_probs.astype(logits.dtype), selected_logps.astype(logits.dtype)


def get_per_token_logps_and_topk(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    topk: int,
    model_kwargs=None,
):
    """Return generated-token log-probs and top-k log-probs for each position.

    This is used during DPPO preprocessing to snapshot the generation-time
    policy distribution over a compact support. The support is then carried in
    the batch so the train step can compare the current policy against the same
    tokens.
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    outputs = model(**call_kwargs)
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits; top-k scoring requires full logits.")
    logits = logits[:, prompt_length - 1 : -1, :]
    targets = input_ids[:, prompt_length:]
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    topk_logps, topk_indices = jax.lax.top_k(log_probs, int(topk))
    return token_log_probs.astype(logits.dtype), topk_indices, topk_logps.astype(logits.dtype)


def compute_per_token_logps(logits, input_ids, prompt_length):
    """Compute per-token log probabilities in a vectorized way.

    Converts raw logits to log-softmax probabilities, then gathers the
    log probability corresponding to each actual target token in the
    completion portion of the sequence.

    Args:
        logits: Pre-trimmed logits of shape ``[batch_size, seq_len, vocab_size]``.
        input_ids: Full input token IDs of shape ``[batch_size, seq_len]``.
        prompt_length: Number of prompt tokens. Targets are extracted from
            ``input_ids[:, prompt_length:]``.

    Returns:
        jax.Array: Per-token log probabilities for the completion portion,
            shape ``[batch_size, completion_len]``.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_ids = input_ids[:, prompt_length:]
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(target_ids, axis=-1),
        axis=-1,
    )
    token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
    return token_log_probs


def get_per_token_logps_and_entropies(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
):
    """Compute per-token log probabilities and entropy for the completion portion.

    Similar to ``get_per_token_logps``, but also returns the per-token entropy
    of the predicted distribution. Entropy is used by GRPO variants that apply
    entropy-based filtering (e.g., top-entropy quantile masking).

    Args:
        model: The language model to run the forward pass on.
        input_ids: Input token IDs including prompt and completion.
            Shape: ``[batch_size, seq_len]``.
        attention_mask: Binary mask indicating valid tokens (1) vs padding (0).
            Shape: ``[batch_size, seq_len]``.
        prompt_length: Number of tokens in the prompt. Log probabilities and
            entropies are computed only for tokens after this position.
        model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
            tensors like ``pixel_values`` or ``inputs_embeds``). Defaults to None.
        logprob_vocab_chunk_size: When set to a positive value, the log-softmax and
            entropy computations over the vocabulary are performed in chunks
            of this size to reduce peak memory usage. ``None`` disables
            vocabulary chunking and computes the full softmax in a single pass.

    Returns:
        tuple[jax.Array, jax.Array]: A pair of arrays:
            - Per-token log probabilities, shape ``[batch_size, completion_len]``.
            - Per-token entropy of the predicted distribution, same shape.

    Note:
        When the model's ``lmhead_chunksize`` is configured, the forward
        pass is run with ``apply_lm_head=False`` and both log probabilities
        and entropies are computed directly from hidden states in a chunked
        fashion, avoiding materialization of the full logit tensor.
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]
    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, entropies = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states,
            targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=True,
            vocab_shard_stage=vocab_shard_stage,
        )
        return token_log_probs, entropies
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits.")
    logits = logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs, entropies = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=True,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs, entropies


def _maybe_extend_inputs_embeds_for_scoring(
    model,
    input_ids,
    model_kwargs,
    *,
    prompt_length: int,
):
    """Pad ``inputs_embeds`` to cover the completion when scoring multimodal prompts.

    GRPO generation may carry ``inputs_embeds`` only for the prompt
    (e.g. when the prompt embeds visual tokens). When the policy is
    later scored on the prompt+completion sequence we need a matching
    embedding tensor; this helper concatenates the model's own token
    embeddings for the completion ids onto the existing prompt
    embeddings. If ``inputs_embeds`` is absent or already full-length
    the kwargs are returned untouched.

    Args:
        model: Bound module exposing :meth:`compute_embedding` for the
            completion-side ids.
        input_ids: Full ``[batch, prompt + completion]`` id array used
            to derive the completion ids.
        model_kwargs: Generation kwargs that may carry ``inputs_embeds``.
        prompt_length: Number of prompt tokens at the start of
            ``input_ids``; used to slice the completion segment.

    Returns:
        A possibly-updated ``model_kwargs`` dict with ``inputs_embeds``
        extended to length ``input_ids.shape[-1]``.

    Raises:
        ValueError: If ``inputs_embeds`` has a length that is neither
            the prompt length nor the target full length, or if the
            model returns embeddings with a different rank than the
            existing prompt embeddings.
    """

    inputs_embeds = model_kwargs.get("inputs_embeds", None)
    if inputs_embeds is None:
        return model_kwargs

    current_length = int(inputs_embeds.shape[-2])
    target_length = int(input_ids.shape[-1])
    if current_length == target_length:
        return model_kwargs
    if current_length != int(prompt_length):
        raise ValueError(
            "GRPO scoring with `inputs_embeds` requires either full-sequence embeddings "
            f"or prompt-length embeddings. Got sequence axis {current_length} for target length {target_length}."
        )

    completion_input_ids = input_ids[:, prompt_length:target_length]
    if completion_input_ids.shape[-1] == 0:
        return model_kwargs

    completion_embeds = model.compute_embedding(completion_input_ids)
    if completion_embeds.ndim != inputs_embeds.ndim:
        raise ValueError(
            "Model `compute_embedding` returned embeddings with an unexpected rank for GRPO scoring: "
            f"{completion_embeds.ndim} vs prompt embeddings rank {inputs_embeds.ndim}."
        )
    if completion_embeds.dtype != inputs_embeds.dtype:
        completion_embeds = completion_embeds.astype(inputs_embeds.dtype)

    updated_model_kwargs = dict(model_kwargs)
    updated_model_kwargs["inputs_embeds"] = jnp.concatenate(
        [inputs_embeds, completion_embeds],
        axis=-2,
    )
    return updated_model_kwargs


def grpo_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    loss_type: str = "dapo",
    epsilon: float = 0.2,
    epsilon_high: float = 0.2,
    delta: float | None = None,
    importance_sampling_level: str = "token",
    top_entropy_quantile: float = 1.0,
    completion_chunk_size: int | None = None,
    max_loss_completion_tokens: int | None = None,
    logprob_vocab_chunk_size: int | None = None,
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    vespo_k_pos: float = 2.0,
    vespo_lambda_pos: float = 3.0,
    vespo_k_neg: float = 3.0,
    vespo_lambda_neg: float = 2.0,
    off_policy_mask_threshold: float | None = None,
    use_bias_correction_kl: bool = False,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    dppo_divergence_type: str | None = None,
    dppo_clip_ratio_c: float = 20.0,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Perform a single GRPO training or evaluation step.

    Computes the group-relative policy optimization loss on a batch of
    pre-processed data (prompt IDs, completion IDs, advantages, and
    optionally reference log-probs). During training the function also
    computes gradients via minibatch accumulation and updates the model state.

    The function supports several loss variants controlled by ``loss_type``:
        - ``"grpo"``: Standard GRPO with per-sequence normalization.
        - ``"bnpo"``: Batch-normalized policy optimization.
        - ``"dr_grpo"``: Denominator-regularized GRPO.
        - ``"dapo"``: Dynamic advantage policy optimization (default).
        - ``"cispo"``: Clipped importance-sampling policy optimization.
        - ``"sapo"``: Soft advantage policy optimization.
        - ``"luspo"``: Length-unbiased sequence policy optimization.
        - ``"vespo"``: Variational sequence-level soft policy optimization.

    Args:
        state: Current model state including parameters and optimizer.
        batch: Mapping containing at minimum ``prompt_ids``, ``prompt_mask``,
            ``completion_ids``, ``completion_mask``, ``advantages``, and
            optionally ``ref_per_token_logps`` and ``old_per_token_logps``.
        num_generations: Number of completions generated per prompt.
        beta: KL divergence penalty coefficient. Set to 0.0 to disable.
        loss_config: Optional loss configuration for gradient clipping etc.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for the batch.
        gradient_accumulation_steps: Number of minibatch accumulation steps.
        is_training: If True, compute and apply gradients. If False, only
            compute metrics (evaluation mode).
        loss_type: Which loss variant to use (see above).
        epsilon: Lower clipping bound for importance-sampling ratios.
        epsilon_high: Upper clipping bound for importance-sampling ratios.
        delta: Optional upper cap on un-clipped importance weights.
        importance_sampling_level: ``"token"`` for per-token or ``"sequence"``
            for per-sequence importance weighting.
        top_entropy_quantile: Fraction of highest-entropy tokens to keep in
            the loss. 1.0 disables filtering.
        completion_chunk_size: Chunk size for memory-saving chunked completion
            loss. Set to ``None`` to disable chunking.
        max_loss_completion_tokens: Optional cap on completion tokens used by
            the GRPO loss. Set to ``None`` to disable truncation.
        sapo_temperature_pos: Soft-clipping temperature for positive SAPO advantages.
        sapo_temperature_neg: Soft-clipping temperature for negative SAPO advantages.
        vespo_k_pos: Gamma-weight exponent for positive VESPO advantages.
        vespo_lambda_pos: Gamma-weight decay for positive VESPO advantages.
        vespo_k_neg: Gamma-weight exponent for negative VESPO advantages.
        vespo_lambda_neg: Gamma-weight decay for negative VESPO advantages.
        off_policy_mask_threshold: Optional sequence-level forward-KL
            threshold for masking high-drift negative-advantage rows from
            the policy objective.
        use_bias_correction_kl: If True, multiply the KL penalty by the
            un-clipped importance weights.
        straight_through_emulator: Optional function for quantization-aware
            straight-through gradient estimation.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: When ``is_training``
            is True, returns the updated state and loss metrics. When False,
            returns only the loss metrics.
    """
    scope_root = "easydel/trainer/grpo/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        # Determine batch size, minibatch size, and enforce partition spec.
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute the GRPO surrogate loss for one minibatch.

        Concatenates prompts and completions, calls the policy forward
        in chunked mode (using ``completion_chunk_size`` /
        ``logprob_vocab_chunk_size`` for memory), masks the
        per-token log-probabilities by the completion mask, applies
        the importance-sampling clip given by ``epsilon`` /
        ``epsilon_high`` / ``delta`` and the configured ``loss_type``
        (``grpo``, ``grpo_token``, ``dr_grpo``, ``bnpo``, ...), folds
        in the KL penalty against the reference policy and the
        optional ``top_entropy_quantile`` mask.

        Args:
            tree: Policy graphstate to differentiate against.
            minibatch: Dict carrying ``prompt_ids``, ``prompt_mask``,
                ``completion_ids``, ``completion_mask``,
                ``advantages``, ``ref_per_token_logps``, and any
                generation-time model kwargs.

        Returns:
            ``(loss, metrics)`` where ``metrics`` is a populated
            :class:`LossMetrics` recording surrogate loss components,
            clip fractions, KL diagnostics, and any straight-through
            quantizer signals.
        """
        if is_training and straight_through_emulator is not None:
            with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                tree = straight_through_emulator(tree)
        with jax.named_scope(scope_root + "/loss_fn/merge_state"):
            module = state.merge(tree)

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            advantages,
        ) = (
            minibatch["prompt_ids"],
            minibatch["prompt_mask"],
            minibatch["completion_ids"],
            minibatch["completion_mask"],
            minibatch["advantages"],
        )

        completion_was_truncated = False
        if max_loss_completion_tokens is not None and completion_ids.shape[1] > max_loss_completion_tokens:
            completion_ids = completion_ids[:, :max_loss_completion_tokens]
            completion_mask = completion_mask[:, :max_loss_completion_tokens]
            completion_was_truncated = True

        # Use runtime batch shapes so filtered-group trainers (e.g. GFPO) can
        # train with a different effective generation count than sampling-time.
        effective_num_generations = completion_ids.shape[0] // max(prompt_ids.shape[0], 1)
        effective_num_generations = max(effective_num_generations, 1)

        input_ids = jnp.concatenate([prompt_ids.repeat(effective_num_generations, 0), completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask.repeat(effective_num_generations, 0), completion_mask], axis=1)
        prompt_len = prompt_ids.shape[-1]
        prompt_model_kwargs = extract_generation_model_kwargs(
            minibatch,
            model_callable=getattr(module, "forward", module),
        )
        completion_model_kwargs = repeat_prompt_aligned_model_kwargs(
            prompt_model_kwargs,
            effective_num_generations,
            prompt_batch_size=prompt_ids.shape[0],
        )

        advantages = minibatch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        has_difficulty_weights = "difficulty_weights" in minibatch
        difficulty_weights = minibatch.get("difficulty_weights")
        if not has_difficulty_weights:
            difficulty_weights = jnp.ones((completion_ids.shape[0], 1), dtype=jnp.float32)
        elif difficulty_weights.ndim == 1:
            difficulty_weights = difficulty_weights[:, None]
        difficulty_weights = difficulty_weights.astype(jnp.float32)

        old_per_token_logps = minibatch.get("old_per_token_logps")
        if old_per_token_logps is not None and old_per_token_logps.shape[1] != completion_ids.shape[1]:
            old_per_token_logps = old_per_token_logps[:, : completion_ids.shape[1]]
        completion_token_count = jnp.sum(completion_mask)
        completion_lengths = jnp.sum(completion_mask, axis=1)

        use_chunked_completion_loss = (
            completion_chunk_size is not None
            and completion_ids.shape[0] > completion_chunk_size
            and top_entropy_quantile >= 1.0
        )
        if use_chunked_completion_loss:
            expanded_prompt_ids = prompt_ids.repeat(effective_num_generations, 0)
            expanded_prompt_mask = prompt_mask.repeat(effective_num_generations, 0)
            completion_batch_size = int(completion_ids.shape[0])
            normalizer = (
                completion_token_count
                if completion_was_truncated
                else minibatch.get(
                    "num_items_in_batch",
                    completion_token_count,
                )
            )

            loss_numerator = jnp.array(0.0, dtype=jnp.float32)
            mean_kl_num = jnp.array(0.0, dtype=jnp.float32)
            mean_kl_den = jnp.array(0.0, dtype=jnp.float32)
            ref_logps_num = jnp.array(0.0, dtype=jnp.float32)
            ref_logps_den = jnp.array(0.0, dtype=jnp.float32)
            low_clip_num = jnp.array(0.0, dtype=jnp.float32)
            low_clip_den = jnp.array(0.0, dtype=jnp.float32)
            high_clip_num = jnp.array(0.0, dtype=jnp.float32)
            high_clip_den = jnp.array(0.0, dtype=jnp.float32)
            region_clip_num = jnp.array(0.0, dtype=jnp.float32)
            region_clip_den = jnp.array(0.0, dtype=jnp.float32)
            cispo_clip_num = jnp.array(0.0, dtype=jnp.float32)
            cispo_clip_den = jnp.array(0.0, dtype=jnp.float32)
            off_policy_keep_num = jnp.array(0.0, dtype=jnp.float32)
            off_policy_keep_den = jnp.array(0.0, dtype=jnp.float32)
            grpo_weight_den = jnp.sum(difficulty_weights)

            for start in range(0, completion_batch_size, completion_chunk_size):
                end = min(start + completion_chunk_size, completion_batch_size)
                chunk_completion_ids = completion_ids[start:end]
                chunk_completion_mask = completion_mask[start:end]
                chunk_prompt_ids = expanded_prompt_ids[start:end]
                chunk_prompt_mask = expanded_prompt_mask[start:end]
                chunk_input_ids = jnp.concatenate([chunk_prompt_ids, chunk_completion_ids], axis=1)
                chunk_attention_mask = jnp.concatenate([chunk_prompt_mask, chunk_completion_mask], axis=1)
                chunk_model_kwargs = slice_prompt_aligned_model_kwargs(
                    completion_model_kwargs,
                    start,
                    end,
                    prompt_batch_size=completion_batch_size,
                )
                with jax.named_scope(scope_root + "/loss_fn/chunked/policy_logps"):
                    chunk_current_topk_logps = None
                    if dppo_divergence_type in {"topk_tv", "topk_kl"} and "sampling_topk_indices" in minibatch:
                        chunk_per_token_logps, chunk_current_topk_logps = get_per_token_logps_and_selected_logps(
                            module,
                            chunk_input_ids,
                            chunk_attention_mask,
                            prompt_len,
                            minibatch["sampling_topk_indices"][start:end],
                            model_kwargs=chunk_model_kwargs,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                    else:
                        chunk_per_token_logps = get_per_token_logps(
                            module,
                            chunk_input_ids,
                            chunk_attention_mask,
                            prompt_len,
                            model_kwargs=chunk_model_kwargs,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                chunk_old_per_token_logps = (
                    old_per_token_logps[start:end]
                    if old_per_token_logps is not None
                    else jax.lax.stop_gradient(chunk_per_token_logps)
                )
                chunk_importance_weights = _compute_importance_weights(
                    per_token_logps=chunk_per_token_logps,
                    old_per_token_logps=chunk_old_per_token_logps,
                    completion_mask=chunk_completion_mask,
                    importance_sampling_level=importance_sampling_level,
                )
                chunk_ref_per_token_logps = (
                    minibatch["ref_per_token_logps"][start:end, : completion_ids.shape[1]]
                    if beta != 0.0
                    else jnp.zeros_like(chunk_per_token_logps)
                )
                chunk_per_token_kl = (
                    jnp.exp(chunk_ref_per_token_logps - chunk_per_token_logps)
                    - (chunk_ref_per_token_logps - chunk_per_token_logps)
                    - 1
                    if beta != 0.0
                    else jnp.zeros_like(chunk_per_token_logps)
                )
                if beta != 0.0 and use_bias_correction_kl:
                    chunk_per_token_kl = chunk_per_token_kl * chunk_importance_weights
                chunk_advantages = advantages[start:end]
                chunk_difficulty_weights = difficulty_weights[start:end]
                chunk_importance_sampling_ratio = (
                    minibatch["importance_sampling_ratio"][start:end]
                    if "importance_sampling_ratio" in minibatch
                    else None
                )

                if dppo_divergence_type is not None:
                    chunk_sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                    if chunk_sampling_per_token_logps is None:
                        chunk_sampling_per_token_logps = chunk_old_per_token_logps
                    else:
                        chunk_sampling_per_token_logps = chunk_sampling_per_token_logps[
                            start:end, : completion_ids.shape[1]
                        ]
                    log_ratio = chunk_per_token_logps - chunk_sampling_per_token_logps
                    coef_1 = jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_ratio, math.log(dppo_clip_ratio_c))))
                    chunk_divergence_mask = _compute_dppo_divergence_mask(
                        per_token_logps=chunk_per_token_logps,
                        sampling_per_token_logps=chunk_sampling_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        divergence_type=dppo_divergence_type,
                        epsilon_low=epsilon,
                        epsilon_high=epsilon_high,
                        current_topk_logps=chunk_current_topk_logps,
                        sampling_topk_logps=(
                            minibatch["sampling_topk_logps"][start:end] if "sampling_topk_logps" in minibatch else None
                        ),
                    )
                    chunk_per_token_loss = -chunk_advantages * coef_1 * chunk_divergence_mask * chunk_per_token_logps
                else:
                    chunk_per_token_loss, coef_1 = _compute_grpo_policy_loss_terms(
                        per_token_logps=chunk_per_token_logps,
                        old_per_token_logps=chunk_old_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        loss_type=loss_type,
                        epsilon=epsilon,
                        epsilon_high=epsilon_high,
                        delta=delta,
                        importance_sampling_level=importance_sampling_level,
                        sapo_temperature_pos=sapo_temperature_pos,
                        sapo_temperature_neg=sapo_temperature_neg,
                        vespo_k_pos=vespo_k_pos,
                        vespo_lambda_pos=vespo_lambda_pos,
                        vespo_k_neg=vespo_k_neg,
                        vespo_lambda_neg=vespo_lambda_neg,
                        importance_sampling_ratio=chunk_importance_sampling_ratio,
                    )

                if off_policy_mask_threshold is not None:
                    chunk_sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                    if chunk_sampling_per_token_logps is None:
                        chunk_sampling_per_token_logps = chunk_old_per_token_logps
                    else:
                        chunk_sampling_per_token_logps = chunk_sampling_per_token_logps[
                            start:end, : completion_ids.shape[1]
                        ]
                    chunk_off_policy_mask = _compute_off_policy_sequence_mask(
                        per_token_logps=chunk_per_token_logps,
                        sampling_per_token_logps=chunk_sampling_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        threshold=off_policy_mask_threshold,
                    )
                    per_chunk_keep_num, per_chunk_keep_den = _masked_sum_and_count(
                        chunk_off_policy_mask,
                        jnp.ones_like(chunk_off_policy_mask),
                    )
                    off_policy_keep_num = off_policy_keep_num + per_chunk_keep_num
                    off_policy_keep_den = off_policy_keep_den + per_chunk_keep_den
                    chunk_per_token_loss = chunk_per_token_loss * chunk_off_policy_mask

                if beta != 0.0:
                    chunk_per_token_loss = chunk_per_token_loss + beta * chunk_per_token_kl
                chunk_per_token_loss = chunk_per_token_loss * chunk_difficulty_weights

                if loss_type in {"grpo", "sapo"}:
                    loss_numerator = loss_numerator + jnp.sum(
                        jnp.sum(chunk_per_token_loss * chunk_completion_mask, axis=1)
                        / jnp.maximum(jnp.sum(chunk_completion_mask, axis=1), 1.0)
                    )
                    grpo_weight_den = jnp.sum(difficulty_weights)
                elif loss_type == "luspo":
                    loss_numerator = loss_numerator + jnp.sum(
                        chunk_per_token_loss * jnp.sum(chunk_completion_mask, axis=1, keepdims=True)
                    )
                else:
                    loss_numerator = loss_numerator + jnp.sum(chunk_per_token_loss * chunk_completion_mask)

                if beta != 0.0:
                    chunk_mean_kl_num, chunk_mean_kl_den = _masked_sum_and_count(
                        chunk_per_token_kl, chunk_completion_mask
                    )
                    mean_kl_num = mean_kl_num + chunk_mean_kl_num
                    mean_kl_den = mean_kl_den + chunk_mean_kl_den
                    chunk_ref_num, chunk_ref_den = _masked_sum_and_count(
                        chunk_ref_per_token_logps,
                        chunk_completion_mask,
                    )
                    ref_logps_num = ref_logps_num + chunk_ref_num
                    ref_logps_den = ref_logps_den + chunk_ref_den

                if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
                    is_low_clipped = (coef_1 < 1 - epsilon) & (chunk_advantages < 0)
                    is_high_clipped = (coef_1 > 1 + epsilon_high) & (chunk_advantages > 0)
                    is_region_clipped = is_low_clipped | is_high_clipped
                    chunk_low_num, chunk_low_den = _masked_sum_and_count(
                        is_low_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    chunk_high_num, chunk_high_den = _masked_sum_and_count(
                        is_high_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    chunk_region_num, chunk_region_den = _masked_sum_and_count(
                        is_region_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    low_clip_num = low_clip_num + chunk_low_num
                    low_clip_den = low_clip_den + chunk_low_den
                    high_clip_num = high_clip_num + chunk_high_num
                    high_clip_den = high_clip_den + chunk_high_den
                    region_clip_num = region_clip_num + chunk_region_num
                    region_clip_den = region_clip_den + chunk_region_den
                elif loss_type == "cispo":
                    is_cispo_clipped = (coef_1 > epsilon_high) & (chunk_advantages > 0)
                    chunk_cispo_num, chunk_cispo_den = _masked_sum_and_count(
                        is_cispo_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    cispo_clip_num = cispo_clip_num + chunk_cispo_num
                    cispo_clip_den = cispo_clip_den + chunk_cispo_den

            if loss_type in {"grpo", "sapo"}:
                loss = loss_numerator / jnp.maximum(grpo_weight_den, 1.0)
            elif loss_type == "luspo":
                # Match the non-chunked path's ``jnp.mean(per_token_loss * completion_lengths)`` which divides
                # by batch*seq; dividing by batch alone made the chunked loss ~seq_len larger, so the gradient
                # magnitude depended on whether LM-head chunking was enabled.
                loss = loss_numerator / jnp.maximum(completion_ids.shape[0] * completion_ids.shape[1], 1.0)
            elif loss_type == "bnpo":
                weighted_token_count = jnp.sum(completion_mask * difficulty_weights)
                loss = loss_numerator / jnp.maximum(weighted_token_count, 1.0)
            elif loss_type == "dr_grpo":
                loss = loss_numerator / jnp.maximum(jnp.sum(difficulty_weights) * completion_ids.shape[1], 1.0)
            elif loss_type in ["cispo", "dapo", "vespo", "dppo"]:
                normalizer = jnp.sum(completion_mask * difficulty_weights) if has_difficulty_weights else normalizer
                loss = loss_numerator / jnp.maximum(normalizer, 1.0)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            other_metrics: dict[str, jax.Array] = {
                "mean_entropy": jnp.array(jnp.nan, dtype=jnp.float32),
                "advantages": jnp.mean(advantages),
            }
            if beta != 0.0:
                mean_kl = mean_kl_num / jnp.maximum(mean_kl_den, 1.0)
                other_metrics["mean_kl"] = mean_kl
                other_metrics["ref_per_token_logps"] = ref_logps_num / jnp.maximum(ref_logps_den, 1.0)
            else:
                mean_kl = None
            if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
                other_metrics["clip_ratio/low_mean"] = low_clip_num / jnp.maximum(low_clip_den, 1.0)
                other_metrics["clip_ratio/high_mean"] = high_clip_num / jnp.maximum(high_clip_den, 1.0)
                other_metrics["clip_ratio/region_mean"] = region_clip_num / jnp.maximum(region_clip_den, 1.0)
            elif loss_type == "cispo":
                other_metrics["cispo_clip_ratio"] = cispo_clip_num / jnp.maximum(cispo_clip_den, 1.0)
            if off_policy_mask_threshold is not None:
                other_metrics["off_policy_keep_ratio"] = off_policy_keep_num / jnp.maximum(off_policy_keep_den, 1.0)

            return loss, LossMetrics(
                loss=loss,
                accuracy=1,
                other_metrics=other_metrics,
            )

        entropies = None
        current_topk_logps = None
        with jax.named_scope(scope_root + "/loss_fn/policy_logps"):
            if dppo_divergence_type in {"topk_tv", "topk_kl"} and "sampling_topk_indices" in minibatch:
                per_token_logps, current_topk_logps = get_per_token_logps_and_selected_logps(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    minibatch["sampling_topk_indices"],
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            elif top_entropy_quantile < 1.0:
                per_token_logps, entropies = get_per_token_logps_and_entropies(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            else:
                per_token_logps = get_per_token_logps(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )

        with jax.named_scope(scope_root + "/loss_fn/kl_to_reference"):
            ref_per_token_logps = None
            if beta != 0.0:
                ref_per_token_logps = minibatch["ref_per_token_logps"][:, : completion_ids.shape[1]]
                per_token_kl = (
                    jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )
            else:
                per_token_kl = jnp.zeros_like(per_token_logps)

        advantages = minibatch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        has_difficulty_weights = "difficulty_weights" in minibatch
        difficulty_weights = minibatch.get("difficulty_weights")
        if not has_difficulty_weights:
            difficulty_weights = jnp.ones((completion_ids.shape[0], 1), dtype=jnp.float32)
        elif difficulty_weights.ndim == 1:
            difficulty_weights = difficulty_weights[:, None]
        difficulty_weights = difficulty_weights.astype(jnp.float32)

        old_per_token_logps = minibatch.get("old_per_token_logps")
        if old_per_token_logps is not None and old_per_token_logps.shape[1] != completion_ids.shape[1]:
            old_per_token_logps = old_per_token_logps[:, : completion_ids.shape[1]]
        if old_per_token_logps is None:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
        importance_weights = _compute_importance_weights(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            completion_mask=completion_mask,
            importance_sampling_level=importance_sampling_level,
        )
        if beta != 0.0 and use_bias_correction_kl:
            per_token_kl = per_token_kl * importance_weights

        with jax.named_scope(scope_root + "/loss_fn/policy_objective"):
            dppo_divergence_mask = None
            if dppo_divergence_type is not None:
                sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                if sampling_per_token_logps is None:
                    sampling_per_token_logps = old_per_token_logps
                elif sampling_per_token_logps.shape[1] != completion_ids.shape[1]:
                    sampling_per_token_logps = sampling_per_token_logps[:, : completion_ids.shape[1]]
                log_ratio = per_token_logps - sampling_per_token_logps
                coef_1 = jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_ratio, math.log(dppo_clip_ratio_c))))
                dppo_divergence_mask = _compute_dppo_divergence_mask(
                    per_token_logps=per_token_logps,
                    sampling_per_token_logps=sampling_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    divergence_type=dppo_divergence_type,
                    epsilon_low=epsilon,
                    epsilon_high=epsilon_high,
                    current_topk_logps=current_topk_logps,
                    sampling_topk_logps=minibatch.get("sampling_topk_logps"),
                )
                per_token_loss = -advantages * coef_1 * dppo_divergence_mask * per_token_logps
            else:
                per_token_loss, coef_1 = _compute_grpo_policy_loss_terms(
                    per_token_logps=per_token_logps,
                    old_per_token_logps=old_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    loss_type=loss_type,
                    epsilon=epsilon,
                    epsilon_high=epsilon_high,
                    delta=delta,
                    importance_sampling_level=importance_sampling_level,
                    sapo_temperature_pos=sapo_temperature_pos,
                    sapo_temperature_neg=sapo_temperature_neg,
                    vespo_k_pos=vespo_k_pos,
                    vespo_lambda_pos=vespo_lambda_pos,
                    vespo_k_neg=vespo_k_neg,
                    vespo_lambda_neg=vespo_lambda_neg,
                    importance_sampling_ratio=minibatch.get("importance_sampling_ratio"),
                )

        off_policy_mask = None
        if off_policy_mask_threshold is not None:
            sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
            if sampling_per_token_logps is None:
                sampling_per_token_logps = old_per_token_logps
            elif sampling_per_token_logps.shape[1] != completion_ids.shape[1]:
                sampling_per_token_logps = sampling_per_token_logps[:, : completion_ids.shape[1]]
            off_policy_mask = _compute_off_policy_sequence_mask(
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                threshold=off_policy_mask_threshold,
            )
            per_token_loss = per_token_loss * off_policy_mask

        if top_entropy_quantile < 1.0 and entropies is not None:
            masked_entropies = jnp.where(completion_mask > 0, entropies, jnp.nan)
            entropy_threshold = jnp.nanquantile(masked_entropies, 1 - top_entropy_quantile)
            entropy_mask = (entropies >= entropy_threshold).astype(completion_mask.dtype) * completion_mask
            per_token_loss = per_token_loss * entropy_mask

        if beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl
        per_token_loss = per_token_loss * difficulty_weights

        completion_token_count = jnp.sum(completion_mask)
        completion_lengths = jnp.sum(completion_mask, axis=1)

        with jax.named_scope(scope_root + "/loss_fn/reduce_loss"):
            if loss_type in {"grpo", "sapo"}:
                sequence_loss = jnp.sum(per_token_loss * completion_mask, axis=1) / jnp.maximum(
                    completion_lengths,
                    1.0,
                )
                loss = jnp.sum(sequence_loss) / jnp.maximum(jnp.sum(difficulty_weights), 1.0)
            elif loss_type == "luspo":
                loss = jnp.mean(per_token_loss * completion_lengths[:, None])
            elif loss_type == "bnpo":
                weighted_token_count = jnp.sum(completion_mask * difficulty_weights)
                loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(weighted_token_count, 1.0)
            elif loss_type == "dr_grpo":
                loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(
                    jnp.sum(difficulty_weights) * per_token_loss.shape[1],
                    1.0,
                )
            elif loss_type in ["cispo", "dapo", "vespo", "dppo"]:
                if has_difficulty_weights:
                    normalizer = jnp.sum(completion_mask * difficulty_weights)
                else:
                    normalizer = (
                        completion_token_count
                        if completion_was_truncated
                        else minibatch.get(
                            "num_items_in_batch",
                            completion_token_count,
                        )
                    )
                loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(normalizer, 1.0)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        def masked_mean(x):
            """Average ``x`` over masked completion tokens.

            For sequence-level (``shape[1] == 1``) tensors falls back
            to a plain mean since there is nothing to mask.

            Args:
                x: Tensor of shape ``[batch, seq_len]`` or ``[batch, 1]``.

            Returns:
                A scalar mean over the masked positions.
            """
            if x.shape[1] == 1:
                return jnp.mean(x)
            return jnp.sum(x * completion_mask) / jnp.maximum(completion_token_count, 1.0)

        other_metrics: dict[str, jax.Array] = {
            "mean_entropy": (
                masked_mean(entropies) if entropies is not None else jnp.array(jnp.nan, dtype=per_token_logps.dtype)
            ),
            "advantages": jnp.mean(advantages),
        }

        if beta != 0.0:
            mean_kl = masked_mean(per_token_kl)
            other_metrics["mean_kl"] = mean_kl
            if ref_per_token_logps is not None:
                other_metrics["ref_per_token_logps"] = jnp.mean(ref_per_token_logps)
        else:
            mean_kl = None

        if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - epsilon) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            other_metrics["clip_ratio/low_mean"] = masked_mean(is_low_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/high_mean"] = masked_mean(is_high_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/region_mean"] = masked_mean(is_region_clipped.astype(jnp.float32))
        elif loss_type == "cispo":
            is_cispo_clipped = (coef_1 > epsilon_high) & (advantages > 0)
            other_metrics["cispo_clip_ratio"] = masked_mean(is_cispo_clipped.astype(jnp.float32))
        elif loss_type == "dppo" and dppo_divergence_mask is not None:
            is_masked = (dppo_divergence_mask == 0) & (completion_mask > 0)
            other_metrics["dppo_mask_ratio/overall_mean"] = masked_mean(is_masked.astype(jnp.float32))
        if off_policy_mask is not None:
            other_metrics["off_policy_keep_ratio"] = jnp.mean(off_policy_mask)

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics=other_metrics,
        )

    # Compute gradients and metrics across minibatches.
    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
            )
        with jax.named_scope(scope_root + "/update_state"):
            state = update_state_respectfully(
                state=state,
                gradients=gradients,
                loss_config=loss_config,
                metrics=update_metrics(
                    metrics=metrics,
                    learning_rate_fn=learning_rate_fn,
                    step=state.step,
                    gradients=gradients,
                ),
            )
        return state, metrics
    else:
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics


def _grpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the scheduled-loss cache key for GRPO-family trainers.

    The key fingerprints all algorithmic knobs that change the loss
    graph (``num_generations``, ``beta``, ``loss_type``, clip bounds,
    importance-sampling level, chunk sizes) plus the object identities
    of the loss config, learning-rate schedule, and any QAT
    straight-through emulator so that recompilation is triggered only
    when those actually change.

    Args:
        call: The :class:`ScheduledStepCall` being compiled by the
            SpectraX VJP pipeline.

    Returns:
        A tuple suitable for use as a SpectraX scheduled-loss cache key.
    """

    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "num_generations",
            "beta",
            "partition_spec",
            "gradient_accumulation_steps",
            "loss_type",
            "epsilon",
            "epsilon_high",
            "delta",
            "importance_sampling_level",
            "top_entropy_quantile",
            "completion_chunk_size",
            "max_loss_completion_tokens",
            "logprob_vocab_chunk_size",
            "sapo_temperature_pos",
            "sapo_temperature_neg",
            "vespo_k_pos",
            "vespo_lambda_pos",
            "vespo_k_neg",
            "vespo_lambda_neg",
            "off_policy_mask_threshold",
            "use_bias_correction_kl",
        ),
        object_fields=("loss_config", "learning_rate_fn", "straight_through_emulator"),
    )


def _make_grpo_scheduled_loss(call):
    """Build a SpectraX-scheduled scalar GRPO-loss closure for ``call``.

    Captures the algorithmic hyperparameters (``num_generations``,
    ``beta``, ``loss_type``, clip bounds, importance-sampling level,
    chunk sizes, ...) and any QAT straight-through emulator from the
    :class:`ScheduledStepCall`, then returns a function
    ``(tree, batch) -> scalar`` which evaluates :func:`grpo_step` in
    eval-mode and returns only the loss field. Used by the MPMD
    pipeline scheduler to obtain a differentiable scalar.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config and
            hyperparameters.

    Returns:
        A closure ``loss_fn(tree, batch) -> jax.Array`` returning the
        scalar GRPO loss for one scheduled microbatch.
    """

    num_generations = call.get("num_generations")
    beta = call.get("beta")
    loss_config = call.get("loss_config")
    learning_rate_fn = call.get("learning_rate_fn")
    partition_spec = call.get("partition_spec")
    gradient_accumulation_steps = call.get("gradient_accumulation_steps")
    loss_type = call.get("loss_type")
    epsilon = call.get("epsilon")
    epsilon_high = call.get("epsilon_high")
    delta = call.get("delta")
    importance_sampling_level = call.get("importance_sampling_level")
    top_entropy_quantile = call.get("top_entropy_quantile")
    completion_chunk_size = call.get("completion_chunk_size")
    max_loss_completion_tokens = call.get("max_loss_completion_tokens")
    logprob_vocab_chunk_size = call.get("logprob_vocab_chunk_size")
    sapo_temperature_pos = call.get("sapo_temperature_pos")
    sapo_temperature_neg = call.get("sapo_temperature_neg")
    vespo_k_pos = call.get("vespo_k_pos")
    vespo_lambda_pos = call.get("vespo_lambda_pos")
    vespo_k_neg = call.get("vespo_k_neg")
    vespo_lambda_neg = call.get("vespo_lambda_neg")
    off_policy_mask_threshold = call.get("off_policy_mask_threshold")
    use_bias_correction_kl = call.get("use_bias_correction_kl")
    straight_through_emulator = call.get("straight_through_emulator")

    def scheduled_loss(tree, batch):
        """Compute the scalar GRPO loss inside the SpectraX scheduled VJP.

        Applies the optional straight-through emulator to the policy
        graphstate, rebuilds an :class:`EasyDeLState` for ``grpo_step``,
        and dispatches in eval mode so only the metrics path runs.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Scheduled microbatch dict consumed by ``grpo_step``
                (``prompt_ids`` / ``completion_ids`` / ``advantages`` /
                ``ref_per_token_logps`` / generation kwargs).

        Returns:
            The scalar GRPO loss for the microbatch.
        """

        with jax.named_scope("easydel/trainer/grpo/scheduled_loss"):
            if straight_through_emulator is not None:
                tree = straight_through_emulator(tree)
            scheduled_state = call.state.replace(graphstate=tree)
            metrics = grpo_step(
                scheduled_state,
                batch,
                num_generations,
                beta,
                loss_config=loss_config,
                learning_rate_fn=learning_rate_fn,
                partition_spec=partition_spec,
                gradient_accumulation_steps=gradient_accumulation_steps,
                is_training=False,
                loss_type=loss_type,
                epsilon=epsilon,
                epsilon_high=epsilon_high,
                delta=delta,
                importance_sampling_level=importance_sampling_level,
                top_entropy_quantile=top_entropy_quantile,
                completion_chunk_size=completion_chunk_size,
                max_loss_completion_tokens=max_loss_completion_tokens,
                logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                sapo_temperature_pos=sapo_temperature_pos,
                sapo_temperature_neg=sapo_temperature_neg,
                vespo_k_pos=vespo_k_pos,
                vespo_lambda_pos=vespo_lambda_pos,
                vespo_k_neg=vespo_k_neg,
                vespo_lambda_neg=vespo_lambda_neg,
                off_policy_mask_threshold=off_policy_mask_threshold,
                use_bias_correction_kl=use_bias_correction_kl,
                straight_through_emulator=None,
            )
            return metrics.loss

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=grpo_step,
    adapter=ScheduledLossAdapter(
        name="grpo",
        make_loss=_make_grpo_scheduled_loss,
        make_cache_key=_grpo_scheduled_loss_cache_key,
    ),
)
