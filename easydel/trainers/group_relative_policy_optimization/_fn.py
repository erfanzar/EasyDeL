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
    compact_generation_model_kwargs,
    extract_generation_model_kwargs,
    make_assertions_and_get_sizes,
    minibatch_call,
    normalize_generation_model_kwargs,
    prepare_generation_model_kwargs_for_call,
    repeat_prompt_aligned_model_kwargs,
    slice_prompt_aligned_model_kwargs,
    update_metrics,
    update_state_respectfully,
)

RewardFunc = EasyDeLState | tp.Callable[[list, list], list[float]]


def _masked_sum_and_count(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return numerator/denominator matching the masked_mean semantics used below."""

    if x.shape[1] == 1:
        return jnp.sum(x), jnp.array(x.shape[0], dtype=jnp.float32)
    return jnp.sum(x * mask), jnp.maximum(jnp.sum(mask), 1.0).astype(jnp.float32)


def get_per_token_logps(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
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
        )
        return token_log_probs
    logits = outputs.logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs


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
        )
        return token_log_probs, entropies
    logits = outputs.logits[:, prompt_length - 1 :]
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
    """Extend prompt-side embeddings so GRPO scores the same prompt representation it sampled from."""

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
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
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
        straight_through_emulator: Optional function for quantization-aware
            straight-through gradient estimation.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: When ``is_training``
            is True, returns the updated state and loss metrics. When False,
            returns only the loss metrics.
    """
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
            tree = straight_through_emulator(tree)
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
                chunk_per_token_logps = get_per_token_logps(
                    module,
                    chunk_input_ids,
                    chunk_attention_mask,
                    prompt_len,
                    model_kwargs=chunk_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
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
                chunk_advantages = advantages[start:end]
                chunk_old_per_token_logps = (
                    old_per_token_logps[start:end]
                    if old_per_token_logps is not None
                    else jax.lax.stop_gradient(chunk_per_token_logps)
                )

                chunk_log_ratio = chunk_per_token_logps - chunk_old_per_token_logps
                if importance_sampling_level == "token":
                    chunk_log_importance_weights = chunk_log_ratio
                elif importance_sampling_level == "sequence":
                    chunk_log_importance_weights = (
                        (chunk_log_ratio * chunk_completion_mask).sum(axis=-1)
                        / jnp.maximum(chunk_completion_mask.sum(axis=-1), 1.0)
                    )[:, None]
                else:
                    raise ValueError(
                        f"Unknown importance sampling level: {importance_sampling_level}. "
                        "Possible values are 'token' and 'sequence'."
                    )

                coef_1 = jnp.exp(chunk_log_importance_weights)
                if loss_type == "cispo":
                    clamped_ratios = jnp.minimum(coef_1, epsilon_high)
                    chunk_per_token_loss = -clamped_ratios * chunk_advantages * chunk_per_token_logps
                elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
                    coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)
                    if delta is not None:
                        coef_1 = jnp.minimum(coef_1, delta)
                    per_token_loss1 = coef_1 * chunk_advantages
                    per_token_loss2 = coef_2 * chunk_advantages
                    chunk_per_token_loss = -jnp.where(
                        chunk_advantages >= 0,
                        jnp.minimum(per_token_loss1, per_token_loss2),
                        jnp.maximum(per_token_loss1, per_token_loss2),
                    )
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                if beta != 0.0:
                    chunk_per_token_loss = chunk_per_token_loss + beta * chunk_per_token_kl

                if loss_type == "grpo":
                    loss_numerator = loss_numerator + jnp.sum(
                        jnp.sum(chunk_per_token_loss * chunk_completion_mask, axis=1)
                        / jnp.maximum(jnp.sum(chunk_completion_mask, axis=1), 1.0)
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

                if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
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

            if loss_type == "grpo":
                loss = loss_numerator / jnp.maximum(jnp.array(completion_ids.shape[0], dtype=jnp.float32), 1.0)
            elif loss_type == "bnpo":
                loss = loss_numerator / jnp.maximum(completion_token_count, 1.0)
            elif loss_type == "dr_grpo":
                loss = loss_numerator / (completion_ids.shape[0] * completion_ids.shape[1])
            elif loss_type in ["cispo", "dapo"]:
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
            if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
                other_metrics["clip_ratio/low_mean"] = low_clip_num / jnp.maximum(low_clip_den, 1.0)
                other_metrics["clip_ratio/high_mean"] = high_clip_num / jnp.maximum(high_clip_den, 1.0)
                other_metrics["clip_ratio/region_mean"] = region_clip_num / jnp.maximum(region_clip_den, 1.0)
            elif loss_type == "cispo":
                other_metrics["cispo_clip_ratio"] = cispo_clip_num / jnp.maximum(cispo_clip_den, 1.0)

            return loss, LossMetrics(
                loss=loss,
                accuracy=1,
                other_metrics=other_metrics,
            )

        entropies = None
        if top_entropy_quantile < 1.0:
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

        if beta != 0.0:
            ref_per_token_logps = minibatch["ref_per_token_logps"][:, : completion_ids.shape[1]]
            per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        else:
            per_token_kl = jnp.zeros_like(per_token_logps)

        advantages = minibatch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        old_per_token_logps = minibatch.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        log_ratio = per_token_logps - old_per_token_logps
        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(axis=-1) / jnp.maximum(
                completion_mask.sum(axis=-1), 1.0
            )
            log_importance_weights = log_importance_weights[:, None]
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = jnp.exp(log_importance_weights)

        if loss_type == "cispo":
            clamped_ratios = jnp.minimum(coef_1, epsilon_high)
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)
            if delta is not None:
                coef_1 = jnp.minimum(coef_1, delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            # Use min for A >= 0, max for A < 0 (pessimistic bound)
            per_token_loss = -jnp.where(
                advantages >= 0,
                jnp.minimum(per_token_loss1, per_token_loss2),
                jnp.maximum(per_token_loss1, per_token_loss2),
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if top_entropy_quantile < 1.0 and entropies is not None:
            masked_entropies = jnp.where(completion_mask > 0, entropies, jnp.nan)
            entropy_threshold = jnp.nanquantile(masked_entropies, 1 - top_entropy_quantile)
            entropy_mask = (entropies >= entropy_threshold).astype(completion_mask.dtype) * completion_mask
            per_token_loss = per_token_loss * entropy_mask

        if beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl

        completion_token_count = jnp.sum(completion_mask)
        completion_lengths = jnp.sum(completion_mask, axis=1)

        if loss_type == "grpo":
            loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / jnp.maximum(completion_lengths, 1.0))
        elif loss_type == "bnpo":
            loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(completion_token_count, 1.0)
        elif loss_type == "dr_grpo":
            loss = jnp.sum(per_token_loss * completion_mask) / (per_token_loss.shape[0] * per_token_loss.shape[1])
        elif loss_type in ["cispo", "dapo"]:
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
            other_metrics["ref_per_token_logps"] = jnp.mean(ref_per_token_logps)
        else:
            mean_kl = None

        if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            is_low_clipped = (coef_1 < 1 - epsilon) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            other_metrics["clip_ratio/low_mean"] = masked_mean(is_low_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/high_mean"] = masked_mean(is_high_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/region_mean"] = masked_mean(is_region_clipped.astype(jnp.float32))
        elif loss_type == "cispo":
            is_cispo_clipped = (coef_1 > epsilon_high) & (advantages > 0)
            other_metrics["cispo_clip_ratio"] = masked_mean(is_cispo_clipped.astype(jnp.float32))

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics=other_metrics,
        )

    # Compute gradients and metrics across minibatches.
    if is_training:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
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
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics
