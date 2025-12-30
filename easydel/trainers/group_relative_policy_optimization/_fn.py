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

import typing as tp

import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)

RewardFunc = tp.Union[EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


def get_per_token_logps(model, input_ids, attention_mask, prompt_length):
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

    Returns:
        Array: Per-token log probabilities for the completion portion.
            Shape: [batch_size, seq_len - prompt_length]

    Note:
        The function shifts logits by one position to align with the autoregressive
        nature of language models, where each position predicts the next token.
    """

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs = compute_per_token_logps(logits, input_ids, prompt_length)
    return token_log_probs


def compute_per_token_logps(logits, input_ids, prompt_length):
    """
    Compute per-token log probabilities in a vectorized way.

    Args:
        logits: Pre-trimmed logits [batch_size, seq_len, vocab_size]
        input_ids: Input token ids [batch_size, seq_len]
        prompt_length: Length of the prompt
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


def get_per_token_logps_and_entropies(model, input_ids, attention_mask, prompt_length):
    """Return per-token log probabilities and entropies for the completion portion."""
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(input_ids[:, prompt_length:], axis=-1),
        axis=-1,
    )
    token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
    entropies = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
    return token_log_probs, entropies


def grpo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
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
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    # Determine batch size, minibatch size, and enforce partition spec.
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
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

        input_ids = jnp.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)
        prompt_len = prompt_ids.shape[-1]

        per_token_logps, entropies = get_per_token_logps_and_entropies(
            module,
            input_ids,
            attention_mask,
            prompt_len,
        )

        if beta != 0.0:
            ref_per_token_logps = minibatch["ref_per_token_logps"]
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

        if top_entropy_quantile < 1.0:
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
            normalizer = minibatch.get("num_items_in_batch", completion_token_count)
            loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(normalizer, 1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        def masked_mean(x):
            if x.shape[1] == 1:
                return jnp.mean(x)
            return jnp.sum(x * completion_mask) / jnp.maximum(completion_token_count, 1.0)

        other_metrics: dict[str, jax.Array] = {
            "mean_entropy": masked_mean(entropies),
            "advantages": jnp.mean(advantages),
        }

        if beta != 0.0:
            mean_kl = masked_mean(per_token_kl)
            other_metrics["mean_kl"] = mean_kl
            other_metrics["ref_per_token_logps"] = jnp.mean(minibatch["ref_per_token_logps"])
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
