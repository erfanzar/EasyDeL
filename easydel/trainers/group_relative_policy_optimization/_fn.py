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

import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully

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
        chex.Array: Per-token log probabilities for the completion portion.
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


def grpo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    per_token_weighting: bool = True,
    is_training: bool = True,
) -> tuple[EasyDeLState, LossMetrics]:
    # Determine batch size, minibatch size, and enforce partition spec.
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(state.graphdef, tree, state.graphother)

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            advantages,
            token_weights,
        ) = (
            minibatch["prompt_ids"],
            minibatch["prompt_mask"],
            minibatch["completion_ids"],
            minibatch["completion_mask"],
            minibatch["advantages"],
            minibatch.get("token_weights"),
        )

        input_ids = jnp.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)

        per_token_logps = get_per_token_logps(module, input_ids, attention_mask, prompt_ids.shape[-1])

        ref_per_token_logps = minibatch["ref_per_token_logps"]

        ratios = jnp.exp(per_token_logps - ref_per_token_logps)
        clipped_ratios = jnp.where(
            jnp.expand_dims(advantages, 1) > 0,
            jnp.minimum(ratios, 1.0 + epsilon_high),
            jnp.maximum(ratios, 1.0 - epsilon_low),
        )

        expanded_adv = jnp.expand_dims(advantages, 1)
        if token_weights is None or not per_token_weighting:
            weights = completion_mask
        else:
            weights = token_weights * completion_mask

        weight_sums = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        policy_terms = jnp.minimum(ratios * expanded_adv, clipped_ratios * expanded_adv)
        policy_loss = -jnp.sum(weights * policy_terms, axis=1) / weight_sums

        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (
            ref_per_token_logps - per_token_logps
        ) - 1.0
        kl_contrib = jnp.sum(weights * per_token_kl, axis=1) / weight_sums
        mean_kl = jnp.mean(kl_contrib)

        loss = policy_loss + beta * kl_contrib
        loss = jnp.mean(loss)

        entropy = -jnp.mean(jnp.sum(weights * per_token_logps, axis=1) / weight_sums)
        weighted_total = jnp.sum(weights)
        ratio_mean = jnp.where(
            weighted_total > 0,
            jnp.sum(ratios * weights) / weighted_total,
            0.0,
        )
        clip_fraction = jnp.where(
            weighted_total > 0,
            jnp.sum(jnp.not_equal(ratios, clipped_ratios) * weights) / weighted_total,
            0.0,
        )

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_kl": mean_kl,
                "policy_loss": jnp.mean(policy_loss),
                "entropy": entropy,
                "ratio_mean": ratio_mean,
                "clip_fraction": clip_fraction,
                "advantages": jnp.mean(advantages),
            },
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
