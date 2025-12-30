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

"""Internal functions for PPO training.

Implements the PPO clipped objective for language-model RLHF training, including:
- Per-token log-probabilities and entropies
- Value head predictions
- Clipped policy loss and clipped value loss
"""

from __future__ import annotations

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


def _masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute the mean of masked elements.

    Args:
        x: Input array of values.
        mask: Binary mask indicating which elements to include.

    Returns:
        Mean of elements where mask is non-zero, or 0 if mask is empty.
    """
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(x * mask) / denom


def compute_per_token_logps(logits: jax.Array, input_ids: jax.Array, prompt_length: int) -> jax.Array:
    """Compute per-token log probabilities for completion tokens.

    Extracts the log probabilities of the actual tokens generated in the
    completion portion of the sequence, given the model's logit predictions.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        input_ids: Full input sequence including prompt and completion.
        prompt_length: Length of the prompt prefix to skip.

    Returns:
        Per-token log probabilities for the completion tokens,
        shape (batch, completion_length).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_ids = input_ids[:, prompt_length:]
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(target_ids, axis=-1),
        axis=-1,
    )
    return jnp.squeeze(token_log_probs, axis=-1)


def get_per_token_logps_values_entropies(model, input_ids: jax.Array, attention_mask: jax.Array, prompt_length: int):
    """Compute per-token log probabilities, values, and entropies for PPO.

    Performs a forward pass through the model with value head to obtain
    all quantities needed for PPO loss computation.

    Args:
        model: CausalLMWithValueHead model instance.
        input_ids: Full input sequence (prompt + completion).
        attention_mask: Attention mask for the sequence.
        prompt_length: Length of the prompt prefix.

    Returns:
        Tuple of (token_log_probs, values, entropies):
            - token_log_probs: Log probabilities for completion tokens.
            - values: Value head predictions for completion positions.
            - entropies: Per-token entropy of the output distribution.

    Raises:
        ValueError: If model outputs don't provide hidden states.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    logits = outputs.logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]

    token_log_probs = compute_per_token_logps(logits, input_ids, prompt_length)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropies = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)

    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model outputs do not provide hidden states; cannot compute value head outputs.")
        hidden_states = hidden_states[-1]

    values_full = model.value_head(hidden_states).squeeze(-1)
    values = values_full[:, prompt_length - 1 : -1]
    return token_log_probs, values, entropies


def ppo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    prompt_length: int,
    cliprange: float,
    vf_coef: float,
    cliprange_value: float,
    entropy_coef: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute a single PPO training or evaluation step.

    Computes the clipped PPO objective including policy loss, value loss,
    and optional entropy bonus. Supports gradient accumulation and
    sharded training across devices.

    Args:
        state: Current model state with parameters and optimizer.
        batch: Dictionary containing:
            - input_ids: Token IDs for prompt + completion.
            - attention_mask: Attention mask for the sequence.
            - completion_mask: Mask for completion tokens only.
            - old_logps: Log probabilities from rollout policy.
            - old_values: Value predictions from rollout.
            - advantages: GAE-computed advantages.
            - returns: Target returns for value function.
        prompt_length: Length of the prompt prefix.
        cliprange: PPO clip range for policy ratio.
        vf_coef: Coefficient for value function loss.
        cliprange_value: Clip range for value function.
        entropy_coef: Coefficient for entropy bonus.
        loss_config: Optional configuration for loss computation.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for distributed training.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        is_training: Whether to compute gradients and update state.

    Returns:
        If is_training: Tuple of (updated_state, metrics).
        If not is_training: Just the metrics.
    """
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

        input_ids = minibatch["input_ids"]
        attention_mask = minibatch["attention_mask"]
        completion_mask = minibatch["completion_mask"].astype(jnp.float32)

        old_logps = minibatch["old_logps"]
        old_values = minibatch["old_values"]
        advantages = jax.lax.stop_gradient(minibatch["advantages"])
        returns = jax.lax.stop_gradient(minibatch["returns"])

        new_logps, new_values, entropies = get_per_token_logps_values_entropies(
            module,
            input_ids,
            attention_mask,
            prompt_length,
        )

        log_ratio = new_logps - old_logps
        ratio = jnp.exp(log_ratio)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = _masked_mean(jnp.maximum(pg_losses, pg_losses2), completion_mask)

        values_clipped = old_values + jnp.clip(new_values - old_values, -cliprange_value, cliprange_value)
        vf_losses1 = jnp.square(new_values - returns)
        vf_losses2 = jnp.square(values_clipped - returns)
        vf_loss = 0.5 * _masked_mean(jnp.maximum(vf_losses1, vf_losses2), completion_mask)

        entropy = _masked_mean(entropies, completion_mask)
        loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy

        approx_kl = 0.5 * _masked_mean(jnp.square(log_ratio), completion_mask)
        pg_clipfrac = _masked_mean(((pg_losses2 > pg_losses).astype(jnp.float32)), completion_mask)
        vf_clipfrac = _masked_mean(((vf_losses2 > vf_losses1).astype(jnp.float32)), completion_mask)

        other_metrics: dict[str, jax.Array] = {
            "policy_loss": pg_loss,
            "value_loss": vf_loss,
            "mean_entropy": entropy,
            "approx_kl": approx_kl,
            "pg_clipfrac": pg_clipfrac,
            "vf_clipfrac": vf_clipfrac,
            "ratio_mean": _masked_mean(ratio, completion_mask),
            "advantages_mean": _masked_mean(advantages, completion_mask),
            "returns_mean": _masked_mean(returns, completion_mask),
        }

        return loss, LossMetrics(loss=loss, accuracy=1, other_metrics=other_metrics)

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

    _, metrics = loss_fn(state.graphstate, batch)
    return metrics
