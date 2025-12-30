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

import typing as tp

import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def _compute_policy_logps(
    module: flax.nnx.Module,
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    completion_ids: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
    """Compute policy log probabilities for completion tokens.

    Args:
        module: Policy model module.
        prompt_ids: Prompt token IDs.
        prompt_mask: Prompt attention mask.
        completion_ids: Completion token IDs.
        completion_mask: Completion attention mask.

    Returns:
        Per-token log probabilities for completions.
    """

    input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
    attention_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
    prompt_length = prompt_ids.shape[-1]
    return get_per_token_logps(module, input_ids, attention_mask, prompt_length)


def nash_md_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    beta: float,
    loss_config: LossConfig | None,
    learning_rate_fn,
    partition_spec: PartitionSpec | None,
    gradient_accumulation_steps: int,
    is_train: bool,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute Nash-MD training or evaluation step.

    Args:
        state: Current model state.
        batch: Input batch with prompts, completions, and rewards.
        beta: Temperature parameter for KL penalty.
        loss_config: Optional loss configuration.
        learning_rate_fn: Function mapping step to learning rate.
        partition_spec: Sharding specification.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        is_train: Whether this is a training step.

    Returns:
        Updated state and metrics (if training) or just metrics (if eval).
    """

    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)
    beta = jnp.asarray(beta, dtype=jnp.float32)

    def loss_fn(tree: flax.nnx.GraphState, minibatch: dict[str, jax.Array]):
        if is_train and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        completion_ids = minibatch["completion_ids"]
        completion_mask = minibatch["completion_mask"]
        ref_token_logps = minibatch["ref_token_logps"]
        probabilities = minibatch["probabilities"]

        policy_token_logps = _compute_policy_logps(module, prompt_ids, prompt_mask, completion_ids, completion_mask)
        mask = completion_mask.astype(policy_token_logps.dtype)
        policy_token_logps = policy_token_logps * mask
        ref_token_logps = ref_token_logps * mask

        policy_logps = policy_token_logps.sum(axis=1)
        log_ratio = policy_token_logps - ref_token_logps
        kl_vector = log_ratio.sum(axis=1)
        kl_loss = (log_ratio * policy_token_logps).sum(axis=1)

        score = (probabilities - 0.5) * policy_logps
        loss_vector = beta * kl_loss - score
        loss = loss_vector.mean()

        metrics = LossMetrics(
            loss=loss,
            other_metrics={
                "score": score.mean(),
                "kl": kl_vector.mean(),
                "probability": probabilities.mean(),
                "policy_logps": policy_logps.mean(),
            },
        )
        return loss, metrics

    if is_train:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        metrics = update_metrics(
            metrics=metrics,
            learning_rate_fn=learning_rate_fn,
            step=state.step,
            gradients=gradients,
        )
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=metrics,
        )
        return state, metrics

    _, metrics = loss_fn(state.graphstate, batch)
    return metrics
