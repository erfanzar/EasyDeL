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
"""SSD eSurge self-distillation trainer."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def ssd_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: tp.Callable[[jax.Array], jax.Array] | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    logprob_vocab_chunk_size: int | None = None,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run an SSD cross-entropy step on eSurge-generated completions.

    The batch contains prompt and completion token tensors produced by the SSD
    trainer. The step concatenates them, computes completion-token log-probs,
    averages negative log-likelihood over active completion tokens, and updates
    state only when ``is_training`` is true.
    """
    scope_root = "easydel/trainer/ssd/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute SSD completion cross-entropy for one minibatch.

        The loss is normalized by the number of active completion tokens. Prompt
        tokens are present only as conditioning context and do not contribute to
        the objective.
        """
        if is_training and straight_through_emulator is not None:
            with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                tree = straight_through_emulator(tree)
        with jax.named_scope(scope_root + "/loss_fn/merge_state"):
            module = state.merge(tree)

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        completion_ids = minibatch["completion_ids"]
        completion_mask = minibatch["completion_mask"]
        input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
        per_token_logps = get_per_token_logps(
            module,
            input_ids,
            attention_mask,
            prompt_ids.shape[-1],
            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
        )
        token_count = jnp.sum(completion_mask)
        loss = -jnp.sum(per_token_logps * completion_mask) / jnp.maximum(token_count, 1.0)
        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "ssd/cross_entropy_loss": loss,
                "ssd/active_token_count": token_count,
            },
        )

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
    _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
    return metrics
