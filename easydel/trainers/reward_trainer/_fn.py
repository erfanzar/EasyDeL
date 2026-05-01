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

"""Internal functions for Reward Model training.

This module contains the core computational functions used by the reward trainer,
implementing training and evaluation steps for reward models in RLHF pipelines.
Reward models learn to predict human preferences between pairs of model outputs,
serving as a proxy for human judgment when training policies with reinforcement learning.

The module provides functions for:
- Training step computation with pairwise ranking losses
- Evaluation step for assessing reward model accuracy
- Support for Bradley-Terry model and margin-based losses
- Reward centering and normalization strategies

The reward model is trained to assign higher scores to preferred (chosen) responses
compared to non-preferred (rejected) responses, learning from human preference data.

All functions are JAX-compatible and support distributed training through sharding.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)


def training_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    center_rewards_coefficient: float | None = None,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one reward-model training step (forward + Bradley-Terry loss + update).

    The reward model produces a scalar score for the chosen and the
    rejected branch via two forward passes; the optimisation target is
    the (optionally margin-shifted) Bradley-Terry log-sigmoid loss

    ``L_BT = -E[ log sigmoid(r_chosen - r_rejected - margin) ]``

    plus an optional centring penalty
    ``center_rewards_coefficient * E[(r_chosen + r_rejected)^2]`` that
    discourages the model from drifting away from mean-zero rewards.
    The loss is differentiated under :func:`minibatch_call`, which
    handles gradient accumulation; the resulting gradients are passed
    to :func:`update_state_respectfully` for the optimiser update and
    to :func:`update_metrics` for diagnostic reporting.

    Args:
        state: Current reward model state (parameters + optimiser).
        batch: Mapping carrying ``input_ids_chosen``,
            ``attention_mask_chosen``, ``input_ids_rejected``,
            ``attention_mask_rejected`` and optionally a per-pair
            ``margin`` array.
        loss_config: Optional :class:`LossConfig` consumed by
            :func:`update_state_respectfully` (clipping, etc.).
        learning_rate_fn: Optional schedule used by
            :func:`update_metrics` for per-step LR logging.
        partition_spec: Sharding spec applied to the batch by
            :func:`make_assertions_and_get_sizes`.
        gradient_accumulation_steps: Number of microbatches whose
            gradients are accumulated per optimiser step.
        center_rewards_coefficient: Optional scalar weight on the
            centring penalty; ``None`` disables centring.
        straight_through_emulator: Optional STE wrapper applied to the
            parameter tree inside the loss closure to simulate
            quantised forwards while keeping gradients differentiable.

    Returns:
        ``(updated_state, metrics)`` where ``metrics`` carries the
        scalar loss alongside the per-branch reward arrays.
    """
    # Determine batch size, minibatch size, and enforce partition spec.
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute the reward-model Bradley-Terry loss for one microbatch.

        The closure rebinds ``tree`` to the trainer state's graph
        definition (optionally through ``straight_through_emulator`` to
        emulate quantised forwards), runs two independent forward passes
        -- one on the chosen branch and one on the rejected branch --
        and reads the scalar score from the model's ``logits`` output.
        Pairwise log-sigmoid is computed on the score difference, with a
        per-pair ``margin`` shift when provided. When
        ``center_rewards_coefficient`` is not ``None`` the centring
        regulariser is added.

        Args:
            tree: Differentiable parameter tree.
            minibatch: Mapping with ``input_ids_chosen``,
                ``attention_mask_chosen``, ``input_ids_rejected``,
                ``attention_mask_rejected``, and optionally ``margin``.

        Returns:
            ``(loss, LossMetrics(loss, chosen_rewards, rejected_rewards))``
            where ``loss`` is the scalar BT objective and the two reward
            arrays carry per-sample scalar predictions for downstream
            accuracy/diagnostic computation.
        """
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)

        rewards_chosen = module(
            input_ids=minibatch["input_ids_chosen"],
            attention_mask=minibatch["attention_mask_chosen"],
        ).logits
        rewards_rejected = module(
            input_ids=minibatch["input_ids_rejected"],
            attention_mask=minibatch["attention_mask_rejected"],
        ).logits
        if "margin" in minibatch:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected - minibatch["margin"]))
        else:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected))

        if center_rewards_coefficient is not None:
            loss += center_rewards_coefficient * jax.numpy.mean((rewards_chosen + rewards_rejected) ** 2)
        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=rewards_chosen,
            rejected_rewards=rewards_rejected,
        )
        return loss, metrics

    # Compute gradients and metrics across minibatches.
    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
    )
    # Update state using the computed gradients and updated metrics.
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


def _reward_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the cache key for a scheduled reward-loss specialization.

    Args:
        call: Scheduled call descriptor capturing the bound static arguments.

    Returns:
        tuple[tp.Any, ...]: Hashable identifier for the
        ``(partition_spec, center_rewards_coefficient,
        straight_through_emulator)`` tuple.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("partition_spec", "center_rewards_coefficient"),
        object_fields=("straight_through_emulator",),
    )


def _make_reward_scheduled_loss(call):
    """Build a scalar reward-model loss closure for the scheduled-loss adapter.

    Concatenates chosen/rejected sequences, runs a single forward pass and
    splits the resulting scalar rewards before applying the (optionally
    margin-shifted) Bradley-Terry log-sigmoid loss with optional
    reward-centering.

    Args:
        call: Scheduled call descriptor providing ``partition_spec`` and
            ``center_rewards_coefficient``.

    Returns:
        tp.Callable: A function ``(tree, batch) -> Array`` returning the
        scalar pairwise ranking loss.
    """
    partition_spec = call.get("partition_spec")
    center_rewards_coefficient = call.get("center_rewards_coefficient")

    def scheduled_loss(tree: spx.State, batch: collections.abc.Mapping[str, jax.Array]):
        """Compute the reward-model scalar loss for the scheduled-loss adapter.

        Args:
            tree (spx.State): Current parameter tree.
            batch (collections.abc.Mapping[str, jax.Array]): Minibatch with
                ``input_ids_chosen`` / ``_rejected`` and optional ``margin``.

        Returns:
            jax.Array: Scalar loss.
        """
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)

        input_ids = jax.numpy.concatenate(
            [call_batch["input_ids_chosen"], call_batch["input_ids_rejected"]],
            axis=0,
        )
        attention_mask = jax.numpy.concatenate(
            [call_batch["attention_mask_chosen"], call_batch["attention_mask_rejected"]],
            axis=0,
        )
        rewards = module(input_ids=input_ids, attention_mask=attention_mask).logits
        rewards_chosen, rewards_rejected = jax.numpy.split(rewards, 2, axis=0)

        if "margin" in call_batch:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected - call_batch["margin"]))
        else:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected))

        if center_rewards_coefficient is not None:
            loss += center_rewards_coefficient * jax.numpy.mean((rewards_chosen + rewards_rejected) ** 2)
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="reward",
        make_loss=_make_reward_scheduled_loss,
        make_cache_key=_reward_scheduled_loss_cache_key,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    center_rewards_coefficient: float | None = None,
) -> LossMetrics:
    """Run one reward-model evaluation step (forward + loss only, no gradients).

    Identical algorithmically to :func:`training_step` minus the
    backward and optimiser update: chosen and rejected branches are
    each forwarded through the reward backbone, the Bradley-Terry
    log-sigmoid loss is computed (with optional per-pair ``margin``
    and centring penalty), and the per-branch scalar rewards are
    returned for downstream accuracy diagnostics. The loss closure
    receives the current ``state.graphstate`` directly because no
    gradient accumulation is needed.

    Args:
        state: Current reward model state.
        batch: Mapping carrying the same keys consumed by
            :func:`training_step`.
        loss_config: Reserved for parity with the training step; not
            consumed by the eval closure.
        partition_spec: Sharding spec applied to the batch.
        center_rewards_coefficient: Optional centring weight; ``None``
            disables centring.

    Returns:
        :class:`LossMetrics` carrying the scalar BT loss together with
        the per-branch reward arrays.
    """
    # Enforce partitioning constraints and determine required sharding.
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree):
        """Compute the reward-model BT loss/metrics for an eval batch (no gradient).

        Mirrors the training-mode closure but receives the full eval
        batch (no microbatch splitting) and skips the
        straight-through-emulator branch -- evaluation runs the model
        in its natural precision. Returns a :class:`LossMetrics`
        instance instead of the ``(loss, metrics)`` pair expected by
        :func:`jax.value_and_grad` because no gradient is taken.

        Args:
            tree: Parameter tree (typically ``state.graphstate``)
                used directly without straight-through emulation.

        Returns:
            :class:`LossMetrics` with ``loss``, ``chosen_rewards``, and
            ``rejected_rewards`` populated.
        """

        # Merge the state with the provided tree update.
        module = state.merge(tree)

        rewards_chosen = module(
            input_ids=batch["input_ids_chosen"],
            attention_mask=batch["attention_mask_chosen"],
        ).logits
        rewards_rejected = module(
            input_ids=batch["input_ids_rejected"],
            attention_mask=batch["attention_mask_rejected"],
        ).logits
        if "margin" in batch:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected - batch["margin"]))
        else:
            loss = -jax.numpy.mean(jax.nn.log_sigmoid(rewards_chosen - rewards_rejected))

        if center_rewards_coefficient is not None:
            loss += center_rewards_coefficient * jax.numpy.mean((rewards_chosen + rewards_rejected) ** 2)
        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=rewards_chosen,
            rejected_rewards=rewards_rejected,
        )
        return metrics

    metrics = loss_fn(state.graphstate)
    return metrics
