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

"""Core training and evaluation step functions.

This module provides the fundamental training and evaluation step implementations
used by the Trainer class. These functions handle:

- Training step: Gradient computation, model updates, and metrics tracking
- Evaluation step: Loss computation and metrics collection without updates
- Minibatch processing for gradient accumulation
- Distributed training with sharding constraints

The functions are designed to be JIT-compiled for optimal performance
and support various model architectures through the EasyDeLState abstraction.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
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


def base_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run the shared base trainer loss path for train or eval."""
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """
        Computes the loss and additional metrics for a given minibatch and tree state.

        The inner function merges the current graph state with an updated tree,
        prepares inputs for a model call, pops the labels from the inputs, and calls
        the model's compute_loss method.

        Args:
            tree: The current update to the model's graph state.
            minibatch: A minibatch of input data.

        Returns:
            A tuple containing:
                - The computed loss (scalar).
                - Additional metrics (LossMetrics) produced during loss computation.
        """
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)
        if not is_training:
            module.eval()
        call_batch = module.prepare_inputs_for_call(**minibatch)
        labels = call_batch.pop("labels", None)
        outputs, metrics = module.compute_loss(
            labels=labels,
            loss_config=loss_config,
            **call_batch,
        )
        return outputs.loss, metrics

    if not is_training:
        _, metrics = loss_fn(state.graphstate, batch)
        return metrics

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


def training_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Perform one base trainer update step."""
    return base_step(
        state=state,
        batch=batch,
        loss_config=loss_config,
        learning_rate_fn=learning_rate_fn,
        partition_spec=partition_spec,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_training=True,
        straight_through_emulator=straight_through_emulator,
    )


def evaluation_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Perform one base trainer evaluation step."""
    return base_step(
        state=state,
        batch=batch,
        loss_config=loss_config,
        partition_spec=partition_spec,
        gradient_accumulation_steps=1,
        is_training=False,
    )


def _base_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    return scheduled_loss_cache_key(
        call,
        value_fields=("partition_spec",),
        object_fields=("loss_config", "straight_through_emulator"),
    )


def _make_base_scheduled_loss(call):
    loss_config = call.get("loss_config")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree, batch):
        module = bind_scheduled_module(call, tree)
        batch = constrain_scheduled_batch(module, batch, partition_spec)
        call_batch = module.prepare_inputs_for_call(**batch)
        labels = call_batch.pop("labels", None)
        outputs, _metrics = module.compute_loss(
            labels=labels,
            loss_config=loss_config,
            **call_batch,
        )
        return outputs.loss

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="base",
        make_loss=_make_base_scheduled_loss,
        make_cache_key=_base_scheduled_loss_cache_key,
    ),
)
