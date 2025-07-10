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

import typing as tp

import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully


def training_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    center_rewards_coefficient: float | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """
    Performs a single training step by computing gradients via minibatch processing,
    updating the model state, and returning updated state and loss metrics.

    The function first determines the batch and minibatch sizes using assertions.
    It then applies sharding constraints to the batch. The loss function is defined
    as an inner function that merges the current model state with an updated tree,
    prepares the inputs, and computes the loss using the model's compute_loss method.
    Gradients are computed using `jax.value_and_grad` over minibatches. The state is updated
    respectfully using the computed gradients and updated metrics.

    Args:
        state (EasyDeLState): The current model state, which includes parameters and model graph.
        batch (tp.Mapping[str, jax.Array]): A mapping of input arrays for the current batch.
        loss_config (tp.Optional[LossConfig], optional): Configuration settings for the loss
            computation. Defaults to None.
        learning_rate_fn (optax.Schedule, optional): A schedule function for the learning rate.
            Defaults to None.
        partition_spec (tp.Optional[PartitionSpec], optional): Specification for data sharding.
            Defaults to None.
        gradient_accumulation_steps (int, optional): Number of steps over which to accumulate gradients.
            Defaults to 1.
        center_rewards_coefficient (int, optional): Coefficient to incentivize the reward model to
        output mean-zero rewards.

    Returns:
        tp.Tuple[EasyDeLState, LossMetrics]:
            A tuple containing:
                - The updated EasyDeLState after applying gradients.
                - LossMetrics containing computed loss and other related metrics.
    """
    # Determine batch size, minibatch size, and enforce partition spec.
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

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
        # Merge the state with the provided tree update.
        module = flax.nnx.merge(state.graphdef, tree, state.graphother)

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


def evaluation_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    center_rewards_coefficient: float | None = None,
) -> tuple[tp.Any, LossMetrics]:
    """
    Performs a single evaluation step by computing loss metrics for the input batch.

    The function determines the required partitioning for the batch, applies sharding constraints,
    and defines an inner loss function. This inner function merges the current state with the graph state,
    sets the model to evaluation mode, and computes loss and metrics via the model's compute_loss method.
    The computed LossMetrics are then returned.

    Args:
        state (EasyDeLState): The current model state.
        batch (tp.Mapping[str, jax.Array]): A mapping of input arrays for evaluation.
        loss_config (tp.Optional[LossConfig], optional): Configuration for loss computation.
            Defaults to None.
        partition_spec (tp.Optional[PartitionSpec], optional): Specification for sharding the batch.
            Defaults to None.
        center_rewards_coefficient (int, optional): Coefficient to incentivize the reward
            model to output mean-zero rewards.

    Returns:
        tp.Tuple[tp.Any, LossMetrics]:
            A tuple containing:
                - (Any): An additional output from loss computation (if any).
                - LossMetrics: The computed loss metrics for the evaluation batch.
    """
    # Enforce partitioning constraints and determine required sharding.
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree):
        """
        Computes loss metrics for the evaluation batch given a merged graph state.

        This inner function merges the provided tree with the current state,
        sets the module to evaluation mode, removes the labels from the batch,
        and computes the loss metrics via the module's compute_loss method.

        Args:
            tree: The current update of the model's graph state.

        Returns:
            LossMetrics: The computed metrics from the loss function.
        """

        # Merge the state with the provided tree update.
        module = flax.nnx.merge(state.graphdef, tree, state.graphother)

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
