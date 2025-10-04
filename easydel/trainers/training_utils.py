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

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.utils.helpers import check_bool_flag

SCAN_TRAINER = check_bool_flag("SCAN_TRAINER")
FAST_COMPILE = check_bool_flag("FAST_COMPILE")


def make_assertions_and_get_sizes(
    batch: dict,
    gradient_accumulation_steps: int,
    batch_partition_spec: PartitionSpec | None = None,
) -> tuple[int, int, PartitionSpec]:
    """
    Validates the input parameters and computes the batch size, minibatch size, and batch partition specification.
    Args:
        batch (tp.Dict): A dictionary containing the batch data. The batch size is inferred from the
            first element's shape.
        gradient_accumulation_steps (int): The number of gradient accumulation steps. Must be greater than 0.
        batch_partition_spec (tp.Optional[PartitionSpec], optional): The partition specification for the batch.
            Defaults to None.
    Returns:
        tp.Tuple[int, int, PartitionSpec]: A tuple containing:
            - batch_size (int): The size of the batch.
            - minibatch_size (int): The size of the minibatch.
            - batch_partition_spec (PartitionSpec): The partition specification for the batch.
    Raises:
            ValueError: If `gradient_accumulation_steps` is not greater than 0.
            ValueError: If the batch size is not divisible by the gradient accumulation steps.
    """

    batch_size = batch[next(iter(batch.keys()))].shape[0]
    minibatch_size = batch_size // gradient_accumulation_steps
    if not gradient_accumulation_steps > 0:
        ValueError("`gradient_accumulation_steps` must be greater than 0.")
    if minibatch_size * gradient_accumulation_steps != batch_size:
        raise ValueError("Batch size must be divisible by gradient accumulation steps.")
    if batch_partition_spec is None:
        batch_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
    return batch_size, minibatch_size, batch_partition_spec


def update_metrics(
    metrics: LossMetrics,
    learning_rate_fn: tp.Callable,
    step: int | jax.Array,
    gradients: jax.Array | None,
) -> LossMetrics:
    """
    Updates the given metrics with the current learning rate and gradient norms.

    Args:
            metrics (LossMetrics): An instance of LossMetrics to be updated.
            learning_rate_fn (tp.Callable): A callable that returns the learning rate given the current step.
            step (int | jax.Array): The current training step.
            gradients (Optional(jax.Array)): The gradients to compute norms from.

    Returns:
            LossMetrics: The updated metrics with learning rate and gradient norms.
    """
    if learning_rate_fn is not None:
        metrics.learning_rate = learning_rate_fn(step)
    if gradients is not None:
        grad_norms = tu.tree_map(jnp.linalg.norm, gradients)
        metrics.max_grad_norm = tu.tree_reduce(jnp.maximum, grad_norms)
        grad_size = tu.tree_reduce(jnp.add, tu.tree_map(jnp.size, grad_norms))
        grad_sum = tu.tree_reduce(jnp.add, tu.tree_map(jnp.sum, grad_norms))
        metrics.mean_grad_norm = grad_sum / grad_size
        metrics.grad_norms = grad_norms
    return metrics


def update_state_respectfully(
    state: EasyDeLState,
    gradients: jax.Array,
    loss_config: LossConfig,
    metrics: LossMetrics,
) -> EasyDeLState:
    """
    Updates the state of the model respectfully based on the provided gradients, loss configuration, and metrics.

    Args:
            state (EasyDeLState): The current state of the model.
            gradients (jax.Array): The gradients to be applied to the model's parameters.
            loss_config (LossConfig): Configuration for the loss, including conditions for breaking on NaN values.
            metrics (LossMetrics): Metrics containing the loss value.

    Returns:
            EasyDeLState: The updated state of the model.
    """
    if FAST_COMPILE:
        return state.apply_gradients(grads=gradients)
    else:

        def update_fn(args):
            state, gradients = args
            return state.apply_gradients(grads=gradients)

        def skip_fn(args):
            state, _ = args
            return state

        should_update = True
        if loss_config is not None:
            should_update = lax.cond(
                loss_config.break_on_nan,
                lambda x: lax.cond(
                    jnp.isnan(metrics.loss),
                    lambda _: False,
                    lambda _: True,
                    None,
                ),
                lambda x: True,
                None,
            )
        state = lax.cond(should_update, update_fn, skip_fn, (state, gradients))
        return state


def minibatch_call(
    state: EasyDeLState,
    batch: dict,
    minibatch_size: int,
    grad_fn: tp.Callable[[jax.Array, dict], tuple[jax.Array, LossMetrics]],
) -> tuple[jax.Array, LossMetrics]:
    """
    Processes batch in smaller chunks for gradient accumulation using jax.lax.scan.
    Uses eval_shape to initialize accumulator structures efficiently.
    """
    num_accum_steps = len(next(iter(batch.values()))) // minibatch_size
    if num_accum_steps > 1:

        def reshape_to_minibatches(arr):
            """Reshape the batch into minibatches for accumulation."""
            batch_shape = (num_accum_steps, minibatch_size, *arr.shape[1:])
            return jnp.reshape(arr, batch_shape)

        batch = jax.tree_util.tree_map(reshape_to_minibatches, batch)

        (_, metrics_shape), grads_shape = jax.eval_shape(
            grad_fn,
            state.graphstate,
            jax.tree_util.tree_map(lambda x: x[0], batch),
        )

        init_acc = {
            "grads": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shape),
            "metrics": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape),
        }

        def accumulate_gradients(acc, minibatch):
            """Accumulate gradients and metrics for each minibatch."""
            (_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
            new_acc = {
                "grads": jax.tree_util.tree_map(jnp.add, acc["grads"], step_grads),
                "metrics": jax.tree_util.tree_map(jnp.add, acc["metrics"], step_aux),
            }
            return new_acc, step_aux

        final_acc, _aux = jax.lax.scan(
            accumulate_gradients,
            init_acc,
            batch,
            length=num_accum_steps,
        )
        gradients = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["grads"])
        metrics = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["metrics"])

    else:
        (_, metrics), gradients = grad_fn(state.graphstate, batch)

    return gradients, metrics


def compute_group_advantages(
    rewards: jax.Array,
    num_generations: int,
    epsilon: float = 1e-4,
    enforce_mixed: bool = True,
    jitter: float = 1e-3,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute group-normalised advantages following the GRPO/DAPO recipe.

    Args:
        rewards: Flat vector of rewards shaped (batch_size * num_generations,).
        num_generations: Number of completions sampled per prompt.
        epsilon: Small stabiliser added to the group standard deviation.
        enforce_mixed: If True, inject a perturbation when the group variance collapses.
        jitter: Magnitude of the perturbation applied to collapsed groups.

    Returns:
        Tuple containing (flattened advantages, group means, group stds, collapsed mask).
    """

    rewards = rewards.reshape(-1, num_generations)
    group_mean = jnp.mean(rewards, axis=1, keepdims=True)
    centered = rewards - group_mean
    group_std = jnp.std(centered, axis=1, keepdims=True)
    safe_std = jnp.maximum(group_std, epsilon)
    advantages = centered / safe_std

    collapsed = (group_std[..., 0] < epsilon) & enforce_mixed
    if enforce_mixed:
        if num_generations == 1:
            fallback = jnp.zeros_like(advantages)
        else:
            neg = -jitter / jnp.maximum(1, num_generations - 1)
            fallback = jnp.full_like(advantages, neg)
            fallback = fallback.at[:, 0].set(jitter)
        advantages = jnp.where(collapsed[:, None], fallback, advantages)

    return (
        advantages.reshape(-1),
        group_mean.reshape(-1),
        safe_std.reshape(-1),
        collapsed,
    )


def compute_length_reward(
    lengths: jax.Array,
    max_completion_length: int,
    cache_tokens: int,
    mode: str,
    scale: float,
) -> jax.Array:
    """DAPO-style reward shaping based on completion length."""

    if mode == "none":
        return jnp.zeros_like(lengths, dtype=jnp.float32)

    lengths = lengths.astype(jnp.float32)
    lmax = jnp.asarray(max_completion_length, dtype=jnp.float32)
    cache = jnp.asarray(cache_tokens, dtype=jnp.float32)
    shoulder = jnp.maximum(0.0, lmax - cache)

    if mode == "linear":
        inside = jnp.where(
            lengths <= shoulder,
            0.0,
            jnp.where(
                lengths <= lmax,
                (shoulder - lengths) / jnp.maximum(cache, 1.0),
                -1.0,
            ),
        )
        shaped = inside
    elif mode == "punitive":
        shaped = -jnp.maximum(0.0, lengths - shoulder) / jnp.maximum(cache, 1.0)
    else:
        raise ValueError(f"Unknown length shaping mode: {mode}")

    return scale * shaped


def update_ema(previous: jax.Array | None, value: jax.Array, horizon: int) -> jax.Array:
    """Update an exponential moving average with horizon H."""

    if horizon <= 0:
        return value
    alpha = 1.0 / float(max(1, horizon))
    if previous is None:
        return value
    return (1.0 - alpha) * previous + alpha * value
