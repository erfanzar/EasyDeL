# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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
import warnings

import chex
import jax
import optax
from jax import numpy as jnp


class OptaxScheduledWeightDecayState(tp.NamedTuple):
    """State for the scheduled weight decay optimizer.

    This named tuple holds the state required by the scheduled weight decay
    transformation, tracking the current step count for schedule evaluation.

    Attributes:
        count (chex.Array): Integer array tracking the current optimization step.
            Used to evaluate the weight decay schedule function.
    """

    count: chex.Array


def optax_add_scheduled_weight_decay(
    schedule_fn: tp.Callable[[chex.Array], chex.Array],
    mask: chex.ArrayTree | None = None,
) -> optax.GradientTransformation:
    """
    Create an optax optimizer that applies weight decay on a schedule.

    This function is similar to `optax.add_decayed_weights`, but it allows for
    the weight decay rate to be scheduled over training steps.

    Args:
        schedule_fn: A function that takes the current step count as input
                      and returns the weight decay rate.
        mask: A PyTree with the same structure as the parameters.
              A value of True at a particular location indicates that weight
              decay should be applied to that parameter.

    Returns:
        An `optax.GradientTransformation` object representing the optimizer.
    """

    def init_fn(params: chex.ArrayTree) -> OptaxScheduledWeightDecayState:
        """Initialize the state of the scheduled weight decay optimizer.

        Args:
            params (chex.ArrayTree): Parameter tree (unused, but required by optax interface).

        Returns:
            OptaxScheduledWeightDecayState: Initial state with step count set to zero.
        """
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(
        updates: chex.ArrayTree,
        state: OptaxScheduledWeightDecayState,
        params: chex.ArrayTree | None = None,
    ) -> tuple[chex.ArrayTree, OptaxScheduledWeightDecayState]:
        """Apply scheduled weight decay to the gradient updates.

        Computes the weight decay rate from the schedule function and adds
        the scaled parameters to the gradient updates.

        Args:
            updates (chex.ArrayTree): Gradient updates to be modified.
            state (OptaxScheduledWeightDecayState): Current optimizer state containing step count.
            params (chex.ArrayTree | None): Model parameters for weight decay computation.

        Returns:
            tuple[chex.ArrayTree, OptaxScheduledWeightDecayState]: Tuple containing:
                - Modified gradient updates with weight decay applied.
                - Updated state with incremented step count.

        Raises:
            ValueError: If params is None, as weight decay requires parameter values.
        """
        if params is None:
            raise ValueError("Params cannot be None for weight decay!")

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        return updates, OptaxScheduledWeightDecayState(count=optax.safe_int32_increment(state.count))

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


def create_linear_scheduler(
    steps: int,
    learning_rate_start: float,
    learning_rate_end: float,
    warmup_steps: int | None = None,
) -> optax.Schedule:
    """
    Creates a linear learning rate scheduler with optional warmup.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Initial learning rate.
        learning_rate_end (float): Final learning rate.
        warmup_steps (tp.Optional[int]): Number of warmup steps.

    Returns:
        optax.Schedule: The configured scheduler.
    """
    if warmup_steps:
        scheduler_warmup = optax.linear_schedule(
            init_value=5e-8,
            end_value=learning_rate_start,
            transition_steps=warmup_steps,
        )
        scheduler_decay = optax.linear_schedule(
            init_value=learning_rate_start,
            end_value=learning_rate_end,
            transition_steps=steps - warmup_steps,
        )
        return optax.join_schedules(schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps])
    else:
        return optax.linear_schedule(
            init_value=learning_rate_start,
            end_value=learning_rate_end,
            transition_steps=steps,
        )


def create_cosine_scheduler(
    steps: int,
    learning_rate: float,
    learning_rate_end: float | None = None,
    warmup_steps: int | None = None,
    exponent: float = 1.0,
) -> optax.Schedule:
    """
    Creates a cosine learning rate scheduler with optional warmup.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate.
        learning_rate_end (tp.Optional[float]): Final learning rate.
        warmup_steps (tp.Optional[int]): Number of warmup steps.
        exponent (float): Exponent for the cosine decay.

    Returns:
        optax.Schedule: The configured scheduler.
    """
    if warmup_steps:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.5e-7,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=steps,
            end_value=learning_rate_end if learning_rate_end is not None else 0.0,
            exponent=exponent,
        )
    if learning_rate_end is not None and learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0 when learning_rate_end is set")

    cosine_alpha = 0.0 if learning_rate_end is None else learning_rate_end / learning_rate
    return optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=steps, alpha=cosine_alpha)


def get_base_optimizer(
    optimizer_type: str,
    scheduler: optax.Schedule,
    optimizer_kwargs: dict,
    weight_decay: float = 0.0,
    weight_decay_mask: tp.Any | None = None,
    gradient_accumulation_steps: int = 1,
    clip_grad: float | None = None,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Base function to create an optimizer with a given scheduler.

    Args:
        optimizer_type (str): Type of optimizer ('adafactor', 'adamw', 'lion', 'rmsprop').
        scheduler (optax.Schedule): Learning rate scheduler.
        optimizer_kwargs (dict): Arguments specific to the optimizer.
        weight_decay (float): Weight decay factor.
        weight_decay_mask (tp.Optional[tp.Any]): Mask for weight decay.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (tp.Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        optax.GradientTransformation: The configured optimizer.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.", stacklevel=1)

    if optimizer_type == "adafactor":
        optimizer = optax.adafactor(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "adamw":
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.setdefault("weight_decay", 0.0)
        optimizer = optax.adamw(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "lion":
        optimizer = optax.lion(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=scheduler, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    chain = [optimizer]

    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))

    if weight_decay != 0.0:
        chain.append(
            optax_add_scheduled_weight_decay(
                lambda step: -scheduler(step) * weight_decay,
                weight_decay_mask,
            )
        )

    tx = optax.chain(*chain)

    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)

    return tx
