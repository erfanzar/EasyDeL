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

"""Loss scaling implementations for mixed precision training.

This module provides loss scaling utilities to prevent gradient underflow and
overflow when training with low-precision dtypes like float16. Dynamic loss
scaling automatically adjusts the scale factor based on gradient health.
"""

import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class LossScaleConfig:
    """Configuration parameters for dynamic loss scaling behavior.

    This dataclass holds the hyperparameters that control how dynamic loss
    scaling behaves during training. The defaults are tuned for typical
    mixed precision training scenarios.

    Attributes:
        initial_scale: The starting loss scale value. Higher values provide
            more headroom for gradients but risk overflow. Default is 2^15
            (32768), which works well for most models.
        growth_interval: Number of consecutive steps with finite gradients
            required before increasing the loss scale. Default is 2000 steps.
        scale_factor: Multiplicative factor for scaling adjustments. The scale
            is multiplied by this factor when increasing and divided by it
            when decreasing. Default is 2.
        min_scale: Minimum allowed loss scale value. Prevents the scale from
            becoming too small, which could cause gradient underflow. Default
            is 1.0.

    Example:
        Default configuration::

            config = LossScaleConfig()
            # initial_scale=32768, growth_interval=2000, scale_factor=2, min_scale=1.0

        Aggressive scaling for stable training::

            config = LossScaleConfig(
                initial_scale=2**16,
                growth_interval=1000,
                scale_factor=2,
                min_scale=1.0
            )

        Conservative scaling for unstable models::

            config = LossScaleConfig(
                initial_scale=2**10,
                growth_interval=5000,
                scale_factor=2,
                min_scale=1.0
            )
    """

    initial_scale: float = 2**15
    growth_interval: int = 2000
    scale_factor: int = 2
    min_scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class NoOpLossScale:
    """No-operation loss scaler that passes values through unchanged.

    This class implements the loss scaler interface but performs no actual
    scaling. It is used when loss scaling is disabled, such as for full
    precision (float32) training where gradient underflow is not a concern.

    Using NoOpLossScale instead of conditionally removing loss scaling logic
    simplifies code by maintaining a consistent interface regardless of whether
    scaling is enabled.

    Attributes:
        loss_scale: Always returns 1 (no scaling applied).

    Example:
        >>> scaler = NoOpLossScale()
        >>> loss = jnp.array(0.5)
        >>> scaled = scaler.scale(loss)
        >>> scaled == loss
        True
        >>> scaler.adjust(True) is scaler
        True
    """

    @property
    def loss_scale(self):
        """Return the loss scale value.

        Returns:
            int: Always returns 1, indicating no scaling is applied.
        """
        return 1

    def scale(self, tree: T) -> T:
        """Scale values by the loss scale factor (no-op).

        Args:
            tree: A PyTree of arrays to scale.

        Returns:
            The input tree unchanged.
        """
        return tree

    def unscale(self, tree: T) -> T:
        """Unscale values by dividing by the loss scale factor (no-op).

        Args:
            tree: A PyTree of scaled arrays to unscale.

        Returns:
            The input tree unchanged.
        """
        return tree

    def adjust(self, grads_finite: jnp.ndarray):
        """Adjust the loss scale based on gradient health (no-op).

        Args:
            grads_finite: A boolean array indicating whether gradients are
                finite (contains no NaN or Inf values). Ignored by NoOpLossScale.

        Returns:
            NoOpLossScale: Returns self unchanged since no adjustment is needed.
        """
        return self


@dataclasses.dataclass(frozen=True)
class DynamicLossScale:
    """Dynamic loss scaling for mixed precision training.

    This class implements dynamic loss scaling, which automatically adjusts
    the loss scale factor during training to prevent gradient underflow and
    overflow. This is essential for stable training with low-precision dtypes
    like float16.

    The algorithm works as follows:
    1. Scale the loss by the current loss_scale before computing gradients
    2. After computing gradients, unscale them by dividing by loss_scale
    3. Check if gradients are finite (no NaN or Inf values)
    4. If finite: increment counter, increase scale after `period` steps
    5. If not finite: reduce scale by `factor`, reset counter

    This class is immutable (frozen dataclass), so adjust() returns a new
    instance rather than modifying in place. This design is compatible with
    JAX's functional programming paradigm.

    Attributes:
        loss_scale: Current loss scale value as a JAX array.
        counter: Number of consecutive steps with finite gradients.
        period: Steps between scale increases when gradients are stable.
        factor: Multiplicative factor for scale adjustments.
        min_loss_scale: Minimum allowed loss scale to prevent underflow.

    Example:
        Basic usage in a training loop::

            scaler = DynamicLossScale(
                loss_scale=jnp.array(2**15),
                period=2000,
                factor=2,
                min_loss_scale=jnp.array(1.0)
            )

            for batch in data_loader:
                # Compute loss and gradients
                loss, grads = compute_loss_and_grads(params, batch)

                # Scale and unscale
                scaled_loss = scaler.scale(loss)
                unscaled_grads = scaler.unscale(grads)

                # Check gradient health and update scaler
                grads_finite = check_finite(unscaled_grads)
                scaler = scaler.adjust(grads_finite)

                # Only update params if gradients are valid
                if grads_finite:
                    params = update_params(params, unscaled_grads)
    """

    loss_scale: jnp.ndarray
    counter: jnp.ndarray = dataclasses.field(default_factory=lambda: np.zeros([], np.int32))
    period: int = 2000
    factor: int = 2
    min_loss_scale: jnp.ndarray = dataclasses.field(default_factory=lambda: np.ones([], np.float32))

    def scale(self, tree: T) -> T:
        """Scale values by multiplying with the loss scale factor.

        This method multiplies all values in the input PyTree by the current
        loss scale. Typically applied to the loss before gradient computation
        to prevent gradient underflow.

        Args:
            tree: A PyTree of arrays to scale. Can be a single array, dict,
                list, tuple, or any nested structure of arrays.

        Returns:
            A new PyTree with the same structure where all arrays have been
            multiplied by the loss_scale.

        Example:
            >>> scaler = DynamicLossScale(loss_scale=jnp.array(1024.0))
            >>> loss = jnp.array(0.001)
            >>> scaled_loss = scaler.scale(loss)
            >>> scaled_loss
            Array(1.024, dtype=float32)
        """
        return jax.tree_util.tree_map(lambda x: x * self.loss_scale, tree)

    def unscale(self, tree: T) -> T:
        """Unscale values by dividing by the loss scale factor.

        This method divides all values in the input PyTree by the current
        loss scale. Typically applied to gradients after computation to
        restore them to their true (unscaled) values.

        Args:
            tree: A PyTree of scaled arrays to unscale. Can be a single array,
                dict, list, tuple, or any nested structure of arrays.

        Returns:
            A new PyTree with the same structure where all arrays have been
            divided by the loss_scale.

        Example:
            >>> scaler = DynamicLossScale(loss_scale=jnp.array(1024.0))
            >>> scaled_grads = {"w": jnp.array(1024.0)}
            >>> unscaled_grads = scaler.unscale(scaled_grads)
            >>> unscaled_grads["w"]
            Array(1.0, dtype=float32)
        """
        return jax.tree_util.tree_map(lambda x: x / self.loss_scale, tree)

    def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
        """Adjust the loss scale based on gradient health.

        This method implements the core dynamic scaling logic:
        - If gradients are finite: increment counter, optionally increase scale
        - If gradients are non-finite: decrease scale, reset counter

        The scale is increased when the counter reaches (period - 1), indicating
        that gradients have been stable for `period` consecutive steps. The scale
        is decreased immediately when non-finite gradients are detected.

        Args:
            grads_finite: A boolean JAX array indicating whether all gradients
                are finite (True) or contain NaN/Inf values (False).

        Returns:
            DynamicLossScale: A new DynamicLossScale instance with updated
            loss_scale and counter values.

        Example:
            >>> scaler = DynamicLossScale(loss_scale=jnp.array(1024.0))
            >>> # Finite gradients - counter increases
            >>> scaler = scaler.adjust(jnp.array(True))
            >>> scaler.counter
            Array(1, dtype=int32)
            >>> # Non-finite gradients - scale decreases
            >>> scaler = scaler.adjust(jnp.array(False))
            >>> scaler.loss_scale
            Array(512., dtype=float32)

        Note:
            The method uses JAX's lax.select for XLA-compatible conditional
            logic, ensuring the operation can be traced and JIT-compiled.
        """

        def first_finite(a, b):
            return jax.lax.select(jnp.isfinite(a).all(), a, b)

        loss_scale = jax.lax.select(
            grads_finite,
            jax.lax.select(
                self.counter == (self.period - 1),
                first_finite(self.loss_scale * self.factor, self.loss_scale),
                self.loss_scale,
            ),
            jnp.maximum(self.min_loss_scale, self.loss_scale / self.factor),
        )

        counter = ((self.counter + 1) % self.period) * grads_finite

        return DynamicLossScale(
            loss_scale=loss_scale,
            counter=counter,
            period=self.period,
            factor=self.factor,
            min_loss_scale=self.min_loss_scale,
        )
