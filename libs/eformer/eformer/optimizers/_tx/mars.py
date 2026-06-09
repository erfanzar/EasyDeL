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


from typing import Any, NamedTuple

import chex
import jax
import optax
from jax import numpy as jnp
from optax import tree_utils as otu
from optax._src import transform


class ScaleByMarsState(NamedTuple):
    """State for the Mars (Matrix-wise Adaptive Regularized Scaling) algorithm.

    This named tuple holds the optimizer state required by the Mars algorithm,
    tracking moments and gradient history for variance reduction.

    Attributes:
        count (chex.Array): Integer array tracking the current optimization step.
            Used for bias correction of the moment estimates.
        mu (optax.Updates): First moment estimates (exponential moving average of gradients).
            Has the same structure as the model parameters.
        nu (optax.Updates): Second moment estimates (exponential moving average of squared gradients).
            Has the same structure as the model parameters.
        mog (optax.Updates): Momentum of gradients from the previous step.
            Used for variance reduction in the Mars algorithm.
    """

    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates
    mog: optax.Updates


def scale_by_mars(
    b1: float = 0.9,
    b2: float = 0.999,
    gamma: float = 0.05,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    max_grad_norm: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the Mars algorithm.

    Mars uses a variance reduction technique that incorporates gradient momentum
    from the previous step, improving upon standard Adam-style optimizers.

    Reference: https://arxiv.org/abs/2411.10438

    Args:
        b1 (float): Decay rate for the exponentially weighted average of gradients.
            Controls how quickly the first moment estimate adapts to new gradients.
            Defaults to 0.9.
        b2 (float): Decay rate for the exponentially weighted average of squared gradients.
            Controls how quickly the second moment estimate adapts. Defaults to 0.999.
        gamma (float): Decay rate for the exponentially weighted average of gradient
            momentum from the previous step. This parameter controls the variance
            reduction strength. Defaults to 0.05.
        eps (float): Small constant added to the denominator to improve numerical stability.
            Prevents division by zero. Defaults to 1e-8.
        eps_root (float): Small constant added inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
            Defaults to 0.0.
        max_grad_norm (float): Maximum gradient norm for clipping. If > 0, gradients
            are clipped to this norm before computing moment estimates. Defaults to 0.0
            (no clipping).
        mu_dtype (Any | None): Optional dtype for the first moment accumulator.
            If None, dtype is inferred from params and updates. Defaults to None.

    Returns:
        optax.GradientTransformation: A gradient transformation that rescales updates
            according to the Mars algorithm.

    Example:
        >>> import optax
        >>> from eformer.optimizers._tx import scale_by_mars
        >>> optimizer = optax.chain(
        ...     scale_by_mars(b1=0.95, b2=0.99, gamma=0.025),
        ...     optax.scale_by_learning_rate(1e-4),
        ... )
    """

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        """Initialize the Mars optimizer state.

        Args:
            params: Model parameters, used to create zero-initialized moment estimates
                with matching structure and shapes.

        Returns:
            ScaleByMarsState: Initial optimizer state with zeroed moments and step count.
        """
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        mog = otu.tree_zeros_like(params, dtype=mu_dtype)
        return ScaleByMarsState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, mog=mog)

    def update_fn(updates, state, params=None):
        """Compute Mars gradient updates.

        Applies the Mars algorithm to scale gradients using variance reduction
        with gradient momentum from the previous step.

        Args:
            updates: Gradient updates to be rescaled.
            state: Current Mars optimizer state containing moments.
            params: Model parameters (unused, but required by optax interface).

        Returns:
            tuple: (rescaled_updates, new_state) where rescaled_updates are the
                Mars-transformed gradients and new_state is the updated optimizer state.
        """
        c = jax.tree.map(
            lambda og, g: None if g is None else g + (gamma * b1 / (1 - b1)) * (g - og),
            state.mog,
            updates,
            is_leaf=lambda x: x is None,
        )
        if max_grad_norm:
            g_norm = optax.global_norm(c)
            scale = jnp.minimum(1.0, max_grad_norm / (g_norm + 1e-6))
            c = jax.tree_map(lambda g: None if g is None else g * scale, c, is_leaf=lambda x: x is None)
        mu = otu.tree_update_moment(c, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(c, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)
        return adam_updates, ScaleByMarsState(count=count_inc, mu=mu, nu=nu, mog=updates)

    return optax.GradientTransformation(init_fn, update_fn)


def mars(learning_rate: float | optax.Schedule, **kwargs) -> optax.GradientTransformation:
    """Mars (Matrix-wise Adaptive Regularized Scaling) optimizer.

    Complete Mars optimizer that combines Mars gradient scaling with learning rate
    scheduling. Mars uses a variance reduction technique that incorporates gradient
    momentum from the previous step, providing improved convergence over Adam.

    Reference: https://arxiv.org/abs/2411.10438

    Args:
        learning_rate (float | optax.Schedule): Learning rate value or schedule function.
            Can be a constant float or an optax.Schedule that takes step count as input.
        **kwargs: Additional keyword arguments passed to scale_by_mars. See scale_by_mars
            for available options including:
            - b1 (float): Decay rate for first moment. Defaults to 0.9.
            - b2 (float): Decay rate for second moment. Defaults to 0.999.
            - gamma (float): Variance reduction strength. Defaults to 0.05.
            - eps (float): Numerical stability constant. Defaults to 1e-8.
            - max_grad_norm (float): Gradient clipping norm. Defaults to 0.0.
            - mu_dtype: Data type for moment accumulators.

    Returns:
        optax.GradientTransformation: The Mars optimizer ready for use with
            optax.apply_updates.

    Example:
        >>> import optax
        >>> from eformer.optimizers._tx import mars
        >>> # With constant learning rate
        >>> optimizer = mars(learning_rate=1e-4, b1=0.95, b2=0.99)
        >>> # With learning rate schedule
        >>> schedule = optax.warmup_cosine_decay_schedule(
        ...     init_value=1e-7, peak_value=1e-4, warmup_steps=1000, decay_steps=10000
        ... )
        >>> optimizer = mars(learning_rate=schedule, gamma=0.025)
    """
    return optax.chain(scale_by_mars(**kwargs), transform.scale_by_learning_rate(learning_rate))
