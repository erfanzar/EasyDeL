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

"""Precision handler implementation for mixed precision operations.

This module provides the PrecisionHandler class that manages dtype casting
and loss scaling for mixed precision training and inference in JAX.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..loss_scaling.loss_scaler import DynamicLossScale, LossScaleConfig, NoOpLossScale
from ..policy.policy import Policy


def _cast_to_dtype(tree: Any, dtype: jnp.dtype) -> Any:
    """Cast floating point values in a PyTree to the specified dtype.

    This function traverses a JAX PyTree and casts all floating-point arrays
    (both numpy and jax.numpy) to the target dtype, while leaving non-floating
    point values unchanged.

    Args:
        tree: A JAX PyTree containing arrays and/or other nested structures.
            Can include numpy arrays, JAX arrays, or any nested combination
            of lists, tuples, and dicts containing arrays.
        dtype: The target JAX numpy dtype to cast floating-point arrays to.
            Common values include jnp.float16, jnp.bfloat16, jnp.float32.

    Returns:
        A new PyTree with the same structure as the input, where all
        floating-point arrays have been cast to the specified dtype.
        Non-floating point arrays and non-array leaves are returned unchanged.

    Example:
        >>> import jax.numpy as jnp
        >>> tree = {"weights": jnp.ones((3, 3), dtype=jnp.float32), "bias": jnp.zeros(3)}
        >>> casted = _cast_to_dtype(tree, jnp.float16)
        >>> casted["weights"].dtype
        dtype('float16')
    """

    def conditional_cast(x):
        if isinstance(x, np.ndarray | jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


class PrecisionHandler:
    """Handles mixed precision operations for training and inference.

    This class provides a unified interface for managing mixed precision training
    and inference in JAX. It combines a precision policy (defining dtypes for
    parameters, computations, and outputs) with optional dynamic loss scaling
    to prevent gradient underflow in low-precision training.

    The handler can wrap training step functions and inference functions to
    automatically handle dtype casting and loss scaling, making it easier to
    implement mixed precision workflows.

    Attributes:
        policy: The precision policy defining dtypes for different operations.
        loss_scale_config: Configuration for loss scaling behavior.
        loss_scaler: The loss scaler instance (DynamicLossScale or NoOpLossScale).

    Example:
        Basic mixed precision training setup::

            from eformer.mpric import PrecisionHandler

            # Create handler with bfloat16 compute, float32 params
            handler = PrecisionHandler(
                policy="p=f32,c=bf16,o=f32",
                use_dynamic_scale=True
            )

            # Wrap your training step
            @handler.training_step_wrapper
            def train_step(params, batch):
                loss, grads = compute_loss_and_grads(params, batch)
                return loss, grads

            # Training loop
            for batch in data_loader:
                loss, grads, grads_finite = train_step(params, batch)
                if grads_finite:
                    params = update_params(params, grads)
    """

    def __init__(
        self,
        policy: str | Policy,
        use_dynamic_scale: bool = True,
        loss_scale_config: LossScaleConfig = None,
    ):
        """Initialize the PrecisionHandler.

        Args:
            policy: The precision policy to use. Can be either a Policy object
                or a string specification. String format supports:
                - Simple dtype: "f32", "bf16", "f16" (applies to all operations)
                - Detailed spec: "p=f32,c=bf16,o=f32" where:
                    - p/params: parameter dtype
                    - c/compute: computation dtype
                    - o/output: output dtype
            use_dynamic_scale: If True, uses dynamic loss scaling to prevent
                gradient underflow/overflow. If False, uses no-op scaling.
                Defaults to True.
            loss_scale_config: Configuration for the loss scaler. If None,
                uses default LossScaleConfig values. Only used when
                use_dynamic_scale is True.

        Example:
            >>> handler = PrecisionHandler("p=f32,c=f16,o=f32")
            >>> handler.policy.compute_dtype
            dtype('float16')
        """
        self.policy = policy if isinstance(policy, Policy) else Policy.from_string(policy)
        self.loss_scale_config = loss_scale_config or LossScaleConfig()

        if use_dynamic_scale:
            self.loss_scaler = DynamicLossScale(
                loss_scale=jnp.array(self.loss_scale_config.initial_scale),
                period=self.loss_scale_config.growth_interval,
                factor=self.loss_scale_config.scale_factor,
                min_loss_scale=jnp.array(self.loss_scale_config.min_scale),
            )
        else:
            self.loss_scaler = NoOpLossScale()

    @partial(jax.jit, static_argnums=(0,))
    def cast_for_compute(self, x: Any) -> Any:
        """Cast input arrays to the computation dtype.

        This method is JIT-compiled for performance and casts all floating-point
        arrays in the input PyTree to the computation dtype specified by the policy.

        Args:
            x: A JAX PyTree containing arrays to cast. Can be a single array,
                a nested dict, list, tuple, or any valid PyTree structure.

        Returns:
            A new PyTree with the same structure where all floating-point arrays
            have been cast to the policy's compute_dtype.

        Note:
            This method is decorated with @jax.jit for efficient execution.
            Non-floating point arrays are returned unchanged.
        """
        return _cast_to_dtype(x, self.policy.compute_dtype)

    @partial(jax.jit, static_argnums=(0,))
    def cast_for_output(self, x: Any) -> Any:
        """Cast arrays to the output dtype.

        This method is JIT-compiled for performance and casts all floating-point
        arrays in the input PyTree to the output dtype specified by the policy.

        Args:
            x: A JAX PyTree containing arrays to cast. Can be a single array,
                a nested dict, list, tuple, or any valid PyTree structure.

        Returns:
            A new PyTree with the same structure where all floating-point arrays
            have been cast to the policy's output_dtype.

        Note:
            This method is decorated with @jax.jit for efficient execution.
            Non-floating point arrays are returned unchanged.
        """
        return _cast_to_dtype(x, self.policy.output_dtype)

    def cast_params(self, params: Any) -> Any:
        """Cast model parameters to the parameter dtype.

        This method casts all floating-point arrays in the parameters PyTree
        to the parameter dtype specified by the policy. Typically used to
        maintain high precision for stored parameters.

        Args:
            params: A PyTree of model parameters. Typically a nested dict
                containing weight and bias arrays.

        Returns:
            A new PyTree with the same structure where all floating-point arrays
            have been cast to the policy's param_dtype.

        Example:
            >>> params = {"layer1": {"weights": jnp.ones((3, 3), dtype=jnp.float16)}}
            >>> casted_params = handler.cast_params(params)
            >>> casted_params["layer1"]["weights"].dtype
            dtype('float32')  # if param_dtype is float32
        """
        return _cast_to_dtype(params, self.policy.param_dtype)

    def training_step_wrapper(self, training_step_fn):
        """Wrap a training step function with precision and loss scaling handling.

        This decorator wraps a training step function to automatically handle:
        1. Casting inputs to the computation dtype before the forward/backward pass
        2. Scaling the loss and unscaling gradients for numerical stability
        3. Checking gradient finiteness and adjusting the loss scale accordingly
        4. Casting outputs to the output dtype

        The wrapped function expects the original training step to return a tuple
        of (loss, grads) and will return (loss, grads, grads_finite).

        Args:
            training_step_fn: A callable that takes model inputs and returns
                a tuple of (loss, gradients). The function signature should be:
                ``def training_step(*args, **kwargs) -> Tuple[Array, PyTree]``

        Returns:
            A wrapped function with signature:
            ``def wrapped_step(*args, **kwargs) -> Tuple[Array, PyTree, bool]``
            returning (loss, gradients, grads_finite) where grads_finite indicates
            whether all gradients are finite (no NaN or Inf values).

        Example:
            >>> handler = PrecisionHandler("p=f32,c=f16,o=f32")
            >>>
            >>> def my_train_step(params, batch):
            ...     loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            ...     return loss, grads
            >>>
            >>> wrapped_step = handler.training_step_wrapper(my_train_step)
            >>> loss, grads, grads_ok = wrapped_step(params, batch)
            >>> if grads_ok:
            ...     params = apply_gradients(params, grads)

        Note:
            The loss scaler state is updated internally after each call. If
            gradients contain NaN or Inf values, the loss scale is reduced.
            After a period of stable gradients, the scale is increased.
        """

        def wrapped_step(*args, **kwargs):
            args = jax.tree_util.tree_map(self.cast_for_compute, args)
            kwargs = jax.tree_util.tree_map(self.cast_for_compute, kwargs)

            loss, grads = training_step_fn(*args, **kwargs)

            scaled_loss = self.loss_scaler.scale(loss)

            grads = self.loss_scaler.unscale(grads)

            grads_finite = jax.tree_util.tree_reduce(
                lambda x, y: x and jnp.all(jnp.isfinite(y)),
                grads,
                True,
            )

            self.loss_scaler = self.loss_scaler.adjust(grads_finite)

            unscaled_loss = self.loss_scaler.unscale(scaled_loss)
            final_loss = self.cast_for_output(unscaled_loss)
            final_grads = jax.tree_util.tree_map(self.cast_for_output, grads)

            return final_loss, final_grads, grads_finite

        return wrapped_step

    def inference_wrapper(self, inference_fn):
        """Wrap an inference function with precision handling.

        This decorator wraps an inference function to automatically handle
        dtype casting for inputs and outputs according to the precision policy.
        Unlike training_step_wrapper, this does not perform loss scaling as
        gradients are not computed during inference.

        Args:
            inference_fn: A callable that performs inference. The function can
                have any signature and return any PyTree structure.

        Returns:
            A wrapped function with the same signature as the input function,
            where inputs are cast to compute_dtype and outputs are cast to
            output_dtype.

        Example:
            >>> handler = PrecisionHandler("p=f32,c=f16,o=f32")
            >>>
            >>> def my_inference(params, inputs):
            ...     return model.apply(params, inputs)
            >>>
            >>> wrapped_inference = handler.inference_wrapper(my_inference)
            >>> outputs = wrapped_inference(params, inputs)

        Note:
            This wrapper is suitable for both single-sample inference and
            batched inference. All floating-point arrays in both args and
            kwargs are cast to the computation dtype before calling the
            wrapped function.
        """

        def wrapped_inference(*args, **kwargs):
            args = jax.tree_util.tree_map(self.cast_for_compute, args)
            kwargs = jax.tree_util.tree_map(self.cast_for_compute, kwargs)
            outputs = inference_fn(*args, **kwargs)
            return jax.tree_util.tree_map(self.cast_for_output, outputs)

        return wrapped_inference
