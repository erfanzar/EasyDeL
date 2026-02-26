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

"""Normalization layers for neural networks.

Provides efficient normalization layers optimized for JAX/Flax,
with support for mixed precision and float8 data types.

Classes:
    RMSNorm: Root Mean Square normalization layer

Constants:
    lowfloats: List of supported float8 data types

Key Features:
    - Efficient RMS normalization
    - Support for float8 quantization
    - Mixed precision computation
    - Automatic dtype promotion

Example:
    >>> from easydel.layers import RMSNorm
    >>> norm = RMSNorm(
    ...     dim=768,
    ...     eps=1e-6,
    ...     dtype=jnp.bfloat16
    ... )
    >>> normalized = norm(inputs)

Note:
    RMSNorm is particularly efficient for large language models
    as it requires fewer parameters than LayerNorm while providing
    similar normalization benefits.
"""

import collections.abc
import typing as tp

import jax
from eformer.common_types import Replicated
from flax import nnx as nn
from flax.nnx.nn import normalization as nutil
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

lowfloats = [
    jnp.float4_e2m1fn,
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2fnuz,
    jnp.float8_e5m2,
    jnp.float8_e3m4,
    jnp.float8_e4m3,
    jnp.float8_e8m0fnu,
]
"""List of supported low-precision floating point data types.

This list contains float8 and float4 data types that require special handling
during normalization operations. When input or parameter dtypes are in this list,
computations are promoted to float32 to maintain numerical stability.

Types included:
    - jnp.float4_e2m1fn: 4-bit float with 2 exponent and 1 mantissa bit
    - jnp.float8_e4m3b11fnuz: 8-bit float with 4 exponent and 3 mantissa bits (brain float variant)
    - jnp.float8_e4m3fn: 8-bit float with 4 exponent and 3 mantissa bits (FP8 E4M3)
    - jnp.float8_e4m3fnuz: 8-bit float with 4 exponent and 3 mantissa bits (unsigned zero)
    - jnp.float8_e5m2fnuz: 8-bit float with 5 exponent and 2 mantissa bits (unsigned zero)
    - jnp.float8_e5m2: 8-bit float with 5 exponent and 2 mantissa bits (FP8 E5M2)
    - jnp.float8_e3m4: 8-bit float with 3 exponent and 4 mantissa bits
    - jnp.float8_e4m3: 8-bit float with 4 exponent and 3 mantissa bits
    - jnp.float8_e8m0fnu: 8-bit float with 8 exponent bits (scaling factor format)

Note:
    These dtypes are commonly used in quantized training and inference for
    memory efficiency. The normalization layer automatically handles promotion
    to float32 when these types are detected.
"""


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    RMSNorm normalizes inputs by their root mean square value, providing a simpler
    and more computationally efficient alternative to Layer Normalization. Unlike
    LayerNorm, RMSNorm does not re-center activations (no learned bias/shift),
    which reduces parameter count while maintaining comparable performance in
    transformer architectures.

    The normalization formula is:
        output = (x / RMS(x)) * scale
        where RMS(x) = sqrt(mean(x^2) + eps)

    This implementation automatically handles low-precision floating point types
    (float8, float4) by promoting computations to float32 for numerical stability.

    Attributes:
        dim (int): Dimension of the input features (last axis size).
        eps (float): Small constant added to denominator for numerical stability.
        dtype (jnp.dtype): Data type for intermediate computations.
        param_dtype (jnp.dtype): Data type for learnable parameters (kernel).
        kernel (nn.Param): Learnable scale parameters of shape (dim,).

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx as nn
        >>> from easydel.layers.norms import RMSNorm
        >>>
        >>> # Create RMSNorm layer for 768-dim hidden states
        >>> norm = RMSNorm(
        ...     dim=768,
        ...     eps=1e-6,
        ...     dtype=jnp.bfloat16,
        ...     param_dtype=jnp.float32,
        ...     rngs=nn.Rngs(0)
        ... )
        >>>
        >>> # Apply to transformer hidden states
        >>> hidden_states = jnp.ones((2, 512, 768))
        >>> normalized = norm(hidden_states)
        >>> assert normalized.shape == (2, 512, 768)

    Note:
        RMSNorm is particularly popular in large language models like LLaMA,
        Mistral, and other recent architectures due to its efficiency and
        effectiveness. It typically provides 10-15% speedup compared to
        LayerNorm while maintaining model quality.

    See Also:
        - :data:`lowfloats`: List of dtypes that trigger float32 promotion.
    """

    kernel_init = staticmethod(nn.initializers.ones)
    """Callable: Static initializer for scale parameters, defaults to ones."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.bfloat16,
        *,
        rngs: nn.Rngs | None = None,
    ) -> None:
        """Initialize the RMSNorm layer.

        Creates a Root Mean Square normalization layer with learnable scale
        parameters. The layer normalizes inputs along the last dimension and
        applies element-wise scaling.

        Args:
            dim: Dimension of input features to normalize. This should match
                the last axis size of input tensors (typically hidden_size
                in transformer models).
            eps: Small constant added to the denominator for numerical stability.
                Prevents division by zero when input variance is very small.
                Defaults to 1e-6.
            dtype: Data type for intermediate computations during forward pass.
                Inputs are cast to this dtype (or float32 for low-precision types)
                before normalization. Defaults to jnp.bfloat16.
            param_dtype: Data type for storing learnable parameters (scale kernel).
                Using float32 for parameters while using bfloat16 for computation
                is a common mixed-precision strategy. Defaults to jnp.bfloat16.
            rngs: Flax NNX random number generators for parameter initialization.
                If None, creates a new Rngs instance with seed 0.

        Example:
            >>> norm = RMSNorm(
            ...     dim=768,
            ...     eps=1e-5,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.float32,
            ...     rngs=nn.Rngs(42)
            ... )
            >>> # Kernel is initialized to ones
            >>> assert norm.kernel.value.shape == (768,)
        """
        if rngs is None:
            rngs = nn.Rngs(0)
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(
            RMSNorm.kernel_init(rngs.params(), (self.dim,), self.param_dtype),
        )

    def _norm(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Compute Root Mean Square normalization without scaling.

        This internal method computes the core RMS normalization operation:
            output = x / sqrt(mean(x^2) + eps)

        The scale parameters (kernel) are applied separately in __call__.

        Args:
            x: Input array to normalize. Shape: (..., dim) where ... represents
                any number of leading batch dimensions. The normalization is
                applied along the last axis.

        Returns:
            Normalized array with the same shape as input. Each element is
            scaled by the reciprocal square root of the mean squared value
            along the last dimension.

        Note:
            Uses lax.rsqrt for efficient computation of 1/sqrt(x), which is
            typically faster than separate division and sqrt operations on
            accelerator hardware.
        """
        return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    @jax.named_scope("easydel-rmsnorm")
    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization to input tensor.

        Normalizes the input using Root Mean Square normalization and applies
        learned scale parameters. Automatically handles dtype promotion for
        numerical stability with low-precision types.

        The computation flow is:
            1. Cast input to computation dtype (float32 for low-precision types)
            2. Compute RMS normalization: x / sqrt(mean(x^2) + eps)
            3. Apply learned scale: output * kernel
            4. Cast back to original input dtype

        Args:
            x: Input array to normalize. Shape: (..., dim) where the last
                dimension must match the `dim` parameter used during initialization.
                Can have any number of leading batch dimensions.

        Returns:
            Normalized and scaled array with the same shape and dtype as input.
            Each feature vector along the last dimension is normalized to have
            unit RMS, then scaled by learned parameters.

        Note:
            The method is decorated with @jax.named_scope("easydel-rmsnorm")
            for profiling and debugging visibility in JAX traces.

        Example:
            >>> norm = RMSNorm(dim=4, dtype=jnp.float32, rngs=nn.Rngs(0))
            >>> x = jnp.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
            >>> y = norm(x)
            >>> # Each row is normalized independently
            >>> assert y.shape == (2, 4)
        """
        org_dtype = x.dtype
        if self.param_dtype in lowfloats or self.dtype in lowfloats:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.promote_types(self.dtype, x.dtype))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.astype(self.dtype)
        return (weight * output).astype(org_dtype)

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return dynamic partition specs for this module's parameters."""
        return {"kernel": Replicated}


class BatchNorm(nn.Module):
    """Batch Normalization layer.

    Normalizes activations across the batch dimension by subtracting the batch
    mean and dividing by the batch standard deviation. Maintains running
    statistics (mean and variance) for use during inference.

    During training, batch statistics are computed from the current mini-batch
    and exponential moving averages are updated. During inference, the stored
    running statistics are used instead.

    Attributes:
        num_features: Number of feature channels to normalize.
        use_running_average: If True, use stored running statistics instead
            of computing batch statistics.
        axis: Feature axis (or axes) to normalize over.
        momentum: Decay rate for exponential moving average of batch statistics.
        epsilon: Small constant for numerical stability in division.
        dtype: Computation dtype. Inputs are promoted to this dtype.
        param_dtype: Dtype for learnable scale and bias parameters.
        use_bias: Whether to include a learnable bias (shift) parameter.
        use_scale: Whether to include a learnable scale parameter.
        mean: Running mean batch statistic.
        var: Running variance batch statistic.
        scale: Learnable scale parameter, or None if ``use_scale=False``.
        bias: Learnable bias parameter, or None if ``use_bias=False``.

    Example:
        >>> norm = BatchNorm(num_features=64, rngs=nn.Rngs(0))
        >>> x = jnp.ones((8, 64))
        >>> y = norm(x, use_running_average=False)
        >>> assert y.shape == (8, 64)
    """

    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: nutil.Initializer = nn.initializers.zeros_init(),  # pyright: ignore[reportCallInDefaultInitializer]
        scale_init: nutil.Initializer = nn.initializers.ones_init(),  # pyright: ignore[reportCallInDefaultInitializer]
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: nutil.PromoteDtypeFn = nutil.dtypes.promote_dtype,
        rngs: nn.Rngs,
        bias_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}),  # pyright: ignore[reportCallInDefaultInitializer]
        scale_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}),  # pyright: ignore[reportCallInDefaultInitializer]
    ):
        """Initialize the BatchNorm layer.

        Args:
            num_features: Size of the feature dimension to normalize.
            use_running_average: If True, use stored running statistics for
                normalization instead of computing from the input batch.
                Typically False during training and True during evaluation.
            axis: Feature axis of the input to normalize over. Defaults to -1
                (last axis).
            momentum: Decay factor for the exponential moving average of running
                statistics. A value of 0.99 means 99% of the old statistic is
                retained per update. Defaults to 0.99.
            epsilon: Small constant added to variance for numerical stability.
                Defaults to 1e-5.
            dtype: Dtype for computation. If None, uses the input dtype.
            param_dtype: Dtype for scale and bias parameters.
                Defaults to jnp.float32.
            use_bias: Whether to add a learnable bias parameter.
                Defaults to True.
            use_scale: Whether to add a learnable scale parameter.
                Defaults to True.
            bias_init: Initializer for bias parameter. Defaults to zeros.
            scale_init: Initializer for scale parameter. Defaults to ones.
            axis_name: Name of the axis used for ``jax.lax.pmean`` for
                cross-replica statistics aggregation. If None, no aggregation
                is performed.
            axis_index_groups: Groups of axis indices for partitioned
                cross-replica statistics.
            use_fast_variance: If True, uses a numerically faster but
                potentially less stable variance computation. Defaults to True.
            promote_dtype: Function used for dtype promotion of inputs and
                parameters.
            rngs: Flax NNX random number generators for parameter initialization.
            bias_metadata: Additional metadata for the bias parameter.
            scale_metadata: Additional metadata for the scale parameter.
        """
        feature_shape = (num_features,)
        self.mean = nn.BatchStat(jnp.zeros(feature_shape, jnp.float32))
        self.var = nn.BatchStat(jnp.ones(feature_shape, jnp.float32))

        self.scale: nn.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nn.Param(scale_init(key, feature_shape, param_dtype), **scale_metadata)
        else:
            self.scale = None

        self.bias: nn.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nn.Param(bias_init(key, feature_shape, param_dtype), **bias_metadata)
        else:
            self.bias = None

        self.num_features = num_features
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance
        self.promote_dtype = promote_dtype

    def __call__(
        self,
        x,
        use_running_average: bool | None = None,
        *,
        mask: jax.Array | None = None,
    ):
        """Apply batch normalization to the input.

        Normalizes the input by subtracting the mean and dividing by the
        standard deviation computed over the batch (and spatial) dimensions.
        During training, batch statistics are computed and running averages
        are updated. During inference, stored running statistics are used.

        Args:
            x: Input array to normalize.
            use_running_average: If provided, overrides the instance-level
                ``use_running_average`` setting. Set to True for inference
                and False for training.
            mask: Optional boolean mask of the same shape as ``x``. If
                provided, masked positions are excluded from statistics
                computation.

        Returns:
            Normalized array with the same shape as the input.
        """
        use_running_average = nutil.first_from(
            use_running_average,
            self.use_running_average,
            error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
        )
        feature_axes = nutil._canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        # Promote dtypes for input and all Variables
        scale = self.scale[...] if self.scale is not None else None
        bias = self.bias[...] if self.bias is not None else None
        x, mean, var, scale, bias = self.promote_dtype((x, self.mean[...], self.var[...], scale, bias), dtype=self.dtype)
        if not use_running_average:
            mean, var = nutil._compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )
            # stop_gradient only for flax_array_ref
            if self.mean._can_update or self.var._can_update:
                stop_gradient = jax.lax.stop_gradient
            else:

                def stop_gradient(x):
                    return x

            self.mean[...] = stop_gradient(self.momentum * self.mean[...] + (1 - self.momentum) * mean)
            self.var[...] = stop_gradient(self.momentum * self.var[...] + (1 - self.momentum) * var)

        return nutil._normalize(
            x,
            mean,
            var,
            scale,
            bias,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.epsilon,
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return dynamic partition specs for this module's parameters."""
        specs: dict[str, object] = {"mean": Replicated, "var": Replicated}
        if self.scale is not None:
            specs["scale"] = Replicated
        if self.bias is not None:
            specs["bias"] = Replicated
        return specs

    def set_mode(
        self,
        use_running_average: bool | None = None,
        **kwargs,
    ) -> dict:
        """Class method used by ``nnx.set_mode``.

        Args:
          use_running_average: if True, the stored batch statistics will be
            used instead of computing the batch statistics on the input.
        """
        if use_running_average is not None:
            self.use_running_average = use_running_average
        return kwargs


class LayerNorm(nn.Module):
    """Layer Normalization layer.

    Normalizes activations across the feature dimension by subtracting the mean
    and dividing by the standard deviation, computed independently for each
    sample in the batch. Unlike BatchNorm, statistics are computed per-sample
    rather than across the batch, making LayerNorm independent of batch size
    and suitable for sequence models.

    The normalization formula is:
        output = ((x - mean) / sqrt(var + eps)) * scale + bias

    Attributes:
        num_features: Size of the feature dimension.
        epsilon: Small constant for numerical stability.
        dtype: Computation dtype. If None, uses the input dtype.
        param_dtype: Dtype for learnable parameters.
        use_bias: Whether a learnable bias parameter is included.
        use_scale: Whether a learnable scale parameter is included.
        reduction_axes: Axes over which mean and variance are computed.
        feature_axes: Axes corresponding to feature dimensions for
            scale/bias application.
        scale: Learnable scale parameter, or None if ``use_scale=False``.
        bias: Learnable bias parameter, or None if ``use_bias=False``.

    Example:
        >>> norm = LayerNorm(num_features=768, rngs=nn.Rngs(0))
        >>> x = jnp.ones((2, 512, 768))
        >>> y = norm(x)
        >>> assert y.shape == (2, 512, 768)
    """

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: nutil.Initializer = nn.initializers.zeros_init(),  # pyright: ignore[reportCallInDefaultInitializer]
        scale_init: nutil.Initializer = nn.initializers.ones_init(),  # pyright: ignore[reportCallInDefaultInitializer]
        reduction_axes: nutil.Axes = -1,
        feature_axes: nutil.Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: nutil.PromoteDtypeFn = nutil.dtypes.promote_dtype,
        rngs: nn.Rngs,
        bias_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}),  # pyright: ignore[reportCallInDefaultInitializer]
        scale_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}),  # pyright: ignore[reportCallInDefaultInitializer]
    ):
        """Initialize the LayerNorm layer.

        Args:
            num_features: Size of the feature dimension. Scale and bias
                parameters are created with this shape.
            epsilon: Small constant added to variance for numerical stability.
                Defaults to 1e-6.
            dtype: Dtype for computation. If None, uses the input dtype.
            param_dtype: Dtype for scale and bias parameters.
                Defaults to jnp.float32.
            use_bias: Whether to add a learnable bias parameter.
                Defaults to True.
            use_scale: Whether to add a learnable scale parameter.
                Defaults to True.
            bias_init: Initializer for bias parameter. Defaults to zeros.
            scale_init: Initializer for scale parameter. Defaults to ones.
            reduction_axes: Axes over which mean and variance are computed.
                Defaults to -1 (last axis).
            feature_axes: Axes that represent features for scale/bias
                broadcasting. Defaults to -1.
            axis_name: Name of the axis used for ``jax.lax.pmean`` for
                cross-replica statistics aggregation. If None, no aggregation
                is performed.
            axis_index_groups: Groups of axis indices for partitioned
                cross-replica statistics.
            use_fast_variance: If True, uses a numerically faster but
                potentially less stable variance computation. Defaults to True.
            promote_dtype: Function used for dtype promotion of inputs and
                parameters.
            rngs: Flax NNX random number generators for parameter initialization.
            bias_metadata: Additional metadata for the bias parameter.
            scale_metadata: Additional metadata for the scale parameter.
        """
        feature_shape = (num_features,)

        self.scale: nn.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nn.Param(scale_init(key, feature_shape, param_dtype), **scale_metadata)
        else:
            self.scale = None

        self.bias: nn.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nn.Param(bias_init(key, feature_shape, param_dtype), **bias_metadata)
        else:
            self.bias = None

        self.num_features = num_features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance
        self.promote_dtype = promote_dtype

    def __call__(self, x, *, mask: jax.Array | None = None):
        """Apply layer normalization to the input.

        Computes per-sample mean and variance over the reduction axes,
        normalizes the input, and applies learned scale and bias.

        Args:
            x: Input array to normalize. The last dimension (by default)
                must match ``num_features``.
            mask: Optional boolean mask of the same shape as ``x``. If
                provided, masked positions are excluded from statistics
                computation.

        Returns:
            Normalized array with the same shape as the input.
        """
        scale = self.scale[...] if self.scale is not None else None
        bias = self.bias[...] if self.bias is not None else None
        x, scale, bias = self.promote_dtype((x, scale, bias), dtype=self.dtype)
        mean, var = nutil._compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return nutil._normalize(
            x,
            mean,
            var,
            scale,
            bias,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.epsilon,
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return dynamic partition specs for this module's parameters."""
        specs: dict[str, object] = {}
        if self.scale is not None:
            specs["scale"] = Replicated
        if self.bias is not None:
            specs["bias"] = Replicated
        return specs
