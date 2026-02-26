# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2024 The Improved Version Contributors.
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

"""Linear layers with parallel and distributed computation support.

Provides optimized linear layers with support for model parallelism,
tensor parallelism, and various sharding strategies for distributed training.

Classes:
    ParallelLinear: Linear layer with tensor/model parallelism support.
    RowParallelLinear: Row-parallel variant (input dimension partitioned).
    ColumnParallelLinear: Column-parallel variant (output dimension partitioned).

Key Features:
    - Automatic sharding and gathering for distributed training
    - Support for various matrix multiplication methods
    - Mixed precision support
    - Efficient initialization strategies with configurable scaling
    - Integration with JAX's shard_map
    - Conversion to quantized variants

Example:
    >>> from easydel.layers.linears import ParallelLinear
    >>> from flax import nnx as nn
    >>>
    >>> # Create a parallel linear layer
    >>> layer = ParallelLinear(
    ...     in_features=768,
    ...     out_features=3072,
    ...     use_bias=True,
    ...     dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0)
    ... )
    >>> output = layer(input_tensor)
"""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
import jax.numpy as jnp
from eformer.common_types import ColumnWise, Replicated, RowWise
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax
from jaxtyping import Array, Shaped

from easydel.layers._sharding import resolve_safe_sharding
from easydel.layers.quantization._configs import QuantizationConfig

if tp.TYPE_CHECKING:
    from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

Dtype = jnp.dtype
Initializer = nn.initializers.Initializer
PrecisionLike = lax.PrecisionLike
Shape = collections.abc.Sequence[int]
AxisNames = str | collections.abc.Sequence[str] | tuple[str, ...]

# Default initializers
default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


class ParallelLinear(nn.Module):
    """A linear transformation layer with optional parallelism support.

    Behaves like `nnx.Linear` but supports distributed computation and parameters
    across multiple devices. This layer implements the transformation:

        y = x @ W + b

    where W has shape (in_features, out_features) and b has shape (out_features,).

    The layer supports optional scaling of the output, which can be specified as
    a fixed value or computed from the fan-in or fan-out dimensions.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to include a bias term. Default is True.
        dtype: The dtype of the computation (defaults to inferred from input).
        param_dtype: The dtype of the parameters. Default is float32.
        precision: JAX precision for the dot product. Default is None.
        kernel_init: Initializer for the kernel weights.
        bias_init: Initializer for the bias.
        kernel: Weight matrix parameter of shape (in_features, out_features).
        bias: Optional bias parameter of shape (out_features,).
        tp_merged: Number of tensor parallel merged outputs.
        distributed_matmul: Optional distributed matrix multiplication function.
        _direction: Parallelism direction ("row", "column", or None).

    Example:
        >>> from easydel.layers.linears import ParallelLinear
        >>> import jax.numpy as jnp
        >>> from flax import nnx as nn
        >>>
        >>> # Create a basic linear layer
        >>> layer = ParallelLinear(
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=nn.Rngs(0)
        ... )
        >>>
        >>> # Forward pass
        >>> x = jnp.ones((32, 768))
        >>> y = layer(x)
        >>> # y.shape = (32, 3072)
        >>>
        >>> # With fan-in scaling (useful for certain architectures)
        >>> layer_scaled = ParallelLinear(
        ...     in_features=768,
        ...     out_features=3072,
        ...     scale="fan_in",
        ...     rngs=nn.Rngs(0)
        ... )
    """

    _direction: tp.Literal["row", "column"] | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float | tp.Literal["fan_in", "fan_out"] = 1.0,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nn.Rngs | None = None,
    ):
        """Initialize a parallel linear layer.

        Creates a linear transformation layer with configurable parameters
        and optional output scaling.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample. Can also be a sequence
                of integers for tensor-parallel merged outputs.
            scale: Output scaling factor. Can be:
                - A float value for direct scaling
                - "fan_in" for 1/sqrt(in_features) scaling
                - "fan_out" for 1/sqrt(out_features) scaling
                Defaults to 1.0 (no scaling).
            use_bias: If True, adds a learnable bias to the output.
                Defaults to True.
            dtype: Data type for computation. If None, uses input dtype.
                Defaults to None.
            param_dtype: Data type for storing parameters. Defaults to float32.
            precision: JAX precision for matrix multiplication. Can be None,
                'default', 'high', 'highest', or specific precision tuples.
                Defaults to None.
            kernel_init: Initializer function for the weight matrix.
                Defaults to lecun_normal().
            bias_init: Initializer function for the bias vector.
                Defaults to zeros.
            rngs: Random number generators for initialization. If None,
                creates a default Rngs with seed 0.
        """
        rngs_computed: nn.Rngs
        if rngs is None:
            rngs_computed = nn.Rngs(0)
        else:
            rngs_computed = rngs

        scale_computed: float
        scale_is_fan_in: bool = scale == "fan_in"
        scale_is_fan_out: bool = scale == "fan_out"
        if scale_is_fan_in:
            scale_computed = in_features**-0.5
        elif scale_is_fan_out:
            scale_computed = out_features**-0.5
        else:
            scale_computed = scale

        scale_is_one: bool = scale_computed != 1.0
        if scale_is_one:

            def _scale_operator(x: Array) -> Array:
                scaled: Array = x * scale_computed
                return scaled

        else:

            def _scale_operator(x: Array) -> Array:
                return x

        self._scale_operator: tp.Callable[[Array], Array] = _scale_operator
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.use_bias: bool = use_bias
        self.dtype: Dtype | None = dtype
        self.param_dtype: Dtype = param_dtype
        self.precision: PrecisionLike = precision
        self.kernel_init: Initializer = kernel_init
        self.bias_init: Initializer = bias_init
        self.rngs: nn.Rngs = rngs_computed

        out_features_is_sequence: bool = isinstance(out_features, collections.abc.Sequence)
        tp_merged: int
        if out_features_is_sequence:
            tp_merged = len(out_features)
        else:
            tp_merged = 1
        self.tp_merged: int = tp_merged

        tp_merged_gt_one: bool = self.tp_merged > 1
        out_features_sum: int
        if tp_merged_gt_one:
            out_features_sum = sum(out_features)
        else:
            out_features_sum = out_features

        kernel_key: tp.Any = rngs_computed.params()
        kernel_shape: tuple[int, int] = (in_features, out_features_sum)
        kernel_initialized: Array = kernel_init(kernel_key, kernel_shape, param_dtype)
        self.kernel: nn.Param = nn.Param(kernel_initialized)

        if use_bias:
            bias_key: tp.Any = rngs_computed.params()
            bias_shape: tuple[int] = (out_features,)
            bias_initialized: Array = bias_init(bias_key, bias_shape, param_dtype)
            self.bias: nn.Param | None = nn.Param(bias_initialized)
        else:
            self.bias = None
        self.distributed_matmul: tp.Any | None = None

    def native_forward(
        self,
        inputs: Shaped[Array, "... in_features"],
        w: Array | None = None,
    ) -> Shaped[Array, "... out_features"]:
        """Apply the linear transformation using native JAX operations.

        Performs the matrix multiplication y = x @ W + b with proper dtype
        promotion and optional scaling.

        Args:
            inputs: The input array of shape (..., in_features). The batch
                dimensions can be arbitrary.
            w: Optional weight matrix to use instead of self.kernel. This is
                useful for weight sharing or external weight injection.
                Defaults to None (uses self.kernel).

        Returns:
            The transformed output array of shape (..., out_features).
            If scale is configured, the output is scaled accordingly.
        """
        w_is_none: bool = w is None
        kernel: Array
        if w_is_none:
            kernel = self.kernel.value
        else:
            kernel = w

        has_bias: bool = self.use_bias
        bias: Array | None
        if has_bias:
            bias = self.bias.value
        else:
            bias = None

        bias_is_not_none: bool = bias is not None
        inputs_promoted: Array
        kernel_promoted: Array
        bias_promoted: Array | None
        if bias_is_not_none:
            inputs_promoted, kernel_promoted, bias_promoted = promote_dtype((inputs, kernel, bias), dtype=self.dtype)
        else:
            inputs_promoted, kernel_promoted = promote_dtype((inputs, kernel), dtype=self.dtype)
            bias_promoted = None

        inputs_ndim: int = inputs_promoted.ndim
        inputs_gt_one_dim: bool = inputs_ndim > 1
        subscript: str
        if inputs_gt_one_dim:
            subscript = "...ik,...kj->...ij"
        else:
            subscript = "...k,...kj->...j"

        y: Shaped[Array, "... out_features"] = jnp.einsum(
            subscript,
            inputs_promoted,
            kernel_promoted,
            precision=self.precision,
            optimize=True,
        )

        y_scaled: Shaped[Array, "... out_features"] = self._scale_operator(y)

        y_final: Shaped[Array, "... out_features"]
        if bias_promoted is not None:
            y_ndim: int = y_scaled.ndim
            num_ones: int = y_ndim - 1
            ones_tuple: tuple[int, ...] = (1,) * num_ones
            final_dim: tuple[int] = (-1,)
            reshape_spec: tuple[int, ...] = ones_tuple + final_dim
            bias_reshaped: Array = jnp.reshape(bias_promoted, reshape_spec)
            y_final = y_scaled + bias_reshaped
        else:
            y_final = y_scaled

        return y_final

    def __call__(
        self,
        inputs: Shaped[Array, "... in_features"],
        w: Array | None = None,
    ) -> Shaped[Array, "... out_features"]:
        """Apply the linear transformation to inputs.

        This is the main entry point for the layer. It delegates to
        native_forward for the actual computation.

        Args:
            inputs: The input array of shape (..., in_features).
            w: Optional weight matrix override. Defaults to None.

        Returns:
            The transformed output array of shape (..., out_features).

        Example:
            >>> layer = ParallelLinear(768, 3072, rngs=nn.Rngs(0))
            >>> x = jnp.ones((32, 768))
            >>> y = layer(x)  # Shape: (32, 3072)
        """
        return self.native_forward(inputs=inputs, w=w)

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, tp.Any]:
        """Return dynamic partition specs for this module's parameters.

        Uses the configured parallelism direction to pick a sharding pattern
        for the kernel. Bias is replicated when present.
        """
        if self._direction is None:
            return {}
        if self._direction == "row":
            kernel_spec = RowWise
        elif self._direction == "column":
            kernel_spec = ColumnWise
        else:
            return {}
        mesh = _kwargs.get("mesh")
        specs: dict[str, tp.Any] = {
            "kernel": resolve_safe_sharding(
                axes=kernel_spec,
                shape=tuple(self.kernel.value.shape),
                partition_manager=partition_manager,
                mesh=mesh,
            )
        }
        if self.use_bias:
            specs["bias"] = resolve_safe_sharding(
                axes=Replicated,
                shape=tuple(self.bias.value.shape),
                partition_manager=partition_manager,
                mesh=mesh,
            )
        return specs

    def to_quantized(
        self,
        config: QuantizationConfig,
        **kwargs,
    ) -> ColumnParallelLinearQuantized | RowParallelLinearQuantized:
        """Convert this layer to a quantized version.

        Creates a quantized linear layer with the same configuration but
        weights stored in a compressed format according to the provided
        quantization configuration.

        Args:
            config: Quantization configuration specifying the quantization
                type (INT8, NF4, etc.) and related parameters.
            **kwargs: Optional runtime quantized-matmul controls forwarded
                to the quantized linear module (for example qmm platform/path
                overrides and tuned-config toggles).

        Returns:
            A RowParallelLinearQuantized or ColumnParallelLinearQuantized
            instance, depending on self._direction.

        Raises:
            ValueError: If _direction is not "row" or "column".

        Example:
            >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
            >>> layer = ColumnParallelLinear(768, 3072, rngs=nn.Rngs(0))
            >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
            >>> quantized_layer = layer.to_quantized(config)
        """
        firend = self._quantized_firend
        lazy_module = jax.eval_shape(
            lambda rngs: firend(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                config=config,
                **kwargs,
                rngs=rngs,
            ),
            self.rngs,
        )

        if isinstance(self.kernel.value, jax.ShapeDtypeStruct):
            return lazy_module

        return lazy_module.restage(kernel=self.kernel, bias=self.bias)

    @property
    def _quantized_firend(self) -> type[RowParallelLinearQuantized] | type[ColumnParallelLinearQuantized]:
        """Get the corresponding quantized layer class.

        Returns:
            The quantized layer class matching this layer's parallelism
            direction (RowParallelLinearQuantized or ColumnParallelLinearQuantized).

        Raises:
            ValueError: If _direction is not "row" or "column".
        """
        from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

        if self._direction == "row":
            return RowParallelLinearQuantized
        elif self._direction == "column":
            return ColumnParallelLinearQuantized
        else:
            raise ValueError("uknown direction, with no firend!")


class RowParallelLinear(ParallelLinear):
    """Row-parallel variant of ParallelLinear.

    This class specializes ParallelLinear for row-wise parallelism, where the
    input dimension is partitioned across devices. In row parallelism, each device
    holds a subset of input features and computes partial results that are then
    reduced (summed) across devices.

    Row parallelism is typically used for the second linear layer in a two-layer
    MLP pattern, where the first layer is column-parallel:

        x -> [Column-Parallel] -> activation -> [Row-Parallel] -> y
                                                     |
                                                 all-reduce

    Attributes:
        _direction: Fixed to "row" to indicate row-wise parallelism.

    Example:
        >>> layer = RowParallelLinear(
        ...     in_features=3072,
        ...     out_features=768,
        ...     rngs=nn.Rngs(0)
        ... )
    """

    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinear(ParallelLinear):
    """Column-parallel variant of ParallelLinear.

    This class specializes ParallelLinear for column-wise parallelism, where the
    output dimension is partitioned across devices. In column parallelism, each
    device computes a different slice of the output features independently
    without requiring any communication.

    Column parallelism is typically used for the first linear layer in a two-layer
    MLP pattern:

        x -> [Column-Parallel] -> activation -> [Row-Parallel] -> y
                  |
              no comm needed (outputs are sharded)

    Attributes:
        _direction: Fixed to "column" to indicate column-wise parallelism.

    Example:
        >>> layer = ColumnParallelLinear(
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=nn.Rngs(0)
        ... )
    """

    _direction: tp.Literal["column"] = "column"
