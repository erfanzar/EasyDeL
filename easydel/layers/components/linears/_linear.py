# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
    ParallelLinear: Linear layer with tensor/model parallelism support
    Linear: Standard linear layer (alias for ParallelLinear)

Key Features:
    - Automatic sharding and gathering for distributed training
    - Support for various matrix multiplication methods
    - Mixed precision support
    - Efficient initialization strategies
    - Integration with JAX's shard_map

Example:
    >>> from easydel.layers import ParallelLinear
    >>> # Create a parallel linear layer
    >>> layer = ParallelLinear(
    ...     features=768,
    ...     use_bias=True,
    ...     gather_output=False,
    ...     axis_name="model",
    ...     dtype=jnp.bfloat16
    ... )
    >>> output = layer(input_tensor)
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax
from jaxtyping import Array, Shaped

from ..quants._quants import QuantizationConfig

if tp.TYPE_CHECKING:
    from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

Dtype = jnp.dtype
Initializer = nn.initializers.Initializer
PrecisionLike = lax.PrecisionLike
Shape = tp.Sequence[int]
AxisNames = str | tp.Sequence[str] | tuple[str, ...]

# Default initializers
default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


class ParallelLinear(nn.Module):
    """A Linear layer with optional parallelism.

    Behaves like `nnx.Linear` but can distribute computation and parameters across devices.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to include a bias term. Default is True.
        dtype: The dtype of the computation (defaults to inferred from input).
        param_dtype: The dtype of the parameters. Default is float32.
        precision: JAX precision for the dot product. Default is None.
        kernel_init: Initializer for the kernel weights.
        bias_init: Initializer for the bias.
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

        out_features_is_sequence: bool = isinstance(out_features, tp.Sequence)
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
        """Applies the linear transformation with optional tensor parallelism.

        Args:
            inputs: The input array. Shape: (..., in_features).
                    For ROW parallelism, the input is expected to be sharded
                    along the feature dimension (`axis_name`).

        Returns:
            The transformed output array.
            Shape: (..., out_features).
            Output is sharded for COLUMN parallelism if `reduce_scatter_output` is True.
            Otherwise, output is fully replicated.
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
        return self.native_forward(inputs=inputs, w=w)

    def to_quantized(self, config: QuantizationConfig) -> ColumnParallelLinearQuantized | RowParallelLinearQuantized:
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
                rngs=rngs,
            ),
            self.rngs,
        )
        return lazy_module.restage(kernel=self.kernel, bias=self.bias)

    @property
    def _quantized_firend(self) -> type[RowParallelLinearQuantized] | type[ColumnParallelLinearQuantized]:
        from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

        if self._direction == "row":
            return RowParallelLinearQuantized
        elif self._direction == "column":
            return ColumnParallelLinearQuantized
        else:
            raise ValueError("uknown direction, with no firend!")


class RowParallelLinear(ParallelLinear):
    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinear(ParallelLinear):
    _direction: tp.Literal["column"] = "column"
