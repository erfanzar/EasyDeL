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

Functions:
    get_sharding: Extract sharding specification from an array
    get_output_partition_spec: Calculate output sharding for matmul
    get_matmul_output_sharding: Determine output sharding from input specs

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

import jax.numpy as jnp
from eformer import escale as es
from eformer.pytree import auto_pytree
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Shaped

Dtype = jnp.dtype
Initializer = nn.initializers.Initializer
PrecisionLike = lax.PrecisionLike
Shape = tp.Sequence[int]
AxisNames = str | tp.Sequence[str] | tuple[str, ...]

# Default initializers
default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


def get_sharding(arr: Shaped[Array, "..."]) -> Ps | None:
    """Get the sharding specification of an array.

    Extracts the PartitionSpec from a sharded JAX array.

    Args:
        arr: Array to get sharding from.

    Returns:
        PartitionSpec of the array, or None if not sharded.
    """
    sharding: tp.Any | None = getattr(arr, "sharding", None)
    has_sharding: bool = sharding is not None
    result: Ps | None
    if has_sharding:
        spec: Ps = sharding.spec
        result = spec
    else:
        result = None
    return result


def get_output_partition_spec(
    lhs: Shaped[Array, "..."],
    rhs: Shaped[Array, "..."],
    method: "MatrixMultiplyMethod",  # noqa #type:ignore
    axis_name: str,
) -> Ps | None:
    """Calculate output partition spec for matrix multiplication.

    Determines the appropriate output sharding based on input
    sharding and the matrix multiplication method used.

    Args:
        lhs: Left-hand side array (inputs).
        rhs: Right-hand side array (weights).
        method: Matrix multiplication method.
        axis_name: Axis name for sharding.

    Returns:
        Output partition specification for the result.
    """
    from jax.sharding import PartitionSpec as P

    lhs_spec: Ps | None = get_sharding(lhs)
    rhs_spec: Ps | None = get_sharding(rhs)

    lhs_is_none: bool = lhs_spec is None
    rhs_is_none: bool = rhs_spec is None
    either_none: bool = lhs_is_none or rhs_is_none
    if either_none:
        return None

    lhs_ndim: int = lhs.ndim
    is_2d: bool = lhs_ndim == 2
    result: Ps
    if is_2d:
        rhs_spec_1: str | None = rhs_spec[1]
        rhs_spec_0: str | None = rhs_spec[0]
        result = P(rhs_spec_1, rhs_spec_0)
    else:
        num_none: int = lhs_ndim - 1
        none_tuple: tuple[None, ...] = (None,) * num_none
        axis_tuple: tuple[str] = (axis_name,)
        combined_tuple: tuple[None | str, ...] = none_tuple + axis_tuple
        result = P(*combined_tuple)
    return result


def get_matmul_output_sharding(lhs_pspec: Ps | None, rhs_pspec: Ps | None) -> Ps:
    """Determine output sharding for matrix multiplication.

    Calculates the output PartitionSpec based on input partition specs,
    following matrix multiplication rules where contracting dimensions
    are reduced and non-contracting dimensions determine output sharding.

    For X @ W:
    - Contracting dimensions are reduced during matmul
    - Non-contracting dimensions determine output sharding
    - Ensures no duplicate sharding dimensions in output

    Args:
        lhs_pspec: PartitionSpec for left-hand side matrix.
        rhs_pspec: PartitionSpec for right-hand side matrix.

    Returns:
        Output PartitionSpec for the multiplication result.
    - Ensures correct output dimensionality with None padding if needed

    Args:
        lhs_pspec: PartitionSpec for the left-hand side matrix X
        rhs_pspec: PartitionSpec for the right-hand side matrix W

    Returns:
        PartitionSpec for the output of X @ W
    """
    lhs_is_none: bool = lhs_pspec is None
    rhs_is_none: bool = rhs_pspec is None
    either_none: bool = lhs_is_none or rhs_is_none
    if either_none:
        empty_spec: Ps = Ps()
        return empty_spec

    lhs_length: int = len(lhs_pspec)
    lhs_gt_one: bool = lhs_length > 1
    lhs_output_dims: tuple
    if lhs_gt_one:
        lhs_output_dims = lhs_pspec[:-1]
    else:
        lhs_output_dims = ()

    rhs_length: int = len(rhs_pspec)
    rhs_ge_two: bool = rhs_length >= 2
    rhs_output_dims: tuple
    if rhs_ge_two:
        rhs_last: str | None = rhs_pspec[-1]
        rhs_output_dims = (rhs_last,)
    else:
        rhs_is_empty: bool = not rhs_pspec
        if rhs_is_empty:
            rhs_output_dims = ()
        else:
            rhs_last_item: str | None = rhs_pspec[-1]
            rhs_output_dims = (rhs_last_item,)

    all_shard_dims: set[str] = set()
    output_dims: list[str | None | tuple] = []

    # Process LHS dimensions
    for dim in lhs_output_dims:
        dim_is_tuple: bool = isinstance(dim, tuple)
        if dim_is_tuple:
            filtered_tuple: tuple = tuple(d for d in dim if d not in all_shard_dims)
            for d in dim:
                all_shard_dims.add(d)
            filtered_is_nonempty: bool = bool(filtered_tuple)
            dim_is_nonempty: bool = bool(dim)
            if filtered_is_nonempty:
                output_dims.append(filtered_tuple)
            elif dim_is_nonempty:
                output_dims.append(None)
        else:
            dim_not_in_set: bool = dim not in all_shard_dims
            if dim_not_in_set:
                output_dims.append(dim)
                all_shard_dims.add(dim)
            else:
                output_dims.append(None)

    # Process RHS dimensions
    for dim in rhs_output_dims:
        dim_is_tuple_rhs: bool = isinstance(dim, tuple)
        if dim_is_tuple_rhs:
            filtered_tuple_rhs: tuple = tuple(d for d in dim if d not in all_shard_dims)
            for d in dim:
                all_shard_dims.add(d)
            filtered_rhs_nonempty: bool = bool(filtered_tuple_rhs)
            dim_rhs_nonempty: bool = bool(dim)
            if filtered_rhs_nonempty:
                output_dims.append(filtered_tuple_rhs)
            elif dim_rhs_nonempty:
                output_dims.append(None)
        else:
            dim_rhs_not_in_set: bool = dim not in all_shard_dims
            if dim_rhs_not_in_set:
                output_dims.append(dim)
                all_shard_dims.add(dim)
            else:
                output_dims.append(None)

    # Pad with None to match expected dimensionality
    output_dims_length: int = len(output_dims)
    lhs_pspec_length: int = len(lhs_pspec)
    target_length: int = lhs_pspec_length - 1 + 1
    needs_padding: bool = output_dims_length < target_length
    while needs_padding:
        output_dims.append(None)
        output_dims_length = len(output_dims)
        needs_padding = output_dims_length < target_length

    result_spec: Ps = Ps(*output_dims)
    return result_spec


@auto_pytree
class TensorParallelConfig:
    """Configuration for Tensor Parallelism.

    Attributes:
        mesh: The JAX device mesh.
        axis_name: The name of the mesh axis to use for tensor parallelism.
        matmul_type: The type of matmul (MatrixMultiplyMethod).
        reduce_scatter_output: If True and parallel_type is COLUMN,
            use reduce-scatter instead of all-gather for the output.
            This keeps the output sharded (useful for sequence parallelism
            or subsequent RowParallel layers). Defaults to False.
    """

    mesh: Mesh = None
    axis_name: str = "tp"
    matmul_method: None = None
    reduce_output: bool = False
    reduce_scatter_output: bool = False

    def __post_init__(self):
        msg: str | None = None
        has_matmul_method: bool = self.matmul_method is not None
        if has_matmul_method:
            mesh_is_none: bool = self.mesh is None
            if mesh_is_none:
                self.mesh = es.get_incontext_mesh()
            axis_names: tuple[str, ...] = self.mesh.axis_names
            axis_not_in_mesh: bool = self.axis_name not in axis_names
            if axis_not_in_mesh:
                axis_name_str: str = self.axis_name
                axis_names_str: str = str(axis_names)
                msg = f"axis_name '{axis_name_str}' not found in mesh axis names: {axis_names_str}"
        has_error: bool = msg is not None
        if has_error:
            raise ValueError(msg)


class ParallelLinear(nn.Module):
    """A Linear layer with optional parallelism.

    Behaves like `nnx.Linear` but can distribute computation and parameters
    across devices based on the `TensorParallelConfig`.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to include a bias term. Default is True.
        dtype: The dtype of the computation (defaults to inferred from input).
        param_dtype: The dtype of the parameters. Default is float32.
        precision: JAX precision for the dot product. Default is None.
        kernel_init: Initializer for the kernel weights.
        bias_init: Initializer for the bias.
        parallel_config: Configuration for tensor parallelism. If None,
          the layer behaves like a standard non-parallel Linear layer.
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
        parallel_config: TensorParallelConfig | None = None,
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
        self.parallel_config: TensorParallelConfig | None = parallel_config
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

    #     if parallel_config is not None and parallel_config.matmul_method is not None:
    #         self.distributed_matmul = create_distributed_matmul(
    #             parallel_config.matmul_method,
    #             parallel_config.axis_name,
    #         )

    # def collective_forward(
    #     self,
    #     inputs: Shaped[Array, "... in_features"],
    #     w: Array | None = None,
    # ) -> Shaped[Array, "... out_features"]:
    #     kernel = self.kernel.value if w is None else w
    #     bias = self.bias.value if self.use_bias else None

    #     if bias is not None:
    #         inputs, kernel, bias = promote_dtype((inputs, kernel, bias), dtype=self.dtype)
    #     else:
    #         inputs, kernel = promote_dtype((inputs, kernel), dtype=self.dtype)

    #     # Ensure inputs are 2D
    #     orig_shape = inputs.shape
    #     inputs_2d = inputs.reshape(-1, inputs.shape[-1])

    #     # Get partition specs
    #     input_spec = get_sharding(inputs_2d)
    #     kernel_spec = get_sharding(kernel)
    #     output_spec = get_output_partition_spec(
    #         inputs_2d,
    #         kernel,
    #         self.parallel_config.matmul_method,
    #         self.parallel_config.axis_name,
    #     )

    #     if self.parallel_config.matmul_method == MatrixMultiplyMethod.REDUCE_SCATTER:
    #         kernel = prepare_matrix_for_reduce_scatter(
    #             kernel,
    #             self.parallel_config.mesh,
    #             self.parallel_config.axis_name,
    #         )
    #     elif self.parallel_config.matmul_method == MatrixMultiplyMethod.ALL_GATHER:
    #         kernel = prepare_matrix_for_all_gather(
    #             kernel,
    #             self.parallel_config.mesh,
    #             self.parallel_config.axis_name,
    #         )

    #     output_2d = shard_map(
    #         self.distributed_matmul,
    #         mesh=self.parallel_config.mesh,
    #         in_specs=(input_spec, kernel_spec),
    #         out_specs=output_spec,
    #         check_vma=False,
    #     )(inputs_2d, kernel)

    #     output = output_2d.reshape((*orig_shape[:-1], self.out_features))

    #     output = self._scale_operator(output)

    #     if bias is not None:
    #         output = output + jnp.reshape(bias, (1,) * (output.ndim - 1) + (-1,))

    #     return output

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
        # if self.distributed_matmul is None:
        return self.native_forward(inputs=inputs, w=w)
        # return self.collective_forward(inputs=inputs, w=w)


class RowParallelLinear(ParallelLinear):
    _direction: tp.Literal["row", "column"] | None = "row"


class ColumnParallelLinear(ParallelLinear):
    _direction: tp.Literal["row", "column"] | None = "column"
