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

"""Per-expert linear layers for Mixture-of-Experts.

This module provides `ParallelMoELinear` with row/column specializations. It is
designed to consume tokens grouped by expert (ragged layout) and multiply them
with per-expert weight shards using grouped matmul kernels.
"""

from __future__ import annotations

import typing

import jax
from eformer import common_types
from eformer.escale import PartitionManager
from ejkernel.modules import grouped_matmul
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

if typing.TYPE_CHECKING:
    pass

BATCH = common_types.BATCH
EMPTY = common_types.EMPTY
EMBED = common_types.EMBED
EXPERT = common_types.EXPERT
MODE_TRAIN = common_types.MODE_TRAIN
EP = common_types.EP
DP = common_types.DP
FSDP = common_types.FSDP
TP = common_types.TP
SP = common_types.SP

ExpertColumnWiseAlt = common_types.ExpertColumnWiseAlt
ExpertRowWiseAlt = common_types.ExpertRowWiseAlt
DynamicShardingAxes = common_types.DynamicShardingAxes


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros
Initializer = nn.initializers.Initializer


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism sharding (experts distributed over TP axis)."""

    axes: typing.ClassVar = [TP, EMPTY, EMPTY]
    mode: typing.ClassVar = MODE_TRAIN


class ParallelMoELinear(nn.Module):
    """A batched linear transformation layer for Mixture of Experts (MoE) models.

    This layer applies separate linear transformations for each expert in a MoE setup.
    The inputs are assumed to be sorted and grouped by expert, with `group_sizes`
    specifying how many tokens belong to each expert. It supports:
    - **Ragged Matrix Multiplication** via `jax.lax.ragged_dot_general`.
    - **Grouped Matrix Multiplication (GMM)** via a Pallas kernel for TPUs.

    Can optionally integrate with a `PartitionManager` to shard parameters and
    use `shard_map` for distributed execution.

    Attributes:
        num_experts: Number of experts.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        use_pallas_group_matmul: Whether to use the optimized GMM kernel (TPU-optimized).
        out_first: If True, kernel shape is `(num_experts, out_features, in_features)`;
            otherwise `(num_experts, in_features, out_features)`.
        dtype: Data type for computation. None means inherits from inputs.
        param_dtype: Data type for parameters (weights, biases).
        kernel_init: Initializer function for the kernel weights.
        bias_init: Initializer function for the bias.
        kernel: Weight matrix parameter for the transformation.
            Shape: (num_experts, out_features, in_features) if out_first else
            (num_experts, in_features, out_features).
        bias: Optional bias parameter. Shape: (num_experts, out_features) if out_first
            else (num_experts, in_features). None if use_bias=False.
        partition_manager: Handles sharding of parameters for distributed execution.
        _direction: Sharding direction for ALT sharding ("row", "column", or None).
    """

    _direction: typing.Literal["row", "column"] | None = None

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        out_first: bool = False,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        use_pallas_group_matmul: bool = False,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        partition_manager: PartitionManager | None = None,
        direction: typing.Literal["row", "column"] | None = None,
        use_expert_tensor_mode: bool = False,
        rngs: nn.Rngs,
    ):
        """Initializes a `ParallelMoELinear` layer.

        Args:
            num_experts: Number of experts in the layer.
            in_features: Size of the input feature dimension.
            out_features: Size of the output feature dimension.
            use_bias: Whether to include a bias term. Defaults to True.
            out_first: If True, kernel shape is `(num_experts, out_features, in_features)`,
                otherwise `(num_experts, in_features, out_features)`.
            kernel_init: Initializer for the kernel weights.
            bias_init: Initializer for the bias.
            use_pallas_group_matmul: Whether to use the TPU-optimized grouped matrix multiplication kernel.
            dtype: Data type for computation. Defaults to None (inherits from inputs).
            param_dtype: Data type for parameters (weights, biases).
            partition_manager: Partition manager for parameter sharding and mapping.
            direction: ALT-sharding direction, either `"row"`, `"column"`, or None.
            rngs: Random number generators for parameter initialization.
        """
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.use_pallas_group_matmul = use_pallas_group_matmul and (jax.default_backend() == "tpu")
        self.out_first = out_first
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.partition_manager = partition_manager
        self.use_expert_tensor_mode = use_expert_tensor_mode

        self.kernel_init = kernel_init
        self.bias_init = bias_init

        if direction is not None:
            assert direction in ["row", "column"]
            self._direction = direction
        kshape = (num_experts, out_features, in_features) if out_first else (num_experts, in_features, out_features)
        self.kernel = nn.Param(kernel_init(rngs.param(), kshape, param_dtype))
        if use_bias:
            bshape = (num_experts, out_features)
            self.bias = nn.Param(bias_init(rngs.param(), bshape, self.param_dtype))
        else:
            self.bias = None

    @property
    def direction(self) -> typing.Literal["row", "column"] | None:
        """Returns the parallelism direction for this layer.

        Returns:
            "row" for row-wise parallelism (input dimension partitioned),
            "column" for column-wise parallelism (output dimension partitioned),
            or None if no parallelism direction is set.
        """
        return self._direction

    @property
    def can_use_shard_map(self) -> bool:
        """Checks if this layer can use shard_map for distributed execution.

        Returns:
            True if both a partition manager and parallelism direction are configured,
            indicating the layer is ready for distributed execution with shard_map.
        """
        return self.partition_manager is not None and self._direction is not None

    @property
    def alt_sharding(self) -> ExpertRowWiseAlt | ExpertColumnWiseAlt | None:
        """Returns the ALT (Alternative) sharding configuration for this layer.

        ALT sharding provides pre-defined sharding patterns for common parallelism
        strategies, simplifying the configuration of distributed execution.

        Returns:
            ExpertRowWiseAlt for row parallelism,
            ExpertColumnWiseAlt for column parallelism,
            or None if no direction is set.

        Raises:
            NotImplementedError: If an unsupported direction is configured.
        """
        if self.direction is None:
            return None
        if self.use_expert_tensor_mode:
            return ExpertTensorParallel
        elif self.direction == "row":
            return ExpertRowWiseAlt
        elif self.direction == "column":
            return ExpertColumnWiseAlt
        else:
            direction = self.direction
            raise NotImplementedError(f"ALT-Sharding Rule for {direction=} is not implemented!.")

    @property
    def alt_sharding_axis(self) -> list[str] | None:
        """Returns the axis names for ALT sharding configuration.

        Returns:
            List of axis names (e.g., ["expert", "tp", "dp"]) for the configured
            ALT sharding pattern, or None if no ALT sharding is configured.
        """
        if self.alt_sharding is None:
            return None
        return self.alt_sharding.axes

    @property
    def expert_axis(self) -> str:
        """Semantic axis name representing the expert dimension."""
        return TP if self.use_expert_tensor_mode else EP

    def _group_axes(self) -> list[str | None]:
        """Sharding axes for group sizes."""
        return [EMPTY]

    def _input_axes(self) -> list[str | None]:
        """Sharding axes for inputs based on parallelism direction."""
        if self.direction == "row":
            return [DP, TP]
        if self.direction == "column":
            return [DP, EMPTY]
        return [DP, EMPTY]

    def _output_axes(self) -> list[str | None]:
        """Sharding axes for outputs based on parallelism direction."""
        if self.direction == "row":
            return [DP, EMPTY]
        if self.direction == "column":
            if self.use_expert_tensor_mode:
                return [DP, TP]
            return [DP, [EP, TP]]
        return [DP, EMPTY]

    def __call__(
        self,
        inputs: Float[Array, "tokens_ragged hidden_dim"],
        group_sizes: Int[Array, "num_groups"],  # noqa
        sorted_experts: Int[Array, "tokens_ragged"] | None = None,  # noqa
    ) -> Float[Array, "tokens_ragged out_dim"]:
        """Applies the batched linear transformation.

        Args:
            inputs: The input array, which is a batch of tokens sorted and grouped
                by expert. Shape: `(total_tokens, in_features)`.
            group_sizes: An array indicating the number of tokens assigned to each
                expert. Shape: `(num_experts,)`.
            sorted_experts: Optional expert ids aligned with `inputs`. Required when
                `use_expert_tensor_mode` so tokens can be localized per shard.

        Returns:
            The output array after the linear transformation.
            Shape: `(total_tokens, out_features)`.
        """
        weight = self.kernel.value

        if weight.dtype in (
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        ):
            weight = weight.astype("f4")

        inputs, weight = promote_dtype((inputs, weight), dtype=self.dtype)
        output = grouped_matmul(
            inputs,
            weight,
            group_sizes,
            preferred_element_type=jnp.bfloat16,
            transpose_rhs=self.out_first,
            platform="xla",
        )
        if self.bias is not None:
            output += self._expand_bias_ragged(group_sizes, sorted_experts=sorted_experts)

        return output

    def _expand_bias_ragged(
        self,
        group_sizes: Int[Array, "num_groups"],  # noqa
        sorted_experts: Int[Array, "tokens_ragged"] | None = None,  # noqa
    ) -> Float[Array, "tokens_ragged out_dim"]:
        """Expands the bias to match the ragged batch structure.

        This method repeats the bias for each expert according to the number of
        tokens assigned to it. This is necessary because tokens are grouped by
        expert, and each group needs its corresponding expert's bias.

        Args:
            group_sizes: The sizes of token groups for each expert.
                Shape: (num_experts,). Each element indicates how many tokens
                are assigned to that expert.

        Returns:
            The expanded bias array where each expert's bias is repeated
            according to its group size. Shape: (total_tokens, out_features).

        Example:
            If expert 0 has 3 tokens, expert 1 has 2 tokens, and expert 2 has 4 tokens,
            this will repeat bias[0] 3 times, bias[1] 2 times, and bias[2] 4 times.
        """
        if sorted_experts is not None:
            indices = sorted_experts
        else:
            bias_rows = self.bias.value.shape[0]
            indices = jnp.repeat(jnp.arange(bias_rows), group_sizes)
        return self.bias.value[indices]


class RowParallelMoELinear(ParallelMoELinear):
    """Row-parallel variant of ParallelMoELinear.

    This class specializes ParallelMoELinear for row-wise parallelism, where the
    input dimension is partitioned across devices. In row parallelism, each device
    holds a subset of input features and computes partial results that are then
    reduced across devices.

    The weight matrix is partitioned along the input dimension (rows), and an
    all-reduce operation is performed after the matrix multiplication to combine
    partial results.

    Attributes:
        _direction: Fixed to "row" to indicate row-wise parallelism.

    Example:
        >>> # Create a row-parallel MoE linear layer
        >>> layer = RowParallelMoELinear(
        ...     num_experts=8,
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=rngs
        ... )
    """

    _direction: typing.Literal["row", "column"] | None = "row"


class ColumnParallelMoELinear(ParallelMoELinear):
    """Column-parallel variant of ParallelMoELinear.

    This class specializes ParallelMoELinear for column-wise parallelism, where the
    output dimension is partitioned across devices. In column parallelism, each device
    computes a subset of output features independently without requiring reduction.

    The weight matrix is partitioned along the output dimension (columns), and each
    device produces its portion of the output directly without communication.

    Attributes:
        _direction: Fixed to "column" to indicate column-wise parallelism.

    Example:
        >>> # Create a column-parallel MoE linear layer
        >>> layer = ColumnParallelMoELinear(
        ...     num_experts=8,
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=rngs
        ... )
    """

    _direction: typing.Literal["row", "column"] | None = "column"
