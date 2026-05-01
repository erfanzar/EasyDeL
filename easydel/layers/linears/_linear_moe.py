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

"""Per-expert linear layers for Mixture-of-Experts.

This module provides `ParallelMoELinear` with row/column specializations. It is
designed to consume tokens grouped by expert (ragged layout) and multiply them
with per-expert weight shards using grouped matmul kernels.

**Key Concepts:**

    **Ragged/Grouped Layout:**
        Unlike standard batched matmul where all batches have the same size, MoE
        layers have variable-sized expert batches. The grouped matmul kernel handles
        this efficiently by processing each expert's tokens as a separate batch.

    **Row vs Column Parallelism:**
        - **Column Parallel**: Output features are partitioned (e.g., W_i, W_u in FFN)
          Each device computes a slice of output features, no reduction needed
        - **Row Parallel**: Input features are partitioned (e.g., W_d in FFN)
          Each device computes partial results that are summed across devices

    **Expert Tensor Mode:**
        An alternative sharding where experts are distributed across the TP axis
        instead of the EP axis. Useful for specific hardware configurations.

Example Workflow:
    >>> # Complete MoE FFN example with row/column parallelism
    >>> from easydel.layers import ColumnParallelMoELinear, RowParallelMoELinear
    >>> import spectrax as spx
from spectrax import nn
    >>>
    >>> # Column-parallel layers (W_i and W_u)
    >>> wi_layer = ColumnParallelMoELinear(8, 768, 3072, rngs=rngs)
    >>> wu_layer = ColumnParallelMoELinear(8, 768, 3072, rngs=rngs)
    >>>
    >>> # Row-parallel layer (W_d)
    >>> wd_layer = RowParallelMoELinear(8, 3072, 768, rngs=rngs)
    >>>
    >>> # Forward pass (assumes tokens are already sorted by expert)
    >>> wi_out = wi_layer(sorted_tokens, group_sizes, sorted_experts)
    >>> wu_out = wu_layer(sorted_tokens, group_sizes, sorted_experts)
    >>> intermediate = jax.nn.silu(wi_out) * wu_out
    >>> output = wd_layer(intermediate, group_sizes, sorted_experts)
"""

from __future__ import annotations

import typing

import jax
import spectrax as spx
from ejkernel.modules import GroupedMatmulConfig, grouped_matmul  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int
from spectrax import common_types

from easydel.infra.sharding import RuntimeShardingResolver, TensorLayout, sharding_for_layout


def promote_dtype(values, *, dtype=None):
    """Cast a tuple of arrays to a shared dtype.

    Lightweight replacement for ``flax.linen.dtypes.promote_dtype`` used by
    the MoE linear layers prior to the grouped matmul.

    Args:
        values: Iterable of arrays (or array-like values) to promote.
        dtype: Target dtype. If ``None``, the values are returned unchanged.

    Returns:
        Tuple of arrays cast to ``dtype`` if it was provided, otherwise the
        original ``values`` argument unchanged.
    """
    if dtype is None:
        return values
    return tuple(jnp.asarray(v, dtype=dtype) for v in values)


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


default_kernel_init = jax.nn.initializers.lecun_normal()
default_bias_init = jax.nn.initializers.zeros
Initializer = jax.nn.initializers.Initializer


def _moe_parameter_layout(
    *,
    direction: typing.Literal["row", "column"] | None,
    use_expert_tensor_mode: bool,
    is_bias: bool = False,
) -> TensorLayout | None:
    """Build the per-expert parameter layout for an MoE linear weight or bias.

    The layout determines how the parameter is partitioned across the mesh:
    along the expert (``EP``) axis, the tensor-parallel (``TP``) axis, or
    the special "expert-on-TP" mode used when experts fit on the TP axis.

    Args:
        direction: Parallelism direction of the linear layer. ``"column"``
            partitions output features, ``"row"`` partitions input features,
            and ``None`` disables the layout (returns ``None``).
        use_expert_tensor_mode: When ``True`` experts are placed on the TP
            axis instead of the EP axis. Useful when ``num_experts`` is small.
        is_bias: ``True`` for the 2-D bias parameter, ``False`` for the 3-D
            weight kernel.

    Returns:
        A :class:`TensorLayout` describing the partitioning of the parameter,
        or ``None`` when ``direction`` is ``None``.
    """
    if direction is None:
        return None
    if use_expert_tensor_mode:
        return TensorLayout((TP, None) if is_bias else (TP, None, None))
    if is_bias:
        return TensorLayout((EP, TP))
    if direction == "column":
        return TensorLayout((EP, None, TP))
    return TensorLayout((EP, TP, None))


class ParallelMoELinear(spx.Module):
    """A batched linear transformation layer for Mixture of Experts (MoE) models.

        This layer applies separate linear transformations for each expert in a MoE setup.
        The inputs are assumed to be sorted and grouped by expert, with `group_sizes`
        specifying how many tokens belong to each expert. It supports:
        - **Ragged Matrix Multiplication** via `jax.lax.ragged_dot_general`.
        - **Grouped Matrix Multiplication (GMM)** via a Pallas kernel for TPUs.

        Can optionally integrate with a runtime sharding resolver to shard parameters and
        use `shard_map` for distributed execution.

        **Distributed Execution:**

            This layer supports multiple parallelism strategies:

            - **Expert Parallelism (EP)**: Partition experts across devices on the expert axis
            - **Tensor Parallelism (TP)**: Partition weight matrices within each expert
            - **Data Parallelism (DP)**: Replicate across data batches
            - **Row/Column Parallelism**: Control which dimension is partitioned (input vs output)

            The sharding strategy is controlled by:
            1. `direction`: "row" or "column" determines which dimension is partitioned
            2. `use_expert_tensor_mode`: Whether experts are on TP axis (True) or EP axis (False)
            3. `runtime_sharding_resolver`: Provides mesh and axis resolution for sharding

        Attributes:
            num_experts: Number of experts.
            in_features: Input feature dimension.
            out_features: Output feature dimension.
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
            runtime_sharding_resolver: Handles sharding metadata for distributed execution.
            _direction: Sharding direction for ALT sharding ("row", "column", or None).


        Example:
            >>> from easydel.layers import ParallelMoELinear
            >>> import spectrax as spx
    from spectrax import nn
            >>>
            >>> # Create a column-parallel MoE linear layer
            >>> layer = ParallelMoELinear(
            ...     num_experts=8,
            ...     in_features=768,
            ...     out_features=3072,
            ...     direction="column",
            ...     rngs=rngs
            ... )
            >>>
            >>> # Inputs are sorted tokens grouped by expert
            >>> sorted_tokens = jnp.ones((1024, 768))  # 1024 tokens, 768 features
            >>> group_sizes = jnp.array([128, 132, 125, 130, 127, 129, 126, 127])  # per expert
            >>> sorted_experts = jnp.repeat(jnp.arange(8), group_sizes)
            >>>
            >>> # Apply expert FFN
            >>> output = layer(sorted_tokens, group_sizes, sorted_experts)
            >>> # output.shape = (1024, 3072)
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
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        runtime_sharding_resolver: RuntimeShardingResolver | None = None,
        partition_manager: RuntimeShardingResolver | None = None,
        direction: typing.Literal["row", "column"] | None = None,
        use_expert_tensor_mode: bool = False,
        weight_modif_fn: typing.Callable[[Array], Array] | None = None,
        rngs: spx.Rngs,
    ):
        """Initialize a ParallelMoELinear layer.

        Creates a batched linear transformation for MoE models with support for
        distributed execution and various parallelism strategies.

        Args:
            num_experts: Number of experts in the layer. Each expert has its own
                weight matrix.
            in_features: Size of the input feature dimension for each expert.
            out_features: Size of the output feature dimension for each expert.
            use_bias: Whether to include a bias term for each expert.
                Defaults to True.
            out_first: If True, kernel shape is `(num_experts, out_features, in_features)`,
                otherwise `(num_experts, in_features, out_features)`. This affects
                whether transpose_rhs is needed in grouped_matmul. Defaults to False.
            kernel_init: Initializer for the kernel weights.
                Defaults to lecun_normal().
            bias_init: Initializer for the bias. Defaults to zeros.
            dtype: Data type for computation. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters (weights, biases).
                Defaults to jnp.bfloat16.
            runtime_sharding_resolver: Runtime sharding resolver for parameter
                placement and axis mapping.
                When provided, enables distributed execution with shard_map.
                Defaults to None.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.
            direction: ALT-sharding direction, either `"row"`, `"column"`, or None.
                - "row": Input features are partitioned across TP axis
                - "column": Output features are partitioned across TP axis
                Defaults to None (no parallelism).
            use_expert_tensor_mode: If True, experts are distributed across the
                TP axis instead of the EP axis. This is useful when the number
                of experts is small and can fit on the TP dimension for better
                hardware utilization. Defaults to False.
            weight_modif_fn: Optional function to modify weights before the forward
                pass. Useful for applying runtime transformations like LoRA or
                quantization. Takes and returns an Array. Defaults to None.
            rngs: Random number generators for parameter initialization.
        """
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.out_first = out_first
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.runtime_sharding_resolver = (
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        self.use_expert_tensor_mode = use_expert_tensor_mode

        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.weight_modif_fn = weight_modif_fn
        if direction is not None:
            if direction not in ("row", "column"):
                raise ValueError(f"direction must be 'row' or 'column', got '{direction}'")
            self._direction = direction
        kshape = (num_experts, out_features, in_features) if out_first else (num_experts, in_features, out_features)
        weight_layout = _moe_parameter_layout(
            direction=self._direction,
            use_expert_tensor_mode=use_expert_tensor_mode,
            is_bias=False,
        )
        self.weight = spx.Parameter(
            kernel_init(rngs.param, kshape, param_dtype),
            sharding=sharding_for_layout(weight_layout),
        )
        if use_bias:
            bshape = (num_experts, out_features)
            bias_layout = _moe_parameter_layout(
                direction=self._direction,
                use_expert_tensor_mode=use_expert_tensor_mode,
                is_bias=True,
            )
            self.bias = spx.Parameter(
                bias_init(rngs.param, bshape, self.param_dtype),
                sharding=sharding_for_layout(bias_layout),
            )
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
            True if both a runtime sharding resolver and parallelism direction are configured,
            indicating the layer is ready for distributed execution with shard_map.
        """
        return self.runtime_sharding_resolver is not None and self._direction is not None

    @property
    def alt_sharding(self) -> PartitionSpec | None:
        """Returns the ALT (Alternative) sharding configuration for this layer.

        ALT sharding provides pre-defined sharding patterns for common parallelism
        strategies, simplifying the configuration of distributed execution.
        """

        from easydel.layers.moe._communication_utils import get_moe_partition_spec

        if self.direction is None:
            return None
        if self.use_expert_tensor_mode:
            return get_moe_partition_spec(self.runtime_sharding_resolver, "column", self.use_expert_tensor_mode, True)
        elif self.direction == "row":
            return get_moe_partition_spec(self.runtime_sharding_resolver, "row", self.use_expert_tensor_mode, False)
        elif self.direction == "column":
            return get_moe_partition_spec(self.runtime_sharding_resolver, "column", self.use_expert_tensor_mode, False)
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
        """Returns sharding axes for expert group sizes array.

        Group sizes specify how many tokens are assigned to each expert and are
        typically replicated across all devices.

        Returns:
            List containing [EMPTY], indicating group_sizes are replicated.
        """
        return [EMPTY]

    def _input_axes(self) -> list[str | None]:
        """Returns sharding axes for input activations based on parallelism direction.

        The input sharding depends on whether this is a row-parallel or column-parallel layer:
        - Row-parallel: Inputs are sharded on the feature dimension [DP, TP]
            because different devices hold different input features
        - Column-parallel: Inputs are replicated on the feature dimension [DP, EMPTY]
            because all devices need the full input to compute their output slice

        Returns:
            List of axis names defining input sharding pattern:
            - Row direction: [DP, TP] - data parallel and tensor parallel sharded
            - Column direction: [DP, EMPTY] - only data parallel sharded
            - No direction: [DP, EMPTY] - default to replicated features
        """
        if self.direction == "row":
            return [DP, TP]
        if self.direction == "column":
            return [DP, EMPTY]
        return [DP, EMPTY]

    def _output_axes(self) -> list[str | list[str] | None]:
        """Returns sharding axes for output activations based on parallelism direction.

        The output sharding depends on the parallelism strategy:
        - Row-parallel: Outputs are fully reduced and replicated [DP, EMPTY]
            because all devices contribute partial results that are summed
        - Column-parallel: Outputs are sharded on the feature dimension
            because each device produces a different slice of the output
            - In expert tensor mode: [DP, TP]
            - In standard mode: [DP, [EP, TP]] (combined expert+tensor parallel)

        Returns:
            List of axis names defining output sharding pattern:
            - Row direction: [DP, EMPTY] - replicated after all-reduce
            - Column direction: [DP, TP] or [DP, [EP, TP]] - sharded features
            - No direction: [DP, EMPTY] - default to replicated
        """
        if self.direction == "row":
            return [DP, EMPTY]
        if self.direction == "column":
            if self.use_expert_tensor_mode:
                return [DP, TP]
            return [DP, [EP, TP]]
        return [DP, EMPTY]

    def forward(
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
        weight = self.weight.value
        if self.weight_modif_fn is not None:
            weight = self.weight_modif_fn(weight)
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
            cfg=GroupedMatmulConfig(bypass_xla_tiling=True),
        )

        if self.bias is not None:
            output += self._expand_bias_ragged(group_sizes, sorted_experts=sorted_experts)

        return output

    def _expand_bias_ragged(
        self,
        group_sizes: Int[Array, "num_groups"],  # noqa
        sorted_experts: Int[Array, "tokens_ragged"] | None = None,  # noqa
    ) -> Float[Array, "tokens_ragged out_dim"]:
        """Expands bias to match the ragged token batch structure.

        This method aligns the per-expert bias with the ragged token layout by
        repeating each expert's bias according to how many tokens are assigned to it.
        This is necessary because tokens are sorted and grouped by expert, and each
        token needs its assigned expert's bias added to the output.

        Two modes of operation:
        1. If `sorted_experts` is provided (expert tensor mode): Uses the expert IDs
           directly to index into the bias array, handling cases where not all experts
           on a shard receive tokens.
        2. If `sorted_experts` is None (standard mode): Generates expert indices by
           repeating each expert ID according to its group size.

        Args:
            group_sizes: The number of tokens assigned to each expert on this shard.
                Shape: (num_local_experts,). Each element indicates how many tokens
                were routed to that expert.
            sorted_experts: Optional pre-computed expert IDs for each token. Shape:
                (total_tokens,). Required in expert tensor mode where expert distribution
                may be sparse. If None, expert indices are generated from group_sizes.

        Returns:
            Expanded bias array aligned with sorted tokens. Shape: (total_tokens, out_features).
            Each row contains the bias for the expert assigned to that token.

        Example:
            >>> # Standard mode: 3 experts with [3, 2, 4] tokens respectively
            >>> bias = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # (3, 2)
            >>> group_sizes = jnp.array([3, 2, 4])
            >>> expanded = _expand_bias_ragged(group_sizes, sorted_experts=None)
            >>> # expanded = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2],  # expert 0 repeated 3 times
            >>> #              [0.3, 0.4], [0.3, 0.4],              # expert 1 repeated 2 times
            >>> #              [0.5, 0.6], [0.5, 0.6], [0.5, 0.6], [0.5, 0.6]]  # expert 2 repeated 4 times
            >>>
            >>> # Expert tensor mode: sparse expert assignment
            >>> sorted_experts = jnp.array([0, 0, 2, 2, 2])  # 5 tokens, only experts 0 and 2
            >>> expanded = _expand_bias_ragged(None, sorted_experts=sorted_experts)
            >>> # expanded = [[0.1, 0.2], [0.1, 0.2], [0.5, 0.6], [0.5, 0.6], [0.5, 0.6]]
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
