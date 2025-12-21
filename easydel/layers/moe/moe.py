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

"""Mixture of Experts (MoE) layer implementations for EasyDeL.

This module provides a comprehensive implementation of Mixture of Experts (MoE) layers
for large-scale neural networks. It includes support for various routing strategies,
load balancing techniques, and distributed training optimizations.

Key Components:
    - **BaseMoeModule**: Abstract base class for MoE implementations with common
      utilities for routing, permutation, and metric computation.
    - **ParallelMoELinear**: Batched linear transformation layer for expert networks
      with support for ragged and grouped matrix multiplication.
    - **Routing Strategies**: Multiple routing algorithms including top-k, switch,
      expert choice, and hash-based routing.
    - **Load Balancing**: Various strategies to ensure balanced expert utilization
      including standard, switch transformer, and expert choice methods.
    - **Distributed Support**: Full support for expert parallelism (EP), tensor
      parallelism (TP), and data parallelism (DP) with optimized all-to-all
      communication patterns.

The module is designed for efficient execution on TPUs and GPUs with optimizations
for:
    - Custom VJP for gradient-efficient sorting operations
    - Pallas-based grouped matrix multiplication kernels for TPUs
    - Ragged tensor operations for variable-length expert assignments
    - Automatic sharding and partitioning for distributed training

Example:
    >>> from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
    >>> # Create a custom MoE layer by extending BaseMoeModule
    >>> class CustomMoE(BaseMoeModule):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         # Initialize gate and expert layers
    ...     def __call__(self, hidden_states):
    ...         # Implement forward pass using _moe_call_standard helper
    ...         return self._moe_call_standard(...)
"""

from __future__ import annotations

import typing
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.extend
from eformer import common_types
from eformer.loggings import get_logger
from ejkernel.modules import GroupedMatmulConfig, grouped_matmul
from flax import nnx as nn
from jax import numpy as jnp
from jax import shard_map
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.utils.helpers import check_bool_flag

from .utils import (
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
    get_all_to_all_params,
    get_experts_location,
    get_moe_partition_spec,
    local_permute,
    permute,
    resolve_eformer_axis,
    sort_activations,
    unpermute,
)

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

logger = get_logger(__name__)


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


class BaseMoeModule(nn.Module, ABC):
    """An abstract base class for Mixture of Experts (MoE) modules.

    This class provides a foundational structure and common utilities for
    implementing various MoE architectures. It includes methods for token routing,
    data permutation for efficient expert computation, load balancing loss
    calculation, and sharding for distributed environments. Subclasses are
    expected to implement the `__call__` method to define the specific MoE forward
    pass.

    Attributes:
        config: The configuration object for the MoE module.
        mesh: The JAX device mesh for distributed computation.
        n_routed_experts: The total number of experts available for routing.
        num_experts_per_tok: The number of experts each token is routed to (k).
        hidden_size: The dimension of the hidden states.
        lbl_coef: The coefficient for the load balancing loss.
        rzl_coef: The coefficient for the router z-loss.
        routing_strategy: The strategy used for routing tokens to experts.
        load_balancing_strategy: The strategy used for calculating the load
            balancing loss.
    """

    def __init__(
        self,
        config: EasyDeLBaseConfig,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        hidden_size: int | None = None,
        lbl_coef: float | None = None,
        rzl_coef: float | None = None,
        routing_strategy: MoeRoutingStrategy = MoeRoutingStrategy.TOP_K,
        load_balancing_strategy: MoeLoadBalancingStrategy = MoeLoadBalancingStrategy.STANDARD,
        moe_hooks: MoeFusedHooks | None = None,
    ):
        """Initializes the BaseMoeModule.

        Args:
            config: The configuration object for this MoE module.
            n_routed_experts: The total number of experts. If None, it's taken
                from `config.n_routed_experts`.
            num_experts_per_tok: The number of experts to route each token to. If
                None, it's taken from `config.num_experts_per_tok`.
            hidden_size: The hidden dimension of the input and output. If None,
                it's taken from `config.hidden_size`.
            lbl_coef: The coefficient for the load balancing loss.
            rzl_coef: The coefficient for the router z-loss.
            routing_strategy: The strategy for routing tokens to experts.
            load_balancing_strategy: The strategy for load balancing.
            moe_hooks: Hook system for custom interventions during MoE execution.
                If None, uses default MoeFusedHooks with no custom hooks.
        """
        super().__init__()
        self.config = config
        self.mesh = config.mesh
        self.partition_manager = config.partition_manager
        self.n_routed_experts = n_routed_experts or config.n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok or config.num_experts_per_tok
        self.hidden_size = hidden_size or config.hidden_size
        self.lbl_coef = lbl_coef
        self.rzl_coef = rzl_coef
        self.routing_strategy = routing_strategy
        self.load_balancing_strategy = load_balancing_strategy
        self.moe_hooks = MoeFusedHooks() if moe_hooks is None else moe_hooks
        self.module_moe_method = self.config.moe_method

        self.expert_mesh = self.config.expert_mesh
        self.auto_expert_mesh = self.config.auto_expert_mesh
        self.expert_abstract_mesh = self.config.expert_abstract_mesh

        self.dtype = getattr(self, "dtype", jnp.bfloat16)

    def get_moe_spec(
        self,
        direction: tp.Literal["row", "column"],
        tensors_are_expert: bool,
        is_bias: bool = False,
    ) -> PartitionSpec:
        """Generate partition spec for MoE weight tensors.

        This helper creates appropriate partition specs for MoE expert weights
        based on the sharding strategy and tensor properties.

        Args:
            direction: Weight matrix orientation:
                - "column": For column-wise sharding (wi/wu kernels)
                - "row": For row-wise sharding (wd kernel)
            tensors_are_expert: If True, uses expert tensor mode (experts on TP axis).
                If False, uses standard mode (experts on EP axis).
            is_bias: If True, generates spec for bias tensor (2D instead of 3D).

        Returns:
            PartitionSpec appropriate for the tensor.

        Examples:
            Standard mode (tensors_are_expert=False):
                - Column weight: [expert, None, tp]  # wi/wu: [E, H, M]
                - Row weight: [expert, tp, None]     # wd: [E, M, H]
                - Bias: [expert, None]               # [E, dim]

            Expert tensor mode (tensors_are_expert=True):
                - Column weight: [tp, None, None]    # Experts on TP
                - Row weight: [tp, None, None]       # Experts on TP
                - Bias: [tp, None]                   # [E, dim]
        """

        return get_moe_partition_spec(
            partition_manager=self.partition_manager,
            direction=direction,
            tensors_are_expert=tensors_are_expert,
            is_bias=is_bias,
            fsdp_is_ep_bound=self.config.fsdp_is_ep_bound,
            sp_is_ep_bound=self.config.sp_is_ep_bound,
            module_view=False,
        )

    def _get_sharding_status(self):
        """Resolves and returns all parallelism axis names and sizes for this MoE layer.

        This method queries the partition manager to resolve logical axis names to
        physical mesh axis names, and retrieves their sizes from the device mesh.
        It handles both standard and expert-tensor parallelism modes.

        In standard mode:
            - Expert axis → EP (expert parallel)
            - Tensor axis → TP (tensor parallel)

        In expert-tensor mode (`use_expert_tensor_mode=True`):
            - Expert axis → TP (tensor parallel)
            - Tensor axis → EP (expert parallel)

        This axis swapping allows alternative sharding strategies for specific use cases.

        Returns:
            A tuple containing 10 elements:
                1. data_axis_name (str): Resolved data parallel axis name
                2. fsdp_axis_name (str): Resolved FSDP axis name
                3. expert_axis_name (str): Resolved expert parallel axis name
                4. tensor_axis_name (str): Resolved tensor parallel axis name
                5. sp_axis_name (str): Resolved sequence parallel axis name
                6. dp_size (int): Data parallel degree (number of devices)
                7. fsdp_size (int): FSDP degree
                8. ep_size (int): Expert parallel degree
                9. tp_size (int): Tensor parallel degree
                10. sp_size (int): Sequence parallel degree

        Note:
            Sizes default to 1 if the axis doesn't exist in the mesh.
        """
        partition_manager = self.partition_manager
        data_axis_name = resolve_eformer_axis(DP, partition_manager)

        if self.config.use_expert_tensor_mode:
            expert_axis_name = resolve_eformer_axis(TP, partition_manager)
            tensor_axis_name = resolve_eformer_axis(EP, partition_manager)
        else:
            expert_axis_name = resolve_eformer_axis(EP, partition_manager)
            tensor_axis_name = resolve_eformer_axis(TP, partition_manager)

        fsdp_axis_name = resolve_eformer_axis(FSDP, partition_manager)
        sp_axis_name = resolve_eformer_axis(SP, partition_manager)

        dp_size = self.mesh.shape.get(data_axis_name, 1)
        ep_size = self.mesh.shape.get(expert_axis_name, 1)
        tp_size = self.mesh.shape.get(tensor_axis_name, 1)
        fsdp_size = self.mesh.shape.get(fsdp_axis_name, 1)
        sp_size = self.mesh.shape.get(sp_axis_name, 1)

        return (
            data_axis_name,
            fsdp_axis_name,
            expert_axis_name,
            tensor_axis_name,
            sp_axis_name,
            dp_size,
            fsdp_size,
            ep_size,
            tp_size,
            sp_size,
        )

    def _replicate_and_sort_tokens(
        self,
        inputs_flat: jax.Array,
        selected_experts: jax.Array,
        use_custom_sort_vjp: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Replicates tokens k times and sorts them by assigned expert ID.

        This function prepares tokens for expert computation by:
        1. Replicating each token k times (once per selected expert)
        2. Sorting all replicated tokens so tokens for the same expert are contiguous
        3. Computing group sizes (how many tokens per expert)
        4. Creating sorted expert ID array aligned with sorted tokens

        The sorted layout enables efficient grouped/ragged matrix multiplication where
        each expert processes its assigned tokens as a contiguous batch.

        Args:
            inputs_flat: Flattened token representations. Shape: (num_tokens, hidden_dim).
            selected_experts: Expert assignments per token. Shape: (num_tokens, k)
                where k = `num_experts_per_tok`.
            use_custom_sort_vjp: Whether to use custom VJP for memory-efficient sorting.
                Defaults to True.

        Returns:
            A tuple containing:
                - sorted_inputs: Token representations sorted by expert. Shape: (num_tokens*k, hidden_dim).
                - sorted_by_expert: Sorting indices for the permutation. Shape: (num_tokens*k,).
                - group_sizes: Number of tokens assigned to each expert. Shape: (n_routed_experts,).
                - sorted_experts: Expert IDs aligned with sorted_inputs. Shape: (num_tokens*k,).

        Example:
            >>> # 2 tokens, 2 experts per token, 4 total experts
            >>> inputs = jnp.ones((2, 128))
            >>> selected = jnp.array([[0, 2], [1, 3]])  # token 0→experts 0,2; token 1→experts 1,3
            >>> sorted_inputs, indices, sizes, expert_ids = _replicate_and_sort_tokens(inputs, selected)
            >>> # sorted_inputs: tokens grouped as [token0_e0, token1_e1, token0_e2, token1_e3]
            >>> # sizes: [1, 1, 1, 1] - one token per expert
        """
        k = selected_experts.shape[-1]
        flat_idx = selected_experts.reshape(-1)
        sorted_by_expert = jnp.argsort(flat_idx)
        replicated = jnp.repeat(inputs_flat, k, axis=0)
        sorted_inputs = sort_activations(replicated, sorted_by_expert, use_custom_sort_vjp)
        group_sizes = jnp.bincount(flat_idx, length=self.n_routed_experts)
        sorted_experts = jnp.repeat(
            jnp.arange(self.n_routed_experts),
            repeats=group_sizes,
            total_repeat_length=flat_idx.shape[0],
        )
        return sorted_inputs, sorted_by_expert, group_sizes, sorted_experts

    def _apply_capacity_mask(
        self,
        selected_experts: jax.Array,
        weights: jax.Array,
        capacity_factor: float,
    ) -> jax.Array:
        """Applies soft capacity constraints to expert assignments.

        This method limits the number of tokens each expert can process by zeroing out
        weights for tokens that exceed the expert's capacity. This helps prevent expert
        overload and improves load balancing during training.

        The capacity is computed as:
            capacity = max(ceil(tokens_per_batch / n_experts) * capacity_factor, capacity_factor)

        Tokens are processed in order, and once an expert reaches capacity, subsequent
        token assignments to that expert receive zero weight.

        Args:
            selected_experts: Expert assignments per token. Shape: (B, S, k) where
                B=batch_size, S=seq_len, k=num_experts_per_tok.
            weights: Expert weights per token. Shape: (B, S, k).
            capacity_factor: Multiplier for base capacity. Values > 1.0 allow more tokens,
                values < 1.0 enforce stricter limits. Typically in range [1.0, 2.0].

        Returns:
            Modified weights with overflow tokens masked to zero. Shape: (B, S, k).
            Tokens within capacity retain their original weights; overflow tokens get 0.

        Example:
            >>> # 2 batches, 4 tokens per batch, 2 experts per token, 4 total experts
            >>> experts = jnp.array([[[0, 1], [0, 2], [1, 3], [2, 3]],  # batch 0
            ...                      [[0, 1], [1, 2], [2, 3], [3, 0]]])  # batch 1
            >>> weights = jnp.ones((2, 4, 2))
            >>> masked_weights = _apply_capacity_mask(experts, weights, capacity_factor=1.5)
            >>> # Some tokens will have zero weight if they exceed expert capacity
        """
        B, S, k = selected_experts.shape
        tokens_per_batch = S * k
        cap = int(max(jnp.ceil(tokens_per_batch / self.n_routed_experts) * capacity_factor, capacity_factor))
        expert_mask = jax.nn.one_hot(selected_experts, num_classes=self.n_routed_experts, dtype=jnp.int32)
        fused = expert_mask.reshape(B, S * k, self.n_routed_experts)
        counts = jnp.cumsum(fused, axis=1)
        counts = counts.reshape(B, S, k, self.n_routed_experts)
        keep = (counts <= cap).astype(weights.dtype)
        keep_for_slot = jnp.sum(keep, axis=-1)
        return weights * keep_for_slot

    def _expert_group_mask(self, gate_logits: jax.Array, n_groups: int, topk_groups: int) -> jax.Array:
        """Creates a mask for hierarchical routing with grouped experts.

        This method implements hierarchical or grouped routing where experts are organized
        into groups, and tokens first select top-k groups, then select experts within
        those groups. This can improve routing efficiency and reduce computation when
        the number of experts is very large.

        The algorithm:
        1. Partition experts into n_groups
        2. For each group, compute a group score (sum of top-2 expert logits in that group)
        3. Select topk_groups with highest scores
        4. Create a mask that zeros out logits for experts in non-selected groups

        Args:
            gate_logits: Router logits for all experts. Shape: (batch*seq, n_experts).
            n_groups: Number of expert groups to partition experts into.
                Must evenly divide n_routed_experts.
            topk_groups: Number of groups to select per token. Typically 1 or 2.

        Returns:
            Binary mask for gate logits. Shape: (batch*seq, n_experts).
            1.0 for experts in selected groups, 0.0 for others.

        Example:
            >>> # 8 experts divided into 4 groups of 2 experts each
            >>> logits = jnp.ones((16, 8))  # 16 tokens, 8 experts
            >>> mask = _expert_group_mask(logits, n_groups=4, topk_groups=2)
            >>> # mask will have 1.0 for experts in 2 selected groups, 0.0 for others
            >>> masked_logits = logits * mask  # Zero out non-selected groups
        """
        BS = gate_logits.shape[0]
        experts_per_group = self.n_routed_experts // n_groups
        scores_grouped = gate_logits.reshape(BS, n_groups, experts_per_group)
        top2_vals, _ = jax.lax.top_k(scores_grouped, k=2)
        group_scores = jnp.sum(top2_vals.astype(jnp.float32), axis=-1)
        _, group_idx = jax.lax.top_k(group_scores, k=topk_groups)
        mask_groups = jax.nn.one_hot(group_idx, num_classes=n_groups, dtype=jnp.float32).sum(axis=-2)
        mask = jnp.broadcast_to(mask_groups[..., None], (BS, n_groups, experts_per_group)).reshape(BS, -1)
        return mask

    def _compute_load_balancing_loss(
        self,
        router_probs: jax.Array,
        expert_loads: jax.Array,
        strategy: MoeLoadBalancingStrategy | None = None,
    ) -> float | None:
        """Computes the load balancing auxiliary loss to distribute tokens evenly across experts."""
        strategy = strategy or self.load_balancing_strategy

        if strategy == MoeLoadBalancingStrategy.NONE or self.lbl_coef is None:
            return None

        if strategy == MoeLoadBalancingStrategy.STANDARD:
            f = expert_loads * self.n_routed_experts / self.num_experts_per_tok
            p = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * jnp.sum(f * p)

        elif strategy == MoeLoadBalancingStrategy.SWITCH_TRANSFORMER:
            num_tokens = router_probs.shape[0]
            expert_fraction = expert_loads / num_tokens
            router_fraction = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * self.n_routed_experts * jnp.sum(expert_fraction * router_fraction)

        elif strategy == MoeLoadBalancingStrategy.EMPTY_CHOICE:
            return self.lbl_coef * jnp.var(expert_loads)

        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

    def _compute_router_z_loss(self, router_logits: Float[Array, "batch_seq num_experts"]) -> float | None:
        """Computes the router z-loss to encourage small logit magnitudes for training stability."""
        if self.rzl_coef is None:
            return None

        log_z = jax.nn.logsumexp(router_logits.astype(jnp.float32), axis=-1)
        return self.rzl_coef * jnp.mean(log_z**2)

    def _compute_metrics(
        self,
        router_logits: jax.Array,
        router_probs: jax.Array,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        expert_loads: jax.Array,
    ) -> MoeMetrics:
        """Computes and aggregates all MoE-related metrics and auxiliary losses."""
        metrics = MoeMetrics(
            expert_loads=expert_loads,
            router_probs=router_probs,
            selected_experts=selected_experts,
            selected_weights=selected_weights,
        )
        metrics.load_balancing_loss = self._compute_load_balancing_loss(router_probs, expert_loads)
        metrics.router_z_loss = self._compute_router_z_loss(router_logits)
        metrics.expert_utilization = jnp.mean(expert_loads > 0)
        metrics.routing_entropy = jnp.mean(-jnp.sum(router_probs * jnp.log(router_probs + 1e-8), axis=-1))
        return metrics

    def _apply_expert_sharding(self, tensor: Float[Array, ...], tensor_type: str = "weight") -> Float[Array, ...]:
        """Applies expert parallel sharding to a tensor for distributed training.

        This method determines the appropriate sharding specification for expert parameters
        based on the tensor type and shape, then places the tensor on devices according
        to that specification. The sharding is currently set to replicate all dimensions
        (EMPTY) but can be extended to support expert-parallel sharding.

        Args:
            tensor: The tensor to shard. Can be weights, biases, or activations.
            tensor_type: Type hint for determining sharding strategy. Options:
                - "weight_col": Column-parallel weight (output dim partitioned)
                - "weight_row": Row-parallel weight (input dim partitioned)
                - "bias": Bias parameters
                - "weight" (default): Generic weight tensor

        Returns:
            The input tensor with sharding applied, placed on appropriate devices
            according to the resolved partition specification.

        Note:
            Current implementation uses EMPTY (replicated) sharding for all dimensions.
            Future versions may shard the expert dimension across EP devices.

        Example:
            >>> weight = jnp.ones((n_experts, hidden_dim, intermediate_dim))
            >>> sharded_weight = _apply_expert_sharding(weight, "weight_col")
            >>> # weight is now sharded across devices according to mesh configuration
        """
        pmag = self.partition_manager

        if tensor_type == "weight_col":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "weight_row":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "bias":
            if tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=tensor.shape)

        else:
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        return jax.device_put(tensor, jax.sharding.NamedSharding(self.mesh, sharding_spec))

    def _get_gate_layer_sharding(self, weight_shape: tuple) -> PartitionSpec:
        """Returns the partition specification for gate/router layer weights.

        The gate layer maps hidden states to expert logits, producing routing decisions.
        This method determines how the gate weight matrix should be sharded across devices.

        Args:
            weight_shape: Shape of the gate weight matrix, typically (hidden_dim, n_experts).

        Returns:
            PartitionSpec defining how to shard the gate weights. Currently uses
            [EMPTY, EMPTY] (replicated across all dimensions).

        Note:
            Gate weights are typically small relative to expert FFN weights and
            are usually replicated for efficient routing computation.
        """
        pmag = self.partition_manager
        return pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=weight_shape)

    def _get_gate_layer_bias_sharding(self, bias_shape: tuple) -> PartitionSpec:
        """Returns the partition specification for gate/router layer bias.

        Args:
            bias_shape: Shape of the gate bias vector, typically (n_experts,).

        Returns:
            PartitionSpec defining how to shard the gate bias. Currently uses
            [EMPTY] (replicated).

        Note:
            Like gate weights, bias is usually replicated for efficient routing.
        """
        pmag = self.partition_manager
        return pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=bias_shape)

    def _validate_routing_inputs(
        self, hidden_states: Float[Array, "batch seq hidden_dim"], router_logits: Float[Array, "batch_seq num_experts"]
    ) -> None:
        """Validates the shapes of inputs for routing operations."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input hidden dimension {hidden_states.shape[-1]} doesn't "
                f"match config hidden dimension {self.hidden_size}"
            )

        if router_logits.shape[-1] != self.n_routed_experts:
            raise ValueError(
                f"Router logits expert dimension {router_logits.shape[-1]} doesn't match "
                f"config expert count {self.n_routed_experts}"
            )

        if router_logits.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]:
            raise ValueError(
                f"Router logits batch dimension {router_logits.shape[0]} doesn't match "
                f"flattened input batch dimension {hidden_states.shape[0] * hidden_states.shape[1]}"
            )

    def _apply_capacity_constraint(
        self,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        capacity_factor: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Applies soft capacity constraint to limit tokens per expert."""
        if capacity_factor is None:
            capacity_factor = 1.0
        num_tokens = selected_experts.shape[0]
        max_capacity = int(capacity_factor * num_tokens / self.n_routed_experts)
        expert_counts = jnp.bincount(selected_experts.flatten(), length=self.n_routed_experts)
        over_capacity_ratio = jnp.maximum(expert_counts / max_capacity, 1.0)
        weight_adjustments = 1.0 / over_capacity_ratio[selected_experts]
        constrained_weights = selected_weights * weight_adjustments
        weight_sum = jnp.sum(constrained_weights, axis=-1, keepdims=True)
        constrained_weights = jnp.where(weight_sum > 0, constrained_weights / weight_sum, constrained_weights)
        return selected_experts, constrained_weights

    def _create_expert_mask(
        self,
        selected_experts: Int[Array, "batch_seq k"],
        expert_id: int,
    ) -> Bool[Array, "batch_seq"]:  # type: ignore #noqa
        """Creates a boolean mask identifying tokens assigned to a specific expert.

        This utility method is useful for per-expert analysis, debugging, or when
        processing experts individually rather than in batched/grouped fashion.

        Args:
            selected_experts: Expert assignments per token. Shape: (batch*seq, k)
                where k = num_experts_per_tok.
            expert_id: The expert ID to create a mask for (0 to n_routed_experts-1).

        Returns:
            Boolean mask where True indicates the token was assigned to the specified
            expert. Shape: (batch*seq,).

        Example:
            >>> selected = jnp.array([[0, 2], [1, 3], [0, 1]])  # 3 tokens, 2 experts each
            >>> mask = _create_expert_mask(selected, expert_id=0)
            >>> # mask = [True, False, True] - tokens 0 and 2 use expert 0
        """
        return jnp.any(selected_experts == expert_id, axis=-1)

    def _sparse_moe_call(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: nn.Module,  # [H, E]
        wi_kernel: jax.Array,  # [E, H, M]
        wu_kernel: jax.Array,  # [E, H, M]
        wd_kernel: jax.Array,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, H]
        wu_bias: jax.Array | None = None,  # [E, H]
        wd_bias: jax.Array | None = None,  # [E, M]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
    ):
        """Fused MoE path using grouped matmul and shard_map.

        This is the core fused MoE implementation that routes tokens to experts,
        permutes them to an expert-grouped layout, applies expert FFNs via grouped
        matmul kernels, and unpermutes/combines outputs. It supports both
        ring-of-experts and all-to-all expert-parallel communication depending on
        configuration and mesh sizes.

        **Architecture Overview:**
            1. **Routing**: Compute router logits via gate_layer and apply softmax
            2. **Permutation**: Sort tokens by expert assignment for grouped computation
            3. **Expert Computation**: Apply grouped matmul for W_i, W_u, activation, W_d
            4. **Communication** (if EP > 1):
               - Ring-of-Experts: All-gather pattern with local expert subsets
               - All-to-All: Ragged all-to-all for token redistribution
            5. **Unpermutation**: Restore token order and combine expert outputs
            6. **Resharding**: Convert from 3D expert mesh back to 5D model mesh

        **Hook Integration:**
            This method reads hooks from `self.moe_hooks` (automatically configured
            by `moe_call()` based on routing strategy). The following hooks
            are used at specific points:

            - `select_hook`: Refines expert selection weights. For TOP_K routing,
              this defaults to weight normalization (softmax). Called during permutation.
            - `refine_weights_hook`: Refines weights before W_i and W_u projections.
            - `refine_inputs_hook`: Refines token representations before expert-parallel
              all-to-all communication in distributed settings.

            Other hooks in `MoeFusedHooks` are not used in this path but could be
            added in future extensions.

        Args:
            hidden_state: Input tensor. Shape: [B, S, H].
            gate_layer: Router module mapping H -> E (produces logits).
            wi_kernel: Expert W_i kernel. Shape: [E, H, M].
            wu_kernel: Expert W_u kernel. Shape: [E, H, M].
            wd_kernel: Expert W_d kernel. Shape: [E, M, H].
            wi_bias: Optional bias for W_i. Shape: [E, H].
            wu_bias: Optional bias for W_u. Shape: [E, H].
            wd_bias: Optional bias for W_d. Shape: [E, M].
            ffn_activation: Optional custom activation combining (w0, w1) -> output.
            act_fn: Activation used when `ffn_activation` is not provided.

        Returns:
            Tuple `(output, router_logits)` where:
            - output: MoE layer output. Shape: [B, S, H].
            - router_logits: Pre-softmax router logits for auxiliary losses. Shape: [B*S, E].

        Example:
            >>> # Setup MoE layer with 8 experts, top-2 routing
            >>> config.n_routed_experts = 8
            >>> config.num_experts_per_tok = 2
            >>> config.use_ring_of_experts = False  # Use all-to-all
            >>>
            >>> # Initialize expert kernels
            >>> wi_kernel = jax.random.normal(key, (8, 768, 3072))  # gate/up
            >>> wu_kernel = jax.random.normal(key, (8, 768, 3072))  # up
            >>> wd_kernel = jax.random.normal(key, (8, 3072, 768))  # down
            >>>
            >>> # Call fused MoE
            >>> hidden_states = jnp.ones((2, 512, 768))  # (batch, seq, hidden)
            >>> output, logits = moe_layer._sparse_moe_call(
            ...     hidden_state=hidden_states,
            ...     gate_layer=gate,
            ...     wi_kernel=wi_kernel,
            ...     wu_kernel=wu_kernel,
            ...     wd_kernel=wd_kernel,
            ...     act_fn=jax.nn.silu,
            ... )
            >>> # output.shape = (2, 512, 768)
            >>> # logits.shape = (1024, 8)  # batch*seq, n_experts
        """

        select_hook = self.moe_hooks.select_hook if self.moe_hooks else None
        refine_weights_hook = self.moe_hooks.refine_weights_hook if self.moe_hooks else None
        refine_inputs_hook = self.moe_hooks.refine_inputs_hook if self.moe_hooks else None
        scale_replicated_inputs = self.moe_hooks.scale_replicated_inputs if self.moe_hooks else None
        output_weights_hook = self.moe_hooks.output_weights_hook if self.moe_hooks else None

        hooks = self.moe_hooks
        _BS, _SQLN, HD = hidden_state.shape

        hidden_state = hidden_state.astype(self.dtype)
        if hooks is not None and hooks.before_gate is not None:
            hidden_state = hooks.before_gate(hidden_state)

        prein_gate_logits = gate_layer(hidden_state.reshape(-1, HD))
        if hooks is not None and hooks.after_gate is not None:
            prein_gate_logits = hooks.after_gate(prein_gate_logits)

        if hooks is not None and hooks.normalize_gate_logits is not None:
            gate_logits = hooks.normalize_gate_logits(prein_gate_logits)
        else:
            gate_logits = jax.nn.softmax(prein_gate_logits.astype("f4"), axis=-1).astype(prein_gate_logits.dtype)
        if hooks is not None and hooks.before_topk is not None:
            gate_logits = hooks.before_topk(gate_logits)

        # Use expert_mesh (3D: dp, ep, tp) for cleaner sharding
        expert_mesh = self.auto_expert_mesh
        pm = self.partition_manager

        # Resolve axis names from partition_manager (not directly from mesh)
        dp_axis_name = resolve_eformer_axis(DP, pm)
        expert_axis_name = resolve_eformer_axis(EP, pm)
        tensor_axis_name = resolve_eformer_axis(TP, pm)

        ep_size = expert_mesh.shape[expert_axis_name]
        tp_size = expert_mesh.shape[tensor_axis_name]

        if self.config.use_expert_tensor_mode:
            assert tp_size == 1, "if using `ExpertTensorMode` Expert Parallel size should be 1."

        # Simplified partition specs using 3D expert_mesh
        input_ps = jax.sharding.PartitionSpec(dp_axis_name, None, None)
        glps = jax.sharding.PartitionSpec(dp_axis_name, None)

        if self.config.use_expert_tensor_mode:
            output_ps = jax.sharding.PartitionSpec(dp_axis_name, None, None)
        else:
            output_ps = jax.sharding.PartitionSpec(dp_axis_name, None, tensor_axis_name)

        if ffn_activation is None:

            def ffn_activation(x0: jax.Array, x1: jax.Array) -> jax.Array:
                return act_fn(x0) * x1

        # Generate weight sharding specs using helper function
        use_expert_tensor = self.config.use_expert_tensor_mode

        wikps = self.get_moe_spec("column", use_expert_tensor, is_bias=False)
        wukps = self.get_moe_spec("column", use_expert_tensor, is_bias=False)
        wdkps = self.get_moe_spec("row", use_expert_tensor, is_bias=False)

        wibps = self.get_moe_spec("column", use_expert_tensor, is_bias=True) if wi_bias is not None else None
        wubps = self.get_moe_spec("column", use_expert_tensor, is_bias=True) if wu_bias is not None else None
        wdbps = self.get_moe_spec("row", use_expert_tensor, is_bias=True) if wd_bias is not None else None

        preferred_element_type = jnp.bfloat16
        if jnp.dtype(self.dtype) == jnp.float32:
            preferred_element_type = jnp.float32
        gmm_kws = {"preferred_element_type": preferred_element_type}
        if self.config.moe_force_xla_gmm:
            gmm_kws.update(dict(cfg=GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True)))
        else:
            if jax.default_backend() == "tpu":
                moe_block_m = int(getattr(self.config, "moe_tiling_size_batch", 1024))
                moe_block_n = int(getattr(self.config, "moe_tiling_size_dim", 1024))
                # Pallas TPU lowering requires block shapes that satisfy hardware constraints.
                # If the configured tile sizes are incompatible, fall back to XLA.
                if (moe_block_m % 8 != 0) or (moe_block_n % 128 != 0):
                    gmm_kws.update(dict(cfg=GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True)))
                else:
                    gmm_kws.update(platform="pallas")
                    if check_bool_flag("DISABLE_MOE_AUTOTUNE_ON_TPU", False):
                        gmm_kws.update(
                            cfg=GroupedMatmulConfig(
                                platform="pallas",
                                bypass_xla_tiling=True,
                                block_m=moe_block_m,
                                block_n=moe_block_n,
                                block_k=512,
                            )
                        )

        @partial(
            shard_map,
            mesh=expert_mesh,  # Use 3D expert_mesh instead of 5D mesh
            in_specs=(input_ps, glps, wikps, wukps, wdkps, wibps, wubps, wdbps),
            out_specs=output_ps,
            check_vma=False,
        )
        def _sparse_call(
            x: jax.Array,
            gate_logits: jax.Array,
            wi_kernel: jax.Array,
            wu_kernel: jax.Array,
            wd_kernel: jax.Array,
            wi_bias: jax.Array | None,
            wu_bias: jax.Array | None,
            wd_bias: jax.Array | None,
        ):
            batch_size, sequence_length, _ = x.shape
            expert_shard_id = jax.lax.axis_index(expert_axis_name)

            if self.config.use_ring_of_experts:
                x, gate_logits = tuple(
                    jax.lax.all_gather(
                        z,
                        axis_name=expert_axis_name,
                        tiled=True,
                    )
                    for z in (x, gate_logits)
                )

                # "Route" tokens within each shard.
                experts_per_shard = self.n_routed_experts // ep_size
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=experts_per_shard * expert_shard_id,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                    select_hook=select_hook,
                    refine_weights_hook=refine_weights_hook,
                    refine_inputs_hook=refine_inputs_hook,
                    scale_replicated_inputs=scale_replicated_inputs,
                )

                group_sizes = group_sizes[:experts_per_shard]  # only the local experts
                # Optimize: use dynamic slice instead of masking to avoid wasted computation
                valid_token_count = jnp.sum(group_sizes)
                x = jax.lax.dynamic_slice_in_dim(x, 0, valid_token_count, axis=0)
            else:
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=None,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                    select_hook=select_hook,
                    refine_weights_hook=refine_weights_hook,
                    refine_inputs_hook=refine_inputs_hook,
                    scale_replicated_inputs=scale_replicated_inputs,
                )

                if ep_size > 1:
                    local_expert_size = self.n_routed_experts // ep_size
                    reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                    global_group_sizes = group_sizes

                    x, _local_sorted_indices, group_sizes, selected_experts = local_permute(
                        x,
                        global_group_sizes[None, :],
                        local_expert_size,
                        shard_index=expert_shard_id,
                        is_offset=True,
                        global_sorted_experts=selected_experts,
                        use_custom_sort_vjp=True,
                    )

            layer_w0 = grouped_matmul(x, wi_kernel, group_sizes, **gmm_kws)

            layer_w0 = checkpoint_name(layer_w0, "mlp_gate")
            if wi_bias is not None:
                layer_w0 = layer_w0 + wi_bias[selected_experts]

            layer_w1 = grouped_matmul(x, wu_kernel, group_sizes, **gmm_kws)

            layer_w1 = checkpoint_name(layer_w1, "mlp_up")
            if wu_bias is not None:
                layer_w1 = layer_w1 + wu_bias[selected_experts]

            intermediate_layer = ffn_activation(layer_w0, layer_w1)

            intermediate_output = grouped_matmul(intermediate_layer, wd_kernel, group_sizes, **gmm_kws)
            intermediate_output = checkpoint_name(intermediate_output, "mlp_down")

            # TP reduction: psum_scatter to shard output across TP on hidden dimension
            # This matches output_ps = [DP, EMPTY, TP]
            if tp_size > 1:
                intermediate_output = jax.lax.psum_scatter(
                    intermediate_output,
                    tensor_axis_name,
                    scatter_dimension=1,
                    tiled=True,
                )

            if wd_bias is not None:
                intermediate_output = intermediate_output + wd_bias[selected_experts]

            if self.config.use_ring_of_experts:
                # No need to mask - intermediate_output was already sliced to valid size
                # If needed for unpermute shape matching, pad back to expected size
                expected_size = sorted_selected_experts.shape[0]
                current_size = intermediate_output.shape[0]
                if current_size < expected_size:
                    padding = jnp.zeros(
                        (expected_size - current_size, intermediate_output.shape[1]), dtype=intermediate_output.dtype
                    )
                    intermediate_output = jnp.concatenate([intermediate_output, padding], axis=0)

                output = unpermute(
                    intermediate_output,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_custom_sort_vjp=self.config.use_custom_sort_vjp,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )

                output = jnp.reshape(output, (-1, sequence_length, HD))
                output = jax.lax.psum_scatter(output, expert_axis_name, scatter_dimension=0, tiled=True)

            else:
                if ep_size > 1:
                    original_inputs_first_dim = batch_size * sequence_length * self.num_experts_per_tok

                    if sorted_selected_experts.shape[0] != original_inputs_first_dim:
                        raise ValueError("original_inputs_first_dim does not match the original tensor shape!")
                    output_shape = jnp.zeros((original_inputs_first_dim, HD // tp_size), dtype=intermediate_output.dtype)

                    input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(
                        reshaped_group_sizes,
                        expert_shard_id,
                        ep_size,
                        is_batch_sharded=False,
                    )

                    intermediate_output = jax.lax.ragged_all_to_all(
                        intermediate_output,
                        output_shape,
                        input_offsets,
                        send_sizes,
                        output_offsets,
                        recv_sizes,
                        axis_name=expert_axis_name,
                    )

                output = unpermute(
                    intermediate_output,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_custom_sort_vjp=True,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )
            return output

        # print(
        #     wi_kernel.shape,
        #     wikps,
        #     wi_bias.shape,
        #     wibps,
        #     wu_kernel.shape,
        #     wukps,
        #     wu_bias.shape,
        #     wubps,
        #     wd_kernel.shape,
        #     wdkps,
        #     wd_bias.shape,
        #     wdbps,
        # )

        output = _sparse_call(
            hidden_state,
            gate_logits,
            wi_kernel,
            wu_kernel,
            wd_kernel,
            wi_bias,
            wu_bias,
            wd_bias,
        )

        # Reshard output back to original 5D mesh for compatibility with rest of model
        # This ensures the output can be used in residual connections
        original_output_ps = self.partition_manager.resolve(
            axes=[DP, EMPTY, TP] if not self.config.use_expert_tensor_mode else [DP, EMPTY, EMPTY],
            mode=MODE_TRAIN,
            shape=output.shape,
        )
        output = jax.lax.with_sharding_constraint(output, jax.sharding.NamedSharding(self.mesh, original_output_ps))

        return output, prein_gate_logits

    def moe_call(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        wi_kernel: jax.Array,  # [E, H, M]
        wu_kernel: jax.Array,  # [E, H, M]
        wd_kernel: jax.Array,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, H]
        wu_bias: jax.Array | None = None,  # [E, H]
        wd_bias: jax.Array | None = None,  # [E, M]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        output_metrics: bool = False,
        layer_idx: int | None = None,
    ):
        """Wrapper for fused MoE call with automatic hook configuration.

        This method dispatches to either standard or fused MoE based on config, and
        automatically configures hooks based on the routing strategy to ensure correct
        expert weight handling.

        **Hook Auto-Configuration:**
            Before calling the fused MoE path, this method automatically configures
            default hooks for the routing strategy if they're not already set by the user:

            - **TOP_K**: Normalizes weights by their sum (softmax-like distribution).
            - **TOP_K_NDIV**: Passes weights through unchanged (raw logit values).
            - **SWITCH**: Enforces hard assignment with weight = 1.0.
            - **EMPTY_CHOICE**: Uniform weights across expert selections.
            - **HASH**: Uniform weights for deterministic assignments.

            Each strategy gets an appropriate default `select_hook` that ensures
            correct weight handling without requiring manual setup. Users can override
            defaults by setting custom hooks on `self.moe_hooks` before calling the layer.

        Args:
            hidden_state: Input tensor. Shape: [B, S, H].
            gate_layer: Router/gate module mapping H -> E (produces logits).
            expert_layer: Expert layer module.
            wi_kernel: Expert W_i (down/first) kernel. Shape: [E, H, M].
            wu_kernel: Expert W_u (up/second) kernel. Shape: [E, H, M].
            wd_kernel: Expert W_d (output/down) kernel. Shape: [E, M, H].
            wi_bias: Optional bias for W_i. Shape: [E, H].
            wu_bias: Optional bias for W_u. Shape: [E, H].
            wd_bias: Optional bias for W_d. Shape: [E, M].
            ffn_activation: Optional custom activation combining (w0, w1) -> output.
            reform_router_probs_fn: Optional function to modify router probabilities
                (used in standard MoE mode only).
            act_fn: Activation function used when `ffn_activation` is not provided.
            output_metrics: Whether to return metrics in standard MoE mode.

        Returns:
            Tuple of (output, logits) where:
            - output: MoE layer output. Shape: [B, S, H].
            - logits: Router logits for auxiliary loss computation. Shape: [B*S, E].
        """
        self._configure_hooks_for_routing_strategy()
        with self.auto_expert_mesh:
            match self.module_moe_method:
                case MoEMethods.STANDARD_MOE:
                    logger.warn_once(
                        "You are using MoEMethods.STANDARD_MOE which is not really recommended please switch to FUSED_MOE"
                    )
                    return self._moe_call_standard(
                        gate_layer=gate_layer,
                        expert_layer=expert_layer,
                        hidden_state=hidden_state,
                        output_metrics=output_metrics,
                        validate_inputs=False,
                        apply_capacity_constraint=False,
                        reform_router_probs_fn=reform_router_probs_fn,
                        layer_idx=layer_idx,
                    )
                case MoEMethods.FUSED_MOE:
                    return self._sparse_moe_call(
                        hidden_state=hidden_state,
                        gate_layer=gate_layer,
                        wi_kernel=wi_kernel,
                        wu_kernel=wu_kernel,
                        wd_kernel=wd_kernel,
                        wi_bias=wi_bias,
                        wu_bias=wu_bias,
                        wd_bias=wd_bias,
                        act_fn=act_fn,
                        ffn_activation=ffn_activation,
                    )
                case MoEMethods.DENSE_MOE:
                    logger.warn_once(
                        "You are using MoEMethods.DENSE_MOE which is not really recommended please switch to FUSED_MOE"
                    )
                    return self._moe_call_dense(
                        hidden_state=hidden_state,
                        gate_layer=gate_layer,
                        wi_kernel=wi_kernel,
                        wu_kernel=wu_kernel,
                        wd_kernel=wd_kernel,
                        wi_bias=wi_bias,
                        wu_bias=wu_bias,
                        wd_bias=wd_bias,
                        act_fn=act_fn,
                        ffn_activation=ffn_activation,
                    )
                case _:
                    raise NotImplementedError()

    def _moe_call_dense(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: nn.Module,
        wi_kernel: jax.Array,  # [E, H, M]
        wu_kernel: jax.Array,  # [E, H, M]
        wd_kernel: jax.Array,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, M]
        wu_bias: jax.Array | None = None,  # [E, M]
        wd_bias: jax.Array | None = None,  # [E, H]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        capacity_factor: float | None = None,
        output_metrics: bool = False,
    ):
        """Dense MoE path using per-token batched matmuls instead of ragged grouping."""
        self._configure_hooks_for_routing_strategy()
        hooks = self.moe_hooks

        hidden_state = hidden_state.astype(self.dtype)
        if hooks.before_gate is not None:
            hidden_state = hooks.before_gate(hidden_state)

        batch_size, seq_len, hidden_dim = hidden_state.shape
        tokens = batch_size * seq_len
        hidden_flat = hidden_state.reshape(tokens, hidden_dim)

        prein_gate_logits = gate_layer(hidden_flat)
        gate_logits = prein_gate_logits
        if hooks.after_gate is not None:
            gate_logits = hooks.after_gate(gate_logits)

        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        if hooks.before_topk is not None:
            router_probs = hooks.before_topk(router_probs)

        selected_weights, selected_experts = get_experts_location(
            gate_logits=router_probs,
            pre_bias_logits=None,
            select_hook=hooks.select_hook,
            refine_weights_hook=hooks.refine_weights_hook,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        weights = selected_weights.astype(self.dtype)
        experts = selected_experts.astype(jnp.int32)

        if capacity_factor is not None and capacity_factor > 0:
            weights_shaped = weights.reshape(batch_size, seq_len, self.num_experts_per_tok)
            experts_shaped = experts.reshape(batch_size, seq_len, self.num_experts_per_tok)
            weights_shaped = self._apply_capacity_mask(experts_shaped, weights_shaped, capacity_factor)
            weights = weights_shaped.reshape(tokens, self.num_experts_per_tok)

        weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
        weights = jnp.where(weight_sum > 0, weights / weight_sum, weights)

        if ffn_activation is None:

            def ffn_activation(x0: jax.Array, x1: jax.Array) -> jax.Array:
                return act_fn(x0) * x1

        precision = getattr(self, "precision", None)
        hidden_expanded = hidden_flat[:, None, :]

        wi_sel = jnp.take(wi_kernel.astype(self.dtype), experts, axis=0)
        w0 = jnp.einsum("tkh,tkhm->tkm", hidden_expanded, wi_sel, precision=precision)
        if wi_bias is not None:
            w0 = w0 + jnp.take(wi_bias.astype(self.dtype), experts, axis=0)

        wu_sel = jnp.take(wu_kernel.astype(self.dtype), experts, axis=0)
        w1 = jnp.einsum("tkh,tkhm->tkm", hidden_expanded, wu_sel, precision=precision)
        if wu_bias is not None:
            w1 = w1 + jnp.take(wu_bias.astype(self.dtype), experts, axis=0)

        intermediate = ffn_activation(w0, w1)
        if hooks.before_wo is not None:
            intermediate = hooks.before_wo(intermediate)

        wd_sel = jnp.take(wd_kernel.astype(self.dtype), experts, axis=0)
        outputs = jnp.einsum("tkm,tkmh->tkh", intermediate, wd_sel, precision=precision)
        if wd_bias is not None:
            outputs = outputs + jnp.take(wd_bias.astype(self.dtype), experts, axis=0)

        if hooks.after_wo is not None:
            outputs = hooks.after_wo(outputs)

        if hooks.before_combine is not None:
            outputs, weights = hooks.before_combine(outputs, weights)

        combined = jnp.sum(outputs * weights[..., None], axis=1)
        output = combined.reshape(batch_size, seq_len, hidden_dim)

        if hooks.finalize_output is not None:
            output = hooks.finalize_output(output)

        if output_metrics:
            expert_mask = (weights > 0).astype(self.dtype)
            expert_loads = jnp.bincount(
                experts.reshape(-1),
                weights=expert_mask.reshape(-1),
                length=self.n_routed_experts,
            ).astype(self.dtype)
            metrics = self._compute_metrics(
                router_logits=prein_gate_logits,
                router_probs=router_probs,
                selected_experts=experts,
                selected_weights=weights,
                expert_loads=expert_loads,
            )
            return output, metrics

        return output, prein_gate_logits

    def _configure_hooks_for_routing_strategy(self) -> None:
        """Configure default hooks based on the current routing strategy.

        This method ensures each routing strategy has appropriate hook configuration
        without requiring manual setup. Only sets hooks if they haven't been explicitly
        configured by the user.

        **Hook Configuration by Strategy:**

            TOP_K: Sets `select_hook` to normalize weights by their sum.
                Ensures expert weights sum to 1.0 for proper weighted combination.

            TOP_K_NDIV: Sets `select_hook` to pass through weights unchanged.
                Uses raw logit values without normalization.

            SWITCH: Sets `select_hook` to enforce hard assignment (weight = 1.0).
                Only one expert gets non-zero weight.

            EMPTY_CHOICE: Sets `select_hook` to normalize per-expert selections.
                Each expert receives equal contribution from selected tokens.

            HASH: Sets `select_hook` to uniform weight distribution.
                All assigned experts get equal weight (1/k).
        """
        # Only set default refine_weights_hook if one wasn't already configured by the user.
        if self.moe_hooks.refine_weights_hook is not None:
            return

        refine_weights_hook = None
        if self.routing_strategy == MoeRoutingStrategy.TOP_K:
            # TOP_K: Normalize weights by their sum (softmax-like)
            if self.moe_hooks.select_hook is None:

                def normalize_selected_weights(weights: jax.Array) -> jax.Array:
                    """Normalize top-k expert weights by their sum.

                    Ensures weights for each token sum to 1.0, creating a proper
                    probability distribution over selected experts.
                    """
                    # Match HF Mixtral-style behavior: for top-1 routing keep the selected
                    # probability as-is; for top-k>1 normalize by the sum across selected experts.
                    if weights.shape[-1] <= 1:
                        return weights
                    return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)

                refine_weights_hook = normalize_selected_weights

        elif self.routing_strategy == MoeRoutingStrategy.TOP_K_NDIV:
            # TOP_K_NDIV: Use weights as-is (raw logits, no normalization)
            if self.moe_hooks.select_hook is None:

                def passthrough_weights(weights: jax.Array) -> jax.Array:
                    """Pass through weights unchanged.

                    For TOP_K_NDIV routing, weights are used as raw logit values
                    without normalization. This allows unnormalized combinations.
                    """
                    return weights

                refine_weights_hook = passthrough_weights

        elif self.routing_strategy == MoeRoutingStrategy.SWITCH:
            # SWITCH: Hard assignment - single expert gets weight 1.0, others 0.0
            if self.moe_hooks.select_hook is None:

                def hard_assignment_weights(weights: jax.Array) -> jax.Array:
                    """Enforce hard assignment for SWITCH routing.

                    Only one expert per token is selected (top-1). This hook ensures
                    the weight is exactly 1.0, creating a hard (non-differentiable)
                    expert assignment.
                    """
                    return jnp.ones_like(weights)

                refine_weights_hook = hard_assignment_weights

        elif self.routing_strategy == MoeRoutingStrategy.EMPTY_CHOICE:
            # EMPTY_CHOICE: Expert-driven selection - normalize per expert
            if self.moe_hooks.select_hook is None:

                def expert_choice_weights(weights: jax.Array) -> jax.Array:
                    """Normalize weights for Expert Choice routing.

                    In Expert Choice routing, each expert selects its own top-k tokens.
                    This hook ensures proper weight distribution across the selected
                    tokens for each expert.
                    """
                    # For expert choice, normalize differently - each expert's selections
                    # should have equal contribution
                    num_experts_selected = weights.shape[-1]
                    return jnp.ones_like(weights) / jnp.maximum(num_experts_selected, 1)

                refine_weights_hook = expert_choice_weights

        elif self.routing_strategy == MoeRoutingStrategy.HASH:
            # HASH: Deterministic routing - uniform weights for all assigned experts
            if self.moe_hooks.select_hook is None:

                def uniform_weights(weights: jax.Array) -> jax.Array:
                    """Uniform weights for hash-based routing.

                    In hash-based routing, tokens are deterministically assigned to experts
                    based on token ID. Each expert in the assignment group gets equal weight.
                    """
                    num_experts_per_token = weights.shape[-1]
                    return jnp.ones_like(weights) / jnp.maximum(num_experts_per_token, 1)

                refine_weights_hook = uniform_weights

        if refine_weights_hook is not None:
            self.moe_hooks = self.moe_hooks.replace(refine_weights_hook=refine_weights_hook)

    def _moe_call_standard(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        apply_capacity_constraint: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        layer_idx: int | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """Standard MoE forward pass: routing, permutation, expert computation, and combining.

        This method uses the MoeFusedHooks system to allow custom interventions at key
        points during execution:

        **Hook Integration:**
            - `before_gate`: Applied before gate/router computation.
            - `after_gate`: Applied after gate logits computation.
            - `before_topk`: Applied before expert selection (top-k).
            - `refine_weights_hook`: Refines expert weights after selection.
            - `refine_inputs_hook`: Refines token representations before expert computation.
            - `before_combine`: Applied before combining expert outputs.
            - `finalize_output`: Applied to the final output.

        Args:
            gate_layer: Router module mapping hidden states to expert logits.
            expert_layer: Expert layer module for computing expert outputs.
            hidden_state: Input tensor. Shape: [B, S, H].
            output_metrics: Whether to return detailed MoE metrics.
            validate_inputs: Whether to validate input shapes.
            apply_capacity_constraint: Whether to apply capacity constraints.
            reform_router_probs_fn: Optional function to modify router probabilities.

        Returns:
            Tuple of (output, metrics_or_logits) where:
            - output: MoE layer output. Shape: [B, S, H].
            - metrics_or_logits: MoeMetrics if output_metrics=True, else router_logits.
        """
        self._configure_hooks_for_routing_strategy()

        hooks = self.moe_hooks

        hidden_state = hidden_state.astype(self.dtype)
        if hooks.before_gate is not None:
            hidden_state = hooks.before_gate(hidden_state)

        batch_size, seq_len, hidden_size = hidden_state.shape
        hidden_state_flat = hidden_state.reshape(-1, hidden_size)

        router_logits = gate_layer(hidden_state_flat).astype(jnp.promote_types(self.dtype, jnp.float32))

        # Store original logits BEFORE any hooks - used for expert selection (matching HF behavior).
        prein_gate_logits = router_logits

        # after_gate hook produces scattered probs for aux loss/logging, but we use original logits for selection.
        if hooks.after_gate is not None:
            router_probs = hooks.after_gate(router_logits)
        else:
            router_probs = jax.nn.softmax(router_logits, axis=-1)

        if reform_router_probs_fn is not None:
            router_probs = reform_router_probs_fn(router_probs)

        if hooks.before_topk is not None:
            router_probs = hooks.before_topk(router_probs)

        if validate_inputs:
            self._validate_routing_inputs(hidden_state, router_logits)

        # Use original logits for expert selection (top-k on logits, then softmax on k selected via refine_weights_hook).
        # This matches HuggingFace behavior where top-k is done on pre-softmax logits.
        selected_weights, selected_experts = get_experts_location(
            gate_logits=prein_gate_logits,
            pre_bias_logits=None,
            select_hook=hooks.select_hook,
            refine_weights_hook=hooks.refine_weights_hook,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        # Detailed logging for debugging
        if layer_idx is not None:
            # Get top-k logits (before softmax) for comparison
            top_k_logits_pre, _ = jax.lax.top_k(prein_gate_logits, self.num_experts_per_tok)
            jax.debug.print("  [ED Router L{}] logits[0]: {}", layer_idx, prein_gate_logits[0])
            jax.debug.print(
                "  [ED Router L{}] top_idx[0]: {}, top_logits[0]: {}",
                layer_idx,
                selected_experts[0],
                top_k_logits_pre[0],
            )
            jax.debug.print(
                "  [ED Router L{}] top_weights[0]: {} (sum={})",
                layer_idx,
                selected_weights[0],
                selected_weights[0].sum(),
            )
            jax.debug.print("  [ED Experts L{}] input[0,:5]: {}", layer_idx, hidden_state_flat[0, :5])

        if apply_capacity_constraint:
            selected_experts, selected_weights = self._apply_capacity_constraint(selected_experts, selected_weights)

        if hooks.refine_inputs_hook is not None:
            hidden_state_flat = hooks.refine_inputs_hook(
                hidden_state_flat,
                selected_weights,
                (batch_size, seq_len, hidden_size),
            )

        (
            sorted_inputs,
            sort_order,
            group_sizes,
            sorted_experts,
        ) = self._replicate_and_sort_tokens(hidden_state_flat, selected_experts)

        out_sorted = expert_layer(sorted_inputs, group_sizes, sorted_experts)
        out_unsorted = sort_activations(out_sorted, jnp.argsort(sort_order))
        out_unflat = out_unsorted.reshape(batch_size * seq_len, self.num_experts_per_tok, hidden_size)

        if hooks.before_combine is not None:
            out_unflat, selected_weights = hooks.before_combine(out_unflat, selected_weights)

        output = jnp.sum(out_unflat * selected_weights[..., None], axis=1).reshape(batch_size, seq_len, hidden_size)

        # Log expert output
        if layer_idx is not None:
            jax.debug.print("  [ED Experts L{}] output[0,:5]: {}", layer_idx, output.reshape(-1, hidden_size)[0, :5])

        if hooks.finalize_output is not None:
            output = hooks.finalize_output(output)

        if output_metrics:
            metrics = self._compute_metrics(
                router_logits,
                router_probs,
                selected_experts,
                selected_weights,
                group_sizes,
            )
            return output, metrics
        return output, router_logits

    @abstractmethod
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden_dim"],
        **kwargs,
    ) -> tuple[Float[Array, "batch seq hidden_dim"], MoeMetrics]:
        """Performs the forward pass of the MoE module.

        Subclasses must implement this method to define the specific logic of their
        MoE layer.

        Args:
            hidden_states: The input tensor.
            **kwargs: Additional keyword arguments that may be required by the
                specific implementation.

        Returns:
            A tuple containing:
                - output: The output tensor from the MoE layer.
                - metrics: A `MoeMetrics` object containing metrics and auxiliary losses.
        """
        pass
