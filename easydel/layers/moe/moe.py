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
from eformer import common_types
from eformer.common_types import DynamicShardingAxes
from ejkernel.modules import grouped_matmul
from flax import nnx as nn
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseConfig

from .utils import (
    MoeFusedHooks,
    MoeFusedPolicy,
    MoeLoadBalancingStrategy,
    MoeMetrics,
    MoeRoutingStrategy,
    get_all_to_all_params,
    local_permute,
    permute,
    resolve_eformer_axis,
    sort_activations,
    unpermute,
)

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

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


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


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
        moe_policy: MoeFusedPolicy | None = None,
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
            moe_policy: Configuration policy for fused MoE operations. If None,
                uses default MoeFusedPolicy.
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
        self.moe_policy = MoeFusedPolicy() if moe_policy is None else moe_policy
        self.moe_hooks = MoeFusedHooks() if moe_hooks is None else moe_hooks
        self.dtype = getattr(self, "dtype", jnp.bfloat16)

    def _route(
        self,
        router_probs: jax.Array,
        routing_strategy: MoeRoutingStrategy | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Selects experts for each token based on the specified routing strategy."""
        return self._route_sharded(router_probs, routing_strategy or self.routing_strategy)

    def _route_sharded(
        self,
        router_probs: Float[Array, "batch_seq num_experts"],
        strategy: MoeRoutingStrategy,
    ) -> tuple[Float[Array, "batch_seq k"], Int[Array, "batch_seq k"]]:
        """Performs sharded routing of tokens to experts across devices using shard_map."""
        pmag = self.partition_manager

        if router_probs.ndim == 2:
            pspec = pmag.resolve(axes=[BATCH, EMPTY], mode=MODE_TRAIN, shape=router_probs.shape)
            in_specs = pspec
            out_specs = (pspec, pspec)
        elif router_probs.ndim == 3:
            pspec = pmag.resolve(axes=[BATCH, EMPTY, EMPTY], mode=MODE_TRAIN, shape=router_probs.shape)
            in_specs = pspec
            out_specs = (pspec, pspec)
        else:
            in_specs = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)
            out_specs = (in_specs, in_specs)

        @partial(shard_map, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        def sharded_route(router_probs_):
            return self._route_local(router_probs_, strategy)

        return sharded_route(router_probs)

    def _route_local(
        self, router_probs: Float[Array, "batch_seq num_experts"], strategy: MoeRoutingStrategy
    ) -> tuple[Float[Array, "batch_seq k"], Int[Array, "batch_seq k"]]:
        """Implements the routing logic on a local device shard."""
        if strategy == MoeRoutingStrategy.TOP_K:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
            selected_weights /= selected_weights.sum(-1, keepdims=True)
        elif strategy == MoeRoutingStrategy.TOP_K_NDIV:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
        elif strategy == MoeRoutingStrategy.SWITCH:
            selected_experts = jnp.argmax(router_probs, axis=-1, keepdims=True)
            selected_weights = jnp.take_along_axis(router_probs, selected_experts, axis=-1)
        elif strategy == MoeRoutingStrategy.EMPTY_CHOICE:
            k = router_probs.shape[0] // self.n_routed_experts
            selected_weights, selected_experts = jax.lax.top_k(router_probs.T, k=k)
            selected_weights = selected_weights.T
            selected_experts = selected_experts.T
        elif strategy == MoeRoutingStrategy.HASH:
            token_ids = jnp.arange(router_probs.shape[0])
            selected_experts = (token_ids % self.n_routed_experts)[..., None]
            selected_weights = jnp.ones_like(selected_experts, dtype=router_probs.dtype)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        return selected_weights, selected_experts

    def _replicate_and_sort_tokens(
        self,
        inputs_flat: jax.Array,
        selected_experts: jax.Array,
        use_custom_sort_vjp: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Replicates tokens k times (once per selected expert) and sorts by expert ID."""
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
        topk_indices: jax.Array,
        weights: jax.Array,
        capacity_factor: float,
    ) -> jax.Array:
        """Applies capacity constraints to expert assignments by masking overflow tokens."""
        B, S, k = topk_indices.shape
        tokens_per_batch = S * k
        cap = int(max(jnp.ceil(tokens_per_batch / self.n_routed_experts) * capacity_factor, capacity_factor))
        expert_mask = jax.nn.one_hot(topk_indices, num_classes=self.n_routed_experts, dtype=jnp.int32)
        fused = expert_mask.reshape(B, S * k, self.n_routed_experts)
        counts = jnp.cumsum(fused, axis=1)
        counts = counts.reshape(B, S, k, self.n_routed_experts)
        keep = (counts <= cap).astype(weights.dtype)
        keep_for_slot = jnp.sum(keep, axis=-1)
        return weights * keep_for_slot

    def _expert_group_mask(self, gate_logits: jax.Array, n_groups: int, topk_groups: int) -> jax.Array:
        """Creates a mask for hierarchical routing where experts are organized into groups."""
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
        """Applies expert parallel sharding to a tensor for distributed training."""
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
        """Returns the partition spec for the gate/router layer weights."""
        pmag = self.partition_manager
        return pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=weight_shape)

    def _get_gate_layer_bias_sharding(self, bias_shape: tuple) -> PartitionSpec:
        """Returns the partition spec for the gate/router layer bias."""
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
        """Creates a boolean mask for tokens assigned to a specific expert."""
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
        top_kfn: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None,
        wmodif_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        refine_idsfn: typing.Callable[[jax.Array, jax.Array, tuple[int]], jax.Array] | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
    ):
        """Fused MoE path using grouped matmul and shard_map.

        This implementation routes tokens to experts, permutes them to an
        expert-grouped layout, applies expert FFNs via grouped matmul kernels,
        and then unpermutes/combines outputs. It supports both ring-of-experts
        and all-to-all expert-parallel communication depending on configuration
        and mesh sizes.

        Args:
            hidden_state: Input tensor. Shape: [B, S, H].
            gate_layer: Router module mapping H -> E (produces logits).
            wi_kernel: Expert W_i kernel. Shape: [E, H, M].
            wu_kernel: Expert W_u kernel. Shape: [E, H, M].
            wd_kernel: Expert W_d kernel. Shape: [E, M, H].
            wi_bias: Optional bias for W_i. Shape: [E, H].
            wu_bias: Optional bias for W_u. Shape: [E, H].
            wd_bias: Optional bias for W_d. Shape: [E, M].
            ffn_activation: Optional fused activation taking (w0, w1) -> act.
            top_kfn: Optional custom top-k selector `(gate_logits, pre_bias_logits, k)`.
            wmodif_fn: Optional weight post-processing function.
            refine_idsfn: Optional function to refine token IDs/inputs before routing.
            act_fn: Activation used when `ffn_activation` is not provided.

        Returns:
            Tuple `(output, None)` where `output` has shape [B, S, H] and the
            second element is a placeholder for parity with other call paths.
        """
        BS, SQLN, HD = hidden_state.shape
        gate_logits = gate_layer(hidden_state)
        gate_logits = jax.nn.softmax(gate_logits.astype("f4"), axis=-1).astype(gate_logits.dtype)

        partition_manager = self.partition_manager
        expert_axis_name = resolve_eformer_axis(EP, partition_manager)
        tensor_axis_name = resolve_eformer_axis(TP, partition_manager)
        fsdp_axis_name = resolve_eformer_axis(FSDP, partition_manager)
        sp_axis_name = resolve_eformer_axis(SP, partition_manager)
        ep_size = self.mesh.shape.get(expert_axis_name, 1)
        tp_size = self.mesh.shape.get(tensor_axis_name, 1)
        fsdp_size = self.mesh.shape.get(fsdp_axis_name, 1)
        sp_size = self.mesh.shape.get(sp_axis_name, 1)
        comb_size = fsdp_size * sp_size
        input_ps = partition_manager.resolve(axes=[DP, EMPTY, EMPTY], mode=MODE_TRAIN, shape=hidden_state.shape)
        output_ps = partition_manager.resolve(axes=[DP, EMPTY, TP], mode=MODE_TRAIN, shape=(BS, SQLN, HD))
        glps = partition_manager.resolve(
            axes=[DP, EMPTY, EMPTY],
            mode=MODE_TRAIN,
            shape=(BS, SQLN, self.n_routed_experts),
        )

        if ffn_activation is None:

            def ffn_activation(x0: jax.Array, x1: jax.Array) -> jax.Array:
                return act_fn(x0) * x1

        if self.config.use_expert_tensor_mode:
            wikps = partition_manager.resolve(dynamic_axes=ExpertTensorParallel, shape=wi_kernel.shape)
            wukps = partition_manager.resolve(dynamic_axes=ExpertTensorParallel, shape=wu_kernel.shape)
            wdkps = partition_manager.resolve(dynamic_axes=ExpertTensorParallel, shape=wd_kernel.shape)
        else:
            wikps = partition_manager.resolve(axes=[EP, [SP, FSDP], TP], mode=MODE_TRAIN, shape=wi_kernel.shape)
            wukps = partition_manager.resolve(axes=[EP, [SP, FSDP], TP], mode=MODE_TRAIN, shape=wu_kernel.shape)
            wdkps = partition_manager.resolve(axes=[EP, TP, [SP, FSDP]], mode=MODE_TRAIN, shape=wd_kernel.shape)

        wibps = None
        wubps = None
        wdbps = None
        if wi_bias is not None:
            wibps = partition_manager.resolve(axes=[EP, [SP, FSDP]], mode=MODE_TRAIN, shape=wi_bias.shape)
        if wu_bias is not None:
            wubps = partition_manager.resolve(axes=[EP, [SP, FSDP]], mode=MODE_TRAIN, shape=wu_bias.shape)
        if wd_bias is not None:
            wdbps = partition_manager.resolve(axes=[EP, EMPTY], mode=MODE_TRAIN, shape=wd_bias.shape)

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(input_ps, glps, wikps, wukps, wdkps, wibps, wubps, wdbps),
            out_specs=(output_ps, None),
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
                    jax.lax.all_gather(z, axis_name=expert_axis_name, tiled=True) for z in (x, gate_logits)
                )

                # "Route" tokens within each shard.
                experts_per_shard = self.n_routed_experts // ep_size
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=experts_per_shard * expert_shard_id,
                    top_kfn=top_kfn,
                    wmodif_fn=wmodif_fn,
                    refine_idsfn=refine_idsfn,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                )

                group_sizes = group_sizes[:experts_per_shard]  # only the local experts
                mask = jnp.arange(x.shape[0]) < jnp.sum(group_sizes)
                x = jnp.where(mask[:, None], x, 0)
            else:
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=None,
                    top_kfn=top_kfn,
                    wmodif_fn=wmodif_fn,
                    refine_idsfn=refine_idsfn,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
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

            layer_w0 = grouped_matmul(
                x,
                wi_kernel,
                group_sizes,
                preferred_element_type=jnp.bfloat16,
                platform="xla",
            )
            if comb_size > 1:
                layer_w0 = jax.lax.psum(layer_w0, (fsdp_axis_name, sp_axis_name))

            if wi_bias is not None:
                layer_w0 = layer_w0 + wi_bias[selected_experts]

            layer_w1 = grouped_matmul(
                x,
                wu_kernel,
                group_sizes,
                preferred_element_type=jnp.bfloat16,
                platform="xla",
            )

            if comb_size > 1:
                layer_w1 = jax.lax.psum(layer_w1, (fsdp_axis_name, sp_axis_name))

            if wu_bias is not None:
                layer_w1 = layer_w1 + wu_bias[selected_experts]

            intermediate_layer = ffn_activation(layer_w0, layer_w1)

            intermediate_output = grouped_matmul(
                intermediate_layer,
                wd_kernel,
                group_sizes,
                preferred_element_type=jnp.bfloat16,
                platform="xla",
            )

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
                mask = jnp.arange(intermediate_output.shape[0]) < jnp.sum(group_sizes)
                intermediate_output = jnp.where(mask[:, None], intermediate_output, 0)

                output = unpermute(
                    intermediate_output,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_custom_sort_vjp=self.config.use_custom_sort_vjp,
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
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )
            return output, None

        output, logits = _sparse_call(
            hidden_state,
            gate_logits,
            wi_kernel,
            wu_kernel,
            wd_kernel,
            wi_bias,
            wu_bias,
            wd_bias,
        )
        return output, logits

    def _moe_call_fused(
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
        top_kfn: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None,
        wmodif_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        refine_idsfn: typing.Callable[[jax.Array, jax.Array, tuple[int]], jax.Array] | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        output_metrics: bool = False,
    ):
        """Wrapper for fused MoE call with shard_map for training/inference."""
        if self.config.moe_method == "standard_moe":
            return self._moe_call_standard(
                gate_layer=gate_layer,
                expert_layer=expert_layer,
                hidden_state=hidden_state,
                output_metrics=output_metrics,
                validate_inputs=False,
                apply_capacity_constraint=False,
                reform_router_probs_fn=reform_router_probs_fn,
            )
        elif self.config.moe_method == "fused_moe":
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
                top_kfn=top_kfn,
                wmodif_fn=wmodif_fn,
                refine_idsfn=refine_idsfn,
                ffn_activation=ffn_activation,
            )

    def _moe_call_standard(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        apply_capacity_constraint: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """Standard MoE forward pass: routing, permutation, expert computation, and combining."""
        batch_size, seq_len, hidden_size = hidden_state.shape
        hidden_state_flat = hidden_state.reshape(-1, hidden_size)
        router_logits = gate_layer(hidden_state_flat).astype(jnp.promote_types(self.dtype, jnp.float32))
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        if reform_router_probs_fn is not None:
            router_probs = reform_router_probs_fn(router_probs)

        if validate_inputs:
            self._validate_routing_inputs(hidden_state, router_logits)

        selected_weights, selected_experts = self._route_local(router_probs, self.routing_strategy)
        if apply_capacity_constraint:
            selected_experts, selected_weights = self._apply_capacity_constraint(selected_experts, selected_weights)

        sorted_inputs, sort_order, group_sizes, _ = self._replicate_and_sort_tokens(hidden_state_flat, selected_experts)
        out_sorted = expert_layer(sorted_inputs, group_sizes)
        out_unsorted = sort_activations(out_sorted, jnp.argsort(sort_order))
        out_unflat = out_unsorted.reshape(batch_size * seq_len, self.num_experts_per_tok, hidden_size)
        output = jnp.sum(out_unflat * selected_weights[..., None], axis=1).reshape(batch_size, seq_len, hidden_size)
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
