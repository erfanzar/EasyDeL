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
from __future__ import annotations

import enum
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig


class MoeRoutingStrategy(enum.Enum):
    """Defines the available strategies for routing tokens to experts in an MoE layer.

    Attributes:
        TOP_K: Standard top-k routing, where each token is routed to the k experts
            with the highest router scores.
        TOP_K_NDIV: Top-k routing without dividing the weights by their sum.
        SWITCH: Switch Transformer-style routing, where each token is routed to
            only the top-1 expert.
        EXPERT_CHOICE: Expert Choice routing, where each expert selects the top-k
            tokens with the highest scores for that expert.
        HASH: A simple hashing-based routing for debugging or baseline comparison.
    """

    TOP_K = "top_k"
    TOP_K_NDIV = "top_k_ndiv"
    SWITCH = "switch"
    EXPERT_CHOICE = "expert_choice"
    HASH = "hash"


class MoeLoadBalancingStrategy(enum.Enum):
    """Defines the available strategies for calculating the load balancing loss.

    Attributes:
        STANDARD: A common load balancing loss based on the product of expert
            loads and mean router probabilities.
        SWITCH_TRANSFORMER: The load balancing loss used in the Switch
            Transformer paper.
        EXPERT_CHOICE: A load balancing loss variant suitable for Expert Choice
            routing, often based on the variance of expert loads.
        NONE: No load balancing loss is applied.
    """

    STANDARD = "standard"
    SWITCH_TRANSFORMER = "switch_transformer"
    EXPERT_CHOICE = "expert_choice"
    NONE = "none"


@dataclass
class MoeMetrics:
    """A container for storing metrics and auxiliary losses from an MoE layer.

    Attributes:
        expert_loads: An array representing the number of tokens routed to each
            expert. Shape: (num_experts,).
        router_probs: The probabilities output by the router for each token and
            expert. Shape: (num_tokens, num_experts).
        selected_experts: The indices of the experts selected for each token.
            Shape: (num_tokens, num_experts_per_tok).
        selected_weights: The weights assigned to the selected experts for each
            token. Shape: (num_tokens, num_experts_per_tok).
        load_balancing_loss: The calculated auxiliary loss to encourage balanced
            load across experts.
        router_z_loss: The calculated auxiliary loss to encourage small router
            logits, promoting stability.
        expert_utilization: The fraction of experts that were utilized (i.e.,
            received at least one token).
        routing_entropy: The entropy of the router probabilities, measuring routing
            confidence.
    """

    expert_loads: jax.Array
    router_probs: jax.Array
    selected_experts: jax.Array
    selected_weights: jax.Array
    load_balancing_loss: float | None = None
    router_z_loss: float | None = None
    expert_utilization: float | None = None
    routing_entropy: float | None = None


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros
Initializer = nn.initializers.Initializer


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
        """
        super().__init__()
        self.config = config
        self.mesh = config.mesh
        self.n_routed_experts = n_routed_experts or config.n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok or config.num_experts_per_tok
        self.hidden_size = hidden_size or config.hidden_size
        self.lbl_coef = lbl_coef
        self.rzl_coef = rzl_coef
        self.routing_strategy = routing_strategy
        self.load_balancing_strategy = load_balancing_strategy

    def _route(
        self,
        router_probs: jax.Array,
        routing_strategy: MoeRoutingStrategy | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Selects experts for each token based on the specified routing strategy.

        This method wraps the sharded routing implementation.

        Args:
            router_probs: An array of router probabilities with shape
                `(batch_size * seq_len, num_experts)`.
            routing_strategy: The routing strategy to use. If None, the default
                strategy from the constructor is used.

        Returns:
            A tuple containing:
                - selected_weights: The weights for the selected experts. Shape
                    `(batch_size * seq_len, num_experts_per_tok)`.
                - selected_experts: The indices of the selected experts. Shape
                    `(batch_size * seq_len, num_experts_per_tok)`.
        """
        return self._route_sharded(router_probs, routing_strategy or self.routing_strategy)

    def _route_sharded(self, router_probs: jax.Array, strategy: MoeRoutingStrategy) -> tuple[jax.Array, jax.Array]:
        """Performs sharded routing of tokens to experts.

        This method uses `jax.sharding.shard_map` to apply the local routing
        logic across devices according to the specified sharding configuration.

        Args:
            router_probs: An array of router probabilities with shape
                `(batch_size * seq_len, num_experts)`.
            strategy: The routing strategy to apply.

        Returns:
            A tuple containing:
                - selected_weights: The sharded weights for the selected experts.
                - selected_experts: The sharded indices of the selected experts.
        """
        if router_probs.ndim == 2:
            in_specs = P("dp", None)
            out_specs = (P("dp", None), P("dp", None))
        elif router_probs.ndim == 3:
            in_specs = P("dp", "fsdp", None)
            out_specs = (P("dp", "fsdp", None), P("dp", "fsdp", None))
        else:
            in_specs = P(None)
            out_specs = (P(None), P(None))

        @partial(shard_map, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
        def sharded_route(router_probs_):
            return self._route_local(router_probs_, strategy)

        return sharded_route(router_probs)

    def _route_local(self, router_probs: jax.Array, strategy: MoeRoutingStrategy) -> tuple[jax.Array, jax.Array]:
        """Implements the routing logic on a local device shard.

        Args:
            router_probs: A shard of router probabilities.
            strategy: The routing strategy to apply.

        Returns:
            A tuple containing the selected weights and expert indices for the
            local shard.

        Raises:
            ValueError: If an unknown routing strategy is provided.
        """
        if strategy == MoeRoutingStrategy.TOP_K:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
            selected_weights /= selected_weights.sum(-1, keepdims=True)
        elif strategy == MoeRoutingStrategy.TOP_K_NDIV:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
        elif strategy == MoeRoutingStrategy.SWITCH:
            selected_experts = jnp.argmax(router_probs, axis=-1, keepdims=True)
            selected_weights = jnp.take_along_axis(router_probs, selected_experts, axis=-1)
        elif strategy == MoeRoutingStrategy.EXPERT_CHOICE:
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

    def _permute(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Permutes tokens to group them by their assigned expert.

        This operation is crucial for efficient batch computation by the experts.
        It sorts the tokens based on their expert index, allowing for a single
        batched computation per expert.

        Args:
            hidden_states_flat: A flattened array of input tokens with shape
                `(batch_size * seq_len, hidden_size)`.
            topk_idx_flat: A flattened array of expert indices for each token-
                expert pair, with shape `(batch_size * seq_len * num_experts_per_tok,)`.

        Returns:
            A tuple containing:
                - x_repeat_sort: The permuted hidden states, ready for expert
                    computation.
                - group_sizes: An array indicating how many tokens are assigned to
                    each expert.
                - sort_idx: The indices used for sorting, needed for the un-permutation step.
        """
        return self._permute_sharded(hidden_states_flat, topk_idx_flat)

    def _permute_sharded(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Performs a sharded permutation of tokens.

        This method uses `jax.sharding.shard_map` to apply the local permutation
        logic across devices.

        Args:
            hidden_states_flat: A sharded, flattened array of input tokens.
            topk_idx_flat: A sharded, flattened array of expert indices.

        Returns:
            A tuple containing the sharded permuted hidden states, group sizes,
            and sorting indices.
        """
        x_in_specs = P("dp", None) if hidden_states_flat.ndim == 2 else P("dp", "fsdp", None)
        idx_in_specs = P("dp") if topk_idx_flat.ndim == 1 else P("dp", "fsdp")

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(x_in_specs, idx_in_specs),
            out_specs=(P("dp", None), P("ep"), P("dp")),
            check_rep=False,
        )
        def permute_sharded(x_flat_: jax.Array, topk_idx_flat_: jax.Array):
            return self._permute_local(x_flat_, topk_idx_flat_)

        return permute_sharded(hidden_states_flat, topk_idx_flat)

    def _permute_local(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Implements the permutation logic on a local device shard.

        Args:
            hidden_states_flat: A shard of flattened hidden states.
            topk_idx_flat: A shard of flattened expert indices.

        Returns:
            A tuple containing the locally permuted hidden states, group sizes,
            and sorting indices.
        """
        sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
        x_repeat_sort = jnp.take(hidden_states_flat, sort_idx // self.num_experts_per_tok, axis=0)
        group_sizes = jnp.bincount(topk_idx_flat, length=self.n_routed_experts)
        return x_repeat_sort, group_sizes, sort_idx

    def _unpermute(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Restores the original order of tokens after expert processing.

        This is the inverse operation of `_permute`.

        Args:
            out_repeat_sort: The output from the experts, in sorted (permuted)
                order.
            sort_idx: The sorting indices generated by the `_permute` step.
            original_shape: The original shape of the hidden states before
                flattening, typically `(batch_size, seq_len, hidden_size)`.

        Returns:
            The un-permuted expert outputs, reshaped to match the input structure.
        """
        return self._unpermute_sharded(out_repeat_sort, sort_idx, original_shape)

    def _unpermute_sharded(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Performs a sharded un-permutation of tokens.

        This method uses `jax.sharding.shard_map` to apply the local
        un-permutation logic across devices.

        Args:
            out_repeat_sort: A sharded array of expert outputs in sorted order.
            sort_idx: The sharded sorting indices.
            original_shape: The original shape of the hidden states.

        Returns:
            The sharded, un-permuted expert outputs.
        """
        out_in_specs = P("dp", None) if out_repeat_sort.ndim == 2 else P("dp", "fsdp", None)
        idx_in_specs = P("dp") if sort_idx.ndim == 1 else P("dp", "fsdp")

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(out_in_specs, idx_in_specs),
            out_specs=P("dp", None, None),
            check_rep=False,
        )
        def unpermute_sharded(out_repeat_sort_: jax.Array, sort_idx_: jax.Array):
            return self._unpermute_local(out_repeat_sort_, sort_idx_, original_shape)

        return unpermute_sharded(out_repeat_sort, sort_idx)

    def _unpermute_local(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Implements the un-permutation logic on a local device shard.

        Args:
            out_repeat_sort: A shard of expert outputs in sorted order.
            sort_idx: A shard of sorting indices.
            original_shape: The original shape of the hidden states.

        Returns:
            The locally un-permuted and reshaped expert outputs.
        """
        inv_sort_idx = jnp.argsort(sort_idx)
        out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
        batch_size, seq_len, hidden_size = original_shape
        out_repeat_unflat = jnp.reshape(out_repeat, (batch_size * seq_len, self.num_experts_per_tok, hidden_size))
        return out_repeat_unflat

    def _compute_load_balancing_loss(
        self,
        router_probs: jax.Array,
        expert_loads: jax.Array,
        strategy: MoeLoadBalancingStrategy | None = None,
    ) -> float | None:
        """Computes the load balancing auxiliary loss.

        This loss encourages the router to distribute tokens evenly across all
        experts, preventing situations where some experts are overloaded while
        others are underutilized.

        Args:
            router_probs: The probabilities output by the router.
            expert_loads: The number of tokens assigned to each expert.
            strategy: The load balancing strategy to use. If None, the default
                strategy from the constructor is used.

        Returns:
            The computed load balancing loss as a scalar float, or None if the
            strategy is `NONE` or the loss coefficient is not set.

        Raises:
            ValueError: If an unknown load balancing strategy is provided.
        """
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

        elif strategy == MoeLoadBalancingStrategy.EXPERT_CHOICE:
            return self.lbl_coef * jnp.var(expert_loads)

        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

    def _compute_router_z_loss(self, router_logits: jax.Array) -> float | None:
        """Computes the router z-loss.

        This auxiliary loss encourages the router to produce logits with small
        magnitudes, which can improve training stability. It is calculated as the
        mean of the squared log-sum-exp of the router logits.

        Args:
            router_logits: The raw logits produced by the router.

        Returns:
            The computed router z-loss as a scalar float, or None if the loss
            coefficient is not set.
        """
        if self.rzl_coef is None:
            return None

        log_z = jax.nn.logsumexp(router_logits, axis=-1)
        return self.rzl_coef * jnp.mean(log_z**2)

    def _compute_metrics(
        self,
        router_logits: jax.Array,
        router_probs: jax.Array,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        expert_loads: jax.Array,
    ) -> MoeMetrics:
        """Computes and aggregates all MoE-related metrics.

        This method consolidates the calculation of various metrics and auxiliary
        losses into a single `MoeMetrics` object.

        Args:
            router_logits: The raw logits from the router.
            router_probs: The probabilities from the router.
            selected_experts: The indices of the chosen experts for each token.
            selected_weights: The weights for the chosen experts.
            expert_loads: The number of tokens assigned to each expert.

        Returns:
            An `MoeMetrics` object containing all computed metrics and losses.
        """
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

    def _apply_expert_sharding(self, hidden_states: jax.Array, axis_name: str = "ep") -> jax.Array:
        """Applies expert parallel sharding to a tensor.

        This utility function shards a tensor, typically expert weights or biases,
        across the expert parallel (`ep`) axis of the device mesh.

        Args:
            hidden_states: The tensor to be sharded. It is expected to have the
                number of experts as its first dimension.
            axis_name: The name of the mesh axis to shard over. Defaults to "ep".

        Returns:
            The sharded tensor.
        """
        if hidden_states.ndim == 3 and hidden_states.shape[0] == self.n_routed_experts:
            sharding = P(axis_name, "tp", None)
        elif hidden_states.ndim == 2 and hidden_states.shape[0] == self.n_routed_experts:
            sharding = P(axis_name, None)
        else:
            sharding = P(None)

        return jax.device_put(hidden_states, jax.sharding.NamedSharding(self.mesh, sharding))

    def _validate_routing_inputs(self, hidden_states: jax.Array, router_logits: jax.Array) -> None:
        """Validates the shapes of inputs for routing operations.

        Args:
            hidden_states: The input tensor to the MoE layer.
            router_logits: The logits produced by the router.

        Raises:
            ValueError: If any of the input shapes are inconsistent with the
                module's configuration.
        """
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
        """Applies a capacity constraint to expert routing.

        This method limits the number of tokens that can be processed by each
        expert, which can prevent imbalances and improve efficiency. Tokens
        routed to an over-capacity expert have their weights down-scaled.

        Args:
            selected_experts: The indices of the selected experts for each token.
                Shape: `(num_tokens, num_experts_per_tok)`.
            selected_weights: The weights for the selected experts.
                Shape: `(num_tokens, num_experts_per_tok)`.
            capacity_factor: The factor to determine the maximum capacity.
                `max_capacity = capacity_factor * num_tokens / num_experts`.
                If None, a default of 1.0 is used.

        Returns:
            A tuple containing:
                - constrained_experts: The expert indices (unchanged in this
                    implementation).
                - constrained_weights: The weights, adjusted for capacity, and
                    re-normalized.
        """
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

    def _create_expert_mask(self, selected_experts: jax.Array, expert_id: int) -> jax.Array:
        """Creates a boolean mask for tokens assigned to a specific expert.

        Args:
            selected_experts: An array of selected expert indices for each token.
            expert_id: The ID of the expert for which to create the mask.

        Returns:
            A boolean array of shape `(num_tokens,)` where `True` indicates that
            the token is routed to the specified expert.
        """
        return jnp.any(selected_experts == expert_id, axis=-1)

    def _moe_call(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        apply_capacity_constraint: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """A generic forward pass implementation for a standard MoE block.

        This method orchestrates the entire MoE process: routing, permutation,
        expert computation, un-permutation, and optional metric calculation.

        Args:
            gate_layer: The module that acts as the router (e.g., a Linear layer).
            expert_layer: The module containing the expert logic (e.g., `MoELinear`).
            hidden_state: The input tensor of shape
                `(batch_size, seq_len, hidden_size)`.
            output_metrics: If True, returns a `MoeMetrics` object along with the
                output. Otherwise, returns the router logits.
            validate_inputs: If True, validates the shapes of routing inputs.
            apply_capacity_constraint: If True, applies capacity constraints to
                the routing.
            reform_router_probs_fn: An optional function to apply to the router
                probabilities after the softmax.

        Returns:
            A tuple containing:
                - output: The final output tensor of shape
                    `(batch_size, seq_len, hidden_size)`.
                - metrics: If `output_metrics` is True, a `MoeMetrics` object.
                    Otherwise, the raw router logits.
        """
        batch_size, seq_len, hidden_size = hidden_state.shape
        hidden_state = apply_logical_sharding(
            hidden_state,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_state_flat = hidden_state.reshape(-1, hidden_size)
        router_logits = gate_layer(hidden_state_flat).astype(jnp.promote_types(self.dtype, jnp.float32))
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        if reform_router_probs_fn is not None:
            router_probs = reform_router_probs_fn(router_probs)

        if validate_inputs:
            self._validate_routing_inputs(hidden_state, router_logits)

        selected_weights, selected_experts = self._route(router_probs)
        if apply_capacity_constraint:
            selected_experts, selected_weights = self._apply_capacity_constraint(selected_experts, selected_weights)

        x_repeat_sort, group_sizes, sort_idx = self._permute(hidden_state_flat, selected_experts.reshape(-1))
        out_repeat_sort = expert_layer(x_repeat_sort, group_sizes)
        out_repeat_unflat = self._unpermute(out_repeat_sort, sort_idx, (batch_size, seq_len, hidden_size))
        output = jnp.sum(out_repeat_unflat * selected_weights[..., None], axis=1)
        output = output.reshape(batch_size, seq_len, hidden_size)

        output = apply_logical_sharding(
            output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
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
    def __call__(self, hidden_states: jax.Array, **kwargs) -> tuple[jax.Array, MoeMetrics]:
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


class MoELinear(nn.Module):
    """A batched Linear layer for Mixture of Experts (MoE).

    This module performs a linear transformation on inputs that have been grouped
    by expert. It supports both standard ragged matrix multiplication and
    optimized Grouped Matrix Multiplication (GMM) on TPUs.

    Attributes:
        num_experts: The number of experts in the layer.
        in_features: The number of input features.
        out_features: The number of output features.
        use_pallas_group_matmul: A boolean indicating whether to use the optimized GMM kernel.
        out_first: If True, the weight matrix has shape `(E, O, I)`. Otherwise,
            `(E, I, O)`.
        dtype: The data type for computations.
        param_dtype: The data type for parameters (weights and biases).
        kernel: The weight parameter for the linear transformation.
        bias: The bias parameter for the linear transformation.
    """

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
        rngs: nn.Rngs,
    ):
        """Initializes the MoELinear layer.

        Args:
            num_experts: The number of experts.
            in_features: The dimension of the input features.
            out_features: The dimension of the output features.
            use_bias: Whether to include a bias term. Defaults to True.
            out_first: If True, the weight kernel is initialized with the output
                dimension first `(num_experts, out_features, in_features)`.
                Defaults to False.
            kernel_init: The initializer for the weight kernel.
            bias_init: The initializer for the bias.
            use_pallas_group_matmul: Whether to use the optimized Grouped Matrix Multiplication
                (GMM) kernel, typically for TPUs. Defaults to False.
            dtype: The data type for the computation.
            param_dtype: The data type for the parameters (weights and bias).
            rngs: The random number generators for parameter initialization.
        """
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.use_pallas_group_matmul = use_pallas_group_matmul
        self.out_first = out_first
        self.dtype = dtype
        self.param_dtype = param_dtype
        kshape = (num_experts, out_features, in_features) if out_first else (num_experts, in_features, out_features)
        self.kernel = nn.Param(kernel_init(rngs.param(), kshape, param_dtype))
        if use_bias:
            bshape = (num_experts, out_features)
            self.bias = nn.Param(bias_init(rngs.param(), bshape, self.param_dtype))
        else:
            self.bias = None

    def __call__(self, inputs: jax.Array, group_sizes: jax.Array) -> jax.Array:
        """Applies the batched linear transformation.

        Args:
            inputs: The input array, which is a batch of tokens sorted and grouped
                by expert. Shape: `(total_tokens, in_features)`.
            group_sizes: An array indicating the number of tokens assigned to each
                expert. Shape: `(num_experts,)`.

        Returns:
            The output array after the linear transformation.
            Shape: `(total_tokens, out_features)`.
        """
        weight = self.kernel.value
        fn = self._ragged_dot
        if self.use_pallas_group_matmul:
            if self.out_first:
                weight = jnp.transpose(self.kernel.value, (0, 2, 1))
            fn = self._grouped_matmul
        inputs, weight = promote_dtype((inputs, weight), dtype=self.dtype)
        output = fn(inputs, weight, group_sizes)

        if self.bias is not None:
            bias_expanded = self._expand_bias_ragged(group_sizes)
            output = output + bias_expanded

        return output

    def _ragged_dot(self, inputs: jax.Array, weight: jax.Array, group_sizes: jax.Array) -> jax.Array:
        """Performs ragged dot product using `jax.lax.ragged_dot_general`.

        This is suitable for inputs where each expert processes a different number
        of tokens.

        Args:
            inputs: The input array.
            weight: The weight array.
            group_sizes: The sizes of token groups for each expert.

        Returns:
            The result of the ragged dot product.
        """
        return jax.lax.ragged_dot_general(
            lhs=inputs,
            rhs=weight,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
                dot_dimension_numbers=(((1,), (2,)) if self.out_first else ((1,), (1,)), ((), ())),
                lhs_ragged_dimensions=(0,),
                rhs_group_dimensions=(0,),
            ),
        )

    def _grouped_matmul(self, inputs: jax.Array, weight: jax.Array, group_sizes: jax.Array) -> jax.Array:
        """Performs grouped matrix multiplication using the Pallas kernel.

        This is a highly optimized operation for TPUs. It may pad the input to
        be a multiple of 512 for efficiency.

        Args:
            inputs: The input array.
            weight: The weight array.
            group_sizes: The sizes of token groups for each expert.

        Returns:
            The result of the grouped matrix multiplication.
        """
        original_batch_size = inputs.shape[0]
        if inputs.shape[0] % 512:
            pad_length = 512 - inputs.shape[0] % 512
            inputs = jnp.pad(inputs, ((0, pad_length), (0, 0)))
        m, k, n = inputs.shape[0], inputs.shape[1], weight.shape[2]
        output = gmm(
            inputs,
            weight,
            group_sizes,
            preferred_element_type=inputs.dtype,
            tiling=(min(m, 512), min(k, 1024), min(n, 1024)),
            interpret=jax.default_backend() != "tpu",
        )
        if original_batch_size % 512:
            output = output[:original_batch_size]
        return output

    def _expand_bias_ragged(self, group_sizes: jax.Array) -> jax.Array:
        """Expands the bias to match the ragged batch structure.

        This method repeats the bias for each expert according to the number of
        tokens assigned to it.

        Args:
            group_sizes: The sizes of token groups for each expert.

        Returns:
            The expanded bias array.
        """
        return self.bias.value[jnp.repeat(jnp.arange(self.num_experts), group_sizes)]
