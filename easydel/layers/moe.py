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
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig


class MoeRoutingStrategy(enum.Enum):
    """Different routing strategies for MoE."""

    TOP_K = "top_k"
    TOP_K_NDIV = "top_k_ndiv"
    SWITCH = "switch"
    EXPERT_CHOICE = "expert_choice"
    HASH = "hash"


class MoeLoadBalancingStrategy(enum.Enum):
    """Different load balancing strategies."""

    STANDARD = "standard"
    SWITCH_TRANSFORMER = "switch_transformer"
    EXPERT_CHOICE = "expert_choice"
    NONE = "none"


@dataclass
class MoeMetrics:
    """Container for MoE metrics and auxiliary losses."""

    expert_loads: jnp.ndarray
    router_probs: jnp.ndarray
    selected_experts: jnp.ndarray
    selected_weights: jnp.ndarray
    load_balancing_loss: float | None = None
    router_z_loss: float | None = None
    expert_utilization: float | None = None
    routing_entropy: float | None = None


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros
Initializer = nn.initializers.Initializer


class BaseMoeModule(nn.Module, ABC):
    """
    Base class for Mixture of Experts modules in EasyDeL, providing common utilities.

    This class offers helper functions and attributes commonly needed by MoE
    implementations, such as routing, permutation, load balancing, and sharding.
    Concrete MoE implementations should inherit from this class.
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
        """
        Initializes the BaseMoeModule.

        Args:
            config: The configuration object for this MoE module.
            mesh: Optional JAX mesh for distributed computation.
            routing_strategy: Strategy for routing tokens to experts.
            load_balancing_strategy: Strategy for load balancing.
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
        router_probs: jnp.ndarray,
        routing_strategy: MoeRoutingStrategy | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Select experts for each token based on routing strategy.

        Args:
            router_probs: Router probabilities of shape (batch*seq, num_experts).
            routing_strategy: Override the default routing strategy.

        Returns:
            Tuple of (selected_weights, selected_experts).
        """
        strategy = routing_strategy or self.routing_strategy

        if self.mesh is not None:
            return self._route_sharded(router_probs, strategy)
        else:
            return self._route_local(router_probs, strategy)

    def _route_sharded(self, router_probs: jnp.ndarray, strategy: MoeRoutingStrategy) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sharded routing implementation."""
        if router_probs.ndim == 2:
            in_specs = P("dp", None)
            out_specs = (P("dp", None), P("dp", None))
        elif router_probs.ndim == 3:
            in_specs = P("dp", "fsdp", None)
            out_specs = (P("dp", "fsdp", None), P("dp", "fsdp", None))
        else:
            in_specs = P(None)
            out_specs = (P(None), P(None))

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        def sharded_route(router_probs_):
            return self._route_local(router_probs_, strategy)

        return sharded_route(router_probs)

    def _route_local(self, router_probs: jnp.ndarray, strategy: MoeRoutingStrategy) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Local routing implementation for different strategies."""
        if strategy == MoeRoutingStrategy.TOP_K:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
            selected_weights /= selected_weights.sum(-1, keepdims=True)
        elif strategy == MoeRoutingStrategy.TOP_K_NDIV:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
        elif strategy == MoeRoutingStrategy.SWITCH:
            selected_experts = jnp.argmax(router_probs, axis=-1, keepdims=True)
            selected_weights = jnp.take_along_axis(router_probs, selected_experts, axis=-1)
        elif strategy == MoeRoutingStrategy.EXPERT_CHOICE:
            selected_weights, selected_experts = jax.lax.top_k(
                router_probs.T,
                k=router_probs.shape[0] // self.n_routed_experts,
            )
            selected_weights = selected_weights.T
            selected_experts = selected_experts.T
        elif strategy == MoeRoutingStrategy.HASH:
            token_ids = jnp.arange(router_probs.shape[0])
            selected_experts = (token_ids % self.n_routed_experts)[..., None]
            selected_weights = jnp.ones_like(selected_experts, dtype=router_probs.dtype)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        return selected_weights, selected_experts

    def _permute(self, x_flat: jnp.ndarray, topk_idx_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Permute tokens to group by expert.

        Args:
            x_flat: Flattened input tokens of shape (batch*seq, hidden_size).
            topk_idx_flat: Flattened expert indices of shape (batch*seq*num_experts_per_tok,).

        Returns:
            Tuple of (x_repeat_sort, group_sizes, sort_idx).
        """
        if self.mesh is not None:
            return self._permute_sharded(x_flat, topk_idx_flat)
        else:
            return self._permute_local(x_flat, topk_idx_flat)

    def _permute_sharded(
        self, x_flat: jnp.ndarray, topk_idx_flat: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sharded permutation implementation."""
        # Determine specs based on actual tensor shapes
        x_in_specs = P("dp", None) if x_flat.ndim == 2 else P("dp", "fsdp", None)
        idx_in_specs = P("dp") if topk_idx_flat.ndim == 1 else P("dp", "fsdp")

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(x_in_specs, idx_in_specs),
            out_specs=(P("dp", None), P("ep"), P("dp")),
            check_rep=False,
        )
        def permute_sharded(x_flat_: jnp.ndarray, topk_idx_flat_: jnp.ndarray):
            return self._permute_local(x_flat_, topk_idx_flat_)

        return permute_sharded(x_flat, topk_idx_flat)

    def _permute_local(
        self,
        x_flat: jnp.ndarray,
        topk_idx_flat: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Local permutation implementation."""
        sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
        x_repeat_sort = jnp.take(x_flat, sort_idx // self.num_experts_per_tok, axis=0)
        group_sizes = jnp.bincount(topk_idx_flat, length=self.n_routed_experts)
        return x_repeat_sort, group_sizes, sort_idx

    def _unpermute(
        self,
        out_repeat_sort: jnp.ndarray,
        sort_idx: jnp.ndarray,
        original_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """
        Restore original token order after expert processing.

        Args:
            out_repeat_sort: Expert outputs in sorted order.
            sort_idx: Sorting indices from permutation.
            original_shape: Original shape before flattening (batch, seq_len, hidden_size).

        Returns:
            Unpermuted outputs.
        """
        if self.mesh is not None:
            return self._unpermute_sharded(out_repeat_sort, sort_idx, original_shape)
        else:
            return self._unpermute_local(out_repeat_sort, sort_idx, original_shape)

    def _unpermute_sharded(
        self,
        out_repeat_sort: jnp.ndarray,
        sort_idx: jnp.ndarray,
        original_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Sharded unpermutation implementation."""
        out_in_specs = P("dp", None) if out_repeat_sort.ndim == 2 else P("dp", "fsdp", None)
        idx_in_specs = P("dp") if sort_idx.ndim == 1 else P("dp", "fsdp")

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(out_in_specs, idx_in_specs),
            out_specs=P("dp", None, None),
            check_rep=False,
        )
        def unpermute_sharded(out_repeat_sort_: jnp.ndarray, sort_idx_: jnp.ndarray):
            return self._unpermute_local(out_repeat_sort_, sort_idx_, original_shape)

        return unpermute_sharded(out_repeat_sort, sort_idx)

    def _unpermute_local(
        self,
        out_repeat_sort: jnp.ndarray,
        sort_idx: jnp.ndarray,
        original_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Local unpermutation implementation."""
        inv_sort_idx = jnp.argsort(sort_idx)
        out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
        batch_size, seq_len, hidden_size = original_shape
        out_repeat_unflat = jnp.reshape(out_repeat, (batch_size * seq_len, self.num_experts_per_tok, hidden_size))
        return out_repeat_unflat

    def _compute_load_balancing_loss(
        self,
        router_probs: jnp.ndarray,
        expert_loads: jnp.ndarray,
        strategy: MoeLoadBalancingStrategy | None = None,
    ) -> float | None:
        """
        Compute load balancing loss based on strategy.

        Args:
            router_probs: Router probabilities.
            expert_loads: Expert load distribution.
            strategy: Load balancing strategy to use.

        Returns:
            Load balancing loss or None.
        """
        strategy = strategy or self.load_balancing_strategy

        if strategy == MoeLoadBalancingStrategy.NONE or self.lbl_coef is None:
            return None

        if strategy == MoeLoadBalancingStrategy.STANDARD:
            # Standard MoE load balancing loss
            f = expert_loads * self.n_routed_experts / self.num_experts_per_tok
            p = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * jnp.sum(f * p)

        elif strategy == MoeLoadBalancingStrategy.SWITCH_TRANSFORMER:
            # Switch Transformer load balancing
            num_tokens = router_probs.shape[0]
            expert_fraction = expert_loads / num_tokens
            router_fraction = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * self.n_routed_experts * jnp.sum(expert_fraction * router_fraction)

        elif strategy == MoeLoadBalancingStrategy.EXPERT_CHOICE:
            # Expert Choice load balancing (simplified)
            return self.lbl_coef * jnp.var(expert_loads)

        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

    def _compute_router_z_loss(self, router_logits: jnp.ndarray) -> float | None:
        """
        Compute router z-loss to encourage low variance in router logits.

        Args:
            router_logits: Raw router logits.

        Returns:
            Router z-loss or None.
        """
        if self.rzl_coef is None:
            return None

        log_z = jax.nn.logsumexp(router_logits, axis=-1)
        return self.rzl_coef * jnp.mean(log_z**2)

    def _compute_metrics(
        self,
        router_logits: jnp.ndarray,
        router_probs: jnp.ndarray,
        selected_experts: jnp.ndarray,
        selected_weights: jnp.ndarray,
        expert_loads: jnp.ndarray,
    ) -> MoeMetrics:
        """
        Compute comprehensive MoE metrics.

        Args:
            router_logits: Raw router logits.
            router_probs: Router probabilities.
            selected_experts: Selected expert indices.
            selected_weights: Selected expert weights.
            expert_loads: Expert load distribution.

        Returns:
            MoeMetrics object with all computed metrics.
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

    def _apply_expert_sharding(self, x: jnp.ndarray, axis_name: str = "ep") -> jnp.ndarray:
        """
        Apply expert parallel sharding to tensors.

        Args:
            x: Tensor to shard.
            axis_name: Name of the expert parallel axis.

        Returns:
            Sharded tensor.
        """
        if x.ndim == 3 and x.shape[0] == self.n_routed_experts:
            sharding = P(axis_name, "tp", None)
        elif x.ndim == 2 and x.shape[0] == self.n_routed_experts:
            sharding = P(axis_name, None)
        else:
            sharding = P(None)

        return jax.device_put(x, jax.sharding.NamedSharding(self.mesh, sharding))

    def _validate_routing_inputs(self, x: jnp.ndarray, router_logits: jnp.ndarray) -> None:
        """
        Validate inputs for routing operations.

        Args:
            x: Input tensor.
            router_logits: Router logits.

        Raises:
            ValueError: If inputs are invalid.
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input hidden dimension {x.shape[-1]} doesn't match config hidden dimension {self.hidden_size}"
            )

        if router_logits.shape[-1] != self.n_routed_experts:
            raise ValueError(
                f"Router logits expert dimension {router_logits.shape[-1]} doesn't match "
                f"config expert count {self.n_routed_experts}"
            )

        if router_logits.shape[0] != x.shape[0] * x.shape[1]:
            raise ValueError(
                f"Router logits batch dimension {router_logits.shape[0]} doesn't match "
                f"flattened input batch dimension {x.shape[0] * x.shape[1]}"
            )

    def _apply_capacity_constraint(
        self,
        selected_experts: jnp.ndarray,
        selected_weights: jnp.ndarray,
        capacity_factor: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply capacity constraints using a simple threshold approach.

        Args:
            selected_experts: Selected expert indices of shape (num_tokens, num_experts_per_tok).
            selected_weights: Selected expert weights of shape (num_tokens, num_experts_per_tok).
            capacity_factor: Maximum capacity factor.

        Returns:
            Tuple of (constrained_experts, constrained_weights).
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

    def _create_expert_mask(self, selected_experts: jnp.ndarray, expert_id: int) -> jnp.ndarray:
        """
        Create a mask for tokens assigned to a specific expert.

        Args:
            selected_experts: Selected expert indices.
            expert_id: ID of the expert to create mask for.

        Returns:
            Boolean mask for the expert.
        """
        return jnp.any(selected_experts == expert_id, axis=-1)

    def _moe_call(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """Forward pass of the MoE block."""
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
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, MoeMetrics]:
        """
        Abstract method for forward pass. Must be implemented by subclasses.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output, metrics).
        """
        pass


class MoELinear(nn.Module):
    """A Linear layer for MoE (Mixture of Experts)."""

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
        use_gmm: bool = False,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nn.Rngs,
    ):
        """
        Args:
            num_experts: Number of experts
            in_features: Input feature dimension
            out_features: Output feature dimension
            rngs: Random number generator
            use_bias: Whether to use bias
            out_first: Whether to put output dimension first in weight matrix (PyTorch style)
            init_scale: Scale for initialization
            use_gmm: Whether to use grouped matrix multiplication
        """
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.use_gmm = use_gmm
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

    def __call__(self, inputs: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Input array of shape (batch, in_features)
            group_sizes: Expert sizes of shape (num_experts,)

        Returns:
            Output array of shape (batch, out_features)
        """
        weight = self.kernel.value
        if self.use_gmm:
            if self.out_first:
                weight = jnp.transpose(self.kernel.value, (0, 2, 1))
            output = self._gmm(inputs, weight, group_sizes)
        else:
            output = self._ragged_dot(inputs, weight, group_sizes)

        if self.bias is not None:
            bias_expanded = self._expand_bias_ragged(group_sizes)
            output = output + bias_expanded

        return output

    def _ragged_dot(self, inputs: jnp.ndarray, weight: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
        """Perform ragged dot product using JAX's ragged_dot_general."""

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

    def _gmm(self, inputs: jnp.ndarray, weight: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
        """Perform grouped matrix multiplication."""
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

    def _expand_bias_ragged(self, group_sizes: jnp.ndarray) -> jnp.ndarray:
        """Expand bias to match the ragged batch structure."""
        return self.bias.value[jnp.repeat(jnp.arange(self.num_experts), group_sizes)]
