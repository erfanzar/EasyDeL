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

"""Kimi Linear model implementation for EasyDeL.

This module implements the Kimi Linear hybrid attention architecture, which combines:
- MLA (Multi-Latent Attention) for full attention layers (like DeepSeek V3)
- KDA (Kernel Delta Attention) for linear attention layers
- MoE (Mixture of Experts) with sigmoid routing and shared experts

The hybrid attention approach allows for efficient long-context processing
with O(N) complexity in linear attention layers while maintaining
expressive power through MLA at regular intervals.

References:
    - Kimi Linear: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
"""

import typing
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    KDACacheView,
    KDAMetadata,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.components import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers.operations import OperationMetadata
from easydel.layers.operations.modules import KDAOutput, KernelDeltaAttnOp, fused_kda_gate

from .kimi_linear_configuration import KimiLinearConfig


def apply_mask_to_padding_states(hidden_states: Array, attention_mask: Array | None) -> Array:
    """Apply attention mask to hidden states by zeroing out padding positions.

    Args:
        hidden_states (Array): Hidden state tensor of shape (batch, seq_len, hidden_dim).
        attention_mask (Array | None): Boolean or float mask of shape (batch, seq_len).

    Returns:
        Array: Masked hidden states with padding positions zeroed out.
    """
    if (
        attention_mask is not None
        and attention_mask.shape[0] == hidden_states.shape[0]
        and attention_mask.shape[1] == hidden_states.shape[1]
        and attention_mask.shape[1] > 1
    ):
        dtype = hidden_states.dtype
        return (hidden_states * attention_mask[:, :, None]).astype(dtype)
    return hidden_states


class KimiRMSNorm(nn.Module):
    """RMSNorm for Kimi Linear.

    Standard RMSNorm implementation with configurable epsilon.

    Attributes:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize KimiRMSNorm layer.

        Args:
            hidden_size (int): Dimension of the input features.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(
            KimiRMSNorm.kernel_init(rngs.params(), (hidden_size,), param_dtype),
        )

    def __call__(self, hidden_states: Float[Array, "... hidden_size"]) -> Float[Array, "... hidden_size"]:
        """Apply RMSNorm normalization.

        Args:
            hidden_states (Array): Input tensor of shape (..., hidden_size).

        Returns:
            Array: Normalized tensor of the same shape as input.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return (self.kernel.value * hidden_states).astype(input_dtype)


class KimiRMSNormGated(nn.Module):
    """Gated RMSNorm for Kimi Linear KDA attention.

    Applies RMSNorm with a gating mechanism: output = silu(gate) * RMSNorm(x).
    This is used in the output path of KDA (Kernel Delta Attention) layers
    for improved gradient flow and expressiveness.

    Attributes:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize KimiRMSNormGated layer.

        Args:
            hidden_size (int): Dimension of the input features.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(
            KimiRMSNormGated.kernel_init(rngs.params(), (hidden_size,), param_dtype),
        )

    def __call__(
        self,
        hidden_states: Float[Array, "... hidden_size"],
        gate: Float[Array, "... hidden_size"],
    ) -> Float[Array, "... hidden_size"]:
        """Apply gated RMSNorm normalization.

        Args:
            hidden_states (Array): Input tensor of shape (..., hidden_size).
            gate (Array): Gate tensor of shape (..., hidden_size) for gating mechanism.

        Returns:
            Array: Gated normalized tensor of the same shape as input.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        hidden_states = self.kernel.value * hidden_states.astype(input_dtype)
        hidden_states = hidden_states * jax.nn.silu(gate.astype(jnp.float32))
        return hidden_states.astype(input_dtype)


class KimiMLP(nn.Module):
    """Kimi Linear dense MLP module.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning. Used in both dense layers
    and as shared experts in MoE configurations.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi MLP block.

        Args:
            config (KimiLinearConfig): Model configuration with MLP parameters.
            intermediate_size (int | None, optional): Override intermediate size.
                Defaults to None (uses config.intermediate_size).
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size

        linear_class = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = linear_class(config.hidden_size, intermediate_size)
        self.up_proj = linear_class(config.hidden_size, intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of shape (batch, seq_len, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class KimiMoEGate(nn.Module):
    """Kimi Linear MoE routing gate with sigmoid activation.

    Implements the routing mechanism for Mixture of Experts with:
    - Sigmoid-based expert scoring with e_score_correction_bias (like DeepSeek V3)
    - Grouped top-k expert selection for load balancing
    - Optional probability renormalization

    Attributes:
        top_k: Number of experts to route each token to.
        n_routed_experts: Total number of routed experts.
        routed_scaling_factor: Scaling factor for routed expert outputs.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of groups to select in top-k routing.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi MoE routing gate.

        Args:
            config (KimiLinearConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.top_k = config.num_experts_per_token
        self.n_routed_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.moe_renormalize
        self.gating_dim = config.hidden_size

        self.kernel = ArrayParam.bound(
            shape=(self.gating_dim, self.n_routed_experts),
            dtype=param_dtype,
            init_method="kaiming_uniform",
            key=rngs.params(),
            init_kwargs={},
        )

        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.n_routed_experts,),
            dtype=param_dtype,
            init_method="constant",
            init_kwargs={"value": 0.0},
            key=rngs.params(),
        )

    def __call__(self, hidden_states: Float[Array, "tokens hidden_dim"]):
        """Route tokens to experts.

        Args:
            hidden_states: Flattened hidden states [num_tokens, hidden_dim]

        Returns:
            Top-k expert weights scaled by routed_scaling_factor
        """
        num_tokens, _ = hidden_states.shape

        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.kernel.value.astype(jnp.float32),
            precision=self.precision,
        )

        scores = jax.nn.sigmoid(logits)

        scores_for_choice = scores + self.e_score_correction_bias.value
        group_scores = scores_for_choice.reshape(num_tokens, self.n_group, -1)
        top2_scores = jax.lax.top_k(group_scores, k=2)[0]
        group_scores_sum = jnp.sum(top2_scores, axis=-1)

        group_idx = jax.lax.top_k(group_scores_sum, k=self.topk_group)[1]

        group_mask = jnp.zeros_like(group_scores_sum)
        indices = jnp.arange(group_mask.shape[0])[:, None]
        group_mask = group_mask.at[indices, group_idx].set(1.0)

        score_mask = jnp.repeat(
            group_mask[:, :, None],
            self.n_routed_experts // self.n_group,
            axis=2,
        ).reshape(num_tokens, -1)

        masked_scores = jnp.where(score_mask > 0, scores_for_choice, 0.0)
        topk_weight, _ = jax.lax.top_k(masked_scores, k=self.top_k)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_weight * self.routed_scaling_factor


class KimiMLPMoE(nn.Module):
    """Kimi Linear MoE MLP using parallel expert linear layers.

    Implements a collection of expert MLPs that can be efficiently
    computed in parallel using grouped matrix operations.
    Each expert follows the same SwiGLU architecture as KimiMLP.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [{"name": "down_proj.kernel", "spliter": lambda x: x}],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi MoE MLP experts.

        Args:
            config (KimiLinearConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        intermediate_size = config.moe_intermediate_size
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=intermediate_size,
            out_features=config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply MoE expert computation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            group_sizes (Array): Number of tokens assigned to each expert.
            sorted_experts (Array | None, optional): Sorted expert indices for each token.
                Defaults to None.

        Returns:
            Array: Expert outputs of shape (batch, seq_len, hidden_dim).
        """
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class KimiSparseMoeBlock(BaseMoeModule):
    """Sparse Mixture of Experts block for Kimi Linear.

    Implements the MoE feedforward layer with:
    - Sigmoid routing with e_score_correction_bias (DeepSeek V3 style)
    - Shared experts that process all tokens
    - Grouped top-k expert selection for load balancing
    - Optional auxiliary load balancing loss

    Attributes:
        config: Model configuration.
        gate: Router module for expert selection.
        experts: Collection of expert MLP modules.
        shared_experts: Optional shared experts applied to all tokens.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi Sparse MoE block.

        Args:
            config (KimiLinearConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.gate = KimiMoEGate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.experts = KimiMLPMoE(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        if config.num_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=shared_intermediate_size,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Apply Sparse MoE feedforward transformation.

        Routes tokens to top-k experts based on router scores, combines
        expert outputs with routing weights, and optionally adds shared
        expert contributions.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            tuple[Array, Array]: Tuple of (output tensor, router logits).
                Output has shape (batch, seq_len, hidden_dim).
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )

        if self.config.num_shared_experts > 0:
            shared_out = self.shared_experts(hidden_states)
            out = out + shared_out

        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class KimiMLAAttention(UnifiedAttention):
    """Kimi Linear MLA (Multi-Latent Attention) layer.

    Implements Multi-Latent Attention with DeepSeek V3-style latent KV compression
    for efficient long-context processing while maintaining full attention expressiveness.

    Features:
        - LoRA-style latent compression for key-value projections
        - Per-head RMSNorm on query/key latent projections
        - Support for YaRN RoPE position scaling
        - Optional LoRA for query projections

    Attributes:
        q_head_dim: Total query head dimension (nope + rope).
        qk_nope_head_dim: Query/key dimension without rotary embedding.
        qk_rope_head_dim: Query/key dimension with rotary embedding.
        v_head_dim: Value head dimension.
        kv_lora_rank: Rank of KV LoRA compression.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
        "output_projection": "o_proj",
    }

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi MLA attention layer.

        Args:
            config (KimiLinearConfig): Model configuration with MLA parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        use_mla_lora = config.q_lora_rank is not None and config.q_lora_rank > 0

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            sliding_window=None,
            use_qk_norm=False,
            use_mla_lora=use_mla_lora,
        )

        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ):
        """Define MLA-specific network structure.

        Creates the query, key-value projections with latent compression,
        layer norms, and output projection. Uses LoRA-style compression
        for KV and optionally for queries.

        Args:
            config (KimiLinearConfig): Model configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Numerical precision.
            rngs (nn.Rngs): Random number generator state.
        """

        if not self.use_mla_lora:
            setattr(
                self,
                self.projection_mapping["mla_q_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
        else:
            setattr(
                self,
                self.projection_mapping["mla_q_a_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.q_lora_rank,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_a_layernorm"],
                KimiRMSNorm(
                    config.q_lora_rank,
                    eps=config.rms_norm_eps,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_b_proj"],
                ColumnParallelLinear(
                    config.q_lora_rank,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )

        setattr(
            self,
            self.projection_mapping["mla_kv_a_proj_with_mqa"],
            ColumnParallelLinear(
                config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_a_layernorm"],
            KimiRMSNorm(
                config.kv_lora_rank,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_b_proj"],
            ColumnParallelLinear(
                config.kv_lora_rank,
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )

        setattr(
            self,
            self.projection_mapping["output_projection"],
            RowParallelLinear(
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )

        self.rotary = self._create_rotary(config, dtype)

        self.attention_performer = self._create_attention_performer(config, rngs)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Forward pass using MLA attention mechanism.

        Applies Multi-Latent Attention with LoRA-style KV compression and
        rotary position embeddings for efficient long-context processing.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional):
                Cache view for KV caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies.
                Defaults to None.
            alibi (Array | None, optional): ALiBi position bias (not used in MLA).
                Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output, optional weights, and cache view.
        """
        return self.forward_mla(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            frequencies=frequencies,
            alibi=alibi,
        )


class KimiDeltaAttention(nn.Module):
    """Kimi Linear KDA (Kernel Delta Attention) layer.

    Implements linear attention with O(N) complexity for efficient long-context
    processing. Uses delta rule updates for recurrent state maintenance.

    Architecture components:
        - Separate causal convolutions for Q, K, V (d_conv kernel size)
        - Decay gate via two-layer MLP (f_a_proj -> f_b_proj) with A_log/dt_bias
        - Update gate (beta) via b_proj with sigmoid activation
        - Output gate via two-layer MLP (g_a_proj -> g_b_proj) with gated RMSNorm
        - Chunk-wise processing for memory efficiency

    HuggingFace-compatible parameter naming:
        - q_proj, k_proj, v_proj: Input linear projections
        - q_conv1d, k_conv1d, v_conv1d: Separate causal convolutions
        - f_a_proj, f_b_proj: Decay gate MLP layers
        - b_proj: Beta/update gate projection
        - g_a_proj, g_b_proj: Output gate MLP layers
        - A_log, dt_bias: Decay parameters
        - o_norm: Gated RMSNorm for output
        - o_proj: Output projection

    Attributes:
        num_heads: Number of attention heads.
        head_k_dim: Dimension per head for keys.
        head_v_dim: Dimension per head for values.
        d_conv: Convolution kernel size.
        chunk_size: Size of chunks for chunk-wise processing.
    """

    reform_param: typing.ClassVar = {
        "q_conv1d.weight$": {
            "splits": [{"name": "q_conv1d.kernel", "spliter": lambda x: x.permute(2, 1, 0)}],
            "inverse_spliter": lambda torch, kernel: kernel.permute(2, 1, 0),
        },
        "k_conv1d.weight$": {
            "splits": [{"name": "k_conv1d.kernel", "spliter": lambda x: x.permute(2, 1, 0)}],
            "inverse_spliter": lambda torch, kernel: kernel.permute(2, 1, 0),
        },
        "v_conv1d.weight$": {
            "splits": [{"name": "v_conv1d.kernel", "spliter": lambda x: x.permute(2, 1, 0)}],
            "inverse_spliter": lambda torch, kernel: kernel.permute(2, 1, 0),
        },
    }

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi KDA (Kernel Delta Attention) layer.

        Args:
            config (KimiLinearConfig): Model configuration with KDA parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        linear_config = config.linear_attn_config or {}
        self.num_heads = linear_config.get("num_heads", config.num_attention_heads)
        self.head_k_dim = linear_config.get("head_k_dim", 128)
        self.head_v_dim = linear_config.get("head_v_dim", 128)
        self.d_conv = linear_config.get("d_conv", 4)
        self.expand_ratio = linear_config.get("expand_ratio", 1)
        self.gate_low_rank_dim = linear_config.get("gate_low_rank_dim", 128)
        self.chunk_size = linear_config.get("chunk_size", 64)

        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim

        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.q_proj = column_linear(config.hidden_size, self.key_dim)
        self.k_proj = column_linear(config.hidden_size, self.key_dim)
        self.v_proj = column_linear(config.hidden_size, self.value_dim)

        self.q_conv1d = nn.Conv(
            in_features=self.key_dim,
            out_features=self.key_dim,
            kernel_size=(self.d_conv,),
            feature_group_count=self.key_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.k_conv1d = nn.Conv(
            in_features=self.key_dim,
            out_features=self.key_dim,
            kernel_size=(self.d_conv,),
            feature_group_count=self.key_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.v_conv1d = nn.Conv(
            in_features=self.value_dim,
            out_features=self.value_dim,
            kernel_size=(self.d_conv,),
            feature_group_count=self.value_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
        )

        self.f_a_proj = column_linear(config.hidden_size, self.gate_low_rank_dim)
        self.f_b_proj = column_linear(self.gate_low_rank_dim, self.num_heads)

        self.b_proj = column_linear(config.hidden_size, self.num_heads)

        self.g_a_proj = column_linear(config.hidden_size, self.gate_low_rank_dim)
        self.g_b_proj = column_linear(self.gate_low_rank_dim, self.value_dim)

        self.o_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.o_norm = KimiRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.A_log = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="constant",
            init_kwargs={"value": 0.0},
            key=rngs.params(),
        )
        self.dt_bias = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="constant",
            init_kwargs={"value": 1.0},
            key=rngs.params(),
        )

        metadata = OperationMetadata(
            runtime_dtype=self.dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=self.config,
        )
        self.kda_op = KernelDeltaAttnOp(metadata)

    def _apply_conv_with_state(
        self,
        x: Float[Array, "batch seq_len dim"],
        conv_layer: nn.Conv,
        conv_state: Float[Array, "batch dim d_conv"] | None,
        is_inference: bool,
    ) -> tuple[Float[Array, "batch seq_len dim"], Float[Array, "batch dim d_conv"] | None]:
        """Apply convolution with optional state management for inference.

        Args:
            x: Input tensor [batch, seq_len, dim]
            conv_layer: Convolution layer
            conv_state: Previous convolution state [batch, dim, d_conv]
            is_inference: Whether in inference mode (single token)

        Returns:
            Tuple of (output, new_conv_state)
        """
        _, seq_len, _ = x.shape

        if is_inference and conv_state is not None:
            new_state = jnp.concatenate([conv_state[:, :, 1:], x.transpose(0, 2, 1)], axis=-1)

            kernel = conv_layer.kernel.value
            kernel = kernel.squeeze(1).T
            output = jnp.sum(new_state * kernel[None, :, :], axis=-1)
            output = output[:, None, :]

            return output, new_state
        else:
            output = conv_layer(x)

            if seq_len >= self.d_conv:
                new_state = x[:, -self.d_conv :, :].transpose(0, 2, 1)
            else:
                pad_len = self.d_conv - seq_len
                if conv_state is not None:
                    new_state = jnp.concatenate(
                        [conv_state[:, :, seq_len:], x.transpose(0, 2, 1)],
                        axis=-1,
                    )
                else:
                    new_state = jnp.pad(x.transpose(0, 2, 1), ((0, 0), (0, 0), (pad_len, 0)))

            return output, new_state

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        cache_view: KDACacheView | None = None,
        cache_metadata: KDAMetadata | None = None,
    ) -> AttentionLayerOutput:
        """Forward pass for KDA attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            mask_info: Attention mask information
            cache_view: Hybrid cache view for inference
            cache_metadata: Hybrid cache metadata

        Returns:
            AttentionLayerOutput with attention output and updated cache
        """
        if mask_info is not None:
            q_mask = mask_info.q_attention_mask
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
            hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)

        batch_size, seq_len, _ = hidden_states.shape
        is_inference = seq_len == 1 and cache_view is not None

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        q_conv_state = cache_view.q_conv_state if cache_view is not None else None
        k_conv_state = cache_view.k_conv_state if cache_view is not None else None
        v_conv_state = cache_view.v_conv_state if cache_view is not None else None

        query, new_q_conv_state = self._apply_conv_with_state(query, self.q_conv1d, q_conv_state, is_inference)
        key, new_k_conv_state = self._apply_conv_with_state(key, self.k_conv1d, k_conv_state, is_inference)
        value, new_v_conv_state = self._apply_conv_with_state(value, self.v_conv1d, v_conv_state, is_inference)

        query = jax.nn.silu(query)
        key = jax.nn.silu(key)
        value = jax.nn.silu(value)

        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        f_hidden = jax.nn.silu(self.f_a_proj(hidden_states))
        gate = self.f_b_proj(f_hidden)
        decay = fused_kda_gate(gate, self.A_log.value, self.dt_bias.value)

        beta = jax.nn.sigmoid(self.b_proj(hidden_states))

        g_hidden = jax.nn.silu(self.g_a_proj(hidden_states))
        output_gate = self.g_b_proj(g_hidden)
        output_gate = output_gate.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        recurrent_state = cache_view.recurrent_state if cache_view is not None else None

        kda_output: KDAOutput = self.kda_op(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            q_conv_state=new_q_conv_state,
            k_conv_state=new_k_conv_state,
            v_conv_state=new_v_conv_state,
            recurrent_state=recurrent_state,
            chunk_size=self.chunk_size,
        )

        output = kda_output.attention_outputs

        output_shape = output.shape
        output = output.reshape(-1, self.head_v_dim)
        output_gate_flat = output_gate.reshape(-1, self.head_v_dim)
        output = self.o_norm(output, output_gate_flat)
        output = output.reshape(output_shape)

        output = output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        new_cache_view = cache_view
        if cache_view is not None:
            new_cache_view = cache_view.update_kda_states(
                new_q_conv_state=kda_output.q_conv_state,
                new_k_conv_state=kda_output.k_conv_state,
                new_v_conv_state=kda_output.v_conv_state,
                new_recurrent_state=kda_output.recurrent_state,
            )

        return AttentionLayerOutput(
            attention_output=output,
            attention_weight=None,
            cache_view=new_cache_view,
        )


class KimiDecoderLayer(nn.Module):
    """Kimi Linear transformer decoder layer.

    Combines attention (MLA or KDA) with feedforward (MLP or MoE) based on
    layer configuration. Uses pre-normalization architecture with RMSNorm
    and residual connections.

    The hybrid attention pattern alternates between:
        - MLA (Multi-Latent Attention): Full attention with latent KV compression
        - KDA (Kernel Delta Attention): Linear attention with O(N) complexity

    The feedforward pattern supports:
        - Dense MLP: Standard SwiGLU feedforward network
        - Sparse MoE: Mixture of Experts with shared experts

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer in the model.
        is_kda_layer: Whether this layer uses KDA linear attention.
        is_moe_layer: Whether this layer uses MoE feedforward.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi decoder layer.

        Args:
            config (KimiLinearConfig): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.is_kda_layer = config.is_kda_layer(layer_idx)
        self.is_moe_layer = config.is_moe_layer(layer_idx)

        mla_block, kda_block, mlp_block, moe_block = auto_remat(
            KimiMLAAttention,
            KimiDeltaAttention,
            KimiMLP,
            KimiSparseMoeBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        if self.is_kda_layer:
            self.self_attn = kda_block(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        elif config.is_mla:
            self.self_attn = mla_block(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            raise NotImplementedError

        if self.is_moe_layer:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        self.input_layernorm = KimiRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | KDACacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + ffn(norm(x)).
        Uses either MLA or KDA attention based on layer configuration, and either
        dense MLP or sparse MoE for the feedforward network.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | KDACacheView | None, optional):
                Cache view for attention caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            output_router_logits (bool, optional): Whether to return MoE router logits.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies.
                Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits,
                and updated cache view.
        """
        normed_hidden = self.input_layernorm(hidden_states)

        if self.is_kda_layer:
            attn_outputs = self.self_attn(
                normed_hidden,
                mask_info,
                cache_view,
                cache_metadata,
            )
        else:
            attn_outputs = self.self_attn.forward(
                normed_hidden,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )

        attn_output = attn_outputs.attention_output
        attn_weight = attn_outputs.attention_weight
        cache_view = attn_outputs.cache_view

        hidden_states = checkpoint_name(hidden_states + attn_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)
        feed_forward_output = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe_layer:
            feed_forward_output, router_logits = feed_forward_output

        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_weight if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=KimiLinearConfig, model_type="kimi_linear")
class KimiLinearModel(EasyDeLBaseModule):
    """Kimi Linear base transformer model.

    Implements the hybrid attention architecture combining MLA (Multi-Latent Attention)
    and KDA (Kernel Delta Attention) for efficient long-context processing. The model
    alternates between full attention (MLA) and linear attention (KDA) layers,
    with optional Mixture of Experts (MoE) feedforward layers.

    Key features:
        - Hybrid attention with O(N) linear attention layers
        - MLA with latent KV compression (DeepSeek V3 style)
        - MoE with sigmoid routing and shared experts
        - Support for very long contexts (100K+ tokens)

    Attributes:
        config (KimiLinearConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi Linear base model.

        Args:
            config (KimiLinearConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            KimiDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = KimiRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the Kimi Linear base model.

        Processes input tokens through embedding, hybrid attention layers
        (MLA and KDA), optional MoE feedforward, and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch, seq_len). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states
                from all layers. Defaults to None.
            output_router_logits (bool | None, optional): Whether to return MoE router logits.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional hidden_states, attentions,
                past_key_values, and router_logits.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_router_logits = () if output_router_logits else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached! "
            f"(Expected <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        if past_key_values is None:
            past_key_values = HybridCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        """Return the encoder part of the model.

        Raises:
            NotImplementedError: This is a decoder-only model and does not have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder part of the model.

        Returns:
            KimiLinearModel: The decoder (this model itself).
        """
        return self

    def get_lm_head(self):
        """Return the language model head.

        Raises:
            NotImplementedError: The base model does not have a language model head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=KimiLinearConfig, model_type="kimi_linear")
class KimiLinearForCausalLM(BaseCausalLMModule[KimiLinearModel, KimiLinearConfig]):
    """Kimi Linear model with a causal language modeling head.

    Extends the base KimiLinearModel with a linear output layer for
    next-token prediction. Supports MoE auxiliary loss for load balancing.

    This model is a hybrid transformer with causal attention masks
    applied to perform autoregressive language generation, combining
    efficient linear attention (KDA) with full attention (MLA).

    Attributes:
        config (KimiLinearConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "kimi_linear"
    _config_class = KimiLinearConfig

    def __init__(
        self,
        config: KimiLinearConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Kimi Linear model for causal language modeling.

        Args:
            config (KimiLinearConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=KimiLinearModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass with language modeling head.

        Processes inputs through the base model and applies the language modeling
        head to produce next-token logits. Computes MoE auxiliary loss if enabled.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
                Defaults to None.
            attention_mask (Array | None, optional): Boolean mask for padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens.
                Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the LM head projection.
                Defaults to True.

        Returns:
            MoeCausalLMOutput: Contains logits, optional hidden_states, attentions,
                past_key_values, router_logits, and aux_loss.
        """
        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=self._compute_aux_loss,
        )

    def _compute_aux_loss(self, outputs, attention_mask):
        """Compute auxiliary loss from router logits for MoE load balancing.

        Calculates the auxiliary load balancing loss to encourage uniform
        expert utilization across the MoE layers.

        Args:
            outputs: Model outputs containing router_logits from MoE layers.
            attention_mask: Attention mask to exclude padding tokens from loss.

        Returns:
            Array | None: Auxiliary loss value, or None if no router logits available.
        """
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_token,
            attention_mask=attention_mask,
        )
        router_aux_loss_coef = getattr(self.config, "router_aux_loss_coef", 0.001)
        return aux_loss + (aux_loss * router_aux_loss_coef)


__all__ = [
    "KimiDecoderLayer",
    "KimiDeltaAttention",
    "KimiLinearConfig",
    "KimiLinearForCausalLM",
    "KimiLinearModel",
    "KimiMLAAttention",
    "KimiSparseMoeBlock",
]
