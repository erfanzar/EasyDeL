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

"""Qwen3Next model implementation for EasyDeL.

This module implements the Qwen3Next hybrid attention architecture, which combines:
- Full attention layers with sigmoid gating and partial RoPE
- Linear attention layers using GatedDeltaNet
- MoE FFN with routed and shared experts

The hybrid attention approach allows for efficient long-context processing
with linear complexity in linear attention layers while maintaining
expressive power through full attention at regular intervals.
"""

import typing
from functools import partial

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
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    LinearCacheView,
    LinearMetadata,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.operations import OperationMetadata
from easydel.layers.operations.modules import GatedDeltaRuleOp, GatedDeltaRuleOutput

from .qwen3_next_configuration import Qwen3NextConfig


def apply_mask_to_padding_states(hidden_states: Array, attention_mask: Array | None) -> Array:
    if (
        attention_mask is not None
        and attention_mask.shape[0] == hidden_states.shape[0]
        and attention_mask.shape[1] == hidden_states.shape[1]
        and attention_mask.shape[1] > 1
    ):
        dtype = hidden_states.dtype
        return (hidden_states * attention_mask[:, :, None]).astype(dtype)
    return hidden_states


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm for Qwen3Next with (1 + weight) scaling formula.

    Qwen3Next uses a modified RMSNorm where the weight is centered at 1:
    output = (1 + weight) * RMSNorm(x)

    This allows initializing weight to zeros while having identity scaling.

    Attributes:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.kernel = nn.Param(jnp.zeros((hidden_size,), dtype=param_dtype))

    def _norm(self, x):
        """Compute RMS normalization."""
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, hidden_states: Float[Array, "... hidden_size"]) -> Float[Array, "... hidden_size"]:
        """
        Apply RMSNorm with (1 + weight) formula.

        Args:
            hidden_states: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        org_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        output = self._norm(hidden_states)
        output = output * (1.0 + self.kernel.value.astype(jnp.float32))
        return output.astype(org_dtype)


class Qwen3NextRMSNormGated(nn.Module):
    """Gated RMSNorm for Qwen3Next linear attention.

    Applies RMSNorm with a gating mechanism: output = z * RMSNorm(x)
    where z is a gating signal passed along with the input.

    Attributes:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.kernel = nn.Param(jnp.ones((hidden_size,), dtype=param_dtype))

    def __call__(
        self,
        hidden_states: Float[Array, "... hidden_size"],
        gate: Float[Array, "... hidden_size"],
    ) -> Float[Array, "... hidden_size"]:
        """Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor to normalize.
            gate: Gating signal to multiply with normalized output.

        Returns:
            Gated normalized tensor.
        """
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        hidden_states = self.kernel.value * hidden_states.astype(input_dtype)
        hidden_states = hidden_states * jax.nn.silu(gate.astype(jnp.float32))
        return hidden_states.astype(input_dtype)


class Qwen3NextMLP(nn.Module):
    """Qwen3Next dense MLP module.

    Standard gated MLP with SiLU activation, used for layers
    that don't use MoE.

    Attributes:
        config: Model configuration.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX precision setting.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.down_proj = row_parallel_linear(intermediate_size, config.hidden_size, rngs=rngs)
        self.up_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
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


class Qwen3NextMLPStack(nn.Module):
    """Qwen3Next MoE MLP using parallel MoE linear layers."""

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
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class Qwen3NextSparseMoeBlock(BaseMoeModule):
    """Sparse Mixture of Experts block for Qwen3Next.

    Routes input to selected experts and combines outputs.
    Includes optional shared expert that processes all tokens.

    Attributes:
        config: Model configuration.
        gate: Router linear layer.
        experts: Stack of expert MLPs.
        shared_expert: Optional shared expert MLP.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Qwen3NextMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert = Qwen3NextMLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert_gate = ColumnParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )

        shared_out = self.shared_expert(hidden_states)
        gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_states))
        shared_out = shared_out * gate
        out = out + shared_out
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3NextFullAttention(UnifiedAttention):
    """Qwen3Next full attention layer with sigmoid gating and partial RoPE.

    Features:
    - q_proj outputs 2x dimension (query + gate), matching HF structure
    - Per-head RMSNorm on Q/K
    - Sigmoid gating applied to attention output

    HuggingFace-compatible structure:
    - q_proj: [hidden_size -> num_heads * head_dim * 2] (query + gate concatenated)
    - k_proj, v_proj, o_proj: Standard projections
    - q_norm, k_norm: Per-head RMSNorm

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=None,
            use_qk_norm=True,
        )
        self.layer_idx = layer_idx

    def _create_q_proj(
        self,
        config,
        dtype,
        param_dtype,
        precision,
        rngs,
    ):
        """Create query projection with 2x output for query + gate.

        HuggingFace Qwen3Next uses q_proj that outputs doubled dimension,
        which is then split into query states and gate values.
        """
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,  # 2x for query + gate
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Use Qwen3Next RMSNorm (1 + weight) for query normalization."""
        return Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Use Qwen3Next RMSNorm (1 + weight) for key normalization."""
        return Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        query_states = self.query_normalization(query_states)
        key_states = self.key_normalization(key_states)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Custom forward for Qwen3Next full attention with gated query.

        The q_proj outputs 2x dimension (query + gate), split and gate applied after attention.
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        q_proj_output = checkpoint_name(self.q_proj(hidden_states), "attn_query")

        q_proj_output = q_proj_output.reshape(batch_size, sequence_length, self.num_heads, self.head_dim * 2)
        query_states, gate = jnp.split(q_proj_output, 2, axis=-1)

        key_states = checkpoint_name(self.k_proj(hidden_states), "attn_key")
        value_states = checkpoint_name(self.v_proj(hidden_states), "attn_value")

        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            sliding_window=self.sliding_window,
        )

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions: AttentionLayerOutput = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=self.causal,
            sliding_window=self.sliding_window,
            softmax_aux=softmax_aux,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attn_output = attentions.attention_outputs

        attn_output = attn_output * jax.nn.sigmoid(gate)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen3NextLinearAttention(nn.Module):
    """Qwen3Next linear attention layer using GatedDeltaNet.

    Implements linear attention with:
    - Causal 1D convolution for local context
    - Gated delta rule recurrence for global context
    - Learnable decay for state forgetting (A_log parameter)
    - Mamba-style dt_bias for time discretization

    HuggingFace-compatible parameter naming:
    - in_proj_qkvz: Projects to query, key, value, and z (gate)
    - in_proj_ba: Projects to beta and alpha
    - A_log: Log of decay matrix A
    - dt_bias: Time discretization bias
    - conv1d: Causal convolution
    - norm: Gated RMSNorm
    - out_proj: Output projection

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
    """

    reform_param: typing.ClassVar = {
        "conv1d.weight$": {
            "splits": [{"name": "conv1d.kernel", "spliter": lambda x: x.permute(2, 1, 0)}],
            "inverse_spliter": lambda torch, kernel: kernel.permute(2, 1, 0),
        },
    }

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        qkvz_dim = self.key_dim * 2 + self.value_dim * 2

        self.in_proj_qkvz = ColumnParallelLinear(
            config.hidden_size,
            qkvz_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        ba_dim = self.num_v_heads * 2
        self.in_proj_ba = ColumnParallelLinear(
            config.hidden_size,
            ba_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        self.norm = Qwen3NextRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.conv1d = nn.Conv(
            in_features=self.conv_dim,
            out_features=self.conv_dim,
            kernel_size=(config.linear_conv_kernel_dim,),
            feature_group_count=self.conv_dim,
            padding=((config.linear_conv_kernel_dim - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
        )

        self.A_log = nn.Param(
            jnp.log(
                jax.random.uniform(
                    rngs.params(),
                    (self.num_v_heads,),
                    dtype=param_dtype,
                    minval=1.0,
                    maxval=16.0,
                )
            )
        )

        self.dt_bias = nn.Param(jnp.ones((self.num_v_heads,), dtype=param_dtype))

        metadata = OperationMetadata(
            runtime_dtype=self.dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=self.config,
        )
        self.gdr_op = GatedDeltaRuleOp(metadata)

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: Float[Array, "batch seq proj_dim"],
        mixed_ba: Float[Array, "batch seq ba_dim"],
    ):
        """Reorder QKV from grouped layout to separate tensors.

        HuggingFace organizes the projection output in a grouped layout:
        [q_h0, k_h0, v_h0, z_h0, q_h1, k_h1, v_h1, z_h1, ...]

        This method unpacks that layout into separate Q, K, V, Z tensors.

        Args:
            mixed_qkvz: Projected QKVZ tensor in grouped layout.
            mixed_ba: Projected beta/alpha tensor.

        Returns:
            Tuple of (query, key, value, z, beta, alpha) tensors.
        """
        batch, seq, _ = mixed_qkvz.shape

        expand_ratio = self.num_v_heads // self.num_k_heads
        per_head_dim = 2 * self.head_k_dim + 2 * self.head_v_dim * expand_ratio
        mixed_qkvz = mixed_qkvz.reshape(batch, seq, self.num_k_heads, per_head_dim)

        split_sizes = [
            self.head_k_dim,
            self.head_k_dim,
            expand_ratio * self.head_v_dim,
            expand_ratio * self.head_v_dim,
        ]
        query, key, value, z = jnp.split(
            mixed_qkvz,
            [split_sizes[0], split_sizes[0] + split_sizes[1], split_sizes[0] + split_sizes[1] + split_sizes[2]],
            axis=-1,
        )

        value = value.reshape(batch, seq, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch, seq, self.num_v_heads, self.head_v_dim)

        ba_per_head = 2 * expand_ratio
        mixed_ba = mixed_ba.reshape(batch, seq, self.num_k_heads, ba_per_head)
        beta, alpha = jnp.split(mixed_ba, [expand_ratio], axis=-1)
        beta = beta.reshape(batch, seq, self.num_v_heads)
        alpha = alpha.reshape(batch, seq, self.num_v_heads)

        return query, key, value, z, beta, alpha

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        cache_view: LinearCacheView | None = None,
        cache_metadata: LinearMetadata | None = None,
    ) -> DecoderLayerOutput:
        if mask_info is not None:
            q_mask = mask_info.q_attention_mask
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
            hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)

        batch_size, seq_len, _ = hidden_states.shape
        is_inference = seq_len == 1 and cache_view is not None

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)

        query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(projected_qkvz, projected_ba)

        query_flat = query.reshape(batch_size, seq_len, -1)
        key_flat = key.reshape(batch_size, seq_len, -1)
        value_flat = value.reshape(batch_size, seq_len, -1)
        conv_input = jnp.concatenate([query_flat, key_flat, value_flat], axis=-1)
        # conv_input: [batch, seq_len, conv_dim]

        conv_state = None
        new_conv_state = None

        if is_inference and cache_view.conv_state is not None:
            # Inference mode: use cached conv_state for incremental convolution
            # conv_state shape: [batch, conv_dim, d_conv]
            conv_state = cache_view.conv_state

            # Roll the conv_state to make room for new input and insert new hidden state
            # conv_input has shape [batch, 1, conv_dim], squeeze seq dim
            new_hidden = conv_input[:, 0, :]  # [batch, conv_dim]
            conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
            conv_state = conv_state.at[:, :, -1].set(new_hidden)
            new_conv_state = conv_state

            # Manual depthwise convolution: sum(conv_state * kernel)
            # kernel shape from nn.Conv: [kernel_size, in_features, out_features]
            # For depthwise (feature_group_count=conv_dim): kernel is [kernel_size, 1, conv_dim]
            kernel = self.conv1d.kernel.value  # [kernel_size, 1, conv_dim]
            kernel = jnp.squeeze(kernel, axis=1)  # [kernel_size, conv_dim]
            kernel = kernel.T  # [conv_dim, kernel_size]

            # conv_state: [batch, conv_dim, d_conv], kernel: [conv_dim, kernel_size]
            # Element-wise multiply and sum over the conv dimension
            conv_output = jnp.sum(conv_state * kernel[None, :, :], axis=-1)  # [batch, conv_dim]
            conv_output = jax.nn.silu(conv_output)
            conv_output = conv_output[:, None, :]  # [batch, 1, conv_dim]
        else:
            # Training/prefill mode: use full convolution
            # conv1d expects [batch, seq, features], outputs same shape with causal padding
            conv_output = jax.nn.silu(self.conv1d(conv_input))

            # Save conv_state for future inference: last d_conv-1 inputs + zero padding
            if cache_view is not None:
                d_conv = self.config.linear_conv_kernel_dim
                # conv_input: [batch, seq_len, conv_dim] -> need [batch, conv_dim, d_conv]
                conv_input_transposed = conv_input.transpose(0, 2, 1)  # [batch, conv_dim, seq_len]
                if seq_len >= d_conv:
                    # Take last d_conv elements
                    new_conv_state = conv_input_transposed[:, :, -d_conv:]
                else:
                    # Pad with zeros on the left
                    pad_width = d_conv - seq_len
                    new_conv_state = jnp.pad(
                        conv_input_transposed,
                        ((0, 0), (0, 0), (pad_width, 0)),
                        mode="constant",
                        constant_values=0,
                    )

        conv_query = conv_output[:, :, : self.key_dim]
        conv_key = conv_output[:, :, self.key_dim : self.key_dim * 2]
        conv_value = conv_output[:, :, self.key_dim * 2 :]

        query = conv_query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = conv_key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = conv_value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        expand_ratio = self.num_v_heads // self.num_k_heads
        if expand_ratio > 1:
            query = jnp.repeat(query, expand_ratio, axis=2)
            key = jnp.repeat(key, expand_ratio, axis=2)

        A = -jnp.exp(self.A_log.value.astype(jnp.float32))
        alpha_biased = alpha.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
        decay = A[None, None, :] * jax.nn.softplus(alpha_biased)

        beta = jax.nn.sigmoid(beta)

        recurrent_state = None
        if cache_view is not None and cache_view.recurrent_state is not None:
            recurrent_state = cache_view.recurrent_state

        gdr_output: GatedDeltaRuleOutput = self.gdr_op(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_state=None,  # conv_state is handled separately above
            recurrent_state=recurrent_state,
        )

        output = gdr_output.attention_outputs

        z_shape_og = z.shape
        output = output.reshape(-1, output.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        output = self.norm(output, z_flat)
        output = output.reshape(z_shape_og)

        output = output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)

        new_cache_view = cache_view
        if cache_view is not None:
            new_cache_view = cache_view.replace(
                conv_state=new_conv_state if new_conv_state is not None else cache_view.conv_state,
                recurrent_state=gdr_output.recurrent_state,
            )

        return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)


class Qwen3NextDecoderLayer(nn.Module):
    """Qwen3Next transformer decoder layer.

    Combines either full or linear attention with MoE or dense MLP,
    based on layer configuration.

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
        is_full_attention: Whether this layer uses full attention.
        is_moe: Whether this layer uses MoE FFN.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.is_full_attention = config.is_full_attention_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)

        full_attn_block, linear_attn_block, mlp_block, moe_block = auto_remat(
            Qwen3NextFullAttention,
            Qwen3NextLinearAttention,
            Qwen3NextMLP,
            Qwen3NextSparseMoeBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        if self.is_full_attention:
            self.self_attn = full_attn_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )
        else:
            self.linear_attn = linear_attn_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )

        if self.is_moe:
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

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        normed_hidden = self.input_layernorm(hidden_states)

        if self.is_full_attention:
            attn_outputs = self.self_attn(
                normed_hidden,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )
        else:
            attn_outputs = self.linear_attn(
                normed_hidden,
                mask_info,
                cache_view,
                cache_metadata,
            )

        attn_output = attn_outputs.attention_output
        attn_weight = attn_outputs.attention_weight
        cache_view = attn_outputs.cache_view
        hidden_states = checkpoint_name(hidden_states + attn_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        feed_forward_output = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_output, router_logits = feed_forward_output

        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_weight if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextModel(EasyDeLBaseModule):
    """Qwen3Next base transformer model.

    Implements the core transformer architecture with hybrid attention
    (alternating between full and linear attention) and MoE FFN layers.

    Attributes:
        config: Model configuration.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Qwen3NextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3NextRMSNorm(
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
            f"Maximum Position Embedding Reached ! "
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
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextForCausalLM(BaseCausalLMModule[Qwen3NextModel, Qwen3NextConfig]):
    """Qwen3Next model with a causal language modeling head.

    Extends the base Qwen3NextModel with a linear output layer for
    next-token prediction.

    Attributes:
        model: Base Qwen3NextModel.
        lm_head: Linear projection to vocabulary.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "qwen3_next"
    _config_class = Qwen3NextConfig

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Qwen3NextModel,
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
        """Compute auxiliary loss from router logits."""
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)


__all__ = [
    "Qwen3NextConfig",
    "Qwen3NextForCausalLM",
    "Qwen3NextModel",
]
