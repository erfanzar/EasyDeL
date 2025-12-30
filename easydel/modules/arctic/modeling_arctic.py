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


import typing
from functools import partial

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm

from .arctic_configuration import ArcticConfig


class ArcticAttention(UnifiedAttention):
    """Arctic Attention module with sliding window support.

    Inherits from UnifiedAttention with Arctic-specific customizations:
    - Sliding window attention
    - Custom bias configuration (uses attention_bias config)
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize ArcticAttention with sliding window configuration.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: JAX precision setting
            rngs: Random number generators
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window,
        )

    def _create_q_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use attention_bias for query projection (Arctic-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            precision=precision,
        )

    def _create_k_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use attention_bias for key projection (Arctic-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            precision=precision,
        )

    def _create_v_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use attention_bias for value projection (Arctic-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            precision=precision,
        )

    def _create_o_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use attention_bias for output projection (Arctic-specific)."""
        from easydel.layers.linear import RowParallelLinear

        return RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            precision=precision,
        )

    def _create_rotary(self, config: ArcticConfig, dtype: jnp.dtype):
        """Create Arctic-specific rotary embedding layer."""
        return config.get_basic_rope(dtype, self.head_dim, self.head_dim, True)

    def _create_attention_performer(self, config: ArcticConfig, rngs: nn.Rngs):
        """Create attention performer with Arctic configuration."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )


class ArcticMLPMoE(nn.Module):
    """
    Arctic Multi-Layer Perceptron (MLP) block. This block implements the feed-forward network
    used in the Arctic model. It can optionally function as a residual MLP.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            is_residual_mlp (bool): Whether this MLP block is a residual MLP. Defaults to False.
            rngs (nn.Rngs): Random number generators for the module.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "w1.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "w3.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "w2.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        is_residual_mlp: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.is_residual_mlp = is_residual_mlp
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size if not self.is_residual_mlp else self.hidden_dim

        self.w1 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=self.hidden_dim,
            out_features=self.ffn_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.w3 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=self.hidden_dim,
            out_features=self.ffn_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.w2 = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=self.ffn_dim,
            out_features=self.hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ):
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return apply_logical_sharding(
            self.w2(
                self.act_fn(self.w1(hidden_states, group_sizes, sorted_experts))
                * self.w3(hidden_states, group_sizes, sorted_experts),
                group_sizes,
                sorted_experts,
            ),
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )


class ArcticMLP(nn.Module):
    """
    Arctic Multi-Layer Perceptron (MLP) block. This block implements the feed-forward network
    used in the Arctic model. It can optionally function as a residual MLP.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            is_residual_mlp (bool): Whether this MLP block is a residual MLP. Defaults to False.
            rngs (nn.Rngs): Random number generators for the module.
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        is_residual_mlp: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.is_residual_mlp = is_residual_mlp
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size if not self.is_residual_mlp else self.hidden_dim
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(),
        )
        self.w1 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w3 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w2 = linear_class(self.ffn_dim, self.hidden_dim, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        w1 = checkpoint_name(self.act_fn(self.w1(hidden_states)), "mlp_gate")
        w3 = checkpoint_name(self.w3(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.w2(w1 * w3), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class ArcticMoeBlock(BaseMoeModule):
    """
    Arctic Mixture of Experts (MoE) block. This module implements the MoE layer used in the Arctic model,
    routing tokens to different experts based on a gating mechanism.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            layer_idx (int): The index of the current layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators for the module.
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        super().__init__(
            config=config,
            n_routed_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=getattr(config, "router_aux_loss_coef", None),
            rzl_coef=getattr(config, "router_z_loss_coef", None),
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.top_k = config.num_experts_per_tok
        self.is_moe_layer = (layer_idx + 1) % config.moe_layer_frequency == 0

        if self.is_moe_layer:
            self.gate = ColumnParallelLinear(
                config.hidden_size,
                config.num_local_experts,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nn.initializers.normal(),
                rngs=rngs,
            )
            self.experts = ArcticMLPMoE(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                is_residual_mlp=False,
                rngs=rngs,
            )

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """
        Forward pass for the ArcticMoeBlock.

        If the current layer is an MoE layer, it calls the MoE logic (_call_moe).
        Otherwise, it passes the input through the standard MLP.

        Args:
                hidden_states (Array): Input hidden states.

        Returns:
                tp.Tuple[Array, Array]: Tuple containing the output
                    hidden state and router_logits (or None if not MoE).
        """
        if self.is_moe_layer:
            out, router_logits = self.moe_call(
                hidden_state=hidden_states,
                gate_layer=self.gate,
                expert_layer=self.experts,
                wi_kernel=self.experts.w1.kernel.value,
                wu_kernel=self.experts.w3.kernel.value,
                wd_kernel=self.experts.w2.kernel.value,
                act_fn=self.experts.act_fn,
            )
            return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")
        return self.mlp(hidden_states), None


class ArcticDecoderLayer(nn.Module):
    """
    Arctic Decoder Layer. This module combines the ArcticAttention and ArcticMoeBlock (or ArcticMLP)
    with layer normalization and residual connections to form a standard Transformer decoder layer.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            layer_idx (int): The index of the current layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators for the module.
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        attn_block = ArcticAttention
        mlp_block = ArcticMoeBlock

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.block_sparse_moe = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.parallel_attn_mlp_res = self.config.parallel_attn_mlp_res and self.block_sparse_moe.is_moe_layer
        if self.parallel_attn_mlp_res:
            self.residual_layernorm = RMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.residual_mlp = ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                is_residual_mlp=True,
                rngs=rngs,
            )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = checkpoint_name(residual_input + hidden_states, "residual")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        residual_attn = hidden_states
        router_logits = None
        if self.parallel_attn_mlp_res:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_residual = checkpoint_name(residual_attn + hidden_states, "residual")
            # parallel mlp moe part
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states, router_logits = self.block_sparse_moe(hidden_states)
            hidden_states = checkpoint_name(residual_residual + hidden_states, "residual")
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, router_logits = self.block_sparse_moe(hidden_states)
            hidden_states = checkpoint_name(residual_attn + hidden_states, "residual")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=ArcticConfig, model_type="arctic")
class ArcticModel(EasyDeLBaseModule):
    """
    Core Arctic model architecture. This module implements the main Transformer stack
    for the Arctic model, including token embeddings and decoder layers.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators for the module.
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the ArcticModel."""
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
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            ArcticDecoderLayer(
                layer_idx=layer_idx,
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
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
        """Forward pass through the ArcticModel.

        Args:
                input_ids (Optional[Array]): Input token IDs.
                inputs_embeds (Optional[Array]): Input embeddings (alternative to input_ids).
                attention_mask (Optional[Array]): Mask to avoid attending to padding tokens.
                position_ids (Optional[Array]): Position IDs for positional embeddings.
                segment_ids (Optional[Array]): Segment IDs (if applicable).
                output_attentions (Optional[bool]): Whether to return attention weights.
                output_hidden_states (Optional[bool]): Whether to return all hidden states.
                output_router_logits (Optional[bool]): Whether to return router logits.
                past_key_values (Optional[TransformerCache | RaggedPagesCache]):
                    Cached key/value states for faster decoding.
                cache_metadata (Optional[TransformerMetadata | RaggedPagesMetadata]):
                    Metadata for paged attention cache.

        Returns:
                MoeModelOutput: Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]

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
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = layer(
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
            hidden_states = outputs.hidden_states

            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )

            if output_attentions:
                all_self_attns += (outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (outputs.router_logits,)

            past_key_values[idx] = outputs.cache_view

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For ArcticModel (decoder-only), this is not applicable.
        """
        # As per instructions, raise NotImplementedError for non-encoder models
        # Or you could return `self` if you consider the whole model the "encoder" context,
        # but raising NotImplementedError is more standard for a decoder-only base.
        raise NotImplementedError("ArcticModel is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For ArcticModel, this is the model itself.
        """
        # The ArcticModel *is* the decoder stack.
        return self

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        ArcticModel does not include the lm_head.
        """
        # The lm_head is part of ArcticForCausalLM, not the base ArcticModel.
        raise NotImplementedError("ArcticModel does not include the language model head. See ArcticForCausalLM.")

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=ArcticConfig, model_type="arctic")
class ArcticForCausalLM(BaseCausalLMModule[ArcticModel, ArcticConfig]):
    """Arctic model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "arctic"
    _config_class = ArcticConfig

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=ArcticModel,
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
        """Forward pass of the ArcticForCausalLM model."""
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
        # Filter out None values (from non-MoE layers based on moe_layer_frequency)
        valid_logits = tuple(l for l in outputs.router_logits if l is not None)  # noqa: E741
        if not valid_logits:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=valid_logits,
            num_experts=self.config.num_local_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=ArcticConfig, model_type="arctic")
class ArcticForSequenceClassification(BaseSequenceClassificationModule[ArcticModel, ArcticConfig]):
    """Arctic model with a Sequence Classification head."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "arctic"
    _config_class = ArcticConfig

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=ArcticModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the ArcticForSequenceClassification model."""
        transformer_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        aux_loss = self.compute_router_aux_loss(transformer_outputs)

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
        )
