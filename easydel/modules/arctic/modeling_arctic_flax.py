# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


from functools import partial

import chex
import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import (
    ACT2FN,
    auto_remat,
    block_wise_ffn,
    get_dot_general_by_bits,
)
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagedAttentionCache,
    PagedAttentionCacheView,
    PagedAttentionMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear
from easydel.layers.norms import RMSNorm

from .arctic_configuration import ArcticConfig


class ArcticAttention(AttentionModule):
    """
    ArcticAttention module. This module implements the attention mechanism for the Arctic model,
    supporting features like rotary position embeddings and flexible attention implementations.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model.
            dtype (jnp.dtype): Data type for computation (e.g., float32). Defaults to float32.
            param_dtype (jnp.dtype): Data type for parameters (e.g., float32). Defaults to float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (e.g., None, 'high', 'highest').
                Defaults to None.
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
    ):
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        linear = partial(
            ParallelLinear,
            use_bias=getattr(self.config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.q_proj = linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            rngs=rngs,
        )
        self.k_proj = linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
        )
        self.v_proj = linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
        )
        self.o_proj = linear(
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
            rngs=rngs,
        )

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            self.head_dim,
            self.head_dim,
            True,
        )
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        (
            query_states,
            key_states,
            value_states,
        ) = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        (
            key_states,
            value_states,
            attention_mask,
            init_attention_bias,
            cache_view,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
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
            ParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.w1 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w3 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w2 = linear_class(self.ffn_dim, self.hidden_dim, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: chex.Array):
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        w1 = self.act_fn(self.w1(hidden_states))
        w3 = self.w3(hidden_states)
        hidden_states = self.w2(w1 * w3)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class ArcticMoeBlock(nn.Module):
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
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
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
            self.gate = ParallelLinear(
                config.hidden_size,
                config.num_local_experts,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nn.initializers.normal(),
                rngs=rngs,
            )
            self.experts = [
                ArcticMLP(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(config.num_local_experts)
            ]
        else:
            self.mlp = ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                is_residual_mlp=False,
                rngs=rngs,
            )

    def _call_moe(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Executes the Mixture of Experts (MoE) logic.

        Args:
                hidden_states (chex.Array): Input hidden states.

        Returns:
                tp.Tuple[chex.Array, chex.Array]: Tuple containing the final hidden state and the router logits.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
            jnp.promote_types(self.dtype, jnp.float32)
        )
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = jax.nn.softmax(
            routing_weights.astype(
                jnp.promote_types(self.dtype, jnp.float32),
            ),
            axis=-1,
        )
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_local_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(selected_experts == index, routing_weights),
                    axis=-1,
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp

        return final_hidden_state, router_logits

    def __call__(self, hidden_states: chex.Array):
        """
        Forward pass for the ArcticMoeBlock.

        If the current layer is an MoE layer, it calls the MoE logic (_call_moe).
        Otherwise, it passes the input through the standard MLP.

        Args:
                hidden_states (chex.Array): Input hidden states.

        Returns:
                tp.Tuple[chex.Array, chex.Array]: Tuple containing the output
                    hidden state and router logits (or 0.0 if not MoE).
        """
        if self.is_moe_layer:
            return self._call_moe(hidden_states=hidden_states)
        return self.mlp(hidden_states), jnp.array(0.0, dtype=hidden_states.dtype)


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
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
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
        )
        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
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
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
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
            attention_mask,
            position_ids,
            causal_mask,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            fcm_mask,
            frequencies,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = residual_input + hidden_states

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        residual_attn = hidden_states
        if self.parallel_attn_mlp_res:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_residual = residual_attn + hidden_states
            # parallel mlp moe part
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_residual + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            cache_view=attn_outputs.cache_view,
            gate_loss=gate_loss,
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
        self.embed_tokens = nn.Embed(
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
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the ArcticModel.

        Args:
                input_ids (Optional[chex.Array]): Input token IDs.
                inputs_embeds (Optional[chex.Array]): Input embeddings (alternative to input_ids).
                attention_mask (Optional[chex.Array]): Mask to avoid attending to padding tokens.
                position_ids (Optional[chex.Array]): Position IDs for positional embeddings.
                segment_ids (Optional[chex.Array]): Segment IDs (if applicable).
                output_attentions (Optional[bool]): Whether to return attention weights.
                output_hidden_states (Optional[bool]): Whether to return all hidden states.
                past_key_values (Optional[TransformerCache | PagedAttentionCache]):
                    Cached key/value states for faster decoding.
                cache_metadata (Optional[TransformerMetadata | PagedAttentionMetadata]):
                    Metadata for paged attention cache.

        Returns:
                MoeModelOutput: Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_losses = ()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

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
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
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

            all_router_losses += (outputs.gate_loss,)

            past_key_values[idx] = outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_router_losses=all_router_losses,
            past_key_values=past_key_values,
        )


@register_module(TaskType.CAUSAL_LM, config=ArcticConfig, model_type="arctic")
class ArcticForCausalLM(EasyDeLBaseModule):
    """
    Arctic model specifically adapted for Causal Language Modeling (CLM).
    This module wraps the core ArcticModel and adds a language modeling head on top.

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
    ):
        """Initializes the ArcticForCausalLM model."""
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = ArcticModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(config.initializer_range),
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> MoeCausalLMOutput | tuple:
        """Forward pass through the ArcticForCausalLM model.

        Args:
                input_ids (Optional[chex.Array]): Input token IDs.
                attention_mask (Optional[chex.Array]): Mask to avoid attending to padding tokens.
                position_ids (Optional[chex.Array]): Position IDs for positional embeddings.
                segment_ids (Optional[chex.Array]): Segment IDs (if applicable).
                past_key_values (Optional[TransformerCache | PagedAttentionCache]):
                    Cached key/value states for faster decoding.
                cache_metadata (Optional[TransformerMetadata | PagedAttentionMetadata]):
                    Metadata for paged attention cache.
                inputs_embeds (Optional[chex.Array]): Input embeddings (alternative to input_ids).
                output_attentions (Optional[bool]): Whether to return attention weights.
                output_hidden_states (Optional[bool]): Whether to return all hidden states.

        Returns:
                Union[MoeCausalLMOutput, Tuple]: Model outputs, including logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
        )

        hidden_states = outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            lm_logits = jax.lax.dot_general(
                hidden_states,
                self.model.embed_tokens.embedding.value.T,
                (((hidden_states.ndim - 1), (0,)), ((), ())),
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        aux_loss = sum(outputs.all_router_losses) * self.config.router_aux_loss_coef

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_router_losses=outputs.all_router_losses,
            past_key_values=outputs.past_key_values,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=ArcticConfig, model_type="arctic")
class ArcticForSequenceClassification(EasyDeLBaseModule):
    """
    Arctic model adapted for sequence classification tasks.
    This module wraps the core ArcticModel and adds a classification head on top.

    Attributes:
            config (ArcticConfig): Configuration object for the Arctic model (must include num_labels).
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
    ):
        """Initializes the ArcticForSequenceClassification model."""
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = ArcticModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )
        self.score = ParallelLinear(
            config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the ArcticForSequenceClassification model.

        Args:
                input_ids (Optional[chex.Array]): Input token IDs.
                inputs_embeds (Optional[chex.Array]): Input embeddings (alternative to input_ids).
                attention_mask (Optional[chex.Array]): Mask to avoid attending to padding tokens.
                position_ids (Optional[chex.Array]): Position IDs for positional embeddings.
                segment_ids (Optional[chex.Array]): Segment IDs (if applicable).
                past_key_values (Optional[TransformerCache | PagedAttentionCache]):
                    Cached key/value states for faster decoding.
                cache_metadata (Optional[TransformerMetadata | PagedAttentionMetadata]):
                    Metadata for paged attention cache.
                output_attentions (Optional[bool]): Whether to return attention weights.
                output_hidden_states (Optional[bool]): Whether to return all hidden states.

        Returns:
                Union[SequenceClassifierOutput, Tuple]: Model outputs, including classification logits
        """

        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
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
        aux_loss = sum(transformer_outputs.all_router_losses) * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
        )
