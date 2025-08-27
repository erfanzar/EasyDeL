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


from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import AttentionLayerOutput, DecoderLayerOutput, MoeCausalLMOutput, MoeModelOutput
from easydel.infra.utils import auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear
from easydel.layers.norms import RMSNorm as FlaxGrok1RMSNorm

from .grok_1_configuration import Grok1Config


class Grok1Attention(AttentionModule):
    """Grok-1 Attention module.

    This module implements the multi-head attention mechanism with rotary position embeddings
    used in the Grok-1 model.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            layer_index (int): The index of the current layer.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
        layer_index: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = ParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.k_proj = ParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.v_proj = ParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.o_proj = ParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim, self.head_dim, True)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (chex.Array): The hidden states with separate head dimensions.

        Returns:
            chex.Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.hidden_size))

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the Grok1Attention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional):
                Cache view for key/value states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional):
                Metadata for cache handling.
            segment_ids (tp.Optional[chex.Array], optional): Segment IDs for segment-based attention.
            output_attentions (bool, optional): Whether to return attention weights.
            fcm_mask (tp.Optional[chex.Array], optional): Forward causal mask.
            frequencies (tp.Optional[chex.Array], optional): Precomputed rotary frequencies.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]: A tuple containing the attention
                output and optionally the attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

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
            cache_metadata,
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

        attn_output = self.resid_dropout(attn_output)
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Grok1BLockSparseMLP(nn.Module):
    """Grok-1 Block Sparse MLP module.

    This module implements the specific MLP structure used within the sparse Mixture of Experts
    layer in the Grok-1 model.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
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
        self.rngs = rngs

        self.linear = ParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.linear_1 = ParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.linear_v = ParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the Grok1BLockSparseMLP module.

        Args:
            hidden_states (chex.Array): Input hidden states.

        Returns:
            chex.Array: Output hidden states after processing through the block sparse MLP.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.linear_1(nn.gelu(self.linear(hidden_states)) * self.linear_v(hidden_states))
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class Grok1SparseMoeBlock(nn.Module):
    """Grok-1 Sparse Mixture of Experts (MoE) block.

    This module implements the sparse MoE layer used in Grok-1. It routes tokens
    to a subset of experts based on learned gating weights.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
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
        self.rngs = rngs
        self.gate = ParallelLinear(
            self.config.hidden_size,
            self.config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(),
        )

        self.experts = [
            Grok1BLockSparseMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_experts)
        ]

    def __call__(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Forward pass of the Grok1SparseMoeBlock.

        Args:
            hidden_states (chex.Array): Input hidden states.

        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing the output hidden states
                and the router logits.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        router_logits = self.gate(hidden_states).astype(jnp.promote_types(self.dtype, jnp.float32))
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = jax.nn.softmax(routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1)
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.layers[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.layers[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(
                        selected_experts == index,
                        routing_weights,
                    ),
                    axis=-1,
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp
        return (final_hidden_state, router_logits)


class Grok1DecoderLayer(nn.Module):
    """Grok-1 Transformer Decoder Layer.

    This module represents a single decoder layer in the Grok-1 model,
    combining self-attention and a sparse MoE block with residual connections
    and layer normalization.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
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
        self.rngs = rngs
        attn_block = Grok1Attention
        mlp_block = Grok1SparseMoeBlock
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.attn = attn_block(
            config=self.config,
            layer_index=self.layer_index,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.moe_block = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pre_attn_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attn_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_moe_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_moe_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass of the Grok1DecoderLayer module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional):
                Cache view for key/value states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional):
                Metadata for cache handling.
            segment_ids (tp.Optional[chex.Array], optional): Segment IDs for segment-based attention.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits from the MoE layer. Defaults to False.
            fcm_mask (tp.Optional[chex.Array], optional): Forward causal mask. Defaults to None.
            frequencies (tp.Optional[chex.Array], optional): Precomputed rotary frequencies.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array], tp.Optional[chex.Array]]: A tuple containing the
                output hidden states, optionally the attention weights, and optionally the router logits.
        """
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        attn_outputs = self.attn(
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
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        hidden_states, router_logits = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)
        hidden_states = residual + hidden_states

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Grok1Config, model_type="grok-1")
class Grok1Model(EasyDeLBaseModule):
    """Grok-1 model implementation.

    This class implements the main Grok-1 transformer model architecture, consisting of
    an embedding layer, multiple Grok1DecoderLayer layers (with sparse MoE), and a final
    RMS normalization layer.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
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
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            Grok1DecoderLayer(
                layer_index=layer_index,
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_index in range(self.config.num_hidden_layers)
        ]

        self.norm = FlaxGrok1RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        return self.config.get_basic_frequencies(
            head_size=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rope_theta,
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
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the Grok1Model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.
            output_router_logits (bool, optional): Whether to return router logits from MoE layers.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing
                precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.


        Returns:
            MoeModelOutput: Model outputs (last hidden state, optional hidden states,
                optional attentions, optional router logits)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        batch_size, sequence_length = inputs_embeds.shape[:2]
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
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                mode=mode,
                cache_view=past_key_values.view[idx],
                cache_metadata=cache_metadata,
                frequencies=self.frequencies,
                causal_mask=self.causal_mask,
                segment_ids=segment_ids,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_self_attns += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Grok1Config, model_type="grok-1")
class Grok1ForCausalLM(EasyDeLBaseModule):
    """Grok-1 model with a language modeling head.

    This model extends the base Grok1Model by adding a linear layer on top to
    predict the next token in a sequence, making it suitable for causal language
    modeling tasks. It also includes handling for the Mixture of Experts auxiliary loss.

    Attributes:
            config (Grok1Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Grok1Config,
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
        self.model = Grok1Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=self.dtype,
            rngs=rngs,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.output_multiplier_scale = self.config.output_multiplier_scale

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput | tuple:
        """Forward pass through the Grok1ForCausalLM model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.
            output_router_logits (bool, optional): Whether to return router logits from MoE layers.
            past_key_values (TransformerCache | PagesCache, optional): Cache
                containing precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.


        Returns:
            MoeCausalLMOutput: Model outputs (logits, optional auxiliary loss, optional hidden states,
                optional attentions, optional router logits)
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            segment_ids=segment_ids,
        )
        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(outputs.last_hidden_state)
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            past_key_values=outputs.past_key_values,
        )

    def apply_lm_head(self, hidden_states: chex.Array) -> chex.Array:
        lm_logits = super().apply_lm_head(hidden_states)
        lm_logits = lm_logits * self.output_multiplier_scale
        return lm_logits

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()
