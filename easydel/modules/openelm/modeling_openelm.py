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
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
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
from easydel.layers.norms import RMSNorm

from .openelm_configuration import OpenELMConfig, make_divisible


class OpenELMMultiHeadCausalAttention(AttentionModule):
    """OpenELM Multi-Head Causal Attention module.

    This module implements the multi-head causal self-attention mechanism used in the OpenELM model.
    It supports Grouped Query Attention (GQA) and optional RMS Normalization of query and key projections.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        layer_idx (int): The index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        qkv_proj (ParallelLinear): Combined linear layer for query, key, and value projections.
        q_norm (RMSNorm, optional): RMS Normalization applied to the query projection if enabled.
        k_norm (RMSNorm, optional): RMS Normalization applied to the key projection if enabled.
        out_proj (ParallelLinear): Linear layer for the output projection.
        head_dim (int): Dimensionality of each attention head.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
        num_q_heads (int): Number of query heads.
        num_k_heads (int): Number of key heads.
        num_v_heads (int): Number of value heads.
        transformer_dim (int): Dimensionality of the transformer model.
        num_groups (int): Number of query groups for GQA.
        rotary (RoPE): Rotary position embedding module.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the OpenELMMultiHeadCausalAttention module.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            layer_idx (int): The index of the current decoder layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = ParallelLinear(
            config.model_dim,
            (q_heads + k_heads + v_heads) * head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        if config.normalize_qk_projections:
            self.q_norm = RMSNorm(
                dim=config.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                dim=config.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = ParallelLinear(
            q_heads * head_dim,
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.head_dim = head_dim

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            base=self.config.rope_freq_constant,
        )

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (chex.Array): The hidden states with separate head dimensions.

        Returns:
            chex.Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_q_heads * self.head_dim))

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
        """
        Forward pass of the OpenELMMultiHeadCausalAttention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

        Returns:
            tp.Union[tp.Tuple[chex.Array, chex.Array], tp.Tuple[chex.Array]]:
                A tuple containing the attention output hidden states. If `output_attentions` is True,
                it also includes the attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(0, 2, 1, 3)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        query_states = qkv[
            :,
            : self.num_q_heads,
            :,
            :,
        ]
        key_states = qkv[
            :,
            self.num_q_heads : self.num_k_heads + self.num_q_heads,
            :,
            :,
        ]
        value_states = qkv[
            :,
            self.num_k_heads + self.num_q_heads :,
            :,
            :,
        ]
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        query_states, key_states, value_states = map(
            lambda x: x.transpose(0, 2, 1, 3),
            [query_states, key_states, value_states],
        )

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

        attn_output = self.out_proj(self.shard_attention_prod(self._merge_heads(attentions.attention_outputs)))

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class OpenELMFeedForwardNetwork(nn.Module):
    """OpenELM Feed-Forward Network (FFN) module.

    This module implements the FFN layer used in the OpenELM model.
    It supports both standard MLP and Gated Linear Unit (GLU) variants.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        layer_idx (int): The index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        ffn_with_glu (bool): Whether the FFN uses a Gated Linear Unit.
        proj_1 (ParallelLinear): First linear projection layer (or gate projection in GLU).
        proj_2 (ParallelLinear): Second linear projection layer (down projection).
        gate_proj (ParallelLinear, optional): Gate projection layer used only if `ffn_with_glu` is True.
        activation_fn (callable): The activation function.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the OpenELMFeedForwardNetwork module.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            layer_idx (int): The index of the current decoder layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,  # type:ignore
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = ParallelLinear(
                config.model_dim,
                2 * intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                **get_dot_general_by_bits(config.bits, config.easy_method),
            )
            self.proj_2 = ParallelLinear(
                intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                **get_dot_general_by_bits(config.bits, config.easy_method),
            )
            self.ffn_with_glu = True
        else:
            self.proj_1 = ParallelLinear(
                config.model_dim,
                intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                **get_dot_general_by_bits(config.bits, config.easy_method),
            )
            self.proj_2 = ParallelLinear(
                intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                **get_dot_general_by_bits(config.bits, config.easy_method),
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        if self.ffn_with_glu:
            y_12 = self.proj_1(hidden_states)
            y_1, y_2 = jnp.split(y_12, 2, axis=-1)
            hidden_states = self.proj_2(self.act(y_1) * y_2)
        else:
            hidden_states = self.proj_2(self.act(self.proj_1(hidden_states)))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class OpenELMDecoderLayer(nn.Module):
    """OpenELM Transformer Decoder Layer.

    This module represents a single decoder layer in the OpenELM model,
    combining self-attention and FFN sub-layers with residual connections
    and layer normalization applied before each sub-layer.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        layer_idx (int): The index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        attn (OpenELMMultiHeadCausalAttention): The self-attention module.
        ffn (OpenELMFeedForwardNetwork): The feed-forward network (FFN) module.
        attn_norm (RMSNorm): Layer normalization before the attention layer.
        ffn_norm (RMSNorm): Layer normalization before the FFN layer.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the OpenELMDecoderLayer.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            layer_idx (int): The index of the current decoder layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        attn_block = OpenELMMultiHeadCausalAttention
        mlp_block = OpenELMFeedForwardNetwork
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        self.attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn = mlp_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn_norm = RMSNorm(
            self.config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
            rngs=rngs,
        )
        self.attn_norm = RMSNorm(
            self.config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
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
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the OpenELMDecoderLayer module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]:
                A tuple containing the output hidden states and optionally the attention weights.
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

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
        hidden_states = residual + attn_outputs.attention_output

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.ffn,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.ffn(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=OpenELMConfig, model_type="openelm")
class OpenELMModel(EasyDeLBaseModule):
    """The base OpenELM model transformer.

    This class represents the core transformer architecture of the OpenELM model,
    consisting of an embedding layer, multiple OpenELMDecoderLayer layers,
    and a final RMS normalization layer.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        token_embeddings (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[OpenELMDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the OpenELMModel.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.token_embeddings = nn.Embed(
            config.vocab_size,
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            OpenELMDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=i,
                rngs=rngs,
            )
            for i in range(self.config.num_transformer_layers)
        ]
        self.norm = RMSNorm(
            config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            eps=1e-6,
            rngs=rngs,
        )
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = ParallelLinear(
                config.model_dim,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
            )
        self.num_transformer_layers = config.num_transformer_layers

    @cached_property
    def frequencies(self):
        return self.config.get_basic_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            base=self.config.rope_freq_constant,
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
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> BaseModelOutput:
        """Forward pass of the OpenELMModel.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.

        Returns:
            BaseModelOutput: The model's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.token_embeddings(input_ids.astype("i4"))
        else:
            raise ValueError("you should specify inputs_embeds or input_ids one of them")
        batch_size, sequence_length, _ = inputs_embeds.shape

        assert (
            sequence_length <= self.config.max_context_length
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_context_length} got {sequence_length})"
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

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, (1, 2))
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            inputs_embeds,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                position_ids=position_ids,
                causal_mask=self.causal_mask,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                output_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
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
        return self.token_embeddings


@register_module(TaskType.CAUSAL_LM, config=OpenELMConfig, model_type="openelm")
class OpenELMForCausalLM(EasyDeLBaseModule):
    """OpenELM model with a Causal Language Modeling head.

    This model consists of the base OpenELM transformer (`OpenELMModel`) followed by a
    linear layer (`lm_head`) that projects the transformer's output hidden states
    to the vocabulary size, producing logits for next token prediction.
    Optionally, the input token embeddings can be tied to the output projection layer.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        transformer (OpenELMModel): The core OpenELM transformer model.
        lm_head (ParallelLinear, optional): The linear layer for projecting hidden states to vocabulary logits.
            This is None if `config.share_input_output_layers` is True.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the OpenELMForCausalLM model.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.transformer = OpenELMModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = ParallelLinear(
            config.model_dim,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
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
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Forward pass of the OpenELMForCausalLM model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.


        Returns:
            CausalLMOutput: The model's output.
                returns a `CausalLMOutput` object containing `logits`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            segment_ids=segment_ids,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(outputs.last_hidden_state)

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
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
        return self.transformer.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.transformer.get_embedding()
