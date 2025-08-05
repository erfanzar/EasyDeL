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


import functools

import chex
import jax.lax
from chex import Array
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

from .phi_configuration import PhiConfig


class PhiMLP(nn.Module):
    """Phi MLP module.

    This module implements the feed-forward network (MLP) used in the Phi model.
    It consists of two linear projections with a GELU activation in between.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        layer_idx (int, optional): Index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        fc1 (ParallelLinear): First linear projection layer (up-projection).
        fc2 (ParallelLinear): Second linear projection layer (down-projection).
        act (callable): Activation function.
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiMLP module.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
            layer_idx (int, optional): Index of the current layer. Defaults to None.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.fc1 = ParallelLinear(
            config.n_embd,
            config.intermediate_size,
            kernel_init=nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = ParallelLinear(
            config.intermediate_size,
            config.n_embd,
            kernel_init=nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        """Forward pass of the PhiMLP module.

        Args:
            hidden_states (Array): Input hidden states.

        Returns:
            Array: Output hidden states after MLP transformation.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class PhiAttention(AttentionModule):
    """Phi Attention module.

    This module implements the multi-head attention mechanism used in the Phi model.
    It supports Grouped Query Attention (GQA), partial Rotary Position Embeddings (RoPE),
    and optional Layer Normalization for query and key projections.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        layer_idx (int, optional): Index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        attention_dropout (float): Dropout probability for attention scores.
        hidden_size (int): Dimensionality of the hidden states.
        num_heads (int): Number of attention query heads.
        head_dim (int): Dimensionality of each attention head.
        num_key_value_heads (int): Number of attention key/value heads (for GQA).
        num_key_value_groups (int): Number of query head groups for each key/value head.
        max_position_embeddings (int): Maximum sequence length supported by RoPE.
        rope_theta (float): Base value for RoPE frequency calculation.
        partial_rotary_factor (float): Factor determining the fraction of head dimension subject to RoPE.
        is_causal (bool): Whether the attention is causal (always True for this implementation).
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        dense (ParallelLinear): Linear layer for the output projection.
        rotary_emb_dim (int): The dimension of the rotary embeddings.
        qk_layernorm (bool): Whether to apply LayerNorm to query and key projections.
        q_layernorm (nn.LayerNorm, optional): Layer normalization for query projections.
        k_layernorm (nn.LayerNorm, optional): Layer normalization for key projections.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
        rotary (RoPE): Rotary position embedding module.
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiAttention module.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
            layer_idx (int, optional): Index of the current layer. Defaults to None.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_heads`.
        """
        super().__init__(config=config)

        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        linear_class = functools.partial(
            ParallelLinear,
            use_bias=True,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.q_proj = linear_class(
            self.hidden_size,
            self.num_heads * self.head_dim,
            rngs=rngs,
        )
        self.k_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
        )
        self.v_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
        )
        self.dense = linear_class(
            self.num_heads * self.head_dim,
            self.hidden_size,
            rngs=rngs,
        )
        self.rotary_emb_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                use_bias=True,
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                use_bias=True,
            )

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            head_size=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
            rotary_dim=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
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
        """
        Forward pass of the PhiAttention module.

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
        (query_states, key_states, value_states) = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

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
        attn_output = self.dense(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class PhiDecoderLayer(nn.Module):
    """Phi Transformer Decoder Layer.

    This module represents a single decoder layer in the Phi model,
    combining self-attention and MLP sub-layers with residual connections
    and layer normalization.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        layer_idx (int, optional): Index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        input_layernorm (nn.LayerNorm): Layer normalization applied before the attention and MLP blocks.
        resid_dropout (nn.Dropout): Dropout applied to the residual connection after the MLP block.
        self_attn (PhiAttention): The self-attention module.
        mlp (PhiMLP): The feed-forward (MLP) module.
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiDecoderLayer.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
            layer_idx (int, optional): Index of the current layer. Defaults to None.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        attn_block = PhiAttention
        mlp_block = PhiMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)

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
        """Forward pass of the PhiDecoderLayer module.

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

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        feed_forward_hidden_states = self.resid_dropout(feed_forward_hidden_states)
        hidden_states = self.resid_dropout(attn_outputs.attention_output) + feed_forward_hidden_states + residual
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


@register_module(TaskType.BASE_MODULE, config=PhiConfig, model_type="phi")
class PhiModel(EasyDeLBaseModule):
    """The base Phi model transformer.

    This class represents the core transformer architecture of the Phi model,
    consisting of an embedding layer, multiple PhiDecoderLayer layers,
    and a final layer normalization.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[PhiDecoderLayer]): List of decoder layers.
        final_layernorm (nn.LayerNorm): Final layer normalization.
        embed_dropout (nn.Dropout): Dropout layer applied after embeddings.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: PhiConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiModel.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embed_dropout = nn.Dropout(config.embd_pdrop, rngs=rngs)
        self.layers = [
            PhiDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=idx,
                rngs=rngs,
            )
            for idx in range(self.config.num_hidden_layers)
        ]
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self):
        return self.config.get_basic_frequencies(
            head_size=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
            rotary_dim=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
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
        """Forward pass of the PhiModel.

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
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
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.final_layernorm(hidden_states)

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
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=PhiConfig, model_type="phi")
class PhiForCausalLM(EasyDeLBaseModule):
    """Phi model with a Causal Language Modeling head.

    This model consists of the base Phi transformer (`PhiModel`) followed by a
    linear layer (`lm_head`) that projects the transformer's output hidden states
    to the vocabulary size, producing logits for next token prediction.
    Optionally, the input token embeddings can be tied to the output projection layer.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        transformer (PhiModel): The core Phi transformer model.
        lm_head (ParallelLinear): The linear layer for projecting hidden states to vocabulary logits.
    """

    def __init__(
        self,
        config: PhiConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiForCausalLM model.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
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
        self.model = PhiModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = self.config.vocab_size
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Forward pass of the PhiForCausalLM model.

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
        outputs = self.model(
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
        hidden_states = outputs.last_hidden_state
        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(hidden_states)

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
