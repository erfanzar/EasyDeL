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


# coding=utf-8
# Copyright 2022 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# THIS SCRIPT IS EDITED FROM ORIGINAL IMPLEMENTATION OF TRANSFORMERS OPT
"""Flax OPT model."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, MaskedLMOutput
from easydel.infra.utils import ACT2FN
from easydel.layers.attention import AttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear

from .opt_configuration import OPTConfig


class OPTAttention(AttentionModule):
    """OPT Attention mechanism module.

    This module implements the multi-head self-attention mechanism used in the OPT model.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        embed_dim (int): The dimensionality of the embedding layer.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability for the attention scores.
        causal (bool): Whether to use causal masking.
        bias (bool): Whether to include bias in the linear projections.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        head_dim (int): Dimensionality of each attention head.
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        out_proj (ParallelLinear): Linear layer for the output projection.
        dropout_layer (nn.Dropout): Dropout layer applied after attention.
        attention_module (AttentionModule): The core attention computation module.
    """

    def __init__(
        self,
        config: OPTConfig,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the OPTAttention module.

        Args:
            config (OPTConfig): The configuration object for the OPT model.
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate for the attention scores. Defaults to 0.0.
            causal (bool, optional): Whether to apply causal masking. Defaults to False.
            bias (bool, optional): Whether to use bias in linear projections. Defaults to True.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()

        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.bias = bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`: {num_heads})."
            )

        linear = partial(
            ParallelLinear,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
        )

        self.q_proj, self.k_proj, self.v_proj = (
            linear(embed_dim, embed_dim, rngs=rngs),
            linear(embed_dim, embed_dim, rngs=rngs),
            linear(embed_dim, embed_dim, rngs=rngs),
        )
        self.out_proj = linear(embed_dim, embed_dim, rngs=rngs)

        self.dropout_layer = nn.Dropout(rate=self.dropout, rngs=rngs)
        self.attention_module: AttentionModule = AttentionModule(
            dropout_prob=config.attention_dropout,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            head_dims=self.head_dim,
            precision=precision,
            force_float32_tpu=True,
            attn_mechanism=config.attn_mechanism,
            dtype=config.attn_dtype,
            mesh=config.mesh,
            softmax_scale=self.head_dim**-0.5,
            axis_name=config.sequence_axis_name,
            base_config=config,
        )

    def _split_heads(self, hidden_states):
        """Splits the hidden states into multiple heads."""
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """Merges the attention heads back into a single hidden state tensor."""
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        causal_mask: chex.Array | None = None,
        key_value_states: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
    ) -> tuple[chex.Array]:
        """Forward pass of the OPTAttention module.

        Args:
            hidden_states (chex.Array): Input hidden states. Shape: (batch_size, sequence_length, embed_dim).
            causal_mask (tp.Optional[chex.Array]): Causal mask for attention.
                Shape: (1, 1, query_len, key_len) or inferred.
            key_value_states (tp.Optional[chex.Array]): Optional hidden states for cross-attention. If provided,
                these are used as keys and values. Shape: (batch_size, key_sequence_length, embed_dim).
            attention_mask (tp.Optional[chex.Array]): Mask to prevent attention to certain positions.
                Shape: (batch_size, 1, query_length, key_length).
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.

        Returns:
            tp.Tuple[chex.Array]: A tuple containing the attention output (shape: batch_size, sequence_length, embed_dim)
                and the attention weights (shape: batch_size, num_heads, sequence_length, key_sequence_length).
                Attention weights are returned only if `output_attentions` is True in the base attention module.
        """
        is_cross_attention = key_value_states is not None
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if attention_mask is not None:
            if self.causal:
                if attention_mask.ndim == 2:
                    attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
                    attention_mask = jnp.logical_and(attention_mask, self.causal_mask[:, :, :sequence_length, :])
                elif attention_mask.ndim == 4:
                    assert attention_mask.shape == (batch_size, 1, sequence_length, 1)
            else:
                if attention_mask.ndim == 2:
                    attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
        if not self.causal:
            causal_mask = None
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
        )

        attentions = self.attention_module(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            causal=self.causal,
            query_sequence_length=query_states.shape[1],
            key_value_sequence_length=key_states.shape[1],
            uses_cache=cache_view is not None,
            causal_mask=causal_mask,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.out_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=None,
            cache_view=cache_view,
        )


class OPTDecoderLayer(nn.Module):
    """OPT Decoder Layer.

    This module represents a single layer in the OPT decoder stack.
    It consists of a self-attention mechanism, optional layer normalization,
    a feed-forward network (FFN), and residual connections.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        embed_dim (int): Dimensionality of the embedding layer.
        self_attn (OPTAttention): The self-attention module.
        do_layer_norm_before (bool): Whether to apply layer normalization before the attention/FFN blocks.
        dropout_layer (nn.Dropout): Dropout layer applied to the hidden states.
        activation_fn (callable): The activation function used in the FFN.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization applied before the self-attention module.
        fc1 (ParallelLinear): The first linear layer of the FFN.
        fc2 (ParallelLinear): The second linear layer (output) of the FFN.
        final_layer_norm (nn.LayerNorm): Layer normalization applied before the FFN module.
    """

    def __init__(
        self,
        config: OPTConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = self.config.hidden_size
        self.self_attn = OPTAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            causal=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.do_layer_norm_before = self.config.do_layer_norm_before
        self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)
        self.activation_fn = ACT2FN[self.config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            epsilon=1e-05,
        )
        self.fc1 = ParallelLinear(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
            rngs=rngs,
        )
        self.fc2 = ParallelLinear(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
            rngs=rngs,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            epsilon=1e-05,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        causal_mask: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
    ) -> tuple[chex.Array]:
        """Forward pass of the OPTDecoderLayer.

        Args:
            hidden_states (chex.Array): Input hidden states. Shape: (batch_size, sequence_length, embed_dim).
            causal_mask (tp.Optional[chex.Array]): Causal mask for self-attention.
                Shape: (1, 1, query_len, key_len) or inferred.
            attention_mask (tp.Optional[chex.Array]): Mask to prevent attention to certain positions.
                Shape: (batch_size, 1, query_length, key_length).
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.

        Returns:
            tp.Tuple[chex.Array]: A tuple containing the output hidden states
                (shape: batch_size, sequence_length, embed_dim)
                and the attention weights (shape: batch_size, num_heads, sequence_length, key_sequence_length).
                Attention weights are returned only if `output_attentions` is True in the base attention module.
        """
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            mode=mode,
            causal_mask=causal_mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)

        hidden_states = (residual + hidden_states).reshape(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states, self_attn_weights


class OPTLearnedPositionalEmbedding(nn.Embed):
    """Learned positional embedding for OPT.

    This module learns positional embeddings up to a maximum specified length.
    It includes an offset, typically used to account for padding tokens.

    Attributes:
        offset (int): The offset added to position IDs before embedding lookup.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        offset: int = 2,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        embedding_init=None,
        rngs: nn.Rngs,
    ):
        """Initializes the OPTLearnedPositionalEmbedding module.

        Args:
            num_embeddings (int): The maximum number of positions to embed (vocabulary size).
            features (int): The dimensionality of the embedding vectors.
            offset (int, optional): The offset added to position IDs. Defaults to 2.
            dtype (tp.Optional[jnp.dtype], optional): Data type for the embeddings. Defaults to None.
            param_dtype (jnp.dtype, optional): Data type for the parameters. Defaults to jnp.float32.
            embedding_init (callable, optional): Initializer function for the embeddings.
                Defaults to `jax.nn.initializers.normal(stddev=1.0)`.
            rngs (nn.Rngs): Random number generators.
        """
        if embedding_init is None:
            embedding_init = nn.initializers.variance_scaling(
                1.0,
                "fan_in",
                "normal",
                out_axis=0,
            )
        self.embedding = nn.Param(embedding_init(rngs.params(), (num_embeddings + offset, features), param_dtype))
        self.offset = offset
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init

    def __call__(self, inputs: chex.Array) -> chex.Array:
        return super().__call__(inputs + self.offset)


class OPTDecoder(EasyDeLBaseModule):
    """OPT Decoder stack.

    This module comprises the main transformer decoder layers for the OPT model,
    including token embeddings, positional embeddings, the decoder layers themselves,
    and optional final layer normalization.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        padding_idx (int): Index of the padding token.
        max_target_positions (int): Maximum sequence length the model can handle.
        embed_scale (float): Scaling factor for embeddings (usually 1.0).
        embed_tokens (nn.Embed): Token embedding layer.
        embed_positions (OPTLearnedPositionalEmbedding): Positional embedding layer.
        project_out (ParallelLinear, optional): Optional linear projection layer after embeddings.
        project_in (ParallelLinear, optional): Optional linear projection layer before embeddings.
        layers (tp.List[OPTDecoderLayer]): List of OPT decoder layers.
        dropout_layer (nn.Dropout): Dropout layer applied after embeddings.
        final_layer_norm (nn.LayerNorm, optional): Optional final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)

        embed_dim = self.config.hidden_size
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings

        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.word_embed_proj_dim,
            embedding_init=nn.initializers.normal(config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.embed_positions = OPTLearnedPositionalEmbedding(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=nn.initializers.normal(config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            offset=offset,
        )

        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = ParallelLinear(
                self.config.word_embed_proj_dim,
                self.config.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.project_out = ParallelLinear(
                self.config.hidden_size,
                self.config.word_embed_proj_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        else:
            self.project_in = None
            self.project_out = None

        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=param_dtype,
                epsilon=1e-05,
                rngs=rngs,
            )
        else:
            self.final_layer_norm = None

        self.layers = [
            OPTDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the OPTDecoder.

        Args:
            input_ids (chex.Array): Input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (tp.Optional[chex.Array]): Mask to prevent attention to padding tokens.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states for all layers. Defaults to False.

        Returns:
            BaseModelOutput: The decoder's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        inputs_embeds = self.embed_tokens(input_ids)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        positions = self.embed_positions(position_ids)
        batch_size, sequence_length = inputs_embeds.shape[:2]
        hidden_states = inputs_embeds + positions
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
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                mode=mode,
                past_key_values=past_key_values.views[idx],
                cache_metadata=cache_metadata,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_state = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_state = self.project_out(hidden_state)

        if output_hidden_states:
            all_hidden_states += (hidden_state,)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class OPTModel(EasyDeLBaseModule):
    """Base OPT Model class.

    This class represents the core OPT model architecture, consisting primarily of the OPTDecoder.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        decoder (OPTDecoder): The OPT decoder stack.
    """

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.decoder = OPTDecoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            offset=offset,
        )

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the OPTModel.

        Args:
            input_ids (chex.Array): Input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (tp.Optional[chex.Array]): Mask to prevent attention to padding tokens.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states for all layers. Defaults to False.

        Returns:
            BaseModelOutput: The model's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        return BaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def set_input_embeddings(self, value):
        """Sets the input embeddings for the model."""
        self.decoder.embed_tokens = value

    def get_input_embeddings(self):
        """Gets the input embeddings from the model."""
        return self.decoder.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=OPTConfig, model_type="opt")
class OPTForCausalLM(EasyDeLBaseModule):
    """OPT Model with a Causal Language Modeling head.

    This model consists of the base OPTModel followed by a linear layer
    (the language modeling head) to predict the next token logits.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        model (OPTModel): The base OPT model.
        lm_head (ParallelLinear): The linear layer for projecting hidden states to vocabulary logits.
    """

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = OPTModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            offset=offset,
        )
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the OPTForCausalLM model.

        Args:
            input_ids (chex.Array): Input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (tp.Optional[chex.Array]): Mask to prevent attention to padding tokens.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states for all layers. Defaults to False.
             Defaults to True.

        Returns:
            tp.Union[MaskedLMOutput, tp.Tuple]: The model's output.
                returns a `MaskedLMOutput` object containing `logits`, `hidden_states` (optional),
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
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.decoder.embed_tokens.embedding.value.T
            self.lm_head.kernel.value = shared_kernel
            lm_logits = self.lm_head.apply(hidden_states)

        else:
            lm_logits = self.lm_head(hidden_states)

        return MaskedLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_input_embeddings(self, value):
        """Sets the input embeddings for the model."""
        self.model.decoder.embed_tokens = value

    def get_input_embeddings(self):
        """Gets the input embeddings from the model."""
        return self.model.decoder.embed_tokens

    def set_decoder(self, decoder):
        """Sets the decoder module for the model."""
        self.model.decoder = decoder

    def get_decoder(self):
        """Gets the decoder module from the model."""
        return self.model.decoder

    def get_output_embeddings(self):
        """Gets the output embeddings (language modeling head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings (language modeling head)."""
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings=None,
        attention_mask: chex.Array | None = None,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        if starts is None:
            starts = self.compute_prefill_length(input_ids, pad_token_id)
        past_key_values = self.init_cache(
            batch_size,
            max_length,
            starts,
            shardings,
            pad_token_id,
        )
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="b1")

        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return self.prepare_inputs_for_call(
            **{
                "past_key_values": past_key_values,
                "attention_mask": extended_attention_mask,
                "position_ids": position_ids,
            }
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
