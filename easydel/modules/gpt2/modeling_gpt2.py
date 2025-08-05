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
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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


import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    DecoderLayerOutput,
)
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

from .gpt2_configuration import GPT2Config as GPT2Config


class Conv1D(nn.Module):
    """Custom 1D Convolution layer used in GPT-2.

    This layer implements a 1D convolution operation often used as a substitute
    for linear layers in transformer models, particularly in earlier GPT architectures.
    It performs a matrix multiplication after transposing the kernel.

    Attributes:
            in_features (int): Dimensionality of the input features.
            out_features (int): Dimensionality of the output features.
            use_bias (bool): Whether to include a bias term. Defaults to True.
            dtype (jnp.dtype): Data type for computations. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            dot_general (tp.Optional[callable]): Custom dot_general function.
                Defaults to None (uses jax.lax.dot_general).
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        dot_general=None,
        *,
        rngs: nn.Rngs,
    ):
        self.kernel = nn.Param(
            nn.initializers.normal(stddev=0.02)(rngs.params(), (out_features, in_features)),
        )

        self.bias = nn.Param(
            nn.initializers.zeros(
                rngs.params(),
                (in_features,),
            )
            if use_bias
            else None
        )

        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dot_general = dot_general

    def __call__(self, inputs):
        """Forward pass of the Conv1D layer.

        Args:
            inputs (chex.Array): Input tensor.

        Returns:
            chex.Array: Output tensor after applying the 1D convolution.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        bias = self.bias.value
        kernel = self.kernel.value.transpose().astype(self.dtype)
        if self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general

        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y = y + bias.astype(self.dtype)
        return y


class GPT2Attention(AttentionModule):
    """GPT-2 Attention module.

    This module implements the standard multi-head self-attention mechanism used in GPT-2.
    It supports both self-attention and cross-attention.

    Attributes:
            config (GPT2Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            causal (bool): Whether the attention is causal.
            is_cross_attention (bool): Whether the attention is cross-attention.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.precision = precision
        self.dtype = dtype
        self.rngs = rngs
        self.is_cross_attention = is_cross_attention
        self.causal = causal
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.is_cross_attention:
            self.c_attn = Conv1D(
                self.embed_dim,
                2 * self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.q_attn = Conv1D(
                self.embed_dim,
                self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.c_attn = Conv1D(
                self.embed_dim,
                3 * self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.c_proj = Conv1D(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attn_pdrop,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (chex.Array): The hidden states with separate head dimensions.

        Returns:
            chex.Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        key_value_states: chex.Array,
        attention_mask: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        causal_mask: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass of the GPT2Attention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            key_value_states (chex.Array, optional): Key/value states for cross-attention.
                Defaults to None (self-attention).
            attention_mask (chex.Array): Mask to apply on the attention scores.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
                Defaults to None.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional):
                Cache view for key/value states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional):
                Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]: A tuple containing the attention output and optionally
                the attention weights.
        """
        is_cross_attention = key_value_states is not None

        if not is_cross_attention:
            qkv_out = self.c_attn(hidden_states)
            query, key, value = jnp.split(qkv_out, 3, axis=2)
        else:
            q_out = self.q_attn(hidden_states)
            (query,) = jnp.split(q_out, 1, axis=2)
            kv_out = self.c_attn(key_value_states)
            key, value = jnp.split(kv_out, 2, axis=2)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        init_attention_bias = lambda: None  # noqa
        if self.causal:
            (
                key,
                value,
                attention_mask,
                init_attention_bias,
                cache_view,
                cache_metadata,
            ) = self.concatenate(
                query=query,
                key=key,
                value=value,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                fcm_mask=None,
            )

        attn = self.attention_performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=mode,
            init_bias=init_attention_bias,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            attention_mask=attention_mask,
            causal=self.causal,
            segment_ids=None,
        )
        attn_output = self.shard_attention_prod(self._merge_heads(attn.attention_outputs))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attn.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GPT2MLP(nn.Module):
    """GPT-2 MLP module.

    This module implements the feed-forward network (MLP) used in the GPT-2 model.
    It consists of two Conv1D layers with a GELU activation in between.

    Attributes:
            config (GPT2Config): Configuration object for the model.
            intermediate_size (int): Dimensionality of the intermediate layer.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPT2Config,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        self.dtype = dtype
        self.rngs = rngs
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(
            embed_dim,
            intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.c_proj = Conv1D(
            intermediate_size,
            embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(
            rate=config.resid_pdrop,
            rngs=rngs,
        )

    def __call__(self, hidden_states):
        """Forward pass of the GPT2MLP module.

        Args:
            hidden_states (chex.Array): Input hidden states.

        Returns:
            chex.Array: Output hidden states after processing through the MLP.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class GPT2Block(nn.Module):
    """GPT-2 Transformer block.

    This module represents a single transformer block in the GPT-2 model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization. It can optionally include cross-attention layers.

    Attributes:
            config (GPT2Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        attn_block = GPT2Attention
        mlp_block = GPT2MLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        self.attn = attn_block(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ln_2 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if config.add_cross_attention:
            self.crossattention = attn_block(
                config=config,
                dtype=dtype,
                causal=True,
                is_cross_attention=True,
            )
            self.ln_cross_attn = nn.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.mlp = mlp_block(
            config=config,
            intermediate_size=inner_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        attention_mask=None,
        causal_mask=None,
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        encoder_hidden_states: chex.Array | None = None,
        encoder_attention_mask: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass of the GPT2Block module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array, optional): Mask to apply on the self-attention scores. Defaults to None.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior. Defaults to None.
            encoder_hidden_states (chex.Array, optional): Hidden states from the encoder for cross-attention.
                Defaults to None.
            encoder_attention_mask (chex.Array, optional): Mask for the encoder hidden states in cross-attention.
                Defaults to None.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional):
                Cache view for key/value states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional):
                Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            tp.Tuple[chex.Array, ...]: A tuple containing the output hidden states and
                optionally attention weights (self and cross).
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            None,
            attention_mask,
            mode,
            cache_view,
            cache_metadata,
            causal_mask,
            output_attentions,
        )

        hidden_states = attn_outputs.attention_output + residual
        cross_attention = None
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)

            cross_attn_outputs = self.crossattention(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                mode,
                None,
                None,
                causal_mask,
                output_attentions,
            )
            cross_attention = cross_attn_outputs.attention_weight
            hidden_states = residual + cross_attn_outputs.attention_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cross_attention=cross_attention,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GPT2Config, model_type="gpt2")
class GPT2Model(EasyDeLBaseModule):
    """GPT-2 model implementation.

    This class implements the main GPT-2 transformer model architecture, consisting of
    embedding layers (token and position), multiple GPT2Block layers, and a final
    layer normalization.

    Attributes:
            config (GPT2Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPT2Config,
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
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.dropout = nn.Dropout(rate=self.config.embd_pdrop, rngs=rngs)
        self.h = [
            GPT2Block(
                self.config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nn.LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        encoder_hidden_states: chex.Array | None = None,
        encoder_attention_mask: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPT2Model.

        Args:
            input_ids (chex.Array): Input token IDs, shape (batch_size, sequence_length).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token. Defaults to None.
            encoder_hidden_states (chex.Array, optional): Hidden states from an encoder model for cross-attention.
                Defaults to None.
            encoder_attention_mask (chex.Array, optional): Mask for the encoder hidden states. Defaults to None.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing
                precomputed key/value states. Defaults to None.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
                Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers. Defaults to False.
                Defaults to True.

        Returns:
            Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]: Model outputs
                (last hidden state, optional past KVs, optional hidden states, optional attentions,
                    optional cross-attentions).
        """
        batch_size, sequence_length = input_ids.shape
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

        inputs_embeds = self.wte(input_ids.astype("i4"))
        position_embeds = self.wpe(position_ids.astype("i4"))

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.h))
        for idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_mask=self.causal_mask,
                mode=mode,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs.cross_attention,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
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
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=GPT2Config, model_type="gpt2")
class GPT2LMHeadModel(EasyDeLBaseModule):
    """GPT-2 model with a language modeling head.

    This model extends the base GPT2Model by adding a linear layer on top to
    predict the next token in a sequence, making it suitable for causal language
    modeling tasks.

    Attributes:
            config (GPT2Config): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    loss_type: str = "ForCausalLM"

    def __init__(
        self,
        config: GPT2Config,
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
        self.transformer = GPT2Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        encoder_hidden_states: chex.Array | None = None,
        encoder_attention_mask: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPT2LMHeadModel.

        Args:
            input_ids (chex.Array): Input token IDs, shape (batch_size, sequence_length).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token. Defaults to None.
            encoder_hidden_states (chex.Array, optional): Hidden states from an encoder model for cross-attention.
                Defaults to None.
            encoder_attention_mask (chex.Array, optional): Mask for the encoder hidden states. Defaults to None.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing precomputed
                key/value states. Defaults to None.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
                Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers. Defaults to False.
             Defaults to True.

        Returns:
            Union[CausalLMOutputWithCrossAttentions, Tuple]: Model outputs (logits, optional past KVs,
                optional hidden states, optional attentions, optional cross-attentions).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
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
        return self.transformer

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.transformer.wte
