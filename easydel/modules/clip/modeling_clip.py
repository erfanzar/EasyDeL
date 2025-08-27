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
import typing as tp
from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CLIPOutput,
    CLIPTextModelOutput,
    EncoderLayerOutput,
    ImageClassifierOutput,
)
from easydel.infra.utils import ACT2FN
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.linear import ParallelLinear

from .clip_configuration import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def contrastive_loss(logits: jax.Array) -> jax.Array:
    """
    Computes the contrastive loss.

    Args:
            logits (jax.Array): Logits from the model.

    Returns:
            jax.Array: Contrastive loss.
    """
    labels = jnp.arange(len(logits))
    return jnp.mean(-jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, len(logits)), axis=-1))


def clip_loss(similarity: jax.Array) -> jax.Array:
    """
    Computes the CLIP loss.

    Args:
            similarity (jax.Array): Similarity matrix.

    Returns:
            jax.Array: CLIP loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(nn.Module):
    """
    Constructs the vision embeddings for CLIP.

    Attributes:
            config (CLIPVisionConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        embed_dim = config.hidden_size
        image_size = config.image_size
        patch_size = config.patch_size

        self.class_embedding = nn.Param(
            jax.nn.initializers.normal(stddev=0.02)(
                rngs.params(),
                shape=(embed_dim,),
                dtype=param_dtype,
            ),
        )

        self.patch_embedding = nn.Conv(
            config.num_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(),
            rngs=rngs,
        )

        self.num_patches = (image_size // patch_size) ** 2
        num_positions = self.num_patches + 1
        self.position_embedding = nn.Embed(
            num_positions,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, pixel_values):
        """
        Forward pass for vision embeddings.

        Args:
                pixel_values (chex.Array): Input pixel values (batch_size, num_channels, height, width).

        Returns:
                chex.Array: Combined class and patch embeddings.
        """
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        class_embeds = jnp.expand_dims(self.class_embedding.value, axis=(0, 1))
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)

        embeddings = embeddings + self.position_embedding(
            jnp.expand_dims(
                jnp.arange(0, ((self.config.image_size // self.config.patch_size) ** 2) + 1, dtype="i4"),
                axis=0,
            )
        )
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    """
    Constructs the text embeddings for CLIP.

    Attributes:
            config (CLIPTextConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embed(
            config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.position_embedding = nn.Embed(
            config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, input_ids, position_ids):
        """
        Forward pass for text embeddings.

        Args:
                input_ids (chex.Array): Input token IDs.
                position_ids (chex.Array): Position IDs.

        Returns:
                chex.Array: Combined token and position embeddings.
        """
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        position_embeds = self.position_embedding(position_ids.astype("i4"))

        embeddings = input_embeds + position_embeds
        return embeddings


class CLIPAttention(AttentionModule):
    """
    CLIP Attention module, supporting both text (causal) and vision (non-causal) attention.

    Attributes:
            config (Union[CLIPTextConfig, CLIPVisionConfig]): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
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
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.dropout = config.attention_dropout
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.k_proj = linear_class(self.embed_dim, self.embed_dim)
        self.v_proj = linear_class(self.embed_dim, self.embed_dim)
        self.q_proj = linear_class(self.embed_dim, self.embed_dim)
        self.out_proj = linear_class(self.embed_dim, self.embed_dim)

        self.causal = isinstance(config, CLIPTextConfig)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _split_heads(self, hidden_states):
        """
        Splits hidden states into multiple heads.

        Args:
                hidden_states (chex.Array): Input hidden states.

        Returns:
                chex.Array: Reshaped hidden states.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """
        Merges multiple heads back into a single hidden state tensor.

        Args:
                hidden_states (chex.Array): Input hidden states.

        Returns:
                chex.Array: Merged hidden states.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array | None = None,
        causal_mask: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        """
        Forward pass for the CLIP attention module.

        Args:
                hidden_states (chex.Array): Input hidden states.
                attention_mask (Optional[chex.Array]): Mask to prevent attention to certain positions.
                causal_mask (Optional[chex.Array]): Causal mask for text attention.
                output_attentions (bool): Whether to output attention weights.

        Returns:
                Tuple[chex.Array, Optional[chex.Array]]: Attention output and optionally attention weights.
        """
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        causal_attention_mask = None
        if self.causal:
            assert causal_mask is not None
            query_length, key_length = query.shape[1], key.shape[1]
            causal_attention_mask = causal_mask[:, :, key_length - query_length : key_length, :key_length]

        if attention_mask is not None and causal_attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = nn.combine_masks(
                attention_mask,
                causal_attention_mask,
                dtype="i4",
            )
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        elif attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_bias = None
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attention_mask = None

        attentions = self.attention_performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            bias=None,
            init_bias=lambda: attention_bias,
            attention_mask=attention_mask,
            segment_ids=None,
            causal=self.causal,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.out_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=None,
        )


class CLIPMLP(nn.Module):
    """
    CLIP MLP (Feed-Forward) layer.

    Attributes:
            config (Union[CLIPTextConfig, CLIPVisionConfig]): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
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
        self.rngs = rngs
        self.activation_fn = ACT2FN[config.hidden_act]
        linear_class = partial(
            ParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc1 = linear_class(config.hidden_size, config.intermediate_size)
        self.fc2 = linear_class(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: chex.Array):
        """
        Forward pass for the MLP layer.

        Args:
                hidden_states (chex.Array): Input hidden states.

        Returns:
                chex.Array: Output hidden states.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """
    Single CLIP encoder layer, combining self-attention and MLP.

    Attributes:
            config (Union[CLIPTextConfig, CLIPVisionConfig]): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
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
        self.rngs = rngs
        self.self_attn = CLIPAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = CLIPMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array | None = None,
        causal_mask: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        """
        Forward pass for the encoder layer.

        Args:
                hidden_states (chex.Array): Input hidden states.
                attention_mask (Optional[chex.Array]): Attention mask.
                causal_mask (Optional[chex.Array]): Causal mask (for text).
                output_attentions (bool): Whether to output attention weights.

        Returns:
                Tuple[chex.Array, ...]: Output hidden states and optional attention weights.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
        )


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `CLIPEncoderLayer` layers.

    Attributes:
            config (Union[CLIPTextConfig, CLIPVisionConfig]): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
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
        self.rngs = rngs
        self.layers = [
            CLIPEncoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]

    @cached_property
    def causal_mask(self):
        """
        Returns the causal mask if the encoder is for text, otherwise None.

        Returns:
                Optional[chex.Array]: Causal mask.
        """
        if isinstance(self.config, CLIPTextConfig):
            return self.config.get_basic_causal_mask()
        return None

    def __call__(
        self,
        inputs_embeds: chex.Array,
        attention_mask: chex.Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Forward pass for the CLIP encoder.

        Args:
                inputs_embeds (chex.Array): Input embeddings.
                attention_mask (Optional[chex.Array]): Attention mask.
                output_attentions (bool): Whether to output attention weights.
                output_hidden_states (bool): Whether to output all hidden states.


        Returns:
                Union[BaseModelOutput, Tuple]: Encoder output
                    (last hidden state, optional hidden states, optional attentions).
        """
        hidden_states = inputs_embeds
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CLIPTextTransformer(EasyDeLBaseModule):
    """
    The transformer encoder for the CLIP text model.

    Attributes:
            config (CLIPTextConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
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
        self.embeddings = CLIPTextEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = CLIPEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.eos_token_id = self.config.eos_token_id

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for the text transformer.

        Args:
                input_ids (chex.Array): Input token IDs.
                attention_mask (chex.Array): Attention mask.
                position_ids (chex.Array): Position IDs.
                output_attentions (bool): Whether to output attention weights.
                output_hidden_states (bool): Whether to output all hidden states.


        Returns:
                Union[BaseModelOutputWithPooling, Tuple]: Transformer output (last hidden state, pooled output,
                    optional hidden states, optional attentions).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                jnp.arange(last_hidden_state.shape[0]),
                input_ids.argmax(axis=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                jnp.arange(last_hidden_state.shape[0]),
                (input_ids == self.eos_token_id).argmax(axis=-1),
            ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has a projection head, not a language model head.
        """
        raise NotImplementedError("This model has a projection head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


class CLIPVisionTransformer(EasyDeLBaseModule):
    """
    The transformer encoder for the CLIP vision model.

    Attributes:
            config (CLIPVisionConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
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
        self.embeddings = CLIPVisionEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pre_layrnorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.encoder = CLIPEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        pixel_values: chex.Array | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """Forward pass for the vision transformer.

        Args:
                pixel_values (Optional[chex.Array]): Input pixel values.
                output_attentions (Optional[bool]): Whether to output attention weights.
                output_hidden_states (Optional[bool]): Whether to output all hidden states.


        Returns:
                Union[BaseModelOutputWithPooling, Tuple]: Transformer output (last hidden state, pooled output, optional
                    hidden states, optional attentions).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is not None and pixel_values.ndim == 4:
            pixel_values = jnp.swapaxes(pixel_values, 1, 3)
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This vision model does not have a language model head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


class CLIPTextModel(EasyDeLBaseModule):
    """
    Bare CLIP text model (transformer) outputting raw hidden-states without any specific head on top.

    Attributes:
            config (CLIPTextConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
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
        self.text_model = CLIPTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for the bare CLIP text model.

        Args:
                input_ids (chex.Array): Input token IDs.
                attention_mask (chex.Array): Attention mask.
                position_ids (chex.Array): Position IDs.
                output_attentions (bool): Whether to output attention weights.
                output_hidden_states (bool): Whether to output all hidden states.


        Returns:
                Union[BaseModelOutputWithPooling, Tuple]: Model output.
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.text_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

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
        return self.text_model.embeddings


class CLIPTextModelWithProjection(EasyDeLBaseModule):
    """
    CLIP text model with a projection layer on top.

    Attributes:
            config (CLIPTextConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
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
        self.text_model = CLIPTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.text_projection = ParallelLinear(
            config.hidden_size,
            config.projection_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Forward pass for the CLIP text model with projection.

        Args:
                input_ids (chex.Array): Input token IDs.
                attention_mask (chex.Array): Attention mask.
                position_ids (chex.Array): Position IDs.
                output_attentions (bool): Whether to output attention weights.
                output_hidden_states (bool): Whether to output all hidden states.


        Returns:
                Union[CLIPTextModelOutput, Tuple]: Model output including projected text embeddings.
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]
        text_embeds = self.text_projection(pooled_output)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.text_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has a projection head, not a language model head.
        """
        raise NotImplementedError("This model has a projection head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.text_model.embeddings


@register_module(config=CLIPVisionConfig, model_type="clip_vision_model", task_type=TaskType.BASE_VISION)
@register_module(config=CLIPVisionConfig, model_type="clip_vision_model", task_type=TaskType.BASE_MODULE)
class CLIPVisionModel(EasyDeLBaseModule):
    """
    Bare CLIP vision model (transformer) outputting raw hidden-states without any specific head on top.

    Attributes:
            config (CLIPVisionConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
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
        self.vision_model = CLIPVisionTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        pixel_values: chex.Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for the bare CLIP vision model.

        Args:
                pixel_values (chex.Array): Input pixel values.
                output_attentions (bool): Whether to output attention weights.
                output_hidden_states (bool): Whether to output all hidden states.


        Returns:
                Union[BaseModelOutputWithPooling, Tuple]: Model output.
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This vision model does not have a language model head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.vision_model.embeddings


@register_module(config=CLIPVisionConfig, model_type="clip", task_type=TaskType.IMAGE_CLASSIFICATION)
class CLIPForImageClassification(EasyDeLBaseModule):
    """
    CLIP vision model with an image classification head on top (a linear layer on the pooled final hidden state).

    Attributes:
            config (CLIPVisionConfig): Configuration object.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): JAX precision level.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the CLIPForImageClassification model."""
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_model = CLIPVisionTransformer(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.classifier = ParallelLinear(
            config.vision_config.hidden_size,
            config.num_labels,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    def __call__(
        self,
        pixel_values: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | ImageClassifierOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = jnp.mean(sequence_output[:, 1:, :], axis=1)
        if self.config.num_labels > 0:
            logits = self.classifier(sequence_output)
        else:
            logits = sequence_output

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model for classification.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has an image classification head, not a language model head.
        """
        raise NotImplementedError("This model has an image classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.vision_model.embeddings


@register_module(config=CLIPConfig, model_type="clip", task_type=TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION)
class CLIPModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: CLIPConfig,
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
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(
            text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_model = CLIPVisionTransformer(
            vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
            rngs=rngs,
        )
        self.visual_projection = linear_class(config.vision_config.hidden_size, self.projection_dim)
        self.text_projection = linear_class(config.text_config.hidden_size, self.projection_dim)

        self.logit_scale = nn.Param(jnp.ones([]) * self.config.logit_scale_init_value)

    def __call__(
        self,
        input_ids: chex.Array,
        pixel_values: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> CLIPOutput:
        if attention_mask is None and input_ids is not None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.cumsum(-1) - 1

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        return CLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_text_features(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    def get_image_features(self, pixel_values: chex.Array):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)
        return image_features

    def compute_loss(
        self,
        *,
        labels=None,  # just to extract
        loss_config=None,  # just to extract
        loss_kwargs=None,  # just to extract
        **batch,
    ) -> tuple[tp.Any, CLIPOutput]:
        outputs = self(**batch)

        loss = LossMetrics(loss=clip_loss(outputs.logits_per_text))
        outputs = outputs.replace(loss=loss.loss)
        return outputs, loss

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        The text model acts as the "decoder" or text processor in this multi-modal setup.
        """
        return self.text_model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model does not have a traditional language model head, but projection heads.
        """
        raise NotImplementedError("This model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the text model.
        """
        return self.text_model.embeddings
