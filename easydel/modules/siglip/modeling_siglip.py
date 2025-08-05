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
from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import image as jimg

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    EncoderLayerOutput,
    ImageClassifierOutput,
    ModelOutput,
)
from easydel.infra.utils import ACT2FN
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.linear import ParallelLinear

from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig


@auto_pytree
class SiglipVisionModelOutput(ModelOutput):
    image_embeds: chex.Array | None = None
    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


@auto_pytree
class SiglipTextModelOutput(ModelOutput):
    text_embeds: chex.Array | None = None
    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


@auto_pytree
class SiglipOutput(ModelOutput):
    loss: chex.Array | None = None
    logits_per_image: chex.Array = None
    logits_per_text: chex.Array = None
    text_embeds: chex.Array = None
    image_embeds: chex.Array = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[tp.Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class SiglipVisionEmbeddings(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embed(
            self.num_positions,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.patch_embedding = nn.Conv(
            in_features=config.num_channels,
            out_features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            precision=precision,
        )

    def interpolate(self, embeddings: chex.Array, height: int, width: int):
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        if num_patches == num_positions and height == width:
            return self.position_embedding(
                jnp.arange(
                    self.num_positions,
                    dtype="i4",
                ).reshape(1, -1)
            )
        patch_pos_embed = self.position_embedding.embedding.unsqueeze(0)

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)

        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, sqrt_num_positions, sqrt_num_positions, dim))
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))

        patch_pos_embed = jimg.resize(
            patch_pos_embed,
            (1, dim, new_height, new_width),
            method="cubic",
        )

        return jnp.reshape(jnp.transpose(patch_pos_embed, (0, 2, 3, 1)), (1, -1, dim))

    def __call__(self, pixel_values: chex.Array, interpolate_pos_encoding=False):
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.kernel.dtype

        pixel_values = pixel_values.transpose(0, 2, 3, 1).astype(dtype=target_dtype)
        patch_embeds = self.patch_embedding(pixel_values).transpose(0, 3, 1, 2)

        embeddings = jnp.reshape(patch_embeds, (*patch_embeds.shape[:2], -1))
        embeddings = jnp.transpose(embeddings, (0, 2, 1))
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(jnp.arange(self.num_positions, dtype="i4").reshape(1, -1))
        return embeddings


class SiglipTextEmbeddings(nn.Module):
    def __init__(
        self,
        config: SiglipTextConfig,
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

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
    ) -> chex.Array:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.embedding.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype="i4").reshape(1, -1)

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class SiglipAttention(AttentionModule):
    def __init__(
        self,
        config,
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

        self.causal = False
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        causal_attention_mask = None
        if self.causal:
            raise NotImplementedError()
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
            attention_bias = jax.lax.select(
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
        )


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config: SiglipTextConfig,
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

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
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


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: SiglipTextConfig,
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
        self.self_attn = SiglipAttention(
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
        self.mlp = SiglipMLP(
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
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
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


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config: SiglipTextConfig,
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
            SiglipEncoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs_embeds: chex.Array,
        attention_mask: chex.Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        hidden_states = inputs_embeds
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
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


class SiglipTextTransformer(EasyDeLBaseModule):
    def __init__(
        self,
        config: SiglipTextConfig,
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
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = SiglipEncoder(
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
        self.head = ParallelLinear(
            embed_dim,
            config.projection_size,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

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


@register_module(TaskType.BASE_MODULE, config=SiglipTextConfig, model_type="siglip_text_model")
class SiglipTextModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: SiglipTextConfig,
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
        self.text_model = SiglipTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
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
        This model has a projection head, not a language model head.
        """
        raise NotImplementedError("This model has a projection head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.text_model.embeddings


class SiglipVisionTransformer(EasyDeLBaseModule):
    def __init__(
        self,
        config: SiglipTextConfig,
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
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = SiglipEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_layernorm = nn.LayerNorm(
            embed_dim,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def __call__(
        self,
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = False,
    ) -> tuple | BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
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


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        def normal_init(*shape):
            return nn.initializers.xavier_uniform()(rngs.param(), shape, param_dtype)

        def ze_init(*shape):
            return jnp.zeros(shape, param_dtype)

        self.in_proj_weight = nn.Param(normal_init(embed_dim * 3, embed_dim))
        self.in_proj_bias = nn.Param(ze_init(3 * embed_dim))
        self.out_proj = ParallelLinear(
            embed_dim,
            embed_dim,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
    ):
        qbs, qss, qds = query.shape
        b, s, d = value.shape

        qb, kb, vb = jnp.split(self.in_proj_bias, 3, -1)
        qw, kw, vw = jnp.split(self.in_proj_weight, 3, -1)

        qout = ((query @ qw) + qb).reshape(qbs, qss, self.num_heads, -1)
        kout = ((key @ kw) + kb).reshape(b, s, self.num_heads, -1)
        vout = ((value @ vw) + vb).reshape(b, s, self.num_heads, -1)

        attn = jnp.einsum(
            "bhqk,bkhd->bqhd",
            jax.nn.softmax(
                jnp.einsum(
                    "bqhd,bkhd->bhqk",
                    qout * (qout.shape[-1] ** -0.5),
                    kout,
                )
            ),
            vout,
        )

        return self.out_proj(attn.reshape(qbs, qss, qds))


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.probe = nn.Param(
            jax.random.normal(
                rngs.param(),
                (1, 1, config.hidden_size),
                param_dtype,
            )
        )
        self.attention = MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = SiglipMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.value.repeat(batch_size, 0)
        hidden_state = self.attention(probe, hidden_state, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


@register_module(TaskType.BASE_VISION, config=SiglipVisionConfig, model_type="siglip_vision_model")
class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.vision_model = SiglipVisionTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
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


@register_module(TaskType.BASE_MODULE, config=SiglipConfig, model_type="siglip")
class SiglipModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: SiglipConfig,
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
        if not isinstance(config.text_config, SiglipTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        text_model = SiglipTextModel(
            text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        vision_model = SiglipVisionModel(
            vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Param(jax.random.normal(rngs.param(), (1,), param_dtype))
        self.logit_bias = nn.Param(jax.random.normal(rngs.param(), (1,), param_dtype))

    def get_text_features(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> chex.Array:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    def get_image_features(
        self,
        pixel_values: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> chex.Array:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        pixel_values: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        return_loss: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | SiglipOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / jnp.linalg.norm(
            image_embeds,
            ord=2,
            axis=-1,
            keepdims=True,
        )
        text_embeds = text_embeds / jnp.linalg.norm(
            text_embeds,
            ord=2,
            axis=-1,
            keepdims=True,
        )

        # cosine similarity as logits
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T)

        logit_scale, logit_bias = (self.logit_scale, self.logit_bias)
        logits_per_text = logits_per_text * jnp.exp(logit_scale) + logit_bias

        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            m1_diag1 = -jnp.ones_like(logits_per_text) + 2 * jnp.eye(logits_per_text.shape[0])
            loglik = jax.nn.log_sigmoid(m1_diag1 * logits_per_text)
            nll = -jnp.sum(loglik, axis=-1)
            loss = nll.mean()

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        The text model acts as the decoder in this multi-modal setup.
        """
        return self.text_model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model does not have a traditional language model head, but a projection head.
        """
        raise NotImplementedError("This model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the text model.
        """
        return self.text_model.embeddings


@register_module(TaskType.IMAGE_CLASSIFICATION, config=SiglipConfig, model_type="siglip")
class SiglipForImageClassification(EasyDeLBaseModule):
    def __init__(
        self,
        config: SiglipConfig,
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
        self.num_labels = config.num_labels
        vision_model = SiglipVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_model = vision_model.vision_model
        self.use_classif = config.num_labels > 0
        # Classifier head
        if self.use_classif:
            self.classifier = ParallelLinear(
                config.vision_config.hidden_size,
                config.num_labels,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def __call__(
        self,
        pixel_values: chex.Array | None = None,
        labels: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | ImageClassifierOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        logits = jnp.mean(sequence_output, axis=1)
        if self.use_classif:
            logits = self.classifier(logits)

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
