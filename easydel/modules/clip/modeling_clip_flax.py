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
import typing as tp
from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxBaseModelOutputWithPooling,
	FlaxCLIPOutput,
	FlaxCLIPTextModelOutput,
	FlaxImageClassifierOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	control_mlp_sharding,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.clip.clip_configuration import (
	CLIPConfig,
	CLIPTextConfig,
	CLIPVisionConfig,
)


def contrastive_loss(logits: jax.Array) -> jax.Array:
	labels = jnp.arange(len(logits))
	return jnp.mean(
		-jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, len(logits)), axis=-1)
	)


def clip_loss(similarity: jax.Array) -> jax.Array:
	caption_loss = contrastive_loss(similarity)
	image_loss = contrastive_loss(similarity.T)
	return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(nn.Module):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		patch_embeds = self.patch_embedding(pixel_values)
		batch_size, height, width, channels = patch_embeds.shape
		patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

		class_embeds = jnp.expand_dims(self.class_embedding.value, axis=(0, 1))
		class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
		embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)

		embeddings = embeddings + self.position_embedding(
			jnp.expand_dims(
				jnp.arange(
					0, ((self.config.image_size // self.config.patch_size) ** 2) + 1, dtype="i4"
				),
				axis=0,
			)
		)
		return embeddings


class CLIPTextEmbeddings(nn.Module):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		input_embeds = self.token_embedding(input_ids.astype("i4"))
		position_embeds = self.position_embedding(position_ids.astype("i4"))

		embeddings = input_embeds + position_embeds
		return embeddings


class CLIPAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: tp.Union[CLIPTextConfig, CLIPVisionConfig],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
			nn.Linear,
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
			attention_dropout=config.attention_dropout,
			num_q_heads=config.num_attention_heads,
			num_kv_heads=config.num_attention_heads,
			head_dims=self.head_dim,
			precision=precision,
			force_float32_tpu=True,
			attn_mechanism=config.attn_mechanism,
			dtype=config.attn_dtype,
			mesh=config.mesh,
			sm_scale=self.head_dim**-0.5,
			axis_name=config.attention_axis_name,
			base_config=config,
		)

	def _split_heads(self, hidden_states):
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.num_heads, self.head_dim)
		)

	def _merge_heads(self, hidden_states):
		return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		causal_mask: tp.Optional[chex.Array] = None,
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
			assert causal_mask is not None
			query_length, key_length = query.shape[1], key.shape[1]
			causal_attention_mask = causal_mask[
				:, :, key_length - query_length : key_length, :key_length
			]

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

		attentions = self.attention_performer(
			query_states=query,
			key_states=key,
			value_states=value,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=self.causal,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query.shape[1],
			key_value_sequence_length=key.shape[1],
			uses_cache=None,
			segment_ids=None,
			causal_mask=causal_mask,
		)
		attn_output = self._merge_heads(attentions.attention_outputs)
		attn_output = self.out_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class CLIPMLP(nn.Module):
	def __init__(
		self,
		config: tp.Union[CLIPTextConfig, CLIPVisionConfig],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
			nn.Linear,
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
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
		return hidden_states


class CLIPEncoderLayer(nn.Module):
	def __init__(
		self,
		config: tp.Union[CLIPTextConfig, CLIPVisionConfig],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		attention_mask: tp.Optional[chex.Array] = None,
		causal_mask: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
	):
		residual = hidden_states

		hidden_states = self.layer_norm1(hidden_states)
		attn_outputs = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			output_attentions=output_attentions,
		)
		hidden_states = attn_outputs[0]
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.layer_norm2(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,) + attn_outputs[1:]

		return outputs


class CLIPEncoder(nn.Module):
	def __init__(
		self,
		config: tp.Union[CLIPTextConfig, CLIPVisionConfig],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		if isinstance(self.config, CLIPTextConfig):
			return self.config.get_basic_causal_mask()
		return None

	def __call__(
		self,
		inputs_embeds: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
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
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (hidden_states,)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
		)


class CLIPTextTransformer(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPTextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		return_dict: bool = True,
	):
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

		encoder_outputs = self.encoder(
			inputs_embeds=hidden_states,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		last_hidden_state = encoder_outputs[0]
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

		if not return_dict:
			return (last_hidden_state, pooled_output) + encoder_outputs[1:]

		return FlaxBaseModelOutputWithPooling(
			last_hidden_state=last_hidden_state,
			pooler_output=pooled_output,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)


class CLIPVisionTransformer(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		pixel_values: tp.Optional[chex.Array] = None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict: bool = True,
	):
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		hidden_states = self.embeddings(pixel_values)
		hidden_states = self.pre_layrnorm(hidden_states)

		encoder_outputs = self.encoder(
			inputs_embeds=hidden_states,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		last_hidden_state = encoder_outputs[0]
		pooled_output = last_hidden_state[:, 0, :]
		pooled_output = self.post_layernorm(pooled_output)

		if not return_dict:
			return (last_hidden_state, pooled_output) + encoder_outputs[1:]

		return FlaxBaseModelOutputWithPooling(
			last_hidden_state=last_hidden_state,
			pooler_output=pooled_output,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)


class CLIPTextModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPTextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		return_dict: bool = True,
	):
		return self.text_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)


class CLIPTextModelWithProjection(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPTextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.text_projection = nn.Linear(
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
		return_dict: bool = True,
	) -> tp.Union[FlaxCLIPTextModelOutput, tp.Tuple]:
		text_outputs = self.text_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		pooled_output = text_outputs[1]
		text_embeds = self.text_projection(pooled_output)

		if not return_dict:
			return (text_embeds, text_outputs[0]) + text_outputs[2:]

		return FlaxCLIPTextModelOutput(
			text_embeds=text_embeds,
			last_hidden_state=text_outputs.last_hidden_state,
			hidden_states=text_outputs.hidden_states,
			attentions=text_outputs.attentions,
		)


@register_module(
	config=CLIPVisionConfig,
	model_type="clip",
	task_type=TaskType.BASE_VISION,
	embedding_layer_names=[
		"position_embedding",
		"token_embedding",
	],
	layernorm_names=[
		"layer_norm1",
		"layer_norm2",
		"pre_layrnorm",
		"post_layernorm",
		"final_layer_norm",
	],
)
class CLIPVisionModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		return_dict: bool = True,
	):
		return self.vision_model(
			pixel_values=pixel_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)


@register_module(
	config=CLIPVisionConfig,
	model_type="clip",
	task_type=TaskType.IMAGE_CLASSIFICATION,
	embedding_layer_names=[
		"position_embedding",
		"token_embedding",
	],
	layernorm_names=[
		"layer_norm1",
		"layer_norm2",
		"pre_layrnorm",
		"post_layernorm",
		"final_layer_norm",
	],
)
class CLIPForImageClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
			config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.classifier = nn.Linear(
			config.vision_config.hidden_size,
			config.num_labels,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)

	def __call__(
		self,
		pixel_values: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
	) -> tp.Union[tuple, FlaxImageClassifierOutput]:
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		outputs = self.vision_model(
			pixel_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]

		sequence_output = jnp.mean(sequence_output[:, 1:, :], axis=1)
		if self.config.num_labels > 0:
			logits = self.classifier(sequence_output)
		else:
			logits = sequence_output

		if not return_dict:
			output = (logits,) + outputs[2:]
			return output

		return FlaxImageClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@register_module(
	config=CLIPConfig,
	model_type="clip",
	task_type=TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION,
	embedding_layer_names=[
		"position_embedding",
		"token_embedding",
	],
	layernorm_names=[
		"layer_norm1",
		"layer_norm2",
		"pre_layrnorm",
		"post_layernorm",
		"final_layer_norm",
	],
)
class CLIPModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: CLIPConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			kernel_init=jax.nn.initializers.normal(0.02),
			use_bias=False,
			rngs=rngs,
		)
		self.visual_projection = linear_class(
			config.vision_config.hidden_size, self.projection_dim
		)
		self.text_projection = linear_class(
			config.text_config.hidden_size, self.projection_dim
		)

		self.logit_scale = nn.Param(jnp.ones([]) * self.config.logit_scale_init_value)

	def __call__(
		self,
		input_ids: chex.Array,
		pixel_values: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	) -> tp.Union[FlaxCLIPOutput, tp.Tuple]:
		if attention_mask is None and input_ids is not None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None and attention_mask is not None:
			position_ids = attention_mask.cumsum(-1) - 1
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		vision_outputs = self.vision_model(
			pixel_values=pixel_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		text_outputs = self.text_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
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

		if not return_dict:
			return (
				logits_per_image,
				logits_per_text,
				text_embeds,
				image_embeds,
				text_outputs,
				vision_outputs,
			)

		return FlaxCLIPOutput(
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
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
	) -> tp.Tuple[tp.Any, FlaxCLIPOutput]:
		batch.pop("return_dict", None)
		outputs = self(**batch, return_dict=True)

		loss = LossMetrics(loss=clip_loss(outputs.logits_per_text))
		outputs = outputs.replace(loss=loss.loss)
		return outputs, loss
