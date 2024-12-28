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


import math
import random
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

# import transformers
from flax import nnx as nn
from jax import lax

from easydel.inference.logits_process import (
	FlaxLogitsProcessorList,
	FlaxStaticForceTokensLogitsProcessor,
	WhisperTimeStampLogitsProcessor,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxBaseModelOutputWithPastAndCrossAttentions,
	FlaxSeq2SeqLMOutput,
	FlaxSeq2SeqModelOutput,
	FlaxSequenceClassifierOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching.transformer_cache import (
	TransformerCache,
	TransformerCacheView,
)
from easydel.modules.whisper.whisper_configuration import WhisperConfig as WhisperConfig

remat = nn.remat


def sinusoidal_embedding_init(key, shape, dtype=jnp.float_) -> jax.Array:
	length, channels = shape
	if channels % 2 != 0:
		raise ValueError(
			f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
		)
	log_timescale_increment = math.log(10000) / (channels // 2 - 1)
	inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
	scaled_time = jnp.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
	return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1).astype(
		dtype
	)


class WhisperAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: WhisperConfig,
		embed_dim: int,
		num_heads: int,
		dropout: float = 0.0,
		causal: bool = False,
		bias: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__(config=config)
		self.rngs = rngs
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.dropout = dropout
		self.causal = causal
		self.bias = bias
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.head_dim = self.embed_dim // self.num_heads
		if self.head_dim * self.num_heads != self.embed_dim:
			raise ValueError(
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
				f" and `num_heads`: {self.num_heads})."
			)

		linear = partial(
			nn.Linear,
			self.embed_dim,
			self.embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(self.config.init_std),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj = linear(use_bias=self.bias, rngs=rngs)
		self.k_proj = linear(use_bias=False, rngs=rngs)
		self.v_proj = linear(use_bias=self.bias, rngs=rngs)
		self.out_proj = linear(use_bias=self.bias, rngs=rngs)

		self.attention_performer = FlexibleAttentionModule(
			use_sharding_constraint=self.config.use_sharding_constraint,
			num_q_heads=self.num_heads,
			num_kv_heads=self.num_heads,
			attention_dropout=self.config.attention_dropout,
			head_dims=self.head_dim,
			shard_attention_computation=self.config.shard_attention_computation,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			dtype=self.config.attn_dtype,
			partition_axis=self.config.partition_axis,
			scan_ring_attention=self.config.scan_ring_attention,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			axis_name=self.config.attention_axis_name,
			backward_pass_impl=self.config.flash_attention_backward_pass_impl,
		)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		key_value_states: tp.Optional[jnp.ndarray] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		attention_mask: tp.Optional[jnp.ndarray] = None,
		causal_mask: tp.Optional[jnp.ndarray] = None,
	) -> tuple[tp.Any, tp.Any]:
		is_cross_attention = key_value_states is not None
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

		if self.causal:
			assert causal_mask is not None, "seems like you forgot to pass causal_mask"
			(
				key_states,
				value_states,
				attention_mask,
				attention_bias,
			) = self.concatenate(
				query=query_states,
				key=key_states,
				cache_view=cache_view,
				value=value_states,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				fcm_mask=None,
			)
		else:
			if attention_mask is not None:
				attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
			attention_bias = None

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=self.causal,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=None,
			causal_mask=causal_mask,
		)

		attn_output = self.out_proj(
			self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
		)

		return attn_output, attentions.attention_outputs

	def _split_heads(self, hidden_state) -> jnp.ndarray:
		return hidden_state.reshape(
			hidden_state.shape[:2] + (self.num_heads, self.head_dim)
		)

	def _merge_heads(self, hidden_state) -> jnp.ndarray:
		return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))


class WhisperEncoderLayer(nn.Module):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.embed_dim = self.config.d_model

		self.self_attn = WhisperAttention(
			config=config,
			embed_dim=self.embed_dim,
			num_heads=config.encoder_attention_heads,
			dropout=config.attention_dropout,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		linear = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(self.config.init_std),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.self_attn_layer_norm = nn.LayerNorm(
			self.embed_dim,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)
		self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)
		self.activation_fn = ACT2FN[self.config.activation_function]
		self.activation_dropout_layer = nn.Dropout(
			rate=self.config.activation_dropout,
			rngs=rngs,
		)
		self.fc1 = linear(self.embed_dim, self.config.encoder_ffn_dim, rngs=rngs)
		self.fc2 = linear(self.config.encoder_ffn_dim, self.embed_dim, rngs=rngs)
		self.final_layer_norm = nn.LayerNorm(
			self.embed_dim,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		attention_mask: jnp.ndarray,
		causal_mask: tp.Optional[jnp.ndarray] = None,
		output_attentions: bool = True,
	) -> tp.Tuple[jnp.ndarray]:
		residual = hidden_states
		hidden_states = self.self_attn_layer_norm(hidden_states)
		hidden_states, attn_weights = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			cache_view=None,
			key_value_states=None,
		)
		hidden_states = self.dropout_layer(hidden_states)
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.final_layer_norm(hidden_states)
		hidden_states = self.activation_fn(self.fc1(hidden_states))
		hidden_states = self.activation_dropout_layer(hidden_states)
		hidden_states = self.fc2(hidden_states)
		hidden_states = self.dropout_layer(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attn_weights,)

		return outputs


class WhisperDecoderLayer(nn.Module):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.embed_dim = self.config.d_model
		self.self_attn = WhisperAttention(
			config=self.config,
			embed_dim=self.embed_dim,
			num_heads=self.config.decoder_attention_heads,
			dropout=self.config.attention_dropout,
			causal=True,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.dropout_layer = nn.Dropout(
			rate=self.config.dropout,
			rngs=rngs,
		)
		self.activation_fn = ACT2FN[self.config.activation_function]
		self.activation_dropout_layer = nn.Dropout(
			rate=self.config.activation_dropout,
			rngs=rngs,
		)

		self.self_attn_layer_norm = nn.LayerNorm(
			self.embed_dim,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)
		self.encoder_attn = WhisperAttention(
			config=self.config,
			embed_dim=self.embed_dim,
			num_heads=self.config.decoder_attention_heads,
			dropout=self.config.attention_dropout,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.encoder_attn_layer_norm = nn.LayerNorm(
			self.embed_dim,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)
		linear = partial(
			nn.Linear,
			param_dtype=self.param_dtype,
			precision=self.precision,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(self.config.init_std),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.fc1 = linear(
			self.embed_dim,
			self.config.decoder_ffn_dim,
			rngs=rngs,
		)
		self.fc2 = linear(
			self.config.decoder_ffn_dim,
			self.embed_dim,
			rngs=rngs,
		)
		self.final_layer_norm = nn.LayerNorm(
			self.embed_dim,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		attention_mask: jnp.ndarray,
		causal_mask: tp.Optional[jnp.ndarray] = None,
		encoder_hidden_states: tp.Optional[jnp.ndarray] = None,
		encoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = True,
	) -> tp.Tuple[jnp.ndarray]:
		residual = hidden_states
		hidden_states = self.self_attn_layer_norm(hidden_states)

		# Self Attention
		hidden_states, self_attn_weights = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			cache_view=cache_view,
		)
		hidden_states = self.dropout_layer(hidden_states)
		hidden_states = residual + hidden_states

		# Cross-Attention Block
		cross_attn_weights = None
		if encoder_hidden_states is not None:
			residual = hidden_states

			hidden_states = self.encoder_attn_layer_norm(hidden_states)
			hidden_states, cross_attn_weights = self.encoder_attn(
				hidden_states=hidden_states,
				causal_mask=causal_mask,
				key_value_states=encoder_hidden_states,
				attention_mask=encoder_attention_mask,
			)
			hidden_states = self.dropout_layer(hidden_states)
			hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.final_layer_norm(hidden_states)
		hidden_states = self.activation_fn(self.fc1(hidden_states))
		hidden_states = self.activation_dropout_layer(hidden_states)
		hidden_states = self.fc2(hidden_states)
		hidden_states = self.dropout_layer(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights, cross_attn_weights)

		return outputs


class WhisperEncoder(EasyDeLBaseModule):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
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
		self.conv1 = nn.Conv(
			self.config.d_model,
			self.config.d_model,
			kernel_size=(3,),
			padding=1,
			kernel_init=jax.nn.initializers.normal(self.config.init_std),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.conv2 = nn.Conv(
			self.config.d_model,
			self.config.d_model,
			kernel_size=(3,),
			strides=2,
			padding=1,
			kernel_init=jax.nn.initializers.normal(self.config.init_std),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.dropout_layer = nn.Dropout(
			rate=self.config.dropout,
			rngs=rngs,
		)

		block = WhisperEncoderLayer
		self.layers = [
			block(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.encoder_layers)
		]

		self.embed_positions = nn.Embed(
			self.config.max_source_positions,
			self.config.d_model,
			dtype=self.dtype,
			embedding_init=sinusoidal_embedding_init,
			param_dtype=self.param_dtype,
			rngs=rngs,
		)

		self.layer_norm = nn.LayerNorm(
			self.config.d_model,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)
		self.layerdrop = self.config.decoder_layerdrop

	def __call__(
		self,
		input_features: jnp.ndarray,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	) -> tuple[tp.Any | None, ...] | FlaxBaseModelOutput:
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		if input_features.shape[1:] != (
			self.config.num_mel_bins,
			self.config.max_source_positions * 2,
		):
			raise ValueError(
				"input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
				f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
				f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
			)

		input_features = input_features.transpose(0, 2, 1)
		hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
		hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

		embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
		# freeze the sinusoidal embeddings by stopping the back-prop
		embed_positions = jax.lax.stop_gradient(embed_positions)
		hidden_states = hidden_states + embed_positions

		hidden_states = self.dropout_layer(hidden_states)

		for encoder_layer in self.layers:
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)
			dropout_probability = random.uniform(0, 1)
			if not self.dropout_layer.deterministic and (
				dropout_probability < self.layerdrop
			):  # skip the layer
				layer_outputs = (None, None)
			else:
				layer_outputs = encoder_layer(
					hidden_states=hidden_states,
					causal_mask=None,
					attention_mask=None,
					output_attentions=output_attentions,
				)
			hidden_states = layer_outputs[0]
			if output_attentions:
				all_attentions = all_attentions + (layer_outputs[1],)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		hidden_states = self.layer_norm(hidden_states)
		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (hidden_states, all_hidden_states, all_attentions)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
		)


class WhisperDecoder(EasyDeLBaseModule):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
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
			self.config.d_model,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.embed_positions = nn.Embed(
			self.config.max_target_positions,
			self.config.d_model,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			WhisperDecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.decoder_layers)
		]

		self.layerdrop = self.config.decoder_layerdrop
		self.dropout_layer = nn.Dropout(
			rate=self.config.dropout,
			rngs=rngs,
		)

		self.layer_norm = nn.LayerNorm(
			self.config.d_model,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			epsilon=1e-05,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: jnp.ndarray,
		attention_mask: jnp.ndarray,
		position_ids: jnp.ndarray,
		encoder_hidden_states: tp.Optional[jnp.ndarray] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	) -> tuple[tp.Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
		inputs_embeds = self.embed_tokens(input_ids)
		position_embeds = self.embed_positions(position_ids)

		hidden_states = inputs_embeds + position_embeds
		hidden_states = self.dropout_layer(hidden_states)

		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_cross_attentions = (
			() if (output_attentions and encoder_hidden_states is not None) else None
		)
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		for idx, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
				# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			dropout_probability = random.uniform(0, 1)
			if not self.dropout_layer.deterministic and (
				dropout_probability < self.layerdrop
			):
				layer_outputs = (None, None, None)
			else:
				layer_outputs = decoder_layer(
					hidden_states=hidden_states,
					attention_mask=attention_mask,
					causal_mask=self.causal_mask,
					encoder_hidden_states=encoder_hidden_states,
					encoder_attention_mask=None,
					cache_view=past_key_values.views[idx],
					output_attentions=output_attentions,
				)

			hidden_states = layer_outputs[0]
			if output_attentions:
				all_self_attns += (layer_outputs[1],)

				if encoder_hidden_states is not None:
					all_cross_attentions += (layer_outputs[2],)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		hidden_states = self.layer_norm(hidden_states)
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = [
			hidden_states,
			all_hidden_states,
			all_self_attns,
			all_cross_attentions,
		]

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			cross_attentions=all_cross_attentions,
		)


@register_module(
	"base-module",
	config=WhisperConfig,
	model_type="whisper",
	embedding_layer_names=["embed_positions", "embed_tokens"],
	layernorm_names=[
		"self_attn_layer_norm",
		"final_layer_norm",
		"encoder_attn_layer_norm",
		"layer_norm",
	],
)
class WhisperModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
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
		self.encoder = WhisperEncoder(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.decoder = WhisperDecoder(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def _get_decoder_module(self):
		return self.decoder

	def _get_encoder_module(self):
		return self.encoder

	def __call__(
		self,
		input_features: jnp.ndarray,
		decoder_input_ids: jnp.ndarray,
		decoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		decoder_position_ids: tp.Optional[jnp.ndarray] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict
		batch_size, sequence_length = decoder_input_ids.shape

		if decoder_attention_mask is None:
			decoder_attention_mask = jnp.ones((batch_size, sequence_length))
		if decoder_position_ids is None:
			if past_key_values is not None:
				raise ValueError(
					"Make sure to provide `decoder_position_ids` when passing `past_key_values`."
				)

			if decoder_attention_mask is not None:
				decoder_position_ids = (
					decoder_attention_mask.cumsum(-1) * decoder_attention_mask
				) - 1
			else:
				decoder_position_ids = jnp.broadcast_to(
					jnp.arange(sequence_length)[None, :],
					(batch_size, sequence_length),
				)

		encoder_outputs = self.encoder(
			input_features=input_features,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			position_ids=decoder_position_ids,
			past_key_values=past_key_values,
			encoder_hidden_states=encoder_outputs[0],
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return FlaxSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)

	def decode(
		self,
		encoder_hidden_states: jnp.ndarray,
		decoder_input_ids: jnp.ndarray,
		decoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		decoder_position_ids: tp.Optional[jnp.ndarray] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict
		batch_size, sequence_length = decoder_input_ids.shape

		if decoder_attention_mask is None:
			decoder_attention_mask = jnp.ones((batch_size, sequence_length))
		if decoder_position_ids is None:
			if past_key_values is not None:
				raise ValueError(
					"Make sure to provide `decoder_position_ids` when passing `past_key_values`."
				)

			if decoder_attention_mask is not None:
				decoder_position_ids = (
					decoder_attention_mask.cumsum(-1) * decoder_attention_mask
				) - 1
			else:
				decoder_position_ids = jnp.broadcast_to(
					jnp.arange(sequence_length)[None, :],
					(batch_size, sequence_length),
				)

		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			position_ids=decoder_position_ids,
			past_key_values=past_key_values,
			encoder_hidden_states=encoder_hidden_states,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs

		return FlaxSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			past_key_values=past_key_values,
		)

	def encode(
		self,
		input_features: jnp.ndarray,
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		encoder_outputs = self.encoder(
			input_features=input_features,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		if not return_dict:
			return encoder_outputs

		return FlaxSeq2SeqModelOutput(
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)


@register_module(
	"speech-sequence-to-sequence",
	config=WhisperConfig,
	model_type="whisper",
	embedding_layer_names=["embed_positions", "embed_tokens"],
	layernorm_names=[
		"self_attn_layer_norm",
		"final_layer_norm",
		"encoder_attn_layer_norm",
		"layer_norm",
	],
)
class WhisperForConditionalGeneration(EasyDeLBaseModule):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
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
		self.model = WhisperModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.proj_out = nn.Linear(
			config.d_model,
			config.vocab_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(config.init_std),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def _get_encoder_module(self):
		return self.model.encoder

	def _get_decoder_module(self):
		return self.model.decoder

	def __call__(
		self,
		input_features,
		decoder_input_ids,
		decoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		decoder_position_ids: tp.Optional[jnp.ndarray] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		outputs = self.model(
			input_features=input_features,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			decoder_position_ids=decoder_position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			past_key_values=past_key_values,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			self.proj_out.kernel.value = (
				self.model.decoder.embed_tokens.embedding.value.T.astype(self.param_dtype)
			)

		lm_logits = self.proj_out(hidden_states)

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return output

		return FlaxSeq2SeqLMOutput(
			logits=lm_logits,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
		)

	def decode(
		self,
		decoder_input_ids,
		encoder_outputs,
		encoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		decoder_attention_mask: tp.Optional[jnp.ndarray] = None,
		decoder_position_ids: tp.Optional[jnp.ndarray] = None,
		past_key_values: tp.Optional[dict] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		encoder_hidden_states = encoder_outputs[0]
		outputs = self.model.decode(
			encoder_hidden_states=encoder_hidden_states,
			decoder_attention_mask=decoder_attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_position_ids=decoder_position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
		)
		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			self.proj_out.kernel.value = (
				self.model.decoder.embed_tokens.embedding.value.T.astype(self.param_dtype)
			)

		lm_logits = self.proj_out(hidden_states)

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return output

		return FlaxSeq2SeqLMOutput(
			logits=lm_logits,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
			past_key_values=outputs.past_key_values,
		)

	def encode(
		self,
		input_features: jnp.ndarray,
		attention_mask: tp.Optional[jnp.ndarray] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		**kwargs,
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		return self.model.encode(
			input_features=jnp.array(input_features, dtype="f4"),
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

	def generate(
		self,
		input_features,
		generation_config=None,
		logits_processor=None,
		return_timestamps=None,
		task=None,
		language=None,
		is_multilingual=None,
		**kwargs,
	):
		if generation_config is None:
			generation_config = self.generation_config

		if return_timestamps is not None:
			generation_config.return_timestamps = return_timestamps

		if task is not None:
			generation_config.task = task

		if is_multilingual is not None:
			generation_config.is_multilingual = is_multilingual

		if language is not None:
			generation_config.language = language

		if kwargs is not None and "decoder_input_ids" in kwargs:
			decoder_input_length = len(kwargs["decoder_input_ids"])
		else:
			decoder_input_length = 1

		forced_decoder_ids = []

		if (
			hasattr(generation_config, "is_multilingual")
			and generation_config.is_multilingual
		):
			if hasattr(generation_config, "language"):
				forced_decoder_ids.append(
					(1, generation_config.lang_to_id[generation_config.language])
				)
			else:
				forced_decoder_ids.append((1, None))

			if hasattr(generation_config, "task"):
				forced_decoder_ids.append(
					(2, generation_config.task_to_id[generation_config.task])
				)
			else:
				forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

		if (
			hasattr(generation_config, "return_timestamps")
			and generation_config.return_timestamps
		) or return_timestamps:
			logits_processor = [
				WhisperTimeStampLogitsProcessor(
					generation_config, self.config, decoder_input_length
				)
			]
		else:
			if (
				forced_decoder_ids
				and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id
			):
				idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
				forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

		if len(forced_decoder_ids) > 0:
			generation_config.forced_decoder_ids = forced_decoder_ids

		return super().generate(
			input_features,
			generation_config,
			logits_processor=logits_processor,
			**kwargs,
		)

	def _force_generate(
		self,
		input_features: jax.Array,
		forced_decoder_ids: jax.Array,
		return_timestamps: bool = False,
		generation_config: tp.Optional["transformers.GenerationConfig"] = None,  # noqa #type:ignore
		**kwargs,
	):
		if generation_config is None:
			generation_config = self.generation_config
		generation_config.forced_decoder_ids = None
		logits_processor = FlaxLogitsProcessorList()
		logits_processor.append(FlaxStaticForceTokensLogitsProcessor(forced_decoder_ids))
		if return_timestamps:
			logits_processor.append(
				WhisperTimeStampLogitsProcessor(generation_config, self.config, 1)
			)
		return super().generate(
			input_features,
			generation_config,
			logits_processor=logits_processor,
			**kwargs,
		)

	def prepare_inputs_for_generation(
		self,
		decoder_input_ids,
		max_length,
		attention_mask: tp.Optional[jax.Array] = None,
		decoder_attention_mask: tp.Optional[jax.Array] = None,
		encoder_outputs=None,
		**kwargs,
	):
		# initializing the cache
		batch_size, seq_length = decoder_input_ids.shape

		past_key_values = self.init_cache(batch_size, max_length)
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if decoder_attention_mask is not None:
			position_ids = decoder_attention_mask.cumsum(-1) - 1
			extended_attention_mask = lax.dynamic_update_slice(
				extended_attention_mask, decoder_attention_mask, (0, 0)
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
			)

		return self.prepare_inputs_for_call(
			**{
				"past_key_values": past_key_values,
				"encoder_outputs": encoder_outputs,
				"encoder_attention_mask": attention_mask,
				"decoder_attention_mask": extended_attention_mask,
				"decoder_position_ids": position_ids,
			}
		)

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs["past_key_values"] = model_outputs.past_key_values
		model_kwargs["decoder_position_ids"] = (
			model_kwargs["decoder_position_ids"][:, -1:] + 1
		)
		return model_kwargs


@register_module(
	"audio-classification",
	config=WhisperConfig,
	model_type="whisper",
	embedding_layer_names=["embed_positions", "embed_tokens"],
	layernorm_names=[
		"self_attn_layer_norm",
		"final_layer_norm",
		"encoder_attn_layer_norm",
		"layer_norm",
	],
)
class WhisperForAudioClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: WhisperConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
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
		self.encoder = WhisperEncoder(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		config.is_encoder_decoder = False
		num_layers = config.num_hidden_layers + 1
		if config.use_weighted_layer_sum:
			self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
		self.projector = nn.Linear(
			config.d_model,
			config.classifier_proj_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.classifier = nn.Linear(
			config.classifier_proj_size,
			config.num_labels,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_features,
		encoder_outputs=None,
		output_attentions=None,
		output_hidden_states: bool = True,
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

		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_features,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

		if self.config.use_weighted_layer_sum:
			hidden_states = jnp.stack(encoder_outputs, axis=1)
			norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
			hidden_states = jnp.sum(
				hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1
			)
		else:
			hidden_states = encoder_outputs[0]

		hidden_states = self.projector(hidden_states)
		pooled_output = jnp.mean(hidden_states, axis=1)

		logits = self.classifier(pooled_output)

		if not return_dict:
			return (logits,) + encoder_outputs[1:]

		return FlaxSequenceClassifierOutput(
			logits=logits,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)
