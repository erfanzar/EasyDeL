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

import math
import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxMaskedLMOutput,
)
from easydel.infra.utils import ACT2FN, control_mlp_sharding
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.opt.opt_configuration import OPTConfig as OPTConfig


class OPTAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: OPTConfig,
		embed_dim: int,
		num_heads: int,
		dropout: float = 0.0,
		causal: bool = False,
		bias: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
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
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
				f" and `num_heads`: {num_heads})."
			)

		linear = partial(
			nn.Linear,
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
		self.attention_module: FlexibleAttentionModule = FlexibleAttentionModule(
			attention_dropout=config.attention_dropout,
			num_q_heads=config.num_attention_heads,
			num_kv_heads=config.num_attention_heads,
			head_dims=self.head_dim,
			precision=precision,
			force_float32_tpu=True,
			attn_mechanism=config.attn_mechanism,
			dtype=config.attn_dtype,
			mesh=config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
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
		causal_mask: tp.Optional[chex.Array] = None,
		key_value_states: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
	) -> tp.Tuple[chex.Array]:
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
					attention_mask = jnp.logical_and(
						attention_mask, self.causal_mask[:, :, :sequence_length, :]
					)
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
			attention_bias,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=cache_view,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
		)

		attentions = self.attention_module(
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
			causal_mask=causal_mask,
		)

		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.out_proj(attn_output)

		return attn_output, attentions.attention_weights


class OPTDecoderLayer(nn.Module):
	def __init__(
		self,
		config: OPTConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.fc1 = nn.Linear(
			self.embed_dim,
			self.embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(config.init_std),
			rngs=rngs,
		)
		self.fc2 = nn.Linear(
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
		causal_mask: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
	) -> tp.Tuple[chex.Array]:
		residual = hidden_states

		# 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
		if self.do_layer_norm_before:
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
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		# 350m applies layer norm AFTER attention
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

		return hidden_states, self_attn_weights


class OPTLearnedPositionalEmbedding(nn.Embed):
	def __init__(
		self,
		num_embeddings: int,
		features: int,
		*,
		offset: int = 2,
		dtype: tp.Optional[jnp.dtype] = None,
		param_dtype: jnp.dtype = jnp.float32,
		embedding_init=None,
		rngs: nn.Rngs,
	):
		if embedding_init is None:
			embedding_init = nn.initializers.variance_scaling(
				1.0,
				"fan_in",
				"normal",
				out_axis=0,
			)
		self.embedding = nn.Param(
			embedding_init(rngs.params(), (num_embeddings + offset, features), param_dtype)
		)
		self.offset = offset
		self.num_embeddings = num_embeddings
		self.features = features
		self.dtype = dtype or self.embedding.value.dtype
		self.param_dtype = param_dtype
		self.embedding_init = embedding_init

	def __call__(self, inputs: chex.Array) -> chex.Array:
		return super().__call__(inputs + self.offset)


class OPTDecoder(EasyDeLBaseModule):
	def __init__(
		self,
		config: OPTConfig,
		offset: int = 2,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
			self.project_in = nn.Linear(
				self.config.word_embed_proj_dim,
				self.config.hidden_size,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			self.project_out = nn.Linear(
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
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
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")

		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		for idx, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = decoder_layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				output_attentions=output_attentions,
				past_key_values=past_key_values.views[idx],
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

		outputs = [hidden_state, all_hidden_states, all_self_attns]

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_state,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


class OPTModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: OPTConfig,
		offset: int = 2,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		decoder_outputs = self.decoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			past_key_values=past_key_values,
		)

		if not return_dict:
			return decoder_outputs

		return FlaxBaseModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			hidden_states=decoder_outputs.hidden_states,
			attentions=decoder_outputs.attentions,
		)

	def set_input_embeddings(self, value):
		self.embed_tokens = value

	def get_input_embeddings(self):
		return self.embed_tokens


@register_module(
	"causal-language-model",
	config=OPTConfig,
	model_type="opt",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["self_attn_layer_norm", "final_layer_norm"],
)
class OPTForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: OPTConfig,
		offset: int = 2,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.lm_head = nn.Linear(
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			shared_kernel = self.modeldecoder.embed_tokens.embedding.value.T
			self.lm_head.kernel.value = shared_kernel
			lm_logits = self.lm_head.apply(hidden_states)

		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxMaskedLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def set_input_embeddings(self, value):
		self.module.model.embed_tokens = value

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_decoder(self, decoder):
		self.module.model = decoder

	def get_decoder(self):
		return self.model

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.module.lm_head = new_embeddings

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		attention_mask: tp.Optional[chex.Array] = None,
	):
		# initializing the cache
		batch_size, seq_length = input_ids.shape

		past_key_values = self.init_cache(batch_size, max_length)
		# Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
		# But since the decoder uses a causal mask, those positions are masked anyway.
		# Thus, we can create a single static attention_mask here, which is more efficient for compilation
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

		if attention_mask is not None:
			position_ids = attention_mask.cumsum(axis=1) - 1
			extended_attention_mask = lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
			)

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
