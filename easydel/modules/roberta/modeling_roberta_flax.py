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
import typing as tp

import chex
import jax
from flax import nnx as nn
from flax.nnx.nn.attention import dot_product_attention_weights
from jax import lax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutputWithPastAndCrossAttentions,
	FlaxBaseModelOutputWithPoolingAndCrossAttentions,
	FlaxCausalLMOutputWithCrossAttentions,
	FlaxMultipleChoiceModelOutput,
	FlaxQuestionAnsweringModelOutput,
	FlaxSequenceClassifierOutput,
	FlaxTokenClassifierOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCacheView
from easydel.layers.caching.transformer_cache import TransformerCache
from easydel.modules.roberta.roberta_configuration import RobertaConfig as RobertaConfig


class RobertaEmbeddings(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.word_embeddings = nn.Embed(
			num_embeddings=self.config.vocab_size,
			features=self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.position_embeddings = nn.Embed(
			num_embeddings=self.config.max_position_embeddings,
			features=self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.token_type_embeddings = nn.Embed(
			num_embeddings=self.config.type_vocab_size,
			features=self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.LayerNorm = nn.LayerNorm(
			self.config.hidden_size,
			epsilon=self.config.layer_norm_eps,
			param_dtype=param_dtype,
			dtype=dtype,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(
			rate=self.config.hidden_dropout_prob,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids,
		token_type_ids,
		position_ids,
		attention_mask,
	):
		inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
		position_embeds = self.position_embeddings(position_ids.astype("i4"))
		token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

		hidden_states = inputs_embeds + token_type_embeddings + position_embeds

		hidden_states = self.LayerNorm(hidden_states)
		hidden_states = self.dropout(hidden_states)
		return hidden_states


class RobertaSelfAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: RobertaConfig,
		causal: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config)
		self.causal = causal
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.head_dim = self.config.hidden_size // self.config.num_attention_heads
		if self.config.hidden_size % self.config.num_attention_heads != 0:
			raise ValueError(
				"`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
				"                   : {self.config.num_attention_heads}"
			)
		self.attention_performer = FlexibleAttentionModule(
			use_sharding_constraint=self.config.use_sharding_constraint,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
			attention_dropout=0.0,
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
			base_config=self.config,
		)
		self.query = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.key = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.value = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def _split_heads(self, hidden_states):
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim)
		)

	def _merge_heads(self, hidden_states):
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (chex.Array): The hidden states with separate head dimensions.

		Returns:
		    chex.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

	def __call__(
		self,
		hidden_states,
		attention_mask,
		layer_head_mask,
		causal_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		key_value_states: tp.Optional[jnp.array] = None,
		output_attentions: bool = False,
	):
		is_cross_attention = key_value_states is not None

		query_states = self.query(hidden_states)
		if is_cross_attention:
			key_states = self.key(key_value_states)
			value_states = self.value(key_value_states)
		else:
			key_states = self.key(hidden_states)
			value_states = self.value(hidden_states)

		query_states = self._split_heads(query_states)
		key_states = self._split_heads(key_states)
		value_states = self._split_heads(value_states)
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
			causal_mask=causal_mask if self.causal else None,
			fcm_mask=None,
			sliding_windows=None,
		)

		if layer_head_mask is None:
			out = self.attention_performer(
				query_states=query_states,
				key_states=key_states,
				value_states=value_states,
				causal=True,
				bias=attention_bias,
				attention_mask=attention_mask,
				uses_cache=cache_view is not None,
				query_sequence_length=query_states.shape[1],
				key_value_sequence_length=key_states.shape[1],
				segment_ids=segment_ids,
				causal_mask=None,
			)
			attn_weights = out.attention_weights
			attn_output = out.attention_outputs
		else:
			attn_weights = dot_product_attention_weights(
				query_states,
				key_states,
				bias=attention_bias,
				dropout_rate=self.config.attention_probs_dropout_prob,
				broadcast_dropout=True,
				dtype=self.dtype,
				precision=None,
			)

			attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)
			attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

		attn_output = self.shard_attention_prod(
			attn_output.reshape(attn_output.shape[:2] + (-1,))
		)

		outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
		return outputs


class RobertaSelfOutput(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.LayerNorm = nn.LayerNorm(
			self.config.hidden_size,
			epsilon=self.config.layer_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob, rngs=rngs)

	def __call__(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class RobertaAttention(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		causal: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.causal = causal
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.self = RobertaSelfAttention(
			config=config,
			causal=causal,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.output = RobertaSelfOutput(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		layer_head_mask,
		causal_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		key_value_states=None,
		output_attentions: bool = False,
	):
		attn_outputs = self.self(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask if self.causal else None,
			layer_head_mask=layer_head_mask,
			cache_view=cache_view,
			key_value_states=key_value_states,
			output_attentions=output_attentions,
		)
		attn_output = attn_outputs[0]
		hidden_states = self.output(attn_output, hidden_states)

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attn_outputs[1],)

		return outputs


class RobertaIntermediate(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.intermediate_size,
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.activation = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.activation(hidden_states)
		return hidden_states


class RobertaOutput(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.hidden_size,
			self.config.intermediate_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=dtype,
			precision=precision,
			param_dtype=param_dtype,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.dropout = nn.Dropout(
			rate=self.config.hidden_dropout_prob,
			rngs=rngs,
		)
		self.LayerNorm = nn.LayerNorm(
			self.config.intermediate_size,
			epsilon=self.config.layer_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, hidden_states, attention_output):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + attention_output)
		return hidden_states


class RobertaLayer(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.attention = RobertaAttention(
			config=config,
			causal=config.is_decoder,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.intermediate = RobertaIntermediate(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.output = RobertaOutput(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		if self.config.add_cross_attention:
			self.crossattention = RobertaAttention(
				config=config,
				causal=True,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		layer_head_mask,
		causal_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
	):
		# Self Attention
		attention_outputs = self.attention(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			layer_head_mask=layer_head_mask,
			cache_view=cache_view,
			output_attentions=output_attentions,
		)
		attention_output = attention_outputs[0]

		# Cross-Attention Block
		if encoder_hidden_states is not None:
			cross_attention_outputs = self.crossattention(
				hidden_states=attention_output,
				attention_mask=encoder_attention_mask,
				layer_head_mask=layer_head_mask,
				cache_view=cache_view,
				key_value_states=encoder_hidden_states,
				output_attentions=output_attentions,
				causal_mask=causal_mask,
			)
			attention_output = cross_attention_outputs[0]

		hidden_states = self.intermediate(attention_output)
		hidden_states = self.output(hidden_states, attention_output)

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attention_outputs[1],)
			if encoder_hidden_states is not None:
				outputs += (cross_attention_outputs[1],)
		return outputs


class RobertaEncoder(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		block = RobertaLayer
		block = auto_remat(
			block,
			policy=config.gradient_checkpointing,
		)
		self.layer = [
			block(
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
		hidden_states,
		attention_mask,
		head_mask,
		causal_mask: tp.Optional[chex.Array] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		all_cross_attentions = (
			() if (output_attentions and encoder_hidden_states is not None) else None
		)

		# Check if head_mask has a correct number of layers specified if desired
		if head_mask is not None:
			if head_mask.shape[0] != (len(self.layer)):
				raise ValueError(
					f"The head_mask should be specified for {len(self.layer)} layer, but it is for                  "
					f"       {head_mask.shape[0]}."
				)
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layer))
		for i, (layer, cache_view) in enumerate(zip(self.layer, past_key_values.views)):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				layer_head_mask=head_mask[i] if head_mask is not None else None,
				cache_view=cache_view,
				causal_mask=causal_mask,
				encoder_hidden_states=encoder_hidden_states,
				encoder_attention_mask=encoder_attention_mask,
				output_attentions=output_attentions,
			)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

				if encoder_hidden_states is not None:
					all_cross_attentions += (layer_outputs[2],)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (
			hidden_states,
			all_hidden_states,
			all_attentions,
			all_cross_attentions,
		)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			cross_attentions=all_cross_attentions,
		)


class RobertaPooler(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def __call__(self, hidden_states):
		cls_hidden_state = hidden_states[:, 0]
		cls_hidden_state = self.dense(cls_hidden_state)
		return nn.tanh(cls_hidden_state)


class RobertaLMHead(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.layer_norm = nn.LayerNorm(
			self.config.hidden_size,
			epsilon=self.config.layer_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.decoder = nn.Linear(
			self.config.vocab_size,
			self.config.hidden_size,
			dtype=dtype,
			use_bias=False,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		self.bias = nn.Param(
			jax.nn.initializers.zeros(
				key=rngs.params(),
				shape=(self.config.vocab_size,),
				dtype=self.param_dtype,
			)
		)

	def __call__(self, hidden_states, shared_embedding=None):
		hidden_states = self.dense(hidden_states)
		hidden_states = ACT2FN["gelu"](hidden_states)
		hidden_states = self.layer_norm(hidden_states)

		if shared_embedding is not None:
			self.decoder.kernel.value = shared_embedding.T
			self.decoder.bias.value = None
			hidden_states = self.decoder(hidden_states)
		else:
			hidden_states = self.decoder(hidden_states)

		bias = self.bias.astype(self.dtype)
		hidden_states += bias
		return hidden_states


class RobertaClassificationHead(nn.Module):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.dense = nn.Linear(
			self.config.hidden_size,
			self.config.hidden_size,
			dtype=dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)
		classifier_dropout = (
			self.config.classifier_dropout
			if self.config.classifier_dropout is not None
			else self.config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(
			rate=classifier_dropout,
			rngs=rngs,
		)
		self.out_proj = nn.Linear(
			self.config.num_labels,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def __call__(self, hidden_states):
		hidden_states = hidden_states[:, 0, :]
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.dense(hidden_states)
		hidden_states = nn.tanh(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.out_proj(hidden_states)
		return hidden_states


@register_module(
	"base-module",
	config=RobertaConfig,
	model_type="roberta",
	embedding_layer_names=[
		"word_embeddings",
		"position_embeddings",
		"token_type_embeddings",
	],
	layernorm_names=["layer_norm", "LayerNorm"],
)
class RobertaModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
		add_pooling_layer: bool = True,
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
		self.embeddings = RobertaEmbeddings(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.encoder = RobertaEncoder(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.pooler = (
			RobertaPooler(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			if add_pooling_layer
			else None
		)
		self.add_pooling_layer = add_pooling_layer

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		head_mask: tp.Optional[chex.Array] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[tp.Tuple[tp.Tuple[chex.Array, chex.Array]]] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# make sure `token_type_ids` is correctly initialized when not passed
		if token_type_ids is None:
			token_type_ids = jnp.zeros_like(input_ids)

		# make sure `position_ids` is correctly initialized when not passed
		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
			)

		hidden_states = self.embeddings(
			input_ids=input_ids,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			attention_mask=attention_mask,
		)
		outputs = self.encoder(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			head_mask=head_mask,
			causal_mask=self.causal_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			past_key_values=past_key_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = outputs[0]
		pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

		if not return_dict:
			# if pooled is None, don't return it
			if pooled is None:
				return (hidden_states,) + outputs[1:]
			return (hidden_states, pooled) + outputs[1:]

		return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=hidden_states,
			pooler_output=pooled,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			cross_attentions=outputs.cross_attentions,
		)


@register_module(
	"sequence-classification",
	config=RobertaConfig,
	model_type="roberta",
	embedding_layer_names=[
		"word_embeddings",
		"position_embeddings",
		"token_type_embeddings",
	],
	layernorm_names=["layer_norm", "LayerNorm"],
)
class RobertaForSequenceClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
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
		self.roberta = RobertaModel(
			config=config,
			dtype=dtype,
			add_pooling_layer=False,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.classifier = RobertaClassificationHead(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		logits = self.classifier(sequence_output)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxSequenceClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


class RobertaForMultipleChoice(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
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
		self.roberta = RobertaModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(
			rate=self.config.hidden_dropout_prob,
			rngs=rngs,
		)
		self.classifier = nn.Linear(
			1,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		num_choices = input_ids.shape[1]
		input_ids = (
			input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
		)
		attention_mask = (
			attention_mask.reshape(-1, attention_mask.shape[-1])
			if attention_mask is not None
			else None
		)
		token_type_ids = (
			token_type_ids.reshape(-1, token_type_ids.shape[-1])
			if token_type_ids is not None
			else None
		)
		position_ids = (
			position_ids.reshape(-1, position_ids.shape[-1])
			if position_ids is not None
			else None
		)

		# Model
		outputs = self.roberta(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		pooled_output = outputs[1]
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		reshaped_logits = logits.reshape(-1, num_choices)

		if not return_dict:
			return (reshaped_logits,) + outputs[2:]

		return FlaxMultipleChoiceModelOutput(
			logits=reshaped_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


class RobertaForTokenClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
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
		self.roberta = RobertaModel(
			config=config,
			dtype=dtype,
			add_pooling_layer=False,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		classifier_dropout = (
			self.config.classifier_dropout
			if self.config.classifier_dropout is not None
			else self.config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(
			rate=classifier_dropout,
			rngs=rngs,
		)
		self.classifier = nn.Linear(
			self.config.num_labels,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		hidden_states = self.dropout(hidden_states)
		logits = self.classifier(hidden_states)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxTokenClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


class RobertaForQuestionAnswering(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
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
		self.roberta = RobertaModel(
			config=config,
			dtype=dtype,
			add_pooling_layer=False,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.qa_outputs = nn.Linear(
			self.config.num_labels,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(bits=config.bits, mode=config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		logits = self.qa_outputs(hidden_states)
		start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)

		if not return_dict:
			return (start_logits, end_logits) + outputs[1:]

		return FlaxQuestionAnsweringModelOutput(
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@register_module(
	"causal-language-model",
	config=RobertaConfig,
	model_type="roberta",
	embedding_layer_names=[
		"word_embeddings",
		"position_embeddings",
		"token_type_embeddings",
	],
	layernorm_names=["layer_norm", "LayerNorm"],
)
class RobertaForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: RobertaConfig,
		dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[lax.Precision] = None,
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
		self.roberta = RobertaModel(
			config=config,
			add_pooling_layer=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = RobertaLMHead(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		head_mask: tp.Optional[chex.Array] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[tp.Tuple[tp.Tuple[chex.Array, chex.Array]]] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			past_key_values=past_key_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		if self.config.tie_word_embeddings:
			shared_embedding = self.roberta.embeddings.word_embeddings.embedding.value
		else:
			shared_embedding = None

		# Compute the prediction scores
		logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxCausalLMOutputWithCrossAttentions(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			cross_attentions=outputs.cross_attentions,
		)
