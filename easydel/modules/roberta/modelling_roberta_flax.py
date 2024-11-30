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
from typing import Optional

import chex
import flax.linen
import jax
from flax import linen as nn
from flax.linen import Dense
from flax.linen.attention import (
	combine_masks,
	dot_product_attention_weights,
	make_causal_mask,
)
from flax.linen.partitioning import remat
from jax import lax
from jax import numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutputWithPastAndCrossAttentions,
	FlaxBaseModelOutputWithPoolingAndCrossAttentions,
	FlaxCausalLMOutputWithCrossAttentions,
	FlaxMaskedLMOutput,
	FlaxMultipleChoiceModelOutput,
	FlaxQuestionAnsweringModelOutput,
	FlaxSequenceClassifierOutput,
	FlaxTokenClassifierOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module
from easydel.modules.roberta.roberta_configuration import RobertaConfig as RobertaConfig


class FlaxRobertaEmbeddings(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.word_embeddings = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
		)
		self.position_embeddings = nn.Embed(
			self.config.max_position_embeddings,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
		)
		self.token_type_embeddings = nn.Embed(
			self.config.type_vocab_size,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
		)
		self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
		self.dropout = flax.linen.Dropout(rate=self.config.hidden_dropout_prob)

	def __call__(
		self,
		input_ids,
		token_type_ids,
		position_ids,
		attention_mask,
		deterministic: bool = True,
	):
		input_embeds = self.word_embeddings(input_ids.astype("i4"))
		position_embeds = self.position_embeddings(position_ids.astype("i4"))
		token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

		hidden_states = input_embeds + token_type_embeddings + position_embeds

		hidden_states = self.LayerNorm(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		return hidden_states


class FlaxRobertaSelfAttention(FlaxAttentionModule):
	config: RobertaConfig
	causal: bool = False
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
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
		self.query = Dense(
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.key = Dense(
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.value = Dense(
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

		if self.causal:
			self.causal_mask = make_causal_mask(
				jnp.ones(
					(
						1,
						getattr(
							self.config,
							"mask_max_position_embeddings",
							self.config.max_position_embeddings,
						),
					),
					dtype="bool",
				),
				dtype="bool",
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
		segment_ids: Optional[chex.Array] = None,
		key_value_states: Optional[jnp.array] = None,
		init_cache: bool = False,
		deterministic=True,
		output_attentions: bool = False,
	):
		is_cross_attention = key_value_states is not None
		batch_size = hidden_states.shape[0]

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

		if self.causal:
			query_length, key_length = query_states.shape[1], key_states.shape[1]
			if self.has_variable("cache", "cached_key"):
				mask_shift = self.variables["cache"]["cache_index"]
				max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
				causal_mask = lax.dynamic_slice(
					self.causal_mask,
					(0, 0, mask_shift, 0),
					(1, 1, query_length, max_decoder_length),
				)
			else:
				causal_mask = self.causal_mask[:, :, :query_length, :key_length]
			causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

		if attention_mask is not None and self.causal:
			attention_mask = jnp.broadcast_to(
				jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
			)
			attention_mask = combine_masks(attention_mask, causal_mask)
		elif self.causal:
			attention_mask = causal_mask
		elif attention_mask is not None:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

		if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
			key_states, value_states, attention_mask = self._concatenate_to_cache(
				query_states,
				key_states,
				value_states,
				attention_mask,
			)

		if attention_mask is not None:
			attention_bias = lax.select(
				attention_mask > 0,
				jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
				jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
			)
		else:
			attention_bias = None

		dropout_rng = None
		if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
			dropout_rng = self.make_rng("dropout")
		if layer_head_mask is None:
			out = self.attention_performer.__call__(
				query_states=query_states,
				key_states=key_states,
				value_states=value_states,
				dropout_rng=dropout_rng,
				deterministic=deterministic,
				causal=True,
				bias=attention_bias,
				attention_mask=attention_mask,
				uses_cache=False,
				query_sequence_length=query_states.shape[1],
				key_value_sequence_length=key_states.shape[1],
				segment_ids=segment_ids,
				causal_mask=causal_mask,
			)
			attn_weights = out.attention_weights
			attn_output = out.attention_outputs
		else:
			attn_weights = dot_product_attention_weights(
				query_states,
				key_states,
				bias=attention_bias,
				dropout_rng=dropout_rng,
				dropout_rate=self.config.attention_probs_dropout_prob,
				broadcast_dropout=True,
				deterministic=deterministic,
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


class FlaxRobertaSelfOutput(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
		self.dropout = flax.linen.Dropout(rate=self.config.hidden_dropout_prob)

	def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class FlaxRobertaAttention(nn.Module):
	config: RobertaConfig
	causal: bool = False
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.self = FlaxRobertaSelfAttention(
			self.config,
			causal=self.causal,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		layer_head_mask,
		key_value_states=None,
		init_cache=False,
		deterministic=True,
		output_attentions: bool = False,
	):
		attn_outputs = self.self(
			hidden_states,
			attention_mask,
			layer_head_mask=layer_head_mask,
			key_value_states=key_value_states,
			init_cache=init_cache,
			deterministic=deterministic,
			output_attentions=output_attentions,
		)
		attn_output = attn_outputs[0]
		hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attn_outputs[1],)

		return outputs


class FlaxRobertaIntermediate(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.intermediate_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.activation = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.activation(hidden_states)
		return hidden_states


class FlaxRobertaOutput(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			precision=self.precision,
			param_dtype=self.param_dtype,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.dropout = flax.linen.Dropout(rate=self.config.hidden_dropout_prob)
		self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

	def __call__(self, hidden_states, attention_output, deterministic: bool = True):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.LayerNorm(hidden_states + attention_output)
		return hidden_states


class FlaxRobertaLayer(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.attention = FlaxRobertaAttention(
			self.config,
			causal=self.config.is_decoder,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
		self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)
		if self.config.add_cross_attention:
			self.crossattention = FlaxRobertaAttention(
				self.config,
				causal=True,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		layer_head_mask,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		output_attentions: bool = False,
	):
		# Self Attention
		attention_outputs = self.attention(
			hidden_states,
			attention_mask,
			layer_head_mask=layer_head_mask,
			init_cache=init_cache,
			deterministic=deterministic,
			output_attentions=output_attentions,
		)
		attention_output = attention_outputs[0]

		# Cross-Attention Block
		if encoder_hidden_states is not None:
			cross_attention_outputs = self.crossattention(
				attention_output,
				attention_mask=encoder_attention_mask,
				layer_head_mask=layer_head_mask,
				key_value_states=encoder_hidden_states,
				deterministic=deterministic,
				output_attentions=output_attentions,
			)
			attention_output = cross_attention_outputs[0]

		hidden_states = self.intermediate(attention_output)
		hidden_states = self.output(
			hidden_states, attention_output, deterministic=deterministic
		)

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attention_outputs[1],)
			if encoder_hidden_states is not None:
				outputs += (cross_attention_outputs[1],)
		return outputs


class FlaxRobertaLayerCollection(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		block = FlaxRobertaLayer
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			block = remat(
				block,
				static_argnums=(5, 6, 7),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)

		self.layers = [
			block(
				self.config,
				name=str(i),
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
			for i in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states,
		attention_mask,
		head_mask,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		init_cache: bool = False,
		deterministic: bool = True,
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
			if head_mask.shape[0] != (len(self.layers)):
				raise ValueError(
					f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
					f"       {head_mask.shape[0]}."
				)

		for i, layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = layer(
				hidden_states,
				attention_mask,
				head_mask[i] if head_mask is not None else None,
				encoder_hidden_states,
				encoder_attention_mask,
				init_cache,
				deterministic,
				output_attentions,
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


class FlaxRobertaEncoder(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.layer = FlaxRobertaLayerCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		head_mask,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		return self.layer(
			hidden_states,
			attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			init_cache=init_cache,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)


class FlaxRobertaPooler(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

	def __call__(self, hidden_states):
		cls_hidden_state = hidden_states[:, 0]
		cls_hidden_state = self.dense(cls_hidden_state)
		return nn.tanh(cls_hidden_state)


class FlaxRobertaLMHead(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
		self.decoder = Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			use_bias=False,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		self.bias = self.param("bias", jax.nn.initializers.zeros, (self.config.vocab_size,))

	def __call__(self, hidden_states, shared_embedding=None):
		hidden_states = self.dense(hidden_states)
		hidden_states = ACT2FN["gelu"](hidden_states)
		hidden_states = self.layer_norm(hidden_states)

		if shared_embedding is not None:
			hidden_states = self.decoder.apply(
				{"params": {"kernel": shared_embedding.T}}, hidden_states
			)
		else:
			hidden_states = self.decoder(hidden_states)

		bias = self.bias.astype(self.dtype)
		hidden_states += bias
		return hidden_states


class FlaxRobertaClassificationHead(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.dense = Dense(
			self.config.hidden_size,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)
		classifier_dropout = (
			self.config.classifier_dropout
			if self.config.classifier_dropout is not None
			else self.config.hidden_dropout_prob
		)
		self.dropout = flax.linen.Dropout(rate=classifier_dropout)
		self.out_proj = Dense(
			self.config.num_labels,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

	def __call__(self, hidden_states, deterministic=True):
		hidden_states = hidden_states[:, 0, :]
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.dense(hidden_states)
		hidden_states = nn.tanh(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.out_proj(hidden_states)
		return hidden_states


@register_module(
	"base-module",
	config=RobertaConfig,
	model_type="roberta",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["layer_norm", "LayerNorm"],
)
@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaModel(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None
	add_pooling_layer: bool = True

	def setup(self):
		self.embeddings = FlaxRobertaEmbeddings(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.encoder = FlaxRobertaEncoder(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.pooler = FlaxRobertaPooler(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids: Optional[jnp.ndarray] = None,
		position_ids: Optional[jnp.ndarray] = None,
		head_mask: Optional[jnp.ndarray] = None,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# make sure `token_type_ids` is correctly initialized when not passed
		if token_type_ids is None:
			token_type_ids = jnp.zeros_like(input_ids)

		# make sure `position_ids` is correctly initialized when not passed
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
			)

		hidden_states = self.embeddings(
			input_ids,
			token_type_ids,
			position_ids,
			attention_mask,
			deterministic=deterministic,
		)
		outputs = self.encoder(
			hidden_states,
			attention_mask,
			head_mask=head_mask,
			deterministic=deterministic,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			init_cache=init_cache,
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


@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForMaskedLM(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			add_pooling_layer=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = FlaxRobertaLMHead(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		if self.config.tie_word_embeddings:
			shared_embedding = self.roberta.variables["params"]["embeddings"][
				"word_embeddings"
			]["embedding"].T.astype(self.param_dtype)
		else:
			shared_embedding = None

		# Compute the prediction scores
		logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxMaskedLMOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@register_module(
	"sequence-classification",
	config=RobertaConfig,
	model_type="roberta",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["layer_norm", "LayerNorm"],
)
@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForSequenceClassification(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			add_pooling_layer=False,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.classifier = FlaxRobertaClassificationHead(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		logits = self.classifier(sequence_output, deterministic=deterministic)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxSequenceClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForMultipleChoice(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.dropout = flax.linen.Dropout(rate=self.config.hidden_dropout_prob)
		self.classifier = Dense(
			1,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		deterministic: bool = True,
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
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		pooled_output = outputs[1]
		pooled_output = self.dropout(pooled_output, deterministic=deterministic)
		logits = self.classifier(pooled_output)

		reshaped_logits = logits.reshape(-1, num_choices)

		if not return_dict:
			return (reshaped_logits,) + outputs[2:]

		return FlaxMultipleChoiceModelOutput(
			logits=reshaped_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForTokenClassification(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			add_pooling_layer=False,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		classifier_dropout = (
			self.config.classifier_dropout
			if self.config.classifier_dropout is not None
			else self.config.hidden_dropout_prob
		)
		self.dropout = flax.linen.Dropout(rate=classifier_dropout)
		self.classifier = Dense(
			self.config.num_labels,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		logits = self.classifier(hidden_states)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxTokenClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForQuestionAnswering(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			add_pooling_layer=False,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.qa_outputs = Dense(
			self.config.num_labels,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		position_ids,
		head_mask,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			deterministic=deterministic,
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
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["layer_norm", "LayerNorm"],
)
@wrap_easydel_module(config_class=RobertaConfig, base_model_prefix="roberta")
class FlaxRobertaForCausalLM(nn.Module):
	config: RobertaConfig
	dtype: jnp.dtype = jnp.float32  # the dtype of the computation
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[lax.Precision] = None

	def setup(self):
		self.roberta = FlaxRobertaModel.flax_module(
			config=self.config,
			add_pooling_layer=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = FlaxRobertaLMHead(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		position_ids,
		token_type_ids: Optional[jnp.ndarray] = None,
		head_mask: Optional[jnp.ndarray] = None,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		# Model
		outputs = self.roberta(
			input_ids,
			attention_mask,
			token_type_ids,
			position_ids,
			head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			init_cache=init_cache,
			deterministic=deterministic,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		if self.config.tie_word_embeddings:
			shared_embedding = self.roberta.variables["params"]["embeddings"][
				"word_embeddings"
			]["embedding"].T.astype(self.param_dtype)
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
