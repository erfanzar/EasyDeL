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

import math
from typing import Any, Optional

import flax.linen
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import Dense, combine_masks, make_causal_mask
from jax import lax

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.gpt2.gpt2_configuration import GPT2Config as GPT2Config
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutputWithPastAndCrossAttentions,
	FlaxCausalLMOutputWithCrossAttentions,
)
from easydel.modules.modeling_utils import wrap_easydel_module


class FlaxConv1D(nn.Module):
	features: int
	use_bias: bool = True
	dtype: Any = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None
	dot_general: Optional[None] = None

	@nn.compact
	def __call__(self, inputs):
		inputs = jnp.asarray(inputs, self.dtype)
		kernel = self.param(
			"kernel",
			jax.nn.initializers.normal(stddev=0.02),
			(self.features, inputs.shape[-1]),
		)

		kernel = kernel.astype(self.dtype).transpose()
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
		if self.use_bias:
			bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
			bias = jnp.asarray(bias, self.dtype)
			y = y + bias
		return y


class FlaxGPT2Attention(FlaxAttentionModule):
	config: GPT2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None
	causal: bool = True
	is_cross_attention: bool = False

	def setup(self):
		config = self.config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads

		if self.is_cross_attention:
			self.c_attn = FlaxConv1D(
				2 * self.embed_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
			self.q_attn = FlaxConv1D(
				self.embed_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
		else:
			self.c_attn = FlaxConv1D(
				3 * self.embed_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
		self.c_proj = FlaxConv1D(
			self.embed_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.attention_performer = FlexibleAttentionModule(
			use_sharding_constraint=self.config.use_sharding_constraint,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
			attention_dropout=self.config.attn_pdrop,
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
			base_config=self.config,
		)
		self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

	def _split_heads(self, hidden_states):
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.num_heads, self.head_dim)
		)

	def _merge_heads(self, hidden_states):
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (chex.Array): The hidden states with separate head dimensions.

		Returns:
		    chex.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

	def __call__(
		self,
		hidden_states,
		key_value_states: Optional[jnp.ndarray] = None,
		attention_mask=None,
		casual_mask=None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
	):
		is_cross_attention = key_value_states is not None
		batch_size = hidden_states.shape[0]
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
		query_length, key_length = query.shape[1], key.shape[1]

		if self.causal:
			if self.has_variable("cache", "cached_key"):
				mask_shift = self.variables["cache"]["cache_index"]
				max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
				causal_mask = lax.dynamic_slice(
					casual_mask,
					(0, 0, mask_shift, 0),
					(1, 1, query_length, max_decoder_length),
				)
			else:
				causal_mask = casual_mask[:, :, :query_length, :key_length]
			causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

		# combine masks if needed
		if attention_mask is not None and self.causal:
			attention_mask = jnp.broadcast_to(
				jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
			)
			attention_mask = combine_masks(attention_mask, causal_mask)
		elif self.causal:
			attention_mask = causal_mask
		elif attention_mask is not None:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

		dropout_rng = None
		if not deterministic and self.config.attn_pdrop > 0.0:
			dropout_rng = self.make_rng("dropout")

		if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
			key, value, attention_mask = self._concatenate_to_cache(
				query, key, value, attention_mask
			)

		if attention_mask is not None:
			attention_bias = lax.select(
				attention_mask > 0,
				jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
				jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
			)
		else:
			attention_bias = None

		attn = self.attention_performer(
			query_states=query,
			key_states=key,
			value_states=value,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=self.causal,
			dropout_rng=dropout_rng,
			deterministic=deterministic,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=self.has_variable("cache", "cached_key") or init_cache,
			segment_ids=None,
			causal_mask=causal_mask,
		)
		attn_output = self.shard_attention_prod(self._merge_heads(attn.attention_outputs))
		attn_output = self.c_proj(attn_output)
		attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

		outputs = (
			(attn_output, attn.attention_weights) if output_attentions else (attn_output,)
		)
		return outputs


class FlaxGPT2MLP(nn.Module):
	config: GPT2Config
	intermediate_size: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		embed_dim = self.config.hidden_size
		self.c_fc = FlaxConv1D(
			self.intermediate_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.c_proj = FlaxConv1D(
			embed_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.act = ACT2FN[self.config.activation_function]
		self.dropout = flax.linen.Dropout(rate=self.config.resid_pdrop)

	def __call__(self, hidden_states, deterministic: bool = True):
		hidden_states = self.c_proj(self.act(self.c_fc(hidden_states)))
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		return hidden_states


class FlaxGPT2Block(nn.Module):
	config: GPT2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		hidden_size = self.config.hidden_size
		inner_dim = (
			self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
		)

		self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

		attn_block = FlaxGPT2Attention
		mlp_block = FlaxGPT2MLP
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = flax.linen.partitioning.remat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(3, 4, 5, 6),
			)

			mlp_block = flax.linen.partitioning.remat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)

		self.attn = attn_block(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

		if self.config.add_cross_attention:
			self.crossattention = attn_block(
				config=self.config,
				dtype=self.dtype,
				causal=True,
				is_cross_attention=True,
			)
			self.ln_cross_attn = nn.LayerNorm(
				epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
			)

		self.mlp = mlp_block(self.config, inner_dim, dtype=self.dtype)

	def __call__(
		self,
		hidden_states,
		attention_mask=None,
		casual_mask=None,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
	):
		residual = hidden_states
		hidden_states = self.ln_1(hidden_states)
		attn_outputs = self.attn(
			hidden_states,
			None,
			attention_mask,
			casual_mask,
			deterministic,
			init_cache,
			output_attentions,
		)
		attn_output = attn_outputs[0]
		outputs = attn_outputs[1:]
		hidden_states = attn_output + residual
		if encoder_hidden_states is not None:
			if not hasattr(self, "crossattention"):
				raise ValueError(
					f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
					"cross-attention layers by setting `config.add_cross_attention=True`"
				)
			residual = hidden_states
			hidden_states = self.ln_cross_attn(hidden_states)
			# hidden_states
			# key_value_states: Optional[jnp.ndarray] = None
			# attention_mask = None
			# casual_mask = None
			# deterministic: bool = True
			# init_cache: bool = False
			# output_attentions: bool = False

			cross_attn_outputs = self.crossattention(
				hidden_states,
				encoder_hidden_states,
				encoder_attention_mask,
				casual_mask,
				deterministic,
				False,
				output_attentions,
			)
			attn_output = cross_attn_outputs[0]
			# residual connection
			hidden_states = residual + attn_output
			outputs = (
				outputs + cross_attn_outputs[1:]
			)  # add cross attentions if we output attention weights

		residual = hidden_states
		hidden_states = self.ln_2(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp, hidden_states, self.config.scan_mlp_chunk_size, deterministic
			)
		else:
			feed_forward_hidden_states = self.mlp(
				hidden_states,
				deterministic,
			)
		# residual connection
		hidden_states = residual + feed_forward_hidden_states

		outputs = (hidden_states,) + outputs

		return outputs


class FlaxGPT2BlockCollection(nn.Module):
	config: GPT2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		self.blocks = [
			FlaxGPT2Block(
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
		attention_mask=None,
		casual_mask=None,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		all_cross_attentions = (
			() if (output_attentions and encoder_hidden_states is not None) else None
		)

		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states,
				attention_mask,
				casual_mask=casual_mask,
				encoder_hidden_states=encoder_hidden_states,
				encoder_attention_mask=encoder_attention_mask,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

				if encoder_hidden_states is not None:
					all_cross_attentions += (layer_outputs[2],)

		outputs = (
			hidden_states,
			all_hidden_states,
			all_attentions,
			all_cross_attentions,
		)

		return outputs


@register_module(
	"base-module",
	config=GPT2Config,
	model_type="gpt2",
	embedding_layer_names=["wte", "wpe"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
@wrap_easydel_module(config_class=GPT2Config, base_model_prefix="transformer")
class FlaxGPT2Model(nn.Module):
	config: GPT2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		self.embed_dim = self.config.hidden_size

		self.wte = nn.Embed(
			self.config.vocab_size,
			self.embed_dim,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
		)
		self.wpe = nn.Embed(
			self.config.max_position_embeddings,
			self.embed_dim,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
		)
		self.casual_mask = make_causal_mask(
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
		self.dropout = flax.linen.Dropout(rate=self.config.embd_pdrop)
		self.h = FlaxGPT2BlockCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

	def __call__(
		self,
		input_ids,
		attention_mask,
		position_ids,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		deterministic=True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		input_embeds = self.wte(input_ids.astype("i4"))
		position_embeds = self.wpe(position_ids.astype("i4"))

		hidden_states = input_embeds + position_embeds
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)

		outputs = self.h(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			casual_mask=self.casual_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		hidden_states = self.ln_f(hidden_states)

		if output_hidden_states:
			all_hidden_states = outputs[1] + (hidden_states,)
			outputs = (hidden_states, all_hidden_states) + outputs[2:]
		else:
			outputs = (hidden_states,) + outputs[1:]

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[2],
			cross_attentions=outputs[3],
		)


@register_module(
	"causal-language-model",
	config=GPT2Config,
	model_type="gpt2",
	embedding_layer_names=["wte", "wpe"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
@wrap_easydel_module(config_class=GPT2Config, base_model_prefix="transformer")
class FlaxGPT2LMHeadModel(nn.Module):
	config: GPT2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		self.transformer = FlaxGPT2Model.flax_module(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(
			self.config.vocab_size,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		position_ids,
		encoder_hidden_states: Optional[jnp.ndarray] = None,
		encoder_attention_mask: Optional[jnp.ndarray] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		outputs = self.transformer(
			input_ids,
			attention_mask,
			position_ids,
			encoder_hidden_states,
			encoder_attention_mask,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T.astype(
				self.param_dtype
			)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				hidden_states,
			)
		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutputWithCrossAttentions(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			cross_attentions=outputs.cross_attentions,
		)
