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
import typing as tp

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutputWithPastAndCrossAttentions,
	FlaxCausalLMOutputWithCrossAttentions,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.gpt2.gpt2_configuration import GPT2Config as GPT2Config


class Conv1D(nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		use_bias: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		dot_general: tp.Optional[None] = None,
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


class GPT2Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: GPT2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		hidden_states: chex.Array,
		key_value_states: chex.Array,
		attention_mask: chex.Array,
		causal_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
	):
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

		attention_bias = None
		if self.causal:
			(
				key,
				value,
				attention_mask,
				attention_bias,
			) = self.concatenate(
				query=query,
				key=key,
				cache_view=cache_view,
				value=value,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				fcm_mask=None,
			)

		attn = self.attention_performer(
			query_states=query,
			key_states=key,
			value_states=value,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=self.causal,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query.shape[1],
			key_value_sequence_length=key.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=None,
			causal_mask=causal_mask,
		)
		attn_output = self.shard_attention_prod(self._merge_heads(attn.attention_outputs))
		attn_output = self.c_proj(attn_output)
		attn_output = self.resid_dropout(attn_output)

		outputs = (
			(attn_output, attn.attention_weights) if output_attentions else (attn_output,)
		)
		return outputs


class GPT2MLP(nn.Module):
	def __init__(
		self,
		config: GPT2Config,
		intermediate_size: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[jax.lax.Precision] = None,
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
		return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


class GPT2Block(nn.Module):
	def __init__(
		self,
		config: GPT2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[jax.lax.Precision] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		hidden_size = self.config.hidden_size
		inner_dim = (
			self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
		)

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
		attention_mask=None,
		causal_mask=None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
	):
		residual = hidden_states
		hidden_states = self.ln_1(hidden_states)
		attn_outputs = self.attn(
			hidden_states,
			None,
			attention_mask,
			causal_mask,
			cache_view,
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

			cross_attn_outputs = self.crossattention(
				hidden_states,
				encoder_hidden_states,
				encoder_attention_mask,
				causal_mask,
				None,
				output_attentions,
			)
			attn_output = cross_attn_outputs[0]
			hidden_states = residual + attn_output
			outputs = outputs + cross_attn_outputs[1:]

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

		outputs = (hidden_states,) + outputs

		return outputs


@register_module(
	"base-module",
	config=GPT2Config,
	model_type="gpt2",
	embedding_layer_names=["wte", "wpe"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
class GPT2Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: GPT2Config,
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		batch_size, sequence_length = input_ids.shape
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
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
		all_cross_attentions = (
			() if (output_attentions and encoder_hidden_states is not None) else None
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
				encoder_hidden_states=encoder_hidden_states,
				encoder_attention_mask=encoder_attention_mask,
				cache_view=past_key_values.views[idx],
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

		hidden_states = outputs[0]
		hidden_states = self.ln_f(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

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
class GPT2LMHeadModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: GPT2Config,
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
		self.transformer = GPT2Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
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
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		encoder_hidden_states: tp.Optional[chex.Array] = None,
		encoder_attention_mask: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		outputs = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			self.lm_head.kernel.value = self.transformer.wte.embedding.value.T
			lm_logits = self.lm_head(hidden_states)
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
