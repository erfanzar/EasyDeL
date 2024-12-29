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
from functools import cached_property, partial

import chex
import jax
from einops import rearrange
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.infra.utils import (
	auto_remat,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.mosaic_mpt.mosaic_configuration import (
	MptConfig as MptConfig,
)


class MptMLP(nn.Module):
	def __init__(
		self,
		config: MptConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		linear_class = partial(
			nn.Linear,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			use_bias=config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.up_proj = linear_class(
			self.config.hidden_size,
			self.config.expansion_ratio * self.config.hidden_size,
			rngs=rngs,
		)
		self.down_proj = linear_class(
			self.config.expansion_ratio * self.config.hidden_size,
			self.config.hidden_size,
			rngs=rngs,
		)
		self.hidden_dropout = nn.Dropout(
			self.config.attn_config.attn_pdrop,
			rngs=rngs,
		)

	def __call__(self, hidden_states: chex.Array, residual: chex.Array):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			jax.nn.gelu(self.up_proj(hidden_states), approximate=False)
		)
		return self.hidden_dropout(hidden_states) + residual


class MptAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: MptConfig,
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
		self.hidden_size = config.hidden_size
		self.Wqkv = nn.Linear(
			config.hidden_size,
			config.hidden_size * 3,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			use_bias=config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.out_proj = nn.Linear(
			config.hidden_size,
			config.hidden_size,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			use_bias=config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.dropout = nn.Dropout(
			self.config.attn_config.attn_pdrop,
			rngs=rngs,
		)

		self.hidden_size = self.config.hidden_size
		self.n_heads = self.config.n_heads
		self.max_seq_length = self.config.max_seq_len
		self.head_dim = self.hidden_size // self.n_heads
		self.softmax_scale = self.config.attn_config.softmax_scale

		if self.softmax_scale is None:
			self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=self.config.attn_config.attn_pdrop,
			num_q_heads=self.config.n_heads,
			num_kv_heads=self.config.n_heads,
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			dtype=self.config.attn_dtype,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			axis_name=self.config.attention_axis_name,
			base_config=self.config,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		position_bias: chex.Array | tp.Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
	):
		inp_shape = hidden_states.shape
		mixed_qkv = self.Wqkv(hidden_states)
		query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

		query_states = rearrange(
			query_states,
			"b s (h d) -> b s h d",
			h=self.config.n_heads,
		)
		key_states = rearrange(
			key_states,
			"b s (h d) -> b s h d",
			h=self.config.n_heads,
		)
		value_states = rearrange(
			value_states,
			"b s (h d) -> b s h d",
			h=self.config.n_heads,
		)
		(
			key_states,
			value_states,
			attention_mask,
			_,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=cache_view,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=fcm_mask,
		)
		if position_bias is not None:
			position_bias_query_index = max(0, position_bias.shape[2] - query_states.shape[1])
			position_bias_key_index = max(0, position_bias.shape[3] - key_states.shape[1])

			position_bias = position_bias[
				:,
				:,
				position_bias_query_index:,
				position_bias_key_index:,
			]
		attention_mask = attention_mask.repeat(position_bias.shape[1], 1)
		attention_bias = lax.select(
			attention_mask.astype("bool"),
			jnp.full(attention_mask.shape, 0.0).astype(self.dtype)
			+ position_bias.astype(self.dtype),
			jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
		)

		attention = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			causal_mask=causal_mask,
			attention_mask=attention_mask,
			deterministic=self.dropout.deterministic,
			segment_ids=segment_ids,
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=cache_view is not None,
			bias=attention_bias,
			causal=False,
		)

		attn_output = self.out_proj(
			self.shard_attention_prod(
				attention.attention_outputs.reshape(inp_shape),
			)
		)

		return (
			(attn_output, attention.attention_weights)
			if output_attentions
			else (attn_output,)
		)


class MptBlock(nn.Module):
	def __init__(
		self,
		config: MptConfig,
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
		attn_block = MptAttention
		mlp_block = MptMLP
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)

		self.norm_1 = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.use_norm_bias,
			rngs=rngs,
		)
		self.attn = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.norm_2 = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.use_norm_bias,
			rngs=rngs,
		)
		self.ffn = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.dropout_rate = self.config.attn_config.attn_pdrop
		self.resid_attn_dropout = nn.Dropout(self.dropout_rate, rngs=rngs)

	def __call__(
		self,
		hidden_states: chex.Array,
		position_bias: chex.Array | tp.Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
	):
		attn_out = self.attn(
			self.norm_1(hidden_states),
			position_bias,
			attention_mask,
			causal_mask,
			segment_ids,
			cache_view,
			output_attentions,
			fcm_mask,
		)
		attn_outputs, attn_weights = attn_out if output_attentions else (attn_out[0], None)
		hidden_states = self.resid_attn_dropout(attn_outputs) + hidden_states
		output = self.ffn(self.norm_2(hidden_states), hidden_states)
		outputs = (output,)
		if output_attentions:
			outputs += (attn_weights,)

		return outputs


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
	alibi = jnp.arange(
		1 - sequence_length,
		1,
		dtype="i4",
	).reshape(
		1,
		1,
		1,
		sequence_length,
	)
	num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
	base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
	base = base * (alibi_bias_max / num_heads_power_of_2)

	slopes = 1.0 / jnp.pow(2, base)
	slopes = slopes.reshape(
		1,
		num_heads_power_of_2,
		1,
		1,
	)

	if num_heads_power_of_2 != num_heads:
		slopes = jnp.concat(
			[slopes[:, 1::2, ...], slopes[:, ::2, ...]],
			axis=1,
		)[:, :num_heads, ...]

	alibi = alibi * slopes
	return alibi


@register_module(
	"base-module",
	config=MptConfig,
	model_type="mpt",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class MptModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: MptConfig,
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
		self.wte = nn.Embed(
			num_embeddings=config.vocab_size,
			features=config.d_model,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
		)

		self.blocks = [
			MptBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.n_layers)
		]

		self.norm_f = nn.LayerNorm(
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			epsilon=config.layer_norm_epsilon,
			use_bias=config.use_norm_bias,
			rngs=rngs,
		)

	@cached_property
	def alibi(self):
		return build_mpt_alibi_tensor(
			sequence_length=self.config.max_seq_len,
			num_heads=self.config.n_heads,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		all_hidden_states = () if output_hidden_states else None
		all_attentions = () if output_attentions else None
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.wte(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.blocks))

		for idx, block in enumerate(self.blocks):
			output = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
				cache_view=past_key_values.views[idx],
				position_bias=self.alibi,
				segment_ids=segment_ids,
			)
			hidden_states = output[0]
			if output_attentions:
				all_attentions += (output[-1],)
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
		hidden_states = self.norm_f(hidden_states)
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (hidden_states, all_hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(value for value in outputs if value is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	"causal-language-model",
	config=MptConfig,
	model_type="mpt",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class MptForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: MptConfig,
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
		self.transformer = MptModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			use_bias=config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
		**kwargs,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		outputs: FlaxBaseModelOutput = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			segment_ids=segment_ids,
			inputs_embeds=inputs_embeds,
			past_key_values=past_key_values,
			output_hidden_states=output_hidden_states,
			output_attentions=output_attentions,
			return_dict=True,
		)
		last_hidden_state = outputs.last_hidden_state

		if self.config.use_lm_head:
			self.lm_head.kernel.value = self.transformer.wte.embedding.value.T
			logits = self.lm_head(last_hidden_state)
		else:
			logits = self.lm_head(last_hidden_state)

		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
