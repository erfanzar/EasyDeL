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

import functools
import math
from typing import Optional, Union

import chex
import flax
import jax
from einops import rearrange
from flax import linen as nn
from flax.linen import Dense
from jax import numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.gpt_neo_x.gpt_neo_x_configuration import (
	GPTNeoXConfig as GPTNeoXConfig,
)
from easydel.modules.modeling_flax_outputs import FlaxBaseModelOutput
from easydel.modules.modeling_utils import wrap_easydel_module


class FlaxGPTNeoXAttention(FlaxAttentionModule):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.head_size = self.config.hidden_size // self.config.num_attention_heads
		self.rotary = self.config.get_basic_rope(
			dtype=self.dtype,
			head_size=self.head_size,
			rotary_dim=self.head_size,
			base=10000,
		)
		dense_class = functools.partial(
			Dense,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			param_dtype=self.dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.w_qkv = dense_class(3 * self.config.hidden_size)
		self.w_o = dense_class(self.config.hidden_size)
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

	def __call__(
		self,
		hidden_states,
		attention_mask,
		position_ids,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		frequencies: Optional[chex.Array] = None,
	):
		b, s, d = hidden_states.shape
		query, key, value = jnp.split(
			self.w_qkv(hidden_states),
			indices_or_sections=3,
			axis=-1,
		)
		query = rearrange(query, "b s (h d) -> b s h d", h=self.config.num_attention_heads)
		key = rearrange(key, "b s (h d) -> b s h d", h=self.config.num_attention_heads)
		value = rearrange(value, "b s (h d) -> b s h d", h=self.config.num_attention_heads)

		query, key = self.rotary(
			positions=position_ids,
			query=query,
			key=key,
			frequencies=frequencies,
		)
		query_length, key_length = query.shape[1], key.shape[1]

		dropout_rng = None
		if not deterministic and self.config.attn_pdrop > 0.0:
			dropout_rng = self.make_rng("dropout")
		(
			query,
			key,
			value,
			attention_mask,
			attention_bias,
		) = self.concatenate_to_cache(
			init_cache=init_cache,
			query=query,
			key=key,
			value=value,
			attention_mask=attention_mask,
			causal_mask=self.causal_mask,
			fcm_mask=None,
		)
		attentions = self.attention_performer(
			query_states=query,
			key_states=key,
			value_states=value,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=dropout_rng,
			deterministic=deterministic,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=self.has_variable("cache", "cached_key") or init_cache,
			segment_ids=segment_ids,
			causal_mask=self.causal_mask,
		)
		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.w_o(attn_output.reshape(b, s, d))
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class FlaxGPTNeoXMlp(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		dense_class = functools.partial(
			Dense,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			param_dtype=self.dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.dense_h_to_4h = dense_class(self.config.intermediate_size)
		self.dense_4h_to_h = dense_class(self.config.hidden_size)
		self.act = ACT2FN[self.config.hidden_act]

	def __call__(self, x):
		x = control_mlp_sharding(x, self.config.partition_axis)
		return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class FlaxGPTNeoXBlock(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.use_parallel_residual = self.config.use_parallel_residual
		self.input_layernorm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps,
			dtype=self.dtype,
		)
		self.post_attention_layernorm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps,
			dtype=self.dtype,
		)

		attn_block = FlaxGPTNeoXAttention
		mlp_block = FlaxGPTNeoXMlp

		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = flax.linen.partitioning.remat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(4, 5, 6, 7),
			)

			mlp_block = flax.linen.partitioning.remat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)
		self.attention = attn_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.mlp = mlp_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		position_ids,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		frequencies: Optional[chex.Array] = None,
	):
		attn_out = self.attention(
			self.input_layernorm(hidden_states),
			attention_mask,
			position_ids,
			segment_ids,
			deterministic,
			init_cache,
			output_attentions,
			frequencies,
		)
		attn = attn_out[0]
		if self.use_parallel_residual:
			mlp = self.mlp(self.post_attention_layernorm(hidden_states))
			hidden_states = mlp + hidden_states + attn
		else:
			hidden_states = attn + hidden_states
			hidden_states = (
				self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
			)
		return (hidden_states,) + attn_out[1:]


class FlaxGPTNeoXCollection(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxGPTNeoXBlock(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_hidden_layers)
		]
		self._frequencies = self.config.get_basic_frequencies(
			head_size=self.head_size,
			rotary_dim=self.head_size,
			base=10000,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask,
		position_ids,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
	):
		for block in self.blocks:
			hidden_out = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				segment_ids=segment_ids,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=False,  # TODO Fix this one
				frequencies=self._frequencies,
			)
			hidden_states = hidden_out[0]
		return hidden_states


@register_module(
	"base-module",
	config=GPTNeoXConfig,
	model_type="gpt_neox",
	embedding_layer_names=["wte"],
)
@wrap_easydel_module(config_class=GPTNeoXConfig, base_model_prefix="transformer")
class FlaxGPTNeoXModel(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.embed_in = nn.Embed(self.config.vocab_size, self.config.hidden_size)
		self.layers = FlaxGPTNeoXCollection(
			config=self.config,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			precision=self.precision,
		)
		self.final_layer_norm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps, dtype=self.dtype
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		position_ids,
		deterministic=True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		hidden_state = self.embed_in(inputs=input_ids)
		hidden_state = self.final_layer_norm(
			self.layers(
				hidden_state=hidden_state,
				attention_mask=attention_mask,
				position_ids=position_ids,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
			)
		)
		if return_dict:
			return FlaxBaseModelOutput(last_hidden_state=hidden_state)
		else:
			return (hidden_state,)


@register_module(
	"causal-language-model",
	config=GPTNeoXConfig,
	model_type="gpt_neox",
	embedding_layer_names=["wte"],
)
@wrap_easydel_module(config_class=GPTNeoXConfig, base_model_prefix="transformer")
class FlaxGPTNeoXForCausalLM(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.transformer = FlaxGPTNeoXModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(self.config.vocab_size, use_bias=False)

	def __call__(self, input_ids, attention_mask, return_dict: bool = False):
		pred = self.transformer(
			input_ids=input_ids, attention_mask=attention_mask, return_dict=True
		).last_hidden_state
		return self.lm_head(pred)
