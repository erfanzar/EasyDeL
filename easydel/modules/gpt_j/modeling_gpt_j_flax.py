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
from functools import partial
from typing import Optional, Union

import chex
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import Dense, make_causal_mask

from easydel.etils.etils import EasyDeLGradientCheckPointers, get_logger
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.gpt_j.gpt_j_configuration import GPTJConfig as GPTJConfig
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module

logger = get_logger(__name__)


class FlaxGPTJAttention(FlaxAttentionModule):
	config: GPTJConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None
	causal: bool = True
	is_cross_attention: bool = False

	def setup(self):
		config = self.config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads

		self.rotary_dim = config.rotary_dim
		dense = partial(
			Dense,
			self.embed_dim,
			use_bias=False,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			param_dtype=self.dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
		self.out_proj = dense()

		self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

		self.causal_mask = make_causal_mask(
			jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=self.embed_dim,
			rotary_dim=self.rotary_dim,
			base=10000,
			is_neox_style=False,
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

	def _split_heads(self, hidden_states):
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.num_heads, self.head_dim)
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
		query = self.q_proj(hidden_states)
		key = self.k_proj(hidden_states)
		value = self.v_proj(hidden_states)

		query = self._split_heads(query)
		key = self._split_heads(key)
		value = self._split_heads(value)

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
		attn_output = self.out_proj(attn_output)
		attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class FlaxGPTJMLP(nn.Module):
	config: GPTJConfig
	intermediate_size: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		embed_dim = self.config.hidden_size
		kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
		self.fc_in = Dense(
			self.intermediate_size,
			dtype=self.dtype,
			param_dtype=self.dtype,
			precision=self.precision,
			kernel_init=kernel_init,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.fc_out = Dense(
			embed_dim,
			dtype=self.dtype,
			param_dtype=self.dtype,
			precision=self.precision,
			kernel_init=kernel_init,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.act = ACT2FN[self.config.activation_function]
		self.dropout = flax.linen.Dropout(rate=self.config.resid_pdrop)

	def __call__(self, hidden_states, deterministic: bool = True):
		hidden_states = self.fc_out(self.act(self.fc_in(hidden_states)))
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		return hidden_states


class FlaxGPTJBlock(nn.Module):
	config: GPTJConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		hidden_size = self.config.hidden_size
		inner_dim = (
			self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
		)

		self.ln_1 = nn.LayerNorm(
			epsilon=self.config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.dtype,
		)
		attn_block = FlaxGPTJAttention

		mlp_block = FlaxGPTJMLP

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
			param_dtype=self.dtype,
			precision=self.precision,
		)

		self.mlp = mlp_block(
			self.config,
			inner_dim,
			dtype=self.dtype,
			param_dtype=self.dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask=None,
		position_ids=None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		frequencies: Optional[chex.Array] = None,
	):
		residual = hidden_states
		hidden_states = self.ln_1(hidden_states)
		# hidden_states
		# attention_mask
		# position_ids
		# deterministic: bool = True
		# init_cache: bool = False
		# output_attentions: bool = False
		attn_outputs = self.attn(
			hidden_states,
			attention_mask,
			position_ids,
			None,
			deterministic,
			init_cache,
			output_attentions,
			frequencies,
		)
		attn_output = attn_outputs[0]
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp, hidden_states, self.config.scan_mlp_chunk_size, deterministic
			)
		else:
			feed_forward_hidden_states = self.mlp(hidden_states, deterministic)
		# residual connection
		hidden_states = attn_output + feed_forward_hidden_states + residual

		return (hidden_states,) + attn_outputs[1:]


class FlaxGPTJBlockCollection(nn.Module):
	config: GPTJConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.blocks = [
			FlaxGPTJBlock(
				self.config,
				name=str(i),
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self._frequencies = self.config.get_basic_frequencies(
			head_size=self.config.hidden_size,
			rotary_dim=self.config.rotary_dim,
			base=10000,
		)

	def __call__(
		self,
		hidden_states,
		attention_mask=None,
		position_ids=None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
				frequencies=self._frequencies,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		outputs = (hidden_states, all_hidden_states, all_attentions)

		return outputs


@register_module(
	"base-module",
	config=GPTJConfig,
	model_type="gptj",
	embedding_layer_names=["wte"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
@wrap_easydel_module(config_class=GPTJConfig, base_model_prefix="transformer")
class FlaxGPTJModel(nn.Module):
	config: GPTJConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.embed_dim = self.config.hidden_size
		self.wte = nn.Embed(
			self.config.vocab_size,
			self.embed_dim,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.dropout = flax.linen.Dropout(rate=self.config.embd_pdrop)
		self.h = FlaxGPTJBlockCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ln_f = nn.LayerNorm(
			epsilon=self.config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
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
		input_embeds = self.wte(input_ids.astype("i4"))

		hidden_states = self.dropout(input_embeds, deterministic=deterministic)

		outputs = self.h(
			hidden_states,
			attention_mask,
			position_ids=position_ids,
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

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


@register_module(
	"causal-language-model",
	config=GPTJConfig,
	model_type="gptj",
	embedding_layer_names=["wte"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
@wrap_easydel_module(config_class=GPTJConfig, base_model_prefix="transformer")
class FlaxGPTJForCausalLM(nn.Module):
	config: GPTJConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.transformer = FlaxGPTJModel.flax_module(
			self.config,
			dtype=self.dtype,
			param_dtype=self.dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			param_dtype=self.dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids,
		attention_mask,
		position_ids,
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

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
