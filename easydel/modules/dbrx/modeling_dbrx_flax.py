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
from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import MoeCausalLMOutput, MoeModelOutput
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.dbrx.dbrx_configuration import (
	DbrxAttentionConfig as DbrxAttentionConfig,
)
from easydel.modules.dbrx.dbrx_configuration import DbrxConfig as DbrxConfig
from easydel.modules.dbrx.dbrx_configuration import DbrxFFNConfig as DbrxFFNConfig


class DbrxAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.num_attention_heads = self.config.n_heads
		self.num_key_value_heads = self.config.attn_config.kv_n_heads
		config = self.config
		self.hidden_size = config.hidden_size
		self.head_dim = self.config.d_model // self.config.n_heads
		self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

		if self.num_key_value_groups == 1:
			assert self.num_attention_heads == self.config.attn_config.kv_n_heads
		self.Wqkv = nn.Linear(
			config.hidden_size,
			self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.out_proj = nn.Linear(
			config.hidden_size,
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.rotary = self.config.get_basic_rope(
			dtype=self.dtype,
			rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			is_neox_style=True,
			base=self.config.attn_config.rope_theta,
		)
		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=self.num_attention_heads,
			num_kv_heads=self.num_key_value_heads,
			attention_dropout=self.config.attn_config.attn_pdrop,
			head_dims=self.head_dim,
			shard_attention_computation=self.config.shard_attention_computation,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			axis_name=self.config.attention_axis_name,
			base_config=self.config,
		)
		self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		batch_size, sequence_length = hidden_states.shape[:2]
		qkv_states = self.Wqkv(hidden_states)
		if self.config.attn_config.clip_qkv is not None:
			qkv_states = qkv_states.clip(
				min=-self.config.attn_config.clip_qkv,
				max=self.config.attn_config.clip_qkv,
			)

		query_size = self.hidden_size
		key_size = self.num_key_value_heads * self.head_dim

		query_states, key_value_states = jnp.split(qkv_states, [query_size], axis=2)
		key_states, value_states = jnp.split(key_value_states, [key_size], axis=2)
		query_states = query_states.reshape(
			batch_size,
			sequence_length,
			self.num_attention_heads,
			self.head_dim,
		)
		key_states = key_states.reshape(
			batch_size,
			sequence_length,
			self.num_key_value_heads,
			self.head_dim,
		)
		value_states = value_states.reshape(
			batch_size,
			sequence_length,
			self.num_key_value_heads,
			self.head_dim,
		)

		query_states, key_states = self.rotary(
			position_ids,
			query=query_states,
			key=key_states,
			frequencies=frequencies,
		)
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
			fcm_mask=fcm_mask,
		)

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=segment_ids,
			causal_mask=causal_mask,
		)

		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.out_proj(attn_output)

		attn_output = self.resid_dropout(attn_output)
		outputs = (attn_output,)
		if output_attentions:
			outputs += (output_attentions,)
		return outputs


class DbrxNormAttentionNorm(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.norm_1 = nn.LayerNorm(
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			rngs=rngs,
		)
		self.attn = DbrxAttention(  # statics 3,5,6,7
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm_2 = nn.LayerNorm(
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			rngs=rngs,
		)

		self.dropout = nn.Dropout(
			self.config.resid_pdrop,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	) -> tp.Tuple[chex.Array, chex.Array, tp.Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array, tp.Optional[chex.Array]]: A tuple containing the residual_states, hidden states, and the attention weights.
		"""
		residual_states = hidden_states
		hidden_states = self.norm_1(hidden_states)

		attn_out = self.attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			causal_mask=causal_mask,
			segment_ids=segment_ids,
			fcm_mask=fcm_mask,
			frequencies=frequencies,
			cache_view=cache_view,
		)
		hidden_states, attn_weights = attn_out if output_attentions else (attn_out[0], None)
		hidden_states = self.dropout(hidden_states)
		hidden_states = hidden_states + residual_states

		residual_states = hidden_states
		hidden_states = self.norm_2(hidden_states)

		return residual_states, hidden_states, attn_weights


class DbrxExpertGLU(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		shape = (
			self.config.ffn_config.moe_num_experts * self.config.ffn_config.ffn_hidden_size,
			self.config.d_model,
		)
		init_fn = nn.initializers.normal(dtype=self.dtype)
		self.w1 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
		self.v1 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
		self.w2 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
		self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

	def __call__(self, x: chex.Array, expert_idx: int) -> chex.Array:
		expert_shape = (
			self.config.ffn_config.moe_num_experts,
			self.config.ffn_config.ffn_hidden_size,
			self.config.d_model,
		)
		expert_w1 = self.w1.value.reshape(expert_shape)[expert_idx]
		expert_v1 = self.v1.value.reshape(expert_shape)[expert_idx]
		expert_w2 = self.w2.value.reshape(expert_shape)[expert_idx]

		x1 = jnp.matmul(
			x,
			jnp.expand_dims(expert_w1.T, 0),
			precision=self.precision,
		)
		x2 = jnp.matmul(
			x,
			jnp.expand_dims(expert_v1.T, 0),
			precision=self.precision,
		)
		x1 = self.activation_fn(x1)
		x1 = x1 * x2
		x1 = jnp.matmul(
			x1,
			jnp.expand_dims(expert_w2, 0),
			precision=self.precision,
		)
		return x1


class DbrxExperts(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.mlp = DbrxExpertGLU(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		x: chex.Array,
		weights: chex.Array,
		top_weights: chex.Array,
		top_experts: chex.Array,
	):
		final_hidden_state = jnp.zeros_like(x)
		for index in range(self.config.ffn_config.moe_num_experts):
			output_moe_layer = self.mlp(x, index)
			final_hidden_state += (
				jnp.sum(jnp.multiply(index == top_experts, top_weights), axis=-1)[:, :, None]
				* output_moe_layer
			)
		return final_hidden_state


class DbrxRouter(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.hidden_size = self.config.d_model
		self.moe_num_experts = self.config.ffn_config.moe_num_experts
		self.moe_top_k = self.config.ffn_config.moe_top_k
		self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
		self.moe_normalize_expert_weights = (
			self.config.ffn_config.moe_normalize_expert_weights
		)
		self.uniform_expert_assignment = self.config.ffn_config.uniform_expert_assignment

		self.layer = nn.Linear(
			config.hidden_size,
			self.moe_num_experts,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def jitter(self, x: chex.Array) -> chex.Array:
		if self.moe_jitter_eps is None:
			raise RuntimeError("The router does not have moe_jitter_eps set.")
		low = 1.0 - self.moe_jitter_eps
		high = 1.0 + self.moe_jitter_eps
		noise = jax.random.normal(self.make_rng("params"), x.shape, dtype=x.dtype)
		return low + noise * (high - low)

	def __call__(
		self, x: chex.Array, deterministic: bool = True
	) -> tp.Tuple[chex.Array, chex.Array, chex.Array]:
		if not deterministic and self.moe_jitter_eps is not None:
			x = x * self.jitter(x)

		weights = self.layer(x.astype(jnp.promote_types(self.dtype, jnp.float32)))
		weights = jax.nn.softmax(weights.astype(jnp.promote_types(self.dtype, jnp.float32)))
		top_weights, top_experts = jax.lax.top_k(weights, self.moe_top_k)

		if self.moe_normalize_expert_weights:
			top_weights = top_weights / jnp.linalg.norm(
				top_weights,
				ord=int(self.moe_normalize_expert_weights),
				axis=-1,
				keepdims=True,
			)

		if self.uniform_expert_assignment:
			top_experts = jax.lax.stop_gradient(
				(
					jnp.arange(
						0,
						jnp.prod(
							jnp.asarray(top_experts.shape, dtype=jnp.int32),
							dtype=jnp.int32,
						),
						dtype=top_experts.dtype,
					)
					% self.moe_num_experts
				).reshape(top_experts.shape)
			)

		weights = weights.astype(x.dtype)
		top_weights = top_weights.astype(x.dtype)
		return weights, top_weights, top_experts


class DbrxFFN(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.router = DbrxRouter(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.experts = DbrxExperts(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, x: chex.Array) -> tp.Tuple[chex.Array, chex.Array]:
		x = control_mlp_sharding(x, self.config.partition_axis)
		weights, top_weights, top_experts = self.router(x)
		out = self.experts(x, weights, top_weights, top_experts)
		return out, weights


class DbrxBlock(nn.Module):
	def __init__(
		self,
		config: DbrxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.hidden_size = self.config.d_model
		self.resid_pdrop = self.config.resid_pdrop
		attn_block = DbrxNormAttentionNorm
		ffn_block = DbrxFFN
		attn_block, ffn_block = auto_remat(
			attn_block,
			ffn_block,
			policy=config.gradient_checkpointing,
		)
		self.norm_attn_norm = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.ffn = ffn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		output_router_logits: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	) -> tp.Tuple[chex.Array, chex.Array, tp.Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights.
		    output_router_logits (bool): If True, outputs router logits.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array, tp.Optional[chex.Array]]: A tuple containing the residual_states, hidden states, and the attention weights.
		"""

		resid_states, hidden_states, self_attn_weights = self.norm_attn_norm(
			hidden_states,
			attention_mask,
			position_ids,
			causal_mask,
			cache_view,
			segment_ids,
			output_attentions,
			fcm_mask,
			frequencies,
		)

		hidden_states, router_logits = self.ffn(hidden_states)
		hidden_states = resid_states + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if output_router_logits:
			outputs += (router_logits,)

		return outputs


@register_module(
	"base-module",
	config=DbrxConfig,
	model_type="dbrx",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class DbrxModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: DbrxConfig,
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
		self.padding_idx = self.config.pad_token_id
		self.vocab_size = self.config.vocab_size
		self.emb_pdrop = self.config.emb_pdrop

		self.wte = nn.Embed(
			self.config.vocab_size,
			self.config.d_model,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.blocks = [
			DbrxBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.n_layers)
		]
		self.norm_f = nn.LayerNorm(
			self.config.hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies(
			rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			base=self.config.attn_config.rope_theta,
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> MoeModelOutput | tp.Tuple:
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.wte(input_ids.astype("i4"))

		batch_size, sequence_length = inputs_embeds.shape[:2]
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)

		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_router_logits = (
			output_router_logits
			if output_router_logits is not None
			else self.config.output_router_logits
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		hidden_states = inputs_embeds
		all_hidden_states = ()
		all_router_logits = ()
		all_attentions = ()
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.blocks))
		for idx, block in enumerate(self.blocks):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				segment_ids=segment_ids,
				cache_view=past_key_values.views[idx],
				output_attentions=output_attentions,
				output_router_logits=output_router_logits,
				frequencies=self.frequencies,
			)
			hidden_states = outputs[0]
			if output_attentions:
				all_attentions += (outputs[1],)
			if output_router_logits:
				all_router_logits += (outputs[-1],)

		hidden_states = self.norm_f(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					all_hidden_states,
					all_attentions,
					all_router_logits,
				]
				if v is not None
			)
		return MoeModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			router_logits=all_router_logits,
		)


@register_module(
	"causal-language-model",
	config=DbrxConfig,
	model_type="dbrx",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class DbrxForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: DbrxConfig,
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
		self.transformer = DbrxModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=False,
			rngs=rngs,
			kernel_init=nn.initializers.normal(config.initializer_range),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | tp.Tuple:
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits
		outputs = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			output_router_logits=output_router_logits,
			past_key_values=past_key_values,
			return_dict=True,
			segment_ids=segment_ids,
		)
		logits = self.lm_head(outputs.last_hidden_state)
		batch_size, seq_length, hd = logits.shape
		aux_loss = None
		if output_router_logits and outputs.router_logits is not None:
			aux_loss = auxiliary_load_balancing_loss_func(
				gate_logits=tuple(  # type:ignore
					[
						logit.reshape(batch_size * seq_length, -1)
						for logit in outputs.router_logits
					]  # type:ignore
				),
				num_experts=self.config.ffn_config.moe_num_experts,
				top_k=self.config.ffn_config.moe_top_k,
				attention_mask=attention_mask,
			)
			aux_loss = aux_loss * self.config.router_aux_loss_coef
		if not return_dict:
			outputs = (logits,) + tuple(
				v
				for v in [
					aux_loss,
					outputs.hidden_states,
					outputs.attentions,
					outputs.router_logits,
				]
				if v is not None
			)
			return outputs

		return MoeCausalLMOutput(
			aux_loss=aux_loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			router_logits=outputs.router_logits,
		)
