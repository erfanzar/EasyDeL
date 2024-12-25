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
from easydel.infra.modeling_outputs import (
	MoeCausalLMOutput,
	MoeModelOutput,
)
from easydel.infra.utils import (
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm as FlaxGrok1RMSNorm
from easydel.modules.grok_1.grok_1_configuration import Grok1Config as Grok1Config


class FlaxGrok1Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: Grok1Config,
		layer_index: int,
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
		self.layer_index = layer_index
		self.hidden_size = config.hidden_size
		self.head_dim = self.config.hidden_size // self.config.num_attention_heads
		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_key_value_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_key_value_heads
		self.q_proj = nn.Linear(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.k_proj = nn.Linear(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.v_proj = nn.Linear(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.o_proj = nn.Linear(
			config.num_attention_heads * self.head_dim,
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
			self.dtype,
			self.head_dim,
			self.head_dim,
			True,
		)
		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_key_value_heads,
			attention_dropout=self.config.attention_dropout,
			head_dims=self.head_dim,
			shard_attention_computation=self.config.shard_attention_computation,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			base_config=self.config,
		)
		self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

	def _merge_heads(self, hidden_states):
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (chex.Array): The hidden states with separate head dimensions.

		Returns:
		    chex.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

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
		query_states, key_states, value_states = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)

		query_states = query_states.reshape(
			batch_size,
			sequence_length,
			self.config.num_attention_heads,
			self.head_dim,
		)
		key_states = key_states.reshape(
			batch_size,
			sequence_length,
			self.config.num_key_value_heads,
			self.head_dim,
		)
		value_states = value_states.reshape(
			batch_size,
			sequence_length,
			self.config.num_key_value_heads,
			self.head_dim,
		)

		query_states, key_states = self.rotary(
			query=query_states,
			key=key_states,
			positions=position_ids,
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
		attn_output = self.o_proj(attn_output)

		attn_output = self.resid_dropout(attn_output)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class FlaxGrok1BLockSparseMLP(nn.Module):
	def __init__(
		self,
		config: Grok1Config,
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

		self.linear = nn.Linear(
			config.hidden_size,
			config.intermediate_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.linear_1 = nn.Linear(
			config.intermediate_size,
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.linear_v = nn.Linear(
			config.hidden_size,
			config.intermediate_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.linear_1(
			nn.gelu(self.linear(hidden_states)) * self.linear_v(hidden_states)
		)


class FlaxGrok1SparseMoeBlock(nn.Module):
	def __init__(
		self,
		config: Grok1Config,
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
		self.gate = nn.Linear(
			self.config.hidden_size,
			self.config.num_experts,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
		)

		self.experts = [
			FlaxGrok1BLockSparseMLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_experts)
		]

	def __call__(self, hidden_states: chex.Array) -> tp.Tuple[chex.Array, chex.Array]:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

		router_logits = self.gate(hidden_states).astype(
			jnp.promote_types(self.dtype, jnp.float32)
		)
		routing_weights, selected_experts = jax.lax.top_k(
			router_logits, k=self.config.num_experts_per_tok
		)
		routing_weights = jax.nn.softmax(
			routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
		)
		final_hidden_state = jnp.zeros_like(hidden_states)

		for index in range(self.config.num_experts):
			expert_layer_output = (
				block_wise_ffn(
					self.layers[index],
					hidden_states,
					self.config.scan_mlp_chunk_size,
					False,
				)
				if self.config.use_scan_mlp
				else self.layers[index](hidden_states)
			)
			expert_layer_output_exp = (
				jnp.sum(jnp.multiply(selected_experts == index, routing_weights), axis=-1)[
					:, :, None
				]
				* expert_layer_output
			)
			final_hidden_state += expert_layer_output_exp
		return (final_hidden_state, router_logits)


class FlaxGrok1DecoderLayer(nn.Module):
	def __init__(
		self,
		config: Grok1Config,
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
		attn_block = FlaxGrok1Attention
		mlp_block = FlaxGrok1SparseMoeBlock
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.attn = attn_block(
			config=self.config,
			layer_index=self.layer_index,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.moe_block = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.pre_attn_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.post_attn_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.pre_moe_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.post_moe_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
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
		output_router_logits: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	) -> tp.Tuple[chex.Array, chex.Array, tp.Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (tp.Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
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
		residual = hidden_states
		hidden_states = self.pre_attn_norm(hidden_states)
		hidden_states, attention_weights = self.attn(
			hidden_states,
			frequencies,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			cache_view,
			output_attentions,
			fcm_mask,
		)

		hidden_states = self.post_attn_norm(hidden_states)
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.pre_moe_norm(hidden_states)
		hidden_states, router_logits = self.moe_block(hidden_states)
		hidden_states = self.post_moe_norm(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)
		if output_attentions:
			outputs += (attention_weights,)
		if output_router_logits:
			outputs += (router_logits,)
		return outputs


@register_module(
	"base-module",
	config=Grok1Config,
	model_type="grok-1",
	embedding_layer_names=["embed_tokens"],
)
class Grok1Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: Grok1Config,
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
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			FlaxGrok1DecoderLayer(
				layer_index=layer_index,
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for layer_index in range(self.config.num_hidden_layers)
		]

		self.norm = FlaxGrok1RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies(
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			base=self.config.rope_theta,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> MoeModelOutput | tp.Tuple:
		"""
		Forward pass through the Grok1 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    output_router_logits (tp.Optional[bool]): If True, output router logits.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    MoeModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""
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
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_router_logits = () if output_router_logits else None

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

		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				output_attentions=output_attentions,
				output_router_logits=output_router_logits,
				cache_view=past_key_values.view[idx],
				frequencies=self.frequencies,
				causal_mask=self.causal_mask,
				segment_ids=segment_ids,
			)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

			if output_router_logits:
				all_router_logits += (layer_outputs[-1],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					all_hidden_states,
					all_self_attns,
					all_router_logits,
				]
				if v is not None
			)
		return MoeModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			router_logits=all_router_logits,
		)


@register_module(
	"causal-language-model",
	config=Grok1Config,
	model_type="grok-1",
	embedding_layer_names=["embed_tokens"],
)
class Grok1ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: Grok1Config,
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
		self.model = Grok1Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			dtype=self.dtype,
			rngs=rngs,
			param_dtype=self.param_dtype,
			precision=self.precision,
			use_bias=False,
			kernel_init=nn.initializers.normal(config.initializer_range),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.output_multiplier_scale = self.config.output_multiplier_scale

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | tp.Tuple:
		"""
		Forward pass through the Grok1 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    output_router_logits (tp.Optional[bool]): If True, output router logits.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    MoeCausalLMOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits
		outputs = self.model(
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
		logits = logits * self.output_multiplier_scale
		batch_size, seq_length, hd = logits.shape
		aux_loss = None
		if output_router_logits and outputs.router_logits is not None:
			aux_loss = auxiliary_load_balancing_loss_func(
				gate_logits=tuple(
					[
						logit.reshape(batch_size * seq_length, -1)
						for logit in outputs.router_logits
					]
				),
				num_experts=self.num_experts,
				top_k=self.num_experts_per_tok,
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
