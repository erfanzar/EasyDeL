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
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import MoeCausalLMOutput, MoeModelOutput
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm
from easydel.modules.arctic.arctic_configuration import ArcticConfig


class ArcticAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: ArcticConfig,
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
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings

		linear = partial(
			nn.Linear,
			use_bias=getattr(self.config, "attention_bias", False),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj = linear(
			config.hidden_size,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear(
			config.hidden_size,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear(
			config.hidden_size,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear(
			self.num_heads * self.head_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
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
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			sm_scale=1 / math.sqrt(self.head_dim),
			backward_pass_impl=self.config.flash_attention_backward_pass_impl,
			base_config=self.config,
			mesh=self.config.mesh,
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
	):
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
			positions=position_ids,
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
		attn_output = self.o_proj(attn_output)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class ArcticMLP(nn.Module):
	def __init__(
		self,
		config: ArcticConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		is_residual_mlp: bool = False,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.is_residual_mlp = is_residual_mlp
		self.hidden_dim = config.hidden_size
		self.ffn_dim = (
			config.intermediate_size if not self.is_residual_mlp else self.hidden_dim
		)
		linear_class = partial(
			nn.Linear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.w1 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
		self.w3 = linear_class(self.hidden_dim, self.ffn_dim, rngs=rngs)
		self.w2 = linear_class(self.ffn_dim, self.hidden_dim, rngs=rngs)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: chex.Array):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		w1 = self.act_fn(self.w1(hidden_states))
		w3 = self.w3(hidden_states)
		return self.w2(w1 * w3)


class ArcticMoeBlock(nn.Module):
	def __init__(
		self,
		config: ArcticConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.rngs = rngs

		self.hidden_dim = config.hidden_size
		self.num_experts = config.num_local_experts

		self.top_k = config.num_experts_per_tok
		self.is_moe_layer = (layer_idx + 1) % config.moe_layer_frequency == 0

		if self.is_moe_layer:
			self.gate = nn.Linear(
				config.hidden_size,
				config.num_local_experts,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				kernel_init=nn.initializers.normal(),
				rngs=rngs,
			)
			self.experts = [
				ArcticMLP(
					config=config,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
				for _ in range(config.num_local_experts)
			]
		else:
			self.mlp = ArcticMLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				is_residual_mlp=False,
				rngs=rngs,
			)

	def _call_moe(self, hidden_states: chex.Array) -> tp.Tuple[chex.Array, chex.Array]:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

		router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
			jnp.promote_types(self.dtype, jnp.float32)
		)
		routing_weights, selected_experts = jax.lax.top_k(
			router_logits, k=self.config.num_experts_per_tok
		)
		routing_weights = jax.nn.softmax(
			routing_weights.astype(
				jnp.promote_types(self.dtype, jnp.float32),
			),
			axis=-1,
		)
		final_hidden_state = jnp.zeros_like(hidden_states)

		for index in range(self.config.num_local_experts):
			expert_layer_output = (
				block_wise_ffn(
					self.experts[index],
					hidden_states,
					self.config.scan_mlp_chunk_size,
				)
				if self.config.use_scan_mlp
				else self.experts[index](hidden_states)
			)
			expert_layer_output_exp = (
				jnp.sum(
					jnp.multiply(selected_experts == index, routing_weights),
					axis=-1,
				)[:, :, None]
				* expert_layer_output
			)
			final_hidden_state += expert_layer_output_exp

		return final_hidden_state, router_logits

	def __call__(self, hidden_states: chex.Array):
		if self.is_moe_layer:
			return self._call_moe(hidden_states=hidden_states)
		return self.mlp(hidden_states), jnp.array(0.0, dtype=hidden_states.dtype)


class ArcticDecoderLayer(nn.Module):
	def __init__(
		self,
		config: ArcticConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.rngs = rngs
		attn_block = ArcticAttention
		mlp_block = ArcticMoeBlock

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.block_sparse_moe = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			layer_idx=layer_idx,
		)
		self.input_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.post_attention_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.parallel_attn_mlp_res = (
			self.config.parallel_attn_mlp_res and self.block_sparse_moe.is_moe_layer
		)
		if self.parallel_attn_mlp_res:
			self.residual_layernorm = RMSNorm(
				dim=config.hidden_size,
				eps=config.rms_norm_eps,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)
			self.residual_mlp = ArcticMLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				is_residual_mlp=True,
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
	) -> tp.Tuple[chex.Array, tp.Optional[chex.Array], chex.Array]:
		residual_input = hidden_states
		hidden_states = self.input_layernorm(hidden_states)
		attn_out = self.self_attn(
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
		hidden_states, self_attn_weights = (
			attn_out if output_attentions else (attn_out[0], None)
		)
		hidden_states = residual_input + hidden_states

		residual_attn = hidden_states
		if self.parallel_attn_mlp_res:
			hidden_states = self.residual_layernorm(hidden_states)
			hidden_states = self.residual_mlp(hidden_states)
			residual_residual = residual_attn + hidden_states
			# parallel mlp moe part
			hidden_states = self.post_attention_layernorm(residual_input)
			hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
			hidden_states = residual_residual + hidden_states
		else:
			hidden_states = self.post_attention_layernorm(hidden_states)
			hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
			hidden_states = residual_attn + hidden_states

		outputs = (hidden_states,)
		if output_attentions:
			outputs += (self_attn_weights,)

		outputs += (gate_loss,)
		return outputs


@register_module(
	"base-module",
	config=ArcticConfig,
	model_type="arctic",
	embedding_layer_names=["embed_tokens"],
)
class ArcticModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: ArcticConfig,
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
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			ArcticDecoderLayer(
				layer_idx=layer_idx,
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for layer_idx in range(config.num_hidden_layers)
		]

		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@cached_property
	def causal_mask(self):
		return self.config.get_basic_causal_mask()

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies()

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeModelOutput, tp.Tuple]:
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)

		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)

		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_router_losses = ()

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

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
		for idx, layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			outputs = layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_view=past_key_values.views[idx],
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
			)
			hidden_states = outputs[0]
			if output_attentions:
				all_self_attns += (outputs[1],)
			all_router_losses += (outputs[-1],)

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
					all_router_losses,
				]
				if v is not None
			)
		return MoeModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			all_router_losses=all_router_losses,
		)


@register_module(
	"causal-language-model",
	config=ArcticConfig,
	model_type="arctic",
	embedding_layer_names=["embed_tokens"],
)
class ArcticForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: ArcticConfig,
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
		self.model = ArcticModel(
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
			kernel_init=nn.initializers.normal(config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | tp.Tuple:
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
		)
		hidden_states = outputs.last_hidden_state
		if self.config.tie_word_embeddings:
			# self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
			# lm_logits = self.lm_head(hidden_states)
			lm_logits = hidden_states @ self.model.embed_tokens.embedding.value.T
		else:
			lm_logits = self.lm_head(hidden_states)
		aux_loss = sum(outputs[-1]) * self.config.router_aux_loss_coef
		if not return_dict:
			outputs = (lm_logits,) + tuple(
				v
				for v in [
					aux_loss,
					outputs.hidden_states,
					outputs.attentions,
					outputs.all_router_losses,
				]
				if v is not None
			)
			return outputs

		return MoeCausalLMOutput(
			aux_loss=aux_loss,
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			all_router_losses=outputs.all_router_losses,
		)
