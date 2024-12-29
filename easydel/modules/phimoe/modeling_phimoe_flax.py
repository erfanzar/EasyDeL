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
import typing as tp

import chex
import jax.lax
from chex import Array
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	with_sharding_constraint,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm as RMSNorm
from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig as PhiMoeConfig


class PhiMoEBlockSparseTop2MLP(nn.Module):
	def __init__(
		self,
		config: PhiMoeConfig,
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
		linear_class = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		ffn_dim = config.intermediate_size
		hidden_dim = config.hidden_size

		self.w1 = linear_class(hidden_dim, ffn_dim, rngs=rngs)
		self.w2 = linear_class(ffn_dim, hidden_dim, rngs=rngs)
		self.w3 = linear_class(hidden_dim, ffn_dim, rngs=rngs)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: Array) -> Array:
		return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class PhiMoEAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: PhiMoeConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings
		self.original_max_position_embeddings = config.rope_scaling.get(
			"original_max_position_embeddings", None
		)
		self.rope_theta = config.rope_theta
		self.rope_scaling = config.rope_scaling
		self.is_causal = True

		if (self.head_dim * self.num_heads) != self.hidden_size:
			raise ValueError(
				f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
				f" and `num_heads`: {self.num_heads})."
			)

		linear_class = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.attention_bias,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear_class(
			config.num_attention_heads * self.head_dim,
			config.hidden_size,
			rngs=rngs,
		)

		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_key_value_heads,
			attention_dropout=self.config.attention_dropout,
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			base_config=self.config,
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=config.hidden_size // config.num_attention_heads,
			rotary_dim=config.hidden_size // config.num_attention_heads,
			is_neox_style=True,
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
		(query_states, key_states, value_states) = (
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

		attn_output = self._merge_heads(attentions.attention_outputs)
		if self.config.shard_attention_computation:
			attn_output = with_sharding_constraint(
				attn_output,
				PartitionSpec(
					self.config.partition_axis.batch_axis,
					(
						self.config.partition_axis.sequence_axis
						if attn_output.shape[1] != 1
						else None
					),
					self.config.partition_axis.hidden_state_axis,
				),
			)
		attn_output = self.o_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class PhiMoeSparseMoeBlock(nn.Module):
	def __init__(
		self,
		config: PhiMoeConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.hidden_dim = config.hidden_size
		self.ffn_dim = config.intermediate_size
		self.num_experts = config.num_local_experts
		self.top_k = config.num_experts_per_tok
		self.router_jitter_noise = config.router_jitter_noise
		self.input_jitter_noise = config.input_jitter_noise
		self.gate = nn.Linear(
			self.config.hidden_size,
			self.config.num_local_experts,
			use_bias=False,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
		)

		self.experts = [
			PhiMoEBlockSparseTop2MLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_local_experts)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		deterministic: bool = False,
	) -> tp.Tuple[chex.Array, chex.Array]:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

		router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
			jnp.promote_types(self.dtype, jnp.float32)
		)
		routing_weights, selected_experts = jax.lax.top_k(
			router_logits, k=self.config.num_experts_per_tok
		)
		routing_weights = jax.nn.softmax(
			routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
		)
		if not deterministic and self.input_jitter_noise > 0:
			final_hidden_state = jax.nn.initializers.uniform(
				1.0 - self.input_jitter_noise,
				1.0 + self.input_jitter_noise,
			)(self.make_rng(), hidden_states.shape, hidden_states.dtype)
		else:
			final_hidden_state = jnp.zeros_like(hidden_states)
		for index in range(self.config.num_local_experts):
			expert_layer_output = (
				block_wise_ffn(
					self.experts[index],
					hidden_states,
					self.config.scan_mlp_chunk_size,
					deterministic,
				)
				if self.config.use_scan_mlp
				else self.experts[index](hidden_states)
			)
			expert_layer_output_exp = (
				jnp.sum(jnp.multiply(selected_experts == index, routing_weights), axis=-1)[
					:, :, None
				]
				* expert_layer_output
			)
			final_hidden_state += expert_layer_output_exp
		return (
			final_hidden_state,
			router_logits,
		)


class FlaxPhiMoeDecoderLayer(nn.Module):
	def __init__(
		self,
		config: PhiMoeConfig,
		layer_idx: int,
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
		attn_block = PhiMoEAttention
		mlp_block = PhiMoeSparseMoeBlock
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.block_sparse_moe = mlp_block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.input_layernorm = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=True,
			rngs=rngs,
		)

		self.post_attention_layernorm = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=True,
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
	):
		residual = hidden_states

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
			(attn_out[0], attn_out[1]) if len(attn_out) == 2 else (attn_out[0], None)
		)

		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states, router_logits = self.block_sparse_moe(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if output_router_logits:
			outputs += (router_logits,)
		return outputs


@register_module(
	"base-module",
	config=PhiMoeConfig,
	model_type="phimoe",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["norm", "input_layernorm", "post_attention_layernorm"],
)
class PhiMoeModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: PhiMoeConfig,
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
		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.embed_dropout = nn.Dropout(config.embd_pdrop)
		self.layers = [
			FlaxPhiMoeDecoderLayer(
				config=config,
				layer_idx=idx,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for idx in range(self.config.num_hidden_layers)
		]
		self.norm = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=True,
			rngs=rngs,
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
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

		batch_size, sequence_length, _ = inputs_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))

		hidden_states = inputs_embeds
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_view=past_key_values.views[idx],
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (hidden_states, all_hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	"causal-language-model",
	config=PhiMoeConfig,
	model_type="phimoe",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["norm", "input_layernorm", "post_attention_layernorm"],
)
class PhiMoeForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: PhiMoeConfig,
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
		self.model = PhiMoeModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.vocab_size = self.config.vocab_size
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			use_bias=config.lm_head_bias,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
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
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		"""
		Forward pass through the PhiMoe module.

		Args:
		    input_ids (tp.Optional[chex.Array]): Input tensor containing token IDs.
		    attention_mask (tp.Optional[chex.Array]): Mask for attention.
		    position_ids (tp.Optional[chex.Array]): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxCausalLMOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""

		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=True,
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
		if not return_dict:
			return (lm_logits,) + outputs[0:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
