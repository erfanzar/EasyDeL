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
from flax import nnx as nn
from jax import numpy as jnp

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
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm
from easydel.modules.openelm.openelm_configuration import (
	OpenELMConfig as OpenELMConfig,
)
from easydel.modules.openelm.openelm_configuration import (
	make_divisible,
)


class OpenELMMultiHeadCausalAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: OpenELMConfig,
		layer_idx: int,
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
		self.layer_idx = layer_idx
		head_dim = config.head_dim
		q_heads = config.num_query_heads[layer_idx]
		k_heads = config.num_kv_heads[layer_idx]
		v_heads = config.num_kv_heads[layer_idx]

		self.qkv_proj = nn.Linear(
			config.model_dim,
			(q_heads + k_heads + v_heads) * head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		if config.normalize_qk_projections:
			self.q_norm = RMSNorm(
				dim=config.head_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				eps=1e-6,
				rngs=rngs,
			)
			self.k_norm = RMSNorm(
				dim=config.head_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				eps=1e-6,
				rngs=rngs,
			)
		else:
			self.q_norm = None
			self.k_norm = None

		self.out_proj = nn.Linear(
			q_heads * head_dim,
			config.model_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			precision=precision,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.head_dim = head_dim

		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=q_heads,
			num_kv_heads=k_heads,
			attention_dropout=0.0,
			head_dims=head_dim,
			shard_attention_computation=self.config.shard_attention_computation,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			base_config=self.config,
		)

		self.head_dim = config.head_dim
		self.num_q_heads = q_heads
		self.num_k_heads = k_heads
		self.num_v_heads = v_heads
		self.transformer_dim = config.model_dim
		self.num_groups = self.num_q_heads // self.num_k_heads

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=self.config.head_dim,
			rotary_dim=self.config.head_dim,
			base=self.config.rope_freq_constant,
		)

	def _merge_heads(self, hidden_states):
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (chex.Array): The hidden states with separate head dimensions.

		Returns:
		    chex.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.num_q_heads * self.head_dim,)
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
		output_attentions = False

		# [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
		qkv = self.qkv_proj(hidden_states)
		# [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
		qkv = qkv.reshape(
			batch_size,
			sequence_length,
			self.num_q_heads + self.num_k_heads + self.num_v_heads,
			self.head_dim,
		)
		# [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
		qkv = qkv.transpose(0, 2, 1, 3)
		# [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
		query_states = qkv[
			:,
			: self.num_q_heads,
			:,
			:,
		]
		key_states = qkv[
			:,
			self.num_q_heads : self.num_k_heads + self.num_q_heads,
			:,
			:,
		]
		value_states = qkv[
			:,
			self.num_k_heads + self.num_q_heads :,
			:,
			:,
		]
		if self.q_norm is not None:
			query_states = self.q_norm(query_states)

		if self.k_norm is not None:
			key_states = self.k_norm(key_states)

		query_states, key_states, value_states = map(
			lambda x: x.transpose(0, 2, 1, 3),
			[query_states, key_states, value_states],
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

		attn_output = self.out_proj(
			self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
		)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class OpenELMFeedForwardNetwork(nn.Module):
	def __init__(
		self,
		config: OpenELMConfig,
		layer_idx: int,
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
		self.layer_idx = layer_idx
		ffn_multiplier = config.ffn_multipliers[layer_idx]
		intermediate_dim = int(
			make_divisible(
				ffn_multiplier * config.model_dim,  # type:ignore
				divisor=config.ffn_dim_divisor,
			)
		)
		if config.ffn_with_glu:
			# FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
			self.proj_1 = nn.Linear(
				config.model_dim,
				2 * intermediate_dim,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
				kernel_init=jax.nn.initializers.normal(config.initializer_range),
				**get_dot_general_by_bits(config.bits, config.easy_method),
			)
			self.proj_2 = nn.Linear(
				intermediate_dim,
				config.model_dim,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
				kernel_init=jax.nn.initializers.normal(config.initializer_range),
				**get_dot_general_by_bits(config.bits, config.easy_method),
			)
			self.ffn_with_glu = True
		else:
			self.proj_1 = nn.Linear(
				config.model_dim,
				intermediate_dim,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
				kernel_init=jax.nn.initializers.normal(config.initializer_range),
				**get_dot_general_by_bits(config.bits, config.easy_method),
			)
			self.proj_2 = nn.Linear(
				intermediate_dim,
				config.model_dim,
				use_bias=False,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
				kernel_init=jax.nn.initializers.normal(config.initializer_range),
				**get_dot_general_by_bits(config.bits, config.easy_method),
			)
			self.ffn_with_glu = False

		self.act = ACT2FN[config.activation_fn_name]

	def __call__(self, hidden_states: chex.Array) -> chex.Array:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

		if self.ffn_with_glu:
			y_12 = self.proj_1(hidden_states)
			y_1, y_2 = jnp.split(y_12, 2, axis=-1)
			return self.proj_2(self.act(y_1) * y_2)
		else:
			return self.proj_2(self.act(self.proj_1(hidden_states)))


class OpenELMDecoderLayer(nn.Module):
	def __init__(
		self,
		config: OpenELMConfig,
		layer_idx: int,
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
		self.layer_idx = layer_idx
		attn_block = OpenELMMultiHeadCausalAttention
		mlp_block = OpenELMFeedForwardNetwork
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)

		self.attn = attn_block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.ffn = mlp_block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.ffn_norm = RMSNorm(
			self.config.model_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			eps=1e-6,
			rngs=rngs,
		)
		self.attn_norm = RMSNorm(
			self.config.model_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			eps=1e-6,
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
	):
		"""
		Forward pass of the module block.

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
		residual = hidden_states
		hidden_states = self.attn_norm(hidden_states)

		hidden_states, self_attn_weights = self.attn(
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
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.ffn_norm(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.ffn,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.ffn(hidden_states)
		hidden_states = residual + feed_forward_hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		return outputs  # type:ignore


@register_module(
	"base-module",
	config=OpenELMConfig,
	model_type="openelm",
	embedding_layer_names=["token_embeddings"],
)
class OpenELMModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: OpenELMConfig,
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
		self.token_embeddings = nn.Embed(
			config.vocab_size,
			config.model_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			OpenELMDecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				layer_idx=i,
				rngs=rngs,
			)
			for i in range(self.config.num_transformer_layers)
		]
		self.norm = RMSNorm(
			config.model_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			eps=1e-6,
			rngs=rngs,
		)
		if config.share_input_output_layers:
			self.classifier = None
		else:
			self.classifier = nn.Linear(
				config.model_dim,
				config.vocab_size,
				use_bias=False,
				rngs=rngs,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
			)
		self.num_transformer_layers = config.num_transformer_layers

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies(
			head_size=self.config.head_dim,
			rotary_dim=self.config.head_dim,
			base=self.config.rope_freq_constant,
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
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		if inputs_embeds is None and input_ids is not None:
			inputs_embeds = self.token_embeddings(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify inputs_embeds or input_ids one of them")
		batch_size, sequence_length, _ = inputs_embeds.shape

		assert (
			sequence_length <= self.config.max_context_length
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_context_length} got {sequence_length})"
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

		for idx, layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			output = layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				cache_view=past_key_values.views[idx],
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				frequencies=self.frequencies,
			)
			hidden_states = output[0]

			if output_attentions:
				output_attentions += (output[1],)

		hidden_states = self.norm(hidden_states)

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
	config=OpenELMConfig,
	model_type="openelm",
	embedding_layer_names=["token_embeddings"],
)
class OpenELMForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: OpenELMConfig,
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
		self.transformer = OpenELMModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.lm_head = nn.Linear(
			config.model_dim,
			config.vocab_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
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
		outputs = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			past_key_values=past_key_values,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]
		if self.config.share_input_output_layers:
			self.lm_head.kernel.value = self.transformer.token_embeddings.embedding.value.T
			lm_logits = self.lm_head(hidden_states)
		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
