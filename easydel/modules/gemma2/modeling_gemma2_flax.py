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


import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from fjformer import with_sharding_constraint
from flax import nnx as nn
from jax.sharding import PartitionSpec

from easydel.etils.etils import get_logger
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
from easydel.modules.gemma2.gemma2_configuration import Gemma2Config as Gemma2Config

logger = get_logger(__name__)


class Gemma2RMSNorm(nn.Module):
	def __init__(self, config: Gemma2Config, dtype: jnp.dtype = jnp.float32):
		self.config = config
		self.epsilon = self.config.rms_norm_eps
		self.dtype = dtype
		self.kernel = nn.Param(jnp.ones(self.config.hidden_size, dtype=dtype))

	def __call__(self, hidden_states):
		variance = hidden_states.astype(jnp.float32)
		variance = jnp.power(variance, 2)
		variance = variance.mean(-1, keepdims=True)
		hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

		return (1 + self.kernel.value.astype(self.dtype)) * jnp.asarray(
			hidden_states, dtype=self.dtype
		)


class Gemma2Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: Gemma2Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		causal: bool = True,
		is_cross_attention: bool = False,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config)
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.is_cross_attention = is_cross_attention
		self.rngs = rngs
		self.causal = causal
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = config.head_dim
		self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads

		kernel = jax.nn.initializers.normal(config.initializer_range)

		linear = partial(
			nn.Linear,
			use_bias=config.attention_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=kernel,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.q_proj = linear(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear(
			self.embed_dim,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear(
			self.embed_dim,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear(
			self.num_heads * self.head_dim,
			self.embed_dim,
			rngs=rngs,
		)
		self.sliding_window = config.sliding_window if (self.layer_idx % 2 == 0) else None
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
			sm_scale=self.config.query_pre_attn_scalar**-0.5,
			base_config=self.config,
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			self.head_dim,
			self.head_dim,
			True,
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
			hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
		)

	def _split_heads(self, hidden_states, num_heads):
		return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

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
		(query_states, key_states, value_states) = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)

		query_states = query_states.reshape(
			batch_size,
			sequence_length,
			self.num_heads,
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
			sliding_windows=self.sliding_window if bool((self.layer_idx % 2) == 0) else None,
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

		return (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)


class Gemma2MLP(nn.Module):
	def __init__(
		self,
		config: Gemma2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		embed_dim = self.config.hidden_size
		inner_dim = (
			self.config.intermediate_size
			if self.config.intermediate_size is not None
			else 4 * embed_dim
		)
		kernel_init = jax.nn.initializers.normal(config.initializer_range)

		self.act = ACT2FN[self.config.hidden_activation]

		linear_class = partial(
			nn.Linear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=kernel_init,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(
			embed_dim,
			inner_dim,
			rngs=rngs,
		)
		self.down_proj = linear_class(
			inner_dim,
			embed_dim,
			rngs=rngs,
		)
		self.up_proj = linear_class(
			embed_dim,
			inner_dim,
			rngs=rngs,
		)

	def __call__(self, hidden_states):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		return hidden_states


class Gemma2DecoderLayer(nn.Module):
	def __init__(
		self,
		config: Gemma2Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		mlp_block = Gemma2MLP
		attn_block = Gemma2Attention

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.is_sliding = bool(self.layer_idx % 2)
		self.self_attn = attn_block(
			self.config,
			layer_idx=self.layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.mlp = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.input_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
		self.post_attention_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
		self.pre_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
		self.post_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)

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

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states, attn_weight = self.self_attn(
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

		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.pre_feedforward_layernorm(hidden_states)
		if self.config.use_scan_mlp:
			hidden_states = block_wise_ffn(
				self.mlp,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			hidden_states = self.mlp(hidden_states)

		hidden_states = self.post_feedforward_layernorm(hidden_states)
		hidden_states = residual + hidden_states
		return hidden_states, attn_weight


@register_module(
	"base-module",
	config=Gemma2Config,
	model_type="gemma2",
	embedding_layer_names=["embed_tokens"],
)
class Gemma2Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: Gemma2Config,
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
		self.hidden_size = self.config.hidden_size
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.layers = [
			Gemma2DecoderLayer(
				self.config,
				layer_idx=i,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.norm = Gemma2RMSNorm(self.config, dtype=self.dtype)

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
		"""
		Forward pass through the Gemma2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""
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
			)
		inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))
		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				cache_view=past_key_values.views[idx],
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
	config=Gemma2Config,
	model_type="gemma2",
	embedding_layer_names=["embed_tokens"],
)
class Gemma2ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: Gemma2Config,
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
		self.model = Gemma2Model(
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
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
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
		"""
		Forward pass through the Gemma2 module.

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
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]
		hidden_states = with_sharding_constraint(
			hidden_states,
			PartitionSpec(
				self.config.partition_axis.batch_axis,
				self.config.partition_axis.sequence_axis,
				None,
			),
		)
		if self.config.tie_word_embeddings:
			# self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
			# lm_logits = self.lm_head(hidden_states)
			lm_logits = hidden_states @ self.model.embed_tokens.embedding.value.T
		else:
			lm_logits = self.lm_head(hidden_states)

		if self.config.final_logit_softcapping is not None:
			cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
			lm_logits = cap * jax.nn.tanh(lm_logits / cap)
			# lm_logits = jnp.where(
			# 	jnp.abs(lm_logits) > cap,
			# 	cap * jnp.sign(lm_logits),
			# 	lm_logits,
			# )

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
