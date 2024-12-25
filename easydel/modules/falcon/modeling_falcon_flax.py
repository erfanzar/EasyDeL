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
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.falcon.falcon_configuration import FalconConfig as FalconConfig


def built_bloom_alibi(attention_mask, num_attention_heads):
	"""The built_bloom_alibi function is used to create a bloom alibi for the attention mask.
	The bloom alibi is used in the Bloom Attention layer to ensure that each token has a unique
	attention vector, even if it's masked out. This ensures that all tokens have an equal chance of being selected as
	the most important token in the sequence, which helps with training stability and performance.

	Args:
	    attention_mask: Mask out the padding tokens in the input
	        sequence
	    num_attention_heads: Determine the number of attention heads in
	        the model

	Returns:
	    A tensor of shape (batch_size, num_attention_heads, 1,
	    sequence_length)
	"""
	batch_size, sequence_length = attention_mask.shape
	cp2 = 2 ** math.floor(math.log2(num_attention_heads))
	base = jnp.asarray(2 ** (-(2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32)
	powers = jnp.arange(1, 1 + cp2, dtype=jnp.float32)
	slops = jnp.power(base, powers)
	if cp2 != num_attention_heads:
		extra_base = jnp.asarray(
			2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32
		)
		num_rem_heads = min(cp2, num_attention_heads - cp2)
		extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
		slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
	arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[
		:, jnp.newaxis, :
	]
	alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
	return alibi.reshape(batch_size, num_attention_heads, 1, sequence_length)


def dropout_add(
	nn_drop: nn.Dropout,
	x: chex.Array,
	residual: chex.Array,
) -> chex.Array:
	"""The dropout_add function is a helper function that adds the residual to the output of
	the dropout layer. This is necessary because we want to use deterministic=True when
	we are evaluating our model, but we still need to add in the residual. The reason for this
	is that during training, we have two paths through our network: one with dropout and one without.
	The path without dropout (residual) allows us to backpropagate gradients through both paths at once.

	Args:
	    nn_drop: nn.Dropout: Specify the dropout layer
	    x: chex.Array: Pass in the input to the dropout layer
	    residual: chex.Array: Add the residual to the output of
	        dropout_add
	    deterministic: bool: Determine whether the dropout layer is
	        active or not

	Returns:
	    A tensor that is the sum of the residual and a dropout layer
	"""
	out = nn_drop(inputs=x)
	out = residual + out
	return out


class FalconAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: FalconConfig,
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
		head_dim = config.hidden_size // config.num_attention_heads
		if config.new_decoder_architecture:
			qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * head_dim
		elif config.multi_query:
			qkv_out_dim = config.hidden_size + 2 * head_dim
		else:
			qkv_out_dim = 3 * config.hidden_size

		self.head_dim = head_dim
		assert self.head_dim * config.num_attention_heads == config.hidden_size
		self.num_kv_heads = (
			config.num_kv_heads
			if (config.new_decoder_architecture or not config.multi_query)
			else 1
		)
		self.new_decoder_architecture = config.new_decoder_architecture
		self.num_heads = config.num_attention_heads
		self.query_key_value = nn.Linear(
			config.hidden_size,
			qkv_out_dim,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=config.bias,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.inv_norm_factor = 1 / math.sqrt(head_dim)
		self.dense = nn.Linear(
			qkv_out_dim,
			config.hidden_size,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.bias,
			precision=self.precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		if not self.config.alibi:
			self.rotary = self.config.get_basic_rope(
				rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
				head_size=self.config.hidden_size // self.config.num_attention_heads,
				base=self.config.rope_theta,
				is_neox_style=True,
				dtype=self.dtype,
			)
		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=0.0,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=config.attn_mechanism,
			dtype=self.config.attn_dtype,
			mesh=config.mesh,
			sm_scale=self.inv_norm_factor,
			axis_name=config.attention_axis_name,
			base_config=config,
			_do_check=False,
		)

	def _split_heads(
		self, qkv: chex.Array
	) -> tp.Tuple[chex.Array, chex.Array, chex.Array]:
		"""
		Splits the query, key, and value tensors into separate heads.

		Args:
		    qkv (chex.Array): Combined query, key, and value tensor.

		Returns:
		    tp.Tuple[chex.Array, chex.Array, chex.Array]: A tuple containing the query, key, and value tensors split into heads.
		"""
		batch_size, sequence_length, _ = qkv.shape

		if self.config.new_decoder_architecture:
			qkv = qkv.reshape(
				batch_size,
				sequence_length,
				-1,
				self.num_heads // self.num_kv_heads + 2,
				self.head_dim,
			)
			query_states = qkv[:, :, :, :-2]
			key_states = qkv[:, :, :, [-2]]
			value_states = qkv[:, :, :, [-1]]
			key_states = jnp.broadcast_to(key_states, query_states.shape)
			value_states = jnp.broadcast_to(value_states, query_states.shape)

			query_states, key_states, value_states = [
				x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
				for x in (query_states, key_states, value_states)
			]

			return query_states, key_states, value_states
		if self.config.multi_query:
			qkv = qkv.reshape(
				batch_size, sequence_length, self.config.num_attention_heads + 2, -1
			)
			query_states, key_states, value_states = (
				qkv[..., :-2, :],
				qkv[..., [-2], :],
				qkv[..., [-1], :],
			)

		else:
			query_states, key_states, value_states = jnp.split(qkv, 3, -1)
		return query_states, key_states, value_states

	def _merge_heads(self, x: chex.Array) -> chex.Array:
		"""
		Merges the attention heads into a single tensor.

		Args:
		    x (chex.Array): Tensor with separate attention heads.

		Returns:
		    chex.Array: Tensor with merged attention heads.
		"""
		batch_size_and_num_heads, seq_length, _ = x.shape
		batch_size = batch_size_and_num_heads // self.num_heads
		x = x.reshape(
			batch_size, self.config.num_attention_heads, seq_length, self.head_dim
		)
		return x.reshape(
			batch_size, seq_length, self.config.num_attention_heads * self.head_dim
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array = None,
		segment_ids: tp.Optional[chex.Array] = None,
		alibi: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		frequencies: tp.Optional[chex.Array] = None,
	):
		fused_qkv = self.query_key_value(hidden_states)
		num_kv_heads = (
			self.num_heads if self.new_decoder_architecture else self.num_kv_heads
		)
		# 3 x [batch_size, seq_length, num_heads, head_dim]
		(query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
		batch_size, query_length, _, _ = query_layer.shape
		key_length = query_length
		query_layer = query_layer.reshape(
			batch_size,
			query_length,
			self.num_heads,
			self.head_dim,
		)
		key_layer = key_layer.reshape(
			batch_size,
			query_length,
			num_kv_heads,
			self.head_dim,
		)
		value_layer = value_layer.reshape(
			batch_size,
			query_length,
			num_kv_heads,
			self.head_dim,
		)
		if alibi is None:
			query_layer, key_layer = self.rotary(
				positions=position_ids,
				query=query_layer,
				key=key_layer,
				frequencies=frequencies,
			)
		(
			key_layer,
			value_layer,
			attention_mask,
			attention_bias,
		) = self.concatenate(
			query=query_layer,
			key=key_layer,
			value=value_layer,
			cache_view=cache_view,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=None,
		)

		if alibi is None:
			attention = self.attention_performer(
				query_states=query_layer,
				key_states=key_layer,
				value_states=value_layer,
				causal_mask=causal_mask,
				attention_mask=attention_mask,
				dropout_rng=self.rngs.params(),
				segment_ids=segment_ids,
				query_sequence_length=query_length,
				key_value_sequence_length=key_length,
				uses_cache=cache_view is not None,
				bias=attention_bias,
				causal=True,
			)
			attention_outputs = attention.attention_outputs
			attention_outputs = attention_outputs.reshape(
				batch_size, query_length, self.num_heads * self.head_dim
			)
			output_tensor = self.dense(attention_outputs)
			return output_tensor, attention.attention_weights

		else:
			attention_scores = jnp.einsum(
				"...qhd,...khd->...hqk",
				query_layer,
				key_layer,
				precision=self.precision,
			)
			attention_scores = attention_scores.reshape(
				batch_size, self.num_heads, query_length, key_length
			)
			attention_scores = attention_scores + alibi.reshape(
				batch_size, self.num_heads, 1, -1
			)
			attention_scores *= self.inv_norm_factor
			attention_scores = jax.nn.softmax(attention_scores + attention_bias, axis=-1)
			attention_scores = attention_scores.reshape(
				batch_size, self.num_heads, query_length, key_length
			)
			# matmul: [batch_size * num_heads, q_length, head_dim]
			attn_output = jax.lax.batch_matmul(
				attention_scores, value_layer.transpose(0, 2, 1, 3)
			)  # noqa
			attn_output = attn_output.reshape(
				(attn_output.shape[1] * attn_output.shape[0],) + attn_output.shape[2:]
			)
			attn_output = self.shard_attention_prod(self._merge_heads(attn_output))

			output_tensor = self.dense(attn_output)

			if output_attentions:
				return output_tensor, attention_scores
			return output_tensor, None


class FalconMlp(nn.Module):
	def __init__(
		self,
		config: FalconConfig,
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
		linear = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=self.config.bias,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.dense_h_to_4h = linear(
			self.config.hidden_size,
			self.config.ff_factor * self.config.hidden_size,
			rngs=rngs,
		)
		self.dense_4h_to_h = linear(
			self.config.ff_factor * self.config.hidden_size,
			self.config.hidden_size,
			rngs=rngs,
		)

	def __call__(self, x: chex.Array, deterministic: bool = True):
		x = control_mlp_sharding(x, self.config.partition_axis)
		return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x), approximate=False))


class FalconBlock(nn.Module):
	def __init__(
		self,
		config: FalconConfig,
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

		if config.new_decoder_architecture and config.num_ln_in_parallel_attn == 2:
			self.ln_attn = nn.LayerNorm(
				self.config.hidden_size,
				epsilon=config.layer_norm_epsilon,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				rngs=rngs,
			)
			self.ln_mlp = nn.LayerNorm(
				self.config.hidden_size,
				epsilon=config.layer_norm_epsilon,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				rngs=rngs,
			)
		else:
			self.input_layernorm = nn.LayerNorm(
				self.config.hidden_size,
				epsilon=config.layer_norm_epsilon,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				rngs=rngs,
			)
			if not config.parallel_attn:
				self.post_attention_layernorm = nn.LayerNorm(
					self.config.hidden_size,
					epsilon=config.layer_norm_epsilon,
					dtype=self.dtype,
					param_dtype=self.param_dtype,
					rngs=rngs,
				)
		attn_block = FalconAttention
		mlp_block = FalconMlp
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)

		self.mlp = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.self_attention = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.dropout = nn.Dropout(self.config.attention_dropout)
		self.dropout_mlp = nn.Dropout(self.config.hidden_dropout)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array = None,
		segment_ids: tp.Optional[chex.Array] = None,
		alibi: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		frequencies: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the FalconBlock module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array], optional): Segment IDs for segment-based attention.
		    alibi (tp.Optional[chex.Array], optional): Alibi tensor for adding positional bias.
		    init_cache (bool, optional): If True, initializes cache for caching keys and values.
		    output_attentions (bool, optional): If True, outputs attention weights alongside the hidden states.
		    deterministic (bool, optional): If True, disables dropout for deterministic behavior.

		Returns:
		    tp.Union[chex.Array, tp.Tuple[chex.Array, chex.Array]]: The output tensor and optionally the attention weights.
		"""
		residual = hidden_states

		if self.config.num_ln_in_parallel_attn == 2:
			attention_layernorm_out = self.ln_attn(hidden_states)
			mlp_layernorm_out = self.ln_mlp(hidden_states)
		else:
			attention_layernorm_out = self.input_layernorm(hidden_states)

		attention_output, attn_score = self.self_attention(
			attention_layernorm_out,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			alibi,
			cache_view,
			output_attentions,
			frequencies,
		)

		if self.config.num_ln_in_parallel_attn == 1:
			if self.config.parallel_attn:
				mlp_layernorm_out = attention_layernorm_out
			else:
				residual = dropout_add(self.dropout, attention_output, residual)
				mlp_layernorm_out = self.post_attention_layernorm(residual)

		if self.config.use_scan_mlp:
			mlp_output = block_wise_ffn(
				self.mlp,
				mlp_layernorm_out,
				self.config.scan_mlp_chunk_size,
			)
		else:
			mlp_output = self.mlp(mlp_layernorm_out)

		if self.config.new_decoder_architecture or self.config.parallel_attn:
			mlp_output += attention_output

		output = dropout_add(self.dropout_mlp, mlp_output, residual)
		return output, attn_score


@register_module(
	"base-module",
	config=FalconConfig,
	model_type="falcon",
	embedding_layer_names=["word_embeddings"],
	layernorm_names=[
		"input_layernorm",
		"ln_f",
		"ln_attn",
		"ln_mlp",
		"post_attention_layernorm",
	],
)
class FalconModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: FalconConfig,
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
		self.word_embeddings = nn.Embed(
			num_embeddings=config.vocab_size,
			features=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.h = [
			FalconBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.ln_f = nn.LayerNorm(
			self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			epsilon=config.layer_norm_epsilon,
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
		"""
		Forward pass through the Falcon module.

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
		all_hidden_states = () if output_hidden_states else None
		all_attentions = () if output_attentions else None
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids.astype("i4"))

		batch_size, sequence_length, _ = inputs_embeds.shape

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")
		alibi = None
		if self.config.alibi:
			alibi = built_bloom_alibi(
				attention_mask,
				self.config.num_attention_heads,
			).astype(inputs_embeds.dtype)
		elif position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (-3, -2))
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.h))
		hidden_states = inputs_embeds
		for idx, layer in enumerate(self.h):
			hidden_states, score = layer(
				hidden_states=hidden_states,
				alibi=alibi,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				cache_view=past_key_values.views[idx],
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
			)
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			if output_attentions:
				all_attentions += (score,)
		hidden_states = self.ln_f(hidden_states)
		if all_hidden_states is not None:
			all_hidden_states += hidden_states
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
	config=FalconConfig,
	model_type="falcon",
	embedding_layer_names=["word_embeddings"],
	layernorm_names=[
		"input_layernorm",
		"ln_f",
		"ln_attn",
		"ln_mlp",
		"post_attention_layernorm",
	],
)
class FalconForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: FalconConfig,
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
		self.transformer = FalconModel(
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
			rngs=rngs,
			use_bias=False,
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
		Forward pass through the Falcon module.

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
		outputs = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			output_hidden_states=output_hidden_states,
			segment_ids=segment_ids,
		)
		if return_dict:
			hidden_state = outputs.last_hidden_state
		else:
			hidden_state = outputs[0]

		logits = self.lm_head(hidden_state)
		if not return_dict:
			return (logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
