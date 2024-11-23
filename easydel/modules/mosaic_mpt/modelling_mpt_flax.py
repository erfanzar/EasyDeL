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
from typing import Optional, Tuple, Union

import chex
import flax
import jax
from einops import rearrange
from flax import linen as nn
from flax.core import FrozenDict
from flax.linen import Dense, combine_masks
from flax.linen.partitioning import remat
from jax import lax
from jax import numpy as jnp

from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import EasyDeLBaseModule
from easydel.modules.mosaic_mpt.mosaic_configuration import (
	MptConfig as MptConfig,
)


class FlaxMptMLP(nn.Module):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		dense_class = partial(
			Dense,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			use_bias=self.config.use_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.up_proj = dense_class(self.config.expansion_ratio * self.config.hidden_size)
		self.down_proj = dense_class(self.config.hidden_size)
		self.hidden_dropout = nn.Dropout(self.config.attn_config.attn_pdrop)

	def __call__(
		self,
		hidden_states: chex.Array,
		residual: chex.Array,
		deterministic: bool = False,
	):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			jax.nn.gelu(self.up_proj(hidden_states), approximate=False)
		)
		return self.hidden_dropout(hidden_states, deterministic=deterministic) + residual


class FlaxMptAttention(FlaxAttentionModule):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.Wqkv = Dense(
			self.config.hidden_size * 3,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			use_bias=self.config.use_bias,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.out_proj = Dense(
			self.config.hidden_size,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			use_bias=self.config.use_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.dropout = nn.Dropout(self.config.attn_config.attn_pdrop)

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
		position_bias: chex.Array | Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    position_bias (Union[chex.Array, Tuple[chex.Array, chex.Array]]): Add a bias to the attention scores.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""

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
		query_length, key_length = query_states.shape[1], key_states.shape[1]

		if self.has_variable("cache", "cached_key"):
			mask_shift = self.variables["cache"]["cache_index"]
			max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
			causal_mask = lax.dynamic_slice(
				causal_mask,
				(0, 0, mask_shift, 0),
				(1, 1, query_length, max_decoder_length),
			)
		else:
			causal_mask = causal_mask[:, :, :query_length, :key_length]

		batch_size = hidden_states.shape[0]
		causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
		attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
		attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

		if self.has_variable("cache", "cached_key") or init_cache:
			key_states, value_states, attention_mask = self._concatenate_to_cache(
				key_states,
				value_states,
				query_states,
				attention_mask,
			)
		if position_bias is not None:
			key_length = key_states.shape[1]

			position_bias_query_index = max(0, position_bias.shape[2] - query_length)
			position_bias_key_index = max(0, position_bias.shape[3] - key_length)

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
			deterministic=deterministic,
			segment_ids=segment_ids,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=self.has_variable("cache", "cached_key") or init_cache,
			bias=attention_bias,
			causal=False,
		)

		attn_output = self.out_proj(attention.attention_outputs.reshape(inp_shape))

		return (
			(attn_output, attention.attention_weights)
			if output_attentions
			else (attn_output,)
		)


class FlaxMptBlock(nn.Module):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		attn_block = FlaxMptAttention
		mlp_block = FlaxMptMLP
		if self.config.gradient_checkpointing != "":
			mlp_block = remat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(2,),
			)
			attn_block = remat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(2, 3, 5, 6, 7, 8),
			)

		self.norm_1 = nn.LayerNorm(
			epsilon=self.config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=self.config.use_norm_bias,
		)
		self.attn = attn_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.norm_2 = nn.LayerNorm(
			epsilon=self.config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=self.config.use_norm_bias,
		)
		self.ffn = mlp_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.dropout_rate = self.config.attn_config.attn_pdrop
		self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

	def __call__(
		self,
		hidden_states: chex.Array,
		position_bias: chex.Array | Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    position_bias (Union[chex.Array, Tuple[chex.Array, chex.Array]]): Add a bias to the attention scores.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""

		attn_out = self.attn(
			self.norm_1(hidden_states),
			position_bias,
			attention_mask,
			causal_mask,
			segment_ids,
			deterministic,
			init_cache,
			output_attentions,
			fcm_mask,
		)
		attn_outputs, attn_weights = attn_out if output_attentions else (attn_out[0], None)
		hidden_states = (
			self.resid_attn_dropout(attn_outputs, deterministic=deterministic) + hidden_states
		)
		output = self.ffn(self.norm_2(hidden_states), hidden_states)
		outputs = (output,)
		if output_attentions:
			outputs += (attn_weights,)

		return outputs


class FlaxMptDecoratorCollection(nn.Module):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		block = FlaxMptBlock
		self.blocks = [
			block(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.n_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		position_bias: chex.Array | Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    position_bias (Union[chex.Array, Tuple[chex.Array, chex.Array]]): Add a bias to the attention scores.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    output_hidden_states (bool): If True, output hidden states.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		all_hidden_states = () if output_hidden_states else None
		all_attentions = () if output_attentions else None
		if not deterministic and self.config.fcm_max_ratio > 0:
			# Apply forgetful causal mask
			batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
			fcm_ratio = jax.random.uniform(
				self.make_rng("fcm"),
				shape=(batch_size, 1, 1, 1),
				minval=self.config.fcm_min_ratio,
				maxval=self.config.fcm_max_ratio,
			)
			fcm_mask = (
				jax.random.uniform(
					self.make_rng("fcm"),
					shape=(batch_size, 1, seq_length, seq_length),
				)
				> fcm_ratio
			)
			fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
			fcm_mask = fcm_mask.astype("bool")
		else:
			fcm_mask = None
		for block in self.blocks:
			output = block(
				hidden_states=hidden_states,
				deterministic=deterministic,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				output_attentions=output_attentions,
				init_cache=init_cache,
				position_bias=position_bias,
				fcm_mask=fcm_mask,
				segment_ids=segment_ids,
			)
			hidden_states = output[0]
			if output_attentions:
				all_attentions += (output[-1],)
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
		return hidden_states, all_hidden_states, all_attentions


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


class FlaxMptModule(nn.Module):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.wte = nn.Embed(
			num_embeddings=self.config.vocab_size,
			features=self.config.d_model,
		)

		self.blocks = FlaxMptDecoratorCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm_f = nn.LayerNorm(
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			epsilon=self.config.layer_norm_epsilon,
			use_bias=self.config.use_norm_bias,
		)
		self.alibi = build_mpt_alibi_tensor(
			sequence_length=self.config.max_seq_len,
			num_heads=self.config.n_heads,
		)
		self.causal_mask = jnp.tril(
			jnp.ones((self.config.max_seq_len, self.config.max_seq_len), dtype="bool")
		).reshape(1, 1, self.config.max_seq_len, self.config.max_seq_len)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		Forward pass through the MPT module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if input_embeds is None:
			input_embeds = self.wte(input_ids)

		hidden_states, all_hidden_states, all_attentions = self.blocks(
			position_bias=self.alibi,
			causal_mask=self.causal_mask,
			init_cache=init_cache,
			output_attentions=output_attentions,
			attention_mask=attention_mask,
			deterministic=deterministic,
			output_hidden_states=output_hidden_states,
			hidden_states=input_embeds,
			segment_ids=segment_ids,
		)
		hidden_states = self.norm_f(hidden_states)
		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		if return_dict:
			return FlaxBaseModelOutput(
				last_hidden_state=hidden_states,
				hidden_states=all_hidden_states,
				attentions=all_attentions,
			)

		return (hidden_states, all_hidden_states, all_attentions)


class FlaxMptPretrainedModel(EasyDeLBaseModule):
	module_class: nn.Module = None
	config_class: MptConfig = MptConfig

	def __init__(
		self,
		config: MptConfig,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: Optional[jax.lax.Precision] = None,
		input_shape: Tuple[int, int] = (1, 1),
		seed: int = 0,
		_do_init: bool = False,
		**kwargs,
	):
		"""
		Initializes the pre-trained model with the given configuration.

		Args:
		    config (MptConfig): Configuration for the model.
		    dtype (jnp.dtype): Data type for computations.
		    param_dtype (jnp.dtype): Data type for model parameters.
		    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
		    input_shape (Tuple[int, int]): Shape of the input tensor.
		    seed (int): Seed for random number generation.
		    _do_init (bool): If True, initialize model weights.
		    **kwargs: Additional keyword arguments.
		"""
		module = self.module_class(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			**kwargs,
		)
		super().__init__(
			dtype=dtype,
			_do_init=_do_init,
			module=module,
			config=config,
			input_shape=input_shape,
			seed=seed,
		)

	def init_cache(self, batch_size, max_length):
		def init_fn():
			input_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32)
			attention_mask = jnp.ones_like(input_ids)

			init_variables = self.module.init(
				jax.random.PRNGKey(0),
				input_ids=input_ids,
				attention_mask=attention_mask,
				return_dict=False,
				init_cache=True,
			)
			return init_variables["cache"]

		return jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), jax.eval_shape(init_fn))

	def init_weights(
		self,
		rng: jax.random.PRNGKey,
		input_shape: Tuple,
		params: Optional[FrozenDict] = None,
	) -> FrozenDict:
		input_ids = jnp.ones(input_shape, dtype="i4")
		if params is None:
			return self.module.init(
				rngs=rng,
				input_ids=input_ids,
				attention_mask=jnp.ones(input_shape, dtype="i4"),
				init_cache=False,
			)["params"]
		else:
			return params

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		params: dict = None,
		past_key_values: Optional[dict] = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		add_params_field: bool = False,
		**kwargs,
	):
		"""
		Forward pass through the model.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    input_embeds (Optional[chex.Array]): embedding inputs to be used instead of input_ids.
		    attention_mask (Optional[chex.Array]): Mask for attention.
		    segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
		    params (dict, optional): Parameters for the model.
		    past_key_values (dict, optional): Past key and value states for caching.
		    dropout_rng (jax.random.PRNGKey, optional): RNG key for dropout.
		    train (bool): If True, the model is in training mode.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    return_dict (Optional[bool]): If True, return a dictionary of outputs.
		    add_params_field (bool): If True, include the parameters in the input dictionary.
		    **kwargs: Additional arguments.

		Returns:
		    Output type depends on the model configuration.
		"""
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
		return_dict = return_dict if return_dict is not None else self.config.return_dict
		batch_size, sequence_length = (
			input_ids.shape if input_ids is not None else input_embeds.shape[:2]
		)

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length))

		rng_s = {}
		if dropout_rng is not None:
			rng_s["dropout"] = dropout_rng

		inputs = (
			{"params": params or self.params} if add_params_field else params or self.params
		)

		if self.config.bits is not None:
			rng_s["params"] = jax.random.key(0)
		if past_key_values is not None:
			inputs["cache"] = past_key_values
			mutable = ["cache"]
		else:
			mutable = False

		predict = self.module.apply(
			params,
			input_ids=input_ids,
			attention_mask=attention_mask,
			input_embeds=input_embeds,
			init_cache=False,
			deterministic=not train,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			segment_ids=segment_ids,
			mutable=mutable,
			rngs=rng_s,
		)
		if past_key_values is not None and return_dict:
			predict, past_key_values = predict
			predict["past_key_values"] = flax.core.unfreeze(past_key_values["cache"])
			return predict
		elif past_key_values is not None and not return_dict:
			predict, past_key_values = predict
			predict = (
				predict[:1] + (flax.core.unfreeze(past_key_values["cache"]),) + predict[1:]
			)
		return predict


@register_module(
	"base-module",
	config=MptConfig,
	model_type="mpt",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class FlaxMptModel(FlaxMptPretrainedModel):
	module_class = FlaxMptModule

	def get_input_embeddings(self):
		return self.module.wte

	def set_input_embeddings(self, value):
		self.module.wte = value


class FlaxMptForCausalLMModule(nn.Module):
	config: MptConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.transformer = FlaxMptModule(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.lm_head = Dense(
			self.config.vocab_size,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			use_bias=self.config.use_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		Forward pass through the MPT module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		predict: FlaxBaseModelOutput = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=True,
			segment_ids=segment_ids,
			output_hidden_states=output_hidden_states,
			init_cache=init_cache,
			output_attentions=output_attentions,
			deterministic=deterministic,
			input_embeds=input_embeds,
		)
		last_hidden_state = predict.last_hidden_state

		if self.config.use_lm_head:
			shared_kernel = self.model.variables["params"]["wte"]["embedding"].T.astype(
				self.param_dtype
			)
			logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				last_hidden_state,
			)
		else:
			logits = self.lm_head(last_hidden_state)

		if return_dict:
			return FlaxCausalLMOutput(logits=logits, hidden_states=predict.hidden_states)
		return logits, predict.hidden_states if output_hidden_states else (logits,)


@register_module(
	"causal-language-model",
	config=MptConfig,
	model_type="mpt",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
class FlaxMptForCausalLM(FlaxMptPretrainedModel):
	module_class = FlaxMptForCausalLMModule

	def get_input_embeddings(self):
		return self.module.transformer.wte

	def get_decoder(self):
		return self.module.transformer

	def set_input_embeddings(self, value):
		self.module.transformer.wte = value

	def set_decoder(self, decoder):
		self.module.transformer = decoder

	def set_output_embeddings(self, new_embeddings):
		self.module.lm_head = new_embeddings

	def get_output_embeddings(self):
		return self.module.lm_head

	def prepare_inputs_for_generation(
		self, input_ids, max_length, attention_mask: Optional[chex.Array] = None
	):
		batch_size, seq_length = input_ids.shape
		past_key_values = self.init_cache(batch_size, max_length)
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			extended_attention_mask = jax.lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)

		return {
			"past_key_values": past_key_values,
			"attention_mask": extended_attention_mask,
		}

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs["past_key_values"] = model_outputs.past_key_values
		return model_kwargs
