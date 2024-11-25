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

from functools import partial
from typing import Optional, Tuple, Union

import chex
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from fjformer import with_sharding_constraint
from flax import linen as nn
from flax.linen import Dense, combine_masks, make_causal_mask
from jax import lax
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers, get_logger
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.gemma2.gemma2_configuration import Gemma2Config as Gemma2Config
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module

logger = get_logger(__name__)


def add_positional_embedding(
	input_embedding: jax.Array,
	position: int,
	theta: int = 10_000,
) -> jax.Array:
	"""Adds positional embeddings to input embeddings. From DeepMind Gemma"""
	embed_dim = input_embedding.shape[-1]
	num_timescales = embed_dim // 2
	log_timescale_increment = jnp.log(float(theta)) / jnp.maximum(
		jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
	)
	inv_timescales = jnp.exp(
		jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
	)
	scaled_time = position * inv_timescales
	signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
	signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
	position_embedding = signal.astype(jnp.float32)

	return input_embedding + position_embedding


def apply_rope(
	inputs: jax.Array,  # [B, L]
	positions: jax.Array,  # [B, L]
	head_dim: int,
	theta: int = 10_000,
) -> jax.Array:
	"""Applies RoPE. From DeepMind Gemma"""
	fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
	timescale = theta**fraction

	sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
	sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
	sin = jnp.sin(sinusoid_inp)
	cos = jnp.cos(sinusoid_inp)

	first_half, second_half = jnp.split(inputs, 2, axis=-1)
	first_part = first_half * cos - second_half * sin
	second_part = second_half * cos + first_half * sin
	out = jnp.concatenate([first_part, second_part], axis=-1)
	return out.astype(inputs.dtype)


class FlaxGemma2RMSNorm(nn.Module):
	config: Gemma2Config
	dtype: jnp.dtype = jnp.float32

	def setup(self):
		self.epsilon = self.config.rms_norm_eps
		self.weight_kernel = self.param(
			"kernel",
			lambda _, shape: jnp.ones(shape),
			self.config.hidden_size,
		)

	def __call__(self, hidden_states):
		variance = jnp.asarray(hidden_states, dtype=jnp.float32)
		variance = jnp.power(variance, 2)
		variance = variance.mean(-1, keepdims=True)
		hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
		w = 1 + self.weight_kernel.astype(self.dtype)
		return (w * jnp.asarray(hidden_states, dtype=self.dtype)).astype(
			hidden_states.dtype
		)


class FlaxGemma2Attention(FlaxAttentionModule):
	config: Gemma2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None
	causal: bool = True
	is_cross_attention: bool = False

	def setup(self):
		config = self.config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = config.head_dim
		self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads

		kernel = jax.nn.initializers.normal(self.config.initializer_range)

		dense_class = partial(
			Dense,
			use_bias=config.attention_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=kernel,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.q_proj = dense_class(self.num_heads * self.head_dim)
		self.k_proj = dense_class(self.num_key_value_heads * self.head_dim)
		self.v_proj = dense_class(self.num_key_value_heads * self.head_dim)
		self.o_proj = dense_class(self.embed_dim)
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
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
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

		dropout_rng = None

		if not deterministic and self.config.attention_dropout > 0.0:
			dropout_rng = self.make_rng("dropout")
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
				query_states,
				key_states,
				value_states,
				attention_mask,
			)

		if bool((self.layer_idx % 2) == 0):
			sliding_window_mask = jnp.tril(
				jnp.ones_like(attention_mask, dtype=jnp.bool),
				k=-self.sliding_window,
			)
			window_mask = jnp.where(sliding_window_mask, 0, 1)
			attention_mask = jnp.logical_and(window_mask, attention_mask)
		attention_bias = lax.select(
			attention_mask > 0,
			jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
			jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
		)
		if bool((self.layer_idx % 2) == 0):
			if attention_bias.shape[-1] <= 1:  # when decoding
				attention_bias = attention_bias[:, :, :, -self.sliding_window :]

		query_length, key_length = query_states.shape[1], key_states.shape[1]

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=dropout_rng,
			deterministic=deterministic,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=self.has_variable("cache", "cached_key") or init_cache,
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


class FlaxGemma2MLP(nn.Module):
	config: Gemma2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

		self.act = ACT2FN[self.config.hidden_activation]
		dense_class = partial(
			Dense,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=kernel_init,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.gate_proj = dense_class(self.config.intermediate_size)
		self.up_proj = dense_class(self.config.intermediate_size)
		self.down_proj = dense_class(self.config.hidden_size)

	def __call__(self, hidden_states, deterministic=False):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

		hidden_states = self.down_proj(
			self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		return hidden_states


class FlaxGemma2DecoderLayer(nn.Module):
	config: Gemma2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		mlp_block = FlaxGemma2MLP
		attn_block = FlaxGemma2Attention

		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			mlp_block = flax.linen.partitioning.remat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)
			attn_block = flax.linen.partitioning.remat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(3, 5, 6, 7, 9),
			)
		self.is_sliding = bool(self.layer_idx % 2)
		self.self_attn = attn_block(
			self.config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.mlp = mlp_block(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.input_layernorm = FlaxGemma2RMSNorm(
			self.config,
			dtype=self.dtype,
		)
		self.post_attention_layernorm = FlaxGemma2RMSNorm(
			self.config,
			dtype=self.dtype,
		)
		self.pre_feedforward_layernorm = FlaxGemma2RMSNorm(
			self.config,
			dtype=self.dtype,
		)
		self.post_feedforward_layernorm = FlaxGemma2RMSNorm(
			self.config,
			dtype=self.dtype,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states, attn_weight = self.self_attn(
			hidden_states,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			deterministic,
			init_cache,
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
				self.mlp, hidden_states, self.config.scan_mlp_chunk_size, deterministic
			)
		else:
			hidden_states = self.mlp(hidden_states, deterministic)

		hidden_states = self.post_feedforward_layernorm(hidden_states)
		hidden_states = residual + hidden_states
		return hidden_states, attn_weight


class FlaxGemma2LayerCollection(nn.Module):
	config: Gemma2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.blocks = [
			FlaxGemma2DecoderLayer(
				self.config,
				layer_idx=i,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_hidden_layers)
		]

		self._frequencies = self.config.get_basic_frequencies()

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	) -> Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		"""
		Forward pass through the collection of decoder layers.

		Args:
		    hidden_states (chex.Array): Input tensor containing the hidden states.
		    attention_mask (chex.Array): Mask to apply during attention.
		    causal_mask (chex.Array): Causal mask for autoregressive decoding.
		    position_ids (chex.Array): Positional indices for the sequence.
		    segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
		    deterministic (bool): If True, disables dropout.
		    init_cache (bool): If True, initializes caching mechanism for fast decoding.
		    output_attentions (bool): If True, returns attention weights.
		    output_hidden_states (bool): If True, returns hidden states.

		Returns:
		    Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		        - hidden_states: The output tensor after layer processing.
		        - all_hidden_states: all of Hidden states (if `output_hidden_states` is True).
		        - self_attn_weights: Attention weights (if `output_attentions` is True).

		"""
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
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
					self.make_rng("fcm"), shape=(batch_size, 1, seq_length, seq_length)
				)
				> fcm_ratio
			)
			fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
			fcm_mask = fcm_mask.astype("bool")
		else:
			fcm_mask = None
		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=causal_mask,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
				fcm_mask=fcm_mask,
				segment_ids=segment_ids,
				frequencies=self._frequencies,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		outputs = (hidden_states, all_hidden_states, all_attentions)

		return outputs


@register_module(
	"base-module",
	config=Gemma2Config,
	model_type="gemma2",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=Gemma2Config, base_model_prefix="model")
class FlaxGemma2Model(nn.Module):
	config: Gemma2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.hidden_size = self.config.hidden_size
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.layers = FlaxGemma2LayerCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm = FlaxGemma2RMSNorm(
			self.config,
			dtype=self.dtype,
		)
		self.causal_mask = make_causal_mask(
			jnp.ones(
				shape=(1, self.config.granted_mask_max_position_embedding),
				dtype="bool",
			),
			dtype="bool",
		)

	# Ignore copy
	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		Forward pass through the Gemma2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
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
		if input_embeds is None and input_ids is not None:
			input_embeds = self.embed_tokens(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
		batch_size, sequence_length, _ = input_embeds.shape

		input_embeds = input_embeds * (self.config.hidden_size**0.5)
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		outputs = self.layers(
			hidden_states=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=self.causal_mask,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]
		hidden_states = self.norm(hidden_states)

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
	config=Gemma2Config,
	model_type="gemma2",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=Gemma2Config, base_model_prefix="model")
class FlaxGemma2ForCausalLM(nn.Module):
	config: Gemma2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		self.model = FlaxGemma2Model.flax_module(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(
			self.config.vocab_size,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxCausalLMOutput, Tuple]:
		"""
		Forward pass through the Gemma2 module.

		Args:
		    input_ids (Optional[chex.Array]): Input tensor containing token IDs.
		    attention_mask (Optional[chex.Array]): Mask for attention.
		    position_ids (Optional[chex.Array]): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""

		batch_size, seq_length = (
			input_ids.shape if input_ids is not None else input_embeds.shape[:2]
		)
		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, seq_length),
			)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			input_embeds=input_embeds,
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
			shared_kernel = self.model.variables["params"]["embed_tokens"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				hidden_states,
			)
		else:
			lm_logits = self.lm_head(hidden_states)

		if self.config.final_logit_softcapping is not None:
			lm_logits = lm_logits / self.config.final_logit_softcapping
			lm_logits = jax.nn.tanh(lm_logits)
			lm_logits = lm_logits * self.config.final_logit_softcapping

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
