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
from typing import Optional, Tuple, Union

import chex
import flax.linen.partitioning
import flax.struct
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import linen as nn
from flax.linen import Dense

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm as FlaxGrok1RMSNorm
from easydel.layers.rotary_embedding import get_rope
from easydel.modules.factory import register_module

# easydel.modules
from easydel.modules.flax_modeling_utils import (
	apply_rotary_pos_emb,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
	precompute_frequencies,
)
from easydel.modules.grok_1.grok_1_configuration import Grok1Config as Grok1Config
from easydel.modules.modeling_flax_outputs import FlaxMaskedLMOutput
from easydel.modules.modeling_utils import wrap_easydel_module

re_mat = flax.linen.partitioning.remat


@flax.struct.dataclass
class MoeModelOutput:
	last_hidden_state: chex.Array = None
	hidden_states: Optional[Tuple[chex.Array]] = None
	attentions: Optional[Tuple[chex.Array]] = None
	router_logits: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
	aux_loss: Optional[chex.Array] = None
	router_logits: Optional[Tuple[chex.Array]] = None


class FlaxGrok1Embedding(nn.Module):
	dtype: jnp.dtype = jnp.float32

	def __call__(self, query, key, frequencies, position_ids):
		sin, cos = frequencies

		sin = sin[position_ids][:, None, :, :]
		cos = cos[position_ids][:, None, :, :]

		key = apply_rotary_pos_emb(key, sin, cos)
		query = apply_rotary_pos_emb(query, sin, cos)

		return query.astype(self.dtype), key.astype(self.dtype)


class FlaxGrok1Attention(FlaxAttentionModule):
	config: Grok1Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		self.hidden_size = config.hidden_size
		self.head_dim = self.config.hidden_size // self.config.num_attention_heads
		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_key_value_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_key_value_heads
		self.q_proj = Dense(
			config.num_attention_heads * self.head_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.k_proj = Dense(
			config.num_key_value_heads * self.head_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.v_proj = Dense(
			config.num_key_value_heads * self.head_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.o_proj = Dense(
			config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.rotary = FlaxGrok1Embedding(self.dtype)
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
		self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)
		initial_rope_kwargs = dict(rope_type="default")
		if getattr(config, "rope_scaling", None) is not None:
			scaling_type = config.rope_scaling["type"]
			scaling_factor = config.rope_scaling["factor"]
			initial_rope_kwargs = dict(scaling_factor=scaling_factor, rope_type=scaling_type)

		self.rotary = get_rope(
			head_size=self.head_dim,
			rotary_dim=self.head_dim,
			max_position=self.config.granted_freq_max_position_embedding,
			base=config.rope_theta,
			rope_scaling=initial_rope_kwargs,
		)

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
		)

		(
			query_states,
			key_states,
			value_states,
			attention_mask,
			attention_bias,
		) = self.concatenate_to_cache(
			init_cache=init_cache,
			query=query_states,
			key=key_states,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=fcm_mask,
		)

		dropout_rng = None

		if not deterministic and self.config.attention_dropout > 0.0:
			dropout_rng = self.make_rng("dropout")

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

		attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class FlaxGrok1BLockSparseMLP(nn.Module):
	config: Grok1Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		config = self.config

		self.linear = Dense(
			config.intermediate_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.linear_1 = Dense(
			config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.linear_v = Dense(
			config.intermediate_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
		"""The __call__ function is the main function of a class.
		It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
		The __call__ method enables instances of a class to be called like standard Python functions.

		Args:
		    self: Represent the instance of the class
		    x: jnp.ndarray: Pass in the input to the layer
		    deterministic: bool: Determine whether to use dropout #
		        IGNORED

		Returns:
		    A tensor that is the result of applying a dropout function
		    to x
		"""

		x = control_mlp_sharding(x, self.config.partition_axis)

		return self.linear_1(nn.gelu(self.linear(x)) * self.linear_v(x))


class FlaxGrok1BlocKSparesTop2MLPCollection(nn.Module):
	config: Grok1Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.layers = [
			FlaxGrok1BLockSparseMLP(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_experts)
		]

	def __call__(
		self,
		selected_experts: chex.Array,
		hidden_states: chex.Array,
		routing_weights: chex.Array,
		batch_size: int,
		sequence_length: int,
		hidden_dim: int,
	) -> chex.Array:
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
		return final_hidden_state


class FlaxGrok1SparseMoeBlock(nn.Module):
	"""This implementation is
	strictly equivalent to standard MoE with full capacity (no
	dropped tokens). It's faster since it formulates MoE operations
	in terms of block-sparse operations to accomodate imbalanced
	assignments of tokens to experts, whereas standard MoE either
	(1) drop tokens at the cost of reduced performance or (2) set
	capacity factor to number of experts and thus waste computation
	and memory on padding.
	"""

	config: Grok1Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision("fastest")

	def setup(self) -> None:
		self.gate = Dense(
			self.config.num_experts,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=nn.initializers.normal(),
		)

		self.experts = FlaxGrok1BlocKSparesTop2MLPCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		e: bool = False,  # Ignored
	) -> Tuple[chex.Array, chex.Array]:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		batch_size, sequence_length, hidden_dim = hidden_states.shape

		router_logits = self.gate(hidden_states).astype(
			jnp.promote_types(self.dtype, jnp.float32)
		)
		routing_weights, selected_experts = jax.lax.top_k(
			router_logits, k=self.config.num_experts_per_tok
		)
		routing_weights = jax.nn.softmax(
			routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
		)

		return (
			self.experts(
				selected_experts=selected_experts,
				batch_size=batch_size,
				sequence_length=sequence_length,
				hidden_dim=hidden_dim,
				hidden_states=hidden_states,
				routing_weights=routing_weights,
			),
			router_logits,
		)


class FlaxGrok1DecoderLayer(nn.Module):
	config: Grok1Config
	layer_index: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		# hidden_states: chex.Array
		# attention_mask: chex.Array
		# causal_mask: chex.Array
		# position_ids: chex.Array
		# deterministic: bool = True
		# init_cache: bool = False
		# output_attentions: bool = True

		attn_block = FlaxGrok1Attention
		mlp_block = FlaxGrok1SparseMoeBlock
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = re_mat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(4, 5, 6),
			)
			mlp_block = re_mat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)
		self.attn = attn_block(
			config=self.config,
			layer_index=self.layer_index,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.moe_block = mlp_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.pre_attn_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.post_attn_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.pre_moe_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.post_moe_norm = FlaxGrok1RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_router_logits: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights.
		    output_router_logits (bool): If True, outputs router logits.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array, Optional[chex.Array]]: A tuple containing the residual_states, hidden states, and the attention weights.
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
			deterministic,
			init_cache,
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


class FlaxGrok1DecoderLayerCollection(nn.Module):
	config: Grok1Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxGrok1DecoderLayer(
				layer_index=layer_index,
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(layer_index),
			)
			for layer_index in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_router_logits: bool = False,
		output_hidden_states: bool = False,
	) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights.
		    output_router_logits (bool): If True, outputs router logits.
		    output_hidden_states (bool): If True, outputs all of hidden states.
		Returns:
		    Tuple[chex.Array, Optional[chex.Array], Optional[chex.Array], Optional[chex.Array]]:
		        A tuple containing the hidden_states, all_attentions, all_hidden_states, all_router_logits.
		"""
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_router_logits = () if output_router_logits else None
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
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				output_attentions=output_attentions,
				output_router_logits=output_router_logits,
				init_cache=init_cache,
				frequencies=frequencies,
				causal_mask=causal_mask,
				deterministic=deterministic,
				segment_ids=segment_ids,
				fcm_mask=fcm_mask,
			)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

			if output_router_logits:
				all_router_logits += (layer_outputs[-1],)

		outputs = (hidden_states,)
		if output_attentions:
			outputs += (all_self_attns,)
		if output_hidden_states:
			outputs += (all_hidden_states,)
		if output_router_logits:
			outputs += (all_router_logits,)
		return outputs


@register_module(
	"base-module",
	config=Grok1Config,
	model_type="grok-1",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=Grok1Config, base_model_prefix="model")
class FlaxGrok1Model(nn.Module):
	config: Grok1Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.layers = FlaxGrok1DecoderLayerCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.norm = FlaxGrok1RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		initial_rope_kwargs = dict(rope_type="none")
		if self.config.rope_scaling is not None:
			scaling_type = self.config.rope_scaling["type"]
			scaling_factor = self.config.rope_scaling["factor"]
			initial_rope_kwargs = dict(scaling_factor=scaling_factor, rope_type=scaling_type)
		self.frequencies = precompute_frequencies(
			max_position_embeddings=self.config.granted_freq_max_position_embedding,
			dim=self.config.hidden_size // self.config.num_attention_heads,
			base=self.config.rope_theta,
			**initial_rope_kwargs,
		)
		self.causal_mask = flax.linen.make_causal_mask(
			jnp.ones(
				shape=(1, self.config.granted_mask_max_position_embedding),
				dtype="bool",
			),
			dtype="bool",
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		output_router_logits: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> MoeModelOutput | Tuple:
		"""
		Forward pass through the Grok1 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    output_router_logits (Optional[bool]): If True, output router logits.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    MoeModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits
		if input_ids is not None and input_embeds is not None:
			raise ValueError(
				"You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
			)

		if input_embeds is None and input_ids is not None:
			input_embeds = self.wte(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
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
		collection_outputs = self.layers(
			hidden_states=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			segment_ids=segment_ids,
			causal_mask=self.causal_mask,
			frequencies=self.frequencies,
			output_attentions=output_attentions,
			output_router_logits=output_router_logits,
			output_hidden_states=output_hidden_states,
			init_cache=init_cache,
			deterministic=deterministic,
		)
		all_self_attns = None
		all_hidden_states = None
		all_router_logits = None
		hidden_states = collection_outputs[0]
		if output_attentions:
			all_self_attns = collection_outputs[1]
		if output_hidden_states:
			all_hidden_states = collection_outputs[2 if output_attentions else 1]
		if output_router_logits:
			all_router_logits = collection_outputs[-1]
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
@wrap_easydel_module(config_class=Grok1Config, base_model_prefix="model")
class FlaxGrok1ForCausalLM(nn.Module):
	"""
	Grok1 model for causal language modeling, including the language model head.

	Attributes:
	    config (Grok1Config): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: Grok1Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.model = FlaxGrok1Model.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			use_bias=False,
			kernel_init=nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.output_multiplier_scale = self.config.output_multiplier_scale

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		output_router_logits: Optional[bool] = None,
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | Tuple:
		"""
		Forward pass through the Grok1 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    output_router_logits (Optional[bool]): If True, output router logits.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    MoeCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			input_embeds=input_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			output_router_logits=output_router_logits,
			init_cache=init_cache,
			deterministic=deterministic,
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
