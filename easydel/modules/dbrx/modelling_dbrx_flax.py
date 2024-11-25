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
import flax.linen
import flax.struct
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import linen as nn
from flax.linen import Dense
from flax.linen.partitioning import remat

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.dbrx.dbrx_configuration import (
	DbrxAttentionConfig as DbrxAttentionConfig,
)
from easydel.modules.dbrx.dbrx_configuration import DbrxConfig as DbrxConfig
from easydel.modules.dbrx.dbrx_configuration import DbrxFFNConfig as DbrxFFNConfig

# easydel.modules
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import FlaxMaskedLMOutput
from easydel.modules.modeling_utils import wrap_easydel_module


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


class FlaxDbrxAttention(FlaxAttentionModule):
	"""
	FlaxDbrxAttention implements an attention mechanism with rotary embeddings.

	Attributes:
	    config (DbrxConfig): Configuration for the attention module.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.num_attention_heads = self.config.n_heads
		self.num_key_value_heads = self.config.attn_config.kv_n_heads
		config = self.config
		self.hidden_size = config.hidden_size
		self.head_dim = self.config.d_model // self.config.n_heads
		self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

		if self.num_key_value_groups == 1:
			assert self.num_attention_heads == self.config.attn_config.kv_n_heads
		self.Wqkv = Dense(
			self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.out_proj = Dense(
			config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
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
		self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

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
		dropout_rng = None

		if not deterministic and self.config.attn_config.attn_pdrop > 0.0:
			dropout_rng = self.make_rng("dropout")
		query_states, key_states = self.rotary(
			position_ids,
			query=query_states,
			key=key_states,
			frequencies=frequencies,
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
		attn_output = self.out_proj(attn_output)

		attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
		outputs = (attn_output,)
		if output_attentions:
			outputs += (output_attentions,)
		return outputs


class FlaxDbrxNormAttentionNorm(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.norm_1 = nn.LayerNorm(
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
		)
		self.attn = FlaxDbrxAttention(  # statics 3,5,6,7
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm_2 = nn.LayerNorm(
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
		)

		self.dropout = flax.linen.Dropout(self.config.resid_pdrop)

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
	) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

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
		    Tuple[chex.Array, chex.Array, Optional[chex.Array]]: A tuple containing the residual_states, hidden states, and the attention weights.
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
			init_cache=init_cache,
			deterministic=deterministic,
			fcm_mask=fcm_mask,
			frequencies=frequencies,
		)
		hidden_states, attn_weights = attn_out if output_attentions else (attn_out[0], None)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = hidden_states + residual_states

		residual_states = hidden_states
		hidden_states = self.norm_2(hidden_states)

		return residual_states, hidden_states, attn_weights


class FlaxDbrxExpertGLU(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		shape = (
			self.config.ffn_config.moe_num_experts * self.config.ffn_config.ffn_hidden_size,
			self.config.d_model,
		)
		init_fn = nn.initializers.normal(dtype=self.dtype)
		self.w1 = self.param("w1", init_fn, shape, self.param_dtype)
		self.v1 = self.param("v1", init_fn, shape, self.param_dtype)
		self.w2 = self.param("w2", init_fn, shape, self.param_dtype)
		self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

	def __call__(self, x: chex.Array, expert_idx: int) -> chex.Array:
		expert_shape = (
			self.config.ffn_config.moe_num_experts,
			self.config.ffn_config.ffn_hidden_size,
			self.config.d_model,
		)
		expert_w1 = self.w1.reshape(expert_shape)[expert_idx]
		expert_v1 = self.v1.reshape(expert_shape)[expert_idx]
		expert_w2 = self.w2.reshape(expert_shape)[expert_idx]

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


class FlaxDbrxExperts(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.mlp = FlaxDbrxExpertGLU(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
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


class FlaxDbrxRouter(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.hidden_size = self.config.d_model
		self.moe_num_experts = self.config.ffn_config.moe_num_experts
		self.moe_top_k = self.config.ffn_config.moe_top_k
		self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
		self.moe_normalize_expert_weights = (
			self.config.ffn_config.moe_normalize_expert_weights
		)
		self.uniform_expert_assignment = self.config.ffn_config.uniform_expert_assignment

		self.layer = Dense(
			self.moe_num_experts,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
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
	) -> Tuple[chex.Array, chex.Array, chex.Array]:
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


class FlaxDbrxFFN(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.router = FlaxDbrxRouter(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.experts = FlaxDbrxExperts(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self, x: chex.Array, deterministic: bool = False
	) -> Tuple[chex.Array, chex.Array]:
		x = control_mlp_sharding(x, self.config.partition_axis)
		weights, top_weights, top_experts = self.router(x, deterministic=deterministic)
		out = self.experts(x, weights, top_weights, top_experts)
		return out, weights


class FlaxDbrxBlock(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.hidden_size = self.config.d_model
		self.resid_pdrop = self.config.resid_pdrop
		attn_block = FlaxDbrxNormAttentionNorm
		ffn_block = FlaxDbrxFFN
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = remat(
				attn_block,
				static_argnums=(3, 5, 6, 7, 9),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)
			ffn_block = remat(
				ffn_block,
				static_argnums=(1,),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)
		self.norm_attn_norm = attn_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ffn = ffn_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
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
		output_router_logits: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
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

		resid_states, hidden_states, self_attn_weights = self.norm_attn_norm(
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

		hidden_states, router_logits = self.ffn(hidden_states, deterministic)
		hidden_states = resid_states + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if output_router_logits:
			outputs += (router_logits,)

		return outputs


class FlaxDbrxBlockCollection(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxDbrxBlock(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=f"{i}",
			)
			for i in range(self.config.n_layers)
		]

		self._frequencies = self.config.get_basic_frequencies(
			rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			base=self.config.attn_config.rope_theta,
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
		output_router_logits: bool = False,
		output_hidden_states: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
		"""
		Forward pass of the attentionNrom module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights.
		    output_router_logits (bool): If True, outputs router logits.
		    output_hidden_states (bool): If True, outputs all of hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, Optional[chex.Array], Optional[chex.Array], Optional[chex.Array]]:
		        A tuple containing the hidden_states, all_attentions, all_hidden_states, all_router_logits.
		"""
		all_hidden_states = ()
		all_router_logits = ()
		all_attentions = ()
		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=causal_mask,
				segment_ids=segment_ids,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
				output_router_logits=output_router_logits,
				fcm_mask=fcm_mask,
				frequencies=self._frequencies,
			)
			hidden_states = outputs[0]
			if output_attentions:
				all_attentions += (outputs[1],)
			if output_router_logits:
				all_router_logits += (outputs[-1],)
		return (
			hidden_states,
			all_attentions,
			all_hidden_states,
			all_router_logits,
		)


@register_module(
	"base-module",
	config=DbrxConfig,
	model_type="dbrx",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
@wrap_easydel_module(config_class=DbrxConfig, base_model_prefix="transformer")
class FlaxDbrxModel(nn.Module):
	config: DbrxConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.padding_idx = self.config.pad_token_id
		self.vocab_size = self.config.vocab_size
		self.emb_pdrop = self.config.emb_pdrop

		self.wte = nn.Embed(
			self.config.vocab_size,
			self.config.d_model,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.blocks = FlaxDbrxBlockCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm_f = nn.LayerNorm(
			use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype
		)

		self.causal_mask = flax.linen.make_causal_mask(
			jnp.ones(
				(1, self.config.granted_mask_max_position_embedding),
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
		Forward pass through the Dbrx module.

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
		collection_outputs = self.blocks(
			hidden_states=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=self.causal_mask,
			output_attentions=output_attentions,
			output_router_logits=output_router_logits,
			output_hidden_states=output_hidden_states,
			init_cache=init_cache,
			deterministic=deterministic,
			segment_ids=segment_ids,
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
		hidden_states = self.norm_f(hidden_states)

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
	config=DbrxConfig,
	model_type="dbrx",
	embedding_layer_names=["wte"],
	layernorm_names=["norm_1", "norm_2", "norm_f"],
)
@wrap_easydel_module(config_class=DbrxConfig, base_model_prefix="transformer")
class FlaxDbrxForCausalLM(nn.Module):
	"""
	Dbrx model for causal language modeling, including the language model head.

	Attributes:
	    config (DbrxConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: DbrxConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.transformer = FlaxDbrxModel.flax_module(
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
		Forward pass through the Dbrx module.

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
		outputs = self.transformer(
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
