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
import flax.core
import jax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module
from easydel.modules.openelm.openelm_configuration import (
	OpenELMConfig as OpenELMConfig,
)
from easydel.modules.openelm.openelm_configuration import (
	make_divisible,
)

re_mat = nn_partitioning.remat


class FlaxOpenELMMLP(nn.Module):
	config: OpenELMConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None


class FlaxOpenELMMultiHeadCausalAttention(FlaxAttentionModule):
	"""
	FlaxOpenELMAttention implements an attention mechanism with rotary embeddings.

	Attributes:
	    config (OpenELMConfig): Configuration for the attention module.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: OpenELMConfig
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		layer_idx = self.layer_idx
		head_dim = config.head_dim
		q_heads = config.num_query_heads[layer_idx]
		k_heads = config.num_kv_heads[layer_idx]
		v_heads = config.num_kv_heads[layer_idx]

		self.qkv_proj = nn.Dense(
			(q_heads + k_heads + v_heads) * head_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		if config.normalize_qk_projections:
			self.q_norm = RMSNorm(
				dim=config.head_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				eps=1e-6,
			)
			self.k_norm = RMSNorm(
				dim=config.head_dim,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				eps=1e-6,
			)
		else:
			self.q_norm = None
			self.k_norm = None

		self.out_proj = nn.Dense(
			config.model_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			precision=self.precision,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
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
		dropout_rng = None

		if not deterministic and self.config.attention_dropout > 0.0:
			dropout_rng = self.make_rng("dropout")

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

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class FlaxOpenELMFeedForwardNetwork(nn.Module):
	config: OpenELMConfig
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		config = self.config
		layer_idx = self.layer_idx
		ffn_multiplier = config.ffn_multipliers[layer_idx]
		intermediate_dim = int(
			make_divisible(
				ffn_multiplier * config.model_dim,  # type:ignore
				divisor=config.ffn_dim_divisor,
			)
		)
		if config.ffn_with_glu:
			# FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
			self.proj_1 = nn.Dense(
				2 * intermediate_dim,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
			self.proj_2 = nn.Dense(
				config.model_dim,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
			self.ffn_with_glu = True
		else:
			self.proj_1 = nn.Dense(
				intermediate_dim,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
			self.proj_2 = nn.Dense(
				config.model_dim,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
				**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
			)
			self.ffn_with_glu = False

		self.act = ACT2FN[config.activation_fn_name]

	def __call__(self, x: chex.Array, e: bool = False) -> chex.Array:
		x = control_mlp_sharding(x, self.config.partition_axis)

		if self.ffn_with_glu:
			y_12 = self.proj_1(x)
			y_1, y_2 = jnp.split(y_12, 2, axis=-1)
			return self.proj_2(self.act(y_1) * y_2)
		else:
			return self.proj_2(self.act(self.proj_1(x)))


class FlaxOpenELMDecoderLayer(nn.Module):
	config: OpenELMConfig
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		attn_block = FlaxOpenELMMultiHeadCausalAttention
		mlp_block = FlaxOpenELMFeedForwardNetwork
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = re_mat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(3, 5, 6, 7, 9),
			)
			mlp_block = re_mat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)

		self.attn = attn_block(
			config=self.config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ffn = mlp_block(
			config=self.config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.ffn_norm = RMSNorm(
			self.config.model_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			eps=1e-6,
		)
		self.attn_norm = RMSNorm(
			self.config.model_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			eps=1e-6,
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
		hidden_states = self.attn_norm(hidden_states)

		hidden_states, self_attn_weights = self.attn(
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
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.ffn_norm(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.ffn,
				hidden_states,
				self.config.scan_mlp_chunk_size,
				deterministic,
			)
		else:
			feed_forward_hidden_states = self.ffn(
				hidden_states,
				deterministic,
			)
		hidden_states = residual + feed_forward_hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		return outputs  # type:ignore


class FlaxOpenELMDecoderLayerCollection(nn.Module):
	config: OpenELMConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		self.layers = [
			FlaxOpenELMDecoderLayer(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				layer_idx=i,
				name=str(i),
			)
			for i in range(self.config.num_transformer_layers)
		]
		self._frequencies = self.config.get_basic_frequencies(
			head_size=self.config.head_dim,
			rotary_dim=self.config.head_dim,
			base=self.config.rope_freq_constant,
		)

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
					self.make_rng("fcm"),
					shape=(batch_size, 1, seq_length, seq_length),
				)
				> fcm_ratio
			)
			fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
			fcm_mask = fcm_mask.astype("bool")
		else:
			fcm_mask = None
		for layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			output = layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				output_attentions=output_attentions,
				init_cache=init_cache,
				segment_ids=segment_ids,
				deterministic=deterministic,
				position_ids=position_ids,
				fcm_mask=fcm_mask,
				frequencies=self._frequencies,
			)
			hidden_states = output[0]

			if output_attentions:
				output_attentions += (output[1],)

		return hidden_states, all_hidden_states, all_attentions


@register_module(
	"base-module",
	config=OpenELMConfig,
	model_type="openelm",
	embedding_layer_names=["token_embeddings"],
)
@wrap_easydel_module(config_class=OpenELMConfig, base_model_prefix="transformer")
class FlaxOpenELMModel(nn.Module):
	"""
	Core module of the OpenELM model, including embedding, decoder layers, and normalization.

	Attributes:
	    config (OpenELMConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: OpenELMConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		config = self.config
		self.token_embeddings = nn.Embed(
			config.vocab_size,
			config.model_dim,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.layers = FlaxOpenELMDecoderLayerCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm = RMSNorm(
			config.model_dim, dtype=self.dtype, param_dtype=self.param_dtype, eps=1e-6
		)
		if config.share_input_output_layers:
			self.classifier = None
		else:
			self.classifier = nn.Dense(
				config.vocab_size,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
		self.num_transformer_layers = config.num_transformer_layers

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
		Forward pass through the OpenELM module.

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
			input_embeds = self.token_embeddings(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
		batch_size, sequence_length, _ = input_embeds.shape

		assert (
			sequence_length <= self.config.max_context_length
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_context_length} got {sequence_length})"
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		outputs = self.layers(
			hidden_states=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			init_cache=init_cache,
			output_attentions=output_attentions,
			deterministic=deterministic,
			causal_mask=self.causal_mask,
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
			return tuple(value for value in outputs if value is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


@register_module(
	"causal-language-model",
	config=OpenELMConfig,
	model_type="openelm",
	embedding_layer_names=["token_embeddings"],
)
@wrap_easydel_module(config_class=OpenELMConfig, base_model_prefix="transformer")
class FlaxOpenELMForCausalLM(nn.Module):
	config: OpenELMConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.transformer = FlaxOpenELMModel.flax_module(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.lm_head = nn.Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			precision=self.precision,
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
		Forward pass through the OpenELM module.

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
		outputs = self.transformer(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			deterministic=deterministic,
			input_embeds=input_embeds,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]

		if self.config.share_input_output_layers:
			shared_kernel = self.transformer.variables["params"]["token_embeddings"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				hidden_states,
			)
		else:
			lm_logits = self.lm_head(hidden_states)

		lm_logits = lm_logits[:, : self.config.vocab_size]
		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
