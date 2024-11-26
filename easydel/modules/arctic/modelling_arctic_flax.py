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
from typing import Optional, Tuple, Union

import chex
import flax
import jax
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import linen as nn
from flax.linen import Dense
from flax.linen import partitioning as nn_partitioning
from jax import numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.modules.arctic.arctic_configuration import ArcticConfig
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import MoeCausalLMOutput, MoeModelOutput
from easydel.modules.modeling_utils import wrap_easydel_module

re_mat = nn_partitioning.remat


class ArcticRMSNorm(nn.Module):
	dim: int
	eps: float = 1e-6
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16

	def setup(self) -> None:
		self.weight = self.param(
			"kernel",
			nn.initializers.ones,
			(self.dim,),
			self.param_dtype,
		)

	def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
		return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
		output = self._norm(x).astype(self.dtype)
		weight = self.weight.astype(self.dtype)
		return output * weight


class FlaxArcticAttention(FlaxAttentionModule):
	"""
	FlaxArcticAttention implements an attention mechanism with rotary embeddings.

	Attributes:
	    config (ArcticConfig): Configuration for the attention module.
	    layer_index (int): Index of the current layer.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: ArcticConfig
	layer_index: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		"""
		Sets up the attention module by initializing projection layers, rotary embeddings, and other parameters.

		The setup method initializes the following:
		- `hidden_size`, `num_heads`, `head_dim`: Dimensions and size attributes based on the configuration.
		- `num_key_value_heads`, `num_key_value_groups`: Configuration for key-value heads.
		- `max_position_embeddings`: Maximum number of position embeddings.
		- `q_proj`, `k_proj`, `v_proj`, `o_proj`: Dense layers for query, key, value, and output projections.
		- `rotary`: Rotary embedding module.
		- `attention_performer`: Flexible attention module handling the main attention computations.
		"""
		config = self.config
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings

		dense = functools.partial(
			Dense,
			use_bias=getattr(self.config, "attention_bias", False),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.q_proj = dense(self.num_heads * self.head_dim)
		self.k_proj = dense(self.num_key_value_heads * self.head_dim)
		self.v_proj = dense(self.num_key_value_heads * self.head_dim)
		self.o_proj = dense(self.num_heads * self.head_dim)

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
		causal_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = True,
		frequencies: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    position_ids (chex.Array): Position indices for the tokens.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.

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
			positions=position_ids,
			query=query_states,
			key=key_states,
			frequencies=frequencies,
		)
		dropout_rng = None
		if not deterministic and self.config.attn_config.attn_pdrop > 0.0:
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
			fcm_mask=None,
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
		attn_output = self.o_proj(attn_output)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class ArcticMLP(nn.Module):
	"""
	ArcticMLP is a multi-layer perceptron (MLP) module for neural network models,
	configured with specific settings and optional residual connections.

	Attributes:
	    config (ArcticConfig): Configuration object containing model parameters.
	    dtype (jnp.dtype): Data type for computation (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for model parameters (default is jnp.bfloat16).
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations (default is "fastest").
	    is_residual_mlp (bool): Flag to determine if the MLP includes residual connections (default is False).
	"""

	config: ArcticConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None
	is_residual_mlp: bool = False

	def setup(self) -> None:
		"""
		Initializes the MLP module by setting up dense layers and activation functions.

		The setup method determines the dimensions for the hidden layers and intermediate layers
		based on the configuration and whether the MLP is residual. It also initializes the dense
		layers and the activation function.
		"""
		config = self.config
		self.hidden_dim = config.hidden_size
		self.ffn_dim = (
			config.intermediate_size if not self.is_residual_mlp else self.hidden_dim
		)
		dense = functools.partial(
			Dense,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.w1 = dense(self.ffn_dim)
		self.w3 = dense(self.ffn_dim)
		self.w2 = dense(self.hidden_dim)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, x: chex.Array, e=None):
		"""
		Forward pass of the MLP module.

		Args:
		    x (chex.Array): Input tensor.
		    e (Optional): Unused parameter (for compatibility).

		Returns:
		    chex.Array: Output tensor after applying dense layers and activation functions.
		"""
		x = control_mlp_sharding(x, self.config.partition_axis)
		w1 = self.act_fn(self.w1(x))
		w3 = self.w3(x)
		return self.w2(w1 * w3)


class FlaxArcticBlocKSparesMLPCollection(nn.Module):
	"""
	FlaxArcticBlocKSparesMLPCollection is a collection of MLP layers that can be selectively activated,
	forming a part of a Mixture of Experts (MoE) architecture.

	Attributes:
	    config (ArcticConfig): Configuration object containing model parameters.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for model parameters (default is jnp.bfloat16).
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations (default is "fastest").
	"""

	config: ArcticConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.layers = [
			ArcticMLP(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_local_experts)
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
		"""
		Forward pass that applies the selected experts to the input hidden states.

		Args:
		    selected_experts (chex.Array): Indices of selected experts for each token.
		    hidden_states (chex.Array): Input tensor containing the hidden states.
		    routing_weights (chex.Array): Weights assigned by the router for each expert.
		    batch_size (int): Batch size of the input data.
		    sequence_length (int): Sequence length of the input data.
		    hidden_dim (int): Hidden dimension of the input data.

		Returns:
		    chex.Array: The final output after processing through the experts.
		"""
		final_hidden_state = jnp.zeros_like(hidden_states)

		for index in range(self.config.num_local_experts):
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


class FlaxArcticMoE(nn.Module):
	"""
	FlaxArcticMoE implements a Mixture of Experts (MoE) layer, where each input can be processed by one or more experts.

	Attributes:
	    config (ArcticConfig): Configuration object containing model parameters.
	    layer_id (int): The index of the layer in the overall network.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for model parameters (default is jnp.bfloat16).
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations (default is "fastest").
	"""

	config: ArcticConfig
	layer_id: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		config = self.config
		layer_id = self.layer_id
		self.hidden_dim = config.hidden_size
		self.num_experts = config.num_local_experts

		self.top_k = config.num_experts_per_tok
		self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0

		if self.is_moe_layer:
			self.gate = Dense(
				self.config.num_local_experts,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=nn.initializers.normal(),
			)
			self.experts = FlaxArcticBlocKSparesMLPCollection(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
		else:
			self.mlp = ArcticMLP(
				config=config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				is_residual_mlp=False,
			)

	def _call_moe(
		self,
		hidden_states: chex.Array,
		e: bool = False,  # Ignored
	) -> Tuple[chex.Array, chex.Array]:
		"""
		Processes the input through the MoE layer, involving gating and expert selection.

		Args:
		    hidden_states (chex.Array): The input tensor.
		    e (bool): An optional parameter, ignored in this implementation.

		Returns:
		    Tuple[chex.Array, chex.Array]: The processed output and an auxiliary loss.
		"""
		if self.is_moe_layer:
			hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
			batch_size, sequence_length, hidden_dim = hidden_states.shape

			router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
				jnp.promote_types(self.dtype, jnp.float32)
			)
			routing_weights, selected_experts = jax.lax.top_k(
				router_logits, k=self.config.num_experts_per_tok
			)
			routing_weights = jax.nn.softmax(
				routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
			)

			return self.experts(
				selected_experts=selected_experts,
				batch_size=batch_size,
				sequence_length=sequence_length,
				hidden_dim=hidden_dim,
				hidden_states=hidden_states,
				routing_weights=routing_weights,
			), auxiliary_load_balancing_loss_func(
				(router_logits,),  # type:ignore
				self.num_experts,
				self.top_k,
				None,
			)
		else:
			return self.mlp(hidden_states), jnp.array([0], hidden_states.dtype)

	def __call__(self, hidden_states: chex.Array, e: bool = False):  # Ignored
		"""
		Determines whether to use the MoE layer or a standard MLP, based on the configuration.

		Args:
		    hidden_states (chex.Array): The input tensor containing hidden states.
		    e (bool): An optional parameter, ignored in this implementation.

		Returns:
		    Tuple[chex.Array, chex.Array]: The output tensor and a scalar auxiliary loss.
		"""
		if self.is_moe_layer:
			return self._call_moe(hidden_states=hidden_states, e=e)
		return self.mlp(hidden_states, e=e), jnp.array(0.0, dtype=hidden_states.dtype)


class FlaxArcticSparseMoeBlock(nn.Module):
	"""This implementation is
	strictly equivalent to standard MoE with full capacity (no
	dropped tokens). It's faster since it formulates MoE operations
	in terms of block-sparse operations to accomodate imbalanced
	assignments of tokens to experts, whereas standard MoE either
	(1) drop tokens at the cost of reduced performance or (2) set
	capacity factor to number of experts and thus waste computation
	and memory on padding.
	"""

	config: ArcticConfig
	layer_index: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision("fastest")

	def setup(self) -> None:
		self.is_moe = (self.layer_index + 1) % self.config.moe_layer_frequency == 0
		if self.is_moe:
			self.gate = Dense(
				self.config.num_local_experts,
				use_bias=False,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				kernel_init=nn.initializers.normal(),
			)

			self.experts = FlaxArcticBlocKSparesMLPCollection(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
		else:
			self.mlp = ArcticMLP(
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

		if self.is_moe:
			router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
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
		else:
			return self.mlp(hidden_states), jnp.asarray([0], hidden_states.dtype)


class FlaxArcticDecoderLayer(nn.Module):
	"""
	FlaxArcticDecoderLayer represents a single layer in a Transformer-like model,
	incorporating self-attention and Mixture of Experts (MoE) mechanisms.

	Attributes:
	    config (ArcticConfig): Configuration object containing model parameters.
	    layer_index (int): The index of the current layer.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for model parameters (default is jnp.bfloat16).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: ArcticConfig
	layer_index: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		"""Initializes the layer components, including attention and MoE blocks, and layer normalization."""

		attn_block = FlaxArcticAttention
		mlp_block = FlaxArcticSparseMoeBlock

		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = re_mat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(2, 5, 6, 7, 8),
			)
			mlp_block = re_mat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)
		self.self_attn = attn_block(
			config=self.config,
			layer_index=self.layer_index,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.block_sparse_moe = mlp_block(
			config=self.config,
			dtype=self.dtype,
			layer_index=self.layer_index,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.input_layernorm = ArcticRMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.post_attention_layernorm = ArcticRMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.parallel_attn_mlp_res = (
			self.config.parallel_attn_mlp_res and self.block_sparse_moe.is_moe_layer
		)
		if self.parallel_attn_mlp_res:
			self.residual_layernorm = ArcticRMSNorm(
				dim=self.config.hidden_size,
				eps=self.config.rms_norm_eps,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
			)
			self.residual_mlp = ArcticMLP(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				is_residual_mlp=True,
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
		output_attentions: bool = True,
		frequencies: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		"""
		Forward pass for the decoder layer, applying self-attention and MoE transformations.

		Args:
		    hidden_states (chex.Array): Input tensor containing the hidden states.
		    attention_mask (chex.Array): Mask to apply during attention.
		    causal_mask (chex.Array): Causal mask for autoregressive decoding.
		    position_ids (chex.Array): Positional indices for the sequence.
		    segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
		    deterministic (bool): If True, disables dropout.
		    init_cache (bool): If True, initializes caching mechanism for fast decoding.
		    output_attentions (bool): If True, returns attention weights.

		Returns:
		    Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		        - hidden_states: The output tensor after layer processing.
		        - self_attn_weights: Attention weights (if `output_attentions` is True).
		        - gate_loss: Loss associated with the MoE gating mechanism.
		"""
		residual_input = hidden_states
		hidden_states = self.input_layernorm(hidden_states)
		attn_out = self.self_attn(
			hidden_states,
			attention_mask,
			causal_mask,
			position_ids,
			segment_ids,
			deterministic,
			init_cache,
			output_attentions,
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


class FlaxArcticDecoderLayerCollection(nn.Module):
	"""
	Collection of Flax Arctic Decoder Layers.

	Attributes:
	    config (ArcticConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: ArcticConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxArcticDecoderLayer(
				layer_index=layer_index,
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(layer_index),
			)
			for layer_index in range(self.config.num_hidden_layers)
		]
		self._frequencies = self.config.get_basic_frequencies()

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: chex.Array,
		deterministic: bool = True,
		init_cache: bool = False,
		output_hidden_states: Optional[bool] = False,
		output_attentions: Optional[bool] = False,
	) -> Tuple:
		"""
		Forward pass through the collection of decoder layers.

		Args:
		    hidden_states (chex.Array): The hidden states input to the decoder.
		    attention_mask (chex.Array): Mask for attention mechanism.
		    causal_mask (chex.Array): Causal mask for autoregressive decoding.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
		    deterministic (bool): If True, disable dropout.
		    init_cache (bool): If True, initialize cache for decoding.
		    output_hidden_states (Optional[bool]): If True, output all hidden states.
		    output_attentions (Optional[bool]): If True, output attention weights.

		Returns:
		    Tuple: The final hidden state, optional attention weights, and router losses.
		"""
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_router_losses = ()
		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				output_attentions=output_attentions,
				init_cache=init_cache,
				causal_mask=causal_mask,
				deterministic=deterministic,
				segment_ids=segment_ids,
				frequencies=self._frequencies,
			)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

			all_router_losses += (layer_outputs[-1],)

		outputs = (hidden_states,)
		if output_attentions:
			outputs += (all_self_attns,)
		if output_hidden_states:
			outputs += (all_hidden_states,)
		outputs += (all_router_losses,)
		return outputs


@register_module(
	"base-module",
	config=ArcticConfig,
	model_type="arctic",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=ArcticConfig, base_model_prefix="model")
class FlaxArcticModel(nn.Module):
	"""
	Core module of the Arctic model, including embedding, decoder layers, and normalization.

	Attributes:
	    config (ArcticConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: ArcticConfig
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

		self.layers = FlaxArcticDecoderLayerCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.norm = ArcticRMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
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
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> MoeModelOutput | Tuple:
		"""
		Forward pass through the Arctic module.

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
		    MoeModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if input_ids is not None and input_embeds is not None:
			raise ValueError(
				"You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
			)

		if input_embeds is None and input_ids is not None:
			input_embeds = self.embed_tokens(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
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
		collection_outputs = self.layers(
			hidden_states=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=self.causal_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			init_cache=init_cache,
			deterministic=deterministic,
			segment_ids=segment_ids,
		)
		all_self_attns = None
		all_hidden_states = None
		hidden_states = collection_outputs[0]
		if output_attentions:
			all_self_attns = collection_outputs[1]
		if output_hidden_states:
			all_hidden_states = collection_outputs[2 if output_attentions else 1]

		all_router_losses = collection_outputs[-1]
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
@wrap_easydel_module(config_class=ArcticConfig, base_model_prefix="model")
class FlaxArcticForCausalLM(nn.Module):
	"""
	Arctic model for causal language modeling, including the language model head.

	Attributes:
	    config (ArcticConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: ArcticConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.model = FlaxArcticModel.flax_module(
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
		init_cache: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | Tuple:
		"""
		Forward pass through the causal language modeling module.

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
		    MoeCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			input_embeds=input_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			init_cache=init_cache,
			deterministic=deterministic,
			return_dict=True,
			segment_ids=segment_ids,
		)
		logits = self.lm_head(outputs.last_hidden_state)

		aux_loss = sum(outputs[-1]) * self.config.router_aux_loss_coef
		if not return_dict:
			outputs = (logits,) + tuple(
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
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			all_router_losses=outputs.all_router_losses,
		)
