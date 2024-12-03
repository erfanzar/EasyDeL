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
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.tree_util
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, initializers
from flax.linen import partitioning as nn_partitioning
from flax.linen.module import compact
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array, lax
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm
from easydel.modules.chatglm.chatglm_configuration import ChatGLMConfig as ChatGLMConfig
from easydel.modules.factory import register_module

# easydel.modules
from easydel.modules.flax_modeling_utils import (
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
	with_sharding_constraint,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
)
from easydel.modules.modeling_utils import EasyDeLBaseModule


def flatten_axes(a: Array, start: int = 0, end: int = -1) -> Array:
	return a.reshape(a.shape[:start] + (-1,) + a.shape[end:][1:])


def split_tensor_along_last_dim(
	tensor: jax.Array,
	num_partitions: int,
	contiguous_split_chunks: bool = False,
) -> tuple[Array, ...] | list[Array]:
	"""Split a tensor along its last dimension.
	Arguments:
	    tensor: input tensor.
	    num_partitions: number of partitions to split the tensor
	    contiguous_split_chunks: If True, make each chunk contiguous
	                             in memory.
	Returns:
	    A list of Tensors
	"""
	# Get the size and dimension.
	last_dim = tensor.ndim - 1
	last_dim_size = tensor.shape[last_dim] // num_partitions
	# Split.
	tensor_list = jnp.split(tensor, last_dim_size, axis=last_dim)
	# Note: torch.split does not create contiguous tensors by default.
	if contiguous_split_chunks:
		return tuple(jax.lax.stop_gradient(chunk) for chunk in tensor_list)

	return tensor_list


def _normalize(
	mdl: nn.Module,
	x: Array,
	mean: Array,
	var: Array,
	reduction_axes,
	feature_axes,
	dtype,
	param_dtype,
	epsilon,
	use_bias,
	use_scale,
	bias_init,
	scale_init,
):
	reduction_axes = nn._canonicalize_axes(x.ndim, reduction_axes)
	feature_axes = nn._canonicalize_axes(x.ndim, feature_axes)
	feature_shape = [1] * x.ndim
	reduced_feature_shape = []
	for ax in feature_axes:
		feature_shape[ax] = x.shape[ax]
		reduced_feature_shape.append(x.shape[ax])

	mean = jnp.expand_dims(mean, reduction_axes)
	var = jnp.expand_dims(var, reduction_axes)
	y = x - mean
	mul = lax.rsqrt(var + epsilon)
	args = [x]
	if use_scale:
		scale = (
			(mdl.param("kernel", scale_init, reduced_feature_shape, param_dtype))
			.astype(param_dtype)
			.reshape(feature_shape)
		)
		mul *= scale
		args.append(scale)
	y *= mul
	if use_bias:
		bias = (
			(mdl.param("bias", bias_init, reduced_feature_shape, param_dtype))
			.astype(param_dtype)
			.reshape(feature_shape)
		)
		y += bias
		args.append(bias)
	dtype = jnp.dtypes.canonicalize_dtype(*args, dtype=dtype)
	return jnp.asarray(y, dtype)


class LayerNorm(nn.Module):
	eps: float = 1e-6
	dtype: Optional[jnp.dtype] = None
	param_dtype: jnp.dtype = jnp.float32
	use_bias: bool = True
	use_scale: bool = True
	bias_init: callable = initializers.zeros
	scale_init: callable = initializers.ones
	reduction_axes = -1
	feature_axes = -1
	axis_name: Optional[str] = None
	axis_index_groups: Any = None
	use_fast_variance: bool = True
	dim: Optional[int] = None

	@compact
	def __call__(self, x, *, mask: Optional[jax.Array] = None):
		"""Applies layer normalization on the input.

		Args:
		  x: the inputs
		  mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
		    the positions for which the mean and variance should be computed.

		Returns:
		  Normalized inputs (the same shape as inputs).
		"""
		mean, var = nn._compute_stats(
			x,
			self.reduction_axes,
			self.dtype,
			self.axis_name,
			self.axis_index_groups,
			use_fast_variance=self.use_fast_variance,
			mask=mask,
		)

		return _normalize(
			self,
			x,
			mean,
			var,
			self.reduction_axes,
			self.feature_axes,
			self.dtype,
			self.param_dtype,
			self.epsilon,
			self.use_bias,
			self.use_scale,
			self.bias_init,
			self.scale_init,
		)


class RotaryEmbedding(nn.Module):
	rope_ratio: float
	dim: int
	dtype: jnp.dtype = jnp.float32

	def setup(self) -> None:
		self.inv_freq = 1.0 / (
			10000 ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim)
		)

	def forward(self, seq_len: int, n_elem: int, base: int = 10000):
		base = base * self.rope_ratio
		theta = 1.0 / (base ** (jnp.arange(0, n_elem, 2, dtype=jnp.float32) / n_elem))
		seq_idx = jnp.arange(seq_len, dtype=jnp.float32)
		idx_theta = jnp.outer(seq_idx, theta).astype(jnp.float32)

		cache = jnp.stack([jnp.cos(idx_theta), jnp.sin(idx_theta)], axis=-1)

		if self.dtype in (jnp.float16, jnp.bfloat16, jnp.int8):
			cache = (
				cache.astype(jnp.bfloat16)
				if self.dtype == jnp.bfloat16
				else cache.astype(jnp.float16)
			)
		return cache


def apply_rotary_pos_emb(x: jax.Array, rope_cache: jax.Array) -> jax.Array:
	# x: [b, np, sq, hn]
	b, np, sq, hn = x.shape
	rot_dim = rope_cache.shape[-2] * 2
	x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
	rope_cache = rope_cache[:, :sq]
	xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
	rope_cache = rope_cache.reshape(-1, 1, sq, xshaped.shape[3], 2)
	x_out2 = jnp.stack(
		[
			xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
			xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
		],
		-1,
	)
	x_out2 = flatten_axes(x_out2, 3)
	return jnp.concatenate((x_out2, x_pass), axis=-1)


class CoreAttention(nn.Module):
	config: ChatGLMConfig
	layer_number: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		layer_number = self.layer_number
		config = self.config
		self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
		self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
		if self.apply_query_key_layer_scaling:
			self.attention_softmax_in_fp32 = True
		self.layer_number = max(1, layer_number)

		projection_size = config.kv_channels * config.num_attention_heads

		# Per attention head and per partition values.
		self.hidden_size_per_partition = projection_size
		self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
		self.num_attention_heads_per_partition = config.num_attention_heads

		coeff = None
		self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
		if self.apply_query_key_layer_scaling:
			coeff = self.layer_number
			self.norm_factor *= coeff
		self.coeff = coeff

		self.attention_dropout = nn.Dropout(config.attention_dropout)
		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=self.config.attention_dropout,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
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
		query_layer: jax.Array,
		key_layer: jax.Array,
		value_layer: jax.Array,
		attention_mask: jax.Array,
		causal_mask: jax.Array,
	):
		batch_size = query_layer.shape[0]
		causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
		mask = causal_mask
		if attention_mask is not None:
			if attention_mask.ndim == 2:
				attention_mask = jnp.expand_dims(attention_mask, (-3, -2))
			mask = jnp.logical_and(causal_mask, attention_mask)
		bias = lax.select(
			mask,
			jnp.full(mask.shape, 0, dtype=query_layer.dtype),
			jnp.full(mask.shape, jnp.finfo(query_layer.dtype).min, dtype=query_layer.dtype),
		)
		context_layer = self.attention_performer(
			query_layer,
			key_layer,
			value_layer,
			bias=bias,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
		).attention_outputs
		new_context_layer_shape = context_layer.reshape[:-2] + (
			self.hidden_size_per_partition,
		)
		context_layer = context_layer.reshape(*new_context_layer_shape)
		return context_layer


class FlaxChatGLMAttention(FlaxAttentionModule):
	config: ChatGLMConfig
	layer_number: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		layer_number = self.layer_number
		self.layer_number = max(1, layer_number)

		self.projection_size = config.kv_channels * config.num_attention_heads

		# Per attention head and per partition values.
		self.hidden_size_per_attention_head = (
			self.projection_size // config.num_attention_heads
		)
		self.num_attention_heads_per_partition = config.num_attention_heads

		self.multi_query_attention = config.multi_query_attention
		self.qkv_hidden_size = 3 * self.projection_size
		if self.multi_query_attention:
			self.num_multi_query_groups_per_partition = config.multi_query_group_num
			self.qkv_hidden_size = (
				self.projection_size
				+ 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
			)
		self.query_key_value = nn.Dense(
			self.qkv_hidden_size,
			use_bias=config.add_bias_linear or config.add_qkv_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		self.core_attention = CoreAttention(
			config,
			self.layer_number,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		# Output.
		self.dense = nn.Dense(
			config.hidden_size,
			use_bias=config.add_bias_linear,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.num_num_key_value_groupsreps = (
			self.num_attention_heads_per_partition
			// self.num_multi_query_groups_per_partition
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

	def apply_rotary(
		self,
		batch_size,
		sequence_length,
		query,
		key,
		value,
		frequencies,
		position_ids,
	):
		"""The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
		The main difference is that it takes in an additional argument, frequencies, which are used to calculate
		the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

		Args:
		    self: Access variables that belong to the class
		    batch_size: Reshape the query, key and value tensors
		    sequence_length: Reshape the query, key and value tensors
		    query: Calculate the attention weights
		    key: Calculate the attention
		    value: Compute the attention weights
		    frequencies: Calculate the frequency of each word in the
		        vocabulary
		    position_ids: Identify the position of each token in the
		        sequence

		Returns:
		    A tuple of 3 tensors: query, key and value
		"""

		query, key, value = self._transpose_sequence_head(query, key, value)
		query, key = self.rotary(
			position_ids=position_ids,
			query=query,
			key=key,
			frequencies=frequencies,
		)
		return self._transpose_sequence_head(query, key, value)

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
		fcm_mask=None,
	):
		"""
		The function takes various inputs related to attention mechanisms in a neural
		network model, processes them, applies attention calculations, and returns the
		output.

		Args:
		  hidden_states (chex.Array): The `hidden_states` parameter is expected to be a 3D
		array representing the input hidden states. The shape of this array should be
		`(batch_size, sequence_length, hidden_size)`, where `batch_size` is the number of
		sequences in the batch, `sequence_length` is the length
		  frequencies (Tuple[chex.Array, chex.Array]): The `frequencies` parameter is a tuple
		containing two arrays.
		  attention_mask (chex.Array): The `attention_mask` parameter is a binary mask
		indicating which positions should be attended to and which should not. It is used to
		prevent the model from attending to padding tokens or future tokens in the case of
		autoregressive models. The mask has a shape of `(batch_size, num_heads, query
		  position_ids (chex.Array): The `position_ids` parameter in the provided code
		represents the positional encoding for each token in the input sequence. It is used
		to provide positional information to the model during the self-attention mechanism.
		The positional encoding helps the model differentiate between tokens based on their
		position in the sequence, allowing the transformer to
		  causal_mask (chex.Array): The `causal_mask` parameter is a binary mask that
		prevents attention to future tokens during self-attention computation in transformer
		models. It is used to ensure that each token can only attend to previous tokens in
		the sequence, maintaining the autoregressive property of the model. The mask is
		typically a lower
		  segment_ids (Optional[chex.Array]): The `segment_ids` parameter in the provided
		code snippet is an optional input that represents the segment IDs for each token in
		the input sequence. Segment IDs are used in models that support segment-level
		attention, such as BERT for handling tasks like sentence pair classification where
		tokens from different segments need to be treated
		  deterministic (bool): The `deterministic` parameter in the function you provided
		is a boolean flag that indicates whether the computation should be deterministic or
		not. When `deterministic=True`, it means that the function should produce the same
		output given the same input every time it is called. This is useful for reproduc.
		Defaults to True
		  init_cache (bool): The `init_cache` parameter in the provided code snippet is a
		boolean flag that indicates whether to initialize the cache for key and value
		layers. If `init_cache` is set to `True`, it will trigger the initialization of the
		cache, otherwise, it will not. Defaults to False
		  output_attentions (bool): The `output_attentions` parameter is a boolean flag that
		determines whether the model should output attention weights along with the final
		attention output. If `output_attentions` is set to `True`, the model will return the
		attention weights in addition to the final attention output. If it is set to `.
		Defaults to False
		  fcm_mask: The `fcm_mask` parameter in the provided code snippet seems to be used
		as an optional argument. It is not explicitly defined in the function signature but
		is used in the `__call__` method.

		Returns:
		  a tuple containing the `attn_output` and `None` if `output_attentions` is set to
		`True`, otherwise it returns a tuple containing only the `attn_output`.
		"""
		batch_size, sequence_length = hidden_states.shape[:2]
		mixed_x_layer = self.query_key_value(hidden_states)

		if self.multi_query_attention:
			(query_layer, key_layer, value_layer) = mixed_x_layer.split(
				[
					self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
					self.num_multi_query_groups_per_partition
					* self.hidden_size_per_attention_head,
					self.num_multi_query_groups_per_partition
					* self.hidden_size_per_attention_head,
				],
				dim=-1,
			)
			query_layer = query_layer.view(
				query_layer.size()[:-1]
				+ (
					self.num_attention_heads_per_partition,
					self.hidden_size_per_attention_head,
				)
			)
			key_layer = key_layer.view(
				key_layer.size()[:-1]
				+ (
					self.num_multi_query_groups_per_partition,
					self.hidden_size_per_attention_head,
				)
			)
			value_layer = value_layer.view(
				value_layer.size()[:-1]
				+ (
					self.num_multi_query_groups_per_partition,
					self.hidden_size_per_attention_head,
				)
			)
		else:
			new_tensor_shape = mixed_x_layer.shape[:-1] + (
				self.num_attention_heads_per_partition,
				3 * self.hidden_size_per_attention_head,
			)
			mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

			# [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
			(query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
				mixed_x_layer, 3
			)
		query_layer, key_layer, value_layer = self.apply_rotary(
			query=query_layer,
			key=key_layer,
			value=value_layer,
			position_ids=position_ids,
			frequencies=frequencies,
			batch_size=batch_size,
			sequence_length=sequence_length,
		)

		query_length, key_length = query_layer.shape[1], key_layer.shape[1]

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
			key_layer, value_layer, attention_mask = self._concatenate_to_cache(
				key_layer, value_layer, query_layer, attention_mask
			)

		attn_output = self.core_attention(
			query_layer, key_layer, value_layer, attention_mask, causal_mask
		)
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
		attn_output = self.dense(attn_output)

		outputs = (attn_output, None) if output_attentions else (attn_output,)
		return outputs


class MLP(nn.Module):
	"""MLP.
	MLP will take the input with h hidden state, project it to 4*h
	hidden dimension, perform nonlinear transformation, and project the
	state back into h hidden dimension.
	"""

	config: ChatGLMConfig
	layer_number: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		self.add_bias = config.add_bias_linear

		# Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
		self.dense_h_to_4h = nn.Dense(
			config.ffn_hidden_size * 2,
			use_bias=self.add_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

		def swiglu(x):
			x = jnp.split(x, 2, axis=-1)
			return jax.nn.silu(x[0]) * x[1]

		self.activation_func = swiglu

		# Project back to h.
		self.dense_4h_to_h = nn.Dense(
			config.hidden_size,
			bias=self.add_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		hidden_states,
		e: bool = True,  # Ignore
	):
		"""
		This function takes hidden states as input, applies some transformations, and
		returns the final output.

		Args:
		  hidden_states: The `hidden_states` parameter in the code snippet represents the
		input hidden states that are passed to the neural network layer for processing.
		These hidden states are typically the output of the previous layer in a neural
		network model. The function processes these hidden states through a series of dense
		layers

		Returns:
		  the result of passing the `hidden_states` through a series of dense layers and
		activation functions. The final output is the result of passing the output of the
		`dense_h_to_4h` layer through an activation function and then through the
		`dense_4h_to_h` layer.
		"""
		return self.dense_4h_to_h(self.activation_func(self.dense_h_to_4h(hidden_states)))


class FlaxChatGLMBlock(nn.Module):
	config: ChatGLMConfig
	layer_number: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(
		self,
	):
		layer_number = self.layer_number
		config = self.config
		self.apply_residual_connection_post_layernorm = (
			config.apply_residual_connection_post_layernorm
		)

		self.fp32_residual_connection = config.fp32_residual_connection

		LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
		# Layernorm on the input data.
		self.input_layernorm = LayerNormFunc(
			config.hidden_size,
			eps=config.layernorm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		# Self attention.
		self.self_attention = FlaxChatGLMAttention(
			config=config,
			layer_number=layer_number,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.hidden_dropout = nn.Dropout(config.hidden_dropout)

		# Layernorm on the attention output
		self.post_attention_layernorm = LayerNormFunc(
			config.hidden_size,
			eps=config.layernorm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.input_layernorm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		attn_block = FlaxChatGLMAttention
		mlp_block = MLP
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = nn_partitioning.remat(
				attn_block,
				static_argnums=(3, 4, 6, 7, 8),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)

			mlp_block = nn_partitioning.remat(
				mlp_block,
				static_argnums=(1,),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)

		self.self_attention = attn_block(
			config=config,
			layer_number=layer_number,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.mlp = mlp_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
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
	):
		"""
		The function takes input hidden states and various masks, applies self-attention and
		MLP layers with residual connections, and returns the output.

		Args:
		  hidden_states (chex.Array): The `hidden_states` parameter in the code snippet you
		provided represents the input hidden states of the model. These hidden states are
		typically the output of the previous layer in a neural network model. In the context
		of the code snippet, `hidden_states` is passed through various layers and operations
		such as self
		  frequencies (Tuple[chex.Array, chex.Array]): The `frequencies` parameter is a tuple
		containing two arrays.
		  attention_mask (chex.Array): The `attention_mask` parameter is used to mask
		certain positions in the input so that the model does not attend to them during
		self-attention. It is a binary mask where the value of 1 indicates that the
		corresponding position should be masked and the value of 0 indicates that the
		position should not
		  position_ids (chex.Array): The `position_ids` parameter in the provided code
		snippet is used to represent the position IDs of tokens in the input sequence. It is
		of type `chex.Array` and is passed as an argument to the `__call__` method along
		with other parameters like `hidden_states`, `freq_c
		  causal_mask (chex.Array): The `causal_mask` parameter is used to mask out elements
		that come after the current position in the sequence during self-attention. It helps
		prevent the model from attending to future tokens during training. The mask is
		typically a lower triangular matrix where the elements above the main diagonal are
		masked out.
		  segment_ids (Optional[chex.Array]): The `segment_ids` parameter in the provided
		function is an optional input that represents the segment IDs for each token in the
		input sequence. Segment IDs are used in models that support multi-segment inputs,
		such as BERT for sentence pair tasks where each token is associated with a segment
		ID indicating which segment
		  deterministic (bool): The `deterministic` parameter in the function you provided
		is a boolean flag that indicates whether the computation should be deterministic or
		not. When `deterministic=True`, it means that the computation should be
		deterministic, i.e., the same input will always produce the same output. This can be
		useful. Defaults to True
		  init_cache (bool): The `init_cache` parameter in the function you provided is a
		boolean flag that indicates whether to initialize cache for the transformer layer.
		When `init_cache` is set to `True`, it means that the cache should be initialized,
		and when set to `False`, it means that the cache should not. Defaults to False

		Returns:
		  the final output after processing the input through self-attention, residual
		connections, layer normalization, and a multi-layer perceptron (MLP) network.
		"""
		layernorm_output = self.input_layernorm(hidden_states)
		# Self attention.

		# hidden_states: chex.Array,
		# frequencies: Tuple[chex.Array, chex.Array],
		# attention_mask: chex.Array,
		# position_ids: chex.Array,
		# causal_mask: chex.Array,
		# segment_ids: Optional[chex.Array] = None,
		# deterministic: bool = True,
		# init_cache: bool = False,
		# output_attentions: bool = False,
		# fcm_mask=None,
		attention_output = self.self_attention(
			layernorm_output,
			frequencies,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			deterministic,
			init_cache,
			False,
			None,
		)

		# Residual connection.
		if self.apply_residual_connection_post_layernorm:
			residual = layernorm_output
		else:
			residual = hidden_states

		layernorm_input = self.hidden_dropout(attention_output, deterministic=deterministic)
		layernorm_input = residual + layernorm_input

		# Layer norm post the self attention.
		layernorm_output = self.post_attention_layernorm(layernorm_input)

		# MLP.
		mlp_output = self.mlp(layernorm_output)

		# Second residual connection.
		if self.apply_residual_connection_post_layernorm:
			residual = layernorm_output
		else:
			residual = layernorm_input

		output = self.hidden_dropout(mlp_output, deterministic=deterministic)
		output = residual + output

		return output


class FlaxChatGLMPreTrainedModel(EasyDeLBaseModule):
	config_class = ChatGLMConfig
	base_model_prefix = "model"
	module_class: nn.Module = None

	def __init__(
		self,
		config: ChatGLMConfig,
		input_shape: Tuple = (1, 1),
		seed: int = 0,
		dtype: jnp.dtype = jnp.float32,
		_do_init: bool = True,
		**kwargs,
	):
		"""The __init__ function is called when the class is instantiated.
		It sets up the instance of the class, and defines what happens when it's created.
		The __init__ function can take arguments, but self is always required (it refers to the instance of the object).

		Args:
		    self: Refer to the object itself
		    config: ChatGLMConfig: Pass the configuration to the module
		    input_shape: Tuple: Specify the shape of the input to the
		        model
		    seed: int: Set the seed for random number generation
		    dtype: jnp.dtype: Specify the data type of the input
		    _do_init: bool: Control whether the module is initialized or
		        not
		    **kwargs: Pass in any additional parameters that the
		        module_class might need
		:param : Specify the number of layers in the network

		Returns:
		    The super() of the class
		"""
		module = self.module_class(config=config, dtype=dtype, **kwargs)
		super().__init__(
			config,
			module,
			input_shape=input_shape,
			seed=seed,
			dtype=dtype,
			_do_init=_do_init,
		)

	def init_weights(
		self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
	) -> FrozenDict:
		"""The init_weights function is used to initialize the weights of a model.

		Args:
		    self: Access variables that belong to the class
		    rng: jax.random.PRNGKey: Initialize the weights of the model
		    input_shape: Tuple: Specify the shape of the input tensor
		    params: FrozenDict: Pass in the parameters of a pre-trained
		        model

		Returns:
		    A frozendict of parameters
		"""
		input_ids = jnp.zeros(input_shape, dtype="i4")
		attention_mask = jnp.ones_like(input_ids)
		position_ids = jnp.broadcast_to(
			jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
		)
		params_rng, dropout_rng = jax.random.split(rng)
		rngs = {"params": params_rng, "dropout": dropout_rng}

		if self.config.add_cross_attention:
			encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
			encoder_attention_mask = attention_mask
			module_init_outputs = self.module.init(
				rngs,
				input_ids,
				attention_mask,
				position_ids,
				encoder_hidden_states,
				encoder_attention_mask,
				return_dict=False,
			)
		else:
			module_init_outputs = self.module.init(
				rngs, input_ids, attention_mask, position_ids, return_dict=False
			)

		random_params = module_init_outputs["params"]

		if params is not None:
			random_params = flatten_dict(unfreeze(random_params))
			params = flatten_dict(unfreeze(params))
			for missing_key in self._missing_keys:
				params[missing_key] = random_params[missing_key]
			self._missing_keys = set()
			return freeze(unflatten_dict(params))
		else:
			return random_params

	def init_cache(self, batch_size, max_length):
		"""The init_cache function is used to initialize the cache for a given batch size and sequence length.
		The cache is a dictionary that contains all the intermediate states from each layer in the model.
		This allows us to run inference on multiple batches without having to re-run forward passes through every layer in
		the model, which would be very slow.

		Args:
		    self: Access the module
		    batch_size: Define the batch size of the input tensors
		    max_length: Set the length of the input sequence

		Returns:
		    A dictionary with the following keys:
		"""

		return super().init_cache(batch_size=batch_size, max_length=max_length)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array = None,
		position_ids: chex.Array = None,
		params: dict = None,
		past_key_values: Optional[dict] = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
		add_params_field: bool = False,
		**kwargs,
	):
		"""The __call__ function is the main function of a JAX module.
		It takes in inputs and returns outputs, but it also has some other important features:
		- It can take in mutable state (e.g., past_key_values) that will be updated during the call and returned at the end.
		- It can take in random number generators (rngs) that are used to generate random numbers for dropout or sampling operations.

		Args:
		    self: Represent the instance of the class
		    input_ids: chex.Array: Pass in the input tokens
		    attention_mask: chex.Array: Mask out certain tokens in the
		        input
		    position_ids: chex.Array: Create the positional embeddings
		    params: dict: Pass in the parameters of the model
		    past_key_values: dict: Pass in the past key values from a
		        previous call to __call__
		    dropout_rng: jax.random.PRNGKey: Make sure that the dropout
		        is applied in a random way
		    train: bool: Determine whether to use dropout or not
		    output_attentions: Optional[bool]: Determine whether to
		        return the attention weights
		    output_hidden_states: Optional[bool]: Return the hidden
		        states of all layers
		    return_dict: Optional[bool]: Determine whether to return a
		        dictionary or not
		    extra_embedding: Optional[Union[jnp.ndarray,None]]: Pass in
		        the embedding for the input_ids
		    add_params_field: bool: Add the params field to the inputs
		        dictionary

		Returns:
		    A tuple of the following:
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

		batch_size, sequence_length = input_ids.shape

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		if position_ids is None:
			if past_key_values is not None:
				raise ValueError(
					"Make sure to provide `position_ids` when passing `past_key_values`."
				)

			position_ids = jnp.broadcast_to(
				jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
			)

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length))

		rngs = {}
		if dropout_rng is not None:
			rngs["dropout"] = dropout_rng

		if self.config.bits is not None:
			rngs["params"] = jax.random.key(0)

		inputs = (
			{"params": params or self.params} if add_params_field else params or self.params
		)

		if past_key_values is not None:
			inputs["cache"] = past_key_values
			mutable = ["cache"]
		else:
			mutable = False

		outputs = self.module.apply(
			inputs,
			jnp.array(input_ids, dtype="i4"),
			jnp.array(attention_mask, dtype="i4"),
			jnp.array(position_ids, dtype="i4"),
			not train,
			False,
			output_attentions,
			output_hidden_states,
			return_dict,
			extra_embedding,
			rngs=rngs,
			mutable=mutable,
		)

		if past_key_values is not None and return_dict:
			outputs, past_key_values = outputs
			outputs["past_key_values"] = unfreeze(past_key_values["cache"])
			return outputs
		elif past_key_values is not None and not return_dict:
			outputs, past_key_values = outputs
			outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

		return outputs


class FlaxChatGLMBlockCollection(nn.Module):
	config: ChatGLMConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.blocks = [
			FlaxChatGLMBlock(
				config=self.config,
				name=str(i),
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				layer_number=i + 1,
			)
			for i in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		"""The __call__ function is the main function of a JAX nn.Module.
		It defines how the module behaves when called as a function, and it's what you'll use to call your model
		 in training loops or inference scripts.
		The __call__ method should take all inputs that are necessary for computing outputs from the module,
		and return all outputs that are computed by this module.

		Args:
		    self: Represent the instance of the class
		    hidden_states: chex.Array: Pass the input tensor to the
		        encoder
		    frequencies: Tuple[chex.Array, chex.Array],: Pass in the
		        frequency of each token
		    attention_mask: chex.Array: Mask out certain tokens in the
		        input sequence
		    position_ids: chex.Array: Specify the position of each token
		        in a sequence
		    causal_mask: chex.Array: Mask the attention weights
		    deterministic: bool: Determine whether the model is in
		        training or evaluation mode
		    init_cache: bool: Initialize the cache for each layer
		    output_attentions: bool: Determine whether to output the
		        attention weights
		    output_hidden_states: bool: Determine whether to return the
		        hidden states of each layer
		    return_dict: bool: Return a dictionary of the outputs
		:param : Determine whether to use the forgetful causal mask

		Returns:
		    A tuple of 3 values
		"""
		all_hidden_states = () if output_hidden_states else None

		for block in self.blocks:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				frequencies=frequencies,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=causal_mask,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
			)
			hidden_states = layer_outputs[0]

		outputs = (hidden_states, all_hidden_states)

		return outputs


class FlaxChatGLMTransformer(nn.Module):
	config: ChatGLMConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		self.fp32_residual_connection = config.fp32_residual_connection
		self.post_layer_norm = config.post_layer_norm

		# Number of layers.
		self.num_layers = config.num_layers

		self.layers = FlaxChatGLMBlockCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		if self.post_layer_norm:
			LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
			# Final layer norm before output.
			self.final_layernorm = LayerNormFunc(
				config.hidden_size,
				eps=config.layernorm_epsilon,
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
		deterministic: bool = True,
		init_cache: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		all_self_attentions = None
		all_hidden_states = () if output_hidden_states else None
		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		hidden_states, hs = self.layers(
			hidden_states=hidden_states,
			frequencies=frequencies,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=causal_mask,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=False,
			output_hidden_states=output_hidden_states,
		)
		# Final layer norm.
		if self.post_layer_norm:
			hidden_states = self.final_layernorm(hidden_states)

		if hs is not None:
			for h in hs:
				all_hidden_states += (h,)
			all_hidden_states += hidden_states
		return hidden_states, all_hidden_states, all_self_attentions


@register_module(
	"base-module",
	config=ChatGLMConfig,
	model_type="glm",
	embedding_layer_names=["embedding"],
)
class FlaxChatGLMModel(nn.Module):
	config: ChatGLMConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		self.embedding = nn.Embed(
			num_embeddings=config.padded_vocab_size,
			features=config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.num_layers = config.num_layers
		self.multi_query_group_num = config.multi_query_group_num
		self.kv_channels = config.kv_channels

		# Rotary positional embeddings
		self.seq_length = config.seq_length
		rotary_dim = (
			config.hidden_size // config.num_attention_heads
			if config.kv_channels is None
			else config.kv_channels
		)

		self.rotary_pos_emb = RotaryEmbedding(
			dim=rotary_dim // 2,
			rope_ratio=config.rope_ratio,
			dtype=self.dtype,
		)
		self.encoder = FlaxChatGLMTransformer(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.output_layer = nn.Dense(
			config.padded_vocab_size,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.causal_mask = nn.make_causal_mask(
			jnp.ones((1, self.config.seq_length), dtype="bool"), dtype="bool"
		)

	def __call__(
		self,
		input_ids,
		position_ids: Optional[jax.Array] = None,
		attention_mask: Optional[jax.Array] = None,
		input_embeds: Optional[jax.Array] = None,
		init_cache: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		deterministic: bool = True,
	):
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		batch_size, seq_length = input_ids.shape

		if input_embeds is None:
			input_embeds = self.embedding(input_ids)
		# Rotary positional embeddings
		rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
		if position_ids is not None:
			rotary_pos_emb = rotary_pos_emb[position_ids]
		else:
			rotary_pos_emb = rotary_pos_emb[None, :seq_length]

		# Run encoder.
		hidden_states, all_hidden_states, all_self_attentions = self.encoder(
			attention_mask=attention_mask,
			causal_mask=self.causal_mask,
			hidden_states=input_embeds,
			frequencies=rotary_pos_emb,
			deterministic=deterministic,
			init_cache=init_cache,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			position_ids=position_ids,
		)

		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					all_hidden_states,
					all_self_attentions,
				]
				if v is not None
			)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
		)
