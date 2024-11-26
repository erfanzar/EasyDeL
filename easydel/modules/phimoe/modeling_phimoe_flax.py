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
import flax.linen.partitioning
import jax.lax
from chex import Array
from flax import linen as nn
from flax.linen import Dense
from flax.linen import partitioning as nn_partitioning
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm as RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
	with_sharding_constraint,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module
from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig as PhiMoeConfig

re_mat = nn_partitioning.remat


class FlaxPhiMoEBlockSparseTop2MLP(nn.Module):
	config: PhiMoeConfig
	layer_idx: Optional[int] = None
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		dense_class = functools.partial(
			nn.Dense,
			kernel_init=nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			use_bias=False,
		)
		self.ffn_dim = self.config.intermediate_size
		self.hidden_dim = self.config.hidden_size

		self.w1 = dense_class(self.ffn_dim)
		self.w2 = dense_class(self.hidden_dim)
		self.w3 = dense_class(self.ffn_dim)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(
		self,
		hidden_states: Array,
		deterministic: bool = False,  # noqa
	) -> Array:
		return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class FlaxPhiMoEAttention(FlaxAttentionModule):
	"""
	FlaxPhiMoEAttention implements an attention mechanism with rotary embeddings.

	Attributes:
	    config (PhiMoeConfig): Configuration for the attention module.
	    dtype (jnp.dtype): Data type for computations (default is jnp.float32).
	    param_dtype (jnp.dtype): Data type for parameters (default is jnp.float32).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: PhiMoeConfig
	layer_idx: Optional[int] = None
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		config = self.config
		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings
		self.original_max_position_embeddings = config.rope_scaling.get(
			"original_max_position_embeddings", None
		)
		self.rope_theta = config.rope_theta
		self.rope_scaling = config.rope_scaling
		self.is_causal = True

		if (self.head_dim * self.num_heads) != self.hidden_size:
			raise ValueError(
				f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
				f" and `num_heads`: {self.num_heads})."
			)

		dense_class = functools.partial(
			Dense,
			use_bias=config.attention_bias,
			precision=self.precision,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits),
		)

		self.q_proj = dense_class(self.num_heads * self.head_dim)
		self.k_proj = dense_class(self.num_key_value_heads * self.head_dim)
		self.v_proj = dense_class(self.num_key_value_heads * self.head_dim)
		self.o_proj = dense_class(self.hidden_size)
		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_key_value_heads,
			attention_dropout=self.config.attention_dropout,
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			base_config=self.config,
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=config.hidden_size // config.num_attention_heads,
			rotary_dim=config.hidden_size // config.num_attention_heads,
			is_neox_style=True,
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
		(query_states, key_states, value_states) = (
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

		attn_output = self._merge_heads(attentions.attention_outputs)
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
		attn_output = self.o_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class FlaxPhiMoeBlocKSparesTop2MLPCollection(nn.Module):
	config: PhiMoeConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		config = self.config
		self.layers = [
			FlaxPhiMoEBlockSparseTop2MLP(
				config=config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_local_experts)
		]

		self.router_jitter_noise = config.router_jitter_noise
		self.input_jitter_noise = config.input_jitter_noise

	def __call__(
		self,
		selected_experts: chex.Array,
		hidden_states: chex.Array,
		routing_weights: chex.Array,
		batch_size: int,
		sequence_length: int,
		hidden_dim: int,
		deterministic: bool,
	) -> chex.Array:
		if not deterministic and self.input_jitter_noise > 0:
			final_hidden_state = jax.nn.initializers.uniform(
				1.0 - self.input_jitter_noise,
				1.0 + self.input_jitter_noise,
			)(self.make_rng(), hidden_states.shape, hidden_states.dtype)
		else:
			final_hidden_state = jnp.zeros_like(hidden_states)
		for index in range(self.config.num_local_experts):
			expert_layer_output = (
				block_wise_ffn(
					self.layers[index],
					hidden_states,
					self.config.scan_mlp_chunk_size,
					deterministic,
				)
				if self.config.use_scan_mlp
				else self.layers[index](hidden_states, deterministic)
			)
			expert_layer_output_exp = (
				jnp.sum(jnp.multiply(selected_experts == index, routing_weights), axis=-1)[
					:, :, None
				]
				* expert_layer_output
			)
			final_hidden_state += expert_layer_output_exp

		return final_hidden_state


class FlaxPhiMoeSparseMoeBlock(nn.Module):
	"""This implementation is
	strictly equivalent to standard MoE with full capacity (no
	dropped tokens). It's faster since it formulates MoE operations
	in terms of block-sparse operations to accomodate imbalanced
	assignments of tokens to experts, whereas standard MoE either
	(1) drop tokens at the cost of reduced performance or (2) set
	capacity factor to number of experts and thus waste computation
	and memory on padding.
	"""

	config: PhiMoeConfig
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision("fastest")

	def setup(self) -> None:
		config = self.config
		self.hidden_dim = config.hidden_size
		self.ffn_dim = config.intermediate_size
		self.num_experts = config.num_local_experts
		self.top_k = config.num_experts_per_tok
		self.gate = Dense(
			self.config.num_local_experts,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=nn.initializers.normal(),
		)

		self.experts = FlaxPhiMoeBlocKSparesTop2MLPCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		deterministic: bool = False,
	) -> Tuple[chex.Array, chex.Array]:
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

		return (
			self.experts(
				selected_experts=selected_experts,
				batch_size=batch_size,
				sequence_length=sequence_length,
				hidden_dim=hidden_dim,
				hidden_states=hidden_states,
				routing_weights=routing_weights,
				deterministic=deterministic,
			),
			router_logits,
		)


class FlaxPhiMoeDecoderLayer(nn.Module):
	config: PhiMoeConfig
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self):
		attn_block = FlaxPhiMoEAttention
		mlp_block = FlaxPhiMoeSparseMoeBlock
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = re_mat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(3, 4, 6, 7, 9),
			)
			mlp_block = re_mat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)
		self.self_attn = attn_block(
			config=self.config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.block_sparse_moe = mlp_block(
			config=self.config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.input_layernorm = nn.LayerNorm(
			epsilon=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=True,
		)

		self.post_attention_layernorm = nn.LayerNorm(
			epsilon=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=True,
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
		    output_router_logits (bool): If True, outputs router logits.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)

		attn_out = self.self_attn(
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
		hidden_states, self_attn_weights = (
			(attn_out[0], attn_out[1]) if len(attn_out) == 2 else (attn_out[0], None)
		)

		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states, router_logits = self.block_sparse_moe(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if output_router_logits:
			outputs += (router_logits,)
		return outputs


class FlaxPhiDecoderLayerCollection(nn.Module):
	"""
	FlaxPhiMoeDecoratorCollection represents a single layer in a Transformer-like model,
	incorporating self-attention and MLP.

	Attributes:
	    config (PhiMoeConfig): Configuration object containing model parameters.
	    dtype (jnp.dtype): Data type for computations (default is jnp.float32).
	    param_dtype (jnp.dtype): Data type for model parameters (default is jnp.float32).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	config: PhiMoeConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.layers = [
			FlaxPhiMoeDecoderLayer(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(idx),
				layer_idx=idx,
			)
			for idx in range(self.config.num_hidden_layers)
		]
		self._frequencies = self.config.get_basic_frequencies(
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
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
		    frequencies (Tuple[chex.Array, chex.Array]): Frequency positional encodings.
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
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
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
		for decoder_layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = decoder_layer(
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
				all_self_attns += (layer_outputs[1],)

		return hidden_states, all_hidden_states, all_self_attns


@register_module(
	"base-module",
	config=PhiMoeConfig,
	model_type="phimoe",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["norm", "input_layernorm", "post_attention_layernorm"],
)
@wrap_easydel_module(config_class=PhiMoeConfig, base_model_prefix="model")
class FlaxPhiMoeModel(nn.Module):
	"""
	Core module of the PhiMoe model, including embedding, decoder layers, and normalization.

	Attributes:
	    config (PhiMoeConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: PhiMoeConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		config = self.config
		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.embed_dropout = flax.linen.Dropout(config.embd_pdrop)
		self.layers = FlaxPhiDecoderLayerCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm = nn.LayerNorm(
			epsilon=config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=True,
		)
		self.causal_mask = nn.make_causal_mask(
			jnp.ones(
				shape=(1, self.config.granted_mask_max_position_embedding),
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
		Forward pass through the PhiMoe module.

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

		if return_dict:
			return FlaxBaseModelOutput(
				last_hidden_state=hidden_states,
				hidden_states=outputs[1] if output_hidden_states else None,
				attentions=outputs[-1] if output_attentions else None,
			)

		return tuple(v for v in outputs if v is not None)


@register_module(
	"causal-language-model",
	config=PhiMoeConfig,
	model_type="phimoe",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["norm", "input_layernorm", "post_attention_layernorm"],
)
@wrap_easydel_module(config_class=PhiMoeConfig, base_model_prefix="model")
class FlaxPhiMoeForCausalLM(nn.Module):
	"""
	PhiMoe model for causal language modeling, including the language model head.

	Attributes:
	    config (PhiMoeConfig): Configuration object with model hyperparameters.
	    dtype (jnp.dtype): Data type for the computations.
	    param_dtype (jnp.dtype): Data type for the model parameters.
	    precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
	"""

	config: PhiMoeConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.model = FlaxPhiMoeModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.vocab_size = self.config.vocab_size
		self.lm_head = Dense(
			self.config.vocab_size,
			use_bias=self.config.lm_head_bias,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
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
		Forward pass through the PhiMoe module.

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
			return_dict=True,
			input_embeds=input_embeds,
			segment_ids=segment_ids,
		)

		if self.config.tie_word_embeddings:
			shared_kernel = self.model.variables["params"]["embed_tokens"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				outputs.last_hidden_state,
			)
		else:
			lm_logits = self.lm_head(outputs.last_hidden_state)

		if not return_dict:
			return (lm_logits,) + outputs[0:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
