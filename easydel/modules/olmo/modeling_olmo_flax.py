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
import typing as tp

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
	BaseModelOutput,
	CausalLMOutput,
	SequenceClassifierOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
	PagedAttentionCache,
	PagedAttentionCacheView,
	PagedAttentionMetadata,
	TransformerCache,
	TransformerCacheView,
	TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear

from .olmo_configuration import OlmoConfig


class OlmoMLP(nn.Module):
	"""OLMo MLP module.

	This module implements the feed-forward network (MLP) used in the OLMo model.
	It consists of gate, up, and down projections with a SiLU activation.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computations.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    gate_proj (ParallelLinear): Linear layer for the gate projection.
	    down_proj (ParallelLinear): Linear layer for the down projection.
	    up_proj (ParallelLinear): Linear layer for the up projection.
	    act_fn (callable): Activation function (SiLU).
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoMLP module.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.
		"""
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		linear_class = functools.partial(
			ParallelLinear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(
			config.hidden_size,
			config.intermediate_size,
			rngs=rngs,
		)
		self.down_proj = linear_class(
			config.intermediate_size,
			config.hidden_size,
			rngs=rngs,
		)
		self.up_proj = linear_class(
			config.hidden_size,
			config.intermediate_size,
			rngs=rngs,
		)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass of the OlmoMLP module.

		Args:
		    hidden_states (jnp.ndarray): Input hidden states.

		Returns:
		    jnp.ndarray: Output hidden states after MLP transformation.
		"""
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		return hidden_states


class OlmoAttention(AttentionModule):
	"""OLMo Attention module.

	This module implements the multi-head attention mechanism with rotary position embeddings
	and Grouped Query Attention (GQA) used in the OLMo model. It also supports optional
	QKV clipping.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computations.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    rngs (nn.Rngs): Random number generators.
	    hidden_size (int): Dimensionality of the hidden states.
	    head_dim (int): Dimensionality of each attention head.
	    num_key_value_groups (int): Number of query head groups for each key/value head.
	    q_proj (ParallelLinear): Linear layer for query projection.
	    k_proj (ParallelLinear): Linear layer for key projection.
	    v_proj (ParallelLinear): Linear layer for value projection.
	    o_proj (ParallelLinear): Linear layer for the output projection.
	    attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
	    rotary (RoPE): Rotary position embedding module.
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoAttention module.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.
		"""
		super().__init__(config=config)
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.hidden_size = config.hidden_size
		self.head_dim = self.config.hidden_size // self.config.num_attention_heads
		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_key_value_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_key_value_heads

		linear_class = functools.partial(
			ParallelLinear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.q_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear_class(
			config.num_attention_heads * self.head_dim,
			config.hidden_size,
			rngs=rngs,
		)

		self.attention_performer = FlexibleAttentionModule(
			dropout_prob=config.attention_dropout,
			base_config=config,
			softmax_scale=self.head_dim**-0.5,
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
			base=self.config.rope_theta,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: tp.Optional[chex.Array | bool],
		cache_view: tp.Optional[TransformerCacheView | PagedAttentionCacheView] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	):
		"""Forward pass of the OlmoAttention module.

		Args:
		    hidden_states (chex.Array): Input hidden states. Shape: (batch_size, sequence_length, hidden_size).
		    attention_mask (chex.Array): Mask to apply on the attention scores. Shape: (batch_size, 1, query_length, key_length).
		    position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
		    causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
		    cache_view (tp.Optional[TransformerCacheView | PagedAttentionCacheView]): Cache view for attention KVs.
		    cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    output_attentions (bool): Whether to return attention weights. Default is False.
		    fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
		    frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

		Returns:
		    tp.Union[tp.Tuple[chex.Array, chex.Array], tp.Tuple[chex.Array]]:
		        A tuple containing the attention output hidden states. If `output_attentions` is True,
		        it also includes the attention weights.
		"""
		batch_size, sequence_length = hidden_states.shape[:2]
		query_states, key_states, value_states = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)

		if self.config.clip_qkv is not None:
			query_states, key_states, value_states = map(
				lambda x: jnp.clip(x, min=-self.config.clip_qkv, max=self.config.clip_qkv),
				[query_states, key_states, value_states],
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

		(
			key_states,
			value_states,
			attention_mask,
			init_attention_bias,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=cache_view,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=fcm_mask,
		)

		attentions = self.attention_performer.forward(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=None,
			cache_metadata=cache_metadata,
			cache_view=cache_view,
			init_bias=init_attention_bias,
			attention_mask=attention_mask,
			segment_ids=segment_ids,
			causal=True,
			dropout_rng=self.rngs.params(),
		)

		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.o_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class OlmoDecoderLayer(nn.Module):
	"""OLMo Transformer Decoder Layer.

	This module represents a single decoder layer in the OLMo model,
	combining self-attention and MLP sub-layers with residual connections.
	Unlike typical transformer blocks, OLMo applies the layer normalization *after*
	the residual connection.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computations.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    rngs (nn.Rngs): Random number generators.
	    self_attn (OlmoAttention): The self-attention module.
	    mlp (OlmoMLP): The feed-forward (MLP) module.
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoDecoderLayer.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.
		"""
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		attn_block = OlmoAttention
		mlp_block = OlmoMLP

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			config=config,
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
		self.input_layernorm = nn.LayerNorm(
			config.hidden_size,
			epsilon=1e-5,
			use_bias=False,
			use_scale=False,
			rngs=rngs,
		)
		self.post_attention_layernorm = nn.LayerNorm(
			config.hidden_size,
			epsilon=1e-5,
			use_bias=False,
			use_scale=False,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: tp.Optional[chex.Array | bool],
		cache_view: tp.Optional[TransformerCacheView | PagedAttentionCacheView] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	):
		"""Forward pass of the OlmoDecoderLayer module.

		Args:
		    hidden_states (chex.Array): Input hidden states. Shape: (batch_size, sequence_length, hidden_size).
		    attention_mask (chex.Array): Mask to apply on the attention scores. Shape: (batch_size, 1, query_length, key_length).
		    position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
		    causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
		    cache_view (tp.Optional[TransformerCacheView | PagedAttentionCacheView]): Cache view for attention KVs.
		    cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    output_attentions (bool): Whether to return attention weights. Default is False.
		    fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
		    frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

		Returns:
		    tp.Tuple[chex.Array, tp.Optional[chex.Array]]:
		        A tuple containing the output hidden states and optionally the attention weights.
		"""
		residual = hidden_states
		attention_output = self.self_attn(
			self.input_layernorm(hidden_states),
			attention_mask,
			position_ids,
			causal_mask,
			cache_view,
			cache_metadata,
			segment_ids,
			output_attentions,
			fcm_mask,
			frequencies,
		)

		hidden_states = attention_output[0] + residual
		ffd_inp = self.post_attention_layernorm(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp, ffd_inp, self.config.scan_mlp_chunk_size
			)
		else:
			feed_forward_hidden_states = self.mlp(ffd_inp)

		hidden_states = hidden_states + feed_forward_hidden_states
		outputs = (hidden_states,)
		if output_attentions:
			outputs += (attention_output[1],)
		return outputs


@register_module(
	TaskType.BASE_MODULE,
	config=OlmoConfig,
	model_type="olmo",
)
class OlmoModel(EasyDeLBaseModule):
	"""The base OLMo model transformer.

	This class represents the core transformer architecture of the OLMo model,
	consisting of an embedding layer and multiple OlmoDecoderLayer layers.
	Note that OLMo does not have a final layer normalization.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computation.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    rngs (nn.Rngs): Random number generators.
	    embed_tokens (nn.Embed): Embedding layer for input tokens.
	    layers (tp.List[OlmoDecoderLayer]): List of decoder layers.
	    gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoModel.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.
		"""
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.embed_tokens = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			OlmoDecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(config.num_hidden_layers)
		]
		self.norm = nn.LayerNorm(
			config.hidden_size,
			epsilon=1e-5,
			use_bias=False,
			use_scale=False,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache | PagedAttentionCache] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[BaseModelOutput, tp.Tuple]:
		"""Forward pass of the OlmoModel.

		Args:
		    input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
		    inputs_embeds (tp.Optional[chex.Array]): Input embeddings. Shape: (batch_size, sequence_length, hidden_size).
		        Either `input_ids` or `inputs_embeds` must be provided.
		    attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
		        Shape: (batch_size, sequence_length).
		    position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
		        Shape: (batch_size, sequence_length).
		    segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
		    past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]): Precomputed key/value states for attention.
		    cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
		    output_attentions (tp.Optional[bool]): Whether to return attention weights. Defaults to `config.output_attentions`.
		    output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
		        Defaults to `config.output_hidden_states`.
		    return_dict (bool): Whether to return a `BaseModelOutput` object or a tuple.

		Returns:
		    tp.Union[BaseModelOutput, tp.Tuple]: The model's output. If `return_dict` is True,
		        returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
		        and `attentions` (optional). Otherwise, returns a tuple with these elements.

		Raises:
		    ValueError: If neither `input_ids` nor `inputs_embeds` is provided.
		"""
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		assert sequence_length <= self.config.max_position_embeddings, (
			f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		)
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "b1")
		else:
			if attention_mask.dtype != jnp.bool:
				attention_mask = jnp.astype(attention_mask == 1, "b1")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)

		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_view=past_key_values.views[idx],
				cache_metadata=cache_metadata,
				causal_mask=self.causal_mask,
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
		else:
			outputs = (hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return BaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	TaskType.CAUSAL_LM,
	config=OlmoConfig,
	model_type="olmo",
)
class OlmoForCausalLM(EasyDeLBaseModule):
	"""OLMo model with a Causal Language Modeling head.

	This model consists of the base OLMo transformer (`OlmoModel`) followed by a
	linear layer (`lm_head`) that projects the transformer's output hidden states
	to the vocabulary size, producing logits for next token prediction.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computation.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    rngs (nn.Rngs): Random number generators.
	    transformer (OlmoModel): The core OLMo transformer model.
	    lm_head (ParallelLinear): The linear layer for projecting hidden states to vocabulary logits.
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoForCausalLM model.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.
		"""
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.model = OlmoModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.lm_head = ParallelLinear(
			config.hidden_size,
			config.vocab_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache | PagedAttentionCache] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[CausalLMOutput, tp.Tuple]:
		"""Forward pass of the OlmoForCausalLM model.

		Args:
		    input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
		    inputs_embeds (tp.Optional[chex.Array]): Input embeddings. Shape: (batch_size, sequence_length, hidden_size).
		        Either `input_ids` or `inputs_embeds` must be provided.
		    attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
		        Shape: (batch_size, sequence_length).
		    position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
		        Shape: (batch_size, sequence_length).
		    segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
		    past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]): Precomputed key/value states for attention.
		    cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
		    output_attentions (tp.Optional[bool]): Whether to return attention weights. Defaults to `config.output_attentions`.
		    output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
		        Defaults to `config.output_hidden_states`.
		    return_dict (bool): Whether to return a `CausalLMOutput` object or a tuple.

		Returns:
		    tp.Union[CausalLMOutput, tp.Tuple]: The model's output. If `return_dict` is True,
		        returns a `CausalLMOutput` object containing `logits`, `hidden_states` (optional),
		        and `attentions` (optional). Otherwise, returns a tuple with these elements.
		"""
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			cache_metadata=cache_metadata,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		if self.config.tie_word_embeddings:
			lm_logits = jax.lax.dot_general(
				hidden_states,
				self.model.embed_tokens.embedding.value.T,
				(((hidden_states.ndim - 1), (0,)), ((), ())),
			)
		else:
			lm_logits = self.lm_head(hidden_states)
		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return CausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)


@register_module(
	TaskType.SEQUENCE_CLASSIFICATION,
	config=OlmoConfig,
	model_type="olmo",
)
class OlmoForSequenceClassification(EasyDeLBaseModule):
	"""OLMo model with a Sequence Classification head.

	This model consists of the base OLMo transformer (`OlmoModel`) followed by a
	linear layer (`score`) that projects the transformer's output hidden states
	(typically the hidden state of the last token) to the number of classes for classification.

	Attributes:
	    config (OlmoConfig): Configuration object for the model.
	    dtype (jnp.dtype): Data type for computation.
	    param_dtype (jnp.dtype): Data type for parameters.
	    precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
	    rngs (nn.Rngs): Random number generators.
	    transformer (OlmoModel): The core OLMo transformer model.
	    score (ParallelLinear): The linear layer for classification.
	"""

	def __init__(
		self,
		config: OlmoConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		"""Initializes the OlmoForSequenceClassification model.

		Args:
		    config (OlmoConfig): The configuration object for the OLMo model.
		        Must include `num_labels`.
		    dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
		    param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
		    precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
		    rngs (nn.Rngs): Random number generators.

		Raises:
		    AssertionError: If `config.num_labels` is not defined.
		"""
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.model = OlmoModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		assert hasattr(config, "num_labels"), (
			"in order to use `SequenceClassification` Models in `EasyDeL` you first need to attach `num_labels` to model `config`"
		)
		self.score = ParallelLinear(
			self.config.hidden_size,
			config.num_labels,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			precision=self.precision,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache | PagedAttentionCache] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[SequenceClassifierOutput, tp.Tuple]:
		"""Forward pass of the OlmoForSequenceClassification model.

		Args:
		    input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
		    inputs_embeds (tp.Optional[chex.Array]): Input embeddings. Shape: (batch_size, sequence_length, hidden_size).
		        Either `input_ids` or `inputs_embeds` must be provided.
		    attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
		        Shape: (batch_size, sequence_length).
		    position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
		        Shape: (batch_size, sequence_length).
		    segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
		    past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]): Precomputed key/value states for attention.
		    cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
		    output_attentions (tp.Optional[bool]): Whether to return attention weights. Defaults to `config.output_attentions`.
		    output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
		        Defaults to `config.output_hidden_states`.
		    return_dict (bool): Whether to return a `SequenceClassifierOutput` object or a tuple.

		Returns:
		    tp.Union[SequenceClassifierOutput, tp.Tuple]: The model's output. If `return_dict` is True,
		        returns a `SequenceClassifierOutput` object containing `logits`, `hidden_states` (optional),
		        and `attentions` (optional). Otherwise, returns a tuple with these elements.

		Raises:
		    ValueError: If `config.pad_token_id` is None and `batch_size > 1`.
		"""
		transformer_outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			cache_metadata=cache_metadata,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
		)

		hidden_states = transformer_outputs[0]
		logits = self.score(hidden_states)
		if input_ids is not None:
			batch_size = input_ids.shape[0]
		else:
			batch_size = inputs_embeds.shape[0]

		if self.config.pad_token_id is None and batch_size != 1:
			raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
		if self.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				sequence_lengths = (
					jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1)
					- 1
				)
				sequence_lengths = sequence_lengths % input_ids.shape[-1]
			else:
				sequence_lengths = -1

		pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

		if not return_dict:
			output = (pooled_logits,) + transformer_outputs[1:]
			return output

		return SequenceClassifierOutput(
			logits=pooled_logits,
			past_key_values=past_key_values,
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
