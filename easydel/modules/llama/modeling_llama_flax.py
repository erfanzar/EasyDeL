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
from typing import List, Optional, Tuple, Union
import flax.nnx as nn
import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec
from easydel.layers.rotary_embedding import get_rope
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.cache_view import AttentionCacheView
from easydel.layers.norms import RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
	with_sharding_constraint,
)

# easydel.modules
from easydel.modules.modeling_utils import EasyDeLBaseModule
from easydel.modules.llama.llama_configuration import LlamaConfig as LlamaConfig
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
)


class LlamaAttention(nn.Module):
	"""
	FlaxLlamaAttention implements an attention mechanism with rotary embeddings.

	Attributes:
	    config (LlamaConfig): Configuration for the attention module.
	    dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
	    param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
	    precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.hidden_size = config.hidden_size
		default_dim = config.hidden_size // config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", default_dim)
		self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
		kernel_range = config.initializer_range / np.sqrt(config.hidden_size)
		if self.num_key_value_groups == 1:
			assert config.num_attention_heads == config.num_key_value_heads

		self.q_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.num_attention_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.attention_bias,
			kernel_init=jax.nn.initializers.normal(kernel_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.k_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.num_key_value_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.attention_bias,
			kernel_init=jax.nn.initializers.normal(kernel_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.v_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.num_key_value_heads * self.head_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.attention_bias,
			kernel_init=jax.nn.initializers.normal(kernel_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.o_proj = nn.Linear(
			in_features=config.num_attention_heads * self.head_dim,
			out_features=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.attention_bias,
			kernel_init=jax.nn.initializers.normal(kernel_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		rope_scaling = None
		if config.rope_scaling is not None:
			original_max_position_embeddings = config.rope_scaling.get(
				"original_max_position_embeddings", None
			)
			rope_scaling = dict(
				rope_type=config.rope_scaling.get("type", config.rope_scaling.get("rope_type")),
				factor=config.rope_scaling.get("factor"),
				low_freq_factor=config.rope_scaling.get("low_freq_factor", None),
				high_freq_factor=config.rope_scaling.get("high_freq_factor", None),
				original_max_position_embeddings=original_max_position_embeddings,
			)

		self.rotary = get_rope(
			head_size=self.head_dim,
			rotary_dim=self.head_dim,
			base=self.config.rope_theta,
			max_position=self.config.granted_freq_max_position_embedding,
			is_neox_style=True,
			rope_scaling=rope_scaling,
		)

		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=self.config.attention_dropout,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_key_value_heads,
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
		self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

	def _merge_heads(self, hidden_states):
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (chex.Array): The hidden states with separate head dimensions.

		Returns:
		    chex.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		cache_view: Optional[AttentionCacheView] = None,
		output_attentions: bool = False,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
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
			positions=position_ids,
			query=query_states,
			key=key_states,
		)

		query_length, key_length = query_states.shape[1], key_states.shape[1]

		if cache_view is not None:
			key_states, value_states, attention_mask = cache_view.concatenate_to_cache(
				query_states,
				key_states,
				value_states,
				attention_mask,
			)

		attention_bias = None
		if attention_mask is not None:
			attention_mask = attention_mask[:, :, :, :key_length]
			attention_bias = lax.select(
				attention_mask > 0,
				jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
				jnp.full(
					attention_mask.shape,
					jnp.finfo(self.dtype).min,
				).astype(self.dtype),
			)

		query_length, key_length = query_states.shape[1], key_states.shape[1]

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=self.rngs(),
			deterministic=self.resid_dropout.deterministic,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=cache_view is not None,
			segment_ids=segment_ids,
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
		attn_output = self.resid_dropout(attn_output)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class LlamaMLP(nn.Module):
	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		init_range = config.initializer_range / np.sqrt(config.hidden_size)
		oinit_range = config.initializer_range / np.sqrt(config.intermediate_size)
		self.gate_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.intermediate_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.mlp_bias,
			kernel_init=jax.nn.initializers.normal(init_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.down_proj = nn.Linear(
			in_features=config.intermediate_size,
			out_features=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.mlp_bias,
			kernel_init=jax.nn.initializers.normal(oinit_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.up_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.intermediate_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.mlp_bias,
			kernel_init=jax.nn.initializers.normal(init_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""The __call__ function is the main function of a class.
		It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
		The __call__ method enables instances of a class to be called like standard Python functions.

		Args:
		    self: Represent the instance of the class
		    x: jnp.ndarray: Pass in the input to the layer
		    deterministic: bool: Determine whether to use dropout

		Returns:
		    A tensor that is the result of applying a dropout function
		    to x
		"""

		x = control_mlp_sharding(x, self.config.partition_axis)
		return self.dropout(
			self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		)


class LlamaBlock(nn.Module):
	def __init__(
		self,
		config: LlamaConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		attn_block = LlamaAttention
		mlp_block = LlamaMLP
		if config.gradient_checkpointing != "":
			attn_block = nn.remat(
				attn_block,
				static_argnums=(1, 4, 5),
				policy=get_gradient_checkpoint_policy(config.gradient_checkpointing),
			)
			mlp_block = nn.remat(
				mlp_block,
				static_argnums=(),
				policy=get_gradient_checkpoint_policy(config.gradient_checkpointing),
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
		self.input_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		self.post_attention_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		cache_view: Optional[AttentionCacheView] = None,
		output_attentions: bool = False,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
				cache_view (Optional(AttentionCacheView))): Past key and values used for generation
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.

		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		attn_outputs = self.self_attn(
			self.input_layernorm(hidden_states),
			attention_mask,
			position_ids,
			segment_ids,
			cache_view,
			output_attentions,
		)
		attn_output = attn_outputs[0]
		hidden_states = hidden_states + attn_output

		feed_forward_input = self.post_attention_layernorm(hidden_states)

		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp,
				feed_forward_input,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.mlp(feed_forward_input)

		hidden_states = hidden_states + feed_forward_hidden_states

		return (hidden_states,) + attn_outputs[1:]


@register_module(
	"base-module",
	config=LlamaConfig,
	model_type="llama",
	embedding_layer_names=["embed_tokens"],
)
class LlamaModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.embed_tokens = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			embedding_init=nn.initializers.normal(stddev=config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(
			rate=config.embd_pdrop,
			rngs=rngs,
		)
		self.layers = [
			LlamaBlock(
				config=config,
				layer_idx=layer_idx,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for layer_idx in range(config.num_hidden_layers)
		]
		self.norm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
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
		cache_views: Optional[List[AttentionCacheView]] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
		extra_embedding: Optional[jnp.ndarray] = None,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
		and returns the output of the model. These optional arguments are passed as keyword arguments when calling a Flax model.

		Args:
				self: Represent the instance of the class
				input_ids: chex.Array: Pass in the input token ids
				attention_mask: (Optional(chex.Array)): Mask out the padding tokens
				position_ids: (Optional(chex.Array)): Indicate the position of each token in a sequence
				segment_ids: (Optional(chex.Array)): Determine the Segment.
				input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
				cache_view: (Optional(List[AttentionCacheView])): Past key and values used for generation
				output_attentions: bool: Determine whether to return the attentions or not
				output_hidden_states: bool: Determine whether to return hidden states
				return_dict: bool: Return a dictionary of the output or not
				extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

		Returns:
				The logits and the hidden states
		"""
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		if input_ids is not None and input_embeds is not None:
			raise ValueError(
				"You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
			)
		if input_embeds is None and input_ids is not None:
			input_embeds = self.embed_tokens(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
		batch_size, sequence_length, _ = input_embeds.shape
		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)
		if attention_mask.ndim == 2:
			attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
			attention_mask = jnp.logical_and(
				attention_mask, self.causal_mask[:, :, :sequence_length, :]
			)

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		hidden_states = (
			input_embeds + extra_embedding if extra_embedding is not None else input_embeds
		)
		hidden_states = self.dropout(hidden_states)
		if cache_views is None:
			cache_views = [None] * self.config.num_hidden_layers
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			hidden_states, attn_weight = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				segment_ids=segment_ids,
				cache_view=cache_views[idx],
				output_attentions=output_attentions,
			)

			if output_attentions:
				all_attentions += (attn_weight,)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (hidden_states, all_hidden_states, all_attentions)
		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


@register_module(
	"causal-language-model",
	config=LlamaConfig,
	model_type="llama",
	embedding_layer_names=["embed_tokens"],
)
class LlamaForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.model = LlamaModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.lm_head = nn.Linear(
			in_features=config.hidden_size,
			out_features=config.vocab_size,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(
				stddev=config.initializer_range / np.sqrt(config.hidden_size)
			),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		cache_views: Optional[List[AttentionCacheView]] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
		extra_embedding: Optional[jnp.ndarray] = None,
	) -> Union[FlaxCausalLMOutput, Tuple]:
		"""
		Forward pass through the Llama module.

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
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			input_embeds=input_embeds,
			segment_ids=segment_ids,
			cache_views=cache_views,
			extra_embedding=extra_embedding,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
			lm_logits = self.lm_head(hidden_states)
		else:
			lm_logits = self.lm_head(hidden_states)

		lm_logits = lm_logits.astype(jnp.float32)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


@register_module(
	"sequence-classification",
	config=LlamaConfig,
	model_type="llama",
	embedding_layer_names=["embed_tokens"],
)
class LlamaForSequenceClassification(EasyDeLBaseModule):
	def __init__(
		self,
		num_classes: int,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.model = LlamaModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.classifier = nn.Linear(
			config.hidden_size,
			num_classes,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=nn.initializers.normal(stddev=config.initializer_range),
			precision=precision,
		)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array = None,
		position_ids: chex.Array = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
		extra_embedding: Optional[jnp.ndarray] = None,
	):
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			extra_embedding=extra_embedding,
			past_key_values=None,
		)

		hidden_states = outputs[0]
		prediction = self.classifier(hidden_states)
		if return_dict:
			return FlaxSequenceClassifierOutput(
				logits=prediction, hidden_states=hidden_states
			)
		else:
			return (prediction,)
