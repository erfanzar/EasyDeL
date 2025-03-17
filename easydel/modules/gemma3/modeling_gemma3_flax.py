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


import typing as tp
from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
	ModelOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import float8s
from easydel.modules.auto.auto_modeling import AutoEasyDeLVisionModel
from easydel.utils import traversals as etr
from easydel.utils.helpers import get_logger

from .gemma3_configuration import Gemma3Config, Gemma3TextConfig

logger = get_logger(__name__)


@etr.auto_pytree
class Gemma3CausalLMOutputWithPast(ModelOutput):
	"""
	Base class for Gemma3 causal language model (or autoregressive) outputs.

	Args:
	    loss (`chex.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
	        Language modeling loss (for next-token prediction).
	    logits (`chex.Array` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
	        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	    past_key_values (`tuple(tuple(chex.Array))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

	        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
	        `past_key_values` input) to speed up sequential decoding.
	    hidden_states (`tuple(chex.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, +
	        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
	    attentions (`tuple(chex.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    image_hidden_states (`chex.Array`, *optional*):
	        A `chex.Array` of size `(batch_size, sequence_length, hidden_size)`.
	        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
	"""

	loss: tp.Optional[chex.Array] = None
	logits: chex.Array = None
	past_key_values: tp.Optional[TransformerCache] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array]] = None
	attentions: tp.Optional[tp.Tuple[chex.Array]] = None
	image_hidden_states: tp.Optional[chex.Array] = None


class Gemma3RMSNorm(nn.Module):
	def __init__(
		self,
		config: Gemma3TextConfig,
		param_dtype: jnp.dtype = jnp.float32,
		dim: tp.Optional[int] = None,
		epsilon: tp.Optional[float] = None,
	):
		self.config = config
		self.epsilon = self.config.rms_norm_eps if epsilon is None else epsilon
		self.param_dtype = param_dtype
		dim = self.config.hidden_size if dim is None else dim
		self.kernel = nn.Param(jnp.ones(dim, dtype=param_dtype))

	def _norm(self, x: jax.Array) -> jax.Array:
		return x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.epsilon))

	def __call__(self, hidden_states: jax.Array) -> jax.Array:
		variance = self._norm(hidden_states.astype(jnp.float32)).astype(self.param_dtype)
		out = (1 + self.kernel.value.astype(self.param_dtype)) * variance

		if out.dtype in float8s:
			out = out.astype(jnp.bfloat16)
		return out


class Gemma3Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: Gemma3TextConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		causal: bool = True,
		is_cross_attention: bool = False,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config)
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.is_cross_attention = is_cross_attention
		self.rngs = rngs
		self.causal = causal
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = getattr(
			config,
			"head_dim",
			config.hidden_size // config.num_attention_heads,
		)

		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads

		kernel = jax.nn.initializers.normal(config.initializer_range)

		linear = partial(
			nn.Linear,
			use_bias=config.attention_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=kernel,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.q_proj = linear(self.embed_dim, self.num_heads * self.head_dim)
		self.k_proj = linear(self.embed_dim, self.num_key_value_heads * self.head_dim)
		self.v_proj = linear(self.embed_dim, self.num_key_value_heads * self.head_dim)
		self.o_proj = linear(self.num_heads * self.head_dim, self.embed_dim)
		self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
		self.sliding_window = config.sliding_window if self.is_sliding else None

		self.q_norm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
			dim=self.head_dim,
		)
		self.k_norm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
			dim=self.head_dim,
		)

		self.attention_performer = FlexibleAttentionModule(
			base_config=config,
			softmax_scale=self.config.query_pre_attn_scalar**-0.5,
			dropout_prob=config.attention_dropout,
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
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		batch_size, sequence_length = hidden_states.shape[:2]
		input_shape = hidden_states.shape[:-1]
		hidden_shape = (*input_shape, -1, self.head_dim)

		(query_states, key_states, value_states) = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)

		query_states = query_states.reshape(*hidden_shape)
		key_states = key_states.reshape(*hidden_shape)
		value_states = value_states.reshape(*hidden_shape)

		query_states = self.q_norm(query_states)
		key_states = self.k_norm(key_states)

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
			token_type_ids=token_type_ids,
			fcm_mask=fcm_mask,
			sliding_windows=None,
		)
		if self.is_sliding:
			attention_mask = jnp.logical_and(
				self._create_sliding_mask(
					cache_pos=self.build_cache_pos(attention_mask, cache_view),
					curr_index=cache_view.index[0] if cache_view is not None else 0,
					cache_length=attention_mask.shape[-1],
					sliding_windows=self.sliding_window,
				),
				attention_mask,
			)

			def init_attention_bias():
				return jax.lax.select(
					attention_mask > 0,
					jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
					jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
				)

		attentions = self.attention_performer.forward(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=None,
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

		return (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)


class Gemma3MLP(nn.Module):
	def __init__(
		self,
		config: Gemma3TextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		embed_dim = self.config.hidden_size
		inner_dim = (
			self.config.intermediate_size
			if self.config.intermediate_size is not None
			else 4 * embed_dim
		)
		kernel_init = jax.nn.initializers.normal(config.initializer_range)

		self.act = ACT2FN[self.config.hidden_activation]

		linear_class = partial(
			nn.Linear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=kernel_init,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(embed_dim, inner_dim)
		self.down_proj = linear_class(inner_dim, embed_dim)
		self.up_proj = linear_class(embed_dim, inner_dim)

	def __call__(self, hidden_states):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		return hidden_states


class Gemma3DecoderLayer(nn.Module):
	def __init__(
		self,
		config: Gemma3TextConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		mlp_block = Gemma3MLP
		attn_block = Gemma3Attention

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)

		self.self_attn = attn_block(
			self.config,
			layer_idx=self.layer_idx,
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

		self.input_layernorm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
		)
		self.post_attention_layernorm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
		)
		self.pre_feedforward_layernorm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
		)
		self.post_feedforward_layernorm = Gemma3RMSNorm(
			self.config,
			param_dtype=self.param_dtype,
		)

		self.is_sliding = self.self_attn.is_sliding
		self.sliding_window = config.sliding_window

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
		default_frequencies: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		residual = hidden_states
		frequencies = default_frequencies if self.is_sliding else frequencies
		hidden_states = self.input_layernorm(hidden_states)
		hidden_states, attn_weight = self.self_attn(
			hidden_states,
			attention_mask,
			position_ids,
			causal_mask,
			cache_view,
			segment_ids,
			token_type_ids,
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
				self.mlp,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			hidden_states = self.mlp(hidden_states)

		hidden_states = self.post_feedforward_layernorm(hidden_states)
		hidden_states = residual + hidden_states
		return hidden_states, attn_weight


@register_module(
	TaskType.BASE_MODULE,
	config=Gemma3TextConfig,
	model_type="gemma3_text", 
)
class Gemma3TextModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: Gemma3TextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.hidden_size = self.config.hidden_size
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.layers = [
			Gemma3DecoderLayer(
				self.config,
				layer_idx=i,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.norm = Gemma3RMSNorm(self.config, param_dtype=self.dtype)

	@cached_property
	def default_frequencies(self):
		from easydel.infra.utils import ModuleCaches
		from easydel.layers.rotary_embedding import get_frequencies

		frequencies = get_frequencies(
			head_size=self.config.head_dim,
			rotary_dim=self.config.head_dim,
			max_position=self.config.granted_freq_max_position_embedding,
			base=self.config.rope_local_base_freq,
			rope_scaling=None,
		)

		return ModuleCaches(frequencies)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		"""
		Forward pass through the Gemma2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4")) * (
				self.config.hidden_size**0.5
			)
		batch_size, sequence_length, _ = inputs_embeds.shape

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "b1")
		else:
			if attention_mask.dtype != jnp.bool:
				attention_mask = jnp.astype(attention_mask == 1, "b1")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			)
		inputs_embeds = inputs_embeds
		assert sequence_length <= self.config.max_position_embeddings, (
			f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		)
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))
		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		causal_mask = self.causal_mask

		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_view=past_key_values.views[idx],
				causal_mask=causal_mask,
				token_type_ids=token_type_ids,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
				default_frequencies=self.default_frequencies,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (hidden_states, all_hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	TaskType.CAUSAL_LM,
	config=Gemma3TextConfig,
	model_type="gemma3_text", 
)
class Gemma3ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: Gemma3TextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		if param_dtype == jnp.float16 or param_dtype == "f2":
			logger.error(
				"Gemma-3's recommended dtype is bfloat16, but you are using float16. "
				"This may result in junk responses or incorrect predictions."
			)
		self.model = Gemma3TextModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			use_bias=False,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		"""
		Forward pass through the Gemma2 module.

		Args:
		    input_ids (tp.Optional[chex.Array]): Input tensor containing token IDs.
		    attention_mask (tp.Optional[chex.Array]): Mask for attention.
		    position_ids (tp.Optional[chex.Array]): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxCausalLMOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""

		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
			token_type_ids=token_type_ids,
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

		if self.config.final_logit_softcapping is not None:
			cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
			lm_logits = cap * jax.nn.tanh(lm_logits / cap)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)


@register_module(
	TaskType.SEQUENCE_CLASSIFICATION,
	config=Gemma3TextConfig,
	model_type="gemma3_text", 
)
class Gemma3ForSequenceClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: Gemma3TextConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.model = Gemma3TextModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		assert hasattr(config, "num_labels"), (
			"in order to use `SequenceClassification` Models in `EasyDeL` you first need to attach `num_labels` to model `config`"
		)
		self.score = nn.Linear(
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
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxSequenceClassifierOutput, tp.Tuple]:
		transformer_outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
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

		return FlaxSequenceClassifierOutput(
			logits=pooled_logits,
			past_key_values=past_key_values,
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)


class Gemma3MultiModalProjector(nn.Module):
	def __init__(
		self,
		config: Gemma3Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.mm_input_projection_weight = nn.Param(
			jnp.zeros(
				(
					config.text_config.hidden_size,
					config.vision_config.hidden_size,
				),
				dtype=param_dtype,
			)
		)
		self.mm_soft_emb_norm = Gemma3RMSNorm(
			config.vision_config,
			param_dtype=param_dtype,
			dim=config.vision_config.hidden_size,
			epsilon=config.vision_config.layer_norm_eps,
		)
		self.patches_per_image = int(
			config.vision_config.image_size // config.vision_config.patch_size
		)
		self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
		kernel_size = self.patches_per_image // self.tokens_per_side
		self.kernel_size = kernel_size
		self.avg_pool = lambda x: jax.lax.reduce_window(
			x,
			init_value=0.0,
			computation=jax.lax.add,
			window_dimensions=(1, 1, kernel_size, kernel_size),
			window_strides=(1, 1, kernel_size, kernel_size),
			padding="VALID",
		) / (kernel_size * kernel_size)

	def __call__(self, vision_outputs):
		batch_size, _, seq_length = vision_outputs.shape

		reshaped_vision_outputs = jnp.transpose(vision_outputs, (0, 2, 1))

		reshaped_vision_outputs = reshaped_vision_outputs.reshape(
			batch_size,
			seq_length,
			self.patches_per_image,
			self.patches_per_image,
		)
		pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
		pooled_vision_outputs = pooled_vision_outputs.reshape(batch_size, seq_length, -1)
		pooled_vision_outputs = jnp.transpose(pooled_vision_outputs, (0, 2, 1))
		normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

		projected_vision_outputs = jax.lax.dot_general(
			normed_vision_outputs,
			self.mm_input_projection_weight.T,
			(((normed_vision_outputs.ndim - 1), (0,)), ((), ())),
		)
		return projected_vision_outputs.astype(vision_outputs.dtype)


@register_module(
	TaskType.IMAGE_TEXT_TO_TEXT,
	config=Gemma3Config,
	model_type="gemma3",
)
class Gemma3ForConditionalGeneration(EasyDeLBaseModule):
	loss_type = "ForCausalLM"

	def __init__(
		self,
		config: Gemma3Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.vision_tower = AutoEasyDeLVisionModel.from_config(
			config=config.vision_config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.multi_modal_projector = Gemma3MultiModalProjector(
			config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.vocab_size = config.text_config.vocab_size
		self.language_model = Gemma3ForCausalLM(
			config=config.text_config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.pad_token_id = (
			self.config.pad_token_id if self.config.pad_token_id is not None else -1
		)

	def get_image_features(self, pixel_values: chex.Array) -> chex.Array:
		vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state

		image_features = self.multi_modal_projector(vision_outputs)

		return image_features

	def __call__(
		self,
		input_ids: chex.Array = None,
		pixel_values: chex.Array = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		**lm_kwargs,
	):
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)
		if input_ids is not None and self.config.image_token_index >= self.vocab_size:
			special_image_mask = input_ids == self.config.image_token_index
			llm_input_ids = input_ids
			llm_input_ids = jnp.where(special_image_mask, 0, llm_input_ids)
		else:
			llm_input_ids = input_ids
		if inputs_embeds is None:
			inputs_embeds = self.language_model.model.embed_tokens(llm_input_ids) * (
				self.config.text_config.hidden_size**0.5
			)
		if pixel_values is not None:
			image_features = self.get_image_features(pixel_values)

			if input_ids is None:
				special_image_mask = inputs_embeds == self.language_model.model.embed_tokens(
					jnp.array(self.config.image_token_index, dtype="i4")
				)
			else:
				special_image_mask = jnp.expand_dims(
					(input_ids == self.config.image_token_index),
					-1,
				)
				special_image_mask = jnp.broadcast_to(special_image_mask, inputs_embeds.shape)
			image_features = image_features.astype(inputs_embeds.dtype)
			inputs_embeds = jnp.place(
				inputs_embeds,
				special_image_mask,
				image_features,
				inplace=False,
			)
		outputs = self.language_model(
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			token_type_ids=token_type_ids,
			segment_ids=None,
			**lm_kwargs,
		)

		return Gemma3CausalLMOutputWithPast(
			loss=None,
			logits=outputs.logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			image_hidden_states=image_features if pixel_values is not None else None,
		)

	def _get_compile_model_kwargs(
		self,
		batch_size: int,
		input_tokens_length: int,
		input_sharding: jax.sharding.PartitionSpec,
		rngs: jax.random.PRNGKey,
		vision_included: bool = False,
		vision_batch_size: int = 1,
		vision_channels: int = 3,
		vision_height: tp.Optional[int] = None,
		vision_width: tp.Optional[int] = None,
		required_props: tp.Optional[tp.Mapping[str, tp.Dict[str, tp.Any]]] = None,
		**kwargs,
	):
		basics = super()._get_compile_model_kwargs(
			batch_size=batch_size,
			input_tokens_length=input_tokens_length,
			input_sharding=input_sharding,
			rngs=rngs,
			vision_included=vision_included,
			vision_batch_size=vision_batch_size,
			vision_channels=vision_channels,
			vision_height=vision_height,
			vision_width=vision_width,
			required_props=required_props,
			**kwargs,
		)
		token_type_ids = jnp.ones(
			(batch_size, input_tokens_length),
			dtype="i4",
			device=input_sharding,
		)
		basics.update({"token_type_ids": token_type_ids})
		if vision_included:
			pixel_values = jnp.ones(
				(
					vision_batch_size or 1,
					vision_channels or 3,
					self.config.vision_config.image_size,
					self.config.vision_config.image_size,
				),
				dtype="f4",
			)
			basics.update({"pixel_values": pixel_values})
		return basics

	def prepare_inputs_for_generation(
		self,
		input_ids: chex.Array,
		max_length: int,
		pixel_values: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		token_type_ids: tp.Optional[chex.Array] = None,
	):
		model_inputs = self.language_model.prepare_inputs_for_generation(
			input_ids=input_ids,
			max_length=max_length,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		model_inputs["pixel_values"] = pixel_values
		return model_inputs

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs = super().update_inputs_for_generation(model_outputs, model_kwargs)
		model_kwargs.pop("pixel_values", None)  # only effect first iter
		model_kwargs.pop("token_type_ids", None)  # only effect first iter
		return model_kwargs
