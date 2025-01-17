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
from einops import rearrange
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.infra.utils import (
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm
from easydel.utils.helpers import get_logger
from easydel.layers.ops.lightning_attention import linear_attn, build_slope_tensor
from .xerxes2_configuration import Xerxes2Config as Xerxes2Config

logger = get_logger(__name__)


class Xerxes2Attention(nn.Module):
	def __init__(
		self,
		config: Xerxes2Config,
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
		self.rngs = rngs

		self.embed_dim = config.hidden_size
		self.head_dim = config.head_dim
		self.num_heads = config.num_attention_heads

		kernel = jax.nn.initializers.normal(config.initializer_range)

		linear_class = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=False,
			kernel_init=kernel,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj = linear_class(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear_class(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear_class(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.g_proj = linear_class(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear_class(
			self.num_heads * self.head_dim,
			self.embed_dim,
			rngs=rngs,
		)
		self.norm = RMSNorm(
			self.num_heads * self.head_dim,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def _split_heads(self, hidden_states, num_heads):
		return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		slope_rate: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		batch_size, sequence_length = hidden_states.shape[:2]
		(query_states, key_states, value_states) = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)
		to_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
		query_states = query_states.reshape(*to_shape)
		key_states = key_states.reshape(*to_shape)
		value_states = value_states.reshape(*to_shape)
		query_states = jnp.transpose(query_states, (0, 2, 1, 3))
		key_states = jnp.transpose(key_states, (0, 2, 1, 3))
		value_states = jnp.transpose(value_states, (0, 2, 1, 3))
		if attention_mask is not None:
			assert attention_mask.ndim == 2
			b, h, s, e = value_states.shape
			value_states = value_states * attention_mask.reshape(b, 1, s, 1)
		output = linear_attn(
			q=query_states,
			k=key_states,
			v=value_states,
			slopes=slope_rate,
			dtype=self.config.attn_dtype,
		)

		output = rearrange(output, "b h n d -> b n (h d)")
		output = self.norm(output)
		output = jax.nn.sigmoid(self.g_proj(hidden_states)) * output
		output = self.o_proj(output)
		return (output, None)


class Xerxes2MLP(nn.Module):
	def __init__(
		self,
		config: Xerxes2Config,
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
		self.rngs = rngs

		self.act = nn.silu
		linear_class = functools.partial(
			nn.Linear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(
			self.config.hidden_size,
			self.config.intermediate_size,
			rngs=rngs,
		)
		self.up_proj = linear_class(
			self.config.hidden_size,
			self.config.intermediate_size,
			rngs=rngs,
		)
		self.down_proj = linear_class(
			self.config.intermediate_size,
			self.config.hidden_size,
			rngs=rngs,
		)

	def __call__(self, hidden_states):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.down_proj(
			self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)


class Xerxes2DecoderLayer(nn.Module):
	def __init__(
		self,
		config: Xerxes2Config,
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
		self.rngs = rngs

		attn_block, mlp_block = auto_remat(
			Xerxes2Attention,
			Xerxes2MLP,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			self.config,
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
		rms = functools.partial(
			RMSNorm,
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		self.input_layernorm = rms()
		self.post_attention_layernorm = rms()
		self.pre_feedforward_layernorm = rms()
		self.post_feedforward_layernorm = rms()

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		slope_rate: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states, attn_weight = self.self_attn(
			hidden_states,
			attention_mask,
			slope_rate,
			cache_view,
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
	"base-module",
	config=Xerxes2Config,
	model_type="xerxes2",
	embedding_layer_names=["embed_tokens"],
)
class Xerxes2Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: Xerxes2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
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
			Xerxes2DecoderLayer(
				self.config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.norm = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		"""Forward pass through the Xerxes module."""
		all_hidden_states = () if output_hidden_states else None
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")

		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		hidden_states = inputs_embeds
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				cache_view=past_key_values.views[idx],
				slope_rate=build_slope_tensor(self.config.num_attention_heads),
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
			hidden_states=all_hidden_states,
			past_key_values=past_key_values,
		)


@register_module(
	"causal-language-model",
	config=Xerxes2Config,
	model_type="xerxes2",
	embedding_layer_names=["embed_tokens"],
)
class Xerxes2ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: Xerxes2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
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
		self.model = Xerxes2Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			self.config.hidden_size,
			self.config.vocab_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		"""
		Forward pass through the Xerxes module.

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
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
		)
		hidden_states = outputs[0]
		if self.config.tie_word_embeddings:
			lm_logits = hidden_states @ self.model.embed_tokens.embedding.value.T
		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
