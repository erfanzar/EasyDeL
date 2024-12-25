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
import typing as tp
from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.etils.etils import get_logger
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.modules.gpt_j.gpt_j_configuration import GPTJConfig as GPTJConfig

logger = get_logger(__name__)


class GPTJAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: GPTJConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		causal: bool = True,
		is_cross_attention: bool = False,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)

		self.precision = precision
		self.dtype = dtype
		self.rngs = rngs
		self.is_cross_attention = is_cross_attention
		self.causal = causal
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads

		self.rotary_dim = config.rotary_dim

		linear = partial(
			nn.Linear,
			self.embed_dim,
			self.embed_dim,
			use_bias=False,
			dtype=dtype,
			kernel_init=nn.initializers.normal(config.initializer_range),
			param_dtype=param_dtype,
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj, self.k_proj, self.v_proj = (
			linear(rngs=rngs),
			linear(rngs=rngs),
			linear(rngs=rngs),
		)
		self.out_proj = linear(rngs=rngs)

		self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			head_size=self.embed_dim,
			rotary_dim=self.rotary_dim,
			base=10000,
			is_neox_style=False,
		)

		self.attention_performer = FlexibleAttentionModule(
			use_sharding_constraint=self.config.use_sharding_constraint,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
			attention_dropout=self.config.attn_pdrop,
			head_dims=self.head_dim,
			shard_attention_computation=self.config.shard_attention_computation,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			dtype=self.config.attn_dtype,
			partition_axis=self.config.partition_axis,
			scan_ring_attention=self.config.scan_ring_attention,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			base_config=self.config,
		)

	def _split_heads(self, hidden_states):
		return hidden_states.reshape(
			hidden_states.shape[:2] + (self.num_heads, self.head_dim)
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		frequencies: tp.Optional[chex.Array] = None,
	):
		query = self.q_proj(hidden_states)
		key = self.k_proj(hidden_states)
		value = self.v_proj(hidden_states)

		query = self._split_heads(query)
		key = self._split_heads(key)
		value = self._split_heads(value)

		query, key = self.rotary(
			positions=position_ids,
			query=query,
			key=key,
			frequencies=frequencies,
		)

		(
			key,
			value,
			attention_mask,
			attention_bias,
		) = self.concatenate(
			query=query,
			key=key,
			cache_view=cache_view,
			value=value,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=None,
		)

		attentions = self.attention_performer(
			query_states=query,
			key_states=key,
			value_states=value,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query.shape[1],
			key_value_sequence_length=key.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=segment_ids,
			causal_mask=causal_mask,
		)
		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.out_proj(attn_output)
		attn_output = self.resid_dropout(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class GPTJMLP(nn.Module):
	def __init__(
		self,
		config: GPTJConfig,
		intermediate_size: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config: GPTJConfig = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.intermediate_size = intermediate_size
		embed_dim = config.hidden_size
		kernel_init = nn.initializers.normal(config.initializer_range)

		self.fc_in = nn.Linear(
			embed_dim,
			intermediate_size,
			dtype=dtype,
			param_dtype=dtype,
			precision=precision,
			kernel_init=kernel_init,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.fc_out = nn.Linear(
			intermediate_size,
			embed_dim,
			dtype=dtype,
			param_dtype=dtype,
			precision=precision,
			kernel_init=kernel_init,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.act = ACT2FN[config.activation_function]
		self.dropout = nn.Dropout(rate=config.resid_pdrop)

	def __call__(self, hidden_states):
		hidden_states = self.dropout(self.fc_out(self.act(self.fc_in(hidden_states))))
		return hidden_states


class GPTJBlock(nn.Module):
	def __init__(
		self,
		config: GPTJConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config: GPTJConfig = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		hidden_size = self.config.hidden_size
		inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

		attn_block = GPTJAttention
		mlp_block = GPTJMLP
		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.ln_1 = nn.LayerNorm(
			self.config.hidden_size,
			epsilon=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=dtype,
			rngs=rngs,
		)

		self.attn = attn_block(
			config,
			dtype=dtype,
			param_dtype=dtype,
			precision=precision,
			rngs=rngs,
		)

		self.mlp = mlp_block(
			config,
			inner_dim,
			dtype=dtype,
			param_dtype=dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		frequencies: tp.Optional[chex.Array] = None,
	):
		residual = hidden_states
		hidden_states = self.ln_1(hidden_states)
		attn_outputs = self.attn(
			hidden_states,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			cache_view,
			output_attentions,
			frequencies,
		)
		attn_output = attn_outputs[0]
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.mlp(hidden_states)
		# residual connection
		hidden_states = attn_output + feed_forward_hidden_states + residual

		return (hidden_states,) + attn_outputs[1:]


@register_module(
	"base-module",
	config=GPTJConfig,
	model_type="gptj",
	embedding_layer_names=["wte"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
class GPTJModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: GPTJConfig,
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
		self.embed_dim = config.hidden_size
		self.wte = nn.Embed(
			self.config.vocab_size,
			self.embed_dim,
			embedding_init=nn.initializers.normal(stddev=config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(
			rate=self.config.embd_pdrop,
			rngs=rngs,
		)
		self.h = [
			GPTJBlock(
				config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.ln_f = nn.LayerNorm(
			self.config.hidden_size,
			epsilon=self.config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@cached_property
	def frequencies(self):
		embed_dim = self.config.hidden_size
		num_heads = self.config.num_attention_heads
		head_dim = embed_dim // num_heads

		rotary_dim = self.config.rotary_dim
		return self.config.get_basic_frequencies(
			rotary_dim=rotary_dim,
			head_size=head_dim,
			base=10000,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		extra_embedding: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.wte(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		hidden_states = (
			inputs_embeds + extra_embedding if extra_embedding is not None else inputs_embeds
		)

		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.h))

		hidden_states = self.dropout(inputs_embeds)
		for idx, block in enumerate(self.h):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			hidden_states, attn_weight = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				cache_view=past_key_values.views[idx],
				position_ids=position_ids,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
				causal_mask=self.causal_mask,
			)
			if output_attentions:
				all_attentions += (attn_weight,)

		hidden_states = self.ln_f(hidden_states)

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
	"causal-language-model",
	config=GPTJConfig,
	model_type="gptj",
	embedding_layer_names=["wte"],
	layernorm_names=["ln_1", "ln_2", "ln_f"],
)
class GPTJForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: GPTJConfig,
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
		self.transformer = GPTJModel(
			self.config,
			dtype=self.dtype,
			param_dtype=self.dtype,
			precision=self.precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			rngs=rngs,
			dtype=self.dtype,
			kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			param_dtype=self.dtype,
			precision=self.precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		extra_embedding: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		outputs = self.transformer(
			input_ids=input_ids,
			extra_embedding=extra_embedding,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			inputs_embeds=inputs_embeds,
			past_key_values=past_key_values,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			shared_kernel = self.transformer.wte.embedding.value.T
			self.lm_head.kernel.value = shared_kernel
			lm_logits = self.lm_head(hidden_states)

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
