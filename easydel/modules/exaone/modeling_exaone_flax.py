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
from flax import nnx as nn
from jax import numpy as jnp

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
from easydel.layers.norms import RMSNorm
from easydel.utils.helpers import get_logger

from .exaone_configuration import ExaoneConfig

logger = get_logger(__name__)


class ExaoneGatedMLP(nn.Module):
	def __init__(
		self,
		config: ExaoneConfig,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		linear = functools.partial(
			ParallelLinear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.c_fc_0 = linear(config.hidden_size, config.intermediate_size, rngs=rngs)
		self.c_fc_1 = linear(config.hidden_size, config.intermediate_size, rngs=rngs)
		self.c_proj = linear(config.intermediate_size, config.hidden_size, rngs=rngs)
		self.act_fn = ACT2FN[config.activation_function]

	def __call__(self, hidden_states: chex.Array):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.c_proj(
			self.act_fn(self.c_fc_0(hidden_states)) * self.c_fc_1(hidden_states)
		)


class ExaoneAttentionInner(AttentionModule):
	def __init__(
		self,
		config: ExaoneConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.attention_dropout_rate = config.attention_dropout
		if self.head_dim * self.num_heads != self.embed_dim:
			raise ValueError(
				"embed_dim must be divisible by num_heads (got `embed_dim`: "
				f"{self.embed_dim} and `num_heads`: {self.num_heads})."
			)

		linear = functools.partial(
			ParallelLinear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

		self.q_proj = linear(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear(
			self.embed_dim,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear(
			self.embed_dim,
			self.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.out_proj = linear(
			self.embed_dim,
			self.num_heads * self.head_dim,
			rngs=rngs,
		)

		dim = int(
			(config.hidden_size // config.num_attention_heads)
			* (
				config.partial_rotary_factor
				if hasattr(config, "partial_rotary_factor")
				else 1.0
			)
		)
		self.rotary = self.config.get_basic_rope(
			dtype=self.dtype,
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			rotary_dim=dim,
			is_neox_style=True,
		)
		self.attention_performer = FlexibleAttentionModule(
			base_config=config,
			softmax_scale=self.head_dim**-0.5,
			dropout_prob=config.attention_dropout,
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
		attn_output = self.out_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class ExaoneAttention(nn.Module):
	def __init__(
		self,
		config: ExaoneConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.attention = ExaoneAttentionInner(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
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
		return self.attention(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=causal_mask,
			cache_view=cache_view,
			segment_ids=segment_ids,
			output_attentions=output_attentions,
			fcm_mask=fcm_mask,
			frequencies=frequencies,
		)


class ExaoneDecoderLayer(nn.Module):
	def __init__(
		self,
		config: ExaoneConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		attn_block = ExaoneAttention
		mlp_block = ExaoneGatedMLP

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.attn = attn_block(
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
		self.ln_1 = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.ln_2 = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
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
		residual = hidden_states
		hidden_states = self.ln_1(hidden_states)
		attention_output = self.attn(
			hidden_states,
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
		residual = hidden_states
		hidden_states = self.ln_2(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.mlp(
				hidden_states,
			)

		hidden_states = residual + feed_forward_hidden_states
		outputs = (hidden_states,)
		if output_attentions:
			outputs += (attention_output[1],)
		return outputs


@register_module(
	TaskType.BASE_MODULE,
	ExaoneConfig,
	model_type="exaone",
)
class ExaoneModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: ExaoneConfig,
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
		self.wte = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.drop = nn.Dropout(self.config.embed_dropout, rngs=rngs)

		self.h = [
			ExaoneDecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.ln_f = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@functools.cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies(
			head_size=self.config.hidden_size // self.config.num_attention_heads,
			rotary_dim=int(
				(self.config.hidden_size // self.config.num_attention_heads)
				* (
					self.config.partial_rotary_factor
					if hasattr(self.config, "partial_rotary_factor")
					else 1.0
				)
			),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache | PagedAttentionCache] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		return_dict: bool = True,
	) -> tp.Union[BaseModelOutput, tp.Tuple]:
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.wte(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

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
			)
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		hidden_states = self.drop(inputs_embeds)
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.h))
		for idx, layer in enumerate(self.h):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			output = layer(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				cache_view=past_key_values.views[idx],
				cache_metadata=cache_metadata,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
			)
			hidden_states = output[0]

			if output_attentions:
				all_attentions += (output[1],)

		hidden_states = self.ln_f(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		outputs = (hidden_states, all_hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(value for value in outputs if value is not None)

		return BaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	TaskType.CAUSAL_LM,
	ExaoneConfig,
	model_type="exaone",
)
class ExaoneForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: ExaoneConfig,
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
		self.transformer = ExaoneModel(
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
			precision=self.precision,
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
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache | PagedAttentionCache] = None,
		cache_metadata: tp.Optional[TransformerMetadata | PagedAttentionMetadata] = None,
		return_dict: bool = True,
	) -> tp.Union[CausalLMOutput, tp.Tuple]:
		outputs = self.transformer(
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
	config=ExaoneConfig,
	model_type="exaone",
)
class ExaoneForSequenceClassification(EasyDeLBaseModule):
	def __init__(
		self,
		config: ExaoneConfig,
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
		self.model = ExaoneModel(
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

		return SequenceClassifierOutput(
			logits=pooled_logits,
			past_key_values=past_key_values,
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
