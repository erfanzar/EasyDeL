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


import copy
import math
import typing as tp
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
	MoeCausalLMOutput,
	MoeModelOutput,
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
from easydel.layers.norms import RMSNorm
from easydel.layers.ops import lightning_attention

from .minimax_text_01_configuration import MiniMaxText01Config


def compute_slops(nhd):
	def get_slopes(n):
		def get_slopes_power_of_2(n):
			start = 2 ** (-(2 ** -(math.log2(n) - 3)))
			ratio = start
			return [start * ratio**i for i in range(n)]

		if math.log2(n).is_integer():
			return get_slopes_power_of_2(n)
		else:
			closest_power_of_2 = 2 ** math.floor(math.log2(n))
			return (
				get_slopes_power_of_2(closest_power_of_2)
				+ get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
			)

	return jnp.asarray(get_slopes(nhd), dtype=jnp.float32).reshape(nhd, 1, 1)


def get_activation_fn(activation):
	if activation == "gelu":
		return partial(jax.nn.gelu, approximate=False)
	elif activation == "relu":
		return jax.nn.relu
	elif activation == "elu":
		return jax.nn.elu
	elif activation == "sigmoid":
		return jax.nn.sigmoid
	elif activation == "exp":

		def f(x):
			x_max = jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
			y = jnp.exp(x - x_max)

			return y

		return f
	elif activation == "leak":
		return jax.nn.leaky_relu
	elif activation == "1+elu":

		def f(x):
			return 1 + jax.nn.elu(x)

		return f
	elif activation == "2+elu":

		def f(x):
			return 2 + jax.nn.elu(x)

		return f
	elif activation == "silu" or activation == "swish":
		return jax.nn.silu
	elif activation == "sine":
		return jax.numpy.sin
	else:
		warnings.warn(
			f"activation: does not support {activation}, use Identity!!!",
			stacklevel=1,
		)
		return lambda x: x


class GLU(nn.Module):
	def __init__(
		self,
		d1,
		d2,
		bias=False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()

		self.l1 = nn.Linear(
			d1,
			d2,
			use_bias=bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.l2 = nn.Linear(
			d1,
			d2,
			use_bias=bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.l3 = nn.Linear(
			d2,
			d1,
			use_bias=bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, x: jax.Array) -> jax.Array:
		return self.l3(self.l1(x) * self.l2(x))


class MiniMaxText01LightningAttention(nn.Module):
	def __init__(
		self,
		config: MiniMaxText01Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

		self.out_proj = nn.Linear(
			self.head_dim * self.num_heads,
			self.hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.act = get_activation_fn(config.hidden_act)
		self.norm = RMSNorm(
			self.head_dim * self.num_heads,
			eps=config.rms_norm_eps,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
		)

		self.qkv_proj = nn.Linear(
			self.hidden_size,
			3 * self.head_dim * self.num_heads,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.output_gate = nn.Linear(
			self.hidden_size,
			self.head_dim * self.num_heads,
			use_bias=False,
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
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
		slope_rate: tp.Optional[chex.Array] = None,
	):
		# TODO: fix these static issues here
		batch_size, sequence_length, _ = hidden_states.shape
		query_states, key_states, value_states = jnp.split(
			self.act(self.qkv_proj(hidden_states)), 3, -1
		)

		to_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
		query_states = query_states.reshape(*to_shape)
		key_states = key_states.reshape(*to_shape)
		value_states = value_states.reshape(*to_shape)
		query_states = jnp.transpose(query_states, (0, 2, 1, 3))
		key_states = jnp.transpose(key_states, (0, 2, 1, 3))
		value_states = jnp.transpose(value_states, (0, 2, 1, 3))
		output, ola = lightning_attention.lightning_attention(
			q=query_states,
			k=key_states,
			v=value_states,
			position_ids=None,
			slope_rate=slope_rate,
			attn_mask=attention_mask,
			past_key_value=cache_view.key_value if cache_view is not None else None,
			init_cache=True if cache_view is not None else False,
			dtype=self.config.attn_dtype,
			softmax_dtype=self.config.attn_softmax_dtype,
		)
		if cache_view is not None:
			cache_view.key_value = ola
		output = rearrange(output, "b h n d -> b n (h d)")
		output = self.norm(output)
		output = jax.nn.sigmoid(self.g_proj(hidden_states)) * output
		output = self.o_proj(output)
		return (output, None)


class MiniMaxText01Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: MiniMaxText01Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)

		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.hidden_size = config.hidden_size
		head_dim = config.hidden_size // config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", head_dim)
		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_key_value_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_key_value_heads

		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.q_proj = linear_class(
			config.hidden_size, config.num_attention_heads * self.head_dim
		)
		self.k_proj = linear_class(
			config.hidden_size, config.num_key_value_heads * self.head_dim
		)
		self.v_proj = linear_class(
			config.hidden_size, config.num_key_value_heads * self.head_dim
		)
		self.o_proj = linear_class(
			config.num_attention_heads * self.head_dim, config.hidden_size
		)
		self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)
		self.rotary = self.config.get_basic_rope(
			self.dtype,
			self.head_dim,
			self.rotary_dim,
			True,
		)

		self.attention_performer = FlexibleAttentionModule(
			dropout_prob=config.attention_dropout,
			base_config=config,
			softmax_scale=self.head_dim**-0.5, 
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
	) -> tp.Tuple[chex.Array, chex.Array]:
		batch_size, sequence_length = hidden_states.shape[:2]
		query_states, key_states, value_states = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)
		qshape = (
			batch_size,
			sequence_length,
			self.config.num_attention_heads,
			self.head_dim,
		)
		kv_shape = (
			batch_size,
			sequence_length,
			self.config.num_key_value_heads,
			self.head_dim,
		)
		query_states = query_states.reshape(qshape)
		key_states = key_states.reshape(kv_shape)
		value_states = value_states.reshape(kv_shape)

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
			init_bias=init_attention_bias,
			attention_mask=attention_mask,
			segment_ids=segment_ids,
			causal=True,
			dropout_rng=self.rngs.params(),
		)

		attn_output = self.o_proj(
			self.shard_attention_prod(
				attn_output=self._merge_heads(attentions.attention_outputs)
			)
		)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class MiniMaxText01MLP(nn.Module):
	def __init__(
		self,
		config: MiniMaxText01Config,
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
		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(config.hidden_size, config.intermediate_size)
		self.down_proj = linear_class(config.intermediate_size, config.hidden_size)
		self.up_proj = linear_class(config.hidden_size, config.intermediate_size)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)


class MiniMaxText01BlockSparseTop2MLP(nn.Module):
	def __init__(
		self,
		config: MiniMaxText01Config,
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
		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.w1 = linear_class(config.hidden_size, config.intermediate_size)
		self.w2 = linear_class(config.intermediate_size, config.hidden_size)
		self.w3 = linear_class(config.hidden_size, config.intermediate_size)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
		current_hidden_states = self.w2(current_hidden_states)
		return current_hidden_states


class MiniMaxText01SparseMoeBlock(nn.Module):
	def __init__(
		self,
		config: MiniMaxText01Config,
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
		self.gate = nn.Linear(
			config.hidden_size,
			config.num_local_experts,
			use_bias=False,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
		)

		self.experts = [
			MiniMaxText01BlockSparseTop2MLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(config.num_local_experts)
		]
		self.jitter_noise = config.router_jitter_noise
		self.deterministic = False

	def __call__(self, hidden_states: chex.Array) -> tp.Tuple[chex.Array, chex.Array]:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		if not self.deterministic and self.jitter_noise > 0:
			hidden_states *= jax.random.uniform(
				self.rngs.param(),
				shape=hidden_states.shape,
				minval=1.0 - self.jitter_noise,
				maxval=1.0 + self.jitter_noise,
			)
		router_logits = self.gate(hidden_states).astype(
			jnp.promote_types(self.dtype, jnp.float32)
		)
		routing_weights, selected_experts = jax.lax.top_k(
			router_logits, k=self.config.num_experts_per_tok
		)
		routing_weights = jax.nn.softmax(
			routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
		)
		routing_weights /= routing_weights.sum(axis=-1, keepdims=True)
		final_hidden_state = jnp.zeros_like(hidden_states)

		for index in range(self.config.num_local_experts):
			expert_layer_output = (
				block_wise_ffn(
					self.experts[index],
					hidden_states,
					self.config.scan_mlp_chunk_size,
				)
				if self.config.use_scan_mlp
				else self.experts[index](hidden_states)
			)
			expert_layer_output_exp = (
				jnp.sum(
					jnp.multiply(
						selected_experts == index,
						routing_weights,
					),
					axis=-1,
				)[:, :, None]
				* expert_layer_output
			)
			final_hidden_state += expert_layer_output_exp
		return (final_hidden_state, router_logits)


class MiniMaxText01DecoderLayer(nn.Module):
	def __init__(
		self,
		config: MiniMaxText01Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		if config.attention_type == 0:
			attn_block = MiniMaxText01LightningAttention
		else:
			attn_block = MiniMaxText01Attention
		mlp_block = MiniMaxText01SparseMoeBlock

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.block_sparse_moe = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.input_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		self.post_attention_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
		)

		self.postnorm = getattr(config, "postnorm", False)
		self.layernorm_attention_alpha = (
			getattr(
				config,
				"layernorm_linear_attention_alpha",
				1,
			)
			if config.attention_type == 0
			else getattr(
				config,
				"layernorm_full_attention_alpha",
				1,
			)
		)
		self.layernorm_attention_beta = (
			getattr(
				config,
				"layernorm_linear_attention_beta",
				1,
			)
			if config.attention_type == 0
			else getattr(
				config,
				"layernorm_full_attention_beta",
				1,
			)
		)
		self.layernorm_mlp_alpha = getattr(
			config,
			"layernorm_mlp_alpha",
			1,
		)
		self.layernorm_mlp_beta = getattr(
			config,
			"layernorm_mlp_beta",
			1,
		)

		shared_intermediate = getattr(
			config,
			"shared_intermediate_size",
			0,
		)
		self.shared_moe = False
		if shared_intermediate > 0:
			self.shared_moe = True
			self.shared_mlp = MiniMaxText01MLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			self.coefficient = nn.Linear(
				self.hidden_size,
				1,
				use_bias=False,
				rngs=rngs,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
			)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		output_router_logits: bool = False,
		slope_rate: tp.Optional[float] = None,
		frequencies: tp.Optional[chex.Array] = None,
	):
		# if self.config.use_scan_mlp:
		# 	feed_forward_hidden_states = block_wise_ffn(
		# 		self.mlp,
		# 		feed_forward_input,
		# 		self.config.scan_mlp_chunk_size,
		# 	)
		# else:
		# 	feed_forward_hidden_states = self.mlp(feed_forward_input)

		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		if self.postnorm:
			residual = hidden_states

		hidden_states, self_attn_weights = self.self_attn(
			hidden_states=hidden_states,
			causal_mask=causal_mask,
			position_ids=position_ids,
			attention_mask=attention_mask,
			cache_view=cache_view,
			output_attentions=output_attentions,
			slope_rate=slope_rate,
			frequencies=frequencies,
		)

		hidden_states = (
			residual * self.layernorm_attention_alpha
			+ hidden_states * self.layernorm_attention_beta
		)

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		if self.postnorm:
			residual = hidden_states

		moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
		if self.shared_moe:
			output_mlp = self.shared_mlp(hidden_states)
			weight_fp32 = self.coefficient.kernel.value.astype(jnp.float32)
			coef = hidden_states.astype(jnp.float32) @ weight_fp32
			coef = jax.nn.sigmoid(coef).to(hidden_states.dtype)
			hidden_states = moe_hidden_states * (1 - coef) + output_mlp * coef
		else:
			hidden_states = moe_hidden_states

		hidden_states = (
			residual * self.layernorm_mlp_alpha + hidden_states * self.layernorm_mlp_beta
		)

		outputs = (
			hidden_states,
			self_attn_weights if output_attentions else None,
			router_logits if output_router_logits else None,
		)
		return outputs


@register_module(
	TaskType.BASE_MODULE,
	config=MiniMaxText01Config,
	model_type="MiniMaxText01",
)
class MiniMaxText01Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: MiniMaxText01Config,
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

		self.embed_tokens = nn.Embed(
			num_embeddings=self.config.vocab_size,
			features=self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			rngs=rngs,
		)

		self.layers: tp.List[MiniMaxText01DecoderLayer] = []

		for i in range(config.num_hidden_layers):
			_config = copy.deepcopy(config)
			if self.attn_type_list[i] == 0:
				_config.attention_type = 0
			else:
				_config.attention_type = 1
			self.layers.append(
				MiniMaxText01DecoderLayer(
					config=_config,
					layer_idx=i,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
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
		output_router_logits: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeModelOutput, tp.Tuple]:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		all_router_logits = () if output_router_logits else None
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

		hidden_states = self.dropout(inputs_embeds)
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))

		sr = compute_slops(nhd=self.config.num_attention_heads)
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_view=past_key_values.views[idx],
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
				output_router_logits=output_router_logits,
				segment_ids=segment_ids,
				frequencies=self.frequencies,
				slope_rate=sr[idx] * (1 - idx / (len(self.layers) - 1) + 1e-5),
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)
			if output_router_logits:
				all_router_logits += (layer_outputs[2],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (
			hidden_states,
			all_hidden_states,
			all_attentions,
			all_router_logits,
			past_key_values,
		)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return MoeModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			router_logits=all_router_logits,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	TaskType.CAUSAL_LM,
	config=MiniMaxText01Config,
	model_type="MiniMaxText01",
)
class MiniMaxText01ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: MiniMaxText01Config,
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
		self.model = MiniMaxText01Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			rngs=rngs,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=False,
			kernel_init=nn.initializers.normal(config.initializer_range),
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
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> MoeCausalLMOutput | tp.Tuple:
		if output_router_logits is None:
			output_router_logits = self.config.output_router_logits
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			output_router_logits=output_router_logits,
			past_key_values=past_key_values,
			return_dict=True,
			segment_ids=segment_ids,
		)
		logits = self.lm_head(outputs.last_hidden_state)
		batch_size, seq_length, hd = logits.shape
		aux_loss = None
		if output_router_logits and outputs.router_logits is not None:
			aux_loss = auxiliary_load_balancing_loss_func(
				gate_logits=tuple(  # type:ignore
					[
						logit.reshape(batch_size * seq_length, -1)
						for logit in outputs.router_logits
					]  # type:ignore
				),
				num_experts=self.config.num_local_experts,
				top_k=self.config.num_experts_per_tok,
				attention_mask=attention_mask,
			)
			aux_loss = aux_loss * self.config.router_aux_loss_coef
		if not return_dict:
			outputs = (logits,) + tuple(
				v
				for v in [
					aux_loss,
					outputs.hidden_states,
					outputs.attentions,
					outputs.router_logits,
				]
				if v is not None
			)
			return outputs

		return MoeCausalLMOutput(
			aux_loss=aux_loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			router_logits=outputs.router_logits,
		)
