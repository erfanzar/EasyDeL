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
from functools import partial
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
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
from easydel.modules.qwen2_vl.qwen2_vl_configuration import (
	Qwen2VLConfig,
	Qwen2VLVisionConfig,
)


def precompute_vl_rotary(dim, theta, max_position):
	inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype="f4") / dim))
	seq = jnp.arange(0, max_position, "f4")
	return jnp.outer(seq, inv)


def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(array: chex.Array, freqs: chex.Array) -> chex.Array:
	orig_dtype = array.dtype
	array = array.astype("f4")
	cos = jnp.cos(freqs)
	sin = jnp.sin(freqs)
	cos = jnp.expand_dims(jnp.repeat(jnp.expand_dims(cos, 1), 2, -1), 0).astype("f4")
	sin = jnp.expand_dims(jnp.repeat(jnp.expand_dims(sin, 1), 2, -1), 0).astype("f4")
	output = (array * cos) + (rotate_half(array) * sin)
	output = output.astype(orig_dtype)
	return output


def create_attention_mask(q, cu_seqlens):
	"""Creates an attention mask based on cumulative sequence lengths.

	Args:
	    q: A JAX array with the dtype from which we will get the min float value for the attention mask
	    cu_seqlens: A JAX array representing cumulative sequence lengths.

	Returns:
	    A JAX array representing the attention mask.
	"""
	seq_length = cu_seqlens[-1] if len(cu_seqlens) > 0 else 0
	attention_mask = jnp.full(
		(1, seq_length, seq_length), jnp.finfo(q.dtype).min, dtype=q.dtype
	)

	def mask_loop(i, attention_mask):
		start = cu_seqlens[i - 1]
		end = cu_seqlens[i]
		return attention_mask.at[..., start:end, start:end].set(0)

	attention_mask = jax.lax.fori_loop(1, len(cu_seqlens), mask_loop, attention_mask)

	return attention_mask


class PatchEmbed(nn.Module):
	def __init__(
		self,
		patch_size: int = 14,
		temporal_patch_size: int = 2,
		in_channels: int = 3,
		embed_dim: int = 1152,
		precision: jax.lax.PrecisionLike = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.dtype = dtype
		self.patch_size = patch_size
		self.temporal_patch_size = temporal_patch_size
		self.in_channels = in_channels
		self.embed_dim = embed_dim

		kernel_size = [temporal_patch_size, patch_size, patch_size]
		self.proj = nn.Conv(
			in_features=in_channels,
			out_features=embed_dim,
			kernel_size=kernel_size,
			strides=kernel_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, hidden_states: chex.Array) -> chex.Array:
		hidden_states = hidden_states.reshape(
			-1,
			self.in_channels,
			self.temporal_patch_size,
			self.patch_size,
			self.patch_size,
		)
		hidden_states = self.proj(
			hidden_states.astype(self.dtype),
		).reshape(-1, self.embed_dim)
		return hidden_states


class PatchMerger(nn.Module):
	def __init__(
		self,
		dim: int,
		context_dim: int,
		spatial_merge_size: int = 2,
		precision: jax.lax.PrecisionLike = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__()
		self.dtype = dtype
		self.hidden_size = context_dim * (spatial_merge_size**2)
		self.ln_q = nn.LayerNorm(context_dim, epsilon=1e-6)
		self.mlp = nn.Sequential(
			nn.Linear(
				self.hidden_size,
				self.hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			),
			partial(nn.gelu, approximate=False),
			nn.Linear(
				self.hidden_size,
				dim,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			),
		)

	def __call__(self, x: chex.Array) -> chex.Array:
		x = self.mlp(self.ln_q(x).reshape(-1, self.hidden_size))
		return x


class VisionMlp(nn.Module):
	def __init__(
		self,
		dim: int,
		hidden_dim: int,
		hidden_act: str,
		precision: jax.lax.PrecisionLike = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__()
		self.fc1 = nn.Linear(
			dim,
			hidden_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.act = ACT2FN[hidden_act]
		self.fc2 = nn.Linear(
			hidden_dim,
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, x: chex.Array) -> chex.Array:
		return self.fc2(self.act(self.fc1(x)))


class VisionAttention(FlaxAttentionModule):
	def __init__(
		self,
		config,
		dim: int,
		num_heads: int = 16,
		precision: jax.lax.PrecisionLike = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config)
		self.rngs = rngs
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.qkv = nn.Linear(
			dim,
			dim * 3,
			bias=True,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.proj = nn.Linear(
			dim,
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=0,
			num_q_heads=num_heads,
			num_kv_heads=num_heads,
			head_dims=self.head_dim,
			precision=precision,
			force_float32_tpu=True,
			attn_mechanism=config.attn_mechanism,
			dtype=config.attn_dtype,
			mesh=config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			axis_name=config.attention_axis_name,
			base_config=config,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		cu_seqlens: chex.Array,
		rotary_pos_emb: chex.Array = None,
	) -> chex.Array:
		seq_length = hidden_states.shape[0]
		q, k, v = jnp.split(
			self.qkv(hidden_states)
			.reshape(seq_length, 3, self.num_heads, -1)
			.transpose(1, 0, 2, 3),
			3,
			0,
		)
		q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
		k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
		attn_msk = create_attention_mask(q, cu_seqlens)
		attentions = self.attention_performer(
			query_states=q,
			key_states=k,
			value_states=v,
			bias=attn_msk,
			attention_mask=None,
			causal=True,
			dropout_rng=self.rngs.params(),
			query_sequence_length=q.shape[1],
			key_value_sequence_length=k.shape[1],
			uses_cache=None,
			segment_ids=None,
			causal_mask=None,
		)
		return self.proj(attentions.attention_outputs.reshape(seq_length, -1))


class Qwen2VLVisionBlock(nn.Module):
	def __init__(
		self,
		config: Qwen2VLVisionConfig,
		precision: jax.lax.PrecisionLike = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(
			config.embed_dim,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.norm2 = nn.LayerNorm(
			config.embed_dim,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

		self.attn = VisionAttention(
			config.embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.mlp = VisionMlp(
			dim=config.embed_dim,
			hidden_dim=mlp_hidden_dim,
			hidden_act=config.hidden_act,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> chex.Array:
		hidden_states = hidden_states + self.attn(
			self.norm1(hidden_states),
			cu_seqlens=cu_seqlens,
			rotary_pos_emb=rotary_pos_emb,
		)
		hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
		return hidden_states


class Qwen2VLMLP(nn.Module):
	def __init__(
		self,
		config: Qwen2VLConfig,
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
		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=self.config.mlp_bias,
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
		self.dropout = nn.Dropout(rate=self.config.resid_pdrop, rngs=rngs)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		hidden_states = self.dropout(hidden_states)
		return hidden_states


class Qwen2VLAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: Qwen2VLConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
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
			use_bias=config.attention_bias,
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

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			self.head_dim,
			self.head_dim,
			True,
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
		self.resid_dropout = nn.Dropout(
			rate=config.resid_pdrop,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: Optional[TransformerCacheView] = None,
		segment_ids: Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, chex.Array]:
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
			attention_bias,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=cache_view,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=fcm_mask,
		)

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=self.rngs.params(),
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=segment_ids,
			causal_mask=causal_mask,
		)
		attn_output = self.resid_dropout(
			self.o_proj(
				self.shard_attention_prod(
					attn_output=self._merge_heads(attentions.attention_outputs)
				)
			),
		)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class Qwen2VLDecoderLayer(nn.Module):
	def __init__(
		self,
		config: Qwen2VLConfig,
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
		attn_block = Qwen2VLAttention
		mlp_block = Qwen2VLMLP
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
		self.input_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.post_attention_layernorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		cache_view: Optional[TransformerCacheView] = None,
		segment_ids: Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	):
		attn_outputs = self.self_attn(
			self.input_layernorm(hidden_states),
			attention_mask,
			position_ids,
			causal_mask,
			cache_view,
			segment_ids,
			output_attentions,
			fcm_mask,
			frequencies,
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
