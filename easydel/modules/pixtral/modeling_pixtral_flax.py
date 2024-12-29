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
import typing as tp

import chex
import jax.lax
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import FlaxBaseModelOutput
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm

from .pixtral_configuration import PixtralVisionConfig


def position_ids_in_meshgrid(patch_embeds_list, max_width):
	positions = []
	for patch in patch_embeds_list:
		height, width = patch.shape[-2:]
		mesh = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
		h_grid, v_grid = jnp.stack(mesh, axis=-1).reshape(-1, 2).T
		ids = h_grid * max_width + v_grid
		positions.append(ids)
	return jnp.concatenate(positions)


# TODO:Make this jitable
def generate_block_attention_mask(patch_embeds_list, tensor):
	dtype = tensor.dtype
	seq_len = tensor.shape[1]
	d_min = jnp.finfo(dtype).min
	causal_mask = jnp.full((seq_len, seq_len), fill_value=d_min, dtype=dtype)

	block_end_idx = jnp.cumsum(jnp.array(patch_embeds_list))
	block_start_idx = jnp.cumsum(jnp.array([0] + patch_embeds_list[:-1]))

	def update_mask(mask, start_end):
		start, end = start_end
		return mask.at[start:end, start:end].set(0)

	causal_mask = jax.lax.fori_loop(
		0,
		len(block_start_idx),
		lambda i, mask: update_mask(mask, (block_start_idx[i], block_end_idx[i])),
		causal_mask,
	)

	causal_mask = jnp.expand_dims(causal_mask, axis=(0, 1))
	causal_mask = jnp.broadcast_to(causal_mask, (tensor.shape[0], 1, seq_len, seq_len))
	return causal_mask


def compute_frequencies(dim: int, max_patches_per_side: int, theta: float = 10000.0):
	"""
	Computes frequencies with a fixed max length for RoPE.

	Args:
	    dim: Embedding dimension.
	    max_patches_per_side: Maximum number of patches per side of the image.
	    theta: Scaling factor for frequencies.

	Returns:
	    inv_freq: Computed frequencies of shape (max_patches_per_side**2, dim).
	"""
	freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))

	h = jnp.arange(max_patches_per_side)
	w = jnp.arange(max_patches_per_side)

	freqs_h = jnp.outer(h, freqs[::2])
	freqs_w = jnp.outer(w, freqs[1::2])

	inv_freq = jnp.concatenate(
		[
			jnp.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
			jnp.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
		],
		axis=-1,
	).reshape(-1, dim // 2)
	# we reshape to only index on the position indexes, not tuple of indexes

	inv_freq = jnp.concatenate((inv_freq, inv_freq), axis=-1)
	return inv_freq


# Adapted from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return jnp.concatenate((-x2, x1), axis=-1)


# Adapted from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
	    q (`jnp.ndarray`): The query tensor.
	    k (`jnp.ndarray`): The key tensor.
	    cos (`jnp.ndarray`): The cosine part of the rotary embedding.
	    sin (`jnp.ndarray`): The sine part of the rotary embedding.
	    position_ids (`jnp.ndarray`, *optional*):
	        Deprecated and unused.
	    unsqueeze_dim (`int`, *optional*):
	        The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and sin
	        so that they can be properly broadcasted to the dimensions of q and k. For example, note
	        that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
	        k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
	        cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
	        the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
	Returns:
	    `tuple(jnp.ndarray)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
	"""
	cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
	sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed


class PixtralMLP(nn.Module):
	def __init__(
		self,
		config: PixtralVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		linear_class = functools.partial(
			nn.Linear,
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
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)


class PixtralAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: PixtralVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.hidden_size = config.hidden_size
		self.head_dim = self.config.head_dim

		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_attention_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_attention_heads

		linear_class = functools.partial(
			nn.Linear,
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
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)

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
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		output_attentions: bool = False,
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
			self.config.num_attention_heads,
			self.head_dim,
		)
		value_states = value_states.reshape(
			batch_size,
			sequence_length,
			self.config.num_attention_heads,
			self.head_dim,
		)

		query_states, key_states = apply_rotary_pos_emb(
			q=query_states,
			k=key_states,
			cos=jnp.cos(frequencies),
			sin=jnp.sin(frequencies),
			position_ids=position_ids,
			unsqueeze_dim=0,
		)

		(
			key_states,
			value_states,
			attention_mask,
			attention_bias,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=None,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=None,
			fcm_mask=None,
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
			uses_cache=False,
			segment_ids=None,
			causal_mask=None,
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


class PixtralBlock(nn.Module):
	def __init__(
		self,
		config: PixtralVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		attn_block = PixtralAttention
		mlp_block = PixtralMLP

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.attention = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.feed_forward = mlp_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.attention_norm = RMSNorm(
			dim=config.hidden_size,
			eps=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.ffn_norm = RMSNorm(
			dim=config.hidden_size,
			eps=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		output_attentions: bool = False,
		frequencies: tp.Optional[chex.Array] = None,
	):
		residual = hidden_states
		attention_output = self.attention(
			self.attention_norm(hidden_states),
			attention_mask,
			position_ids,
			output_attentions,
			frequencies,
		)

		hidden_states = attention_output[0] + residual
		ffd_inp = self.ffn_norm(hidden_states)
		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.feed_forward, ffd_inp, self.config.scan_mlp_chunk_size
			)
		else:
			feed_forward_hidden_states = self.feed_forward(ffd_inp)

		hidden_states = hidden_states + feed_forward_hidden_states
		outputs = (hidden_states,)
		if output_attentions:
			outputs += (attention_output[1],)
		return outputs


class PixtralTransformer(nn.Module):
	def __init__(
		self,
		config: PixtralVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.layers = [
			PixtralBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		inputs_embeds: chex.Array,
		position_embeddings: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		batch_size, sequence_length, _ = inputs_embeds.shape

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")

		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		hidden_states = inputs_embeds
		for idx, block in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				output_attentions=output_attentions,
				position_embeddings=position_embeddings,
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (hidden_states, all_hidden_states, all_attentions, None)

		if not return_dict:
			return tuple(value for value in outputs if value is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=None,
		)


@register_module(
	TaskType.BASE_VISION,
	config=PixtralVisionConfig,
	model_type="pixtral",
	embedding_layer_names=["embed_tokens"],
)
class PixtralVisionModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: PixtralVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.patch_conv = nn.Conv(
			in_features=config.num_channels,
			out_features=config.hidden_size,
			kernel_size=(config.patch_size,) * 2,
			strides=(config.patch_size,) * 2,
			use_bias=False,
			precision=precision,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.ln_pre = RMSNorm(
			config.hidden_size,
			eps=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.transformer = PixtralTransformer(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	@functools.cached_property
	def frequencies(self):
		return compute_frequencies(
			dim=self.config.head_dim,
			theta=self.config.rope_theta,
			max_patches_per_side=self.config.image_size // self.config.patch_size,
		)

	def __call__(
		self,
		pixel_values: tp.List[chex.Array],
		output_hidden_states: tp.Optional[bool] = False,
		output_attentions: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		*args,
		**kwargs,
	) -> tp.Union[tp.Tuple, FlaxBaseModelOutput]:
		patch_embeds_list = [
			self.patch_conv(jnp.expand_dims(img, 0).astype(self.dtype).transpose(0, 2, 3, 1))
			for img in pixel_values
		]
		patch_embeds_list = [p.transpose(0, 3, 1, 2) for p in patch_embeds_list]
		# flatten to a single sequence
		patch_embeds = jnp.concatenate(
			[
				jnp.transpose(jnp.reshape(p, (p.shape[0], p.shape[1], -1)), (0, 2, 1))
				for p in patch_embeds_list
			],
			axis=1,
		)
		patch_embeds = self.ln_pre(patch_embeds)

		# positional embeddings
		position_ids = position_ids_in_meshgrid(
			patch_embeds_list,
			max_width=self.config.image_size // self.config.patch_size,
		)

		attention_mask = generate_block_attention_mask(
			[p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
		)
		return self.transformer(
			inputs_embeds=patch_embeds,
			attention_mask=attention_mask,
			position_embeddings=self.frequencies[position_ids],
		)
