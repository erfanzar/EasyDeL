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
import warnings
from functools import partial

import chex
import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
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
from easydel.layers.norms import RMSNorm
from easydel.modules.qwen2_vl.qwen2_vl_configuration import (
	Qwen2VLConfig,
	Qwen2VLVisionConfig,
)


# TODO: Convert this to a jitable jax fn and use that inside model instead of precall
def get_rope_index(
	input_ids: np.ndarray,
	image_grid_thw: tp.Optional[np.ndarray] = None,
	video_grid_thw: tp.Optional[np.ndarray] = None,
	attention_mask: tp.Optional[np.ndarray] = None,
	spatial_merge_size: int = 1,
	image_token_id: int = -1,
	video_token_id: int = -1,
	vision_start_token_id: int = -1,
) -> tp.Tuple[np.ndarray, np.ndarray]:
	"""
	Calculate the 3D rope index based on image and video's temporal, height, and width in LLM.

	Args:
	    input_ids (`np.ndarray` of shape `(batch_size, sequence_length)`):
	        Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
	        it.
	    image_grid_thw (`np.ndarray` of shape `(num_images, 3)`, *optional*):
	        The temporal, height, and width of feature shape of each image in LLM.
	    video_grid_thw (`np.ndarray` of shape `(num_videos, 3)`, *optional*):
	        The temporal, height, and width of feature shape of each video in LLM.
	    attention_mask (`np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
	        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
	        - 1 for tokens that are **not masked**,
	        - 0 for tokens that are **masked**.
	    spatial_merge_size (int):
	        The spatial merge size for vision embeddings.
	    image_token_id (int):
	        The token ID representing an image.
	    video_token_id (int):
	        The token ID representing a video.
	    vision_start_token_id (int):
	        The token ID representing the start of a vision sequence.

	Returns:
	    position_ids (`np.ndarray` of shape `(3, batch_size, sequence_length)`)
	    mrope_position_deltas (`np.ndarray` of shape `(batch_size)`)
	"""
	if input_ids.shape[-1] != 1:
		attention_mask = attention_mask[:, : input_ids.shape[-1]]
	if input_ids is not None and (
		image_grid_thw is not None or video_grid_thw is not None
	):
		total_input_ids = input_ids
		if attention_mask is None:
			attention_mask = np.ones_like(total_input_ids)

		position_ids = np.ones(
			(3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
		)
		image_index, video_index = 0, 0
		mrope_position_deltas = []

		for i in range(input_ids.shape[0]):
			input_ids_masked = input_ids[i][attention_mask[i] == 1]
			vision_start_indices = np.where(input_ids_masked == vision_start_token_id)[0]
			vision_tokens = input_ids_masked[vision_start_indices + 1]
			image_nums = np.sum(vision_tokens == image_token_id)
			video_nums = np.sum(vision_tokens == video_token_id)
			input_tokens = input_ids_masked.tolist()
			llm_pos_ids_list = []
			st = 0
			remain_images, remain_videos = image_nums, video_nums

			for _ in range(image_nums + video_nums):
				if image_token_id in input_tokens and remain_images > 0:
					ed_image = input_tokens.index(image_token_id, st)
				else:
					ed_image = len(input_tokens) + 1
				if video_token_id in input_tokens and remain_videos > 0:
					ed_video = input_tokens.index(video_token_id, st)
				else:
					ed_video = len(input_tokens) + 1
				if ed_image < ed_video:
					t, h, w = (
						image_grid_thw[image_index][0],
						image_grid_thw[image_index][1],
						image_grid_thw[image_index][2],
					)
					image_index += 1
					remain_images -= 1
					ed = ed_image
				else:
					t, h, w = (
						video_grid_thw[video_index][0],
						video_grid_thw[video_index][1],
						video_grid_thw[video_index][2],
					)
					video_index += 1
					remain_videos -= 1
					ed = ed_video
				llm_grid_t, llm_grid_h, llm_grid_w = (
					int(t),
					int(h) // spatial_merge_size,
					int(w) // spatial_merge_size,
				)
				text_len = ed - st

				st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
				llm_pos_ids_list.append(
					np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
				)

				t_index = (
					np.arange(llm_grid_t)
					.reshape(-1, 1)
					.repeat(llm_grid_h * llm_grid_w, axis=1)
					.flatten()
				)
				h_index = (
					np.arange(llm_grid_h)
					.reshape(1, -1, 1)
					.repeat(llm_grid_t, axis=0)
					.repeat(llm_grid_w, axis=2)
					.flatten()
				)
				w_index = (
					np.arange(llm_grid_w)
					.reshape(1, 1, -1)
					.repeat(llm_grid_t, axis=0)
					.repeat(llm_grid_h, axis=1)
					.flatten()
				)
				llm_pos_ids_list.append(
					np.stack([t_index, h_index, w_index]) + text_len + st_idx
				)
				st = ed + llm_grid_t * llm_grid_h * llm_grid_w

			if st < len(input_tokens):
				st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
				text_len = len(input_tokens) - st
				llm_pos_ids_list.append(
					np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
				)

			llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
			position_ids[:, i, attention_mask[i] == 1] = llm_positions
			mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

		mrope_position_deltas = np.array(mrope_position_deltas).reshape(-1, 1)
		return position_ids, mrope_position_deltas
	else:
		if attention_mask is not None:
			position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
			position_ids = jnp.where(attention_mask == 0, 1, position_ids)
			position_ids = jnp.expand_dims(position_ids, axis=0).repeat(3, axis=0)
			max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True)
			mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
		else:
			position_ids = (
				np.arange(input_ids.shape[1])
				.reshape(1, 1, -1)
				.repeat(3, axis=0)
				.repeat(input_ids.shape[0], axis=1)
			)
			mrope_position_deltas = np.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)

		return position_ids, mrope_position_deltas


@flax.struct.dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
	"""
	Base class for Qwen2VL causal language model (or autoregressive) outputs.
	"""

	loss: tp.Optional[chex.Array] = None
	logits: chex.Array = None
	past_key_values: tp.Optional[tp.List[chex.Array]] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array]] = None
	attentions: tp.Optional[tp.Tuple[chex.Array]] = None
	rope_deltas: tp.Optional[chex.Array] = None


def create_attention_mask(cu_seqlens, seq_length, dtype):
	"""
	Creates an attention mask matrix.

	Args:
	    cu_seqlens: Cumulative sequence lengths.
	    seq_length: Length of each sequence.
	    dtype: Data type of the mask.

	Returns:
	    Attention mask matrix.
	"""
	attention_mask = jnp.full(
		(1, seq_length, seq_length),
		jnp.finfo(dtype).min,
		dtype=dtype,
	)

	mask_updates = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

	for i in range(1, len(cu_seqlens)):
		start_idx = cu_seqlens[i - 1]
		end_idx = cu_seqlens[i]
		mask_updates = mask_updates.at[
			...,
			start_idx:end_idx,
			start_idx:end_idx,
		].set(0)

	attention_mask = jax.lax.dynamic_update_slice(attention_mask, mask_updates, (0, 0, 0))
	return attention_mask


# some of my garbage ideas but they always endup workin
# TODO: Fix this structure somehow
@partial(jax.jit, static_argnames=["TKN_ID"])
def jax_scatter(sec_embeds, ids, fir_embeds, TKN_ID):
	image_embeds = sec_embeds.astype(fir_embeds.dtype)

	image_indices = (
		jnp.where(
			jnp.broadcast_to(
				jnp.expand_dims(ids == TKN_ID, axis=-1), fir_embeds.shape
			).reshape(-1),
			size=fir_embeds.size,
			fill_value=-1,
		)[0]
		+ 1
	)
	flatten_emb = fir_embeds.reshape(-1)
	flatten_img_emb = image_embeds.reshape(-1)[: len(image_indices)]
	flatten_emb = jnp.pad(flatten_emb, (1, 0))
	flatten_img_emb = jnp.pad(
		flatten_img_emb,
		(0, flatten_emb.size - flatten_img_emb.size),
		# this will default be known as 0 so it wont be used anyway
	)
	image_indices = jnp.pad(
		image_indices,
		(0, flatten_emb.size - image_indices.size),
		# this will default be known as 0 so it wont be used anyway
	)
	scattered_embeds = flatten_emb.at[image_indices].set(flatten_img_emb)[1:]
	fir_embeds = scattered_embeds.reshape(fir_embeds.shape)
	return fir_embeds


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
	return output.squeeze(0)


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

		kernel_size = (temporal_patch_size, patch_size, patch_size)
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
		hidden_states = jnp.transpose(
			hidden_states.reshape(
				-1,
				self.in_channels,
				self.temporal_patch_size,
				self.patch_size,
				self.patch_size,
			),
			(0, 2, 3, 4, 1),
		)
		hidden_states = self.proj(hidden_states.astype(self.dtype))
		hidden_states = hidden_states.reshape(-1, self.embed_dim)
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
		self.ln_q = nn.LayerNorm(
			context_dim,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.mlp = [
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
		]

	def __call__(self, x: chex.Array) -> chex.Array:
		x = self.ln_q(x).reshape(-1, self.hidden_size)
		for mlp in self.mlp:  # make easy attach work with no effort
			x = mlp(x)
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
			use_bias=True,
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
		q, k, v = map(
			lambda x: x.squeeze(0),
			jnp.split(
				self.qkv(hidden_states)
				.reshape(seq_length, 3, self.num_heads, -1)
				.transpose(1, 0, 2, 3),  # seq spl nhd fea -> spl seq nhd fea
				3,
				0,
			),
		)
		q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
		k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

		# q = jnp.expand_dims(q, 0)
		# k = jnp.expand_dims(k, 0)
		# v = jnp.expand_dims(v, 0)
		# attention_mask = jnp.full(
		# 	(1, seq_length, seq_length),
		# 	jnp.finfo(q.dtype).min,
		# 	dtype=q.dtype,
		# )
		# for i in range(1, len(cu_seqlens)):
		# 	mask = attention_mask.at[
		# 		...,
		# 		cu_seqlens[i - 1] : cu_seqlens[i],
		# 		cu_seqlens[i - 1] : cu_seqlens[i],
		# 	].set(0)
		# 	attention_mask = mask
		row_ids = jnp.arange(seq_length)[None, None, :]
		col_ids = jnp.arange(seq_length)[None, :, None]
		starts = cu_seqlens[:-1][:, None, None]
		ends = cu_seqlens[1:][:, None, None]
		is_valid = (
			(row_ids >= starts) & (row_ids < ends) & (col_ids >= starts) & (col_ids < ends)
		)
		combined_mask = jnp.any(is_valid, axis=0)
		attention_mask = jnp.where(combined_mask, 0.0, jnp.finfo(q.dtype).min)
		# attention_mask = jnp.expand_dims(attention_mask, 0)
		# attn_output = self.attention_performer(
		# 	query_states=q,
		# 	key_states=k,
		# 	value_states=v,
		# 	bias=attention_mask,
		# 	attention_mask=None,
		# 	causal=True,
		# 	dropout_rng=self.rngs.params(),
		# 	query_sequence_length=q.shape[1],
		# 	key_value_sequence_length=k.shape[1],
		# 	uses_cache=None,
		# 	segment_ids=None,
		# 	causal_mask=None,
		# )
		# attn_output = self.proj(
		# 	attn_output.attention_outputs.reshape(seq_length, -1),
		# )
		# return attn_output
		q = q.swapaxes(0, 1)
		k = k.swapaxes(0, 1)
		v = v.swapaxes(0, 1)
		attn_weights = jnp.matmul(q, k.swapaxes(1, 2)) / math.sqrt(self.head_dim)
		attn_weights = attn_weights + attention_mask
		attn_weights = jax.nn.softmax(
			attn_weights.astype(jnp.float32),
			axis=-1,
		).astype(q.dtype)
		attn_output = jnp.matmul(attn_weights, v)
		attn_output = attn_output.swapaxes(0, 1)
		attn_output = attn_output.reshape(seq_length, -1)
		attn_output = self.proj(attn_output)
		return attn_output


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
			config=config,
			dim=config.embed_dim,
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

	def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> chex.Array:
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
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		hidden_states = self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)

		return hidden_states


class Qwen2VLAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: Qwen2VLConfig,
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
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.q_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
			use_bias=True,
		)
		self.k_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
			use_bias=True,
		)
		self.v_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
			use_bias=True,
		)
		self.o_proj = linear_class(
			config.num_attention_heads * self.head_dim,
			config.hidden_size,
			rngs=rngs,
			use_bias=False,
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
		if position_ids.ndim == 3:
			position_ids = position_ids[0]
			# cond vision gen issue will be fixed with no mem issue.
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


class Qwen2VLDecoderLayer(nn.Module):
	def __init__(
		self,
		config: Qwen2VLConfig,
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
		cache_view: tp.Optional[TransformerCacheView] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
		frequencies: tp.Optional[chex.Array] = None,
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


@register_module(
	TaskType.BASE_VISION,
	config=Qwen2VLConfig,
	model_type="qwen2_vl",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["ln_q", "norm1", "norm2"],
)
class Qwen2VisionTransformerPretrainedModel(EasyDeLBaseModule):
	config_class = Qwen2VLVisionConfig

	def __init__(
		self,
		config: Qwen2VLConfig,
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

		self.patch_embed = PatchEmbed(
			patch_size=config.patch_size,
			temporal_patch_size=config.temporal_patch_size,
			in_channels=config.in_channels,
			embed_dim=config.embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.spatial_merge_size = config.spatial_merge_size
		head_dim = config.embed_dim // config.num_heads
		self._head_dim_ro = head_dim // 2

		self.blocks = [
			Qwen2VLVisionBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(config.depth)
		]

		self.merger = PatchMerger(
			dim=config.hidden_size,
			context_dim=config.embed_dim,
			spatial_merge_size=config.spatial_merge_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def get_dtype(self) -> jnp.dtype:
		return self.blocks[0].mlp.fc2.kernel.value.dtype

	def rot_pos_emb(self, grid_thw, max_grid_size):
		pos_ids = []

		for t, h, w in grid_thw:
			# Create height position ids
			hpos_ids = jnp.arange(h)
			hpos_ids = jnp.expand_dims(hpos_ids, 1)
			hpos_ids = jnp.broadcast_to(hpos_ids, (h, w))

			# Reshape and permute height positions
			hpos_ids = hpos_ids.reshape(
				h // self.spatial_merge_size,
				self.spatial_merge_size,
				w // self.spatial_merge_size,
				self.spatial_merge_size,
			)
			hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3))
			hpos_ids = hpos_ids.flatten()

			# Create width position ids
			wpos_ids = jnp.arange(w)
			wpos_ids = jnp.expand_dims(wpos_ids, 0)
			wpos_ids = jnp.broadcast_to(wpos_ids, (h, w))

			# Reshape and permute width positions
			wpos_ids = wpos_ids.reshape(
				h // self.spatial_merge_size,
				self.spatial_merge_size,
				w // self.spatial_merge_size,
				self.spatial_merge_size,
			)
			wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3))
			wpos_ids = wpos_ids.flatten()

			# Stack and repeat
			stacked = jnp.stack([hpos_ids, wpos_ids], axis=-1)
			repeated = jnp.repeat(stacked, t, axis=1)
			pos_ids.append(repeated)

		pos_ids = jnp.concatenate(pos_ids, axis=0)
		rotary_pos_emb_full = jnp.outer(
			jnp.arange(0, max_grid_size, dtype="f4"),
			1.0
			/ (
				10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)
			),
		)
		# Index into embeddings and flatten
		rotary_pos_emb = jnp.take(rotary_pos_emb_full, pos_ids, axis=0)
		rotary_pos_emb = rotary_pos_emb.reshape(pos_ids.shape[0], -1)

		return rotary_pos_emb

	def __call__(
		self,
		hidden_states: chex.Array,
		grid_thw: chex.Array,
		max_grid_size,
	) -> chex.Array:
		hidden_states = self.patch_embed(hidden_states)
		rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)

		grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
		repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
		cu_seqlens = jnp.cumsum(repeated, dtype="i4")
		cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)
		for block in self.blocks:
			hidden_states = block(
				hidden_states,
				cu_seqlens=cu_seqlens,
				rotary_pos_emb=rotary_pos_emb,
			)

		return self.merger(hidden_states)


@register_module(
	TaskType.BASE_MODULE,
	config=Qwen2VLConfig,
	model_type="qwen2_vl",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["ln_q", "norm1", "norm2"],
)
class Qwen2VLModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: Qwen2VLConfig,
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

		self.embed_tokens = nn.Embed(
			num_embeddings=self.config.vocab_size,
			features=self.config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			rngs=rngs,
		)

		self.layers = [
			Qwen2VLDecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(self.config.num_hidden_layers)
		]
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
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
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
			outputs = (hidden_states, all_attentions)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	TaskType.IMAGE_TEXT_TO_TEXT,
	config=Qwen2VLConfig,
	model_type="qwen2_vl",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["ln_q", "norm1", "norm2"],
)
class Qwen2VLForConditionalGeneration(EasyDeLBaseModule):
	def __init__(
		self,
		config: Qwen2VLConfig,
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

		self.visual = Qwen2VisionTransformerPretrainedModel(
			config.vision_config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.model = Qwen2VLModel(
			config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def get_output_embeddings(self):
		return self.lm_head

	def get_decoder(self):
		return self.model

	def __call__(
		self,
		input_ids: chex.Array = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		pixel_values: tp.Optional[chex.Array] = None,
		pixel_values_videos: tp.Optional[chex.Array] = None,
		image_grid_thw: tp.Optional[tuple] = None,
		video_grid_thw: tp.Optional[tuple] = None,
		rope_deltas: tp.Optional[chex.Array] = None,
		image_max_grid_size: int = None,
		video_max_grid_size: int = None,
	) -> tp.Union[tp.Tuple, Qwen2VLCausalLMOutputWithPast]:
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

		if inputs_embeds is None:
			inputs_embeds = self.model.embed_tokens(input_ids)
			if pixel_values is not None:
				pixel_values = pixel_values.astype(self.visual.get_dtype())
				image_embeds = self.visual(
					pixel_values,
					grid_thw=np.array(image_grid_thw),
					max_grid_size=image_max_grid_size,
				)
				inputs_embeds = jax_scatter(
					image_embeds,
					input_ids,
					inputs_embeds,
					self.config.image_token_id,
				)

			if pixel_values_videos is not None:
				pixel_values_videos = pixel_values_videos.astype(self.visual.get_dtype())
				video_embeds = self.visual(
					pixel_values_videos,
					grid_thw=np.array(video_grid_thw),
					max_grid_size=video_max_grid_size,
				)

				inputs_embeds = jax_scatter(
					video_embeds,
					input_ids,
					inputs_embeds,
					self.config.video_token_id,
				)

		if (
			position_ids is None
			and input_ids is not None
			and (attention_mask is None or attention_mask.ndim == 2)
		):
			if past_key_values is not None or rope_deltas is None:
				warnings.warn(
					"You shouldn't be here (make sure to call `prepare_inputs_for_call`)",
					stacklevel=1,
				)
				position_ids, rope_deltas = get_rope_index(
					input_ids=input_ids,
					image_grid_thw=image_grid_thw,
					video_grid_thw=video_grid_thw,
					attention_mask=attention_mask,
					spatial_merge_size=self.visual.spatial_merge_size,
					image_token_id=self.config.image_token_id,
					video_token_id=self.config.video_token_id,
					vision_start_token_id=self.config.vision_start_token_id,
				)
			else:
				batch_size, sequence_length = inputs_embeds.shape[:2]
				position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
				position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)
		outputs = self.model(
			input_ids=None,
			position_ids=position_ids,
			attention_mask=attention_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		logits = self.lm_head(hidden_states)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return output

		return Qwen2VLCausalLMOutputWithPast(
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			rope_deltas=rope_deltas,
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		past_key_values=None,
		attention_mask=None,
		inputs_embeds=None,
		position_ids=None,
		pixel_values=None,
		pixel_values_videos=None,
		image_grid_thw=None,
		video_grid_thw=None,
		**kwargs,
	):
		batch_size, seq_length = input_ids.shape

		if past_key_values is None:
			past_key_values = self.init_cache(batch_size, max_length)

		if inputs_embeds is not None:
			model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
		else:
			model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			extended_attention_mask = jax.lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)
		model_inputs.update(
			{
				"position_ids": position_ids,
				"past_key_values": past_key_values,
				"attention_mask": extended_attention_mask,
				"pixel_values": pixel_values,
				"pixel_values_videos": pixel_values_videos,
				"image_grid_thw": image_grid_thw,
				"video_grid_thw": video_grid_thw,
			}
		)
		return self.prepare_inputs_for_call(**model_inputs)

	def prepare_inputs_for_call(
		self,
		image_grid_thw: tp.Optional[chex.Array] = None,
		video_grid_thw: tp.Optional[chex.Array] = None,
		image_max_grid_size: int = None,
		video_max_grid_size: int = None,
		drop_ids: bool = True,
		**others,
	):
		if image_grid_thw is not None:
			if image_max_grid_size is None:
				image_max_grid_size = int(np.max(image_grid_thw[:, 1:]).item())
			image_grid_thw = tuple(map(tuple, np.asarray(image_grid_thw)))

		if video_grid_thw is not None:
			if video_max_grid_size is None:
				video_max_grid_size = int(np.max(video_grid_thw[:, 1:]).item())
			video_grid_thw = tuple(map(tuple, np.asarray(video_grid_thw)))
		attention_mask = others.get("attention_mask", None)
		rope_deltas = others.get("rope_deltas", None)
		position_ids = others.get("position_ids", None)

		if (
			position_ids is None
			and others.get("input_ids", None) is not None
			and (attention_mask is None or attention_mask.ndim == 2)
		):
			if (
				others.get("past_key_values", None) is not None
				or others.get("rope_deltas", None) is None
			):
				position_ids, rope_deltas = get_rope_index(
					input_ids=others.get("input_ids"),
					image_grid_thw=image_grid_thw,
					video_grid_thw=video_grid_thw,
					attention_mask=attention_mask,
					spatial_merge_size=self.visual.spatial_merge_size,
					image_token_id=self.config.image_token_id,
					video_token_id=self.config.video_token_id,
					vision_start_token_id=self.config.vision_start_token_id,
				)
			else:
				batch_size, sequence_length = others.get("input_ids").shape
				position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
				position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)
		if drop_ids:
			others.pop("input_ids", None)
		others.update(
			dict(
				video_max_grid_size=video_max_grid_size,
				image_max_grid_size=image_max_grid_size,
				video_grid_thw=video_grid_thw,
				image_grid_thw=image_grid_thw,
				position_ids=position_ids,
				rope_deltas=rope_deltas,
			)
		)
		return others

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs["past_key_values"] = model_outputs.past_key_values
		model_kwargs["position_ids"] = model_kwargs["position_ids"][..., -1:] + 1
		return model_kwargs

	def get_static_arguments(self):
		return (
			"video_max_grid_size",
			"image_max_grid_size",
			"image_grid_thw",
			"video_grid_thw",
		)
