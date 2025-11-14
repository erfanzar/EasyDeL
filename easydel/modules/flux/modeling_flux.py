# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
Flux Transformer implementation for state-of-the-art image generation.

This module implements the Flux Transformer architecture, a diffusion model that uses
rectified flow, rotary position embeddings (RoPE), and dual transformer blocks for
high-quality image generation with text conditioning.
"""

import math
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import auto_remat

from .flux_configuration import FluxConfig


def get_1d_rotary_pos_embed(
	dim: int,
	pos: tp.Union[jnp.ndarray, int],
	theta: float = 10000.0,
	use_real: bool = True,
) -> jnp.ndarray:
	"""
	Precompute the frequency tensor for rotary position embeddings (RoPE).

	Args:
		dim: Embedding dimension (must be even).
		pos: Position indices, either an integer or array.
		theta: Base value for frequency calculation.
		use_real: If True, return real-valued rotary matrix (Flux-style).

	Returns:
		Rotary position embeddings.
	"""
	assert dim % 2 == 0

	if isinstance(pos, int):
		pos = jnp.arange(pos)

	freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim))
	freqs = jnp.outer(pos, freqs)

	if use_real:
		# Flux-style: return rotation matrix components
		freqs_cos = jnp.cos(freqs)
		freqs_sin = jnp.sin(freqs)
		out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)
	else:
		# Complex exponential style
		out = jnp.exp(1j * freqs)

	return out


def apply_rope(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Apply rotary position embeddings to query and key tensors.

	Args:
		xq: Query tensor of shape [B, H, L, D].
		xk: Key tensor of shape [B, H, L, D].
		freqs_cis: Rotary position embeddings.

	Returns:
		Tuple of (rotated_query, rotated_key).
	"""
	xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
	xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

	xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
	xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

	return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)


class FluxPosEmbed(nn.Module):
	"""Position embedding module for Flux using multi-dimensional RoPE."""

	def __init__(
		self,
		theta: int,
		axes_dim: list[int],
		dtype: jnp.dtype = jnp.float32,
	):
		self.theta = theta
		self.axes_dim = axes_dim
		self.dtype = dtype

	def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
		"""
		Generate position embeddings.

		Args:
			ids: Position IDs of shape [..., n_axes].

		Returns:
			Position embeddings.
		"""
		n_axes = ids.shape[-1]
		out_freqs = []
		pos = ids.astype(self.dtype)

		for i in range(n_axes):
			out = get_1d_rotary_pos_embed(self.axes_dim[i], pos[..., i], theta=self.theta, use_real=True)
			out_freqs.append(out)

		out_freqs = jnp.concatenate(out_freqs, axis=1)
		return out_freqs


class FluxTimestepEmbedding(nn.Module):
	"""Time step embedding module using sinusoidal encoding and MLP."""

	def __init__(
		self,
		time_embed_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.time_embed_dim = time_embed_dim
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.linear_1 = nn.Linear(
			time_embed_dim,
			time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.linear_2 = nn.Linear(
			time_embed_dim,
			time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, temb: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass with SiLU activation."""
		temb = self.linear_1(temb)
		temb = nn.silu(temb)
		temb = self.linear_2(temb)
		return temb


class PixArtAlphaTextProjection(nn.Module):
	"""Projects text embeddings with GELU activation."""

	def __init__(
		self,
		in_features: int,
		hidden_size: int,
		out_features: int = None,
		act_fn: str = "gelu_tanh",
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		if out_features is None:
			out_features = hidden_size

		self.linear_1 = nn.Linear(
			in_features,
			hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.act_fn = act_fn
		self.linear_2 = nn.Linear(
			hidden_size,
			out_features,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, caption: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass with activation."""
		hidden_states = self.linear_1(caption)
		if self.act_fn == "gelu_tanh":
			hidden_states = nn.gelu(hidden_states, approximate="tanh")
		elif self.act_fn == "silu":
			hidden_states = nn.silu(hidden_states)
		else:
			hidden_states = nn.gelu(hidden_states)
		hidden_states = self.linear_2(hidden_states)
		return hidden_states


class CombinedTimestepTextProjEmbeddings(nn.Module):
	"""Combined timestep and text projection embeddings (for flux-schnell)."""

	def __init__(
		self,
		embedding_dim: int,
		pooled_projection_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.time_proj = FluxTimestepEmbedding(
			time_embed_dim=embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.text_proj = PixArtAlphaTextProjection(
			in_features=pooled_projection_dim,
			hidden_size=embedding_dim,
			act_fn="silu",
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, timestep: jnp.ndarray, pooled_projection: jnp.ndarray) -> jnp.ndarray:
		"""Combine timestep and text embeddings."""
		timestep_emb = self.time_proj(timestep)
		pooled_proj = self.text_proj(pooled_projection)
		conditioning = timestep_emb + pooled_proj
		return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
	"""Combined timestep, guidance, and text projection embeddings (for flux-dev)."""

	def __init__(
		self,
		embedding_dim: int,
		pooled_projection_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.time_proj = FluxTimestepEmbedding(
			time_embed_dim=embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.guidance_proj = FluxTimestepEmbedding(
			time_embed_dim=embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.text_proj = PixArtAlphaTextProjection(
			in_features=pooled_projection_dim,
			hidden_size=embedding_dim,
			act_fn="silu",
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self, timestep: jnp.ndarray, guidance: jnp.ndarray, pooled_projection: jnp.ndarray
	) -> jnp.ndarray:
		"""Combine timestep, guidance, and text embeddings."""
		timestep_emb = self.time_proj(timestep.astype(pooled_projection.dtype))
		guidance_emb = self.guidance_proj(guidance.astype(pooled_projection.dtype))
		time_guidance_emb = timestep_emb + guidance_emb
		pooled_proj = self.text_proj(pooled_projection)
		conditioning = time_guidance_emb + pooled_proj
		return conditioning


class AdaLayerNormZero(nn.Module):
	"""Adaptive Layer Normalization Zero for double transformer blocks."""

	def __init__(
		self,
		embedding_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		eps: float = 1e-6,
		*,
		rngs: nn.Rngs,
	):
		self.embedding_dim = embedding_dim
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.eps = eps

		self.lin = nn.Linear(
			6 * embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm = nn.LayerNorm(
			embedding_dim,
			epsilon=eps,
			use_bias=False,
			use_scale=False,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> tuple:
		"""Apply adaptive layer norm and return modulation parameters."""
		emb = nn.silu(emb)
		emb = self.lin(emb)
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(emb[:, None, :], 6, axis=-1)
		x = self.norm(x)
		x = x * (1 + scale_msa) + shift_msa
		return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
	"""Adaptive Layer Normalization Zero for single transformer blocks."""

	def __init__(
		self,
		embedding_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		eps: float = 1e-6,
		*,
		rngs: nn.Rngs,
	):
		self.embedding_dim = embedding_dim
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.eps = eps

		self.lin = nn.Linear(
			3 * embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm = nn.LayerNorm(
			embedding_dim,
			epsilon=eps,
			use_bias=False,
			use_scale=False,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> tuple:
		"""Apply adaptive layer norm and return gate."""
		emb = nn.silu(emb)
		emb = self.lin(emb)
		shift_msa, scale_msa, gate_msa = jnp.split(emb[:, None, :], 3, axis=-1)
		x = self.norm(x)
		x = x * (1 + scale_msa) + shift_msa
		return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
	"""Adaptive Layer Normalization Continuous for output normalization."""

	def __init__(
		self,
		embedding_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		eps: float = 1e-5,
		elementwise_affine: bool = True,
		*,
		rngs: nn.Rngs,
	):
		self.embedding_dim = embedding_dim
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.eps = eps
		self.elementwise_affine = elementwise_affine

		self.dense = nn.Linear(
			embedding_dim * 2,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm = nn.LayerNorm(
			embedding_dim,
			epsilon=eps,
			use_bias=elementwise_affine,
			use_scale=elementwise_affine,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, x: jnp.ndarray, conditioning_embedding: jnp.ndarray) -> jnp.ndarray:
		"""Apply continuous adaptive layer norm."""
		emb = nn.silu(conditioning_embedding)
		emb = self.dense(emb)
		shift, scale = jnp.split(emb, 2, axis=1)
		x = self.norm(x)
		x = (1 + scale[:, None, :]) * x + shift[:, None, :]
		return x


class FluxAttention(nn.Module):
	"""
	Flux attention module with RoPE and dual QKV projections.

	This module handles both image and text attention with separate QKV projections
	for each, RMS normalization, and rotary position embeddings.
	"""

	def __init__(
		self,
		query_dim: int,
		heads: int = 8,
		dim_head: int = 64,
		qkv_bias: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.query_dim = query_dim
		self.heads = heads
		self.dim_head = dim_head
		self.inner_dim = dim_head * heads
		self.scale = dim_head**-0.5

		# Image QKV projection
		self.i_qkv = nn.Linear(
			self.inner_dim * 3,
			use_bias=qkv_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Encoder (text) QKV projection
		self.e_qkv = nn.Linear(
			self.inner_dim * 3,
			use_bias=qkv_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Output projections
		self.i_proj = nn.Linear(
			query_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.e_proj = nn.Linear(
			query_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# RMS normalizations
		self.query_norm = nn.RMSNorm(dim_head, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
		self.key_norm = nn.RMSNorm(dim_head, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
		self.encoder_query_norm = nn.RMSNorm(dim_head, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
		self.encoder_key_norm = nn.RMSNorm(dim_head, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray = None,
		image_rotary_emb: jnp.ndarray = None,
	) -> tuple[jnp.ndarray, jnp.ndarray]:
		"""
		Forward pass with dual attention (image and text).

		Args:
			hidden_states: Image features [B, L_img, D].
			encoder_hidden_states: Text features [B, L_txt, D].
			image_rotary_emb: Rotary position embeddings.

		Returns:
			Tuple of (image_attn_output, text_attn_output).
		"""
		# Process image QKV
		qkv_proj = self.i_qkv(hidden_states)
		B, L = hidden_states.shape[:2]
		H, D, K = self.heads, qkv_proj.shape[-1] // (self.heads * 3), 3
		qkv_proj = qkv_proj.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
		query_proj, key_proj, value_proj = qkv_proj

		query_proj = self.query_norm(query_proj)
		key_proj = self.key_norm(key_proj)

		if encoder_hidden_states is not None:
			# Process encoder (text) QKV
			encoder_qkv_proj = self.e_qkv(encoder_hidden_states)
			B, L = encoder_hidden_states.shape[:2]
			encoder_qkv_proj = encoder_qkv_proj.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
			encoder_query_proj, encoder_key_proj, encoder_value_proj = encoder_qkv_proj

			encoder_query_proj = self.encoder_query_norm(encoder_query_proj)
			encoder_key_proj = self.encoder_key_norm(encoder_key_proj)

			# Concatenate image and text
			query_proj = jnp.concatenate((encoder_query_proj, query_proj), axis=2)
			key_proj = jnp.concatenate((encoder_key_proj, key_proj), axis=2)
			value_proj = jnp.concatenate((encoder_value_proj, value_proj), axis=2)

		# Apply RoPE
		if image_rotary_emb is not None:
			image_rotary_emb = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)
			query_proj, key_proj = apply_rope(query_proj, key_proj, image_rotary_emb)

		# Reshape for attention computation
		query_proj = query_proj.transpose(0, 2, 1, 3).reshape(query_proj.shape[0], query_proj.shape[2], -1)
		key_proj = key_proj.transpose(0, 2, 1, 3).reshape(key_proj.shape[0], key_proj.shape[2], -1)
		value_proj = value_proj.transpose(0, 2, 1, 3).reshape(value_proj.shape[0], value_proj.shape[2], -1)

		# Compute attention (simple dot-product attention)
		attn_weights = jnp.einsum("bqd,bkd->bqk", query_proj, key_proj) * self.scale
		attn_weights = nn.softmax(attn_weights, axis=-1)
		attn_output = jnp.einsum("bqk,bkd->bqd", attn_weights, value_proj)

		context_attn_output = None

		if encoder_hidden_states is not None:
			# Split back into text and image
			context_attn_output, attn_output = (
				attn_output[:, : encoder_hidden_states.shape[1]],
				attn_output[:, encoder_hidden_states.shape[1] :],
			)

			attn_output = self.i_proj(attn_output)
			context_attn_output = self.e_proj(context_attn_output)
		else:
			attn_output = self.i_proj(attn_output)

		return attn_output, context_attn_output


class FluxTransformerBlock(nn.Module):
	"""
	Double transformer block processing both image and text features.

	This block uses adaptive layer normalization, dual attention, and separate MLPs
	for image and text modalities.
	"""

	def __init__(
		self,
		dim: int,
		num_attention_heads: int,
		attention_head_dim: int,
		qkv_bias: bool = False,
		mlp_ratio: float = 4.0,
		eps: float = 1e-6,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.dim = dim
		self.mlp_ratio = mlp_ratio

		self.img_norm1 = AdaLayerNormZero(
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			eps=eps,
			rngs=rngs,
		)
		self.txt_norm1 = AdaLayerNormZero(
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			eps=eps,
			rngs=rngs,
		)

		self.attn = FluxAttention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			qkv_bias=qkv_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.img_norm2 = nn.LayerNorm(
			dim,
			use_bias=False,
			use_scale=False,
			epsilon=eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		mlp_hidden_dim = int(dim * mlp_ratio)
		self.img_mlp = [
			nn.Linear(mlp_hidden_dim, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs),
			nn.Linear(dim, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs),
		]

		self.txt_norm2 = nn.LayerNorm(
			dim,
			use_bias=False,
			use_scale=False,
			epsilon=eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.txt_mlp = [
			nn.Linear(mlp_hidden_dim, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs),
			nn.Linear(dim, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs),
		]

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray,
		temb: jnp.ndarray,
		image_rotary_emb: jnp.ndarray = None,
	) -> tuple[jnp.ndarray, jnp.ndarray]:
		"""
		Forward pass.

		Args:
			hidden_states: Image features.
			encoder_hidden_states: Text features.
			temb: Time embeddings.
			image_rotary_emb: Rotary position embeddings.

		Returns:
			Tuple of (updated_image_features, updated_text_features).
		"""
		# Normalize and prepare for attention
		norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.img_norm1(hidden_states, emb=temb)
		norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.txt_norm1(
			encoder_hidden_states, emb=temb
		)

		# Attention
		attn_output, context_attn_output = self.attn(
			hidden_states=norm_hidden_states,
			encoder_hidden_states=norm_encoder_hidden_states,
			image_rotary_emb=image_rotary_emb,
		)

		# Process image features
		attn_output = gate_msa * attn_output
		hidden_states = hidden_states + attn_output
		norm_hidden_states = self.img_norm2(hidden_states)
		norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

		ff_output = self.img_mlp[0](norm_hidden_states)
		ff_output = nn.gelu(ff_output)
		ff_output = self.img_mlp[1](ff_output)
		ff_output = gate_mlp * ff_output
		hidden_states = hidden_states + ff_output

		# Process text features
		context_attn_output = c_gate_msa * context_attn_output
		encoder_hidden_states = encoder_hidden_states + context_attn_output
		norm_encoder_hidden_states = self.txt_norm2(encoder_hidden_states)
		norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

		context_ff_output = self.txt_mlp[0](norm_encoder_hidden_states)
		context_ff_output = nn.gelu(context_ff_output)
		context_ff_output = self.txt_mlp[1](context_ff_output)
		encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

		# Clip for fp16 stability
		if encoder_hidden_states.dtype == jnp.float16:
			encoder_hidden_states = jnp.clip(encoder_hidden_states, -65504, 65504)

		return hidden_states, encoder_hidden_states


class FluxSingleTransformerBlock(nn.Module):
	"""
	Single transformer block processing concatenated image and text features.

	This block uses adaptive layer normalization and processes the combined
	features through attention and MLP layers.
	"""

	def __init__(
		self,
		dim: int,
		num_attention_heads: int,
		attention_head_dim: int,
		mlp_ratio: float = 4.0,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
		*,
		rngs: nn.Rngs,
	):
		self.dim = dim
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
		self.mlp_ratio = mlp_ratio
		self.mlp_hidden_dim = int(dim * mlp_ratio)

		self.norm = AdaLayerNormZeroSingle(
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.linear1 = nn.Linear(
			dim * 3 + self.mlp_hidden_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.linear2 = nn.Linear(
			dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Attention norms
		self.query_norm = nn.RMSNorm(attention_head_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
		self.key_norm = nn.RMSNorm(attention_head_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
		self.scale = attention_head_dim**-0.5

	def __call__(
		self, hidden_states: jnp.ndarray, temb: jnp.ndarray, image_rotary_emb: jnp.ndarray = None
	) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Concatenated image and text features.
			temb: Time embeddings.
			image_rotary_emb: Rotary position embeddings.

		Returns:
			Updated features.
		"""
		residual = hidden_states
		norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
		qkv, mlp = jnp.split(self.linear1(norm_hidden_states), [3 * self.dim], axis=-1)

		# Reshape for attention
		B, L = hidden_states.shape[:2]
		H, D, K = self.num_attention_heads, qkv.shape[-1] // (self.num_attention_heads * 3), 3
		qkv_proj = qkv.reshape(B, L, K, H, D).transpose(2, 0, 3, 1, 4)
		q, k, v = qkv_proj

		q = self.query_norm(q)
		k = self.key_norm(k)

		# Apply RoPE
		if image_rotary_emb is not None:
			image_rotary_emb_reordered = rearrange(image_rotary_emb, "n d (i j) -> n d i j", i=2, j=2)
			q, k = apply_rope(q, k, image_rotary_emb_reordered)

		# Reshape for attention computation
		q = q.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
		k = k.transpose(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1)
		v = v.transpose(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)

		# Compute attention
		attn_weights = jnp.einsum("bqd,bkd->bqk", q, k) * self.scale
		attn_weights = nn.softmax(attn_weights, axis=-1)
		attn_output = jnp.einsum("bqk,bkd->bqd", attn_weights, v)

		# Combine attention and MLP
		attn_mlp = jnp.concatenate([attn_output, nn.gelu(mlp)], axis=2)
		hidden_states = self.linear2(attn_mlp)
		hidden_states = gate * hidden_states
		hidden_states = residual + hidden_states

		# Clip for fp16 stability
		if hidden_states.dtype == jnp.float16:
			hidden_states = jnp.clip(hidden_states, -65504, 65504)

		return hidden_states


@register_module("flux", TaskType.IMAGE_DIFFUSION)
class FluxTransformer2DModel(EasyDeLBaseModule):
	"""
	Flux Transformer 2D Model for state-of-the-art image generation.

	This model implements the Flux architecture with:
	- Dual transformer blocks (double and single)
	- Rotary position embeddings (RoPE)
	- Adaptive layer normalization
	- Text and image conditioning
	- Rectified flow for diffusion

	The model processes images in latent space with text conditioning from
	a text encoder (T5) and pooled text embeddings (CLIP).
	"""

	def __init__(
		self,
		config: FluxConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision = None,
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

		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.out_channels = config.in_channels
		self.inner_dim = config.num_attention_heads * config.attention_head_dim

		# Position embeddings
		self.pe_embedder = FluxPosEmbed(
			theta=config.theta,
			axes_dim=list(config.axes_dims_rope),
			dtype=dtype,
		)

		# Time and text embeddings
		text_time_guidance_cls = (
			CombinedTimestepGuidanceTextProjEmbeddings
			if config.guidance_embeds
			else CombinedTimestepTextProjEmbeddings
		)

		self.time_text_embed = text_time_guidance_cls(
			embedding_dim=self.inner_dim,
			pooled_projection_dim=config.pooled_projection_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Input projections
		self.txt_in = nn.Linear(
			self.inner_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.img_in = nn.Linear(
			self.inner_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Double transformer blocks
		self.double_blocks = [
			FluxTransformerBlock(
				dim=self.inner_dim,
				num_attention_heads=config.num_attention_heads,
				attention_head_dim=config.attention_head_dim,
				qkv_bias=config.qkv_bias,
				mlp_ratio=config.mlp_ratio,
				eps=config.eps,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(config.num_layers)
		]

		# Single transformer blocks
		self.single_blocks = [
			FluxSingleTransformerBlock(
				dim=self.inner_dim,
				num_attention_heads=config.num_attention_heads,
				attention_head_dim=config.attention_head_dim,
				mlp_ratio=config.mlp_ratio,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(config.num_single_layers)
		]

		# Output layers
		self.norm_out = AdaLayerNormContinuous(
			self.inner_dim,
			elementwise_affine=False,
			eps=config.eps,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.proj_out = nn.Linear(
			config.patch_size**2 * self.out_channels,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def timestep_embedding(
		self, t: jax.Array, dim: int, max_period: int = 10000, time_factor: float = 1000.0
	) -> jax.Array:
		"""
		Generate sinusoidal timestep embeddings.

		Args:
			t: Timesteps tensor.
			dim: Embedding dimension.
			max_period: Maximum period for sinusoidal encoding.
			time_factor: Scaling factor for timesteps.

		Returns:
			Timestep embeddings.
		"""
		t = time_factor * t
		half = dim // 2

		freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.bfloat16) / half).astype(
			dtype=t.dtype
		)

		args = t[:, None].astype(jnp.bfloat16) * freqs[None]
		embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

		if dim % 2:
			embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

		if jnp.issubdtype(t.dtype, jnp.floating):
			embedding = embedding.astype(t.dtype)

		return embedding

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray,
		pooled_projections: jnp.ndarray,
		timestep: jnp.ndarray,
		img_ids: jnp.ndarray,
		txt_ids: jnp.ndarray,
		guidance: jnp.ndarray = None,
		return_dict: bool = True,
		train: bool = False,
	) -> tp.Union[BaseModelOutput, tuple]:
		"""
		Forward pass.

		Args:
			hidden_states: Input latent images [B, H*W, C].
			encoder_hidden_states: Text embeddings from T5 [B, seq_len, 4096].
			pooled_projections: Pooled text embeddings from CLIP [B, 768].
			timestep: Diffusion timesteps [B].
			img_ids: Image position IDs [B, H*W, 3].
			txt_ids: Text position IDs [B, seq_len, 3].
			guidance: Guidance scale values [B] (only for flux-dev).
			return_dict: Whether to return a dict or tuple.
			train: Whether in training mode.

		Returns:
			Model output with denoised latents.
		"""
		# Project inputs
		hidden_states = self.img_in(hidden_states)

		# Create timestep embeddings
		timestep = self.timestep_embedding(timestep, 256)

		if self.config.guidance_embeds:
			guidance = self.timestep_embedding(guidance, 256) if guidance is not None else None
		else:
			guidance = None

		# Combine timestep and text embeddings
		temb = (
			self.time_text_embed(timestep, pooled_projections)
			if guidance is None
			else self.time_text_embed(timestep, guidance, pooled_projections)
		)

		# Project text inputs
		encoder_hidden_states = self.txt_in(encoder_hidden_states)

		# Handle position IDs
		if txt_ids.ndim == 3:
			txt_ids = txt_ids[0]
		if img_ids.ndim == 3:
			img_ids = img_ids[0]

		ids = jnp.concatenate((txt_ids, img_ids), axis=0)
		image_rotary_emb = self.pe_embedder(ids)

		# Process through double blocks
		for i, double_block in enumerate(self.double_blocks):
			if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
				hidden_states, encoder_hidden_states = auto_remat(
					double_block,
					policy=self.config.gradient_checkpointing,
				)(
					hidden_states=hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					temb=temb,
					image_rotary_emb=image_rotary_emb,
				)
			else:
				hidden_states, encoder_hidden_states = double_block(
					hidden_states=hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					temb=temb,
					image_rotary_emb=image_rotary_emb,
				)

		# Concatenate image and text features
		hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

		# Process through single blocks
		for i, single_block in enumerate(self.single_blocks):
			if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
				hidden_states = auto_remat(
					single_block,
					policy=self.config.gradient_checkpointing,
				)(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
			else:
				hidden_states = single_block(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)

		# Extract image features (remove text portion)
		hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

		# Output projection
		hidden_states = self.norm_out(hidden_states, temb)
		output = self.proj_out(hidden_states)

		if not return_dict:
			return (output,)

		return BaseModelOutput(last_hidden_state=output)
