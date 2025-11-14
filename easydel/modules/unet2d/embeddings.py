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

"""Embedding layers for UNet2D models."""

import math
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx as nn


def get_sinusoidal_embeddings(
	timesteps: jnp.ndarray,
	embedding_dim: int,
	freq_shift: float = 1,
	min_timescale: float = 1,
	max_timescale: float = 1.0e4,
	flip_sin_to_cos: bool = False,
	scale: float = 1.0,
) -> jnp.ndarray:
	"""
	Generate sinusoidal positional embeddings.

	Args:
		timesteps: 1D tensor of N indices (timesteps)
		embedding_dim: Number of output channels
		freq_shift: Frequency shift
		min_timescale: Smallest time unit
		max_timescale: Largest time unit
		flip_sin_to_cos: Whether to flip sin/cos order
		scale: Scaling factor for embeddings

	Returns:
		Sinusoidal embeddings of shape [N, embedding_dim]
	"""
	assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
	assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"

	num_timescales = float(embedding_dim // 2)
	log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
	inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
	emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

	# Scale embeddings
	scaled_time = scale * emb

	if flip_sin_to_cos:
		signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
	else:
		signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)

	signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
	return signal


class Timesteps(nn.Module):
	"""
	Wrapper module for sinusoidal timestep embeddings.

	Args:
		dim: Timestep embedding dimension
		flip_sin_to_cos: Whether to flip sin/cos order
		freq_shift: Frequency shift
		scale: Scaling factor
	"""

	def __init__(
		self,
		dim: int = 32,
		flip_sin_to_cos: bool = False,
		freq_shift: float = 1.0,
		scale: int = 1,
	):
		self.dim = dim
		self.flip_sin_to_cos = flip_sin_to_cos
		self.freq_shift = freq_shift
		self.scale = scale

	def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
		"""Generate timestep embeddings."""
		return get_sinusoidal_embeddings(
			timesteps,
			embedding_dim=self.dim,
			flip_sin_to_cos=self.flip_sin_to_cos,
			freq_shift=self.freq_shift,
			scale=self.scale,
		)


class TimestepEmbedding(nn.Module):
	"""
	Timestep embedding module that projects timestep features to hidden dimension.

	Args:
		in_channels: Input feature dimension
		time_embed_dim: Output embedding dimension
		act_fn: Activation function name
		out_dim: Optional different output dimension
		post_act_fn: Optional post-activation function
		dtype: Data type for computation
		param_dtype: Data type for parameters
		precision: Precision for computation
		rngs: Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		time_embed_dim: int,
		act_fn: str = "silu",
		out_dim: int | None = None,
		post_act_fn: str | None = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.linear_1 = nn.Linear(
			in_channels,
			time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		if act_fn == "silu":
			self.act = nn.silu
		elif act_fn == "gelu":
			self.act = nn.gelu
		else:
			raise ValueError(f"Unknown activation function: {act_fn}")

		time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
		self.linear_2 = nn.Linear(
			time_embed_dim,
			time_embed_dim_out,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		if post_act_fn is None:
			self.post_act = None
		elif post_act_fn == "silu":
			self.post_act = nn.silu
		elif post_act_fn == "gelu":
			self.post_act = nn.gelu
		else:
			raise ValueError(f"Unknown post-activation function: {post_act_fn}")

	def __call__(self, sample: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		sample = self.linear_1(sample)
		if self.act is not None:
			sample = self.act(sample)
		sample = self.linear_2(sample)
		if self.post_act is not None:
			sample = self.post_act(sample)
		return sample


class TextTimeEmbedding(nn.Module):
	"""
	Text-time embedding for SDXL-style models.

	Combines text embeddings with time embeddings for additional conditioning.

	Args:
		text_embed_dim: Dimension of text embeddings
		time_embed_dim: Dimension of time embeddings
		num_heads: Number of attention heads (unused, kept for compatibility)
		dtype: Data type for computation
		param_dtype: Data type for parameters
		precision: Precision for computation
		rngs: Random number generators
	"""

	def __init__(
		self,
		text_embed_dim: int,
		time_embed_dim: int,
		num_heads: int = 64,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.linear_1 = nn.Linear(
			text_embed_dim,
			time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.act = nn.silu
		self.linear_2 = nn.Linear(
			time_embed_dim,
			time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, text_embeds: jnp.ndarray, time_embeds: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			text_embeds: Text embeddings from encoder
			time_embeds: Timestep embeddings

		Returns:
			Combined text-time embeddings
		"""
		# Concatenate text and time embeddings
		hidden_states = jnp.concatenate([text_embeds, time_embeds], axis=-1)
		hidden_states = self.linear_1(hidden_states)
		hidden_states = self.act(hidden_states)
		hidden_states = self.linear_2(hidden_states)
		return hidden_states


class CombinedTimestepTextEmbedding(nn.Module):
	"""
	Combined timestep and text projection embeddings for SDXL.

	Args:
		embedding_dim: Output embedding dimension
		pooled_projection_dim: Dimension of pooled text projections
		dtype: Data type for computation
		param_dtype: Data type for parameters
		precision: Precision for computation
		rngs: Random number generators
	"""

	def __init__(
		self,
		embedding_dim: int,
		pooled_projection_dim: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.time_proj = TimestepEmbedding(
			in_channels=embedding_dim,
			time_embed_dim=embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.text_proj = nn.Linear(
			pooled_projection_dim,
			embedding_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, timestep: jnp.ndarray, pooled_projection: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			timestep: Timestep tensor
			pooled_projection: Pooled text projections

		Returns:
			Combined timestep-text embeddings
		"""
		timestep_emb = self.time_proj(timestep)
		pooled_emb = self.text_proj(pooled_projection)
		pooled_emb = nn.silu(pooled_emb)
		conditioning = timestep_emb + pooled_emb
		return conditioning
