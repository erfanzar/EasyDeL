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

"""Attention modules for UNet2D diffusion models."""

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.layers.attention import FlexibleAttentionModule


class GEGLU(nn.Module):
	"""
	GEGLU activation function.

	Implements the Gated Linear Unit with GELU activation from:
	https://arxiv.org/abs/2002.05202

	Args:
		dim_in: Input dimension
		dim_out: Output dimension
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		dim_in: int,
		dim_out: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.proj = nn.Linear(
			dim_in,
			dim_out * 2,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		hidden_states = self.proj(hidden_states)
		hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
		return hidden_linear * nn.gelu(hidden_gelu)


class FeedForward(nn.Module):
	"""
	Feed-forward network with GEGLU activation.

	Args:
		dim: Input/output dimension
		dim_out: Output dimension (if None, same as dim)
		mult: Multiplier for inner dimension
		dropout: Dropout rate
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		dim: int,
		dim_out: int | None = None,
		mult: int = 4,
		dropout: float = 0.0,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		inner_dim = int(dim * mult)
		dim_out = dim_out if dim_out is not None else dim

		self.net_0 = GEGLU(
			dim,
			inner_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(dropout, rngs=rngs)
		self.net_2 = nn.Linear(
			inner_dim,
			dim_out,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
		"""Forward pass."""
		hidden_states = self.net_0(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.net_2(hidden_states)
		return hidden_states


class BasicTransformerBlock(nn.Module):
	"""
	Basic transformer block with self-attention, cross-attention, and feed-forward.

	Args:
		dim: Hidden dimension
		num_attention_heads: Number of attention heads
		attention_head_dim: Dimension per attention head
		dropout: Dropout rate
		only_cross_attention: Whether to only use cross-attention
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		dim: int,
		num_attention_heads: int,
		attention_head_dim: int,
		dropout: float = 0.0,
		only_cross_attention: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.only_cross_attention = only_cross_attention
		self.dropout = dropout

		# Self-attention (or first cross-attention if only_cross_attention)
		self.attn1 = FlexibleAttentionModule(
			num_q_heads=num_attention_heads,
			num_kv_heads=num_attention_heads,
			head_dims=attention_head_dim,
			dropout_rate=dropout,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm1 = nn.LayerNorm(
			dim,
			epsilon=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Cross-attention
		self.attn2 = FlexibleAttentionModule(
			num_q_heads=num_attention_heads,
			num_kv_heads=num_attention_heads,
			head_dims=attention_head_dim,
			dropout_rate=dropout,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm2 = nn.LayerNorm(
			dim,
			epsilon=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Feed-forward
		self.ff = FeedForward(
			dim,
			dropout=dropout,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.norm3 = nn.LayerNorm(
			dim,
			epsilon=1e-5,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		context: jnp.ndarray | None = None,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor [batch, seq_len, dim]
			context: Context for cross-attention [batch, ctx_len, dim]
			deterministic: Whether to use deterministic mode (no dropout)

		Returns:
			Output tensor [batch, seq_len, dim]
		"""
		# Self-attention (or cross-attention if only_cross_attention)
		residual = hidden_states
		hidden_states = self.norm1(hidden_states)

		if self.only_cross_attention:
			# Use context for both Q and KV
			attn_output = self.attn1(
				hidden_states,
				context,
				deterministic=deterministic,
			)
		else:
			# Self-attention
			attn_output = self.attn1(
				hidden_states,
				hidden_states,
				deterministic=deterministic,
			)

		hidden_states = attn_output + residual

		# Cross-attention
		if context is not None:
			residual = hidden_states
			hidden_states = self.norm2(hidden_states)
			attn_output = self.attn2(
				hidden_states,
				context,
				deterministic=deterministic,
			)
			hidden_states = attn_output + residual

		# Feed-forward
		residual = hidden_states
		hidden_states = self.norm3(hidden_states)
		hidden_states = self.ff(hidden_states, deterministic=deterministic)
		hidden_states = hidden_states + residual

		return hidden_states


class Transformer2DModel(nn.Module):
	"""
	2D Transformer model for spatial transformations in diffusion models.

	Args:
		num_attention_heads: Number of attention heads
		attention_head_dim: Dimension per attention head
		in_channels: Number of input channels
		num_layers: Number of transformer layers
		dropout: Dropout rate
		norm_num_groups: Number of groups for group normalization
		use_linear_projection: Whether to use linear projection instead of conv
		only_cross_attention: Whether to only use cross-attention
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		num_attention_heads: int,
		attention_head_dim: int,
		in_channels: int,
		num_layers: int = 1,
		dropout: float = 0.0,
		norm_num_groups: int = 32,
		use_linear_projection: bool = False,
		only_cross_attention: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.use_linear_projection = use_linear_projection
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
		inner_dim = num_attention_heads * attention_head_dim

		# Group norm
		self.norm = nn.GroupNorm(
			num_groups=norm_num_groups,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Projection in
		if use_linear_projection:
			self.proj_in = nn.Linear(
				in_channels,
				inner_dim,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)
		else:
			self.proj_in = nn.Conv(
				in_channels,
				inner_dim,
				kernel_size=(1, 1),
				strides=(1, 1),
				padding="VALID",
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)

		# Transformer blocks
		self.transformer_blocks = [
			BasicTransformerBlock(
				inner_dim,
				num_attention_heads,
				attention_head_dim,
				dropout=dropout,
				only_cross_attention=only_cross_attention,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(num_layers)
		]

		# Projection out
		if use_linear_projection:
			self.proj_out = nn.Linear(
				inner_dim,
				in_channels,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)
		else:
			self.proj_out = nn.Conv(
				inner_dim,
				in_channels,
				kernel_size=(1, 1),
				strides=(1, 1),
				padding="VALID",
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray | None = None,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor [batch, height, width, channels]
			encoder_hidden_states: Context tensor [batch, seq_len, dim]
			deterministic: Whether to use deterministic mode

		Returns:
			Output tensor [batch, height, width, channels]
		"""
		batch, height, width, channels = hidden_states.shape
		residual = hidden_states

		# Normalize
		hidden_states = self.norm(hidden_states)

		# Project in
		if self.use_linear_projection:
			hidden_states = hidden_states.reshape(batch, height * width, channels)
			hidden_states = self.proj_in(hidden_states)
		else:
			hidden_states = self.proj_in(hidden_states)
			hidden_states = hidden_states.reshape(batch, height * width, -1)

		# Transformer blocks
		for block in self.transformer_blocks:
			hidden_states = block(
				hidden_states,
				encoder_hidden_states,
				deterministic=deterministic,
			)

		# Project out
		if self.use_linear_projection:
			hidden_states = self.proj_out(hidden_states)
			hidden_states = hidden_states.reshape(batch, height, width, channels)
		else:
			hidden_states = hidden_states.reshape(batch, height, width, -1)
			hidden_states = self.proj_out(hidden_states)

		return hidden_states + residual
