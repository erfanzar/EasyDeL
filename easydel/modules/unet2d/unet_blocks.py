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

"""UNet building blocks for 2D diffusion models."""

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx as nn

from .attention import Transformer2DModel


class Upsample2D(nn.Module):
	"""
	2D upsampling layer.

	Args:
		channels: Number of channels
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		channels: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.conv = nn.Conv(
			channels,
			channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass with 2x upsampling."""
		batch, height, width, channels = hidden_states.shape
		hidden_states = jax.image.resize(
			hidden_states,
			shape=(batch, height * 2, width * 2, channels),
			method="nearest",
		)
		hidden_states = self.conv(hidden_states)
		return hidden_states


class Downsample2D(nn.Module):
	"""
	2D downsampling layer.

	Args:
		channels: Number of channels
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		channels: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		self.conv = nn.Conv(
			channels,
			channels,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass with 2x downsampling."""
		hidden_states = self.conv(hidden_states)
		return hidden_states


class ResnetBlock2D(nn.Module):
	"""
	2D ResNet block with time embedding.

	Args:
		in_channels: Number of input channels
		out_channels: Number of output channels
		temb_channels: Number of time embedding channels
		dropout: Dropout rate
		groups: Number of groups for group normalization
		use_nin_shortcut: Whether to use 1x1 conv for shortcut
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int | None = None,
		temb_channels: int = 512,
		dropout: float = 0.0,
		groups: int = 32,
		use_nin_shortcut: bool | None = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		out_channels = out_channels or in_channels
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.norm1 = nn.GroupNorm(
			num_groups=groups,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.conv1 = nn.Conv(
			in_channels,
			out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.time_emb_proj = nn.Linear(
			temb_channels,
			out_channels,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.norm2 = nn.GroupNorm(
			num_groups=groups,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.dropout = nn.Dropout(dropout, rngs=rngs)

		self.conv2 = nn.Conv(
			out_channels,
			out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		use_nin_shortcut = use_nin_shortcut if use_nin_shortcut is not None else (in_channels != out_channels)
		self.conv_shortcut = None
		if use_nin_shortcut:
			self.conv_shortcut = nn.Conv(
				in_channels,
				out_channels,
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
		temb: jnp.ndarray,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor [batch, height, width, channels]
			temb: Time embedding [batch, temb_channels]
			deterministic: Whether to use deterministic mode

		Returns:
			Output tensor [batch, height, width, out_channels]
		"""
		residual = hidden_states

		hidden_states = self.norm1(hidden_states)
		hidden_states = nn.silu(hidden_states)
		hidden_states = self.conv1(hidden_states)

		temb = nn.silu(temb)
		temb = self.time_emb_proj(temb)
		temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
		hidden_states = hidden_states + temb

		hidden_states = self.norm2(hidden_states)
		hidden_states = nn.silu(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.conv2(hidden_states)

		if self.conv_shortcut is not None:
			residual = self.conv_shortcut(residual)

		return hidden_states + residual


class CrossAttnDownBlock2D(nn.Module):
	"""
	Downsampling block with cross-attention.

	Args:
		in_channels: Number of input channels
		out_channels: Number of output channels
		temb_channels: Number of time embedding channels
		dropout: Dropout rate
		num_layers: Number of ResNet layers
		num_attention_heads: Number of attention heads
		attention_head_dim: Dimension per attention head
		add_downsample: Whether to add downsampling layer
		use_linear_projection: Whether to use linear projection in attention
		only_cross_attention: Whether to only use cross-attention
		transformer_layers_per_block: Number of transformer layers per block
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		num_attention_heads: int = 1,
		attention_head_dim: int | None = None,
		add_downsample: bool = True,
		use_linear_projection: bool = False,
		only_cross_attention: bool = False,
		transformer_layers_per_block: int = 1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		resnets = []
		attentions = []

		for i in range(num_layers):
			in_ch = in_channels if i == 0 else out_channels
			resnets.append(
				ResnetBlock2D(
					in_channels=in_ch,
					out_channels=out_channels,
					temb_channels=temb_channels,
					dropout=dropout,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

			attn_head_dim = attention_head_dim or out_channels // num_attention_heads
			attentions.append(
				Transformer2DModel(
					num_attention_heads=num_attention_heads,
					attention_head_dim=attn_head_dim,
					in_channels=out_channels,
					num_layers=transformer_layers_per_block,
					dropout=dropout,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.resnets = resnets
		self.attentions = attentions

		self.downsamplers = None
		if add_downsample:
			self.downsamplers = [
				Downsample2D(
					out_channels,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			]

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		temb: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray | None = None,
		deterministic: bool = True,
	) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor
			temb: Time embedding
			encoder_hidden_states: Encoder hidden states for cross-attention
			deterministic: Whether to use deterministic mode

		Returns:
			Tuple of (output, output_states)
		"""
		output_states = ()

		for resnet, attn in zip(self.resnets, self.attentions):
			hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
			hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
			output_states = output_states + (hidden_states,)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)
			output_states = output_states + (hidden_states,)

		return hidden_states, output_states


class DownBlock2D(nn.Module):
	"""
	Downsampling block without attention.

	Args:
		in_channels: Number of input channels
		out_channels: Number of output channels
		temb_channels: Number of time embedding channels
		dropout: Dropout rate
		num_layers: Number of ResNet layers
		add_downsample: Whether to add downsampling layer
		dtype: Data type
		param_dtype: Parameter data type
		precision: Computation precision
		rngs: Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		add_downsample: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		resnets = []

		for i in range(num_layers):
			in_ch = in_channels if i == 0 else out_channels
			resnets.append(
				ResnetBlock2D(
					in_channels=in_ch,
					out_channels=out_channels,
					temb_channels=temb_channels,
					dropout=dropout,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.resnets = resnets

		self.downsamplers = None
		if add_downsample:
			self.downsamplers = [
				Downsample2D(
					out_channels,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			]

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		temb: jnp.ndarray,
		deterministic: bool = True,
	) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
		"""Forward pass."""
		output_states = ()

		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
			output_states = output_states + (hidden_states,)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)
			output_states = output_states + (hidden_states,)

		return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
	"""
	Upsampling block with cross-attention.

	Similar to CrossAttnDownBlock2D but for upsampling path.
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		prev_output_channel: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		num_attention_heads: int = 1,
		attention_head_dim: int | None = None,
		add_upsample: bool = True,
		use_linear_projection: bool = False,
		only_cross_attention: bool = False,
		transformer_layers_per_block: int = 1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		resnets = []
		attentions = []

		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels

			resnets.append(
				ResnetBlock2D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					dropout=dropout,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

			attn_head_dim = attention_head_dim or out_channels // num_attention_heads
			attentions.append(
				Transformer2DModel(
					num_attention_heads=num_attention_heads,
					attention_head_dim=attn_head_dim,
					in_channels=out_channels,
					num_layers=transformer_layers_per_block,
					dropout=dropout,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.resnets = resnets
		self.attentions = attentions

		self.upsamplers = None
		if add_upsample:
			self.upsamplers = [
				Upsample2D(
					out_channels,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			]

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		res_hidden_states_tuple: tuple[jnp.ndarray, ...],
		temb: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray | None = None,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""Forward pass."""
		for resnet, attn in zip(self.resnets, self.attentions):
			# Pop residual hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]

			# Concatenate with skip connection
			hidden_states = jnp.concatenate([hidden_states, res_hidden_states], axis=-1)

			hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
			hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states)

		return hidden_states


class UpBlock2D(nn.Module):
	"""
	Upsampling block without attention.
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		prev_output_channel: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		add_upsample: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		resnets = []

		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels

			resnets.append(
				ResnetBlock2D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					dropout=dropout,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.resnets = resnets

		self.upsamplers = None
		if add_upsample:
			self.upsamplers = [
				Upsample2D(
					out_channels,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			]

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		res_hidden_states_tuple: tuple[jnp.ndarray, ...],
		temb: jnp.ndarray,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""Forward pass."""
		for resnet in self.resnets:
			# Pop residual hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]

			# Concatenate with skip connection
			hidden_states = jnp.concatenate([hidden_states, res_hidden_states], axis=-1)
			hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states)

		return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
	"""
	Middle block with cross-attention.
	"""

	def __init__(
		self,
		in_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		num_attention_heads: int = 1,
		attention_head_dim: int | None = None,
		use_linear_projection: bool = False,
		transformer_layers_per_block: int = 1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
		*,
		rngs: nn.Rngs,
	):
		# First ResNet
		resnets = [
			ResnetBlock2D(
				in_channels=in_channels,
				out_channels=in_channels,
				temb_channels=temb_channels,
				dropout=dropout,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
		]

		attentions = []

		for _ in range(num_layers):
			attn_head_dim = attention_head_dim or in_channels // num_attention_heads
			attentions.append(
				Transformer2DModel(
					num_attention_heads=num_attention_heads,
					attention_head_dim=attn_head_dim,
					in_channels=in_channels,
					num_layers=transformer_layers_per_block,
					dropout=dropout,
					use_linear_projection=use_linear_projection,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

			resnets.append(
				ResnetBlock2D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					dropout=dropout,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			)

		self.resnets = resnets
		self.attentions = attentions

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		temb: jnp.ndarray,
		encoder_hidden_states: jnp.ndarray | None = None,
		deterministic: bool = True,
	) -> jnp.ndarray:
		"""Forward pass."""
		hidden_states = self.resnets[0](hidden_states, temb, deterministic=deterministic)

		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
			hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

		return hidden_states
