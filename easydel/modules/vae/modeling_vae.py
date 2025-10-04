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

# JAX implementation of Variational Autoencoder for latent diffusion models

import math
import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import ModelOutput

from .vae_configuration import VAEConfig


@auto_pytree
class DecoderOutput(ModelOutput):
	"""
	Output of decoding method.

	Args:
		sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
			The decoded output sample from the last layer of the model.
	"""

	sample: chex.Array


@auto_pytree
class AutoencoderKLOutput(ModelOutput):
	"""
	Output of AutoencoderKL encoding method.

	Args:
		latent_dist (`DiagonalGaussianDistribution`):
			Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
			`DiagonalGaussianDistribution` allows for sampling latents from the distribution.
	"""

	latent_dist: "DiagonalGaussianDistribution"


class Upsample2D(nn.Module):
	"""
	Flax NNX implementation of 2D Upsample layer

	Args:
		in_channels (`int`):
			Input channels
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.conv = nn.Conv(
			in_features=in_channels,
			out_features=in_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, hidden_states: chex.Array) -> chex.Array:
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
	Flax NNX implementation of 2D Downsample layer

	Args:
		in_channels (`int`):
			Input channels
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.conv = nn.Conv(
			in_features=in_channels,
			out_features=in_channels,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding="VALID",
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, hidden_states: chex.Array) -> chex.Array:
		pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
		hidden_states = jnp.pad(hidden_states, pad_width=pad)
		hidden_states = self.conv(hidden_states)
		return hidden_states


class ResnetBlock2D(nn.Module):
	"""
	Flax NNX implementation of 2D Resnet Block.

	Args:
		in_channels (`int`):
			Input channels
		out_channels (`int`):
			Output channels
		dropout (:obj:`float`, *optional*, defaults to 0.0):
			Dropout rate
		groups (:obj:`int`, *optional*, defaults to `32`):
			The number of groups to use for group norm.
		use_nin_shortcut (:obj:`bool`, *optional*, defaults to `None`):
			Whether to use `nin_shortcut`. This activates a new layer inside ResNet block
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int | None = None,
		dropout: float = 0.0,
		groups: int = 32,
		use_nin_shortcut: bool | None = None,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.out_channels = in_channels if out_channels is None else out_channels
		self.dropout_rate = dropout
		self.groups = groups
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.norm1 = nn.GroupNorm(
			num_groups=groups,
			epsilon=1e-6,
			dtype=dtype,
			rngs=rngs,
		)
		self.conv1 = nn.Conv(
			in_features=in_channels,
			out_features=self.out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.norm2 = nn.GroupNorm(
			num_groups=groups,
			epsilon=1e-6,
			dtype=dtype,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(dropout, rngs=rngs)
		self.conv2 = nn.Conv(
			in_features=self.out_channels,
			out_features=self.out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		use_nin_shortcut = (
			in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut
		)

		self.conv_shortcut = None
		if use_nin_shortcut:
			self.conv_shortcut = nn.Conv(
				in_features=in_channels,
				out_features=self.out_channels,
				kernel_size=(1, 1),
				strides=(1, 1),
				padding="VALID",
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

	def __call__(self, hidden_states: chex.Array, deterministic: bool = True) -> chex.Array:
		residual = hidden_states
		hidden_states = self.norm1(hidden_states)
		hidden_states = nn.swish(hidden_states)
		hidden_states = self.conv1(hidden_states)

		hidden_states = self.norm2(hidden_states)
		hidden_states = nn.swish(hidden_states)
		hidden_states = self.dropout(hidden_states, deterministic=deterministic)
		hidden_states = self.conv2(hidden_states)

		if self.conv_shortcut is not None:
			residual = self.conv_shortcut(residual)

		return hidden_states + residual


class AttentionBlock(nn.Module):
	"""
	Flax NNX Convolutional based multi-head attention block for diffusion-based VAE.

	Args:
		channels (:obj:`int`):
			Input channels
		num_head_channels (:obj:`int`, *optional*, defaults to `None`):
			Number of attention heads
		num_groups (:obj:`int`, *optional*, defaults to `32`):
			The number of groups to use for group norm
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		channels: int,
		num_head_channels: int | None = None,
		num_groups: int = 32,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.channels = channels
		self.num_head_channels = num_head_channels
		self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
		self.num_groups = num_groups
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		dense = partial(
			nn.Linear,
			in_features=channels,
			out_features=channels,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.group_norm = nn.GroupNorm(
			num_groups=num_groups,
			epsilon=1e-6,
			dtype=dtype,
			rngs=rngs,
		)
		self.query = dense()
		self.key = dense()
		self.value = dense()

		self.proj_attn = dense()

	def transpose_for_scores(self, projection: chex.Array) -> chex.Array:
		new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
		# move heads to 2nd position (B, T, H * D) -> (B, T, H, D)
		new_projection = projection.reshape(new_projection_shape)
		# (B, T, H, D) -> (B, H, T, D)
		new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
		return new_projection

	def __call__(self, hidden_states: chex.Array) -> chex.Array:
		residual = hidden_states
		batch, height, width, channels = hidden_states.shape

		hidden_states = self.group_norm(hidden_states)

		hidden_states = hidden_states.reshape((batch, height * width, channels))

		query = self.query(hidden_states)
		key = self.key(hidden_states)
		value = self.value(hidden_states)

		# transpose
		query = self.transpose_for_scores(query)
		key = self.transpose_for_scores(key)
		value = self.transpose_for_scores(value)

		# compute attentions
		scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
		attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)

		attn_weights = nn.softmax(attn_weights, axis=-1)

		# attend to values
		hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)

		hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
		new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
		hidden_states = hidden_states.reshape(new_hidden_states_shape)

		hidden_states = self.proj_attn(hidden_states)
		hidden_states = hidden_states.reshape((batch, height, width, channels))
		hidden_states = hidden_states + residual
		return hidden_states


class DownEncoderBlock2D(nn.Module):
	"""
	Flax NNX Resnet blocks-based Encoder block for diffusion-based VAE.

	Args:
		in_channels (:obj:`int`):
			Input channels
		out_channels (:obj:`int`):
			Output channels
		dropout (:obj:`float`, *optional*, defaults to 0.0):
			Dropout rate
		num_layers (:obj:`int`, *optional*, defaults to 1):
			Number of Resnet layer block
		resnet_groups (:obj:`int`, *optional*, defaults to `32`):
			The number of groups to use for the Resnet block group norm
		add_downsample (:obj:`bool`, *optional*, defaults to `True`):
			Whether to add downsample layer
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_groups: int = 32,
		add_downsample: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dropout = dropout
		self.num_layers = num_layers
		self.resnet_groups = resnet_groups
		self.add_downsample = add_downsample
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		resnets = []
		for i in range(num_layers):
			in_ch = in_channels if i == 0 else out_channels

			res_block = ResnetBlock2D(
				in_channels=in_ch,
				out_channels=out_channels,
				dropout=dropout,
				groups=resnet_groups,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			resnets.append(res_block)
		self.resnets = resnets

		if add_downsample:
			self.downsamplers_0 = Downsample2D(
				out_channels,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

	def __call__(self, hidden_states: chex.Array, deterministic: bool = True) -> chex.Array:
		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, deterministic=deterministic)

		if self.add_downsample:
			hidden_states = self.downsamplers_0(hidden_states)

		return hidden_states


class UpDecoderBlock2D(nn.Module):
	"""
	Flax NNX Resnet blocks-based Decoder block for diffusion-based VAE.

	Args:
		in_channels (:obj:`int`):
			Input channels
		out_channels (:obj:`int`):
			Output channels
		dropout (:obj:`float`, *optional*, defaults to 0.0):
			Dropout rate
		num_layers (:obj:`int`, *optional*, defaults to 1):
			Number of Resnet layer block
		resnet_groups (:obj:`int`, *optional*, defaults to `32`):
			The number of groups to use for the Resnet block group norm
		add_upsample (:obj:`bool`, *optional*, defaults to `True`):
			Whether to add upsample layer
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_groups: int = 32,
		add_upsample: bool = True,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dropout = dropout
		self.num_layers = num_layers
		self.resnet_groups = resnet_groups
		self.add_upsample = add_upsample
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		resnets = []
		for i in range(num_layers):
			in_ch = in_channels if i == 0 else out_channels
			res_block = ResnetBlock2D(
				in_channels=in_ch,
				out_channels=out_channels,
				dropout=dropout,
				groups=resnet_groups,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			resnets.append(res_block)

		self.resnets = resnets

		if add_upsample:
			self.upsamplers_0 = Upsample2D(
				out_channels,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

	def __call__(self, hidden_states: chex.Array, deterministic: bool = True) -> chex.Array:
		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, deterministic=deterministic)

		if self.add_upsample:
			hidden_states = self.upsamplers_0(hidden_states)

		return hidden_states


class UNetMidBlock2D(nn.Module):
	"""
	Flax NNX Unet Mid-Block module.

	Args:
		in_channels (:obj:`int`):
			Input channels
		dropout (:obj:`float`, *optional*, defaults to 0.0):
			Dropout rate
		num_layers (:obj:`int`, *optional*, defaults to 1):
			Number of Resnet layer block
		resnet_groups (:obj:`int`, *optional*, defaults to `32`):
			The number of groups to use for the Resnet and Attention block group norm
		num_attention_heads (:obj:`int`, *optional*, defaults to `1`):
			Number of attention heads for each attention block
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_groups: int = 32,
		num_attention_heads: int | None = 1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.dropout = dropout
		self.num_layers = num_layers
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

		# there is always at least one resnet
		resnets = [
			ResnetBlock2D(
				in_channels=in_channels,
				out_channels=in_channels,
				dropout=dropout,
				groups=resnet_groups,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
		]

		attentions = []

		for _ in range(num_layers):
			attn_block = AttentionBlock(
				channels=in_channels,
				num_head_channels=num_attention_heads,
				num_groups=resnet_groups,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			attentions.append(attn_block)

			res_block = ResnetBlock2D(
				in_channels=in_channels,
				out_channels=in_channels,
				dropout=dropout,
				groups=resnet_groups,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			resnets.append(res_block)

		self.resnets = resnets
		self.attentions = attentions

	def __call__(self, hidden_states: chex.Array, deterministic: bool = True) -> chex.Array:
		hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			hidden_states = attn(hidden_states)
			hidden_states = resnet(hidden_states, deterministic=deterministic)

		return hidden_states


class Encoder(nn.Module):
	"""
	Flax NNX Implementation of VAE Encoder.

	Args:
		in_channels (:obj:`int`, *optional*, defaults to 3):
			Input channels
		out_channels (:obj:`int`, *optional*, defaults to 3):
			Output channels
		down_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
			DownEncoder block type
		block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
			Tuple containing the number of output channels for each block
		layers_per_block (:obj:`int`, *optional*, defaults to `2`):
			Number of Resnet layer for each block
		norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
			norm num group
		act_fn (:obj:`str`, *optional*, defaults to `silu`):
			Activation function
		double_z (:obj:`bool`, *optional*, defaults to `False`):
			Whether to double the last output channels
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			Parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		down_block_types: tp.Tuple[str, ...] = ("DownEncoderBlock2D",),
		block_out_channels: tp.Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		double_z: bool = False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.down_block_types = down_block_types
		self.block_out_channels = block_out_channels
		self.layers_per_block = layers_per_block
		self.norm_num_groups = norm_num_groups
		self.act_fn = act_fn
		self.double_z = double_z
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		# in
		self.conv_in = nn.Conv(
			in_features=in_channels,
			out_features=block_out_channels[0],
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# downsampling
		down_blocks = []
		output_channel = block_out_channels[0]
		for i, _ in enumerate(down_block_types):
			input_channel = output_channel
			output_channel = block_out_channels[i]
			is_final_block = i == len(block_out_channels) - 1

			down_block = DownEncoderBlock2D(
				in_channels=input_channel,
				out_channels=output_channel,
				num_layers=layers_per_block,
				resnet_groups=norm_num_groups,
				add_downsample=not is_final_block,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			down_blocks.append(down_block)
		self.down_blocks = down_blocks

		# middle
		self.mid_block = UNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			num_attention_heads=None,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# end
		conv_out_channels = 2 * out_channels if double_z else out_channels
		self.conv_norm_out = nn.GroupNorm(
			num_groups=norm_num_groups,
			epsilon=1e-6,
			dtype=dtype,
			rngs=rngs,
		)
		self.conv_out = nn.Conv(
			in_features=block_out_channels[-1],
			out_features=conv_out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, sample: chex.Array, deterministic: bool = True) -> chex.Array:
		# in
		sample = self.conv_in(sample)

		# downsampling
		for block in self.down_blocks:
			sample = block(sample, deterministic=deterministic)

		# middle
		sample = self.mid_block(sample, deterministic=deterministic)

		# end
		sample = self.conv_norm_out(sample)
		sample = nn.swish(sample)
		sample = self.conv_out(sample)

		return sample


class Decoder(nn.Module):
	"""
	Flax NNX Implementation of VAE Decoder.

	Args:
		in_channels (:obj:`int`, *optional*, defaults to 3):
			Input channels
		out_channels (:obj:`int`, *optional*, defaults to 3):
			Output channels
		up_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
			UpDecoder block type
		block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
			Tuple containing the number of output channels for each block
		layers_per_block (:obj:`int`, *optional*, defaults to `2`):
			Number of Resnet layer for each block
		norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
			norm num group
		act_fn (:obj:`str`, *optional*, defaults to `silu`):
			Activation function
		dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			parameters `dtype`
		param_dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
			parameters `dtype`
		precision (:obj:`lax.PrecisionLike`, *optional*):
			Precision for computations
		rngs (`nn.Rngs`):
			Random number generators
	"""

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		up_block_types: tp.Tuple[str, ...] = ("UpDecoderBlock2D",),
		block_out_channels: tp.Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.up_block_types = up_block_types
		self.block_out_channels = block_out_channels
		self.layers_per_block = layers_per_block
		self.norm_num_groups = norm_num_groups
		self.act_fn = act_fn
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		# z to block_in
		self.conv_in = nn.Conv(
			in_features=in_channels,
			out_features=block_out_channels[-1],
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# middle
		self.mid_block = UNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			num_attention_heads=None,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# upsampling
		reversed_block_out_channels = list(reversed(block_out_channels))
		output_channel = reversed_block_out_channels[0]
		up_blocks = []
		for i, _ in enumerate(up_block_types):
			prev_output_channel = output_channel
			output_channel = reversed_block_out_channels[i]

			is_final_block = i == len(block_out_channels) - 1

			up_block = UpDecoderBlock2D(
				in_channels=prev_output_channel,
				out_channels=output_channel,
				num_layers=layers_per_block + 1,
				resnet_groups=norm_num_groups,
				add_upsample=not is_final_block,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			up_blocks.append(up_block)
			prev_output_channel = output_channel

		self.up_blocks = up_blocks

		# end
		self.conv_norm_out = nn.GroupNorm(
			num_groups=norm_num_groups,
			epsilon=1e-6,
			dtype=dtype,
			rngs=rngs,
		)
		self.conv_out = nn.Conv(
			in_features=block_out_channels[0],
			out_features=out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, sample: chex.Array, deterministic: bool = True) -> chex.Array:
		# z to block_in
		sample = self.conv_in(sample)

		# middle
		sample = self.mid_block(sample, deterministic=deterministic)

		# upsampling
		for block in self.up_blocks:
			sample = block(sample, deterministic=deterministic)

		sample = self.conv_norm_out(sample)
		sample = nn.swish(sample)
		sample = self.conv_out(sample)

		return sample


class DiagonalGaussianDistribution:
	"""
	Diagonal Gaussian Distribution for VAE latent space.

	Args:
		parameters (chex.Array): Concatenated mean and logvar parameters
		deterministic (bool): If True, use mean without sampling
	"""

	def __init__(self, parameters: chex.Array, deterministic: bool = False):
		# Last axis to account for channels-last
		self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
		self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
		self.deterministic = deterministic
		self.std = jnp.exp(0.5 * self.logvar)
		self.var = jnp.exp(self.logvar)
		if self.deterministic:
			self.var = self.std = jnp.zeros_like(self.mean)

	def sample(self, key: jax.Array) -> chex.Array:
		"""Sample from the distribution using the provided RNG key."""
		return self.mean + self.std * jax.random.normal(key, self.mean.shape)

	def kl(self, other: "DiagonalGaussianDistribution | None" = None) -> chex.Array:
		"""Compute KL divergence."""
		if self.deterministic:
			return jnp.array([0.0])

		if other is None:
			return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])

		return 0.5 * jnp.sum(
			jnp.square(self.mean - other.mean) / other.var
			+ self.var / other.var
			- 1.0
			- self.logvar
			+ other.logvar,
			axis=[1, 2, 3],
		)

	def nll(self, sample: chex.Array, axis: list[int] = [1, 2, 3]) -> chex.Array:
		"""Compute negative log likelihood."""
		if self.deterministic:
			return jnp.array([0.0])

		logtwopi = jnp.log(2.0 * jnp.pi)
		return 0.5 * jnp.sum(
			logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis
		)

	def mode(self) -> chex.Array:
		"""Return the mode (mean) of the distribution."""
		return self.mean


@register_module(
	"AutoencoderKL",
	config=VAEConfig,
	model_type="vae",
)
class AutoencoderKL(EasyDeLBaseModule):
	"""
	Flax NNX implementation of a VAE model with KL loss for decoding latent representations.

	This model uses Flax NNX and inherits from EasyDeLBaseModule.

	Args:
		config (VAEConfig):
			Model configuration.
		dtype (jnp.dtype, *optional*, defaults to `jnp.float32`):
			The dtype of the computation.
		param_dtype (jnp.dtype, *optional*, defaults to `jnp.float32`):
			The dtype of the parameters.
		precision (lax.PrecisionLike, *optional*):
			Numerical precision for operations.
		rngs (nn.Rngs):
			Random number generators for initialization.
	"""

	config_class = VAEConfig

	def __init__(
		self,
		config: VAEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: lax.PrecisionLike = None,
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

		self.encoder = Encoder(
			in_channels=config.in_channels,
			out_channels=config.latent_channels,
			down_block_types=config.down_block_types,
			block_out_channels=config.block_out_channels,
			layers_per_block=config.layers_per_block,
			act_fn=config.act_fn,
			norm_num_groups=config.norm_num_groups,
			double_z=True,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.decoder = Decoder(
			in_channels=config.latent_channels,
			out_channels=config.out_channels,
			up_block_types=config.up_block_types,
			block_out_channels=config.block_out_channels,
			layers_per_block=config.layers_per_block,
			norm_num_groups=config.norm_num_groups,
			act_fn=config.act_fn,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.quant_conv = nn.Conv(
			in_features=2 * config.latent_channels,
			out_features=2 * config.latent_channels,
			kernel_size=(1, 1),
			strides=(1, 1),
			padding="VALID",
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.post_quant_conv = nn.Conv(
			in_features=config.latent_channels,
			out_features=config.latent_channels,
			kernel_size=(1, 1),
			strides=(1, 1),
			padding="VALID",
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def encode(
		self,
		sample: chex.Array,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> AutoencoderKLOutput | tp.Tuple[DiagonalGaussianDistribution]:
		"""
		Encode images into latent representations.

		Args:
			sample (chex.Array): Input images of shape (batch_size, channels, height, width)
			deterministic (bool): Whether to use deterministic mode (no dropout)
			return_dict (bool): Whether to return a dict or tuple

		Returns:
			AutoencoderKLOutput or tuple containing the latent distribution
		"""
		sample = jnp.transpose(sample, (0, 2, 3, 1))

		hidden_states = self.encoder(sample, deterministic=deterministic)
		moments = self.quant_conv(hidden_states)
		posterior = DiagonalGaussianDistribution(moments)

		if not return_dict:
			return (posterior,)

		return AutoencoderKLOutput(latent_dist=posterior)

	def decode(
		self,
		latents: chex.Array,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> DecoderOutput | tp.Tuple[chex.Array]:
		"""
		Decode latent representations into images.

		Args:
			latents (chex.Array): Latent representations
			deterministic (bool): Whether to use deterministic mode (no dropout)
			return_dict (bool): Whether to return a dict or tuple

		Returns:
			DecoderOutput or tuple containing the decoded images
		"""
		if latents.shape[-1] != self.config.latent_channels:
			latents = jnp.transpose(latents, (0, 2, 3, 1))

		hidden_states = self.post_quant_conv(latents)
		hidden_states = self.decoder(hidden_states, deterministic=deterministic)

		hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

		if not return_dict:
			return (hidden_states,)

		return DecoderOutput(sample=hidden_states)

	def __call__(
		self,
		sample: chex.Array,
		sample_posterior: bool = False,
		deterministic: bool = True,
		return_dict: bool = True,
	) -> DecoderOutput | tp.Tuple[chex.Array]:
		"""
		Forward pass through the VAE.

		Args:
			sample (chex.Array): Input images of shape (batch_size, channels, height, width)
			sample_posterior (bool): Whether to sample from the posterior distribution
			deterministic (bool): Whether to use deterministic mode (no dropout)
			return_dict (bool): Whether to return a dict or tuple

		Returns:
			DecoderOutput or tuple containing the reconstructed images
		"""
		posterior = self.encode(sample, deterministic=deterministic, return_dict=return_dict)
		if sample_posterior:
			rng = self.rngs.params()  # Use params rng for sampling
			hidden_states = posterior.latent_dist.sample(rng)
		else:
			hidden_states = posterior.latent_dist.mode()

		sample = self.decode(hidden_states, deterministic=deterministic, return_dict=return_dict).sample

		if not return_dict:
			return (sample,)

		return DecoderOutput(sample=sample)
