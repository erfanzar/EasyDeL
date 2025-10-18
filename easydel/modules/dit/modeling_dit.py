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
DiT (Diffusion Transformer) implementation for image diffusion with rectified flow.

This module implements the Diffusion Transformer architecture for image generation,
supporting both unconditional and class-conditional generation with rectified flow.
"""

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import auto_remat, get_dot_general_by_bits
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear

from .dit_configuration import DiTConfig


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
	"""Apply adaptive layer norm modulation: (1 + scale) * x + shift."""
	return x * (1 + scale) + shift


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jnp.ndarray:
	"""
	Generate 2D sinusoidal positional embeddings.

	Args:
		embed_dim: Embedding dimension (must be divisible by 2)
		grid_size: Size of the spatial grid (grid_size x grid_size patches)

	Returns:
		Positional embeddings of shape [1, grid_size**2, embed_dim]
	"""
	assert embed_dim % 2 == 0

	# Create grid coordinates
	grid_h = jnp.arange(grid_size, dtype=jnp.float32)
	grid_w = jnp.arange(grid_size, dtype=jnp.float32)
	grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")  # (W, H)
	grid = jnp.stack(grid, axis=0)  # (2, H, W)
	grid = grid.reshape(2, -1)  # (2, H*W)

	# Generate embeddings for each dimension
	omega = jnp.arange(embed_dim // 4, dtype=jnp.float32)
	omega /= embed_dim / 4.0
	omega = 1.0 / (10000**omega)  # (D/4,)

	# Compute sinusoidal embeddings
	out = jnp.einsum("hw,d->hwd", grid, omega)  # (2, H*W, D/4)
	emb_sin = jnp.sin(out)  # (2, H*W, D/4)
	emb_cos = jnp.cos(out)  # (2, H*W, D/4)

	# Concatenate sin and cos for both H and W
	emb = jnp.concatenate([emb_sin, emb_cos], axis=-1)  # (2, H*W, D/2)
	emb = emb.reshape(2, -1, embed_dim // 2)
	emb = jnp.concatenate([emb[0], emb[1]], axis=-1)  # (H*W, D)

	return jnp.expand_dims(emb, 0)  # (1, H*W, D)


class TimestepEmbedding(nn.Module):
	"""Embeds scalar timesteps into vector representations using sinusoidal encoding."""

	def __init__(
		self,
		hidden_size: int,
		frequency_embedding_size: int = 256,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.hidden_size = hidden_size
		self.frequency_embedding_size = frequency_embedding_size
		self.dtype = dtype
		self.param_dtype = param_dtype

		# MLP to project frequency embeddings to hidden_size
		self.mlp = [
			nn.Linear(
				frequency_embedding_size,
				hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
			nn.Linear(
				hidden_size,
				hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
		]

	def timestep_embedding(self, t: jnp.ndarray, max_period: float = 10000.0) -> jnp.ndarray:
		"""Generate sinusoidal timestep embeddings."""
		t = jax.lax.convert_element_type(t, jnp.float32)
		half = self.frequency_embedding_size // 2
		freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
		args = t[:, None] * freqs[None]
		embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
		return embedding.astype(self.dtype)

	def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			t: Timesteps of shape [batch_size]

		Returns:
			Timestep embeddings of shape [batch_size, hidden_size]
		"""
		t_freq = self.timestep_embedding(t)
		t_emb = self.mlp[0](t_freq)
		t_emb = nn.silu(t_emb)
		t_emb = self.mlp[1](t_emb)
		return t_emb


class LabelEmbedding(nn.Module):
	"""Embeds class labels into vector representations."""

	def __init__(
		self,
		num_classes: int,
		hidden_size: int,
		dropout_prob: float = 0.1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob

		# +1 for unconditional class token
		self.embedding_table = nn.Embed(
			num_classes + 1,
			hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, labels: jnp.ndarray, force_drop_ids: jnp.ndarray | None = None) -> jnp.ndarray:
		"""
		Forward pass with classifier-free guidance dropout.

		Args:
			labels: Class labels of shape [batch_size]
			force_drop_ids: Optional boolean mask for dropping labels

		Returns:
			Label embeddings of shape [batch_size, hidden_size]
		"""
		use_dropout = self.dropout_prob > 0
		if (force_drop_ids is None) and use_dropout:
			# Randomly drop labels for classifier-free guidance
			drop_ids = jax.random.bernoulli(
				nn.rng_key("dropout"),
				self.dropout_prob,
				shape=labels.shape,
			)
		else:
			drop_ids = force_drop_ids if force_drop_ids is not None else jnp.zeros_like(labels, dtype=bool)

		# Use num_classes as the unconditional token
		labels = jnp.where(drop_ids, self.num_classes, labels)
		embeddings = self.embedding_table(labels)
		return embeddings


class PatchEmbed(nn.Module):
	"""Converts images into a sequence of patch embeddings."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		in_channels: int,
		hidden_size: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.image_size = image_size
		self.patch_size = patch_size
		self.in_channels = in_channels
		self.hidden_size = hidden_size
		self.num_patches = (image_size // patch_size) ** 2

		self.proj = nn.Conv(
			in_features=in_channels,
			out_features=hidden_size,
			kernel_size=(patch_size, patch_size),
			strides=(patch_size, patch_size),
			padding="VALID",
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			x: Images of shape [batch_size, height, width, channels]

		Returns:
			Patch embeddings of shape [batch_size, num_patches, hidden_size]
		"""
		B, H, W, C = x.shape
		assert H == self.image_size and W == self.image_size, \
			f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"
		assert C == self.in_channels, \
			f"Input channels ({C}) doesn't match model ({self.in_channels})"

		x = self.proj(x)  # [B, H', W', hidden_size]
		x = x.reshape(B, -1, self.hidden_size)  # [B, num_patches, hidden_size]
		return x


class DiTBlock(nn.Module):
	"""DiT transformer block with adaptive layer norm conditioning."""

	def __init__(
		self,
		config: DiTConfig,
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

		# Attention
		self.attn = FlexibleAttentionModule(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# MLP
		linear_cls = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.use_bias,
			rngs=rngs,
		)

		self.mlp = nn.Sequential(
			linear_cls(config.hidden_size, config.intermediate_size),
			nn.gelu if config.hidden_act == "gelu" else nn.silu,
			nn.Dropout(config.mlp_dropout, rngs=rngs) if config.mlp_dropout > 0 else lambda x: x,
			linear_cls(config.intermediate_size, config.hidden_size),
			nn.Dropout(config.mlp_dropout, rngs=rngs) if config.mlp_dropout > 0 else lambda x: x,
		)

		# Adaptive layer norm modulation
		self.adaLN_modulation = nn.Sequential(
			nn.silu,
			linear_cls(config.hidden_size, 6 * config.hidden_size),
		)

		# Layer norms
		self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
		self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		conditioning: jnp.ndarray,
		attention_mask: jnp.ndarray | None = None,
	) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor [batch_size, seq_len, hidden_size]
			conditioning: Conditioning vector [batch_size, hidden_size]
			attention_mask: Optional attention mask

		Returns:
			Output tensor [batch_size, seq_len, hidden_size]
		"""
		# Adaptive layer norm parameters
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
			self.adaLN_modulation(conditioning).chunk(6, axis=-1)

		# Attention block with adaptive LN
		normed = modulate(self.norm1(hidden_states), shift_msa[:, None, :], scale_msa[:, None, :])
		attn_output = self.attn(
			normed,
			attention_mask=attention_mask,
			causal_mask=None,
		)
		hidden_states = hidden_states + gate_msa[:, None, :] * attn_output

		# MLP block with adaptive LN
		normed = modulate(self.norm2(hidden_states), shift_mlp[:, None, :], scale_mlp[:, None, :])
		mlp_output = self.mlp(normed)
		hidden_states = hidden_states + gate_mlp[:, None, :] * mlp_output

		return hidden_states


class FinalLayer(nn.Module):
	"""Final layer that unpatchifies and outputs the denoised image/velocity."""

	def __init__(
		self,
		config: DiTConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype

		self.norm_final = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.layer_norm_eps,
			rngs=rngs,
		)

		self.linear = nn.Linear(
			config.hidden_size,
			config.patch_size * config.patch_size * config.out_channels,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.adaLN_modulation = nn.Sequential(
			nn.silu,
			nn.Linear(
				config.hidden_size,
				2 * config.hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
		)

	def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			x: Hidden states [batch_size, num_patches, hidden_size]
			conditioning: Conditioning vector [batch_size, hidden_size]

		Returns:
			Output tensor [batch_size, num_patches, patch_size**2 * out_channels]
		"""
		shift, scale = self.adaLN_modulation(conditioning).chunk(2, axis=-1)
		x = modulate(self.norm_final(x), shift[:, None, :], scale[:, None, :])
		x = self.linear(x)
		return x


@register_module("dit", TaskType.BASE_MODEL)
class DiTModel(EasyDeLBaseModule):
	"""
	DiT (Diffusion Transformer) base model.

	This model implements the core DiT architecture without the final
	unpatchification layer, useful for feature extraction.
	"""

	def __init__(
		self,
		config: DiTConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		# Patch embedding
		self.patch_embed = PatchEmbed(
			image_size=config.image_size,
			patch_size=config.patch_size,
			in_channels=config.in_channels,
			hidden_size=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Positional embedding (learned or fixed)
		grid_size = config.image_size // config.patch_size
		pos_embed = get_2d_sincos_pos_embed(config.hidden_size, grid_size)
		self.pos_embed = nn.Variable(pos_embed.astype(param_dtype))

		# Timestep embedding
		self.time_embed = TimestepEmbedding(
			hidden_size=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Label embedding
		if config.use_conditioning:
			self.label_embed = LabelEmbedding(
				num_classes=config.num_classes,
				hidden_size=config.hidden_size,
				dropout_prob=config.class_dropout_prob,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)

		# Transformer blocks
		self.blocks = [
			DiTBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(config.num_hidden_layers)
		]

	def __call__(
		self,
		pixel_values: jnp.ndarray,
		timesteps: jnp.ndarray,
		labels: jnp.ndarray | None = None,
		attention_mask: jnp.ndarray | None = None,
		return_dict: bool = True,
	) -> BaseModelOutput | tuple:
		"""
		Forward pass.

		Args:
			pixel_values: Input images [batch_size, height, width, channels]
			timesteps: Diffusion timesteps [batch_size]
			labels: Optional class labels [batch_size]
			attention_mask: Optional attention mask
			return_dict: Whether to return a ModelOutput object

		Returns:
			BaseModelOutput or tuple of hidden states
		"""
		# Embed patches
		hidden_states = self.patch_embed(pixel_values)

		# Add positional embedding
		hidden_states = hidden_states + self.pos_embed.value

		# Embed timesteps
		time_emb = self.time_embed(timesteps)

		# Embed labels and combine with timestep embedding
		if self.config.use_conditioning and labels is not None:
			label_emb = self.label_embed(labels)
			conditioning = time_emb + label_emb
		else:
			conditioning = time_emb

		# Apply transformer blocks
		for block in self.blocks:
			if self.config.gradient_checkpointing != "nothing_saveable":
				hidden_states = auto_remat(
					block,
					policy=self.config.gradient_checkpointing,
				)(hidden_states, conditioning, attention_mask)
			else:
				hidden_states = block(hidden_states, conditioning, attention_mask)

		if return_dict:
			return BaseModelOutput(last_hidden_state=hidden_states)
		return (hidden_states,)


@register_module("dit", TaskType.IMAGE_DIFFUSION)
class DiTForImageDiffusion(EasyDeLBaseModule):
	"""
	DiT model for image diffusion with rectified flow.

	This model adds the final unpatchification layer to convert transformer
	outputs back to image space for diffusion training.
	"""

	def __init__(
		self,
		config: DiTConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype

		# Base DiT model
		self.model = DiTModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Final layer
		self.final_layer = FinalLayer(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
		"""
		Convert patch embeddings back to image format.

		Args:
			x: Patch outputs [batch_size, num_patches, patch_size**2 * channels]

		Returns:
			Image tensor [batch_size, height, width, channels]
		"""
		B = x.shape[0]
		p = self.config.patch_size
		h = w = self.config.image_size // p
		c = self.config.out_channels

		x = x.reshape(B, h, w, p, p, c)
		x = jnp.einsum("bhwpqc->bhpwqc", x)
		x = x.reshape(B, h * p, w * p, c)
		return x

	def __call__(
		self,
		pixel_values: jnp.ndarray,
		timesteps: jnp.ndarray,
		labels: jnp.ndarray | None = None,
		attention_mask: jnp.ndarray | None = None,
		return_dict: bool = True,
	) -> BaseModelOutput | tuple:
		"""
		Forward pass.

		Args:
			pixel_values: Input images [batch_size, height, width, channels]
			timesteps: Diffusion timesteps [batch_size]
			labels: Optional class labels [batch_size]
			attention_mask: Optional attention mask
			return_dict: Whether to return a ModelOutput object

		Returns:
			Model outputs with predicted velocity/noise
		"""
		# Get hidden states from base model
		outputs = self.model(
			pixel_values=pixel_values,
			timesteps=timesteps,
			labels=labels,
			attention_mask=attention_mask,
			return_dict=True,
		)

		hidden_states = outputs.last_hidden_state

		# Get conditioning for final layer
		time_emb = self.model.time_embed(timesteps)
		if self.config.use_conditioning and labels is not None:
			label_emb = self.model.label_embed(labels)
			conditioning = time_emb + label_emb
		else:
			conditioning = time_emb

		# Apply final layer
		patch_outputs = self.final_layer(hidden_states, conditioning)

		# Unpatchify to image space
		predictions = self.unpatchify(patch_outputs)

		if return_dict:
			return BaseModelOutput(last_hidden_state=predictions)
		return (predictions,)
