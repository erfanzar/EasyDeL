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

"""UNet 2D Conditional model for diffusion."""

import typing as tp

import chex
import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import ModelOutput

from .embeddings import CombinedTimestepTextEmbedding, TimestepEmbedding, Timesteps
from .unet2d_configuration import UNet2DConfig
from .unet_blocks import (
	CrossAttnDownBlock2D,
	CrossAttnUpBlock2D,
	DownBlock2D,
	UNetMidBlock2DCrossAttn,
	UpBlock2D,
)


@auto_pytree
class UNet2DConditionOutput(ModelOutput):
	"""
	Output of UNet2DConditionModel.

	Args:
		sample: The output sample from the model
	"""

	sample: chex.Array


@register_module(
	config_class=UNet2DConfig,
	model_type="unet2d",
	embedding_layer_names=[],
)
class UNet2DConditionModel(EasyDeLBaseModule):
	"""
	2D UNet Conditional Model for diffusion.

	A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep
	and returns a sample-shaped output. Used in Stable Diffusion and other latent diffusion models.

	Args:
		config: Model configuration
		dtype: Data type for computation
		param_dtype: Data type for parameters
		precision: Precision for computation
		rngs: Random number generators
	"""

	def __init__(
		self,
		config: UNet2DConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.Precision | None = None,
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

		self.sample_size = config.sample_size
		block_out_channels = config.block_out_channels
		time_embed_dim = block_out_channels[0] * 4

		# Time embeddings
		self.time_proj = Timesteps(
			dim=block_out_channels[0],
			flip_sin_to_cos=config.flip_sin_to_cos,
			freq_shift=config.freq_shift,
		)

		self.time_embedding = TimestepEmbedding(
			in_channels=block_out_channels[0],
			time_embed_dim=time_embed_dim,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Additional embeddings for SDXL
		self.add_time_proj = None
		self.add_embedding = None
		if config.addition_embed_type == "text_time":
			if config.addition_time_embed_dim is None:
				raise ValueError("addition_time_embed_dim must be specified for text_time addition_embed_type")

			self.add_time_proj = Timesteps(
				dim=config.addition_time_embed_dim,
				flip_sin_to_cos=config.flip_sin_to_cos,
				freq_shift=config.freq_shift,
			)

			self.add_embedding = CombinedTimestepTextEmbedding(
				embedding_dim=time_embed_dim,
				pooled_projection_dim=config.projection_class_embeddings_input_dim or 1280,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

		# Input convolution
		self.conv_in = nn.Conv(
			config.in_channels,
			block_out_channels[0],
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Process attention configuration
		if isinstance(config.only_cross_attention, bool):
			only_cross_attention = [config.only_cross_attention] * len(config.down_block_types)
		else:
			only_cross_attention = list(config.only_cross_attention)

		if isinstance(config.attention_head_dim, int):
			attention_head_dim = [config.attention_head_dim] * len(config.down_block_types)
		else:
			attention_head_dim = list(config.attention_head_dim)

		if isinstance(config.num_attention_heads, int):
			num_attention_heads = [config.num_attention_heads] * len(config.down_block_types)
		elif config.num_attention_heads is not None:
			num_attention_heads = list(config.num_attention_heads)
		else:
			num_attention_heads = attention_head_dim

		if isinstance(config.transformer_layers_per_block, int):
			transformer_layers_per_block = [config.transformer_layers_per_block] * len(config.down_block_types)
		else:
			transformer_layers_per_block = list(config.transformer_layers_per_block)

		# Down blocks
		down_blocks = []
		output_channel = block_out_channels[0]

		for i, down_block_type in enumerate(config.down_block_types):
			input_channel = output_channel
			output_channel = block_out_channels[i]
			is_final_block = i == len(block_out_channels) - 1

			if down_block_type == "CrossAttnDownBlock2D":
				down_block = CrossAttnDownBlock2D(
					in_channels=input_channel,
					out_channels=output_channel,
					temb_channels=time_embed_dim,
					dropout=config.dropout,
					num_layers=config.layers_per_block,
					num_attention_heads=num_attention_heads[i],
					attention_head_dim=attention_head_dim[i],
					add_downsample=not is_final_block,
					use_linear_projection=config.use_linear_projection,
					only_cross_attention=only_cross_attention[i],
					transformer_layers_per_block=transformer_layers_per_block[i],
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			else:  # DownBlock2D
				down_block = DownBlock2D(
					in_channels=input_channel,
					out_channels=output_channel,
					temb_channels=time_embed_dim,
					dropout=config.dropout,
					num_layers=config.layers_per_block,
					add_downsample=not is_final_block,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)

			down_blocks.append(down_block)

		self.down_blocks = down_blocks

		# Mid block
		self.mid_block = UNetMidBlock2DCrossAttn(
			in_channels=block_out_channels[-1],
			temb_channels=time_embed_dim,
			dropout=config.dropout,
			num_attention_heads=num_attention_heads[-1],
			attention_head_dim=attention_head_dim[-1],
			use_linear_projection=config.use_linear_projection,
			transformer_layers_per_block=transformer_layers_per_block[-1],
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Up blocks
		up_blocks = []
		reversed_block_out_channels = list(reversed(block_out_channels))
		reversed_num_attention_heads = list(reversed(num_attention_heads))
		reversed_attention_head_dim = list(reversed(attention_head_dim))
		reversed_only_cross_attention = list(reversed(only_cross_attention))
		reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

		output_channel = reversed_block_out_channels[0]

		for i, up_block_type in enumerate(config.up_block_types):
			prev_output_channel = output_channel
			output_channel = reversed_block_out_channels[i]
			input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
			is_final_block = i == len(block_out_channels) - 1

			if up_block_type == "CrossAttnUpBlock2D":
				up_block = CrossAttnUpBlock2D(
					in_channels=input_channel,
					out_channels=output_channel,
					prev_output_channel=prev_output_channel,
					temb_channels=time_embed_dim,
					dropout=config.dropout,
					num_layers=config.layers_per_block + 1,
					num_attention_heads=reversed_num_attention_heads[i],
					attention_head_dim=reversed_attention_head_dim[i],
					add_upsample=not is_final_block,
					use_linear_projection=config.use_linear_projection,
					only_cross_attention=reversed_only_cross_attention[i],
					transformer_layers_per_block=reversed_transformer_layers_per_block[i],
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
			else:  # UpBlock2D
				up_block = UpBlock2D(
					in_channels=input_channel,
					out_channels=output_channel,
					prev_output_channel=prev_output_channel,
					temb_channels=time_embed_dim,
					dropout=config.dropout,
					num_layers=config.layers_per_block + 1,
					add_upsample=not is_final_block,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)

			up_blocks.append(up_block)

		self.up_blocks = up_blocks

		# Output
		self.conv_norm_out = nn.GroupNorm(
			num_groups=config.norm_num_groups,
			epsilon=1e-6,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.conv_out = nn.Conv(
			block_out_channels[0],
			config.out_channels,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding=((1, 1), (1, 1)),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		sample: chex.Array,
		timesteps: chex.Array | int | float,
		encoder_hidden_states: chex.Array,
		added_cond_kwargs: dict[str, chex.Array] | None = None,
		return_dict: bool = True,
		deterministic: bool = True,
	) -> UNet2DConditionOutput | tuple:
		"""
		Forward pass of the UNet.

		Args:
			sample: Noisy input tensor [batch, height, width, channels] or [batch, channels, height, width]
			timesteps: Timestep tensor [batch] or scalar
			encoder_hidden_states: Text encoder hidden states [batch, seq_len, dim]
			added_cond_kwargs: Additional conditioning for SDXL (text_embeds, time_ids)
			return_dict: Whether to return a dictionary or tuple
			deterministic: Whether to use deterministic mode (no dropout)

		Returns:
			UNet2DConditionOutput or tuple containing the sample
		"""
		# Handle different input formats
		if sample.ndim == 4:
			# Check if input is in NCHW format and convert to NHWC
			if sample.shape[1] == self.config.in_channels:
				sample = jnp.transpose(sample, (0, 2, 3, 1))

		# Timestep embedding
		if not isinstance(timesteps, jnp.ndarray):
			timesteps = jnp.array([timesteps], dtype=jnp.int32)
		elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
			timesteps = jnp.expand_dims(timesteps, 0)

		t_emb = self.time_proj(timesteps)
		t_emb = self.time_embedding(t_emb)

		# Additional embeddings for SDXL
		if self.config.addition_embed_type == "text_time":
			if added_cond_kwargs is None:
				raise ValueError("Need to provide added_cond_kwargs for text_time addition_embed_type")

			text_embeds = added_cond_kwargs.get("text_embeds")
			time_ids = added_cond_kwargs.get("time_ids")

			if text_embeds is None or time_ids is None:
				raise ValueError("text_embeds and time_ids required in added_cond_kwargs for text_time")

			# Compute time embeds
			time_embeds = self.add_time_proj(jnp.ravel(time_ids))
			time_embeds = jnp.reshape(time_embeds, (text_embeds.shape[0], -1))

			# Combine with text embeds
			add_embeds = jnp.concatenate([text_embeds, time_embeds], axis=-1)
			aug_emb = self.add_embedding(add_embeds, add_embeds)  # Using same for both timestep and pooled
			t_emb = t_emb + aug_emb

		# Input convolution
		sample = self.conv_in(sample)

		# Down blocks
		down_block_res_samples = (sample,)
		for down_block in self.down_blocks:
			if isinstance(down_block, CrossAttnDownBlock2D):
				sample, res_samples = down_block(
					sample,
					t_emb,
					encoder_hidden_states,
					deterministic=deterministic,
				)
			else:
				sample, res_samples = down_block(
					sample,
					t_emb,
					deterministic=deterministic,
				)
			down_block_res_samples = down_block_res_samples + res_samples

		# Mid block
		sample = self.mid_block(
			sample,
			t_emb,
			encoder_hidden_states,
			deterministic=deterministic,
		)

		# Up blocks
		for up_block in self.up_blocks:
			res_samples = down_block_res_samples[-(self.config.layers_per_block + 1) :]
			down_block_res_samples = down_block_res_samples[: -(self.config.layers_per_block + 1)]

			if isinstance(up_block, CrossAttnUpBlock2D):
				sample = up_block(
					sample,
					res_samples,
					t_emb,
					encoder_hidden_states,
					deterministic=deterministic,
				)
			else:
				sample = up_block(
					sample,
					res_samples,
					t_emb,
					deterministic=deterministic,
				)

		# Output
		sample = self.conv_norm_out(sample)
		sample = nn.silu(sample)
		sample = self.conv_out(sample)

		# Convert back to NCHW format if needed
		sample = jnp.transpose(sample, (0, 3, 1, 2))

		if not return_dict:
			return (sample,)

		return UNet2DConditionOutput(sample=sample)
