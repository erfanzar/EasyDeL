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

"""Configuration for UNet 2D Conditional model."""

import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("unet2d")
class UNet2DConfig(EasyDeLBaseConfig):
	"""
	Configuration class for UNet2D Conditional models.

	UNet2D is the core architecture used in Stable Diffusion and other latent diffusion models
	for image generation. It processes noisy latent representations conditioned on timesteps
	and text embeddings to predict the noise that should be removed.

	Args:
		sample_size (`int`, *optional*, defaults to 32):
			The size of the input sample (in latent space).
		in_channels (`int`, *optional*, defaults to 4):
			Number of input channels (latent channels).
		out_channels (`int`, *optional*, defaults to 4):
			Number of output channels (latent channels).
		down_block_types (`tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
			Tuple of downsample block types.
		up_block_types (`tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
			Tuple of upsample block types.
		block_out_channels (`tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
			Tuple of block output channels.
		layers_per_block (`int`, *optional*, defaults to 2):
			Number of ResNet layers per block.
		attention_head_dim (`int | tuple[int]`, *optional*, defaults to 8):
			Dimension of attention heads. Can be a single int or tuple matching down_block_types.
		num_attention_heads (`int | tuple[int]`, *optional*):
			Number of attention heads. If None, uses attention_head_dim.
		cross_attention_dim (`int`, *optional*, defaults to 1280):
			Dimension of cross-attention features (text embeddings).
		dropout (`float`, *optional*, defaults to 0.0):
			Dropout probability.
		flip_sin_to_cos (`bool`, *optional*, defaults to True):
			Whether to flip sin to cos in timestep embedding.
		freq_shift (`int`, *optional*, defaults to 0):
			Frequency shift for timestep embedding.
		use_linear_projection (`bool`, *optional*, defaults to False):
			Whether to use linear projection in attention blocks.
		only_cross_attention (`bool | tuple[bool]`, *optional*, defaults to False):
			Whether to only use cross-attention (no self-attention).
		transformer_layers_per_block (`int | tuple[int]`, *optional*, defaults to 1):
			Number of transformer layers per block.
		addition_embed_type (`str`, *optional*):
			Type of additional embeddings. "text_time" for SDXL.
		addition_time_embed_dim (`int`, *optional*):
			Dimension of additional time embeddings for SDXL.
		projection_class_embeddings_input_dim (`int`, *optional*):
			Input dimension for projection class embeddings (SDXL).
		norm_num_groups (`int`, *optional*, defaults to 32):
			Number of groups for group normalization.
		attention_type (`str`, *optional*, defaults to "default"):
			Type of attention mechanism to use.
		use_memory_efficient_attention (`bool`, *optional*, defaults to False):
			Whether to use memory efficient attention.
		gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
			Gradient checkpointing configuration.
		bits (`int`, *optional*):
			Number of bits for quantization.
	"""

	model_type: str = "unet2d"

	def __init__(
		self,
		sample_size: int = 32,
		in_channels: int = 4,
		out_channels: int = 4,
		down_block_types: tuple[str, ...] = (
			"CrossAttnDownBlock2D",
			"CrossAttnDownBlock2D",
			"CrossAttnDownBlock2D",
			"DownBlock2D",
		),
		up_block_types: tuple[str, ...] = (
			"UpBlock2D",
			"CrossAttnUpBlock2D",
			"CrossAttnUpBlock2D",
			"CrossAttnUpBlock2D",
		),
		block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
		layers_per_block: int = 2,
		attention_head_dim: int | tuple[int, ...] = 8,
		num_attention_heads: int | tuple[int, ...] | None = None,
		cross_attention_dim: int = 1280,
		dropout: float = 0.0,
		flip_sin_to_cos: bool = True,
		freq_shift: int = 0,
		use_linear_projection: bool = False,
		only_cross_attention: bool | tuple[bool, ...] = False,
		transformer_layers_per_block: int | tuple[int, ...] = 1,
		addition_embed_type: str | None = None,
		addition_time_embed_dim: int | None = None,
		projection_class_embeddings_input_dim: int | None = None,
		norm_num_groups: int = 32,
		attention_type: str = "default",
		use_memory_efficient_attention: bool = False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: int | None = None,
		**kwargs,
	):
		self.sample_size = sample_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.down_block_types = down_block_types
		self.up_block_types = up_block_types
		self.block_out_channels = block_out_channels
		self.layers_per_block = layers_per_block
		self.attention_head_dim = attention_head_dim
		self.num_attention_heads = num_attention_heads
		self.cross_attention_dim = cross_attention_dim
		self.dropout = dropout
		self.flip_sin_to_cos = flip_sin_to_cos
		self.freq_shift = freq_shift
		self.use_linear_projection = use_linear_projection
		self.only_cross_attention = only_cross_attention
		self.transformer_layers_per_block = transformer_layers_per_block
		self.addition_embed_type = addition_embed_type
		self.addition_time_embed_dim = addition_time_embed_dim
		self.projection_class_embeddings_input_dim = projection_class_embeddings_input_dim
		self.norm_num_groups = norm_num_groups
		self.attention_type = attention_type
		self.use_memory_efficient_attention = use_memory_efficient_attention
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

		super().__init__(
			bits=bits,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.

		Returns:
			`tuple[tuple[str, PartitionSpec]]`: The partition rules.
		"""
		pmag = self.partition_manager
		return (
			# Input convolution
			(r"conv_in/kernel", pmag.resolve(Replicated)),
			(r"conv_in/bias", pmag.resolve(Replicated)),
			# Time embeddings
			(r"time_embedding/.*/kernel", pmag.resolve(ColumnWise)),
			(r"time_embedding/.*/bias", pmag.resolve(Replicated)),
			# Additional embeddings (SDXL)
			(r"add_time_proj/.*", pmag.resolve(Replicated)),
			(r"add_embedding/.*/kernel", pmag.resolve(ColumnWise)),
			(r"add_embedding/.*/bias", pmag.resolve(Replicated)),
			# Down blocks
			(r"down_blocks/.*/resnets/.*/norm./scale", pmag.resolve(Replicated)),
			(r"down_blocks/.*/resnets/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"down_blocks/.*/resnets/.*/conv./bias", pmag.resolve(Replicated)),
			(r"down_blocks/.*/resnets/.*/time_emb_proj/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/resnets/.*/time_emb_proj/bias", pmag.resolve(Replicated)),
			# Attention in down blocks
			(r"down_blocks/.*/attentions/.*/norm/scale", pmag.resolve(Replicated)),
			(r"down_blocks/.*/attentions/.*/proj_in/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/proj_out/kernel", pmag.resolve(RowWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/ff/net_0/proj/kernel", pmag.resolve(ColumnWise)),
			(r"down_blocks/.*/attentions/.*/transformer_blocks/.*/ff/net_2/kernel", pmag.resolve(RowWise)),
			# Downsampling
			(r"down_blocks/.*/downsamplers_0/conv/kernel", pmag.resolve(Replicated)),
			(r"down_blocks/.*/downsamplers_0/conv/bias", pmag.resolve(Replicated)),
			# Mid block
			(r"mid_block/resnets/.*/norm./scale", pmag.resolve(Replicated)),
			(r"mid_block/resnets/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"mid_block/resnets/.*/conv./bias", pmag.resolve(Replicated)),
			(r"mid_block/resnets/.*/time_emb_proj/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/resnets/.*/time_emb_proj/bias", pmag.resolve(Replicated)),
			(r"mid_block/attentions/.*/norm/scale", pmag.resolve(Replicated)),
			(r"mid_block/attentions/.*/proj_in/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/proj_out/kernel", pmag.resolve(RowWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn1/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn1/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn1/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn1/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn2/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn2/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn2/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/attn2/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/ff/net_0/proj/kernel", pmag.resolve(ColumnWise)),
			(r"mid_block/attentions/.*/transformer_blocks/.*/ff/net_2/kernel", pmag.resolve(RowWise)),
			# Up blocks
			(r"up_blocks/.*/resnets/.*/norm./scale", pmag.resolve(Replicated)),
			(r"up_blocks/.*/resnets/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"up_blocks/.*/resnets/.*/conv./bias", pmag.resolve(Replicated)),
			(r"up_blocks/.*/resnets/.*/time_emb_proj/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/resnets/.*/time_emb_proj/bias", pmag.resolve(Replicated)),
			# Attention in up blocks
			(r"up_blocks/.*/attentions/.*/norm/scale", pmag.resolve(Replicated)),
			(r"up_blocks/.*/attentions/.*/proj_in/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/proj_out/kernel", pmag.resolve(RowWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn1/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_q/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_k/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_v/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/attn2/to_out_0/kernel", pmag.resolve(RowWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/ff/net_0/proj/kernel", pmag.resolve(ColumnWise)),
			(r"up_blocks/.*/attentions/.*/transformer_blocks/.*/ff/net_2/kernel", pmag.resolve(RowWise)),
			# Upsampling
			(r"up_blocks/.*/upsamplers_0/conv/kernel", pmag.resolve(Replicated)),
			(r"up_blocks/.*/upsamplers_0/conv/bias", pmag.resolve(Replicated)),
			# Output
			(r"conv_norm_out/scale", pmag.resolve(Replicated)),
			(r"conv_out/kernel", pmag.resolve(Replicated)),
			(r"conv_out/bias", pmag.resolve(Replicated)),
			# Norms
			(r".*/norm.*/scale", pmag.resolve(Replicated)),
			(r".*/norm.*/bias", pmag.resolve(Replicated)),
			# Default
			(r".*bias", pmag.resolve(Replicated)),
			(r".*", pmag.resolve(Replicated)),
		)

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns tuple of parameter names to exclude from weight decay."""
		return ("bias", "norm", "time_proj", "time_embedding")

	@staticmethod
	def rng_keys():
		"""Returns the RNG key names."""
		return ("params", "dropout")

	@property
	def time_embed_dim(self) -> int:
		"""Dimension of the timestep embedding."""
		return self.block_out_channels[0] * 4
