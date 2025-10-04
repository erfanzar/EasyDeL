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

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("flux")
class FluxConfig(EasyDeLBaseConfig):
	"""
	Configuration class for Flux Transformer models.

	Flux is a state-of-the-art transformer-based diffusion model for image generation
	that uses rectified flow, rotary position embeddings (RoPE), and dual transformer
	blocks (double and single) for processing image and text conditioning.

	Args:
		patch_size (`int`, *optional*, defaults to 1):
			Size of patches for patchifying the input latent images.
		in_channels (`int`, *optional*, defaults to 64):
			Number of input channels in the latent space.
		num_layers (`int`, *optional*, defaults to 19):
			Number of double transformer blocks (processing both image and text).
		num_single_layers (`int`, *optional*, defaults to 38):
			Number of single transformer blocks (processing concatenated features).
		attention_head_dim (`int`, *optional*, defaults to 128):
			Dimensionality of each attention head.
		num_attention_heads (`int`, *optional*, defaults to 24):
			Number of attention heads in each attention layer.
		joint_attention_dim (`int`, *optional*, defaults to 4096):
			Dimensionality of the text encoder hidden states.
		pooled_projection_dim (`int`, *optional*, defaults to 768):
			Dimensionality of the pooled text projections (CLIP embeddings).
		guidance_embeds (`bool`, *optional*, defaults to False):
			Whether to use guidance embeddings for classifier-free guidance.
			Set to True for flux-dev, False for flux-schnell.
		axes_dims_rope (`tuple`, *optional*, defaults to (16, 56, 56)):
			Dimensions for the rotary position embeddings (RoPE).
			Format: (channels_dim, height_dim, width_dim).
		theta (`int`, *optional*, defaults to 10000):
			Base value for rotary position embedding frequencies.
		mlp_ratio (`float`, *optional*, defaults to 4.0):
			Ratio of MLP hidden dimension to embedding dimension.
		qkv_bias (`bool`, *optional*, defaults to True):
			Whether to use bias in QKV projections.
		eps (`float`, *optional*, defaults to 1e-6):
			Epsilon value for layer normalization.
		attention_kernel (`str`, *optional*, defaults to "dot_product"):
			Type of attention kernel to use. Options: "dot_product", "flash", "cudnn_flash_te".
		gradient_checkpointing (`str`, *optional*, defaults to "nothing_saveable"):
			Gradient checkpointing configuration.
		bits (`int`, *optional*):
			Number of bits for quantization.
	"""

	model_type: str = "flux"

	def __init__(
		self,
		patch_size: int = 1,
		in_channels: int = 64,
		num_layers: int = 19,
		num_single_layers: int = 38,
		attention_head_dim: int = 128,
		num_attention_heads: int = 24,
		joint_attention_dim: int = 4096,
		pooled_projection_dim: int = 768,
		guidance_embeds: bool = False,
		axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
		theta: int = 10000,
		mlp_ratio: float = 4.0,
		qkv_bias: bool = True,
		eps: float = 1e-6,
		attention_kernel: str = "dot_product",
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: int | None = None,
		**kwargs,
	):
		self.patch_size = patch_size
		self.in_channels = in_channels
		self.num_layers = num_layers
		self.num_single_layers = num_single_layers
		self.attention_head_dim = attention_head_dim
		self.num_attention_heads = num_attention_heads
		self.joint_attention_dim = joint_attention_dim
		self.pooled_projection_dim = pooled_projection_dim
		self.guidance_embeds = guidance_embeds
		self.axes_dims_rope = axes_dims_rope
		self.theta = theta
		self.mlp_ratio = mlp_ratio
		self.qkv_bias = qkv_bias
		self.eps = eps
		self.attention_kernel = attention_kernel
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

		# Computed properties
		self.inner_dim = num_attention_heads * attention_head_dim
		self.out_channels = in_channels

		super().__init__(
			bits=bits,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.

		Returns:
			`tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		pmag = self.partition_manager
		return (
			# Position embeddings
			(r"pe_embedder/.*", pmag.resolve(Replicated)),
			# Time and text embeddings
			(r"time_text_embed/.*/linear_1/kernel", pmag.resolve(ColumnWise)),
			(r"time_text_embed/.*/linear_1/bias", pmag.resolve(Replicated)),
			(r"time_text_embed/.*/linear_2/kernel", pmag.resolve(RowWise)),
			(r"time_text_embed/.*/linear_2/bias", pmag.resolve(Replicated)),
			# Input projections
			(r"txt_in/kernel", pmag.resolve(ColumnWise)),
			(r"txt_in/bias", pmag.resolve(Replicated)),
			(r"img_in/kernel", pmag.resolve(ColumnWise)),
			(r"img_in/bias", pmag.resolve(Replicated)),
			# Double blocks - attention
			(r"double_blocks_\d+/attn/i_qkv/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/attn/i_qkv/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/attn/e_qkv/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/attn/e_qkv/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/attn/i_proj/kernel", pmag.resolve(RowWise)),
			(r"double_blocks_\d+/attn/i_proj/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/attn/e_proj/kernel", pmag.resolve(RowWise)),
			(r"double_blocks_\d+/attn/e_proj/bias", pmag.resolve(Replicated)),
			# Double blocks - MLP
			(r"double_blocks_\d+/img_mlp/layers_0/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/img_mlp/layers_0/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/img_mlp/layers_2/kernel", pmag.resolve(RowWise)),
			(r"double_blocks_\d+/img_mlp/layers_2/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/txt_mlp/layers_0/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/txt_mlp/layers_0/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/txt_mlp/layers_2/kernel", pmag.resolve(RowWise)),
			(r"double_blocks_\d+/txt_mlp/layers_2/bias", pmag.resolve(Replicated)),
			# Double blocks - norms
			(r"double_blocks_\d+/img_norm1/lin/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/img_norm1/lin/bias", pmag.resolve(Replicated)),
			(r"double_blocks_\d+/txt_norm1/lin/kernel", pmag.resolve(ColumnWise)),
			(r"double_blocks_\d+/txt_norm1/lin/bias", pmag.resolve(Replicated)),
			# Single blocks
			(r"single_blocks_\d+/linear1/kernel", pmag.resolve(ColumnWise)),
			(r"single_blocks_\d+/linear1/bias", pmag.resolve(Replicated)),
			(r"single_blocks_\d+/linear2/kernel", pmag.resolve(RowWise)),
			(r"single_blocks_\d+/linear2/bias", pmag.resolve(Replicated)),
			(r"single_blocks_\d+/norm/lin/kernel", pmag.resolve(ColumnWise)),
			(r"single_blocks_\d+/norm/lin/bias", pmag.resolve(Replicated)),
			# Output projection
			(r"norm_out/Dense_0/kernel", pmag.resolve(ColumnWise)),
			(r"norm_out/Dense_0/bias", pmag.resolve(Replicated)),
			(r"proj_out/kernel", pmag.resolve(ColumnWise)),
			(r"proj_out/bias", pmag.resolve(Replicated)),
			# RMS norms
			(r".*/query_norm/scale", pmag.resolve(Replicated)),
			(r".*/key_norm/scale", pmag.resolve(Replicated)),
			(r".*/encoder_query_norm/scale", pmag.resolve(Replicated)),
			(r".*/encoder_key_norm/scale", pmag.resolve(Replicated)),
			# Layer norms
			(r".*LayerNorm.*/scale", pmag.resolve(Replicated)),
			(r".*LayerNorm.*/bias", pmag.resolve(Replicated)),
			# Default
			(r".*bias", pmag.resolve(Replicated)),
			(r".*", pmag.resolve(Replicated)),
		)

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns tuple of parameter names to exclude from weight decay."""
		return ("bias", "norm", "scale", "embedding")

	@staticmethod
	def rng_keys():
		"""Returns the RNG key names."""
		return ("params", "dropout")

	@property
	def head_dim(self) -> int:
		"""Dimensionality of each attention head."""
		return self.attention_head_dim
