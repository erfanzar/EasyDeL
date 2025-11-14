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

from eformer.common_types import Replicated

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("vae")
class VAEConfig(EasyDeLBaseConfig):
	"""
	Configuration class for Variational Autoencoder (VAE) models.

	VAE is used to encode images into a latent space and decode latents back to images.
	Commonly used in latent diffusion models like Stable Diffusion.

	Args:
		in_channels (`int`, *optional*, defaults to 3):
			Number of input channels (3 for RGB images).
		out_channels (`int`, *optional*, defaults to 3):
			Number of output channels.
		latent_channels (`int`, *optional*, defaults to 4):
			Number of channels in the latent space.
		down_block_types (`tuple[str]`, *optional*):
			Types of downsampling blocks. Defaults to ResNet blocks.
		up_block_types (`tuple[str]`, *optional*):
			Types of upsampling blocks. Defaults to ResNet blocks.
		block_out_channels (`tuple[int]`, *optional*, defaults to (128, 256, 512, 512)):
			Number of output channels for each block.
		layers_per_block (`int`, *optional*, defaults to 2):
			Number of ResNet layers per block.
		act_fn (`str`, *optional*, defaults to "silu"):
			Activation function to use.
		norm_num_groups (`int`, *optional*, defaults to 32):
			Number of groups for group normalization.
		sample_size (`int`, *optional*, defaults to 512):
			Sample size of the input image.
		scaling_factor (`float`, *optional*, defaults to 0.18215):
			Scaling factor for latent space (SD 1.x/2.x use 0.18215, SDXL uses 0.13025).
		force_upcast (`bool`, *optional*, defaults to True):
			Whether to force upcasting to float32 for certain operations.
	"""

	model_type: str = "vae"

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		latent_channels: int = 4,
		down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",) * 4,
		up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",) * 4,
		block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
		layers_per_block: int = 2,
		act_fn: str = "silu",
		norm_num_groups: int = 32,
		sample_size: int = 512,
		scaling_factor: float = 0.18215,
		force_upcast: bool = True,
		**kwargs,
	):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.latent_channels = latent_channels
		self.down_block_types = down_block_types
		self.up_block_types = up_block_types
		self.block_out_channels = block_out_channels
		self.layers_per_block = layers_per_block
		self.act_fn = act_fn
		self.norm_num_groups = norm_num_groups
		self.sample_size = sample_size
		self.scaling_factor = scaling_factor
		self.force_upcast = force_upcast

		super().__init__(**kwargs)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.

		Returns:
			`tuple[tuple[str, PartitionSpec]]`: The partition rules.
		"""
		pmag = self.partition_manager
		return (
			# Encoder
			(r"encoder/conv_in/kernel", pmag.resolve(Replicated)),
			(r"encoder/conv_in/bias", pmag.resolve(Replicated)),
			(r"encoder/down_blocks/.*/resnets/.*/norm./scale", pmag.resolve(Replicated)),
			(r"encoder/down_blocks/.*/resnets/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"encoder/down_blocks/.*/resnets/.*/conv./bias", pmag.resolve(Replicated)),
			(r"encoder/down_blocks/.*/downsamplers/.*/conv/kernel", pmag.resolve(Replicated)),
			(r"encoder/down_blocks/.*/downsamplers/.*/conv/bias", pmag.resolve(Replicated)),
			(r"encoder/mid_block/.*/norm./scale", pmag.resolve(Replicated)),
			(r"encoder/mid_block/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"encoder/mid_block/.*/conv./bias", pmag.resolve(Replicated)),
			(r"encoder/conv_norm_out/scale", pmag.resolve(Replicated)),
			(r"encoder/conv_out/kernel", pmag.resolve(Replicated)),
			(r"encoder/conv_out/bias", pmag.resolve(Replicated)),
			# Quant/Post-quant conv
			(r"quant_conv/kernel", pmag.resolve(Replicated)),
			(r"quant_conv/bias", pmag.resolve(Replicated)),
			(r"post_quant_conv/kernel", pmag.resolve(Replicated)),
			(r"post_quant_conv/bias", pmag.resolve(Replicated)),
			# Decoder
			(r"decoder/conv_in/kernel", pmag.resolve(Replicated)),
			(r"decoder/conv_in/bias", pmag.resolve(Replicated)),
			(r"decoder/up_blocks/.*/resnets/.*/norm./scale", pmag.resolve(Replicated)),
			(r"decoder/up_blocks/.*/resnets/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"decoder/up_blocks/.*/resnets/.*/conv./bias", pmag.resolve(Replicated)),
			(r"decoder/up_blocks/.*/upsamplers/.*/conv/kernel", pmag.resolve(Replicated)),
			(r"decoder/up_blocks/.*/upsamplers/.*/conv/bias", pmag.resolve(Replicated)),
			(r"decoder/mid_block/.*/norm./scale", pmag.resolve(Replicated)),
			(r"decoder/mid_block/.*/conv./kernel", pmag.resolve(Replicated)),
			(r"decoder/mid_block/.*/conv./bias", pmag.resolve(Replicated)),
			(r"decoder/conv_norm_out/scale", pmag.resolve(Replicated)),
			(r"decoder/conv_out/kernel", pmag.resolve(Replicated)),
			(r"decoder/conv_out/bias", pmag.resolve(Replicated)),
			# Default
			(r".*", pmag.resolve(Replicated)),
		)

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns tuple of parameter names to exclude from weight decay."""
		return ("norm", "bias")

	@staticmethod
	def rng_keys():
		"""Returns the RNG key names."""
		return ("params", "dropout")
