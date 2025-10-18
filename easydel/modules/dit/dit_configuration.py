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


@register_config("dit")
class DiTConfig(EasyDeLBaseConfig):
	"""
	Configuration class for DiT (Diffusion Transformer) models.

	DiT is a transformer-based architecture for image diffusion models that operates on
	image patches similar to Vision Transformers. It uses adaptive layer norm conditioning
	for timestep and class label embeddings.

	Args:
		image_size (`int`, *optional*, defaults to 32):
			Input image resolution (assumes square images).
		patch_size (`int`, *optional*, defaults to 2):
			Size of image patches.
		in_channels (`int`, *optional*, defaults to 4):
			Number of input channels (3 for RGB, 4 for latent space).
		hidden_size (`int`, *optional*, defaults to 1152):
			Dimensionality of the transformer layers.
		num_hidden_layers (`int`, *optional*, defaults to 28):
			Number of transformer blocks.
		num_attention_heads (`int`, *optional*, defaults to 16):
			Number of attention heads in each attention layer.
		intermediate_size (`int`, *optional*, defaults to None):
			Dimensionality of the MLP layer. If None, defaults to 4 * hidden_size.
		hidden_act (`str`, *optional*, defaults to `"gelu"`):
			Activation function for MLP layers.
		num_classes (`int`, *optional*, defaults to 1000):
			Number of class labels for conditional generation.
		class_dropout_prob (`float`, *optional*, defaults to 0.1):
			Dropout probability for classifier-free guidance during training.
		learn_sigma (`bool`, *optional*, defaults to True):
			Whether to learn the variance in addition to the mean.
		use_conditioning (`bool`, *optional*, defaults to True):
			Whether to use timestep and class conditioning.
		attention_dropout (`float`, *optional*, defaults to 0.0):
			Dropout probability for attention weights.
		mlp_dropout (`float`, *optional*, defaults to 0.0):
			Dropout probability for MLP layers.
		initializer_range (`float`, *optional*, defaults to 0.02):
			Standard deviation for weight initialization.
		layer_norm_eps (`float`, *optional*, defaults to 1e-6):
			Epsilon for layer normalization.
		use_bias (`bool`, *optional*, defaults to True):
			Whether to use bias in linear layers.
		gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
			Gradient checkpointing configuration.
		bits (`int`, *optional*):
			Number of bits for quantization.
	"""

	model_type: str = "dit"

	def __init__(
		self,
		image_size: int = 32,
		patch_size: int = 2,
		in_channels: int = 4,
		hidden_size: int = 1152,
		num_hidden_layers: int = 28,
		num_attention_heads: int = 16,
		intermediate_size: int | None = None,
		hidden_act: str = "gelu",
		num_classes: int = 1000,
		class_dropout_prob: float = 0.1,
		learn_sigma: bool = True,
		use_conditioning: bool = True,
		attention_dropout: float = 0.0,
		mlp_dropout: float = 0.0,
		initializer_range: float = 0.02,
		layer_norm_eps: float = 1e-6,
		use_bias: bool = True,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: int | None = None,
		**kwargs,
	):
		self.image_size = image_size
		self.patch_size = patch_size
		self.in_channels = in_channels
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.intermediate_size = intermediate_size or 4 * hidden_size
		self.hidden_act = hidden_act
		self.num_classes = num_classes
		self.class_dropout_prob = class_dropout_prob
		self.learn_sigma = learn_sigma
		self.use_conditioning = use_conditioning
		self.attention_dropout = attention_dropout
		self.mlp_dropout = mlp_dropout
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_bias = use_bias
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

		# Computed properties
		self.num_patches = (image_size // patch_size) ** 2
		self.out_channels = in_channels * 2 if learn_sigma else in_channels

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
			# Patch embedding
			(r"patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
			(r"patch_embed/proj/bias", pmag.resolve(Replicated)),
			# Positional embedding
			(r"pos_embed", pmag.resolve(Replicated)),
			# Time embedding
			(r"time_embed/mlp/\d+/kernel", pmag.resolve(ColumnWise)),
			(r"time_embed/mlp/\d+/bias", pmag.resolve(Replicated)),
			# Label embedding
			(r"label_embed/embedding", pmag.resolve(ColumnWise)),
			# Transformer blocks
			(r"blocks/\d+/attn/(q|k|v)_proj/kernel", pmag.resolve(ColumnWise)),
			(r"blocks/\d+/attn/o_proj/kernel", pmag.resolve(RowWise)),
			(r"blocks/\d+/attn/.*proj/bias", pmag.resolve(Replicated)),
			(r"blocks/\d+/mlp/fc1/kernel", pmag.resolve(ColumnWise)),
			(r"blocks/\d+/mlp/fc2/kernel", pmag.resolve(RowWise)),
			(r"blocks/\d+/mlp/.*bias", pmag.resolve(Replicated)),
			# Adaptive layer norm
			(r"blocks/\d+/adaLN_modulation/\d+/kernel", pmag.resolve(ColumnWise)),
			(r"blocks/\d+/adaLN_modulation/\d+/bias", pmag.resolve(Replicated)),
			# Final layer
			(r"final_layer/adaLN_modulation/\d+/kernel", pmag.resolve(ColumnWise)),
			(r"final_layer/adaLN_modulation/\d+/bias", pmag.resolve(Replicated)),
			(r"final_layer/linear/kernel", pmag.resolve(ColumnWise)),
			(r"final_layer/linear/bias", pmag.resolve(Replicated)),
			# Norms
			(r".*(norm|ln)/scale", pmag.resolve(Replicated)),
			(r".*(norm|ln)/bias", pmag.resolve(Replicated)),
			# Default
			(r".*bias", pmag.resolve(Replicated)),
			(r".*", pmag.resolve(Replicated)),
		)

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns tuple of parameter names to exclude from weight decay."""
		return ("bias", "norm", "pos_embed", "time_embed", "label_embed")

	@staticmethod
	def rng_keys():
		"""Returns the RNG key names."""
		return ("params", "dropout")

	@property
	def head_dim(self) -> int:
		"""Dimensionality of each attention head."""
		return self.hidden_size // self.num_attention_heads
