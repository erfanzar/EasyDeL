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

from eformer.common_types import ColumnWise, ExpertColumnWiseAlt, ExpertRowWiseAlt, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("dit_moe")
class DiTMoEConfig(EasyDeLBaseConfig):
	"""
	Configuration for DiT-MoE (Mixture of Experts Diffusion Transformer).

	This extends DiT with sparse mixture of experts in the MLP blocks, following
	DeepSeek V2's MoE architecture for efficient scaling.

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
		moe_intermediate_size (`int`, *optional*, defaults to None):
			Dimensionality of each MoE expert. If None, defaults to intermediate_size.
		num_classes (`int`, *optional*, defaults to 1000):
			Number of class labels for conditional generation.
		class_dropout_prob (`float`, *optional*, defaults to 0.1):
			Dropout probability for classifier-free guidance during training.
		learn_sigma (`bool`, *optional*, defaults to True):
			Whether to learn the variance in addition to the mean.
		use_conditioning (`bool`, *optional*, defaults to True):
			Whether to use timestep and class conditioning.

		# MoE Parameters (from DeepSeek V2)
		n_shared_experts (`int`, *optional*, defaults to 2):
			Number of shared experts (always active).
		n_routed_experts (`int`, *optional*, defaults to 64):
			Number of routed experts (selected via routing).
		num_experts_per_tok (`int`, *optional*, defaults to 6):
			Number of experts to activate per token (top-k routing).
		ep_size (`int`, *optional*, defaults to 1):
			Expert parallel size for distributed training.
		routed_scaling_factor (`float`, *optional*, defaults to 1.0):
			Scaling factor for routed expert outputs.
		topk_method (`str`, *optional*, defaults to "greedy"):
			Method for top-k expert selection.
		n_group (`int`, *optional*, defaults to None):
			Number of expert groups for grouped routing.
		topk_group (`int`, *optional*, defaults to None):
			Top-k groups for grouped routing.
		moe_layer_freq (`int`, *optional*, defaults to 1):
			Frequency of MoE layers (1 = every layer, 2 = every other layer).
		first_k_dense_replace (`int`, *optional*, defaults to 0):
			First k layers to replace with dense MLPs instead of MoE.
		norm_topk_prob (`bool`, *optional*, defaults to False):
			Whether to normalize top-k probabilities.
		scoring_func (`str`, *optional*, defaults to "softmax"):
			Scoring function for expert selection.
		aux_loss_alpha (`float`, *optional*, defaults to 0.001):
			Auxiliary loss weight for load balancing.
		seq_aux (`bool`, *optional*, defaults to True):
			Whether to use sequence-level auxiliary loss.

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

	model_type: str = "dit_moe"

	def __init__(
		self,
		image_size: int = 32,
		patch_size: int = 2,
		in_channels: int = 4,
		hidden_size: int = 1152,
		num_hidden_layers: int = 28,
		num_attention_heads: int = 16,
		intermediate_size: int | None = None,
		moe_intermediate_size: int | None = None,
		hidden_act: str = "gelu",
		num_classes: int = 1000,
		class_dropout_prob: float = 0.1,
		learn_sigma: bool = True,
		use_conditioning: bool = True,
		# MoE parameters
		n_shared_experts: int = 2,
		n_routed_experts: int = 64,
		num_experts_per_tok: int = 6,
		ep_size: int = 1,
		routed_scaling_factor: float = 1.0,
		topk_method: str = "greedy",
		n_group: int | None = None,
		topk_group: int | None = None,
		moe_layer_freq: int = 1,
		first_k_dense_replace: int = 0,
		norm_topk_prob: bool = False,
		scoring_func: str = "softmax",
		aux_loss_alpha: float = 0.001,
		seq_aux: bool = True,
		# Standard DiT parameters
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
		self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
		self.hidden_act = hidden_act
		self.num_classes = num_classes
		self.class_dropout_prob = class_dropout_prob
		self.learn_sigma = learn_sigma
		self.use_conditioning = use_conditioning

		# MoE parameters
		self.n_shared_experts = n_shared_experts
		self.n_routed_experts = n_routed_experts
		self.num_experts_per_tok = num_experts_per_tok
		self.ep_size = ep_size
		self.routed_scaling_factor = routed_scaling_factor
		self.topk_method = topk_method
		self.n_group = n_group
		self.topk_group = topk_group
		self.moe_layer_freq = moe_layer_freq
		self.first_k_dense_replace = first_k_dense_replace
		self.norm_topk_prob = norm_topk_prob
		self.scoring_func = scoring_func
		self.aux_loss_alpha = aux_loss_alpha
		self.seq_aux = seq_aux

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
		Get the partition rules for the model with MoE-specific sharding.

		Returns:
			`tuple[tuple[str, PartitionSpec]]`: The partition rules.
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
			# Transformer blocks - Attention
			(r"blocks/\d+/attn/(q|k|v)_proj/kernel", pmag.resolve(ColumnWise)),
			(r"blocks/\d+/attn/o_proj/kernel", pmag.resolve(RowWise)),
			(r"blocks/\d+/attn/.*proj/bias", pmag.resolve(Replicated)),
			# Transformer blocks - MoE (following DeepSeek V2 patterns)
			(r"blocks/\d+/moe/gate/kernel", pmag.resolve(Replicated)),
			(r"blocks/\d+/moe/shared_experts/.*/kernel", pmag.resolve(ColumnWise)),
			(r"blocks/\d+/moe/experts/.*/up_proj/kernel", pmag.resolve(ExpertColumnWiseAlt)),
			(r"blocks/\d+/moe/experts/.*/down_proj/kernel", pmag.resolve(ExpertRowWiseAlt)),
			(r"blocks/\d+/moe/.*/bias", pmag.resolve(Replicated)),
			# Transformer blocks - Dense MLP (for first_k_dense_replace layers)
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
		return ("bias", "norm", "pos_embed", "time_embed", "label_embed", "gate")

	@staticmethod
	def rng_keys():
		"""Returns the RNG key names."""
		return ("params", "dropout")

	@property
	def head_dim(self) -> int:
		"""Dimensionality of each attention head."""
		return self.hidden_size // self.num_attention_heads

	@property
	def total_experts(self) -> int:
		"""Total number of experts (shared + routed)."""
		return self.n_shared_experts + self.n_routed_experts
