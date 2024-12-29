# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


import math

from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("mamba2")
class Mamba2Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 50280):
	        Vocabulary size of the Mamba model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 768):
	        Dimensionality of the encoder layers and the pooler layer.
	    state_size (`int`, *optional*, defaults to 16):
	        State size of the Mamba model.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    pad_token_id (`int`, *optional*, defaults to 0):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 0):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 0):
	        The id of the *end-of-sequence* token.
	    expand (`int`, *optional*, defaults to 2):
	        Expansion factor for the intermediate size.
	    conv_kernel (`int`, *optional*, defaults to 4):
	        Kernel size of the convolution layer.
	    use_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use bias in the linear layers.
	    use_conv_bias (`bool`, *optional*, defaults to `True`):
	        Whether to use bias in the convolution layer.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    initializer_range (`float`, *optional*, defaults to 0.1):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
	        Whether to compute the residual connection in float32.
	    time_step_rank (`str` or `int`, *optional*, defaults to `"auto"`):
	        The rank of the time step embedding. If set to `"auto"`, the rank is calculated as
	        `math.ceil(self.hidden_size / 16)`.
	    time_step_scale (`float`, *optional*, defaults to 1.0):
	        The scale factor for the time step embedding.
	    time_step_min (`float`, *optional*, defaults to 0.001):
	        The minimum value for the time step embedding.
	    time_step_max (`float`, *optional*, defaults to 0.1):
	        The maximum value for the time step embedding.
	    time_step_floor (`float`, *optional*, defaults to 1e-4):
	        The floor value for the time step embedding.
	    rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
	        Whether to rescale the pre-norm residual.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	"""

	model_type: str = "mamba2"

	def __init__(
		self,
		num_heads=128,
		head_dim=64,
		vocab_size=32768,
		hidden_size=4096,
		state_size=128,
		num_hidden_layers=64,
		layer_norm_epsilon=1e-5,
		pad_token_id=1,
		bos_token_id=0,
		eos_token_id=2,
		expand=2,
		conv_kernel=4,
		n_groups=8,
		use_bias=False,
		use_conv_bias=True,
		hidden_act="silu",
		initializer_range=0.1,
		residual_in_fp32=True,
		time_step_rank="auto",
		time_step_min=0.001,
		time_step_max=0.1,
		time_step_floor=1e-4,
		time_step_limit=(0.0, float("inf")),
		rescale_prenorm_residual=False,
		use_cache=True,
		norm_before_gate=True,
		rms_norm=True,
		chunk_size=256,
		tie_word_embeddings=False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.state_size = state_size
		self.num_hidden_layers = num_hidden_layers
		self.layer_norm_epsilon = layer_norm_epsilon
		self.conv_kernel = conv_kernel
		self.expand = expand

		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.pad_token_id = pad_token_id
		self.use_bias = use_bias
		self.use_conv_bias = use_conv_bias
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.time_step_rank = (
			math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
		)
		self.time_step_min = time_step_min
		self.time_step_max = time_step_max
		self.time_step_floor = time_step_floor
		self.rescale_prenorm_residual = rescale_prenorm_residual
		self.residual_in_fp32 = residual_in_fp32
		self.use_cache = use_cache
		self.n_groups = n_groups
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.norm_before_gate = norm_before_gate
		self.rms_norm = rms_norm
		self.state_size = state_size
		self.chunk_size = chunk_size
		self.time_step_limit = time_step_limit
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.intermediate_size = int(expand * hidden_size)
		super().__init__(**kwargs)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			# Embeddings
			("backbone/embeddings/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			# Language model head
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("lm_head/bias", PartitionSpec(None)),
			# Mixer layers
			("mixer/in_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mixer/in_proj/bias", PartitionSpec(None)),
			("mixer/out_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("mixer/out_proj/bias", PartitionSpec(None)),
			# Conv1d in mixer (3D kernel)
			("mixer/conv1d/kernel", PartitionSpec("tp", None, None)),
			("mixer/conv1d/bias", PartitionSpec("tp")),
			# State space parameters
			("mixer/A_log", PartitionSpec(None)),
			("mixer/D", PartitionSpec(None)),
			("mixer/dt_bias", PartitionSpec(None)),
			# Normalization layers
			("mixer/norm/kernel", PartitionSpec(None)),
			("backbone/layers/.*/norm/kernel", PartitionSpec(None)),
			("backbone/norm_f/kernel", PartitionSpec(None)),
			# Catch-all
			(".*", PartitionSpec(None)),
		)

	def add_jax_args(
		self,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
	):
		self.gradient_checkpointing = gradient_checkpointing
