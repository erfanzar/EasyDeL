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


import typing as tp

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("cohere2")
class Cohere2Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 256000):
	        Vocabulary size of the Cohere model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 8192):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 22528):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    logit_scale (`float`, *optional*, defaults to 0.0625):
	        A logit scale value used in the attention layer.
	    num_hidden_layers (`int`, *optional*, defaults to 40):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 64):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*):
	        Number of key and value heads for each attention layer in the Transformer encoder. Will default to
	        `num_attention_heads` if not set.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 8192):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*, defaults to 0):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 5):
	        The index of the beginning of sequence token in the vocabulary.
	    eos_token_id (`int`, *optional*, defaults to 255001):
	        The index of the end of sequence token in the vocabulary.
	    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    attention_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use attention bias.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    use_qk_norm (`bool`, *optional*, defaults to `False`):
	        Whether to use query and key normalization.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	"""

	model_type: str = "cohere"

	def __init__(
		self,
		vocab_size=256000,
		hidden_size=8192,
		intermediate_size=22528,
		logit_scale=0.0625,
		num_hidden_layers=40,
		num_attention_heads=64,
		num_key_value_heads=None,
		hidden_act="silu",
		max_position_embeddings=8192,
		initializer_range=0.02,
		layer_norm_eps=1e-5,
		use_cache=True,
		pad_token_id=0,
		bos_token_id=5,
		eos_token_id=255001,
		tie_word_embeddings=True,
		rope_theta=10000.0,
		rope_scaling=None,
		attention_bias=False,
		attention_dropout=0.0,
		sliding_window=4096,
		sliding_window_pattern=4,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: tp.Optional[int] = None,
		**kwargs,
	):
		"""Initializes the Cohere2Config instance.

		Args:
		    vocab_size (int): Vocabulary size.
		    hidden_size (int): Dimensionality of the hidden layers.
		    intermediate_size (int): Dimensionality of the intermediate feed-forward layer.
		    logit_scale (float): Logit scale for attention.
		    num_hidden_layers (int): Number of hidden layers.
		    num_attention_heads (int): Number of attention heads.
		    num_key_value_heads (Optional[int]): Number of key/value heads.
		    hidden_act (str): Activation function.
		    max_position_embeddings (int): Maximum sequence length.
		    initializer_range (float): Initializer range for weights.
		    layer_norm_eps (float): Epsilon for layer normalization.
		    use_cache (bool): Whether to use caching.
		    pad_token_id (int): Padding token ID.
		    bos_token_id (int): Beginning of sequence token ID.
		    eos_token_id (int): End of sequence token ID.
		    tie_word_embeddings (bool): Whether to tie word embeddings.
		    rope_theta (float): RoPE theta value.
		    rope_scaling (Optional[dict]): RoPE scaling configuration.
		    attention_bias (bool): Whether to use attention bias.
		    attention_dropout (float): Dropout rate for attention.
		    sliding_window (int): Sliding window size for attention.
		    sliding_window_pattern (int): Pattern for sliding window attention (unused in current Flax implementation).
		    gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing strategy.
		    bits (Optional[int]): Number of bits for quantization.
		    **kwargs: Additional keyword arguments.
		"""
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.logit_scale = logit_scale
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads

		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout
		self.sliding_window = sliding_window
		self.sliding_window_pattern = sliding_window_pattern
		self.head_dim = hidden_size // num_attention_heads
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
			("self_attn/q_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/k_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/v_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("linear/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("linear_1/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("linear_v/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("gate/kernel", PartitionSpec("tp")),
			("post_attn_norm/kernel", PartitionSpec(None)),
			("pre_attn_norm/kernel", PartitionSpec(None)),
			("pre_moe_norm/kernel", PartitionSpec(None)),
			("post_moe_norm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns a tuple of parameter names for which weight decay should be excluded."""
		return tuple()

	@staticmethod
	def rng_keys():
		"""Returns the names of the random number generator keys used by the model."""
		return "params", "dropout"

	@property
	def granted_freq_max_position_embedding(self) -> int:
		"""Returns the maximum position embedding size for frequency-based position embeddings, falling back to max_position_embeddings."""
		return getattr(
			self,
			"freq_max_position_embeddings",
			self.max_position_embeddings,
		)

	@property
	def granted_mask_max_position_embedding(self) -> int:
		"""Returns the maximum position embedding size for mask-based position embeddings, falling back to max_position_embeddings."""
		return getattr(
			self,
			"mask_max_position_embeddings",
			self.max_position_embeddings,
		)
