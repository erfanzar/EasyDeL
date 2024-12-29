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

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("phi3")
class Phi3Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32064):
	        Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 3072):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 8192):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*):
	        Number of key and value heads for each attention layer in the Transformer encoder. Will default to
	        `num_attention_heads` if not set.
	    resid_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    embd_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the embeddings.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 4096):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    original_max_position_embeddings (`int`, *optional*, defaults to 4096):
	        The original maximum sequence length that this model might ever be used with. Typically set this to
	        something large just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The configuration for rope scaling.
	    bos_token_id (`int`, *optional*, defaults to 1):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 32000):
	        The id of the *end-of-sequence* token.
	    pad_token_id (`int`, *optional*, defaults to 32000):
	        The index of the padding token in the vocabulary.
	    sliding_window (`int`, *optional*):
	        The sliding window size.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	"""

	model_type: str = "phi3"

	def __init__(
		self,
		vocab_size=32064,
		hidden_size=3072,
		intermediate_size=8192,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=None,
		resid_pdrop=0.0,
		embd_pdrop=0.0,
		attention_dropout=0.0,
		hidden_act="silu",
		max_position_embeddings=4096,
		original_max_position_embeddings=4096,
		initializer_range=0.02,
		rms_norm_eps=1e-5,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=10000.0,
		rope_scaling=None,
		bos_token_id=1,
		eos_token_id=32000,
		pad_token_id=32000,
		sliding_window=None,
		bits: tp.Optional[int] = None,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	) -> None:
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads

		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.resid_pdrop = resid_pdrop
		self.embd_pdrop = embd_pdrop
		self.attention_dropout = attention_dropout
		self.hidden_act = hidden_act
		self.max_position_embeddings = max_position_embeddings
		self.original_max_position_embeddings = original_max_position_embeddings
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self._rope_scaling_validation()
		self.sliding_window = sliding_window

		self.bits = bits
		self.gradient_checkpointing = gradient_checkpointing
		self.head_dim = hidden_size // num_attention_heads

		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			bits=bits,
			**kwargs,
		)

	def add_jax_args(
		self,
		bits: tp.Optional[int] = None,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		self.bits = bits
		self.gradient_checkpointing = gradient_checkpointing
		for k, v in kwargs.items():
			if not hasattr(self, k):
				setattr(self, k, v)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.

		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
			("norm/kernel", PartitionSpec(("fsdp", "sp"))),
			("post_attention_layernorm/kernel", PartitionSpec(("fsdp", "sp"))),
			("input_layernorm/kernel", PartitionSpec(("fsdp", "sp"))),
			("mlp/gate_up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("self_attn/qkv_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)

	def _rope_scaling_validation(self):
		"""Validate the `rope_scaling` configuration."""
		if self.rope_scaling is None:
			return

		rope_scaling_type = self.rope_scaling.get("type", None)

		# For backward compatibility if previous version used "su" or "yarn"
		if rope_scaling_type is not None and rope_scaling_type in ["su", "yarn"]:
			self.rope_scaling["type"] = "longrope"

	@property
	def granted_freq_max_position_embedding(self) -> int:
		return getattr(
			self,
			"freq_max_position_embeddings",
			self.max_position_embeddings,
		)

	@property
	def granted_mask_max_position_embedding(self) -> int:
		return getattr(
			self,
			"mask_max_position_embeddings",
			self.max_position_embeddings,
		)
