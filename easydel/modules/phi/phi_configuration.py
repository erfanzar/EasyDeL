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


@register_config("phi")
class PhiConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 51200):
	        Vocabulary size of the Phi model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 2048):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 8192):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 24):
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
	    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The configuration for rope scaling.
	    partial_rotary_factor (`float`, *optional*, defaults to 0.5):
	        The factor for partial rotary embeddings.
	    qk_layernorm (`bool`, *optional*, defaults to `False`):
	        Whether to apply layer normalization to the query and key tensors.
	    bos_token_id (`int`, *optional*, defaults to 1):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The id of the *end-of-sequence* token.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	"""

	model_type: str = "phi"
	attribute_map = {
		"max_position_embeddings": "n_positions",
		"hidden_size": "n_embd",
		"num_attention_heads": "num_attention_heads",
		"num_hidden_layers": "num_hidden_layers",
	}

	def __init__(
		self,
		vocab_size=51200,
		hidden_size=2048,
		intermediate_size=8192,
		num_hidden_layers=24,
		num_attention_heads=32,
		num_key_value_heads=None,
		resid_pdrop=0.0,
		embd_pdrop=0.0,
		attention_dropout=0.0,
		hidden_act="gelu_new",
		max_position_embeddings=2048,
		initializer_range=0.02,
		layer_norm_eps=1e-5,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=10000.0,
		rope_scaling=None,
		partial_rotary_factor=0.5,
		qk_layernorm=False,
		bos_token_id=1,
		eos_token_id=2,
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
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.partial_rotary_factor = partial_rotary_factor
		self.qk_layernorm = qk_layernorm
		self.bits = bits
		self.gradient_checkpointing = gradient_checkpointing
		super().__init__(
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

	def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
		"""
		Get the partition rules for the model.

		Args:
		    fully_sharded_data_parallel (`bool`, *optional*, defaults to `True`):
		        Whether to use fully sharded data parallelism.

		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
				(
					"final_layernorm/(scale|bias)",
					PartitionSpec(
						None,
					),
				),
				(
					"final_layernorm/(scale|bias)",
					PartitionSpec(
						None,
					),
				),
				("mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(
					"mlp/fc1/bias",
					PartitionSpec(
						"tp",
					),
				),
				("mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				(
					"mlp/fc2/bias",
					PartitionSpec(
						("fsdp", "sp"),
					),
				),
				("self_attn/dense/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("self_attn/dense/bias", PartitionSpec("tp")),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				(
					"self_attn/(q_proj|k_proj|v_proj)/bias",
					PartitionSpec(
						"tp",
					),
				),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("lm_head/bias", PartitionSpec("tp")),
				(
					".*",
					PartitionSpec(
						None,
					),
				),
			)
			if fully_sharded_data_parallel
			else (
				(
					"embed_tokens/embedding",
					PartitionSpec(
						"tp",
						("fsdp", "sp"),
					),
				),
				(
					"final_layernorm/(scale|bias)",
					PartitionSpec(
						None,
					),
				),
				(
					"final_layernorm/(scale|bias)",
					PartitionSpec(
						None,
					),
				),
				("mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(
					"mlp/fc1/bias",
					PartitionSpec(
						"tp",
					),
				),
				("mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				(
					"mlp/fc2/bias",
					PartitionSpec(
						("fsdp", "sp"),
					),
				),
				(
					"self_attn/dense/kernel",
					PartitionSpec(
						"tp",
						("fsdp", "sp"),
					),
				),
				("self_attn/dense/bias", PartitionSpec("tp")),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				(
					"self_attn/(q_proj|k_proj|v_proj)/bias",
					PartitionSpec(
						"tp",
					),
				),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("lm_head/bias", PartitionSpec("tp")),
				(
					".*",
					PartitionSpec(
						None,
					),
				),
			)
		)

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
