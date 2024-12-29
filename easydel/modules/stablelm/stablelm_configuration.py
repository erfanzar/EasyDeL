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


@register_config("stablelm")
class StableLmConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 50304):
	        Vocabulary size of the StableLM model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed when calling [`~easydel.modules.StableLmModel`].
	    hidden_size (`int`, *optional*, defaults to 2560):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 6912):
	        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 32):
	        Number of key-value heads for each attention layer in the Transformer encoder.
	    hidden_act (`str` or `tp.Callable`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
	        `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 4096):
	        The maximum sequence length that this model might ever be used with.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models).
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`int`, *optional*, defaults to 10000):
	        The theta value for the rotary position embeddings.
	    rope_scaling (`str`, *optional*):
	        The scaling to use for the rotary position embeddings.
	    qk_layernorm (`bool`, *optional*, defaults to `False`):
	        Whether to use layer normalization on the queries and keys in the attention layer.
	    use_parallel_residual (`bool`, *optional*, defaults to `False`):
	        Whether to use a parallel residual connection in the attention layer.
	    hidden_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    partial_rotary_factor (`float`, *optional*, defaults to 0.25):
	        The factor to scale the partial rotary embeddings by.
	    bos_token_id (`int`, *optional*, defaults to 0):
	        The id for the beginning of stream token.
	    eos_token_id (`int`, *optional*, defaults to 0):
	        The id for the end of stream token.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to. If None, the model is not quantized.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        What to save during gradient checkpointing. Choose one of `"nothing_saveable"`, `"first_half_saveable"`,
	        `"full_saveable"`.
	"""

	model_type: str = "stablelm"

	def __init__(
		self,
		vocab_size=50304,
		intermediate_size=6912,
		hidden_size=2560,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=32,
		hidden_act="silu",
		max_position_embeddings=4096,
		initializer_range=0.02,
		layer_norm_eps=1.0e-5,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=10_000,
		rope_scaling=None,
		use_qkv_bias=False,
		qk_layernorm=False,
		use_parallel_residual=False,
		hidden_dropout=0.0,
		attention_dropout=0.0,
		partial_rotary_factor=0.25,
		bos_token_id=0,
		eos_token_id=0,
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
		self.qk_layernorm = qk_layernorm
		self.use_parallel_residual = use_parallel_residual
		self.num_key_value_heads = num_key_value_heads
		self.use_qkv_bias = use_qkv_bias
		self.hidden_dropout = hidden_dropout
		self.attention_dropout = attention_dropout
		self.hidden_act = hidden_act
		self.max_position_embeddings = max_position_embeddings
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.partial_rotary_factor = partial_rotary_factor
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
				("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
				("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("input_layernorm/kernel", PartitionSpec(None)),
				("post_attention_layernorm/kernel", PartitionSpec(None)),
				("model/norm/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else (
				("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
				("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("input_layernorm/kernel", PartitionSpec(None)),
				("post_attention_layernorm/kernel", PartitionSpec(None)),
				("model/norm/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
				(".*", PartitionSpec(("fsdp", "sp"))),
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
