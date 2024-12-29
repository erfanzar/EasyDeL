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


@register_config("qwen2_moe")
class Qwen2MoeConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 151936):
	        Vocabulary size of the Qwen-2 MoE model. Defines the number of different tokens that can be represented by
	        the `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 2048):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 5632):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 24):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 16):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 16):
	        Number of key and value heads for each attention layer in the Transformer encoder.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 32768):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-6):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    use_sliding_window (`bool`, *optional*, defaults to `False`):
	        Whether to use a sliding window attention.
	    sliding_window (`int`, *optional*, defaults to 4096):
	        The sliding window size.
	    max_window_layers (`int`, *optional*, defaults to 28):
	        The maximum number of layers to use for the sliding window attention.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    decoder_sparse_step (`int`, *optional*, defaults to 1):
	        The sparse step for the decoder.
	    moe_intermediate_size (`int`, *optional*, defaults to 1408):
	        The intermediate size of the MoE layer.
	    shared_expert_intermediate_size (`int`, *optional*, defaults to 5632):
	        The intermediate size of the shared expert.
	    num_experts_per_tok (`int`, *optional*, defaults to 4):
	        The number of experts per token.
	    num_experts (`int`, *optional*, defaults to 60):
	        The number of experts.
	    norm_topk_prob (`bool`, *optional*, defaults to `False`):
	        Whether to normalize the top-k probabilities.
	    output_router_logits (`bool`, *optional*, defaults to `False`):
	        Whether to output the router logits.
	    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
	        The coefficient for the router auxiliary loss.
	    mlp_only_layers (`list` of `int`, *optional*):
	        The layers that should only contain an MLP.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	"""

	model_type: str = "qwen2_moe"

	def __init__(
		self,
		vocab_size=151936,
		hidden_size=2048,
		intermediate_size=5632,
		num_hidden_layers=24,
		num_attention_heads=16,
		num_key_value_heads=16,
		hidden_act="silu",
		max_position_embeddings=32768,
		initializer_range=0.02,
		rms_norm_eps=1e-6,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=10000.0,
		use_sliding_window=False,
		sliding_window=4096,
		max_window_layers=28,
		attention_dropout=0.0,
		decoder_sparse_step=1,
		moe_intermediate_size=1408,
		shared_expert_intermediate_size=5632,
		num_experts_per_tok=4,
		num_experts=60,
		norm_topk_prob=False,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		mlp_only_layers=None,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: tp.Optional[int] = None,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.use_sliding_window = use_sliding_window
		self.sliding_window = sliding_window
		self.max_window_layers = max_window_layers

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.attention_dropout = attention_dropout

		# MoE arguments
		self.decoder_sparse_step = decoder_sparse_step
		self.moe_intermediate_size = moe_intermediate_size
		self.shared_expert_intermediate_size = shared_expert_intermediate_size
		self.num_experts_per_tok = num_experts_per_tok
		self.num_experts = num_experts
		self.norm_topk_prob = norm_topk_prob
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits
		self.mlp_only_layers = mlp_only_layers or []
		super().__init__(
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

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
				("gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("shared_expert_gate/kernel", PartitionSpec(("fsdp", "sp"))),
				("gate/kernel", PartitionSpec(("fsdp", "sp"))),
				("input_layernorm/kernel", PartitionSpec(None)),
				("post_attention_layernorm/kernel", PartitionSpec(None)),
				("model/norm/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else (
				("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
				("gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("up_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("shared_expert_gate/kernel", PartitionSpec(("fsdp", "sp"))),
				("gate/kernel", PartitionSpec(("fsdp", "sp"))),
				("input_layernorm/kernel", PartitionSpec(None)),
				("post_attention_layernorm/kernel", PartitionSpec(None)),
				("model/norm/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(("fsdp", "sp"))),
			)
		)

	def add_jax_args(
		self,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: tp.Optional[int] = None,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the Transformer class:

		Args:
		    self: Refer to the current object
		    gradient_checkpointing: str: Control the amount of memory
		        used by jax
		    bits: tp.Optional[int]: Determine the number of bits used in
		        the quantization

		Returns:
		    The following:
		"""
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

	@staticmethod
	def get_weight_decay_exclusions():
		return tuple()

	@staticmethod
	def rng_keys():
		return "params", "dropout"

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
