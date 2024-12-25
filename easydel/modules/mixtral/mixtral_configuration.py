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


@register_config("mixtral")
class MixtralConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32000):
	        Vocabulary size of the Mixtral model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 14336):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 8):
	        Number of key and value heads for each attention layer in the Transformer encoder.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 4096 * 32):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 1):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The id of the *end-of-sequence* token.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 1e6):
	        The theta value to use for rotary position embeddings.
	    sliding_window (`int`, *optional*, defaults to 4096):
	        The sliding window size.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    num_experts_per_tok (`int`, *optional*, defaults to 2):
	        The number of experts per token.
	    num_local_experts (`int`, *optional*, defaults to 8):
	        The number of local experts.
	    output_router_logits (`bool`, *optional*, defaults to `False`):
	        Whether to output router logits.
	    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
	        The router auxiliary loss coefficient.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    use_scan_mlp (`bool`, *optional*, defaults to `False`):
	        Whether to use the scan implementation for the MLP.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size to use when scanning the MLP.
	    number_rep_kv (`int`, *optional*, defaults to 1):
	        Number of repetitions for the key and value vectors.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The configuration for rope scaling.
	    attention_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use bias in the attention layer.
	    initialization_of_moe (`bool`, *optional*, defaults to `False`):
	        Whether to initialize the MoE layers.
	    router_jitter_noise (`float`, *optional*, defaults to 0.0):
	        The jitter noise for the router.
	"""

	model_type: str = "mixtral"

	def __init__(
		self,
		vocab_size=32000,
		hidden_size=4096,
		intermediate_size=14336,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=8,
		hidden_act="silu",
		max_position_embeddings=4096 * 32,
		initializer_range=0.02,
		rms_norm_eps=1e-5,
		use_cache=True,
		pad_token_id=None,
		bos_token_id=1,
		eos_token_id=2,
		tie_word_embeddings=False,
		rope_theta=1e6,
		sliding_window=4096,
		attention_dropout=0.0,
		num_experts_per_tok=2,
		num_local_experts=8,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		number_rep_kv: int = 1,
		bits: tp.Optional[int] = None,
		rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
		attention_bias: bool = False,
		initialization_of_moe: bool = False,
		router_jitter_noise=0.0,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.sliding_window = sliding_window
		self.bits = bits
		self.attention_dropout = attention_dropout
		self.num_local_experts = num_local_experts
		self.num_experts_per_tok = num_experts_per_tok
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef
		self.attention_bias = attention_bias
		# for backward compatibility
		self.rope_scaling = rope_scaling
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initialization_of_moe = initialization_of_moe
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.number_rep_kv = number_rep_kv
		self.gradient_checkpointing = gradient_checkpointing
		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
		self.router_jitter_noise = router_jitter_noise
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			use_scan_mlp=use_scan_mlp,
			scan_mlp_chunk_size=scan_mlp_chunk_size,
			bits=bits,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			(
				"self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("w1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("w2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("w3/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("gate/kernel", PartitionSpec(("fsdp", "sp"))),
			("input_layernorm/kernel", PartitionSpec(None)),
			("post_attention_layernorm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)

	def add_jax_args(
		self,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		number_rep_kv: int = 1,
		bits: tp.Optional[int] = None,
		attention_dropout: float = 0.0,
		rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
		attention_bias: bool = False,
		initialization_of_moe: bool = False,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the model:

		Args:
		    self: Bind the attributes and methods of a class to an
		        instance of that class
		    gradient_checkpointing: str: Determine whether to use
		        gradient checkpointing
		    use_scan_mlp: bool: Determine whether to use the scan_mlp
		        function or not
		    scan_mlp_chunk_size: int: Chunk the input to the mlp
		    number_rep_kv: int: Control the number of times that the key
		        and value vectors are repeated
		    bits: tp.Optional[int]: Specify the number of bits to use for
		        quantization
		    attention_dropout: float: Set the dropout rate for the
		        attention layer
		    attention_bias: bool: when ever to use attention_bias
		    initialization_of_moe: bool: initialization of moe needs to
		        disable some dynamic part's this boolean variable will
		        turn them off.
		    rope_scaling: tp.Dict[str, tp.Union[str, float]]: rope_scaling for
		        rope

		Returns:
		    A tuple of the following:
		"""
		self.attention_dropout = attention_dropout
		self.attention_bias = attention_bias
		self.rope_scaling = rope_scaling
		self.number_rep_kv = number_rep_kv
		self.gradient_checkpointing = gradient_checkpointing
		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
		self.bits = bits
		self.initialization_of_moe = initialization_of_moe

	@staticmethod
	def get_weight_decay_exclusions():
		return tuple()

	@staticmethod
	def rng_keys():
		return "params", "dropout", "fcm"

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
