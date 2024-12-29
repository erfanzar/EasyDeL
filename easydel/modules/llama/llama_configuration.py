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


@register_config("llama")
class LlamaConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32000):
	        Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 11008):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    number_rep_kv (`int`, *optional*, defaults to 1):
	        Number of repetitions for the key and value vectors.
	    num_key_value_heads (`int`, *optional*):
	        Number of key and value heads for each attention layer in the Transformer encoder. Will default to
	        `number_rep_kv * num_attention_heads` if not set.
	    max_position_embeddings (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
			head_dim (`int`, *optional*):
					head_dim for attention qkv.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-6):
	        The epsilon used by the rms normalization layers.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    bos_token_id (`int`, *optional*, defaults to 0):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 1):
	        The id of the *end-of-sequence* token.
	    resid_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    embd_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the embeddings.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    attention_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use attention bias.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    fcm_min_ratio (`float`, *optional*, defaults to -1):
	        The minimum ratio for Flash Attention.
	    fcm_max_ratio (`float`, *optional*, defaults to -1):
	        The maximum ratio for Flash Attention.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The configuration for rope scaling.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size to use when scanning the MLP.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    hidden_act (`str`, *optional*, defaults to `"silu"`):
	        The hidden activation function to use.
	    pretraining_tp (`int`, *optional*, defaults to 1):
	        The tensor parallelism degree used during pretraining.
	    mlp_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use bias in the MLP.
	    scan_layers (`bool`, *optional*, defaults to `False`):
	        Whether to use the scan implementation for the layers.
	"""

	model_type: str = "llama"

	def __init__(
		self,
		vocab_size: int = 32000,
		hidden_size: int = 4096,
		intermediate_size: int = 11008,
		num_hidden_layers: int = 32,
		num_attention_heads: int = 32,
		number_rep_kv: int = 1,
		head_dim: tp.Optional[int] = None,
		num_key_value_heads: tp.Optional[int] = None,
		max_position_embeddings: int = 2048,
		rms_norm_eps: float = 1e-6,
		initializer_range: float = 0.02,
		use_cache: bool = True,
		bos_token_id: int = 0,
		eos_token_id: int = 1,
		resid_pdrop: float = 0.0,
		embd_pdrop: float = 0.0,
		attention_dropout: float = 0.0,
		rope_theta: float = 10000.0,
		attention_bias: bool = False,
		tie_word_embeddings: bool = False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		fcm_min_ratio: float = -1,
		fcm_max_ratio: float = -1,
		rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
		scan_mlp_chunk_size: int = 1024,
		bits: tp.Optional[int] = None,
		hidden_act: str = "silu",
		pretraining_tp: int = 1,
		mlp_bias: bool = False,
		scan_layers: bool = False,
		**kwargs,
	):
		num_key_value_heads = num_key_value_heads or number_rep_kv * num_attention_heads
		self.num_key_value_heads = num_key_value_heads
		self.vocab_size = vocab_size

		self.number_rep_kv = number_rep_kv
		self.hidden_size = hidden_size
		self.initializer_range = initializer_range
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.rope_theta = rope_theta
		self.attention_bias = attention_bias
		self.num_attention_heads = num_attention_heads
		self.max_position_embeddings = max_position_embeddings
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.pretraining_tp = pretraining_tp
		self.resid_pdrop = resid_pdrop
		self.embd_pdrop = embd_pdrop
		self.attention_dropout = attention_dropout
		self.gradient_checkpointing = gradient_checkpointing
		self.mlp_bias = mlp_bias
		self.fcm_min_ratio = fcm_min_ratio
		self.hidden_act = hidden_act
		self.fcm_max_ratio = fcm_max_ratio
		self.rope_scaling = rope_scaling
		self.bits = bits
		self.scan_layers = scan_layers
		self.head_dim = (
			head_dim if head_dim is not None else hidden_size // num_attention_heads
		)
		super().__init__(
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
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

	def add_jax_args(
		self,
		resid_pdrop: float = 0.0,
		embd_pdrop: float = 0.0,
		attention_dropout: float = 0.0,
		tie_word_embeddings: bool = False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		fcm_min_ratio: float = 0.0,
		fcm_max_ratio: float = 0.0,
		number_rep_kv: int = 1,
		bits: tp.Optional[int] = None,
		rope_theta: float = 10000.0,
		attention_bias: bool = False,
		hidden_act: str = "silu",
		scan_layers: bool = True,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the Transformer class:

		Args:
		    self: Refer to the current object
		    resid_pdrop: float: Set the dropout rate for residual
		        connections
		    embd_pdrop: float: Set the probability of dropping an
		        embedding
		    attention_dropout: float: Set the probability of dropping
		        out the attention layer
		    tie_word_embeddings: bool: Tie the word embeddings to the
		        decoder
		    gradient_checkpointing: str: Control the amount of memory
		        used by jax
		    fcm_min_ratio: float: Control the minimum ratio of the
		        number of chunks to be used in flash-based computation
		    fcm_max_ratio: float: Set the maximum ratio of the number of
		        input tokens to output tokens
		    number_rep_kv: int: Determine how many times the key and
		        value vectors are repeated
		    bits: tp.Optional[int]: Determine the number of bits used in
		        the quantization
		    rope_theta: float : rope_theta for compute rope
		    attention_bias: bool : whenever to use attention bias or no
		    hidden_act: str : hidden_act for mlp
		    scan_layers: bool: Determine whether to use scan layers or
		        not
		"""
		self.scan_layers = scan_layers
		self.embd_pdrop = embd_pdrop
		self.number_rep_kv = number_rep_kv
		self.resid_pdrop = resid_pdrop
		self.rope_theta = rope_theta
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout
		self.hidden_act = hidden_act
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.fcm_min_ratio = fcm_min_ratio
		self.fcm_max_ratio = fcm_max_ratio
		self.bits = bits

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


class VisionLlamaConfig(LlamaConfig):
	def __init__(
		self,
		vision_vocab_size=8448,
		tie_vision_embeddings=False,
		sample_mode="all",
		**kwargs,
	):
		super().__init__(**kwargs)
		self.vision_vocab_size = vision_vocab_size
		self.tie_vision_embeddings = tie_vision_embeddings
		self.sample_mode = sample_mode

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			("model/embed_vision/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			(
				"self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("input_layernorm/kernel", PartitionSpec(None)),
			("post_attention_layernorm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("vision_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
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
