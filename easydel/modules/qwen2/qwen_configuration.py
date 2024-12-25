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


@register_config("qwen2")
class Qwen2Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read
	the documentation from [`PretrainedConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 151936):
	        Vocabulary size of the Qwen-2 model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 22016):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 32):
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
	    resid_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    embd_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the embeddings.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    fcm_min_ratio (`float`, *optional*, defaults to 0.0):
	        The minimum ratio for Flash Attention.
	    fcm_max_ratio (`float`, *optional*, defaults to 0.0):
	        The maximum ratio for Flash Attention.
	    use_scan_mlp (`bool`, *optional*, defaults to `False`):
	        Whether to use the scan implementation for the MLP.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size to use when scanning the MLP.
	    number_rep_kv (`int`, *optional*, defaults to 1):
	        Number of repetitions for the key and value vectors.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    scan_layers (`bool`, *optional*, defaults to `True`):
	        Whether to use the scan implementation for the layers.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The configuration for rope scaling.
	"""

	model_type: str = "qwen2"

	def __init__(
		self,
		vocab_size=151936,
		hidden_size=4096,
		intermediate_size=22016,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=32,
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
		resid_pdrop: float = 0.0,
		embd_pdrop: float = 0.0,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		fcm_min_ratio: float = 0.0,
		fcm_max_ratio: float = 0.0,
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		number_rep_kv: int = 1,
		bits: tp.Optional[int] = None,
		scan_layers: bool = True,
		rope_scaling: tp.Optional[tp.Mapping[str, str | float]] = None,
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

		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.rope_scaling = rope_scaling
		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.scan_layers = scan_layers
		self.embd_pdrop = embd_pdrop
		self.number_rep_kv = number_rep_kv
		self.resid_pdrop = resid_pdrop
		self.attention_dropout = attention_dropout
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.fcm_min_ratio = fcm_min_ratio
		self.fcm_max_ratio = fcm_max_ratio
		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
		self.bits = bits

		if self.rope_scaling is not None and "type" in self.rope_scaling:
			self.rope_scaling["rope_type"] = self.rope_scaling["type"]
		super().__init__(
			tie_word_embeddings=tie_word_embeddings,
			use_scan_mlp=use_scan_mlp,
			scan_mlp_chunk_size=scan_mlp_chunk_size,
			bits=bits,
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
			("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
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
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		number_rep_kv: int = 1,
		bits: tp.Optional[int] = None,
		rope_theta: float = 10000.0,
		hidden_act: str = "silu",
		scan_layers: bool = True,
		rope_scaling: tp.Optional[tp.Mapping[str, str | float]] = None,
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
		    use_scan_mlp: bool: Determine whether to use the scan_mlp
		        function or not
		    scan_mlp_chunk_size: int: Set the chunk size for scan_mlp
		    number_rep_kv: int: Determine how many times the key and
		        value vectors are repeated
		    bits: tp.Optional[int]: Determine the number of bits used in
		        the quantization
		    rope_theta: float : rope_theta for compute rope
		    hidden_act: str : hidden_act for mlp
		    scan_layers: bool: Determine whether to use scan layers or
		        not

		Returns:
		    The following:
		"""
		self.scan_layers = scan_layers
		self.embd_pdrop = embd_pdrop
		self.number_rep_kv = number_rep_kv
		self.resid_pdrop = resid_pdrop
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_dropout = attention_dropout
		self.hidden_act = hidden_act
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.fcm_min_ratio = fcm_min_ratio
		self.fcm_max_ratio = fcm_max_ratio

		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
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
