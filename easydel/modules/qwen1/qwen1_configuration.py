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

from typing import Optional

from jax.sharding import PartitionSpec

from easydel.modules.factory import register_config
from easydel.modules.modeling_utils import EDPretrainedConfig


@register_config("qwen")
class Qwen1Config(EDPretrainedConfig):
	"""Configuration objects inherit from [`EDPretrainedConfig`] and can be used to control the model outputs. Read
	the documentation from [`EDPretrainedConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 151936):
	        Vocabulary size of the Qwen model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    emb_dropout_prob (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the embeddings.
	    attn_dropout_prob (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
	        The epsilon used by the layer normalization layers.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    seq_length (`int`, *optional*, defaults to 8192):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 512 or 1024 or 2048).
	    scale_attn_weights (`bool`, *optional*, defaults to `True`):
	        Scale attention weights by dividing by sqrt(hidden_size).
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    kv_channels (`int`, *optional*, defaults to 128):
	        The number of key-value channels.
	    rotary_pct (`float`, *optional*, defaults to 1.0):
	        The percentage of the hidden dimension to use for rotary embeddings.
	    rotary_emb_base (`int`, *optional*, defaults to 10000):
	        The base for the rotary position embedding.
	    use_dynamic_ntk (`bool`, *optional*, defaults to `True`):
	        Whether to use dynamic NTK scaling.
	    use_logn_attn (`bool`, *optional*, defaults to `True`):
	        Whether to use log-n attention.
	    intermediate_size (`int`, *optional*, defaults to 22016):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    no_bias (`bool`, *optional*, defaults to `True`):
	        Whether to use bias in the linear layers.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    softmax_in_fp32 (`bool`, *optional*, defaults to `False`):
	        Whether to compute the softmax in float32.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    use_scan_mlp (`bool`, *optional*, defaults to `False`):
	        Whether to use scan for the MLP.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size for the MLP scan.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    scan_layers (`bool`, *optional*, defaults to `True`):
	        Whether to use the scan implementation for the layers.
	    init_rope_cache_auto (`bool`, *optional*, defaults to `False`):
	        Whether to initialize the RoPE cache automatically.
	"""

	model_type: str = "qwen"

	def __init__(
		self,
		vocab_size=151936,
		hidden_size=4096,
		num_hidden_layers=32,
		num_attention_heads=32,
		emb_dropout_prob=0.0,
		attn_dropout_prob=0.0,
		layer_norm_epsilon=1e-6,
		initializer_range=0.02,
		seq_length=8192,
		scale_attn_weights=True,
		use_cache=True,
		kv_channels=128,
		rotary_pct=1.0,
		rotary_emb_base=10000,
		use_dynamic_ntk=True,
		use_logn_attn=True,
		intermediate_size=22016,
		no_bias=True,
		tie_word_embeddings=False,
		softmax_in_fp32=False,
		gradient_checkpointing: str = "nothing_saveable",
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		bits: Optional[int] = None,
		scan_layers: bool = True,
		init_rope_cache_auto: bool = False,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.seq_length = seq_length
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.scale_attn_weights = scale_attn_weights
		self.no_bias = no_bias
		self.kv_channels = kv_channels
		self.use_dynamic_ntk = use_dynamic_ntk
		self.use_logn_attn = use_logn_attn
		self.rotary_emb_base = rotary_emb_base
		self.rotary_pct = rotary_pct
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.layer_norm_epsilon = layer_norm_epsilon
		self.softmax_in_fp32 = softmax_in_fp32
		self.initializer_range = initializer_range
		self.use_cache = use_cache
		self.scan_layers = scan_layers
		self.emb_dropout_prob = emb_dropout_prob
		self.attn_dropout_prob = attn_dropout_prob
		self.init_rope_cache_auto = init_rope_cache_auto
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
		self.bits = bits
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
		    `Tuple[Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				("model/wte/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
				("self_attn/c_attn/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("self_attn/c_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("mlp/w1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("mlp/w2/kernel", PartitionSpec(("fsdp", "sp")), "tp"),
				("mlp/c_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("ln_1/kernel", PartitionSpec(None)),
				("ln_2/kernel", PartitionSpec(None)),
				("model/ln_f/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else (
				("model/wte/embedding", PartitionSpec(("fsdp", "sp"))),
				(
					"self_attn/(q_proj|k_proj|v_proj)/kernel",
					PartitionSpec(("fsdp", "sp"), "tp"),
				),
				("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
				("mlp/w1/kernel", PartitionSpec(("fsdp", "sp"))),
				("mlp/w2/kernel", PartitionSpec(("fsdp", "sp"))),
				("mlp/c_proj/kernel", PartitionSpec(("fsdp", "sp"))),
				("ln_1/kernel", PartitionSpec(None)),
				("ln_2/kernel", PartitionSpec(None)),
				("model/ln_f/kernel", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
				(".*", PartitionSpec(None)),
			)
		)

	def add_jax_args(
		self,
		gradient_checkpointing: str = "nothing_saveable",
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		bits: Optional[int] = None,
		scan_layers: bool = True,
		init_rope_cache_auto: bool = False,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the Transformer class:

		Args:
		    self: Refer to the current object
		    gradient_checkpointing: str: Control the amount of memory
		        used by jax
		    use_scan_mlp: bool: Determine whether to use the scan_mlp
		        function or not
		    scan_mlp_chunk_size: int: Set the chunk size for scan_mlp
		    init_rope_cache_auto: bool: Whether to use the
		        rope_cache_auto in model
		    bits: Optional[int]: Determine the number of bits used in
		        the quantization
		    scan_layers: bool: Determine whether to use scan layers or
		        not

		Returns:
		    The following:
		"""
		self.scan_layers = scan_layers
		self.gradient_checkpointing = gradient_checkpointing
		self.use_scan_mlp = use_scan_mlp
		self.scan_mlp_chunk_size = scan_mlp_chunk_size
		self.bits = bits
		self.init_rope_cache_auto = init_rope_cache_auto

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
			self.seq_length,
		)

	@property
	def granted_mask_max_position_embedding(self) -> int:
		return getattr(
			self,
			"mask_max_position_embeddings",
			self.seq_length,
		)
