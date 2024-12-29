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


@register_config("deepseek_v2")
class DeepseekV2Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 102400):
	        Vocabulary size of the DeepseekV2 model. Defines the number of different tokens that can be represented by
	        the `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 11008):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    moe_intermediate_size (`int`, *optional*, defaults to 1407):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the MoE layer.
	    num_hidden_layers (`int`, *optional*, defaults to 30):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 32):
	        Number of key and value heads for each attention layer in the Transformer encoder.
	    n_shared_experts (`int`, *optional*):
	        Number of shared experts.
	    n_routed_experts (`int`, *optional*):
	        Number of routed experts.
	    ep_size (`int`, *optional*, defaults to 1):
	        Expert parallel size.
	    routed_scaling_factor (`float`, *optional*, defaults to 1.0):
	        Routed scaling factor.
	    kv_lora_rank (`int`, *optional*, defaults to 512):
	        KV LoRA rank.
	    q_lora_rank (`int`, *optional*, defaults to 1536):
	        Q LoRA rank.
	    qk_rope_head_dim (`int`, *optional*, defaults to 64):
	        QK rope head dimension.
	    v_head_dim (`int`, *optional*, defaults to 128):
	        V head dimension.
	    qk_nope_head_dim (`int`, *optional*, defaults to 128):
	        QK nope head dimension.
	    topk_method (`str`, *optional*, defaults to `"gready"`):
	        Top-k method.
	    n_group (`int`, *optional*):
	        Number of groups.
	    topk_group (`int`, *optional*):
	        Top-k group.
	    num_experts_per_tok (`int`, *optional*):
	        Number of experts per token.
	    moe_layer_freq (`int`, *optional*, defaults to 1):
	        MoE layer frequency.
	    first_k_dense_replace (`int`, *optional*, defaults to 0):
	        First k dense replace.
	    norm_topk_prob (`bool`, *optional*, defaults to `False`):
	        Whether to normalize top-k probabilities.
	    scoring_func (`str`, *optional*, defaults to `"softmax"`):
	        Scoring function.
	    aux_loss_alpha (`float`, *optional*, defaults to 0.001):
	        Auxiliary loss alpha.
	    seq_aux (`bool`, *optional*, defaults to `True`):
	        Whether to use sequence auxiliary loss.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    max_position_embeddings (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-6):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 100000):
	        The index of the beginning of sequence token in the vocabulary.
	    eos_token_id (`int`, *optional*, defaults to 100001):
	        The index of the end of sequence token in the vocabulary.
	    pretraining_tp (`int`, *optional*, defaults to 1):
	        Pretraining TP.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    rope_theta (`float`, *optional*, defaults to 10000.0):
	        The theta value to use for rotary position embeddings.
	    attention_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use attention bias.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    use_scan_mlp (`bool`, *optional*, defaults to `False`):
	        Whether to use scan for MLP.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size for scan MLP.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
	        The rope scaling configuration.
	"""

	model_type: str = "deepseek_v2"

	def __init__(
		self,
		vocab_size=102400,
		hidden_size=4096,
		intermediate_size=11008,
		moe_intermediate_size=1407,
		num_hidden_layers=30,
		num_attention_heads=32,
		num_key_value_heads=32,
		n_shared_experts=None,
		n_routed_experts=None,
		ep_size=1,
		routed_scaling_factor=1.0,
		kv_lora_rank=512,
		q_lora_rank=1536,
		qk_rope_head_dim=64,
		v_head_dim=128,
		qk_nope_head_dim=128,
		topk_method="gready",
		n_group=None,
		topk_group=None,
		num_experts_per_tok=None,
		moe_layer_freq=1,
		first_k_dense_replace=0,
		norm_topk_prob=False,
		scoring_func="softmax",
		aux_loss_alpha=0.001,
		seq_aux=True,
		hidden_act="silu",
		max_position_embeddings=2048,
		initializer_range=0.02,
		rms_norm_eps=1e-6,
		use_cache=True,
		pad_token_id=None,
		bos_token_id=100000,
		eos_token_id=100001,
		pretraining_tp=1,
		tie_word_embeddings=False,
		rope_theta=10000.0,
		attention_bias=False,
		attention_dropout=0.0,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		use_scan_mlp: bool = False,
		scan_mlp_chunk_size: int = 1024,
		bits: tp.Optional[int] = None,
		rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.moe_intermediate_size = moe_intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.n_shared_experts = n_shared_experts
		self.n_routed_experts = n_routed_experts
		self.ep_size = ep_size
		self.routed_scaling_factor = routed_scaling_factor
		self.kv_lora_rank = kv_lora_rank
		self.q_lora_rank = q_lora_rank
		self.qk_rope_head_dim = qk_rope_head_dim
		self.v_head_dim = v_head_dim
		self.qk_nope_head_dim = qk_nope_head_dim
		self.topk_method = topk_method
		self.n_group = n_group
		self.topk_group = topk_group
		self.num_experts_per_tok = num_experts_per_tok
		self.moe_layer_freq = moe_layer_freq
		self.first_k_dense_replace = first_k_dense_replace
		self.norm_topk_prob = norm_topk_prob
		self.scoring_func = scoring_func
		self.aux_loss_alpha = aux_loss_alpha
		self.seq_aux = seq_aux
		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.pretraining_tp = pretraining_tp
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout
		self.gradient_checkpointing = gradient_checkpointing
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
			("model/embed_tokens/embedding", PartitionSpec("tp", ("sp", "fsdp"))),
			(
				"self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
			("gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
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
		bits: tp.Optional[int] = None,
		rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
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
		    bits: tp.Optional[int]: Specify the number of bits to use for
		        quantization

		Returns:
		    A tuple of the following:
		"""
		self.rope_scaling = rope_scaling
		self.gradient_checkpointing = gradient_checkpointing
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
