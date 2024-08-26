from typing import Optional, Union

from jax.sharding import PartitionSpec

from easydel.modules.modeling_utils import EDPretrainedConfig


class MptAttentionConfig(EDPretrainedConfig):
	"""
	This is the configuration class to store the attention related configuration of a [`MptModel`].

	Args:
	    attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
	        The type of attention to use. Can be either `"multihead_attention"` or `"multiquery_attention"`.
	    attn_pdrop (`float`, *optional*, defaults to 0.0):
	        The dropout probability applied to the attention output.
	    attn_impl (`str`, *optional*, defaults to `"torch"`):
	        The implementation of the attention mechanism. Can be either `"torch"` or `"flash"`.
	    clip_qkv (`float`, *optional*):
	        The clip value applied to the query, key, and value tensors.
	    softmax_scale (`float`, *optional*):
	        The scale factor applied to the softmax function in the attention layer.
	    prefix_lm (`bool`, *optional*, defaults to `False`):
	        Whether to use a prefix LM.
	    qk_ln (`bool`, *optional*, defaults to `False`):
	        Whether to apply layer normalization to the query and key tensors.
	    attn_uses_sequence_id (`bool`, *optional*, defaults to `False`):
	        Whether the attention layer uses sequence IDs.
	    alibi (`bool`, *optional*, defaults to `True`):
	        Whether to use the ALiBi (Attention with Linear Biases) method.
	    alibi_bias_max (`int`, *optional*, defaults to 8):
	        The maximum value for the ALiBi bias.
	"""

	def __init__(
		self,
		attn_type="multihead_attention",
		attn_pdrop=0,
		attn_impl="torch",
		clip_qkv=None,
		softmax_scale=None,
		prefix_lm=False,
		qk_ln=False,
		attn_uses_sequence_id=False,
		alibi=True,
		alibi_bias_max=8,
		**kwargs,
	):
		super().__init__()
		self.attn_type = attn_type
		self.attn_pdrop = attn_pdrop
		self.attn_impl = attn_impl
		self.clip_qkv = clip_qkv
		self.softmax_scale = softmax_scale
		self.prefix_lm = prefix_lm
		self.attn_uses_sequence_id = attn_uses_sequence_id
		self.alibi = alibi
		self.qk_ln = qk_ln
		self.alibi_bias_max = alibi_bias_max

		if attn_type not in ["multihead_attention", "multiquery_attention"]:
			raise ValueError(
				f"`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}"
			)

	@classmethod
	def from_pretrained(
		cls, pretrained_model_name_or_path, **kwargs
	) -> "EDPretrainedConfig":
		cls._set_token_in_kwargs(kwargs)
		config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
		if config_dict.get("model_type") == "mpt":
			config_dict = config_dict["attn_config"]
		return cls.from_dict(config_dict, **kwargs)


class MptConfig(EDPretrainedConfig):
	"""
	Configuration objects inherit from [`EDPretrainedConfig`] and can be used to control the model outputs. Read
	the documentation from [`EDPretrainedConfig`] for more information.

	Args:
	    d_model (`int`, *optional*, defaults to 2048):
	        Dimensionality of the encoder layers and the pooler layer.
	    n_heads (`int`, *optional*, defaults to 16):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    n_layers (`int`, *optional*, defaults to 24):
	        Number of hidden layers in the Transformer encoder.
	    expansion_ratio (`int`, *optional*, defaults to 4):
	        Expansion ratio of the feed-forward layer.
	    max_seq_len (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    vocab_size (`int`, *optional*, defaults to 50368):
	        Vocabulary size of the MPT model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    resid_prob_drop (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    emb_prob_drop (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the embeddings.
	    learned_pos_emb (`bool`, *optional*, defaults to `True`):
	        Whether to learn positional embeddings.
	    attn_config ([`MptAttentionConfig`], *optional*):
	        The configuration of the attention layer.
	    init_device (`str`, *optional*, defaults to `"cpu"`):
	        The device to initialize the model on.
	    logit_scale (`float` or `str`, *optional*):
	        The logit scale. If set to `"inv_sqrt_d_model"`, the logit scale is calculated as `1 / math.sqrt(d_model)`.
	    no_bias (`bool`, *optional*, defaults to `True`):
	        Whether to use bias in the linear layers.
	    verbose (`int`, *optional*, defaults to 0):
	        The verbosity level.
	    embedding_fraction (`float`, *optional*, defaults to 1.0):
	        The fraction of the embedding matrix to use.
	    norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
	        The type of layer normalization to use.
	    use_cache (`bool`, *optional*, defaults to `False`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    alibi (`bool`, *optional*, defaults to `True`):
	        Whether to use ALiBi (Attention with Linear Biases) method.
	    use_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use bias in the linear layers.
	    act_fn (`str`, *optional*, defaults to `"gelu"`):
	        The activation function to use.
	    qk_ln (`bool`, *optional*, defaults to `False`):
	        Whether to apply layer normalization to the query and key tensors.
	    use_lm_head (`bool`, *optional*, defaults to `False`):
	        Whether to use a language modeling head.
	    use_norm_bias (`bool`, *optional*, defaults to `False`):
	        Whether to use bias in the layer normalization layers.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	"""

	model_type = "mpt"
	attribute_map = {
		"num_attention_heads": "n_heads",
		"hidden_size": "d_model",
		"num_hidden_layers": "n_layers",
	}

	def __init__(
		self,
		d_model: int = 2048,
		n_heads: int = 16,
		n_layers: int = 24,
		expansion_ratio: int = 4,
		max_seq_len: int = 2048,
		vocab_size: int = 50368,
		resid_prob_drop: float = 0.0,
		layer_norm_epsilon: float = 1e-5,
		emb_prob_drop: float = 0.0,
		learned_pos_emb: bool = True,
		attn_config: MptAttentionConfig = None,
		init_device: str = "cpu",
		logit_scale: Optional[Union[float, str]] = None,
		no_bias: bool = True,
		verbose: int = 0,
		embedding_fraction: float = 1.0,
		norm_type: str = "low_precision_layernorm",
		use_cache: bool = False,
		initializer_range=0.02,
		alibi: bool = True,
		use_bias: bool = False,
		act_fn: str = "gelu",
		qk_ln: bool = False,
		use_lm_head: bool = False,
		use_norm_bias: bool = False,
		gradient_checkpointing: str = "nothing_saveable",
		bits: Optional[int] = None,
		**kwargs,
	):
		if attn_config is None:
			self.attn_config = MptAttentionConfig()
		elif isinstance(attn_config, dict):
			self.attn_config = MptAttentionConfig(**attn_config)
		else:
			self.attn_config = attn_config
		self.d_model = d_model
		self.use_norm_bias = use_norm_bias
		self.use_lm_head = use_lm_head
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.expansion_ratio = expansion_ratio
		self.max_seq_len = max_seq_len
		self.vocab_size = vocab_size
		self.resid_prob_drop = resid_prob_drop
		self.use_bias = use_bias
		self.emb_prob_drop = emb_prob_drop
		self.gradient_checkpointing = gradient_checkpointing
		self.norm_type = norm_type
		self.learned_pos_emb = learned_pos_emb
		self.act_fn = act_fn
		self.logit_scale = logit_scale
		self.no_bias = no_bias
		self.qk_ln = qk_ln
		self.alibi = alibi
		self.verbose = verbose
		self.initializer_range = initializer_range
		self.embedding_fraction = embedding_fraction
		self.init_device = init_device
		self.use_cache = use_cache
		self.bits = bits
		self.layer_norm_epsilon = layer_norm_epsilon
		self.from_pt = False
		super().__init__(bits=bits, **kwargs)

	@staticmethod
	def _set_config_defaults(config, config_defaults):
		for k, v in config_defaults.items():
			if k not in config:
				config[k] = v
		return config

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
				("transformer/wte/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
				("attn/Wqkv/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("attn/out_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
				("ffn/down_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("ffn/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("transformer/norm_1/scale", PartitionSpec(None)),
				("transformer/norm_2/scale", PartitionSpec(None)),
				("transformer/norm_f/scale", PartitionSpec(None)),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else (
				("transformer/wte/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
				("attn/Wqkv/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("attn/out_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("ffn/down_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("ffn/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				("transformer/norm_1/scale", PartitionSpec(("fsdp", "sp"))),
				("transformer/norm_2/scale", PartitionSpec(("fsdp", "sp"))),
				("transformer/norm_f/scale", PartitionSpec(("fsdp", "sp"))),
				("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
				(".*", PartitionSpec(("fsdp", "sp"))),
			)
		)

	def add_jax_args(
		self,
		gradient_checkpointing: str = "nothing_saveable",
		bits: Optional[int] = None,
		**kwargs,
	):
		if hasattr(self, "attn_config"):
			for k, v in self.attn_config.__dict__.items():
				setattr(self, k, v)
		basics = dict(bits=bits, gradient_checkpointing=gradient_checkpointing, **kwargs)
		for k, v in basics.items():
			if not hasattr(self, k):
				setattr(self, k, v)
		self.from_pt = False

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
