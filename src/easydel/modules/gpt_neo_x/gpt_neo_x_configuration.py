from jax.sharding import PartitionSpec

from easydel.modules.modeling_utils import EDPretrainedConfig


class GPTNeoXConfig(EDPretrainedConfig):
	"""
	Configuration objects inherit from [`EDPretrainedConfig`] and can be used to control the model outputs. Read
	the documentation from [`EDPretrainedConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 50432):
	        Vocabulary size of the GPT NeoX model. Defines the number of different tokens that can be represented by
	        the `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 6144):
	        Dimensionality of the encoder layers and the pooler layer.
	    num_hidden_layers (`int`, *optional*, defaults to 44):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 64):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    intermediate_size (`int`, *optional*, defaults to 24576):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    rotary_pct (`float`, *optional*, defaults to 0.25):
	        The percentage of hidden dimensions to allocate to rotary embeddings.
	    rotary_emb_base (`int`, *optional*, defaults to 10000):
	        The base for the rotary position embedding.
	    classifier_dropout (`float`, *optional*, defaults to 0.1):
	        The dropout ratio for the classifier layer.
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
	    bos_token_id (`int`, *optional*, defaults to 0):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The id of the *end-of-sequence* token.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    gradient_checkpointing (`str`, *optional*, defaults to `"everything_saveable"`):
	        The gradient checkpointing configuration.
	    use_parallel_residual (`bool`, *optional*, defaults to `True`):
	        Whether to use a parallel residual connection in the attention layer.
	"""

	model_type: str = "gpt_neox"

	def __init__(
		self,
		vocab_size=50432,
		hidden_size=6144,
		num_hidden_layers=44,
		num_attention_heads=64,
		intermediate_size=24576,
		hidden_act="gelu",
		rotary_pct=0.25,
		rotary_emb_base=10000,
		classifier_dropout=0.1,
		max_position_embeddings=2048,
		initializer_range=0.02,
		layer_norm_eps=1e-5,
		use_cache=True,
		bos_token_id=0,
		eos_token_id=2,
		tie_word_embeddings=False,
		gradient_checkpointing="everything_saveable",
		use_parallel_residual=True,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.intermediate_size = intermediate_size
		self.hidden_act = hidden_act
		self.rotary_pct = rotary_pct
		self.rotary_emb_base = rotary_emb_base
		self.classifier_dropout = classifier_dropout
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_cache = use_cache
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing

		self.use_parallel_residual = use_parallel_residual
		self.from_pt = False
		super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

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
				("wte/embedding", PartitionSpec("fsdp", "dp")),
				("attention/w_qkv/(kernel|bias)", PartitionSpec("fsdp", "dp")),
				("attention/wo/(kernel|bias)", PartitionSpec("fsdp", "dp")),
				("mlp/dense_h_to_4h/(kernel|bias)", PartitionSpec("fsdp", "dp")),
				("mlp/dense_4h_to_h/(kernel|bias)", PartitionSpec("dp", "fsdp")),
				("post_attention_layernorm/(bias|scale)", PartitionSpec("fsdp", "dp")),
				("input_layernorm/(bias|scale)", PartitionSpec("fsdp", "dp")),
				(
					"transformer/final_layer_norm/(scale|bias)",
					PartitionSpec("dp", "fsdp"),
				),
				("lm_head/kernel", PartitionSpec("dp", "fsdp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else (
				("embed_in/embedding", PartitionSpec("fsdp")),
				("attention/w_qkv/(kernel|bias)", PartitionSpec("fsdp")),
				("attention/wo/(kernel|bias)", PartitionSpec("fsdp")),
				("mlp/dense_h_to_4h/(kernel|bias)", PartitionSpec("fsdp")),
				("mlp/dense_4h_to_h/(kernel|bias)", PartitionSpec("fsdp")),
				("post_attention_layernorm/(bias|scale)", PartitionSpec("fsdp")),
				("input_layernorm/(bias|scale)", PartitionSpec("fsdp")),
				("transformer/final_layer_norm/(scale|bias)", PartitionSpec("fsdp")),
				("lm_head/kernel", PartitionSpec("fsdp")),
				(".*", PartitionSpec(("fsdp", "sp"))),
			)
		)

	@staticmethod
	def get_mesh_names():
		return "dp", "fsdp", "tp", "sp"

	def add_jax_args(
		self,
		**kwargs,
	):
		self.from_pt = False
