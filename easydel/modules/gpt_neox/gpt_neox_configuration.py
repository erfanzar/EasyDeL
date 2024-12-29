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


from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("gpt_neox")
class GPTNeoXConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

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
		attention_dropout=0.0,
		hidden_dropout=0.0,
		classifier_dropout=0.1,
		max_position_embeddings=2048,
		initializer_range=0.02,
		layer_norm_eps=1e-5,
		use_cache=True,
		bos_token_id=0,
		eos_token_id=2,
		tie_word_embeddings=False,
		use_parallel_residual=True,
		rope_scaling=None,
		attention_bias=True,
		gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
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
		self.rope_theta = rotary_emb_base
		self.classifier_dropout = classifier_dropout
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.use_cache = use_cache
		self.tie_word_embeddings = tie_word_embeddings
		self.hidden_dropout = hidden_dropout
		self.gradient_checkpointing = gradient_checkpointing
		self.attention_dropout = attention_dropout
		self.use_parallel_residual = use_parallel_residual
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.from_pt = False
		super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("wte/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			("attention/w_qkv/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("attention/w_qkv/bias", PartitionSpec(("fsdp", "sp"))),  # 1D for bias
			("attention/wo/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("attention/wo/bias", PartitionSpec(("fsdp", "sp"))),  # 1D for bias
			("mlp/dense_h_to_4h/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/dense_h_to_4h/bias", PartitionSpec(("fsdp", "sp"))),  # 1D for bias
			("mlp/dense_4h_to_h/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("mlp/dense_4h_to_h/bias", PartitionSpec(("fsdp", "sp"))),  # 1D for bias
			("post_attention_layernorm/(bias|scale)", PartitionSpec(None)),
			("input_layernorm/(bias|scale)", PartitionSpec(None)),
			("transformer/final_layer_norm/(scale|bias)", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)

	@staticmethod
	def get_mesh_names():
		return "dp", "fsdp", "tp", "sp"

	def add_jax_args(
		self,
		**kwargs,
	):
		self.from_pt = False
