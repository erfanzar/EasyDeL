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


@register_config("opt")
class OPTConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 50272):
	        Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 768):
	        Dimensionality of the encoder layers and the pooler layer.
	    num_hidden_layers (`int`, *optional*, defaults to 12):
	        Number of hidden layers in the Transformer encoder.
	    ffn_dim (`int`, *optional*, defaults to 3072):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    max_position_embeddings (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 512 or 1024 or 2048).
	    do_layer_norm_before (`bool`, *optional*, defaults to `True`):
	        Whether to perform layer normalization before the attention block.
	    _remove_final_layer_norm (`bool`, *optional*, defaults to `False`):
	        Whether to remove the final layer norm.
	    word_embed_proj_dim (`int`, *optional*):
	        The dimension of the word embedding projection. If not provided, it will default to `hidden_size`.
	    dropout (`float`, *optional*, defaults to 0.1):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    num_attention_heads (`int`, *optional*, defaults to 12):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
	        The non-linear activation function (function or string) to use in the encoder and pooler. If string,
	        `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
	    layerdrop (`float`, *optional*, defaults to 0.0):
	        The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
	        for more details.
	    init_std (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*, defaults to 1):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 2):
	        The id of the *beginning-of-sequence* token.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The id of the *end-of-sequence* token.
	    enable_bias (`bool`, *optional*, defaults to `True`):
	        Whether to use bias in the linear layers.
	    layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
	        Whether to use elementwise affine in the layer normalization layers.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	"""

	model_type: str = "opt"
	keys_to_ignore_at_inference = ["past_key_values"]

	def __init__(
		self,
		vocab_size: int = 50272,
		hidden_size: int = 768,
		num_hidden_layers: int = 12,
		ffn_dim: int = 3072,
		max_position_embeddings: int = 2048,
		do_layer_norm_before: bool = True,
		_remove_final_layer_norm: bool = False,
		word_embed_proj_dim: int = None,
		dropout: float = 0.1,
		attention_dropout: float = 0.0,
		num_attention_heads: int = 12,
		activation_function: str = "relu",
		layerdrop: float = 0.0,
		init_std: float = 0.02,
		use_cache: bool = True,
		pad_token_id: int = 1,
		bos_token_id: int = 2,
		eos_token_id: int = 2,
		enable_bias: bool = True,
		layer_norm_elementwise_affine: bool = True,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			**kwargs,
		)
		self.vocab_size = vocab_size
		self.gradient_checkpointing = gradient_checkpointing
		self.max_position_embeddings = max_position_embeddings
		self.num_attention_heads = num_attention_heads
		self.word_embed_proj_dim = (
			word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
		)
		self.ffn_dim = ffn_dim
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.dropout = dropout
		self.attention_dropout = attention_dropout
		self.activation_function = activation_function
		self.init_std = init_std
		self.layerdrop = layerdrop
		self.use_cache = use_cache
		self.do_layer_norm_before = do_layer_norm_before
		self.enable_bias = enable_bias
		self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
		self._remove_final_layer_norm = _remove_final_layer_norm
		self.from_pt = False

	def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
		"""
		Get the partition rules for the model.

		Args:
		    fully_sharded_data_parallel (`bool`, *optional*, defaults to `True`):
		        Whether to use fully sharded data parallelism.

		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		if not fully_sharded_data_parallel:
			raise NotImplementedError
		else:
			return (".*", PartitionSpec(("fsdp", "sp")))

	def add_jax_args(
		self,
		vocab_size: int = 50272,
		hidden_size: int = 768,
		num_hidden_layers: int = 12,
		ffn_dim: int = 3072,
		max_position_embeddings: int = 2048,
		do_layer_norm_before: bool = True,
		_remove_final_layer_norm: bool = False,
		word_embed_proj_dim: int = None,
		dropout: float = 0.1,
		attention_dropout: float = 0.0,
		num_attention_heads: int = 12,
		activation_function: str = "relu",
		layerdrop: float = 0.0,
		init_std: float = 0.02,
		use_cache: bool = True,
		pad_token_id: int = 1,
		bos_token_id: int = 2,
		eos_token_id: int = 2,
		enable_bias: bool = True,
		layer_norm_elementwise_affine: bool = True,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		basics = dict(
			vocab_size=vocab_size,
			hidden_size=hidden_size,
			num_hidden_layers=num_hidden_layers,
			ffn_dim=ffn_dim,
			max_position_embeddings=max_position_embeddings,
			do_layer_norm_before=do_layer_norm_before,
			_remove_final_layer_norm=_remove_final_layer_norm,
			word_embed_proj_dim=word_embed_proj_dim,
			dropout=dropout,
			attention_dropout=attention_dropout,
			num_attention_heads=num_attention_heads,
			activation_function=activation_function,
			layerdrop=layerdrop,
			init_std=init_std,
			use_cache=use_cache,
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			enable_bias=enable_bias,
			layer_norm_elementwise_affine=layer_norm_elementwise_affine,
			gradient_checkpointing=gradient_checkpointing,
			**kwargs,
		)
		for k, v in basics.items():
			if not hasattr(self, k):
				setattr(self, k, v)
		self.from_pt = False
