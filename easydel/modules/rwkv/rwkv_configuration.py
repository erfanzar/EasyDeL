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


@register_config("rwkv")
class RwkvConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 50277):
	        Vocabulary size of the RWKV model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed when calling [`~easydel.modules.RwkvModel`].
	    context_length (`int`, *optional*, defaults to 1024):
	        The maximum sequence length that this model might ever be used with.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    attention_hidden_size (`int`, *optional*):
	        Dimensionality of the query/key/value of the MultiHead Attention layer of the RWKV* model. If None, it is
	        set to `hidden_size`.
	    intermediate_size (`int`, *optional*):
	        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder. If None,
	        it is set to `4 * hidden_size`.
	    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    rescale_every (`int`, *optional*, defaults to 6):
	        Interval of layers at which to rescale the attention scores.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models).
	    bos_token_id (`int`, *optional*, defaults to 0):
	        The id for the beginning of stream token.
	    eos_token_id (`int`, *optional*, defaults to 0):
	        The id for the end of stream token.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to. If None, the model is not quantized.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        What to save during gradient checkpointing. Choose one of `"nothing_saveable"`, `"first_half_saveable"`,
	        `"full_saveable"`.
	"""

	model_type: str = "rwkv"
	attribute_map = {"max_position_embeddings": "context_length"}

	def __init__(
		self,
		vocab_size=50277,
		context_length=1024,
		hidden_size=4096,
		num_hidden_layers=32,
		attention_hidden_size=None,
		intermediate_size=None,
		layer_norm_epsilon=1e-5,
		bos_token_id=0,
		eos_token_id=0,
		rescale_every=6,
		tie_word_embeddings=False,
		use_cache=True,
		bits: tp.Optional[int] = None,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	) -> None:
		self.bits = bits
		self.gradient_checkpointing = gradient_checkpointing
		self.vocab_size = vocab_size
		self.context_length = context_length
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.attention_hidden_size = (
			attention_hidden_size if attention_hidden_size is not None else hidden_size
		)
		self.intermediate_size = (
			intermediate_size if intermediate_size is not None else 4 * hidden_size
		)
		self.layer_norm_epsilon = layer_norm_epsilon
		self.rescale_every = rescale_every
		self.use_cache = use_cache

		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id

		super().__init__(
			tie_word_embeddings=tie_word_embeddings,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			bits=bits,
			**kwargs,
		)

	def add_jax_args(
		self,
		bits: tp.Optional[int] = None,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		self.bits = bits
		self.gradient_checkpointing = gradient_checkpointing
		for k, v in kwargs.items():
			if not hasattr(self, k):
				setattr(self, k, v)

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
			((".*", PartitionSpec(("sp", "fsdp"))),)
			if fully_sharded_data_parallel
			else ((".*", PartitionSpec(("sp", "fsdp"))),)
		)
