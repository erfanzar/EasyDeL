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

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("grok-1")
class Grok1Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32000):
	        Vocabulary size of the Grok-1 model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 32768):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 32):
	        Number of key and value heads for each attention layer in the Transformer encoder.
	    attn_output_multiplier (`float`, *optional*, defaults to 1.0):
	        The multiplier value applied to the attention output.
	    max_attn_value (`float`, *optional*, defaults to 1.0):
	        The maximum value of the attention weights.
	    max_position_embeddings (`int`, *optional*, defaults to 4096):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 2048 or 4096).
	    embedding_multiplier_scale (`float`, *optional*, defaults to 1.0):
	        The scale factor for the embedding layer.
	    output_multiplier_scale (`float`, *optional*, defaults to 1.0):
	        The scale factor for the output layer.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*):
	        The index of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 1):
	        The index of the beginning of sequence token in the vocabulary.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The index of the end of sequence token in the vocabulary.
	    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
	        Whether to tie the weights of the input embeddings and the output embeddings.
	    num_experts_per_tok (`int`, *optional*, defaults to 2):
	        The number of experts per token.
	    num_experts (`int`, *optional*, defaults to 8):
	        The number of experts.
	    output_router_logits (`bool`, *optional*, defaults to `False`):
	        Whether to output router logits.
	    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
	        The router auxiliary loss coefficient.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	"""

	model_type: str = "grok-1"

	def __init__(
		self,
		vocab_size=32000,
		hidden_size=4096,
		intermediate_size=32768,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=32,
		attn_output_multiplier=1.0,
		max_attn_value=1.0,
		max_position_embeddings=4096,
		embedding_multiplier_scale: float = 1.0,
		output_multiplier_scale: float = 1.0,
		rms_norm_eps=1e-5,
		use_cache=True,
		pad_token_id=None,
		bos_token_id=1,
		eos_token_id=2,
		tie_word_embeddings=True,
		num_experts_per_tok=2,
		num_experts=8,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: tp.Optional[int] = None,
		**kwargs,
	):
		"""Initializes a Grok1Config object.

		Args:
		    vocab_size (int, optional): Vocabulary size. Defaults to 32000.
		    hidden_size (int, optional): Hidden size. Defaults to 4096.
		    intermediate_size (int, optional): Intermediate size of the feed-forward network. Defaults to 32768.
		    num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
		    num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
		    num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 32.
		    attn_output_multiplier (float, optional): Multiplier for attention output. Defaults to 1.0.
		    max_attn_value (float, optional): Maximum attention value. Defaults to 1.0.
		    max_position_embeddings (int, optional): Maximum sequence length. Defaults to 4096.
		    embedding_multiplier_scale (float, optional): Scale factor for embeddings. Defaults to 1.0.
		    output_multiplier_scale (float, optional): Scale factor for the output layer. Defaults to 1.0.
		    rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-5.
		    use_cache (bool, optional): Whether to use KV cache. Defaults to True.
		    pad_token_id (int, optional): Padding token ID. Defaults to None.
		    bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 1.
		    eos_token_id (int, optional): End-of-sequence token ID. Defaults to 2.
		    tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to True.
		    num_experts_per_tok (int, optional): Number of experts to route per token. Defaults to 2.
		    num_experts (int, optional): Total number of experts. Defaults to 8.
		    output_router_logits (bool, optional): Whether to output router logits. Defaults to False.
		    router_aux_loss_coef (float, optional): Coefficient for router auxiliary loss. Defaults to 0.001.
		    gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
		        Defaults to EasyDeLGradientCheckPointers.NONE.
		    bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
		    **kwargs: Additional keyword arguments.
		"""
		self.vocab_size = vocab_size
		self.attn_output_multiplier = attn_output_multiplier
		self.max_attn_value = max_attn_value
		self.max_position_embeddings = max_position_embeddings
		self.embedding_multiplier_scale = embedding_multiplier_scale
		self.output_multiplier_scale = output_multiplier_scale
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads

		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache

		self.num_experts_per_tok = num_experts_per_tok
		self.num_experts = num_experts
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model. This method defines how the model's parameters are
		partitioned across devices for distributed training and inference.

		Args:
		    *args: Additional positional arguments (unused).
		    **kwargs: Additional keyword arguments (unused).

		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: A tuple of partition rules, where each rule is a tuple
		        containing a regex pattern for parameter names and the corresponding `PartitionSpec`.
		"""
		return (
			("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
			("attn/q_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("attn/k_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("attn/v_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("linear/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("linear_1/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("linear_v/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("gate/kernel", PartitionSpec(("fsdp", "sp"))),
			("post_attn_norm/kernel", PartitionSpec(None)),
			("pre_attn_norm/kernel", PartitionSpec(None)),
			("pre_moe_norm/kernel", PartitionSpec(None)),
			("post_moe_norm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)

	def attach_custom_arguments(
		self,
		tie_word_embeddings: bool = False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: tp.Optional[int] = None,
		**kwargs,
	):
		"""Attaches custom arguments to the configuration object.

		This method allows adding or overriding configuration attributes dynamically.
		It primarily sets attributes related to word embeddings, gradient checkpointing, and quantization bits.

		Args:
		    tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
		    gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
		        Defaults to EasyDeLGradientCheckPointers.NONE.
		    bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
		    **kwargs: Additional keyword arguments (ignored).
		"""
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

	@staticmethod
	def get_weight_decay_exclusions():
		"""Returns a tuple of parameter names for which weight decay should be excluded.

		Returns:
		    tuple: An empty tuple, indicating no specific weight decay exclusions for this model.
		"""
		return tuple()

	@staticmethod
	def rng_keys():
		"""Returns the names of the random number generator keys used by the model.

		Returns:
		    tuple: A tuple containing "params" and "dropout" as the RNG keys.
		"""
		return "params", "dropout"

	@property
	def granted_freq_max_position_embedding(self) -> int:
		"""Returns the maximum position embedding size specifically for frequency-based position embeddings.

		If `freq_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
		`max_position_embeddings`.

		Returns:
		    int: The granted maximum position embedding size for frequency encoding.
		"""
		return getattr(
			self,
			"freq_max_position_embeddings",
			self.max_position_embeddings,
		)

	@property
	def granted_mask_max_position_embedding(self) -> int:
		"""Returns the maximum position embedding size specifically for mask-based position embeddings.

		If `mask_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
		`max_position_embeddings`.

		Returns:
		    int: The granted maximum position embedding size for mask encoding.
		"""
		return getattr(
			self,
			"mask_max_position_embeddings",
			self.max_position_embeddings,
		)
