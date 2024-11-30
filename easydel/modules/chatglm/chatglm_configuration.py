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

from typing import Dict, Optional, Union

from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.modules.factory import register_config
from easydel.modules.modeling_utils import EasyDeLBaseConfig


@register_config("glm")
class ChatGLMConfig(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    num_layers (`int`, *optional*, defaults to 28):
	        Number of hidden layers in the Transformer encoder.
	    padded_vocab_size (`int`, *optional*, defaults to 65024):
	        Vocabulary size of the ChatGLM model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed to the forward method.
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimensionality of the encoder layers and the pooler layer.
	    ffn_hidden_size (`int`, *optional*, defaults to 13696):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    kv_channels (`int`, *optional*, defaults to 128):
	        Dimensionality of the key and value projection layers in the attention layer.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    seq_length (`int`, *optional*, defaults to 2048):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 512 or 1024 or 2048).
	    hidden_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
	    classifier_dropout (`float`, *optional*):
	        The dropout ratio for classifier.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
	        The epsilon used by the layer normalization layers.
	    rmsnorm (`bool`, *optional*, defaults to `True`):
	        Whether to use RMS norm instead of layer norm.
	    apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
	        Whether to apply residual connection after layer normalization.
	    post_layer_norm (`bool`, *optional*, defaults to `True`):
	        Whether to use post-layernorm architecture.
	    add_bias_linear (`bool`, *optional*, defaults to `False`):
	        Whether to add bias to the linear layers.
	    add_qkv_bias (`bool`, *optional*, defaults to `False`):
	        Whether to add bias to the query, key, and value projection layers in the attention layer.
	    bias_dropout_fusion (`bool`, *optional*, defaults to `True`):
	        Whether to use bias dropout fusion.
	    multi_query_attention (`bool`, *optional*, defaults to `False`):
	        Whether to use multi-query attention.
	    multi_query_group_num (`int`, *optional*, defaults to 1):
	        The number of groups in multi-query attention.
	    rope_ratio (`float`, *optional*, defaults to 1.0):
	        The ratio of the rotary position embedding.
	    apply_query_key_layer_scaling (`bool`, *optional*, defaults to `True`):
	        Whether to apply query key layer scaling.
	    attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
	        Whether to compute attention softmax in float32.
	    fp32_residual_connection (`bool`, *optional*, defaults to `False`):
	        Whether to compute residual connection in float32.
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        The gradient checkpointing configuration.
	    rope_scaling (`Dict[str, Union[str, float]]`, *optional*):
	        The rope scaling configuration.
	    scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
	        The chunk size of the scanned MLP.
	    bits (`int`, *optional*):
	        The number of bits to quantize the model to.
	    mlp_bias (`bool`, *optional*, defaults to `False`):
	        Whether the MLP layers should have bias.
	    scan_layers (`bool`, *optional*, defaults to `False`):
	        Whether to use the scan implementation of the layers.
	"""

	model_type: str = "chatglm"

	def __init__(
		self,
		num_layers=28,
		padded_vocab_size=65024,
		hidden_size=4096,
		ffn_hidden_size=13696,
		kv_channels=128,
		num_attention_heads=32,
		seq_length=2048,
		hidden_dropout=0.0,
		classifier_dropout=None,
		attention_dropout=0.0,
		layernorm_epsilon=1e-5,
		rmsnorm=True,
		apply_residual_connection_post_layernorm=False,
		post_layer_norm=True,
		add_bias_linear=False,
		add_qkv_bias=False,
		bias_dropout_fusion=True,
		multi_query_attention=False,
		multi_query_group_num=1,
		rope_ratio=1,
		apply_query_key_layer_scaling=True,
		attention_softmax_in_fp32=True,
		fp32_residual_connection=False,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		rope_scaling: Dict[str, Union[str, float]] = None,
		scan_mlp_chunk_size: int = 1024,
		bits: Optional[int] = None,
		mlp_bias: bool = False,
		scan_layers: bool = False,
		**kwargs,
	):
		self.num_layers = num_layers
		self.vocab_size = padded_vocab_size
		self.padded_vocab_size = padded_vocab_size
		self.hidden_size = hidden_size
		self.ffn_hidden_size = ffn_hidden_size
		self.kv_channels = kv_channels
		self.num_attention_heads = num_attention_heads
		self.seq_length = seq_length
		self.hidden_dropout = hidden_dropout
		self.classifier_dropout = classifier_dropout
		self.attention_dropout = attention_dropout
		self.layernorm_epsilon = layernorm_epsilon
		self.rmsnorm = rmsnorm
		self.apply_residual_connection_post_layernorm = (
			apply_residual_connection_post_layernorm
		)
		self.post_layer_norm = post_layer_norm
		self.add_bias_linear = add_bias_linear
		self.add_qkv_bias = add_qkv_bias
		self.bias_dropout_fusion = bias_dropout_fusion
		self.multi_query_attention = multi_query_attention
		self.multi_query_group_num = multi_query_group_num
		self.rope_ratio = rope_ratio
		self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
		self.attention_softmax_in_fp32 = attention_softmax_in_fp32
		self.fp32_residual_connection = fp32_residual_connection
		self.gradient_checkpointing = gradient_checkpointing
		self.mlp_bias = mlp_bias
		self.rope_scaling = rope_scaling
		self.bits = bits
		self.scan_layers = scan_layers
		super().__init__(
			scan_mlp_chunk_size=scan_mlp_chunk_size,
			bits=bits,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `Tuple[Tuple[str, PartitionSpec]]`: The partition rules.
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
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		bits: Optional[int] = None,
		scan_layers: bool = False,
		**kwargs,
	):
		self.scan_layers = scan_layers
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

	@staticmethod
	def get_weight_decay_exclusions():
		return tuple()

	@staticmethod
	def rng_keys():
		return "params", "dropout", "fcm"
