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


from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("MiniMaxText01")
class MiniMaxText01Config(EasyDeLBaseConfig):
	r"""
	This is the configuration class to store the configuration of a [`MiniMaxText01Model`]. It is used to instantiate an
	MiniMaxText01 model according to the specified arguments, defining the model architecture. Instantiating a configuration
	with the defaults will yield a similar configuration to that of the MiniMaxText01.
	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
	documentation from [`PretrainedConfig`] for more information.
	Args:
	    vocab_size (`int`, *optional*, defaults to 32000):
	        Vocabulary size of the MiniMaxText01 model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed when calling [`MiniMaxText01Model`]
	    hidden_size (`int`, *optional*, defaults to 4096):
	        Dimension of the hidden representations.
	    intermediate_size (`int`, *optional*, defaults to 14336):
	        Dimension of the MLP representations.
	    num_hidden_layers (`int`, *optional*, defaults to 32):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 32):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 8):
	        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
	        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
	        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
	        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
	        by meanpooling all the original heads within that group. For more details checkout [this
	        paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) in the decoder.
	    max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
	        The maximum sequence length that this model might ever be used with. MiniMaxText01's sliding window attention
	        allows sequence of up to 4096*32 tokens.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-05):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    pad_token_id (`int`, *optional*):
	        The id of the padding token.
	    bos_token_id (`int`, *optional*, defaults to 1):
	        The id of the "beginning-of-sequence" token.
	    eos_token_id (`int`, *optional*, defaults to 2):
	        The id of the "end-of-sequence" token.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether the model's input and output word embeddings should be tied.
	    rope_theta (`float`, *optional*, defaults to 1000000.0):
	        The base period of the RoPE embeddings.
	    sliding_window (`int`, *optional*):
	        Sliding window attention window size. If not specified, will default to `4096`.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    num_experts_per_tok (`int`, *optional*, defaults to 2):
	        The number of experts to route per-token, can be also interpreted as the `top-k` routing
	        parameter
	    num_local_experts (`int`, *optional*, defaults to 8):
	        Number of experts per Sparse MLP layer.
	    output_router_logits (`bool`, *optional*, defaults to `False`):
	        Whether or not the router logits should be returned by the model. Enabeling this will also
	        allow the model to output the auxiliary loss. See [here]() for more details
	    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
	        The aux loss factor for the total loss.
	    router_jitter_noise (`float`, *optional*, defaults to 0.0):
	        Amount of noise to add to the router.
	```python
	>>> from transformers import MiniMaxText01Model, MiniMaxText01Config
	>>> # Initializing a MiniMaxText01 style configuration
	>>> configuration = MiniMaxText01Config()
	>>> # Initializing a model from the MiniMaxText01 style configuration
	>>> model = MiniMaxText01Model(configuration)
	>>> # Accessing the model configuration
	>>> configuration = model.config
	```"""

	model_type = "MiniMaxText01"
	keys_to_ignore_at_inference = ["past_key_values"]

	def __init__(
		self,
		vocab_size=32000,
		hidden_size=4096,
		intermediate_size=14336,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=8,
		hidden_act="silu",
		max_position_embeddings=4096 * 32,
		initializer_range=0.02,
		rms_norm_eps=1e-5,
		use_cache=True,
		pad_token_id=None,
		bos_token_id=None,
		eos_token_id=None,
		tie_word_embeddings=False,
		rope_theta=1e6,
		sliding_window=None,
		attention_dropout=0.0,
		num_experts_per_tok=2,
		num_local_experts=8,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		router_jitter_noise=0.0,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.sliding_window = sliding_window

		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.attention_dropout = attention_dropout

		self.num_experts_per_tok = num_experts_per_tok
		self.num_local_experts = num_local_experts
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef
		self.router_jitter_noise = router_jitter_noise
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for distributing the MiniMaxText01 model parameters across multiple devices.

		These rules define how parameters should be partitioned when using techniques like
		Fully Sharded Data Parallelism (FSDP), Sharded Parallelism (SP), and Tensor Parallelism (TP).
		Each rule consists of a regex pattern matching parameter names and a corresponding PartitionSpec.

		Returns:
		    tuple: A tuple of tuples where each inner tuple contains:
		        - A regex pattern matching parameter names
		        - A PartitionSpec object specifying how to partition matching parameters
		"""
		from jax.sharding import PartitionSpec

		return (
			# Embeddings
			("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
			# Attention layers
			("model/layers/.*/self_attn/q_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("model/layers/.*/self_attn/k_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("model/layers/.*/self_attn/v_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("model/layers/.*/self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			# MoE layers
			(
				"model/layers/.*/block_sparse_moe/experts/.*/w[13]/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"model/layers/.*/block_sparse_moe/experts/.*/w[24]/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			(
				"model/layers/.*/block_sparse_moe/gate/kernel",
				PartitionSpec(("fsdp", "sp"), None),
			),
			# Normalization
			("model/norm/kernel", PartitionSpec(None)),
			("model/layers/.*/input_layernorm/kernel", PartitionSpec(None)),
			("model/layers/.*/post_attention_layernorm/kernel", PartitionSpec(None)),
			# LM head
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			# Catch-all
			(".*", PartitionSpec(None)),
		)
