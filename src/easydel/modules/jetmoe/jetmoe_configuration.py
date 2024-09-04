
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

from easydel.modules.modeling_utils import EDPretrainedConfig


class JetMoEConfig(EDPretrainedConfig):
	model_type: str = "jetmoe"

	def __init__(
		self,
		vocab_size=32000,
		hidden_size=2048,
		num_hidden_layers=12,
		num_attention_heads=32,
		num_key_value_heads=16,
		kv_channels=128,
		ffn_hidden_size=5632,
		max_position_embeddings=4096,
		activation_function="silu",
		glu=True,
		moe_num_experts=8,
		moe_top_k=2,
		use_cache=True,
		bos_token_id=1,
		eos_token_id=2,
		tie_word_embeddings=True,
		bias=True,
		rope_theta=10000.0,
		rms_norm_eps=1e-6,
		initializer_range=0.01,
		gradient_checkpointing: str = "nothing_saveable",
		bits: Optional[int] = None,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.ffn_hidden_size = ffn_hidden_size
		self.kv_channels = kv_channels
		self.bias = bias
		self.glu = glu
		self.moe_num_experts = moe_num_experts
		self.moe_top_k = moe_top_k
		self.activation_function = activation_function
		self.rope_theta = rope_theta
		self.initializer_range = initializer_range
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache

		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits
		super().__init__(
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
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
		return ((".*", PartitionSpec(("fsdp", "sp"))),)

	def add_jax_args(
		self,
		tie_word_embeddings: bool = False,
		gradient_checkpointing: str = "nothing_saveable",
		bits: Optional[int] = None,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the Transformer class:

		Args:
		    self: Refer to the current object
		    tie_word_embeddings: bool: Tie the word embeddings to the
		        decoder
		    gradient_checkpointing: str: Control the amount of memory
		        used by jax
		    bits: Optional[int]: Determine the number of bits used in
		        the quantization
		"""
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

	@staticmethod
	def get_weight_decay_exclusions():
		return tuple()

	@staticmethod
	def rng_keys():
		return "params", "dropout"
