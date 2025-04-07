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

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


@register_config("qwen3")
class Qwen3Config(EasyDeLBaseConfig):
	model_type = "qwen3"

	def __init__(
		self,
		vocab_size=151936,
		hidden_size=4096,
		intermediate_size=22016,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=32,
		head_dim=128,
		hidden_act="silu",
		max_position_embeddings=32768,
		initializer_range=0.02,
		rms_norm_eps=1e-6,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=10000.0,
		rope_scaling=None,
		attention_bias=False,
		use_sliding_window=False,
		sliding_window=4096,
		max_window_layers=28,
		attention_dropout=0.0,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.use_sliding_window = use_sliding_window
		self.sliding_window = (
			sliding_window  # we check `use_sliding_window` in the modeling code
		)
		self.max_window_layers = max_window_layers

		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.head_dim = head_dim
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout
		# Validate the correctness of rotary position embeddings parameters
		# BC: if there is a 'type' field, move it to 'rope_type'.
		if self.rope_scaling is not None and "type" in self.rope_scaling:
			self.rope_scaling["rope_type"] = self.rope_scaling["type"]

		super().__init__(
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.

		Args:
		    fully_sharded_data_parallel (`bool`, *optional*, defaults to `True`):
		        Whether to use fully sharded data parallelism.

		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
			("self_attn/q_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/k_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/v_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("input_layernorm/kernel", PartitionSpec(None)),
			("post_attention_layernorm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(None)),
		)


__all__ = ["Qwen3Config"]
