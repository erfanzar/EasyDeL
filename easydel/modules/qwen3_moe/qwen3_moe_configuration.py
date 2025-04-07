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


@register_config("qwen3_moe")
class Qwen3MoeConfig(EasyDeLBaseConfig):
	model_type = "qwen3_moe"

	def __init__(
		self,
		vocab_size=151936,
		hidden_size=2048,
		intermediate_size=6144,
		num_hidden_layers=24,
		num_attention_heads=32,
		num_key_value_heads=4,
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
		decoder_sparse_step=1,
		moe_intermediate_size=768,
		num_experts_per_tok=8,
		num_experts=128,
		norm_topk_prob=False,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		mlp_only_layers=None,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.use_sliding_window = use_sliding_window
		self.sliding_window = sliding_window if use_sliding_window else None
		self.max_window_layers = max_window_layers

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout
		if self.rope_scaling is not None and "type" in self.rope_scaling:
			self.rope_scaling["rope_type"] = self.rope_scaling["type"]

		self.decoder_sparse_step = decoder_sparse_step
		self.moe_intermediate_size = moe_intermediate_size
		self.num_experts_per_tok = num_experts_per_tok
		self.num_experts = num_experts
		self.norm_topk_prob = norm_topk_prob
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef
		self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

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
			("self_attn/o_proj/kernel", PartitionSpec(("sp", "fsdp"), "tp")),
			("gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("shared_expert_gate/kernel", PartitionSpec(("fsdp", "sp"))),
			("gate/kernel", PartitionSpec(("fsdp", "sp"))),
			("input_layernorm/kernel", PartitionSpec(None)),
			("post_attention_layernorm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(".*", PartitionSpec(("fsdp", "sp"))),
		)


__all__ = ["Qwen3MoeConfig"]
