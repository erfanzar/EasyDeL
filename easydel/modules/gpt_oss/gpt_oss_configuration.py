# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""GPT-OSS Model Configuration

This module provides configuration classes for the GPT-OSS model,
a transformer-based language model with Mixture of Experts (MoE) architecture. The model
features sparse routing, sliding window attention, and efficient parameter sharding for
distributed training.

The configuration includes custom sharding specifications for MoE components and
comprehensive model hyperparameters.
"""

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers.moe.utils import get_moe_partition_spec


@register_config("gpt_oss")
class GptOssConfig(EasyDeLBaseConfig):
    """Configuration class for GPT-OSS model.

    GPT-OSS is a transformer-based language model featuring:
    - Mixture of Experts (MoE) architecture with sparse routing
    - Alternating sliding window and full attention layers
    - RMSNorm for layer normalization
    - Rotary Position Embeddings (RoPE) with optional scaling
    - Efficient parameter sharding for distributed training

    Attributes:
        num_hidden_layers (int): Number of transformer layers. Default: 36
        num_local_experts (int): Number of expert networks per MoE layer. Default: 128
        vocab_size (int): Size of the vocabulary. Default: 201088
        hidden_size (int): Dimension of hidden representations. Default: 2880
        intermediate_size (int): Dimension of MLP intermediate layer. Default: 2880
        head_dim (int): Dimension of each attention head. Default: 64
        num_attention_heads (int): Number of attention heads. Default: 64
        num_key_value_heads (int): Number of key-value heads for GQA. Default: 8
        sliding_window (int): Size of sliding window for local attention. Default: 128
        rope_theta (float): Base frequency for RoPE. Default: 150000.0
        tie_word_embeddings (bool): Whether to tie input/output embeddings. Default: False
        hidden_act (str): Activation function for MLP. Default: "silu"
        initializer_range (float): Standard deviation for weight initialization. Default: 0.02
        max_position_embeddings (int): Maximum sequence length. Default: 131072
        rms_norm_eps (float): Epsilon for RMS normalization. Default: 1e-5
        rope_scaling (dict): Configuration for RoPE scaling. Default: YARN scaling with factor 32
        attention_dropout (float): Dropout rate for attention weights. Default: 0.0
        num_experts_per_tok (int): Number of experts to route each token to. Default: 4
        router_aux_loss_coef (float): Coefficient for load balancing auxiliary loss. Default: 0.9
        output_router_logits (bool): Whether to output router logits. Default: False
        use_cache (bool): Whether to use key-value caching. Default: True
        layer_types (list): Attention type for each layer. Default: alternating sliding/full

    Example:
        >>> config = GptOssConfig(
        ...     num_hidden_layers=24,
        ...     num_local_experts=64,
        ...     hidden_size=2048,
        ...     num_attention_heads=32
        ... )
        >>> model = GptOssForCausalLM(config)
    """

    model_type = "gpt_oss"

    def __init__(
        self,
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings=False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings=131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling=None,
        attention_dropout: float = 0.0,
        num_experts_per_tok=4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits=False,
        use_cache=True,
        layer_types=None,
        mlp_activations_limit: float = 7.0,
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False}
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_local_experts = num_local_experts
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            # Default: alternating sliding window and full attention
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        self.mlp_activations_limit = mlp_activations_limit
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_partition_rules(self, *args, **kwargs):
        """Get the partition rules for distributed training of GPT-OSS model.

        Returns partition specifications for different parameter groups to enable
        efficient model parallelism. The rules specify how to shard parameters
        across devices for:
        - Embeddings: Column-wise sharding
        - Attention: Column-wise for QKV, row-wise for output projection
        - MoE: Custom expert-parallel sharding for expert parameters
        - Normalization: Replicated across devices

        Returns:
            tuple: Partition rules as (regex_pattern, PartitionSpec) pairs
        """
        pmag = self.partition_manager

        kws = dict(
            fsdp_is_ep_bound=self.fsdp_is_ep_bound,
            sp_is_ep_bound=self.sp_is_ep_bound,
            module_view=True,
            tensors_are_expert=self.use_expert_tensor_mode,
            partition_manager=self.partition_manager,
        )

        eck = get_moe_partition_spec(direction="column", is_bias=False, **kws)
        erk = get_moe_partition_spec(direction="row", is_bias=False, **kws)

        ecb = get_moe_partition_spec(direction="column", is_bias=True, **kws)
        erb = get_moe_partition_spec(direction="row", is_bias=True, **kws)

        return (
            (r".*embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r".*self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r".*self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r".*self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*mlp/gate/kernel",
                pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise),
            ),
            (r".*mlp/gate/bias", pmag.resolve(Replicated)),
            # Legacy paths (original HuggingFace parameter names)
            (r".*mlp/experts/gate_up_proj$", eck),
            (r".*mlp/experts/down_proj$", erk),
            (r".*mlp/experts/gate_up_proj_bias$", ecb),
            (r".*mlp/experts/down_proj_bias$", erb),
            # New split paths (after reform_param transformation)
            (r".*mlp/experts/(gate_proj|up_proj)/kernel", eck),
            (r".*mlp/experts/down_proj/kernel", erk),
            (r".*mlp/experts/(gate_proj|up_proj)/bias", ecb),
            (r".*mlp/experts/down_proj/bias", erb),
            (r".*layernorm/scale", pmag.resolve(Replicated)),
            (r".*rms_norm/scale", pmag.resolve(Replicated)),
            (r".*norm/scale", pmag.resolve(Replicated)),
            (r".*lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*score/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


__all__ = ["GptOssConfig"]
