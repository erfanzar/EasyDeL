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

"""Configuration for the Qwen3Next hybrid attention model."""

import typing as tp

from eformer.common_types import (
    EMPTY,
    MODE_TRAIN,
    TP,
    ColumnWise,
    DynamicShardingAxes,
    Replicated,
    RowWise,
)
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers.moe.utils import get_moe_partition_spec

logger = get_logger(__name__)


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


@register_config("qwen3_next")
class Qwen3NextConfig(EasyDeLBaseConfig):
    """Configuration for Qwen3Next - a hybrid attention model with GatedDeltaRule.

    Qwen3Next alternates between full attention layers (standard MHA with sigmoid gating
    and partial RoPE) and linear attention layers (GatedDeltaNet with causal convolution).
    The model uses MoE FFN layers with both routed and shared experts.

    Architecture:
        - Full Attention: Standard MHA with sigmoid output gating, per-head RMSNorm,
          and partial RoPE (25% of head_dim by default)
        - Linear Attention: GatedDeltaNet with causal 1D convolution and gated delta
          rule recurrence
        - MoE FFN: Multiple routed experts with optional shared expert

    Attributes:
        vocab_size: Vocabulary size of the tokenizer.
        hidden_size: Hidden representation dimension.
        intermediate_size: Dimension of the dense MLP layers.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads for full attention layers.
        num_key_value_heads: Number of KV heads for grouped query attention.
        head_dim: Dimension per attention head.
        hidden_act: Activation function type.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether to use KV cache for decoding.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base for rotary position embeddings.
        rope_scaling: Configuration for RoPE scaling.
        attention_bias: Whether to use bias in attention projections.
        attention_dropout: Dropout probability for attention weights.
        partial_rotary_factor: Fraction of head_dim to apply RoPE (default 0.25).
        layer_types: List specifying attention type per layer.
        full_attention_interval: Interval for full attention layers.
        linear_conv_kernel_dim: Kernel size for linear attention convolution.
        linear_key_head_dim: Key head dimension for linear attention.
        linear_value_head_dim: Value head dimension for linear attention.
        linear_num_key_heads: Number of key heads for linear attention.
        linear_num_value_heads: Number of value heads for linear attention.
        decoder_sparse_step: Step interval for MoE layers.
        moe_intermediate_size: Intermediate size for routed experts.
        shared_expert_intermediate_size: Intermediate size for shared expert.
        num_experts_per_tok: Number of experts selected per token.
        num_experts: Total number of routed experts.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        output_router_logits: Whether to output router logits.
        router_aux_loss_coef: Coefficient for auxiliary routing loss.
        mlp_only_layers: Layer indices that use dense MLP instead of MoE.
    """

    model_type = "qwen3_next"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.25,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 10,
        num_experts: int = 512,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.partial_rotary_factor = partial_rotary_factor

        self.full_attention_interval = full_attention_interval
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def rotary_dim(self) -> int:
        """Return the dimension used for rotary embeddings (partial RoPE)."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_d_inner(self) -> int:
        """Return the inner dimension for linear attention convolution state.

        This is the conv_dim used in Qwen3NextLinearAttention:
        conv_dim = key_dim * 2 + value_dim
        where key_dim = num_k_heads * head_k_dim and value_dim = num_v_heads * head_v_dim
        """
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        return key_dim * 2 + value_dim

    @property
    def linear_d_state(self) -> int:
        """Return the state dimension for linear attention recurrence."""
        return self.linear_value_head_dim

    def get_partition_rules(self, *args, **kwargs):
        """Get the partition rules for the model.

        Returns:
            Tuple of partition rules mapping regex patterns to partition specs.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"full_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"full_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"full_attn/q_gate_proj/kernel", pmag.resolve(ColumnWise)),
            (r"full_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"full_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"linear_attn/in_proj/kernel", pmag.resolve(ColumnWise)),
            (r"linear_attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"linear_attn/out_norm/kernel", pmag.resolve(Replicated)),
            (r"linear_attn/conv1d/kernel", pmag.resolve(Replicated)),
            (r"linear_attn/beta_proj/kernel", pmag.resolve(ColumnWise)),
            (r"linear_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r"mlp/gate/kernel",
                pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise),
            ),
            (r"mlp/gate/bias", pmag.resolve(Replicated)),
            (
                r"mlp/experts/(gate_proj|up_proj)/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="column",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (
                r"mlp/experts/down_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="row",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (r"mlp/experts/.*bias", pmag.resolve(Replicated)),
            (r"shared_expert/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"shared_expert/down_proj/kernel", pmag.resolve(RowWise)),
            (r"shared_expert/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm)/kernel",
                pmag.resolve(Replicated),
            ),
            (r"norm/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses full attention.

        Args:
            layer_idx: Layer index.

        Returns:
            True if the layer uses full attention, False for linear attention.
        """
        return self.layer_types[layer_idx] == "full_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses MoE FFN.

        Args:
            layer_idx: Layer index.

        Returns:
            True if the layer uses MoE, False for dense MLP.
        """
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0


__all__ = ["Qwen3NextConfig"]
