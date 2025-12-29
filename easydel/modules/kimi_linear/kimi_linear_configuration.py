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

"""
Kimi Linear configuration class for EasyDeL.

This module provides the configuration class for Kimi Linear models (moonshotai),
which combine MLA (Multi-Latent Attention) with KDA (Kernel Delta Attention)
linear attention layers and MoE (Mixture of Experts).

References:
    - Kimi Linear: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
"""

import typing
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

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers.caching.hybrid import FULL_ATTENTION, KDA_LINEAR_ATTENTION
from easydel.layers.moe.utils import get_moe_partition_spec
from easydel.layers.rotary_embedding import RopeConfig

KIMI_LINEAR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class KimiExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes for Kimi Linear."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


@register_config("kimi_linear")
class KimiLinearConfig(EasyDeLBaseConfig):
    r"""
    Configuration class for Kimi Linear models.

    Kimi Linear combines:
    - MLA (Multi-Latent Attention) for full attention layers
    - KDA (Kernel Delta Attention) for linear attention layers
    - MoE (Mixture of Experts) with shared experts

    Args:
        vocab_size (`int`, *optional*, defaults to 163840):
            Vocabulary size of the Kimi Linear model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the dense MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*):
            Number of key_value heads for GQA. Defaults to `num_attention_heads`.
        head_dim (`int`, *optional*):
            Dimension of each attention head. Defaults to `hidden_size // num_attention_heads`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            RoPE scaling configuration.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.

        MLA Parameters:
        q_lora_rank (`int`, *optional*):
            Rank for query LoRA projection in MLA.
        kv_lora_rank (`int`, *optional*):
            Rank for KV LoRA projection in MLA.
        qk_nope_head_dim (`int`, *optional*):
            Non-positional embedding dimension for Q/K in MLA.
        qk_rope_head_dim (`int`, *optional*):
            Positional embedding dimension for Q/K in MLA.
        v_head_dim (`int`, *optional*):
            Value head dimension in MLA.
        mla_use_nope (`bool`, *optional*, defaults to `False`):
            Whether to use nope (non-positional) embeddings in MLA.

        MoE Parameters:
        moe_intermediate_size (`int`, *optional*):
            Intermediate dimension for MoE experts.
        moe_renormalize (`bool`, *optional*, defaults to `True`):
            Whether to renormalize expert weights.
        moe_router_activation_func (`str`, *optional*, defaults to `"sigmoid"`):
            Router activation function ("sigmoid" or "softmax").
        num_experts (`int`, *optional*):
            Number of routed experts.
        num_experts_per_token (`int`, *optional*):
            Number of experts selected per token.
        num_shared_experts (`int`, *optional*, defaults to 0):
            Number of shared experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed expert outputs.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of initial dense layers before MoE layers.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers (every Nth layer uses MoE).
        use_grouped_topk (`bool`, *optional*, defaults to `True`):
            Whether to use grouped top-k selection.
        num_expert_group (`int`, *optional*, defaults to 1):
            Number of expert groups.
        topk_group (`int`, *optional*, defaults to 1):
            Number of groups to select in top-k.

        Linear Attention Parameters:
        num_nextn_predict_layers (`int`, *optional*, defaults to 0):
            Number of next-n prediction layers.
        linear_attn_config (`dict`, *optional*):
            Configuration for linear attention with keys:
            - `kda_layers`: List of layer indices using KDA (1-indexed)
            - `full_attn_layers`: List of layer indices using MLA (1-indexed)
            - Other KDA-specific parameters
    """

    model_type = "kimi_linear"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        mla_use_nope: bool | None = True,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        max_position_embeddings: int = 2**16,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.max_position_embeddings = max_position_embeddings
        if linear_attn_config is not None:
            assert linear_attn_config.get("kda_layers") is not None
            assert linear_attn_config.get("full_attn_layers") is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self) -> bool:
        """Check if model uses Multi-Latent Attention."""
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self) -> bool:
        """Check if model uses Mixture of Experts."""
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        """Check if model uses linear attention (KDA)."""
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config.get("kda_layers") is not None
                and len(self.linear_attn_config.get("kda_layers", [])) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses KDA linear attention (1-indexed in config).

        Args:
            layer_idx: Layer index (0-indexed).

        Returns:
            True if layer uses KDA, False otherwise.
        """
        return self.linear_attn_config is not None and (layer_idx + 1) in self.linear_attn_config.get("kda_layers", [])

    def is_mla_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses MLA full attention (1-indexed in config).

        Args:
            layer_idx: Layer index (0-indexed).

        Returns:
            True if layer uses MLA, False otherwise.
        """
        return self.linear_attn_config is not None and (layer_idx + 1) in self.linear_attn_config.get(
            "full_attn_layers", []
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses MoE.

        Args:
            layer_idx: Layer index (0-indexed).

        Returns:
            True if layer uses MoE, False otherwise.
        """
        return (
            self.num_experts is not None
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def get_layer_types(self) -> tuple[str, ...]:
        """Get layer types tuple for HybridCache initialization.

        Returns:
            Tuple of layer type strings ("full_attention" or "kda_linear_attention").
        """
        layer_types = []
        for i in range(self.num_hidden_layers):
            if self.is_kda_layer(i):
                layer_types.append(KDA_LINEAR_ATTENTION)
            else:
                layer_types.append(FULL_ATTENTION)
        return tuple(layer_types)

    @property
    def q_head_dim(self) -> int:
        """Get query head dimension for MLA."""
        if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.head_dim

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for distributed training.

        Returns:
            Tuple of partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_a_proj|q_b_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/(kv_a_proj_with_mqa|kv_b_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/(q_a_layernorm|kv_a_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"self_attn/(q_conv1d|k_conv1d|v_conv1d)/kernel", pmag.resolve(Replicated)),
            (r"self_attn/(f_a_proj|f_b_proj|b_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/(g_a_proj|g_b_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/(A_log|dt_bias)", pmag.resolve(Replicated)),
            (r"self_attn/o_norm/kernel", pmag.resolve(Replicated)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/gate/kernel", pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise)),
            (r"mlp/gate/e_score_correction_bias", pmag.resolve(Replicated)),
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
            (r"mlp/shared_experts/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/shared_experts/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/shared_experts/.*proj/bias", pmag.resolve(Replicated)),
            (r".*/(input_layernorm|post_attention_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"norm/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def get_rope_config(self) -> RopeConfig:
        """Get RoPE configuration.

        Returns:
            RopeConfig instance.
        """
        return RopeConfig(
            rope_type="default",
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
        )
