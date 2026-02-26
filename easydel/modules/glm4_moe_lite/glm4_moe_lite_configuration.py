# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing
import typing as tp

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


def _rope_scaling_from_rope_parameters(
    rope_parameters: dict[str, typing.Any] | None,
    rope_scaling: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    if rope_scaling is not None:
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]
        return rope_scaling

    if rope_parameters is None:
        return None

    rope_scaling_out: dict[str, typing.Any] = {
        "rope_type": rope_parameters.get("rope_type", "default"),
    }
    for key in (
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
        "short_factor",
        "long_factor",
        "beta_fast",
        "beta_slow",
        "extrapolation_factor",
        "attn_factor",
        "mscale",
        "mscale_all_dim",
    ):
        if key in rope_parameters:
            rope_scaling_out[key] = rope_parameters[key]
    return rope_scaling_out


@register_config("glm4_moe_lite")
class Glm4MoeLiteConfig(EasyDeLBaseConfig):
    r"""
    Configuration class for GLM-4-MoE-Lite models.

    This configuration mirrors the GLM-4 MoE Lite architecture with MLA attention,
    grouped top-k MoE routing, and a dense-to-sparse MLP schedule.

    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10240):
            Dimension of the dense MLP intermediate representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of routed expert MLPs.
        num_hidden_layers (`int`, *optional*, defaults to 47):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 20):
            Number of key/value heads (GQA).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.8):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank for KV compression in MLA.
        q_lora_rank (`int`, *optional*, defaults to 768):
            Rank for Q compression in MLA.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            RoPE head dimension for queries/keys.
        v_head_dim (`int`, *optional*, defaults to 256):
            Value head dimension.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Non-RoPE head dimension for queries/keys.
        n_group (`int`, *optional*, defaults to 1):
            Number of expert groups.
        topk_group (`int`, *optional*, defaults to 1):
            Number of expert groups to select per token.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of routed experts per token.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize routed expert weights.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Weight initialization range.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMS normalization epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV caching.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Pretraining tensor parallel factor.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input embeddings with the LM head.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for rotary embeddings.
        rope_parameters (`dict`, *optional*):
            RoPE configuration dictionary (HF-compatible).
        rope_scaling (`dict`, *optional*):
            EasyDeL rope scaling configuration.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to use interleaved RoPE layout.
        mlp_layer_types (`list[str]`, *optional*):
            List of "dense"/"sparse" specifying MLP type per layer.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
    """

    model_type: str = "glm4_moe_lite"
    attribute_map: tp.ClassVar = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_shared_experts: int = 1,
        n_routed_experts: int | None = 64,
        routed_scaling_factor: float = 1.8,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 768,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int | None = 4,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_parameters: dict[str, typing.Any] | None = None,
        rope_scaling: dict[str, typing.Any] | None = None,
        rope_interleave: bool = True,
        mlp_layer_types: list[str] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = _rope_scaling_from_rope_parameters(rope_parameters, rope_scaling)

        if rope_theta is None and rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.rope_theta = rope_theta if rope_theta is not None else 10000.0

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types must have length {self.num_hidden_layers}, got {len(self.mlp_layer_types)}."
            )
        for layer_type in self.mlp_layer_types:
            if layer_type not in ("dense", "sparse"):
                raise ValueError(f"Invalid layer type {layer_type}. Expected 'dense' or 'sparse'.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding.

        Providing explicit partition rules is preferred over automatic sharding resolution,
        as it gives full control over parameter distribution across the device mesh.
        Returns ``None`` by default, which triggers automatic sharding via
        module-level ``craft_sharding`` hooks.

        Returns:
            Partition rules as ``tuple[tuple[str, PartitionSpec], ...] | None``.
        """
        return None


__all__ = ["Glm4MoeLiteConfig"]
