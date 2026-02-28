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
    """Convert ``rope_parameters`` or ``rope_scaling`` dict into a normalised rope-scaling dict.

    HuggingFace checkpoints may store the rope configuration under either
    ``rope_scaling`` (older) or ``rope_parameters`` (newer).  This helper
    normalises both representations into a single ``rope_scaling`` dict that
    the rest of the code can consume.

    Args:
        rope_parameters: Newer-style rope configuration dict (may be ``None``).
        rope_scaling: Legacy rope scaling dict (takes precedence if provided).

    Returns:
        A normalised rope-scaling dict, or ``None`` when neither input is set.
    """
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


@register_config("glm_moe_dsa")
class GlmMoeDsaConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the GLM-MoE-DSA model.
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimensionality of the dense MLP intermediate layer.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the MoE expert intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 78):
            Number of decoder layers in the transformer.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for Multi-head Latent Attention.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            Number of key-value heads (typically equal to ``num_attention_heads`` for MLA).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts that always process every token.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Total number of routed experts in MoE layers.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor applied to routed expert weights after normalisation.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the low-rank KV compression in MLA.
        q_lora_rank (`int`, *optional*, defaults to 2048):
            Rank of the low-rank query decomposition. Set to ``None`` to disable.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the query/key RoPE subspace.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimensionality of the query/key non-RoPE subspace.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimensionality of each value head.
        n_group (`int`, *optional*, defaults to 1):
            Number of expert groups for grouped top-k routing.
        topk_group (`int`, *optional*, defaults to 1):
            Number of top groups to activate per token.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of routed experts activated per token.
        norm_topk_prob (`bool`, *optional*, defaults to ``True``):
            Whether to normalise top-k routing probabilities.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation function used in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialisation.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalisation layers.
        use_cache (`bool`, *optional*, defaults to ``True``):
            Whether to return past key/values for caching.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period for rotary position embeddings.
        rope_interleave (`bool`, *optional*, defaults to ``False``):
            Whether to use interleaved RoPE layout for the main attention.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top-k tokens selected by the dynamic sparse attention indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Head dimension used in the sparse attention indexer.
        index_n_heads (`int`, *optional*, defaults to 32):
            Number of heads in the sparse attention indexer (auto-calculated as
            ``num_attention_heads // 2`` when ``None``).
        indexer_rope_interleave (`bool`, *optional*, defaults to ``False``):
            Whether to use interleaved RoPE layout for the indexer.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer MLP type schedule (``"dense"`` or ``"sparse"``). Defaults to the first
            3 layers dense and the remainder sparse.
        attention_bias (`bool`, *optional*, defaults to ``False``):
            Whether to use bias in attention projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention weights.
    """

    model_type: str = "glm_moe_dsa"
    attribute_map: tp.ClassVar = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: int = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 2048,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 192,
        v_head_dim: int = 256,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int | None = 8,
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
        rope_interleave: bool = False,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int | None = 32,
        indexer_rope_interleave: bool = False,
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
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_interleave = rope_interleave
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads if index_n_heads is not None else max(1, num_attention_heads // 2)
        self.indexer_rope_interleave = indexer_rope_interleave
        self.rope_scaling = _rope_scaling_from_rope_parameters(rope_parameters, rope_scaling)

        if rope_theta is None and rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.rope_theta = rope_theta if rope_theta is not None else 10000.0

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            dense_layers = min(3, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (self.num_hidden_layers - dense_layers)

        if self.n_routed_experts is not None and self.n_group > 0 and self.n_routed_experts % self.n_group != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})."
            )

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
        return None


__all__ = ["GlmMoeDsaConfig"]
