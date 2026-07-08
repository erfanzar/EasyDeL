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

"""Configuration class for the Tencent Hunyuan V3 (``hy_v3``) model family.

Defines :class:`HyV3Config`, the EasyDeL configuration object for HYV3.
The architecture is a decoder-only transformer combining Qwen3-style
QK-normalized grouped-query attention (per-head RMSNorm on Q/K before
RoPE, full-dim rotary with a large default base frequency) with a
DeepSeek-style sigmoid MoE: per-layer dense/sparse MLP scheduling
(``mlp_layer_types``), a bias-corrected sigmoid router with an extra
``router_scaling_factor``, one always-on shared expert, and optional
fp32 combination of routed and shared outputs.
"""

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

logger = get_logger(__name__)

# HF ``HYV3Config.default_theta`` — HYV3 uses a much larger rotary base
# frequency than the Llama (1e4) / Qwen (1e6) families. Do not "fix" this.
DEFAULT_ROPE_THETA = 11_158_840.0


@register_config("hy_v3")
class HyV3Config(EasyDeLBaseConfig):
    """Configuration for the Tencent Hunyuan V3 (HYV3) decoder architecture.

    HYV3 layers use full causal attention with per-head RMSNorm on queries
    and keys (applied before RoPE, exactly like Qwen3). The MLP of each layer
    is scheduled per-layer via ``mlp_layer_types``: ``"dense"`` layers use a
    SwiGLU MLP of width ``intermediate_size``, while ``"sparse"`` layers use
    a sigmoid-routed mixture of ``num_experts`` experts (top
    ``num_experts_per_tok`` selected on sigmoid scores plus a per-expert
    ``e_score_correction_bias``, weighted by the raw sigmoid scores,
    normalized, and scaled by ``router_scaling_factor``) plus one shared
    SwiGLU expert of width ``moe_intermediate_size * num_shared_experts``.

    Args:
        vocab_size (`int`, *optional*, defaults to 120832):
            Vocabulary size of the HYV3 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden layers.
        intermediate_size (`int`, *optional*, defaults to 13312):
            Dimensionality of the dense MLP intermediate layer (``"dense"`` layers).
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of transformer decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for grouped query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head (explicit; not derived from
            ``hidden_size // num_attention_heads``).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length this model supports.
        initializer_range (`float`, *optional*, defaults to 0.006):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values for caching during generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output word embeddings.
        rope_theta (`float`, *optional*, defaults to 11158840.0):
            Base frequency for rotary position embeddings (HF ``default_theta``).
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in QKV and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP projection layers.
        num_experts (`int`, *optional*, defaults to 192):
            Total number of routed experts in each sparse MoE layer.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts activated per token.
        num_shared_experts (`int`, *optional*, defaults to 1):
            Multiplier on ``moe_intermediate_size`` for the always-on shared expert.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of each routed expert's MLP intermediate layer.
        router_scaling_factor (`float`, *optional*, defaults to 2.826):
            Scaling factor applied to the normalized top-k routing weights.
        enable_moe_fp32_combine (`bool`, *optional*, defaults to `True`):
            Whether to add routed and shared expert outputs in float32 before
            casting back to the activation dtype.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer MLP type, ``"dense"`` or ``"sparse"``. Defaults to
            ``["dense"] + ["sparse"] * (num_hidden_layers - 1)``.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output MoE router logits.
    """

    model_type = "hy_v3"

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = DEFAULT_ROPE_THETA,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        num_experts: int = 192,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        moe_intermediate_size: int = 1536,
        router_scaling_factor: float = 2.826,
        enable_moe_fp32_combine: bool = True,
        mlp_layer_types: list[str] | None = None,
        output_router_logits: bool = False,
        **kwargs,
    ):
        """Initialize HyV3Config with model architecture hyperparameters.

        See class docstring for detailed parameter descriptions.

        Raises:
            ValueError: If ``mlp_layer_types`` has the wrong length or
                contains values other than ``"dense"`` / ``"sparse"``.
        """
        # HF v5 configs serialize the rotary base inside ``rope_parameters``
        # instead of a top-level ``rope_theta`` key; honor it when present.
        rope_parameters = kwargs.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            if rope_scaling is None and rope_parameters.get("rope_type", "default") != "default":
                rope_scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.router_scaling_factor = router_scaling_factor
        self.enable_moe_fp32_combine = enable_moe_fp32_combine
        self.output_router_logits = output_router_logits
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * (1 if num_hidden_layers > 0 else 0) + ["sparse"] * max(
                num_hidden_layers - 1, 0
            )
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError(
                f"`mlp_layer_types` must have exactly `num_hidden_layers` ({num_hidden_layers}) entries, "
                f"got {len(mlp_layer_types)}."
            )
        invalid_types = sorted({t for t in mlp_layer_types if t not in ("dense", "sparse")})
        if invalid_types:
            raise ValueError(f"`mlp_layer_types` entries must be 'dense' or 'sparse', got {invalid_types}.")
        self.mlp_layer_types = list(mlp_layer_types)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HyV3Config"]
