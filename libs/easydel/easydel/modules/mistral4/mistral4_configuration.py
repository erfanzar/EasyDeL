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

"""Configuration for Mistral4.

Mistral4 is a DeepSeek-V3-shaped architecture from Mistral AI: Multi-head
Latent Attention (optional low-rank query projection, compressed KV latents
with a shared single-head RoPE key) combined with a DeepseekV3-style MoE
block (stacked routed experts plus one shared expert, grouped top-k
routing). It differs from DeepSeek-V3 in three ways:

- routing scores are a plain softmax over the router logits (no sigmoid, no
  ``e_score_correction_bias``);
- queries are scaled by the Llama-4 attention temperature
  ``1 + beta * log(1 + floor(position / original_max_position_embeddings))``
  with ``beta = rope_parameters["llama_4_scaling_beta"]``;
- the attention softmax scale is a plain ``qk_head_dim ** -0.5`` (no YaRN
  mscale adjustment).

This module exposes :class:`Mistral4Config`, registered for the HF model
type ``"mistral4"``.
"""

import typing

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers import RopeConfig


def _default_rope_parameters(qk_rope_head_dim: int, qk_head_dim: int, max_position_embeddings: int) -> dict:
    """Build the default Mistral4 RoPE parameter dict.

    Mirrors ``transformers.Mistral4Config.__post_init__``: a YaRN spec with
    base 10000, factor 128, an 8192-token pre-extension window, unit mscale
    values, the Llama-4 attention-temperature beta, and a partial rotary
    factor covering only the ``qk_rope_head_dim`` slice of the head.

    Args:
        qk_rope_head_dim: Width of the RoPE-rotated query/key slice.
        qk_head_dim: Full query/key head width (``nope + rope``).
        max_position_embeddings: Deployed maximum context length.

    Returns:
        dict: HF v5-style ``rope_parameters`` mapping.
    """
    return {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "max_position_embeddings": max_position_embeddings,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale_all_dim": 1.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": 0.1,
        "partial_rotary_factor": qk_rope_head_dim / qk_head_dim,
    }


def _normalize_rope_dict(
    rope_parameters: dict[str, typing.Any] | None,
    rope_scaling: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    """Merge ``rope_parameters`` / ``rope_scaling`` into one normalized dict.

    HF checkpoints newer than v5 store the RoPE spec under
    ``rope_parameters``; older EasyDeL-flavoured configs use ``rope_scaling``
    (which takes precedence when both are present). Unlike the whitelist
    helpers used by other families, this keeps *every* key so
    Mistral4-specific entries (``llama_4_scaling_beta``,
    ``partial_rotary_factor``) survive the round trip.

    Args:
        rope_parameters: Newer-style RoPE dict (may be ``None``).
        rope_scaling: Legacy RoPE dict (takes precedence when set).

    Returns:
        dict | None: Normalized copy with both ``type`` and ``rope_type``
        keys populated, or ``None`` when neither input was provided.
    """
    source = rope_scaling if rope_scaling is not None else rope_parameters
    if source is None:
        return None
    normalized = dict(source)
    if "type" in normalized and "rope_type" not in normalized:
        normalized["rope_type"] = normalized["type"]
    normalized.setdefault("rope_type", "default")
    return normalized


@register_config("mistral4")
class Mistral4Config(EasyDeLBaseConfig):
    r"""
    Configuration class for Mistral4 models. Instantiating a configuration
    with the defaults yields a configuration close to
    ``mistralai/Mistral-Small-4-119B-2603``.

    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used
    to control the model outputs. Read the documentation from
    [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the Mistral4 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimensionality of the dense MLP intermediate layer.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of each routed expert's intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of decoder layers in the transformer.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for Multi-head Latent Attention.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads (equal to ``num_attention_heads`` for MLA).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts that always process every token.
        n_routed_experts (`int`, *optional*, defaults to 128):
            Total number of routed experts in MoE layers.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to routed expert weights after normalisation.
        kv_lora_rank (`int`, *optional*, defaults to 256):
            Rank of the low-rank KV compression in MLA.
        q_lora_rank (`int`, *optional*, defaults to 1024):
            Rank of the low-rank query decomposition. ``None`` uses a direct
            ``q_proj``.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the query/key RoPE subspace.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each value head.
        qk_nope_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the query/key non-RoPE subspace.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of top groups kept per token before per-group top-k.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of routed experts activated per token.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers
            (embed->dense->dense->...->dense->moe->moe...->lm_head).
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalise top-k routing probabilities.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialisation.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalisation layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values for caching.
        pad_token_id (`int`, *optional*, defaults to 11):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Pretraining tensor-parallel factor (kept for checkpoint
            compatibility).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input embeddings with the LM head.
        rope_theta (`float`, *optional*):
            RoPE base frequency. Falls back to
            ``rope_parameters["rope_theta"]`` and finally ``10000.0``.
        rope_parameters (`dict`, *optional*):
            HF v5-style RoPE parameter dict. When neither this nor
            ``rope_scaling`` is given, the Mistral4 YaRN defaults are used
            (theta 10000, factor 128, original max 8192, unit mscales,
            ``llama_4_scaling_beta=0.1``).
        rope_scaling (`dict`, *optional*):
            Legacy-style RoPE scaling dict (takes precedence over
            ``rope_parameters`` when both are given).
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether the RoPE-rotated query/key channels use the interleaved
            (GPT-J even/odd) pairing instead of NeoX half-split pairing.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the ``q_a`` / ``kv_a`` / ``o`` projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention probabilities.
        layer_types (`list[str]`, *optional*):
            Per-layer attention types; defaults to ``"full_attention"`` for
            every layer.
    """

    model_type = "mistral4"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]
    attribute_map: typing.ClassVar = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 128,
        routed_scaling_factor: float = 1.0,
        kv_lora_rank: int = 256,
        q_lora_rank: int | None = 1024,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 64,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int | None = 4,
        first_k_dense_replace: int = 0,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = 11,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_parameters: dict[str, typing.Any] | None = None,
        rope_scaling: dict[str, typing.Any] | None = None,
        rope_interleave: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a new Mistral4Config instance.

        Args:
            vocab_size: Token vocabulary size.
            hidden_size: Residual-stream dimension.
            intermediate_size: Inner width of the dense MLP.
            moe_intermediate_size: Inner width of each routed expert FFN.
            num_hidden_layers: Total decoder layers.
            num_attention_heads: Total query heads for MLA.
            num_key_value_heads: KV heads (defaults to ``num_attention_heads``).
            n_shared_experts: Shared experts always applied to every token.
            n_routed_experts: Total routed experts in MoE layers.
            routed_scaling_factor: Multiplier on routed expert weights.
            kv_lora_rank: Rank of the MLA KV down-projection.
            q_lora_rank: Rank of the MLA Q down-projection (``None``
                disables the low-rank Q step).
            qk_rope_head_dim: RoPE-rotated portion of the Q/K head dim.
            v_head_dim: Value head dim.
            qk_nope_head_dim: Unrotated portion of the Q/K head dim.
            n_group: Number of expert groups for grouped routing.
            topk_group: Groups kept per token before per-group top-k.
            num_experts_per_tok: Routed experts selected per token.
            first_k_dense_replace: Leading layers that use a dense MLP.
            norm_topk_prob: Whether to renormalise top-k routing weights.
            hidden_act: Activation applied to the gate half of MLPs.
            max_position_embeddings: Context-length bound used by RoPE.
            initializer_range: Stddev for normal weight init.
            rms_norm_eps: Epsilon for every RMSNorm.
            use_cache: Whether downstream code should return KV cache.
            pad_token_id: Padding token id.
            bos_token_id: Beginning-of-stream token id.
            eos_token_id: End-of-stream token id.
            pretraining_tp: Pretraining tensor-parallel factor.
            tie_word_embeddings: Tie input embeddings with the LM head.
            rope_theta: RoPE base frequency override.
            rope_parameters: HF v5-style RoPE parameter dict.
            rope_scaling: Legacy RoPE scaling dict (precedence over
                ``rope_parameters``).
            rope_interleave: Interleaved (GPT-J) vs NeoX RoPE pairing.
            attention_bias: Whether attention projections carry biases.
            attention_dropout: Dropout on the attention probabilities.
            layer_types: Per-layer attention type schedule.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If ``n_routed_experts`` is not divisible by
                ``n_group``.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
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
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        rope_dict = _normalize_rope_dict(rope_parameters, rope_scaling)
        if rope_dict is None:
            rope_dict = _default_rope_parameters(
                qk_rope_head_dim=qk_rope_head_dim,
                qk_head_dim=self.qk_head_dim,
                max_position_embeddings=max_position_embeddings,
            )
        rope_dict.setdefault("partial_rotary_factor", qk_rope_head_dim / self.qk_head_dim)
        if rope_theta is None:
            rope_theta = rope_dict.get("rope_theta", 10000.0)
        rope_dict.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope_theta
        # `rope_scaling` feeds EasyDeL's rotary builders; `rope_parameters`
        # is what HF v5 modeling code reads. Keep both in sync.
        self.rope_scaling = dict(rope_dict)
        self.rope_parameters = dict(rope_dict)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if self.n_routed_experts is not None and self.n_group > 0 and self.n_routed_experts % self.n_group != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _get_rope_config(self) -> RopeConfig:
        """Build the RoPE configuration for Mistral4.

        Reads ``self.rope_scaling`` (always populated by ``__init__``,
        falling back to the Mistral4 YaRN defaults when the checkpoint did
        not specify RoPE parameters) and backfills
        ``original_max_position_embeddings`` from the legacy attribute when
        absent.

        Returns:
            RopeConfig: Resolved RoPE configuration.
        """
        rope_scaling = getattr(self, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = _default_rope_parameters(
                qk_rope_head_dim=self.qk_rope_head_dim,
                qk_head_dim=self.qk_head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
        config = RopeConfig.from_dict(rope_scaling)
        if config.original_max_position_embeddings is None:
            config.original_max_position_embeddings = getattr(self, "original_max_position_embeddings", None)
        return config


__all__ = ["Mistral4Config"]
