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

"""Configuration classes for the MiniMax M3 VL (``minimax_m3_vl``) model family.

MiniMax M3 VL is a self-contained vision-language family:

- :class:`MiniMaxM3VLTextConfig` (``minimax_m3_vl_text``) — a MiniMax-M2-style
  full-attention MoE decoder with per-head Gemma-style ``(1 + weight)`` RMSNorm
  on Q/K, partial RoPE (``rope_parameters.partial_rotary_factor``), a
  sigmoid-scored bias-corrected top-k router, clamped SwiGLU-OAI experts, a
  shared expert per MoE layer, per-layer dense/sparse MLP scheduling
  (``mlp_layer_types``), and optional block-sparse "Lightning Indexer"
  attention on layers whose ``layer_types`` entry is ``"minimax_m3_sparse"``.
- :class:`MiniMaxM3VLVisionConfig` (``minimax_m3_vl_vision``) — a CLIP-style
  tower (biased QKV, LayerNorm, GELU MLP) with a Conv3d patch embedding and a
  3D (T/H/W) rotary position embedding applied to Q/K.
- :class:`MiniMaxM3VLConfig` (``minimax_m3_vl``) — the composite config wiring
  both towers plus the two-stage GELU patch-merge projector.
"""

import typing
from collections.abc import Mapping

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

logger = get_logger(__name__)

# HF ``MiniMaxM2Config.default_theta`` (inherited by the M3 text config).
DEFAULT_TEXT_ROPE_THETA = 5_000_000.0
# HF ``MiniMaxM3VLVisionConfig.default_theta``.
DEFAULT_VISION_ROPE_THETA = 10_000.0

_VALID_ATTENTION_LAYER_TYPES = ("full_attention", "minimax_m3_sparse")
_VALID_MLP_LAYER_TYPES = ("dense", "sparse")


@register_config("minimax_m3_vl_text")
class MiniMaxM3VLTextConfig(EasyDeLBaseConfig):
    """Configuration for the MiniMax M3 VL text decoder.

    The decoder pairs MiniMax-M2-style grouped-query attention (biasless
    projections, per-head Gemma-style ``(1 + weight)`` RMSNorm on queries and
    keys before partial RoPE) with a mixed dense/sparse MLP schedule. Sparse
    layers run a sigmoid top-k router (selection on
    ``sigmoid(logits) + e_score_correction_bias``, weights from the raw
    sigmoid scores normalized and scaled by ``routed_scaling_factor``) over
    stacked clamped SwiGLU-OAI experts, plus one always-on shared expert.
    Layers whose ``layer_types`` entry is ``"minimax_m3_sparse"`` additionally
    run a block-sparse Lightning Indexer that restricts attention to the
    top-scoring key blocks.

    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the text model.
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the hidden layers.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Intermediate width of each routed expert.
        dense_intermediate_size (`int`, *optional*, defaults to 12288):
            Intermediate width of the dense MLP on ``"dense"`` layers.
        shared_intermediate_size (`int`, *optional*, defaults to 3072):
            Intermediate width of the shared expert in MoE layers.
        num_hidden_layers (`int`, *optional*, defaults to 60):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Per-head channel dimension (explicit; not derived).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Pointwise fallback activation key. The checkpoint declares
            ``"swigluoai"`` but the gate is computed inline from
            ``swiglu_alpha`` / ``swiglu_limit``; HF normalizes this to
            ``"silu"`` and so does EasyDeL.
        max_position_embeddings (`int`, *optional*, defaults to 524288):
            Maximum supported sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Stddev for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon of the Gemma-style RMSNorm layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values during generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            Rotary base frequency (HF ``default_theta``).
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration.
        partial_rotary_factor (`float`, *optional*, defaults to 1.0):
            Fraction of ``head_dim`` rotated by RoPE. Mirrors HF, where the
            rotated width comes exclusively from
            ``rope_parameters["partial_rotary_factor"]`` (the ``rotary_dim``
            field is metadata only).
        rotary_dim (`int`, *optional*, defaults to 64):
            Stored metadata for the number of rotated head channels. Not used
            at runtime (see ``partial_rotary_factor``), kept for checkpoint
            fidelity.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether QKV/output projections carry bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate on attention weights.
        num_local_experts (`int`, *optional*, defaults to 128):
            Number of routed experts per MoE layer.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of experts activated per token.
        routed_scaling_factor (`float`, *optional*, defaults to 2.0):
            Multiplier applied to the routed expert combination.
        swiglu_alpha (`float`, *optional*, defaults to 1.702):
            Sigmoid gain of the SwiGLU-OAI activation.
        swiglu_limit (`float`, *optional*, defaults to 7.0):
            Clamp bound on the gate/up projections of SwiGLU-OAI.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer MLP selector, ``"sparse"`` (MoE) or ``"dense"``.
            Defaults to all ``"sparse"``.
        layer_types (`list[str]`, *optional*):
            Per-layer attention dispatch, ``"full_attention"`` or
            ``"minimax_m3_sparse"``. Defaults to all ``"full_attention"``.
        index_n_heads (`int`, *optional*, defaults to 4):
            Number of Lightning-Indexer scoring heads (one per KV group).
        index_head_dim (`int`, *optional*, defaults to 128):
            Per-head channel dimension of the indexer.
        index_block_size (`int`, *optional*, defaults to 128):
            Number of key tokens pooled into one scored block.
        index_topk_blocks (`int`, *optional*, defaults to 16):
            Number of top-scoring key blocks each query attends to.
        index_local_blocks (`int`, *optional*, defaults to 1):
            Number of key blocks immediately preceding the query that are
            always kept visible.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output MoE router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Auxiliary router loss coefficient (HF M2 default).
    """

    model_type = "minimax_m3_vl_text"
    attribute_map: typing.ClassVar = {"num_experts": "num_local_experts"}

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 6144,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 12288,
        shared_intermediate_size: int = 3072,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = DEFAULT_TEXT_ROPE_THETA,
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 1.0,
        rotary_dim: int = 64,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        routed_scaling_factor: float = 2.0,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        mlp_layer_types: list[str] | None = None,
        layer_types: list[str] | None = None,
        index_n_heads: int = 4,
        index_head_dim: int = 128,
        index_block_size: int = 128,
        index_topk_blocks: int = 16,
        index_local_blocks: int = 1,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ):
        """Initialize MiniMaxM3VLTextConfig with model architecture hyperparameters.

        See the class docstring for detailed parameter descriptions.

        Raises:
            ValueError: If ``layer_types`` / ``mlp_layer_types`` have the wrong
                length or contain unknown entries.
        """
        # HF v5 serializes the rotary base (and partial factor) inside
        # ``rope_parameters``; honor it when present.
        rope_parameters = kwargs.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", partial_rotary_factor)
            if rope_scaling is None and rope_parameters.get("rope_type", "default") != "default":
                rope_scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
        # Legacy MiniMax checkpoint layout: ``sparse_attention_config`` +
        # ``moe_layer_freq`` describe the per-layer schedules.
        sparse_cfg = kwargs.pop("sparse_attention_config", None) or {}
        moe_layer_freq = kwargs.pop("moe_layer_freq", None)
        legacy_index_map = {
            "index_n_heads": "sparse_num_index_heads",
            "index_head_dim": "sparse_index_dim",
            "index_block_size": "sparse_block_size",
            "index_topk_blocks": "sparse_topk_blocks",
            "index_local_blocks": "sparse_local_block",
        }
        legacy_index_values = {flat: sparse_cfg[legacy] for flat, legacy in legacy_index_map.items() if legacy in sparse_cfg}
        index_n_heads = legacy_index_values.get("index_n_heads", index_n_heads)
        index_head_dim = legacy_index_values.get("index_head_dim", index_head_dim)
        index_block_size = legacy_index_values.get("index_block_size", index_block_size)
        index_topk_blocks = legacy_index_values.get("index_topk_blocks", index_topk_blocks)
        index_local_blocks = legacy_index_values.get("index_local_blocks", index_local_blocks)
        if layer_types is None and "sparse_attention_freq" in sparse_cfg:
            layer_types = ["minimax_m3_sparse" if f else "full_attention" for f in sparse_cfg["sparse_attention_freq"]]
        if mlp_layer_types is None and moe_layer_freq is not None:
            mlp_layer_types = ["sparse" if f else "dense" for f in moe_layer_freq]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        # The checkpoint declares ``"swigluoai"``; the gate is computed inline
        # from ``swiglu_alpha`` / ``swiglu_limit`` and ``hidden_act`` is only
        # the pointwise fallback — normalize it like HF does.
        self.hidden_act = "silu" if hidden_act == "swigluoai" else hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = rotary_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.index_local_blocks = index_local_blocks
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"`layer_types` must have exactly `num_hidden_layers` ({num_hidden_layers}) entries, "
                f"got {len(layer_types)}."
            )
        invalid_attn = sorted({t for t in layer_types if t not in _VALID_ATTENTION_LAYER_TYPES})
        if invalid_attn:
            raise ValueError(
                f"`layer_types` entries must be one of {_VALID_ATTENTION_LAYER_TYPES}, got {invalid_attn}."
            )
        self.layer_types = list(layer_types)

        if mlp_layer_types is None:
            mlp_layer_types = ["sparse"] * num_hidden_layers
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError(
                f"`mlp_layer_types` must have exactly `num_hidden_layers` ({num_hidden_layers}) entries, "
                f"got {len(mlp_layer_types)}."
            )
        invalid_mlp = sorted({t for t in mlp_layer_types if t not in _VALID_MLP_LAYER_TYPES})
        if invalid_mlp:
            raise ValueError(f"`mlp_layer_types` entries must be 'dense' or 'sparse', got {invalid_mlp}.")
        self.mlp_layer_types = list(mlp_layer_types)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # HF's rotary embedding reads the partial factor exclusively from
        # ``rope_parameters``; keep the duck-typed surface in sync so parity
        # tests that feed this config to the HF class see the same rotation.
        if self.partial_rotary_factor != 1.0 and isinstance(getattr(self, "rope_parameters", None), dict):
            self.rope_parameters = {**self.rope_parameters, "partial_rotary_factor": self.partial_rotary_factor}


@register_config("minimax_m3_vl_vision")
class MiniMaxM3VLVisionConfig(EasyDeLBaseConfig):
    """Configuration for the MiniMax M3 VL vision tower.

    A CLIP-style encoder (pre-LN blocks with biased QKV projections and GELU
    MLPs) fed by a bias-free Conv3d patch embedding, with a 3D rotary position
    embedding: ``2 * (head_dim // 2)`` rotary channels are split evenly across
    the temporal/height/width axes and rotated by each patch's grid position.
    The tower has no final LayerNorm; its raw last hidden state feeds the
    multimodal projector.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Width of the encoder MLP.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        image_size (`int`, *optional*, defaults to 2016):
            Nominal input image size.
        patch_size (`int`, *optional*, defaults to 14):
            Spatial patch size.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Temporal patch size for video frames.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the patch-merge block used by the projector.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            MLP activation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon of the LayerNorm layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate on attention weights.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency of the 3D rotary embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Stddev for weight initialization.
    """

    model_type = "minimax_m3_vl_vision"

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 2016,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        rope_theta: float = DEFAULT_VISION_ROPE_THETA,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize MiniMaxM3VLVisionConfig.

        See the class docstring for detailed parameter descriptions.
        """
        rope_parameters = kwargs.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


@register_config("minimax_m3_vl")
class MiniMaxM3VLConfig(EasyDeLBaseConfig):
    """Composite configuration for the MiniMax M3 VL vision-language model.

    Wires the :class:`MiniMaxM3VLVisionConfig` tower and the
    :class:`MiniMaxM3VLTextConfig` decoder together with the two-stage GELU
    patch-merge projector (``projector_hidden_size`` wide; the merge stage
    consumes ``merged_hidden_size = text_hidden * spatial_merge_size**2``
    channels).

    Args:
        vision_config (`dict` or `MiniMaxM3VLVisionConfig`, *optional*):
            Vision tower configuration.
        text_config (`dict` or `MiniMaxM3VLTextConfig`, *optional*):
            Text decoder configuration.
        image_token_index (`int`, *optional*, defaults to 200025):
            Placeholder token id for image patches.
        video_token_index (`int`, *optional*, defaults to 200026):
            Placeholder token id for video patches.
        projector_hidden_size (`int`, *optional*, defaults to 6144):
            Hidden width of both projector MLP stages.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input embeddings with the LM head.
    """

    model_type = "minimax_m3_vl"
    sub_configs: typing.ClassVar = {
        "vision_config": MiniMaxM3VLVisionConfig,
        "text_config": MiniMaxM3VLTextConfig,
    }
    attribute_map: typing.ClassVar = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vision_config: Mapping[str, typing.Any] | MiniMaxM3VLVisionConfig | None = None,
        text_config: Mapping[str, typing.Any] | MiniMaxM3VLTextConfig | None = None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        projector_hidden_size: int = 6144,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the composite MiniMax M3 VL configuration.

        Args:
            vision_config: Mapping or :class:`MiniMaxM3VLVisionConfig` for the
                vision tower; ``None`` materializes the defaults.
            text_config: Mapping or :class:`MiniMaxM3VLTextConfig` for the
                text decoder; ``None`` materializes the defaults.
            image_token_index: Placeholder token id for image patches.
            video_token_index: Placeholder token id for video patches.
            projector_hidden_size: Hidden width of the projector MLP stages.
            tie_word_embeddings: Whether to tie input embeddings with the LM
                head.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projector_hidden_size = projector_hidden_size
        if not tie_word_embeddings and getattr(self.text_config, "tie_word_embeddings", False):
            tie_word_embeddings = self.text_config.tie_word_embeddings
        # Channel dim after grouping ``spatial_merge_size**2`` projected
        # patches, consumed by the projector's patch-merge MLP.
        self.merged_hidden_size = self.text_config.hidden_size * (self.vision_config.spatial_merge_size**2)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> MiniMaxM3VLTextConfig:
        """Return the text decoder configuration.

        Args:
            decoder (bool, optional): Unused; kept for API compatibility with
                upstream Hugging Face configs. Defaults to True.

        Returns:
            MiniMaxM3VLTextConfig: The text sub-config.
        """
        del decoder
        return self.text_config  # pyright: ignore[reportReturnType]

    def get_vision_config(self) -> MiniMaxM3VLVisionConfig:
        """Return the vision tower configuration.

        Returns:
            MiniMaxM3VLVisionConfig: The vision sub-config.
        """
        return self.vision_config  # type: ignore


__all__ = ["MiniMaxM3VLConfig", "MiniMaxM3VLTextConfig", "MiniMaxM3VLVisionConfig"]
