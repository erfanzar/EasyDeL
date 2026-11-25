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

"""Configuration classes for the Inkling 2 (``inkling2`` and ``inkling2_loop``) model families.

This module defines the five configuration classes covering Thinking Machines'
Inkling 2 vision-language Mixture-of-Experts architectures:

- :class:`Inkling2TextConfig` (``inkling2_text``) — flat decoder stack with
  alternating sliding-window (``"hybrid_sliding"``) and full (``"hybrid"``)
  attention layers, hidden-state conditioned relative position bias, L2-clipped
  Q/K/relative states, logarithmic query/relative scaling on full layers,
  per-token attention sinks, sigmoid-gated attention output projections, four
  causal short convolutions per layer (dynamic K/V taps on full layers),
  block attention residuals, softcapped SwiGLU MoE experts, and an unpadded
  softcapped language-modeling head.
- :class:`Inkling2VisionConfig` (``inkling2_vision``) — ViT encoder with linear
  patch projection, bilinearly interpolated 2-D learned position embeddings,
  per-image CLS tokens, pre-LayerNorm, axial 2-D RoPE, LayerScale, GELU MLPs,
  and a 2x2 spatial-merge projector.
- :class:`Inkling2Config` (``inkling2``) — composite vision-language config
  combining :class:`Inkling2TextConfig` and :class:`Inkling2VisionConfig`.
- :class:`Inkling2LoopTextConfig` (``inkling2_loop_text``) — looped variant of
  :class:`Inkling2TextConfig` (e.g. Inkling-2-Mini) that repeats layers
  ``[loop_layer_index_start, loop_layer_index_end)`` for ``loop_num_passes``
  passes with cross-pass message attention residuals and low-rank message gates.
- :class:`Inkling2LoopConfig` (``inkling2_loop``) — composite vision-language
  config combining :class:`Inkling2LoopTextConfig` and
  :class:`Inkling2VisionConfig`.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

DEFAULT_VISION_ROPE_THETA: tp.Final[float] = 10_000.0
DEFAULT_VOCAB_SIZE: tp.Final[int] = 201_024
DEFAULT_UNPADDED_VOCAB_SIZE: tp.Final[int] = 200_058

_LAYER_TYPE_ALIASES: tp.Final[dict[str, str]] = {
    "hybrid": "hybrid",
    "hybrid_sliding": "hybrid_sliding",
    "full_attention": "hybrid",
    "sliding_attention": "hybrid_sliding",
}


def _normalize_layer_types(
    layer_types: tp.Sequence[str] | None,
    num_hidden_layers: int,
    n_layers_per_attn_res_block: int,
    local_layer_ids: tp.Sequence[int] | None = None,
) -> list[str]:
    """Resolve and validate per-layer attention types for an Inkling 2 decoder.

    Accepts either an explicit ``layer_types`` list (canonical HF strings
    ``"hybrid"`` / ``"hybrid_sliding"`` or EasyDeL aliases
    ``"full_attention"`` / ``"sliding_attention"``) or Thinking Machines'
    ``local_layer_ids`` index list from Hub ``config.json`` files.

    Args:
        layer_types: Explicit per-layer type sequence, or ``None``.
        num_hidden_layers: Number of physical decoder layers.
        n_layers_per_attn_res_block: Block size for default full-attention
            placement when neither ``layer_types`` nor ``local_layer_ids`` is
            given (every ``n_layers_per_attn_res_block``-th layer is full).
        local_layer_ids: Optional list of layer indices that use sliding-window
            attention, as emitted by vendor checkpoints.

    Returns:
        Normalized list of length ``num_hidden_layers`` whose elements are
        ``"hybrid"`` or ``"hybrid_sliding"``.

    Raises:
        ValueError: If lengths mismatch, indices are out of range, or unknown
            layer types are supplied.
    """
    if layer_types is None:
        if local_layer_ids is not None:
            local_set = {int(idx) for idx in local_layer_ids}
            out_of_range = sorted(idx for idx in local_set if not (0 <= idx < num_hidden_layers))
            if out_of_range:
                raise ValueError(f"`local_layer_ids` contains indices outside [0, {num_hidden_layers}): {out_of_range}.")
            return ["hybrid_sliding" if i in local_set else "hybrid" for i in range(num_hidden_layers)]
        period = max(1, int(n_layers_per_attn_res_block))
        return ["hybrid" if (i + 1) % period == 0 else "hybrid_sliding" for i in range(num_hidden_layers)]

    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            f"`layer_types` must have length `num_hidden_layers` ({num_hidden_layers}), got {len(layer_types)}."
        )
    normalized: list[str] = []
    for entry in layer_types:
        mapped = _LAYER_TYPE_ALIASES.get(str(entry))
        if mapped is None:
            raise ValueError(f"Unknown Inkling 2 layer type {entry!r}; expected one of {sorted(_LAYER_TYPE_ALIASES)}.")
        normalized.append(mapped)
    return normalized


@register_config("inkling2_text")
class Inkling2TextConfig(EasyDeLBaseConfig):
    """Configuration for the Inkling 2 flat MoE text decoder.

    Args:
        vocab_size: Padded vocabulary size of the input token embedding table.
            Defaults to 201024.
        unpadded_vocab_size: Output width of ``lm_head``. Checkpoints store
            ``unembed.weight`` padded to ``vocab_size``; padded rows are dropped
            on load and restored as zeros on export. Defaults to 200058 when
            ``vocab_size == 201024``, otherwise ``vocab_size``.
        hidden_size: Hidden representation dimension. Defaults to 4608.
        num_hidden_layers: Number of physical decoder layers. Defaults to 48.
        num_attention_heads: Query attention heads in full-attention layers.
            Defaults to 48.
        num_key_value_heads: Key/value heads in full-attention layers.
            Defaults to 8.
        head_dim: Head dimension in full-attention layers. Defaults to 128.
        swa_num_attention_heads: Query attention heads in sliding-window layers.
            Defaults to ``num_attention_heads``.
        swa_num_key_value_heads: Key/value heads in sliding-window layers.
            Defaults to ``num_key_value_heads``.
        swa_head_dim: Head dimension in sliding-window layers. Defaults to
            ``head_dim``.
        sliding_window_size: Causal sliding-window size for ``"hybrid_sliding"``
            layers. Defaults to 1024.
        d_rel: Per-head width of relative-position states. Defaults to 16.
        rel_extent: Maximum backward distance for relative position bias in
            full-attention layers. Defaults to 1024.
        log_scaling_n_floor: Position threshold for logarithmic scaling of full
            attention queries and relative states. Defaults to 128000.
        log_scaling_alpha: Slope of the logarithmic scaling factor. Defaults to
            0.1.
        layer_types: Optional per-layer attention types (``"hybrid"`` or
            ``"hybrid_sliding"``). When omitted, built from ``local_layer_ids``
            if present or every ``n_layers_per_attn_res_block``-th layer full.
        max_position_embeddings: Maximum sequence length. Defaults to 1048576.
        rms_norm_eps: Epsilon for RMSNorm layers. Defaults to 1e-6.
        conv_kernel_size: Kernel width for all four short causal convolutions
            per layer. Defaults to 4.
        hidden_act: Activation identifier. Defaults to ``"silu"``.
        moe_intermediate_size: Intermediate width of each routed MoE expert.
            Defaults to 2304.
        n_routed_experts: Total number of routed MoE experts per layer.
            Defaults to 384.
        num_experts_per_tok: Number of active routed experts per token.
            Defaults to 8.
        route_scale: Multiplier applied to normalized top-k routing weights.
            Defaults to 8.0.
        attention_dropout: Dropout probability on attention weights. Defaults
            to 0.0.
        initializer_range: Standard deviation for parameter initialization.
            Defaults to 0.02.
        pad_token_id: Optional padding token id. Defaults to ``None``.
        bos_token_id: Optional beginning-of-sequence token id. Defaults to
            ``None``.
        eos_token_id: End-of-sequence token id. Defaults to 200006.
        qk_l2_clip_threshold: L2 norm clip threshold for query and key head
            vectors. Defaults to ``22.627416997969522`` (``16 * sqrt(2)``).
        rel_l2_clip_threshold: L2 norm clip threshold for relative-state head
            vectors. Defaults to 8.0.
        n_layers_per_attn_res_block: Number of decoder layers accumulated into
            one residual block before becoming an attention-residual source.
            Defaults to 6.
        mlp_gate_softcap: Tanh softcap applied to the linear gate factor of
            each MoE expert's SwiGLU activation. Defaults to 4.0.
        mlp_up_softcap: Tanh softcap applied to the up projection of each MoE
            expert's SwiGLU activation. Defaults to 25.0.
        final_logit_softcapping: Tanh softcap applied to float32 LM head logits.
            Defaults to 30.0.
        scoring_func: Router score activation (``"sigmoid"`` or ``"softmax"``).
            Defaults to ``"sigmoid"``.
        use_cache: Whether to return past KV and convolution states. Defaults
            to ``True``.
        tie_word_embeddings: Kept ``False``; Inkling 2 never ties embeddings and
            the unembedding head. Defaults to ``False``.
        output_router_logits: Whether to return router logits from forward
            passes. Defaults to ``False``.
        router_aux_loss_coef: Optional router load-balancing loss coefficient.
            Defaults to 0.0.
        **kwargs: Additional Hugging Face / EasyDeL config kwargs (including
            vendor aliases ``model_max_length``, ``sconv_kernel_size``,
            ``intermediate_size``, ``num_local_experts``,
            ``routed_scaling_factor``, ``sliding_window``, and
            ``local_layer_ids``).
    """

    model_type = "inkling2_text"
    base_config_key = "text_config"
    number_of_conv_states: tp.ClassVar[int] = 4
    attribute_map: tp.ClassVar[dict[str, str]] = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "num_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        sliding_window_size: int = 1024,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 128000,
        log_scaling_alpha: float = 0.1,
        layer_types: list[str] | None = None,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        hidden_act: str = "silu",
        moe_intermediate_size: int = 2304,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        route_scale: float = 8.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 200006,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        final_logit_softcapping: float = 30.0,
        scoring_func: str = "sigmoid",
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.0,
        **kwargs,
    ):
        """Initialize :class:`Inkling2TextConfig` and validate architectural invariants."""
        # Accept Hub/HF attribute_map aliases when passed directly to __init__.
        if "sliding_window" in kwargs:
            sliding_window_size = int(kwargs.pop("sliding_window"))
        if "model_max_length" in kwargs:
            max_position_embeddings = int(kwargs.pop("model_max_length"))
        if "sconv_kernel_size" in kwargs:
            conv_kernel_size = int(kwargs.pop("sconv_kernel_size"))
        if "intermediate_size" in kwargs:
            moe_intermediate_size = int(kwargs.pop("intermediate_size"))
        if "num_local_experts" in kwargs:
            n_routed_experts = int(kwargs.pop("num_local_experts"))
        if "num_experts" in kwargs:
            n_routed_experts = int(kwargs.pop("num_experts"))
        if "routed_scaling_factor" in kwargs:
            route_scale = float(kwargs.pop("routed_scaling_factor"))
        local_layer_ids = kwargs.pop("local_layer_ids", None)
        hf_quantization_config = kwargs.pop("quantization_config", None)

        if unpadded_vocab_size is None:
            unpadded_vocab_size = (
                DEFAULT_UNPADDED_VOCAB_SIZE if int(vocab_size) == DEFAULT_VOCAB_SIZE else int(vocab_size)
            )
        if not (1 <= int(unpadded_vocab_size) <= int(vocab_size)):
            raise ValueError(
                f"`unpadded_vocab_size` ({unpadded_vocab_size}) must satisfy 1 <= unpadded_vocab_size <= vocab_size "
                f"({vocab_size})."
            )
        if int(n_layers_per_attn_res_block) <= 0:
            raise ValueError(f"`n_layers_per_attn_res_block` ({n_layers_per_attn_res_block}) must be positive.")
        if int(conv_kernel_size) <= 0:
            raise ValueError(f"`conv_kernel_size` ({conv_kernel_size}) must be positive.")

        self.vocab_size = int(vocab_size)
        self.unpadded_vocab_size = int(unpadded_vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.swa_num_attention_heads = (
            int(num_attention_heads) if swa_num_attention_heads is None else int(swa_num_attention_heads)
        )
        self.swa_num_key_value_heads = (
            int(num_key_value_heads) if swa_num_key_value_heads is None else int(swa_num_key_value_heads)
        )
        self.swa_head_dim = int(head_dim) if swa_head_dim is None else int(swa_head_dim)
        self.sliding_window_size = int(sliding_window_size)
        self.d_rel = int(d_rel)
        self.rel_extent = int(rel_extent)
        self.log_scaling_n_floor = int(log_scaling_n_floor)
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.conv_kernel_size = int(conv_kernel_size)
        self.hidden_act = str(hidden_act)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.n_routed_experts = int(n_routed_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.route_scale = float(route_scale)
        self.attention_dropout = float(attention_dropout)
        self.initializer_range = float(initializer_range)
        self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.n_layers_per_attn_res_block = int(n_layers_per_attn_res_block)
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.final_logit_softcapping = float(final_logit_softcapping)
        self.scoring_func = str(scoring_func)
        self.use_cache = bool(use_cache)
        self.output_router_logits = bool(output_router_logits)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.layer_types = _normalize_layer_types(
            layer_types=layer_types,
            num_hidden_layers=self.num_hidden_layers,
            n_layers_per_attn_res_block=self.n_layers_per_attn_res_block,
            local_layer_ids=local_layer_ids,
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if hf_quantization_config is not None:
            self.quantization_config = hf_quantization_config

    @property
    def total_num_hidden_layers(self) -> int:
        """Total number of decoder layer executions (and cache slots) per forward pass."""
        return self.num_hidden_layers

    @property
    def effective_layer_types(self) -> list[str]:
        """Per-cache-slot attention type list matching the physical/looped cache layout."""
        return list(self.layer_types)

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """Return whether physical layer ``layer_idx`` uses sliding-window attention."""
        return self.layer_types[layer_idx] == "hybrid_sliding"

    def layer_kv_shape(self, cache_layer_idx: int) -> tuple[int, int]:
        """Return ``(num_key_value_heads, head_dim)`` for cache slot ``cache_layer_idx``."""
        is_sliding = self.effective_layer_types[cache_layer_idx] == "hybrid_sliding"
        if is_sliding:
            return self.swa_num_key_value_heads, self.swa_head_dim
        return self.num_key_value_heads, self.head_dim

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Return per-cache-slot attention mask metadata for cache construction."""
        details: dict[int, AttnMaskDetail] = {}
        for idx, layer_type in enumerate(self.effective_layer_types):
            if layer_type == "hybrid_sliding":
                details[idx] = AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=self.sliding_window_size)
            else:
                details[idx] = AttnMaskDetail(mask_type=AttnMaskType.FULL, size=self.max_position_embeddings)
        return details


@register_config("inkling2_vision")
class Inkling2VisionConfig(EasyDeLBaseConfig):
    """Configuration for the Inkling 2 vision encoder and spatial-merge projector.

    Args:
        hidden_size: Vision encoder hidden dimension. Defaults to 1024.
        intermediate_size: Vision encoder MLP hidden dimension. Defaults to
            ``int(hidden_size * mlp_ratio)``.
        num_hidden_layers: Number of vision encoder blocks. Defaults to 23.
        num_attention_heads: Number of attention heads per block. Defaults to 16.
        num_channels: Number of input image channels. Defaults to 3.
        patch_size: Native spatial patch side length in pixels. Defaults to 20.
        hidden_act: Activation function used in encoder MLPs and the merger.
            Defaults to ``"gelu"``.
        layer_norm_eps: Epsilon for LayerNorm modules. Defaults to 1e-5.
        attention_dropout: Attention dropout probability. Defaults to 0.0.
        mlp_ratio: Ratio of ``intermediate_size`` to ``hidden_size`` when
            ``intermediate_size`` is omitted. Defaults to 4.0.
        layer_scale_init_value: Initial value for LayerScale parameters
            ``lambda_1`` and ``lambda_2``. Defaults to 0.1.
        rope_parameters: Optional rotary configuration mapping. Defaults to
            ``{"rope_type": "axial", "rope_theta": 10000.0}``.
        rope_theta: Base frequency for axial 2-D RoPE when ``rope_parameters``
            omits ``rope_theta``. Defaults to 10000.0.
        spatial_merge_size: Side length of the square patch block concatenated
            into one language-model image token. Defaults to 2.
        posemb_grid_size: Side length of the learned 2-D position-embedding
            grid interpolated onto each image's patch grid. Defaults to 32.
        projection_hidden_dim: Intermediate dimension of the two-layer vision
            merger MLP. Defaults to 4096.
        projector_bias: Whether the vision merger linear layers include bias.
            Defaults to ``False``.
        text_hidden_size: Hidden size of the receiving language model. Defaults
            to 4608.
        initializer_range: Standard deviation for parameter initialization.
            Defaults to 0.02.
        **kwargs: Additional Hugging Face / EasyDeL config kwargs (including
            vendor aliases ``native_patch_size`` and ``adapter_merge_size``).
    """

    model_type = "inkling2_vision"
    base_config_key = "vision_config"
    default_rope_type: tp.ClassVar[str] = "axial"
    attribute_map: tp.ClassVar[dict[str, str]] = {
        "native_patch_size": "patch_size",
        "adapter_merge_size": "spatial_merge_size",
    }

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 23,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 20,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 0.1,
        rope_parameters: dict[str, tp.Any] | None = None,
        rope_theta: float = DEFAULT_VISION_ROPE_THETA,
        spatial_merge_size: int = 2,
        posemb_grid_size: int = 32,
        projection_hidden_dim: int = 4096,
        projector_bias: bool = False,
        text_hidden_size: int = 4608,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize :class:`Inkling2VisionConfig`."""
        if "native_patch_size" in kwargs:
            patch_size = int(kwargs.pop("native_patch_size"))
        if "adapter_merge_size" in kwargs:
            spatial_merge_size = int(kwargs.pop("adapter_merge_size"))

        resolved_rope: dict[str, tp.Any] = dict(rope_parameters) if isinstance(rope_parameters, Mapping) else {}
        resolved_rope.setdefault("rope_type", self.default_rope_type)
        resolved_rope.setdefault("rope_theta", float(rope_theta))
        if resolved_rope["rope_type"] != "axial":
            raise ValueError(f"Inkling2VisionConfig supports only 'axial' RoPE, got {resolved_rope['rope_type']!r}.")

        self.hidden_size = int(hidden_size)
        self.mlp_ratio = float(mlp_ratio)
        self.intermediate_size = (
            int(self.hidden_size * self.mlp_ratio) if intermediate_size is None else int(intermediate_size)
        )
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_channels = int(num_channels)
        self.patch_size = int(patch_size)
        self.hidden_act = str(hidden_act)
        self.layer_norm_eps = float(layer_norm_eps)
        self.attention_dropout = float(attention_dropout)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.rope_theta = float(resolved_rope["rope_theta"])
        self.spatial_merge_size = int(spatial_merge_size)
        self.posemb_grid_size = int(posemb_grid_size)
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.projector_bias = bool(projector_bias)
        self.text_hidden_size = int(text_hidden_size)
        self.initializer_range = float(initializer_range)
        super().__init__(**kwargs)
        self.rope_parameters = resolved_rope


@register_config("inkling2")
class Inkling2Config(EasyDeLBaseConfig):
    """Composite vision-language configuration for Inkling 2.

    Args:
        text_config: Text decoder configuration or mapping. Defaults to
            :class:`Inkling2TextConfig`.
        vision_config: Vision encoder configuration or mapping. Defaults to
            :class:`Inkling2VisionConfig` with ``text_hidden_size`` matched to
            ``text_config.hidden_size``.
        image_token_id: Placeholder token id replaced by image features.
            Defaults to 200054.
        eos_token_id: End-of-sequence token id. Defaults to 200006.
        tie_word_embeddings: Kept ``False``; Inkling 2 never ties embeddings and
            the LM head. Defaults to ``False``.
        **kwargs: Additional Hugging Face / EasyDeL config kwargs (including
            optional ``mtp_config`` and ``quantization_config``).
    """

    model_type = "inkling2"
    sub_configs: tp.ClassVar[dict[str, type[EasyDeLBaseConfig]]] = {
        "text_config": Inkling2TextConfig,
        "vision_config": Inkling2VisionConfig,
    }
    keys_to_ignore_at_inference: tp.ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, tp.Any] | Inkling2TextConfig | None = None,
        vision_config: Mapping[str, tp.Any] | Inkling2VisionConfig | None = None,
        image_token_id: int = 200054,
        eos_token_id: int | None = 200006,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the composite :class:`Inkling2Config` and validate tower width alignment."""
        hf_quantization_config = kwargs.pop("quantization_config", None)
        mtp_config = kwargs.pop("mtp_config", None)

        text_cls = self.sub_configs["text_config"]
        vision_cls = self.sub_configs["vision_config"]

        if isinstance(text_config, Mapping):
            text_kwargs = dict(text_config)
            text_kwargs.pop("model_type", None)
            self.text_config = text_cls(**self._fix_parent_kws(text_kwargs, kwargs))
        elif text_config is None:
            self.text_config = text_cls()
        else:
            self.text_config = text_config

        if isinstance(vision_config, Mapping):
            vision_kwargs = dict(vision_config)
            vision_kwargs.pop("model_type", None)
            # Hub checkpoints such as thinkingmachines/Inkling-2-Small omit
            # `text_hidden_size` in `vision_config`; inherit it from the text tower.
            vision_kwargs.setdefault("text_hidden_size", self.text_config.hidden_size)
            self.vision_config = vision_cls(**self._fix_parent_kws(vision_kwargs, kwargs))
        elif vision_config is None:
            self.vision_config = vision_cls(text_hidden_size=self.text_config.hidden_size)
        else:
            self.vision_config = vision_config

        self.image_token_id = int(image_token_id)
        self.mtp_config = dict(mtp_config) if isinstance(mtp_config, Mapping) else mtp_config
        self.validate_architecture()

        super().__init__(
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if hf_quantization_config is not None:
            self.quantization_config = hf_quantization_config

    def validate_architecture(self) -> None:
        """Verify that the vision merger output width matches the text hidden size."""
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`vision_config.text_hidden_size` ({self.vision_config.text_hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )

    def get_text_config(self, decoder: bool = True) -> Inkling2TextConfig:
        """Return the text decoder configuration."""
        del decoder
        return self.text_config  # pyright: ignore[reportReturnType]

    def get_vision_config(self) -> Inkling2VisionConfig:
        """Return the vision encoder configuration."""
        return self.vision_config  # pyright: ignore[reportReturnType]

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Delegate per-layer attention mask metadata to the text decoder config."""
        return self.text_config.get_mask_details()


@register_config("inkling2_loop_text")
class Inkling2LoopTextConfig(Inkling2TextConfig):
    """Configuration for the looped Inkling 2 text decoder (e.g. Inkling-2-Mini).

    Extends :class:`Inkling2TextConfig` with a recurrent middle loop spanning
    layers ``[loop_layer_index_start, loop_layer_index_end)`` executed
    ``loop_num_passes`` times. Passes after the first restart from the pre-loop
    residual stream and blend cross-pass messages retrieved from the previous
    pass's block states via low-rank message gates of width ``loop_gate_rank``.

    Args:
        loop_layer_index_start: First physical layer in the looped core. Must
            be positive and divisible by ``n_layers_per_attn_res_block``.
            Defaults to 6.
        loop_layer_index_end: Exclusive end index of the looped core. Must be
            divisible by ``n_layers_per_attn_res_block`` and at most
            ``num_hidden_layers``. Defaults to 18.
        loop_num_passes: Total number of passes through the looped core. Must
            be at least 1. Defaults to 3.
        loop_gate_rank: Bottleneck rank of the cross-pass message gate
            controllers. Defaults to 64.
        **kwargs: Remaining :class:`Inkling2TextConfig` parameters, plus
            optional Hub nested ``loop_config`` mapping (``core_start``,
            ``core_end``, ``num_passes``, ``gate_rank``).
    """

    model_type = "inkling2_loop_text"

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 2304,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        sliding_window_size: int = 1024,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 8192,
        log_scaling_alpha: float = 0.1,
        layer_types: list[str] | None = None,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        hidden_act: str = "silu",
        moe_intermediate_size: int = 1152,
        n_routed_experts: int = 128,
        num_experts_per_tok: int = 8,
        route_scale: float = 8.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 200006,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        final_logit_softcapping: float = 30.0,
        scoring_func: str = "sigmoid",
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.0,
        loop_layer_index_start: int = 6,
        loop_layer_index_end: int = 18,
        loop_num_passes: int = 3,
        loop_gate_rank: int = 64,
        **kwargs,
    ):
        """Initialize :class:`Inkling2LoopTextConfig` and validate loop boundaries."""
        loop_config = kwargs.pop("loop_config", None)
        if isinstance(loop_config, Mapping):
            loop_layer_index_start = int(loop_config.get("core_start", loop_layer_index_start))
            loop_layer_index_end = int(loop_config.get("core_end", loop_layer_index_end))
            loop_num_passes = int(loop_config.get("num_passes", loop_num_passes))
            loop_gate_rank = int(loop_config.get("gate_rank", loop_gate_rank))

        self.loop_layer_index_start = int(loop_layer_index_start)
        self.loop_layer_index_end = int(loop_layer_index_end)
        self.loop_num_passes = int(loop_num_passes)
        self.loop_gate_rank = int(loop_gate_rank)

        super().__init__(
            vocab_size=vocab_size,
            unpadded_vocab_size=unpadded_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            swa_num_attention_heads=swa_num_attention_heads,
            swa_num_key_value_heads=swa_num_key_value_heads,
            swa_head_dim=swa_head_dim,
            sliding_window_size=sliding_window_size,
            d_rel=d_rel,
            rel_extent=rel_extent,
            log_scaling_n_floor=log_scaling_n_floor,
            log_scaling_alpha=log_scaling_alpha,
            layer_types=layer_types,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            conv_kernel_size=conv_kernel_size,
            hidden_act=hidden_act,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            route_scale=route_scale,
            attention_dropout=attention_dropout,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            qk_l2_clip_threshold=qk_l2_clip_threshold,
            rel_l2_clip_threshold=rel_l2_clip_threshold,
            n_layers_per_attn_res_block=n_layers_per_attn_res_block,
            mlp_gate_softcap=mlp_gate_softcap,
            mlp_up_softcap=mlp_up_softcap,
            final_logit_softcapping=final_logit_softcapping,
            scoring_func=scoring_func,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            **kwargs,
        )
        self.validate_architecture()

    def validate_architecture(self) -> None:
        """Validate loop boundaries against block-residual alignment and layer count.

        Raises:
            ValueError: If loop pass count, rank, or start/end indices violate
                Inkling 2 Loop invariants.
        """
        if self.loop_num_passes < 1:
            raise ValueError(f"`loop_num_passes` ({self.loop_num_passes}) must be at least 1.")
        if self.loop_gate_rank < 1:
            raise ValueError(f"`loop_gate_rank` ({self.loop_gate_rank}) must be positive.")
        if self.loop_layer_index_start <= 0:
            raise ValueError(f"`loop_layer_index_start` ({self.loop_layer_index_start}) must be greater than 0.")
        if self.loop_layer_index_end <= self.loop_layer_index_start:
            raise ValueError(
                f"`loop_layer_index_end` ({self.loop_layer_index_end}) must be greater than "
                f"`loop_layer_index_start` ({self.loop_layer_index_start})."
            )
        if self.loop_layer_index_end > self.num_hidden_layers:
            raise ValueError(
                f"`loop_layer_index_end` ({self.loop_layer_index_end}) must not exceed "
                f"`num_hidden_layers` ({self.num_hidden_layers})."
            )
        if self.loop_layer_index_start % self.n_layers_per_attn_res_block != 0:
            raise ValueError(
                f"`loop_layer_index_start` ({self.loop_layer_index_start}) must be a multiple of "
                f"`n_layers_per_attn_res_block` ({self.n_layers_per_attn_res_block})."
            )
        if self.loop_layer_index_end % self.n_layers_per_attn_res_block != 0:
            raise ValueError(
                f"`loop_layer_index_end` ({self.loop_layer_index_end}) must be a multiple of "
                f"`n_layers_per_attn_res_block` ({self.n_layers_per_attn_res_block})."
            )

    @property
    def num_loop_layers(self) -> int:
        """Number of physical layers inside the looped core."""
        return self.loop_layer_index_end - self.loop_layer_index_start

    @property
    def total_num_hidden_layers(self) -> int:
        """Total number of decoder layer executions (and cache slots) per forward pass."""
        return self.num_hidden_layers + (self.loop_num_passes - 1) * self.num_loop_layers

    @property
    def effective_layer_types(self) -> list[str]:
        """Per-cache-slot attention types: physical layers followed by extra loop passes."""
        loop_types = list(self.layer_types[self.loop_layer_index_start : self.loop_layer_index_end])
        return list(self.layer_types) + loop_types * (self.loop_num_passes - 1)


@register_config("inkling2_loop")
class Inkling2LoopConfig(Inkling2Config):
    """Composite vision-language configuration for Inkling 2 Loop (Inkling-2-Mini).

    Uses :class:`Inkling2LoopTextConfig` for ``text_config`` and
    :class:`Inkling2VisionConfig` (defaulting ``text_hidden_size=2304``) for
    ``vision_config``.
    """

    model_type = "inkling2_loop"
    sub_configs: tp.ClassVar[dict[str, type[EasyDeLBaseConfig]]] = {
        "text_config": Inkling2LoopTextConfig,
        "vision_config": Inkling2VisionConfig,
    }

    def __init__(
        self,
        text_config: Mapping[str, tp.Any] | Inkling2LoopTextConfig | None = None,
        vision_config: Mapping[str, tp.Any] | Inkling2VisionConfig | None = None,
        image_token_id: int = 200054,
        eos_token_id: int | None = 200006,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the composite :class:`Inkling2LoopConfig`."""
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = [
    "Inkling2Config",
    "Inkling2LoopConfig",
    "Inkling2LoopTextConfig",
    "Inkling2TextConfig",
    "Inkling2VisionConfig",
]
