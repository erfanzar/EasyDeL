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

"""Configuration classes for the Muse-Glimmer multimodal model.

Mirrors the HuggingFace ``muse_glimmer`` implementation:

- :class:`MuseGlimmerVisionConfig` — windowed ViT tower with 2-D RoPE and a
  ``merge_size**2`` pixel-shuffle head.
- :class:`MuseGlimmerTextConfig` — decoder with gated attention, a scale-less
  QK-norm, per-layer RoPE base theta (``0`` meaning NoPE), a sliding/full
  attention schedule and tanh logit soft-capping.
- :class:`MuseGlimmerConfig` — composite config binding the two together with
  the vision adapter/projection widths.
"""

import typing
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


def _normalize_rope_dict(
    rope_parameters: Mapping[str, typing.Any] | None,
    rope_scaling: Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    """Merge ``rope_parameters`` / ``rope_scaling`` into one normalized dict.

    HF checkpoints newer than v5 store the RoPE spec under ``rope_parameters``;
    older EasyDeL-flavoured configs use ``rope_scaling`` (which wins when both
    are given). Both ``type`` and ``rope_type`` are populated on the result so
    the value survives a round trip through either convention.

    Args:
        rope_parameters: Newer-style RoPE dict (may be ``None``).
        rope_scaling: Legacy RoPE scaling dict (takes precedence when set).

    Returns:
        dict | None: Normalized copy, or ``None`` when neither input was given.
    """
    source = rope_scaling if rope_scaling is not None else rope_parameters
    if source is None:
        return None
    normalized = dict(source)
    if "type" in normalized and "rope_type" not in normalized:
        normalized["rope_type"] = normalized["type"]
    normalized.setdefault("rope_type", "default")
    return normalized


@register_config("muse_glimmer_vision")
class MuseGlimmerVisionConfig(EasyDeLBaseConfig):
    """Configuration for the Muse-Glimmer vision tower.

    The tower is a NaViT-style packed ViT: images and videos arrive already
    flattened into ``patch_temporal * in_channels * patch_size**2`` rows, are
    embedded by a single linear projection, receive a bilinearly-resampled
    learned position embedding, and run through ``num_hidden_layers`` blocks
    that alternate window attention with full attention. The tower output is
    pixel-shuffled by ``merge_size`` in both spatial axes, so the language
    model sees ``hidden_size * merge_size**2`` channels per merged token.

    Args:
        patch_size: Spatial patch edge in pixels. Defaults to 14.
        pos_emb_height: Height of the learned position-embedding grid. Defaults to 32.
        pos_emb_width: Width of the learned position-embedding grid. Defaults to 32.
        num_attention_heads: Attention heads per vision block. Defaults to 16.
        num_hidden_layers: Number of vision blocks. Defaults to 50.
        hidden_size: Vision hidden width. Defaults to 1536.
        intermediate_size: Vision FFN width. Defaults to 8960.
        hidden_act: Vision FFN activation. Defaults to ``"gelu"``.
        in_channels: Input image channels. Defaults to 3.
        rope_theta: Base frequency of the 2-D vision RoPE. Defaults to 10000.0.
        rope_parameters: HF v5-style RoPE dict (alternative to ``rope_theta``).
        rope_scaling: Legacy RoPE dict; takes precedence over ``rope_parameters``.
        max_position_embeddings: ``pos_emb_height * pos_emb_width``. Defaults to 1024.
        patch_temporal: Temporal patch depth used when packing video. Defaults to 2.
        merge_size: Pixel-shuffle merge factor per spatial axis. Defaults to 2.
        layer_norm_eps: Epsilon of ``ln_pre`` / ``ln_post``. Defaults to 1e-5.
        layer_types: Per-layer ``"window_attention"`` / ``"full_attention"``
            schedule. When ``None``, every 4th layer (and the last) is full.
        initializer_range: Stddev of the normal weight initializer. Defaults to 0.02.
    """

    model_type = "muse_glimmer_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        patch_size: int = 14,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 50,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        hidden_act: str = "gelu",
        in_channels: int = 3,
        rope_theta: float | None = None,
        rope_parameters: Mapping[str, typing.Any] | None = None,
        rope_scaling: Mapping[str, typing.Any] | None = None,
        max_position_embeddings: int | None = None,
        patch_temporal: int = 2,
        merge_size: int = 2,
        layer_norm_eps: float = 1e-5,
        layer_types: list[str] | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize MuseGlimmerVisionConfig.

        See the class docstring for parameter semantics; ``**kwargs`` are
        forwarded to :class:`EasyDeLBaseConfig` for the standard EasyDeL
        plumbing (sharding, dtype, attention mechanism, ...).
        """
        self.patch_size = patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.in_channels = in_channels
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.head_dim = hidden_size // num_attention_heads

        rope_dict = _normalize_rope_dict(rope_parameters, rope_scaling)
        if rope_theta is None:
            rope_theta = float(rope_dict.get("rope_theta", 10000.0)) if rope_dict else 10000.0
        if rope_dict is None:
            rope_dict = {"rope_type": "default"}
        rope_dict.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_dict

        if max_position_embeddings is None:
            max_position_embeddings = pos_emb_height * pos_emb_width
        self.max_position_embeddings = max_position_embeddings

        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % 4 == 0 or i == num_hidden_layers - 1 else "window_attention"
                for i in range(num_hidden_layers)
            ]

        # `layer_types` is assigned *after* the base initializer on purpose: the
        # HuggingFace base class validates it against a global allow-list that
        # only gained "window_attention" in the release that introduced this
        # architecture, so on older `transformers` the init-time check would
        # reject a perfectly valid vision schedule. `validate` below re-runs the
        # base validators with a vision-aware layer-type check substituted in.
        super().__init__(**kwargs)
        self.layer_types = layer_types
        self.validate_layer_type()

    def validate_layer_type(self) -> None:
        """Validate :attr:`layer_types` against the vision tower's own vocabulary.

        Applies the same shape and length rules as the HuggingFace base
        validator, but against the two layer types this tower actually
        supports, so the check does not depend on the installed
        ``transformers`` release knowing about ``"window_attention"``.

        Raises:
            ValueError: If an entry is not a known vision layer type, or if the
                schedule length does not match ``num_hidden_layers``.
        """
        layer_types = getattr(self, "layer_types", None)
        if layer_types is None:
            return
        allowed = ("full_attention", "window_attention")
        if not all(layer_type in allowed for layer_type in layer_types):
            raise ValueError(f"The `layer_types` entries must be in {allowed} but got {layer_types}")
        if self.num_hidden_layers != len(layer_types):
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must be equal to the number of "
                f"`layer_types` ({len(layer_types)})"
            )

    def validate(self) -> None:
        """Run the HuggingFace class validators with the vision layer-type check.

        The strict-dataclass decorator captures validator functions on
        ``PreTrainedConfig`` itself, so a plain method override is never
        consulted. This re-dispatches the collected validators through the
        instance, which routes ``validate_layer_type`` to the vision-aware
        implementation above while leaving every other check untouched.
        """
        for validator in type(self).__class_validators__:
            getattr(self, validator.__name__)()

    @property
    def window_size(self) -> int:
        """Window edge, in pixels, used to group patches for window attention.

        Matches HF's ``MuseGlimmerVisionModel.window_size`` — the learned
        position grid side times the patch edge.
        """
        return self.pos_emb_height * self.patch_size

    @property
    def spatial_merge_size(self) -> int:
        """Alias for :attr:`merge_size` under EasyDeL's conventional name.

        The stored field keeps the HuggingFace spelling so ``config.json``
        round-trips, but the shared multimodal preprocessing in eSurge probes
        ``spatial_merge_size`` to align patch grids; without this alias it would
        read the default of ``1`` and skip the divisibility padding.
        """
        return self.merge_size

    @property
    def temporal_patch_size(self) -> int:
        """Alias for :attr:`patch_temporal` under EasyDeL's conventional name.

        Same rationale as :attr:`spatial_merge_size` — the shared patchifier
        reads ``temporal_patch_size`` when building flat patch inputs.
        """
        return self.patch_temporal

    @property
    def out_hidden_size(self) -> int:
        """Channel width emitted by the tower after ``merge_size`` pixel shuffle."""
        return self.hidden_size * self.merge_size**2


@register_config("muse_glimmer_text")
class MuseGlimmerTextConfig(EasyDeLBaseConfig):
    """Configuration for the Muse-Glimmer language model.

    Differences from a vanilla Llama-style decoder that this config carries:

    - **Gated attention** — an extra ``gate_proj`` of width
      ``num_attention_heads * head_dim`` whose sigmoid multiplies the attention
      output before ``o_proj``.
    - **Scale-less QK-norm** — Q and K are RMS-normalized without a learnable
      scale, and Q is then multiplied by ``qk_scale_factor`` on top of the usual
      ``1/sqrt(head_dim)``.
    - **Per-layer RoPE base** — ``layer_rope_theta[i] == 0`` marks a NoPE layer.
      By default every 4th layer counted backward from the last is NoPE, and
      those same layers use full attention while the rest slide.
    - **Sandwich norms** — a post-attention and post-FFN norm (each with
      ``post_norm_eps``) sit between the sub-layer output and the residual add.
    - **Soft-capped logits** — logits are scaled by ``output_multiplier`` then
      passed through ``T * tanh(x / T)`` with ``T = final_logit_softcapping``.

    Args:
        vocab_size: Vocabulary size. Defaults to 202048.
        hidden_size: Residual stream width. Defaults to 6656.
        intermediate_size: SwiGLU FFN width. Defaults to 19968.
        num_hidden_layers: Number of decoder layers. Defaults to 52.
        num_attention_heads: Query heads. Defaults to 32.
        num_key_value_heads: Key/value heads (GQA). Defaults to 2.
        head_dim: Per-head width. Defaults to 128.
        hidden_activation: FFN activation. Defaults to ``"silu"``.
        max_position_embeddings: Context-length bound used to size RoPE. Defaults to 131072.
        initializer_range: Stddev of the normal weight initializer. Defaults to 0.02.
        rms_norm_eps: Epsilon of the pre-norms, QK-norm and final norm. Defaults to 1e-5.
        use_cache: Whether downstream code should return a KV cache. Defaults to True.
        tie_word_embeddings: Tie the LM head to the input embedding. Defaults to False.
        rope_theta: Global RoPE base frequency. Defaults to 500000.0.
        rope_parameters: HF v5-style RoPE dict (alternative to ``rope_theta``).
        rope_scaling: Legacy RoPE dict; takes precedence over ``rope_parameters``.
        attention_bias: Whether Q/K/V/O carry biases. Defaults to False.
        attention_dropout: Dropout on attention probabilities. Defaults to 0.0.
        sliding_window: Window size of the sliding layers. Defaults to 2048.
        layer_types: Per-layer ``"sliding_attention"`` / ``"full_attention"``
            schedule. When ``None``, every 4th layer counted backward from the
            last is full and the rest slide.
        final_logit_softcapping: Tanh soft-cap applied to the final logits. Defaults to 20.0.
        qk_scale_factor: Extra multiplier on Q after the scale-less QK-norm. Defaults to 3.87.
        output_multiplier: Pre-softcap logit scale. Defaults to ``1/sqrt(hidden_size/256)``.
        post_norm_eps: Epsilon of the post-attention / post-FFN norms. Defaults to 1e-8.
        layer_rope_theta: Per-layer RoPE base theta, ``0`` disabling rotary for
            that layer. When ``None``, derived from ``rope_theta`` and the NoPE
            schedule.
        pad_token_id: Padding token id.
        bos_token_id: Beginning-of-stream token id. Defaults to 200000.
        eos_token_id: End-of-stream token id. Defaults to 200001.
    """

    model_type = "muse_glimmer_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202_048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_parameters: Mapping[str, typing.Any] | None = None,
        rope_scaling: Mapping[str, typing.Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = 2048,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = 20.0,
        qk_scale_factor: float = 3.87,
        output_multiplier: float = 0.19611613513818404,
        post_norm_eps: float = 1e-8,
        layer_rope_theta: list[float] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 200_000,
        eos_token_id: int | list[int] | None = 200_001,
        **kwargs,
    ):
        """Initialize MuseGlimmerTextConfig.

        See the class docstring for parameter semantics. ``**kwargs`` are
        forwarded to :class:`EasyDeLBaseConfig`.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        # `hidden_act` is what EasyDeL's shared helpers look for; keep both in sync.
        self.hidden_act = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.post_norm_eps = post_norm_eps

        rope_dict = _normalize_rope_dict(rope_parameters, rope_scaling)
        if rope_theta is None:
            rope_theta = float(rope_dict.get("rope_theta", 500_000.0)) if rope_dict else 500_000.0
        if rope_dict is None:
            rope_dict = {"rope_type": "default"}
        rope_dict.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_dict

        if layer_types is None:
            # Full attention on the NoPE layers (every 4th, counted backward from
            # the last); the reference config's [w, w, w, 0] sliding pattern.
            layer_types = [
                "full_attention" if (num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        if layer_rope_theta is None:
            layer_rope_theta = [
                0.0 if (num_hidden_layers - 1 - i) % 4 == 0 else rope_theta for i in range(num_hidden_layers)
            ]
        self.layer_rope_theta = [float(theta) for theta in layer_rope_theta]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Map each decoder layer to its attention-mask category.

        Layers marked ``"sliding_attention"`` in :attr:`layer_types` get a
        sliding-window mask of width :attr:`sliding_window`; the rest get a
        full causal mask. Consumed by the attention dispatcher and by eSurge
        when grouping KV-cache layers.

        Returns:
            dict[int, AttnMaskDetail]: One entry per decoder layer.
        """
        mapping: dict[int, AttnMaskDetail] = {}
        if self.layer_types is None:
            return mapping
        for layer_idx in range(self.num_hidden_layers):
            mapping[layer_idx] = AttnMaskDetail(
                mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                size=self.sliding_window,
            )
        return mapping


@register_config("muse_glimmer")
class MuseGlimmerConfig(EasyDeLBaseConfig):
    """Composite configuration for the Muse-Glimmer vision-language model.

    Args:
        text_config: Text decoder config, dict or :class:`MuseGlimmerTextConfig`.
        vision_config: Vision tower config, dict or :class:`MuseGlimmerVisionConfig`.
        image_token_id: Placeholder token replaced by image features. Defaults to 200092.
        video_token_id: Placeholder token replaced by video features. Defaults to 200091.
        out_hidden_size: Vision-tower output width feeding the adapter. Defaults to 6144.
        projector_hidden_size: Adapter hidden width. Defaults to 4096.
        projector_hidden_act: Adapter activation. Defaults to ``"gelu"``.
        tie_word_embeddings: Tie the LM head to the input embedding. Defaults to False.
    """

    model_type = "muse_glimmer"
    sub_configs: typing.ClassVar = {
        "text_config": MuseGlimmerTextConfig,
        "vision_config": MuseGlimmerVisionConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, typing.Any] | MuseGlimmerTextConfig | None = None,
        vision_config: Mapping[str, typing.Any] | MuseGlimmerVisionConfig | None = None,
        image_token_id: int = 200_092,
        video_token_id: int = 200_091,
        out_hidden_size: int = 6144,
        projector_hidden_size: int = 4096,
        projector_hidden_act: str = "gelu",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize MuseGlimmerConfig.

        Accepts ``text_config`` / ``vision_config`` either as dicts (built via
        the registered sub-config classes) or as already-built configuration
        objects. ``**kwargs`` are forwarded to :class:`EasyDeLBaseConfig`.
        """
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(dict(text_config), kwargs))
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(dict(vision_config), kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.projector_hidden_act = projector_hidden_act

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> MuseGlimmerTextConfig:
        """Return the text sub-configuration.

        Args:
            decoder: Ignored, kept for HuggingFace API compatibility.

        Returns:
            MuseGlimmerTextConfig: The language-model configuration.
        """
        return self.text_config  # pyright: ignore[reportReturnType]
