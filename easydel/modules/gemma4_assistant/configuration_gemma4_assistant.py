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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the Gemma4 Assistant (MTP drafter) model.

Mirrors the HF ``google/gemma-4-*-it-assistant`` config layout:

- :class:`Gemma4AssistantTextConfig` describes the small 4-layer
  Gemma4-style decoder (hidden 256 / 1024 depending on variant).
- :class:`Gemma4AssistantConfig` is the composite top-level config
  carrying the text sub-config plus drafter-specific fields:
  ``backbone_hidden_size`` (the target model's hidden size that
  ``pre_projection`` and ``post_projection`` bridge), centroid head
  hyper-parameters (``num_centroids``, ``centroid_intermediate_top_k``,
  ``use_ordered_embeddings``), and multimodal special-token IDs
  inherited from the target Gemma4 tokenizer.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.modules.gemma4.gemma4_configuration import Gemma4TextConfig


@register_config("gemma4_assistant_text")
class Gemma4AssistantTextConfig(Gemma4TextConfig):
    """Text-side config for the Gemma4 Assistant drafter.

    Subclasses :class:`Gemma4TextConfig` with much smaller defaults
    (E4B drafter ships 4 layers at hidden=256). All MoE-related fields
    inherited from Gemma4 are kept as no-ops since the drafter is
    purely dense.

    Args:
        vocab_size: Drafter vocabulary size; matches the target
            tokenizer. Default ``262144`` (Gemma4 tokenizer size).
        hidden_size: Drafter hidden dimension. Default ``256``
            (E4B-assistant). 26B-assistant uses ``1024``.
        intermediate_size: GeGLU intermediate dimension. Default
            ``2048``. 26B-assistant uses ``8192``.
        num_hidden_layers: Always ``4`` for published assistants.
        num_attention_heads: ``4`` (E4B) or ``16`` (26B).
        num_key_value_heads: ``2`` (E4B) or ``8`` (26B). Set to match
            the TARGET model's KV head layout so cross-model KV
            sharing has the right shapes.
        head_dim: Per-head dim for sliding-attention layers (256).
        global_head_dim: Per-head dim for the full-attention layer
            (512 for E4B). The HF Gemma4 attention layer uses this
            for global layers.
        layer_types: Per-layer attention type. The published
            assistants use ``["sliding_attention"] * 3 +
            ["full_attention"]``.
        sliding_window: Sliding-attention window size (512 for E4B,
            1024 for 26B). Mirrors the target model's window.
        num_kv_shared_layers: For the assistant this equals
            ``num_hidden_layers`` — every drafter layer pulls K/V
            from the target's KV cache instead of computing its own.
        max_position_embeddings: Context length the drafter accepts;
            should match the target's context.
        rms_norm_eps: 1e-6 for Gemma4.
        rope_parameters: Per-layer-type RoPE config dict.
        tie_word_embeddings: Always ``True`` for the drafter — its
            ``embed_tokens`` doubles as the LM head.
        attention_k_eq_v: Inherited Gemma4 flag; published assistants
            set ``False``. Has no effect since the drafter has no
            K/V projections of its own.
        **kwargs: Forwarded to :class:`Gemma4TextConfig`.
    """

    model_type = "gemma4_assistant_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 256,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        global_head_dim: int | None = 512,
        layer_types: list[str] | None = None,
        sliding_window: int = 512,
        num_kv_shared_layers: int | None = None,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        rope_parameters: dict | None = None,
        tie_word_embeddings: bool = True,
        attention_k_eq_v: bool = False,
        **kwargs,
    ):
        """Initialize a :class:`Gemma4AssistantTextConfig`.

        See the class docstring for the full semantics of each field.
        Highlights: ``num_kv_shared_layers`` defaults to ``num_hidden_layers``
        so every drafter layer pulls K/V from the target model's cache, and
        ``layer_types`` defaults to ``["sliding_attention"] * (L-1) +
        ["full_attention"]`` to match the published HF assistants. The
        ``global_head_dim`` field is mirrored onto ``self`` so the underlying
        Gemma4 attention layer can read the larger head dim for the global
        layer.

        Args:
            **kwargs: Forwarded to :class:`Gemma4TextConfig`.
        """
        if layer_types is None:
            layer_types = ["sliding_attention"] * (num_hidden_layers - 1) + ["full_attention"]
        if num_kv_shared_layers is None:
            num_kv_shared_layers = num_hidden_layers
        if rope_parameters is None:
            rope_parameters = {
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
            }
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            layer_types=layer_types,
            sliding_window=sliding_window,
            num_kv_shared_layers=num_kv_shared_layers,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_parameters=rope_parameters,
            tie_word_embeddings=tie_word_embeddings,
            attention_k_eq_v=attention_k_eq_v,
            **kwargs,
        )
        self.global_head_dim = global_head_dim if global_head_dim is not None else head_dim


@register_config("gemma4_assistant")
class Gemma4AssistantConfig(EasyDeLBaseConfig):
    """Top-level Gemma4 Assistant (drafter) config.

    The drafter operates as an HF-style ``assistant_model`` paired with
    a Gemma4 target. The target's hidden states + next-token embedding
    are fused via ``pre_projection`` to seed the drafter's input; the
    drafter's output projects back via ``post_projection`` for the
    feedback buffer and through the centroid head for sparse logits.

    Args:
        text_config: Drafter decoder sub-config. ``Gemma4AssistantTextConfig``
            or a dict.
        backbone_hidden_size: TARGET model's hidden size (2560 for
            Gemma4-E4B, 2816 for 26B, etc.). Drives
            ``pre_projection``'s input width (``2 * backbone_hidden_size``)
            and ``post_projection``'s output width.
        num_centroids: Number of centroid clusters in the output head.
            ``2048`` for E4B; ``None`` / 0 disables the centroid head
            (26B-assistant uses ``use_ordered_embeddings=False``).
        centroid_intermediate_top_k: Number of top centroids selected
            per token. ``32`` for E4B. Picks
            ``top_k * (vocab_size / num_centroids)`` candidate
            vocabulary IDs to score per position.
        use_ordered_embeddings: Whether to use the centroid head. When
            ``True``, the drafter scores only the top-k clustered
            tokens (sparse softmax); when ``False`` it falls back to
            standard full-vocab projection.
        image_token_id: Image placeholder token ID (Gemma4 default
            ``258880``).
        video_token_id: Reserved for symmetry; Gemma4 video tokens are
            audio-driven. Defaults to ``None``.
        audio_token_id, boa_token_id, eoa_token_id: Audio placeholder
            IDs (E4B drafter inherits the target's audio support).
        boi_token_id, eoi_token_id: Image span markers.
        tie_word_embeddings: Whether ``embed_tokens`` doubles as the
            LM head. Always ``True`` for the published assistants.
        **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
    """

    model_type = "gemma4_assistant"
    sub_configs: typing.ClassVar = {"text_config": Gemma4AssistantTextConfig}
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, typing.Any] | Gemma4AssistantTextConfig | None = None,
        backbone_hidden_size: int = 2560,
        num_centroids: int = 2048,
        centroid_intermediate_top_k: int = 32,
        use_ordered_embeddings: bool = True,
        image_token_id: int = 258880,
        video_token_id: int | None = None,
        audio_token_id: int | None = 258881,
        boa_token_id: int | None = 256000,
        eoa_token_id: int | None = 258883,
        boi_token_id: int | None = 255999,
        eoi_token_id: int | None = 258882,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize a :class:`Gemma4AssistantConfig`.

        See the class docstring for a description of every field. Highlights:

        - ``text_config`` may be a :class:`Gemma4AssistantTextConfig`, a plain
          dict (unpacked into ``Gemma4AssistantTextConfig(**text_config)``), or
          ``None`` for the default 4-layer drafter geometry.
        - ``backbone_hidden_size`` must match the target model's hidden size
          since the drafter's ``pre_projection`` consumes
          ``2 * backbone_hidden_size`` features and ``post_projection`` writes
          back ``backbone_hidden_size``.
        - ``num_centroids`` of ``0`` (or ``None``) disables the centroid head
          and falls back to a standard full-vocab projection.
        - Special-token IDs (``image_token_id``, ``audio_token_id``, …) mirror
          the upstream Gemma4 tokenizer; pass through for compatibility.

        Args:
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.backbone_hidden_size = int(backbone_hidden_size)
        self.num_centroids = int(num_centroids) if num_centroids else 0
        self.centroid_intermediate_top_k = int(centroid_intermediate_top_k)
        self.use_ordered_embeddings = bool(use_ordered_embeddings)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> Gemma4AssistantTextConfig:
        """Return the embedded drafter text config."""
        return self.text_config


__all__ = ["Gemma4AssistantConfig", "Gemma4AssistantTextConfig"]
