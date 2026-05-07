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

"""Configuration classes for the GLM-4V vision-language model.

This module exposes three :class:`EasyDeLBaseConfig` subclasses and one
helper:

- :class:`Glm4vVisionConfig`: vision-tower hyper-params (patch size,
  spatial / temporal merge factors, projector width).
- :class:`Glm4vTextConfig`: text-decoder hyper-params (GQA, partial RoPE,
  multi-dimensional RoPE sections).
- :class:`Glm4vConfig`: top-level VLM config that composes the two
  sub-configs and stores the special-token ids that mark image and video
  spans inside the text stream.
- :func:`_rope_scaling_from_rope_parameters`: normalises HF ``rope_parameters``
  into EasyDeL ``rope_scaling`` mappings, defaulting to GLM-4V's mRoPE
  ``[8, 12, 12]`` sections.
"""

import typing
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


def _rope_scaling_from_rope_parameters(
    rope_parameters: dict[str, typing.Any] | None,
    rope_scaling: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    """Normalise HF ``rope_parameters`` into EasyDeL ``rope_scaling`` for GLM-4V.

    When ``rope_scaling`` is supplied it wins (a stray ``type`` key is
    aliased to ``rope_type``). Otherwise the relevant subset of
    ``rope_parameters`` is copied out, including the GLM-4V-specific
    ``mrope_section`` and ``mrope_interleaved`` fields. When neither is
    supplied, defaults to the GLM-4V mRoPE preset
    ``{"rope_type": "default", "mrope_section": [8, 12, 12]}``.

    Args:
        rope_parameters: HF-style RoPE parameters mapping or ``None``.
        rope_scaling: EasyDeL RoPE scaling mapping or ``None``.

    Returns:
        A normalised ``rope_scaling`` mapping (never ``None`` for GLM-4V
        since a default is always emitted when both inputs are missing).
    """
    if rope_scaling is not None:
        # HF sometimes uses "type" instead of "rope_type"
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]
        return rope_scaling

    if rope_parameters is None:
        # Default mRoPE for GLM4V-family models.
        # Matches upstream GLM4V-family configs (e.g. THUDM/GLM-4.1V-9B-Thinking).
        return {"rope_type": "default", "mrope_section": [8, 12, 12]}

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
        "mrope_section",
        "mrope_interleaved",
    ):
        if key in rope_parameters:
            rope_scaling_out[key] = rope_parameters[key]
    return rope_scaling_out


@register_config("glm4v_vision")
class Glm4vVisionConfig(EasyDeLBaseConfig):
    """
    Configuration class for the GLM4V vision encoder.

    This class stores the configuration for the vision transformer component of GLM4V,
    which processes image and video inputs.

    Args:
        depth (`int`, *optional*, defaults to 24):
            Number of transformer layers in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder hidden states.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in each transformer layer.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels (RGB = 3).
        image_size (`int`, *optional*, defaults to 336):
            Input image resolution.
        patch_size (`int`, *optional*, defaults to 14):
            Size of each image patch for the patch embedding.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Factor for spatial downsampling of visual features.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Temporal patch size for video processing.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            Output projection dimension to match the language model.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimensionality of the MLP intermediate layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
    """

    model_type = "glm4v_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1536,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_heads: int = 12,
        in_channels: int = 3,
        image_size: int = 336,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        intermediate_size: int = 13696,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize the GLM-4V vision encoder configuration.

        Args:
            depth: Number of vision-transformer layers.
            hidden_size: Vision encoder model dim.
            hidden_act: Activation used inside the vision MLPs.
            attention_bias: Whether vision attention projections carry biases.
            attention_dropout: Vision attention dropout probability.
            num_heads: Vision attention heads (also used for ``num_attention_heads``).
            in_channels: Input image channel count (3 for RGB).
            image_size: Square input image side length in pixels.
            patch_size: Patch side length used by the patch embedder.
            rms_norm_eps: Vision RMSNorm epsilon.
            spatial_merge_size: Spatial downsampling factor of the merger.
            temporal_patch_size: Temporal patch length for video inputs.
            out_hidden_size: Width to which the merger projects features
                before they are concatenated into the text stream.
            intermediate_size: Inner width of the vision MLP.
            initializer_range: Stddev for truncated-normal init.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range


@register_config("glm4v_text")
class Glm4vTextConfig(EasyDeLBaseConfig):
    """
    Configuration class for the GLM4V text decoder.

    This class stores the configuration for the language model component of GLM4V,
    which generates text responses based on visual and textual inputs.

    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden states.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimensionality of the MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key-value heads for grouped-query attention (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length the model can handle.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use key-value cache for generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in attention layers.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Fraction of head dimension to apply rotary embeddings.
        rope_theta (`float`, *optional*):
            Base frequency for rotary position embeddings.
        rope_scaling (`dict`, *optional*):
            Configuration for RoPE scaling (e.g., for extended context).
        rope_parameters (`dict`, *optional*):
            Alternative RoPE configuration format (converted to rope_scaling).
    """

    model_type = "glm4v_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 2,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        partial_rotary_factor: float = 0.5,
        rope_theta: float | None = None,
        rope_scaling: dict[str, typing.Any] | None = None,
        rope_parameters: dict[str, typing.Any] | None = None,
        **kwargs,
    ):
        """Initialize the GLM-4V text-decoder configuration.

        Args:
            vocab_size: Token vocabulary size.
            hidden_size: Decoder model dim.
            intermediate_size: Inner width of the gated MLP.
            num_hidden_layers: Decoder block count.
            num_attention_heads: Total query heads per attention layer.
            num_key_value_heads: KV heads for grouped-query attention;
                falls back to ``num_attention_heads`` when ``None``.
            head_dim: Per-head dim; falls back to ``hidden_size //
                num_attention_heads`` when ``None``.
            hidden_act: Activation applied to the gate half of the MLP.
            max_position_embeddings: Context-length bound used for RoPE
                tables and asserts.
            initializer_range: Stddev for truncated-normal weight init.
            rms_norm_eps: Epsilon for every RMSNorm.
            use_cache: Whether downstream code should return KV cache.
            tie_word_embeddings: Tie input embeddings with the LM head.
            attention_dropout: Attention probability dropout.
            attention_bias: Whether attention projections carry biases.
            partial_rotary_factor: Fraction of each head dim that
                receives RoPE; remaining channels are unrotated.
            rope_theta: RoPE base frequency. Falls back to
                ``rope_parameters["rope_theta"]`` and finally ``10000.0``.
            rope_scaling: EasyDeL-flavoured RoPE scaling dict.
            rope_parameters: HF-style RoPE parameters (merged via
                :func:`_rope_scaling_from_rope_parameters`).
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.partial_rotary_factor = partial_rotary_factor

        if rope_theta is None and rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.rope_theta = 10000.0 if rope_theta is None else float(rope_theta)
        self.rope_scaling = _rope_scaling_from_rope_parameters(rope_parameters, rope_scaling)

        self._external_rope_config_kwargs = {"repetition_style": True}

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@register_config("glm4v")
class Glm4vConfig(EasyDeLBaseConfig):
    """
    Configuration class for the GLM4V multimodal vision-language model.

    GLM4V is a multimodal model that combines a vision encoder with a language model
    decoder for tasks like image understanding, visual question answering, and
    image-based conversation.

    Args:
        text_config (`dict` or `Glm4vTextConfig`, *optional*):
            Configuration for the text decoder. If a dict is provided, it will be
            converted to `Glm4vTextConfig`.
        vision_config (`dict` or `Glm4vVisionConfig`, *optional*):
            Configuration for the vision encoder. If a dict is provided, it will be
            converted to `Glm4vVisionConfig`.
        image_token_id (`int`, *optional*, defaults to 151343):
            Token ID used to represent image placeholders in the input.
        video_token_id (`int`, *optional*, defaults to 151344):
            Token ID used to represent video placeholders in the input.
        image_start_token_id (`int`, *optional*, defaults to 151339):
            Token ID marking the start of an image sequence.
        image_end_token_id (`int`, *optional*, defaults to 151340):
            Token ID marking the end of an image sequence.
        video_start_token_id (`int`, *optional*, defaults to 151341):
            Token ID marking the start of a video sequence.
        video_end_token_id (`int`, *optional*, defaults to 151342):
            Token ID marking the end of a video sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.

    Example:
        ```python
        from easydel.modules.glm4v import Glm4vConfig

        # Load from pretrained
        config = Glm4vConfig.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

        # Create custom config
        config = Glm4vConfig(
            text_config={"hidden_size": 4096, "num_hidden_layers": 40},
            vision_config={"hidden_size": 1536, "depth": 24},
        )
        ```
    """

    model_type = "glm4v"
    sub_configs: typing.ClassVar = {
        "vision_config": Glm4vVisionConfig,
        "text_config": Glm4vTextConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, typing.Any] | Glm4vTextConfig | None = None,
        vision_config: Mapping[str, typing.Any] | Glm4vVisionConfig | None = None,
        image_token_id: int = 151343,
        video_token_id: int = 151344,
        image_start_token_id: int = 151339,
        image_end_token_id: int = 151340,
        video_start_token_id: int = 151341,
        video_end_token_id: int = 151342,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the top-level GLM-4V configuration.

        Args:
            text_config: Mapping or :class:`Glm4vTextConfig` for the text
                decoder; ``None`` materialises the default text config.
            vision_config: Mapping or :class:`Glm4vVisionConfig` for the
                vision encoder; ``None`` materialises the default vision
                config.
            image_token_id: Placeholder token id used for image patches.
            video_token_id: Placeholder token id used for video patches.
            image_start_token_id: Token id marking the start of an image
                sequence in the text stream.
            image_end_token_id: Token id marking the end of an image
                sequence.
            video_start_token_id: Token id marking the start of a video
                sequence.
            video_end_token_id: Token id marking the end of a video
                sequence.
            tie_word_embeddings: Whether to tie input embeddings with the
                LM head.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> Glm4vTextConfig:
        """Return the text decoder configuration.

        Args:
            decoder (bool, optional): Unused; kept for API compatibility with
                upstream Hugging Face configs. Defaults to True.

        Returns:
            Glm4vTextConfig: The text sub-config.
        """
        del decoder
        return self.text_config  # pyright: ignore[reportReturnType]

    def get_vision_config(self) -> Glm4vVisionConfig:
        """Return the vision encoder configuration.

        Returns:
            Glm4vVisionConfig: The vision sub-config.
        """
        return self.vision_config  # type: ignore


__all__ = ["Glm4vConfig", "Glm4vTextConfig", "Glm4vVisionConfig"]
