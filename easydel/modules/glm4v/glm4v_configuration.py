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

import typing

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


def _rope_scaling_from_rope_parameters(
    rope_parameters: dict[str, typing.Any] | None,
    rope_scaling: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
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

    def get_partition_rules(self, *args, **kwargs):
        pmag = self.partition_manager
        return (
            (r"patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
            (r"patch_embed/proj/bias", pmag.resolve(Replicated)),
            (r"pos_embed/embedding", pmag.resolve(ColumnWise)),
            (r"blocks/.*/attn/qkv/kernel", pmag.resolve(ColumnWise)),
            (r"blocks/.*/attn/qkv/bias", pmag.resolve(Replicated)),
            (r"blocks/.*/attn/proj/kernel", pmag.resolve(RowWise)),
            (r"blocks/.*/attn/proj/bias", pmag.resolve(Replicated)),
            (r"blocks/.*/mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"blocks/.*/mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"blocks/.*/norm(1|2)/scale", pmag.resolve(Replicated)),
            (r"(post_conv_layernorm|post_layernorm)/scale", pmag.resolve(Replicated)),
            (r"downsample/kernel", pmag.resolve(ColumnWise)),
            (r"downsample/bias", pmag.resolve(Replicated)),
            (r"merger/proj/kernel", pmag.resolve(ColumnWise)),
            (r"merger/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"merger/down_proj/kernel", pmag.resolve(RowWise)),
            (r"merger/norm/.*", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


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

    def get_partition_rules(self, *args, **kwargs):
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"layers/.*/self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"layers/.*/self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"layers/.*/mlp/gate_up_proj/kernel", pmag.resolve(ColumnWise)),
            (r"layers/.*/mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"layers/.*/(input_layernorm|post_attention_layernorm)/scale", pmag.resolve(Replicated)),
            (r"layers/.*/(post_self_attn_layernorm|post_mlp_layernorm)/scale", pmag.resolve(Replicated)),
            (r"norm/scale", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
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
        text_config: typing.Mapping[str, typing.Any] | Glm4vTextConfig | None = None,
        vision_config: typing.Mapping[str, typing.Any] | Glm4vVisionConfig | None = None,
        image_token_id: int = 151343,
        video_token_id: int = 151344,
        image_start_token_id: int = 151339,
        image_end_token_id: int = 151340,
        video_start_token_id: int = 151341,
        video_end_token_id: int = 151342,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
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
        del decoder
        return self.text_config

    def get_vision_config(self) -> Glm4vVisionConfig:
        return self.vision_config

    def get_partition_rules(self, *args, **kwargs):
        pmag = self.partition_manager
        return (
            (r"model/visual/patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/patch_embed/proj/bias", pmag.resolve(Replicated)),
            (r"model/visual/pos_embed/embedding", pmag.resolve(ColumnWise)),
            (r"model/visual/blocks/.*/attn/qkv/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/blocks/.*/attn/proj/kernel", pmag.resolve(RowWise)),
            (r"model/visual/blocks/.*/mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/blocks/.*/mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"model/visual/downsample/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/merger/proj/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/merger/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"model/visual/merger/down_proj/kernel", pmag.resolve(RowWise)),
            (r"model/language_model/embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"model/language_model/layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"model/language_model/layers/.*/self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"model/language_model/layers/.*/mlp/gate_up_proj/kernel", pmag.resolve(ColumnWise)),
            (r"model/language_model/layers/.*/mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*(norm|layernorm).*", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


__all__ = ["Glm4vConfig", "Glm4vTextConfig", "Glm4vVisionConfig"]
