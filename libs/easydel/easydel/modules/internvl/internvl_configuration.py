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

"""Configuration for the InternVL family of vision-language conditional generators.

Defines :class:`InternVLVisionConfig` for the InternViT vision tower (a
BEiT/timm-style ViT with CLS token, learned absolute position embeddings,
layer-scale residual multipliers and optional full-width QK RMSNorm) and
:class:`InternVLConfig`, the composite config wrapping the vision tower and an
auto-resolved text-decoder config (Qwen2 by default; Llama variants exist)
plus the pixel-shuffle multimodal projector parameters.
"""

import typing

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config, registry

from ..auto import AutoEasyDeLConfig

logger = get_logger(__name__)


@register_config("internvl_vision")
class InternVLVisionConfig(EasyDeLBaseConfig):
    """Configuration for the InternViT vision tower used by InternVL.

    This mirrors HuggingFace's ``InternVLVisionConfig``: a timm-style ViT with
    a CLS token, learned absolute position embeddings, per-layer layer-scale
    multipliers (``lambda_1``/``lambda_2``), configurable layer/rms norms and
    optional full-width RMSNorm on the query/key projections.

    Args:
        hidden_size (int, optional): Encoder hidden width. Defaults to 1024.
        num_hidden_layers (int, optional): Number of encoder layers. Defaults to 24.
        num_attention_heads (int, optional): Attention heads per layer. Defaults to 16.
        attention_bias (bool, optional): Whether q/k/v projections carry a bias.
            Defaults to False.
        use_qk_norm (bool, optional): Whether to apply a full-width RMSNorm to the
            query/key projections before the head split. Defaults to False.
        intermediate_size (int, optional): Feed-forward width. Defaults to 4096.
        hidden_act (str, optional): MLP activation name. Defaults to "gelu".
        hidden_dropout_prob (float, optional): Hidden dropout probability.
            Defaults to 0.0.
        attention_dropout (float, optional): Attention dropout probability.
            Defaults to 0.0.
        projection_dropout (float, optional): Output-projection dropout probability.
            Defaults to 0.0.
        initializer_range (float, optional): Stddev for weight init. Defaults to 0.02.
        norm_type (str, optional): ``"layer_norm"`` or ``"rms_norm"`` for the
            per-layer norms. Defaults to "layer_norm".
        layer_norm_eps (float, optional): Epsilon for the per-layer norms.
            Defaults to 1e-6.
        image_size (int | tuple[int, int], optional): Input image resolution.
            Defaults to (448, 448).
        patch_size (int | tuple[int, int], optional): Patch resolution.
            Defaults to (14, 14).
        num_channels (int, optional): Input channels. Defaults to 3.
        use_mask_token (bool, optional): Whether a mask token exists (masked image
            modeling). Defaults to False.
        use_absolute_position_embeddings (bool, optional): Whether learned absolute
            position embeddings are added. Defaults to True.
        layer_scale_init_value (float, optional): Initial value of the layer-scale
            multipliers. Defaults to 0.1.
        use_mean_pooling (bool, optional): When True the final layernorm is the
            identity (mean pooling happens in downstream heads). Defaults to True.
        **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
    """

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        attention_bias: bool = False,
        use_qk_norm: bool = False,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        initializer_range: float = 0.02,
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        image_size: int | list[int] | tuple[int, ...] = (448, 448),
        patch_size: int | list[int] | tuple[int, ...] = (14, 14),
        num_channels: int = 3,
        use_mask_token: bool = False,
        use_absolute_position_embeddings: bool = True,
        layer_scale_init_value: float = 0.1,
        use_mean_pooling: bool = True,
        **kwargs,
    ):
        """Initialize the InternVL vision encoder configuration.

        Args:
            hidden_size (int, optional): Encoder hidden width.
            num_hidden_layers (int, optional): Number of encoder layers.
            num_attention_heads (int, optional): Attention heads per layer.
            attention_bias (bool, optional): Whether q/k/v projections carry a bias.
            use_qk_norm (bool, optional): Whether to RMS-normalize q/k projections.
            intermediate_size (int, optional): Feed-forward width.
            hidden_act (str, optional): MLP activation name.
            hidden_dropout_prob (float, optional): Hidden dropout probability.
            attention_dropout (float, optional): Attention dropout probability.
            projection_dropout (float, optional): Output-projection dropout.
            initializer_range (float, optional): Stddev for weight init.
            norm_type (str, optional): Per-layer norm flavor
                (``"layer_norm"`` or ``"rms_norm"``).
            layer_norm_eps (float, optional): Epsilon for the per-layer norms.
            image_size (int | list[int] | tuple[int, ...], optional): Input image
                resolution; ints are normalized to a square tuple.
            patch_size (int | list[int] | tuple[int, ...], optional): Patch
                resolution; ints are normalized to a square tuple.
            num_channels (int, optional): Input channels.
            use_mask_token (bool, optional): Whether a mask token exists.
            use_absolute_position_embeddings (bool, optional): Whether learned
                absolute position embeddings are added.
            layer_scale_init_value (float, optional): Initial layer-scale value.
            use_mean_pooling (bool, optional): Whether the final layernorm is
                replaced by the identity.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If ``norm_type`` is not ``"layer_norm"`` or ``"rms_norm"``.
        """
        if norm_type not in ("layer_norm", "rms_norm"):
            raise ValueError(f"norm_type should be one of 'layer_norm', 'rms_norm'. Got: {norm_type}")

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.initializer_range = initializer_range
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        # HF's model code indexes image_size/patch_size, so normalize to tuples.
        self.image_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.patch_size = tuple(patch_size) if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.layer_scale_init_value = layer_scale_init_value
        self.use_mean_pooling = use_mean_pooling

        super().__init__(**kwargs)


@register_config("internvl")
class InternVLConfig(EasyDeLBaseConfig):
    """Composite configuration for InternVL conditional generation.

    Wraps an :class:`InternVLVisionConfig` vision tower and an auto-resolved
    text-decoder config (Qwen2 by default; Llama variants exist) together with
    the pixel-shuffle downsampling ratio and MLP projector parameters, matching
    HuggingFace's ``InternVLConfig``.

    Args:
        vision_config (dict | EasyDeLBaseConfig | None, optional): Vision tower
            configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
            ``None`` for the InternViT defaults. Defaults to None.
        text_config (dict | EasyDeLBaseConfig | None, optional): Text decoder
            configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
            ``None`` for a Qwen2 default. Defaults to None.
        image_token_id (int, optional): Placeholder token id for image features.
            Defaults to 151667.
        image_seq_length (int, optional): Sequence length contributed by one image.
            Defaults to 256.
        downsample_ratio (float, optional): Pixel-shuffle downsampling factor.
            Defaults to 0.5.
        projector_hidden_act (str, optional): Projector activation. Defaults to "gelu".
        vision_feature_layer (int | list[int], optional): Vision hidden layer to
            extract features from (-1 selects the tower's final output).
            Defaults to -1.
        vision_feature_select_strategy (str, optional): ``"default"`` (drop CLS)
            or ``"full"``. Defaults to "default".
        **kwargs: Forwarded to :class:`EasyDeLBaseConfig` (including
            ``tie_word_embeddings``, which HF defaults to True for InternVL).
    """

    model_type = "internvl"
    sub_configs: typing.ClassVar = {"text_config": AutoEasyDeLConfig, "vision_config": AutoEasyDeLConfig}

    def __init__(
        self,
        vision_config: dict | EasyDeLBaseConfig | None = None,
        text_config: dict | EasyDeLBaseConfig | None = None,
        image_token_id: int = 151667,
        image_seq_length: int = 256,
        downsample_ratio: float = 0.5,
        projector_hidden_act: str = "gelu",
        vision_feature_layer: int | list[int] = -1,
        vision_feature_select_strategy: str = "default",
        **kwargs,
    ):
        """Initialize the InternVL composite configuration.

        Args:
            vision_config (dict | EasyDeLBaseConfig | None, optional): Vision tower
                configuration (dict, config instance, or None for defaults).
            text_config (dict | EasyDeLBaseConfig | None, optional): Text decoder
                configuration (dict, config instance, or None for a Qwen2 default).
            image_token_id (int, optional): Placeholder token id for image features.
            image_seq_length (int, optional): Tokens contributed by one image.
            downsample_ratio (float, optional): Pixel-shuffle downsampling factor.
            projector_hidden_act (str, optional): Projector activation name.
            vision_feature_layer (int | list[int], optional): Vision hidden layer
                index to extract features from.
            vision_feature_select_strategy (str, optional): ``"default"`` or
                ``"full"`` vision feature selection.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If ``vision_feature_select_strategy`` is not ``"default"``
                or ``"full"``.
        """
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "internvl_vision")
            vision_config = registry.get_config(vision_config["model_type"])(**vision_config)
        elif vision_config is None:
            vision_config = registry.get_config("internvl_vision")()

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = registry.get_config(text_config["model_type"])(**text_config)
        elif text_config is None:
            text_config = registry.get_config("qwen2")()

        self.text_config = text_config

        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
