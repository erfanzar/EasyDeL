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

"""Configuration for the Aya Vision multimodal model.

Defines :class:`AyaVisionConfig` — a composite config bundling a vision tower
config (defaults to a SigLIP encoder) and a text decoder config (defaults to
Cohere2). Adds AyaVision-specific knobs: vision feature selection strategy
and layer, multi-modal projector downsample factor, adapter layer-norm
epsilon, and the image token index used to splice image features into the
text token stream.
"""

import typing

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config, registry

from ..auto import AutoEasyDeLConfig

logger = get_logger(__name__)


@register_config("aya_vision")
class AyaVisionConfig(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`AyaVisionForConditionalGeneration`]. It is used
    to instantiate an AyaVision model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of AyaVision.
    e.g. [CohereForAI/aya-vision-8b](https://huggingface.co/CohereForAI/aya-vision-8b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        downsample_factor (`int`, *optional*, defaults to 2):
            The downsample factor to apply to the vision features.
        adapter_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon value used for layer normalization in the adapter.
        image_token_index (`int`, *optional*, defaults to 255036):
            The image token index to encode the image prompt.
    """

    model_type = "aya_vision"
    attribute_map: typing.ClassVar = {"image_token_id": "image_token_index"}
    sub_configs: typing.ClassVar = {"text_config": AutoEasyDeLConfig, "vision_config": AutoEasyDeLConfig}

    def __init__(
        self,
        vision_config: dict | EasyDeLBaseConfig | None = None,
        text_config: dict | EasyDeLBaseConfig | None = None,
        vision_feature_select_strategy: str = "full",
        vision_feature_layer: int = -1,
        downsample_factor: int = 2,
        adapter_layer_norm_eps: float = 1e-6,
        image_token_index: int = 255036,
        **kwargs,
    ):
        """Initialize the AyaVision configuration.

        Args:
            vision_config (dict | EasyDeLBaseConfig | None, optional): Vision
                tower config or its dict form. ``None`` falls back to a default
                SigLIP encoder (``hidden_size=1152``, ``patch_size=14``,
                ``image_size=384``, ``num_hidden_layers=26``,
                ``num_attention_heads=14``).
            text_config (dict | EasyDeLBaseConfig | None, optional): Text
                decoder config or its dict form. ``None`` falls back to a
                default :class:`Cohere2Config`.
            vision_feature_select_strategy (str, optional): Vision feature
                selection. ``"default"`` strips the CLS token; ``"full"``
                keeps every patch token. Defaults to ``"full"``.
            vision_feature_layer (int, optional): Index of the vision encoder
                layer used as feature source. Defaults to ``-1`` (last layer).
            downsample_factor (int, optional): Spatial downsampling factor for
                the multi-modal projector's pixel-shuffle. Defaults to ``2``.
            adapter_layer_norm_eps (float, optional): Epsilon for the
                projector's layer normalization. Defaults to ``1e-6``.
            image_token_index (int, optional): Token id reserved for image
                placeholders in the text stream. Defaults to ``255036``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If ``vision_feature_select_strategy`` is not one of
                ``"default"`` or ``"full"``.
        """
        self.image_token_index = image_token_index
        self.downsample_factor = downsample_factor
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = registry.get_config(vision_config["model_type"])(**vision_config)
        elif vision_config is None:
            from ..siglip import SiglipVisionConfig

            vision_config = SiglipVisionConfig(
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=384,
                num_hidden_layers=26,
                num_attention_heads=14,
                vision_use_head=False,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = registry.get_config(text_config["model_type"])(**text_config)
        elif text_config is None:
            from ..cohere2 import Cohere2Config

            text_config = Cohere2Config()

        self.text_config = text_config

        super().__init__(**kwargs)
