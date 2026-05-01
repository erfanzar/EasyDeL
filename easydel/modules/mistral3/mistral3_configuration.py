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


import typing as tp

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config, registry


@register_config("mistral3")
class Mistral3Config(EasyDeLBaseConfig):
    """Composite configuration for Mistral-3 / Pixtral vision-language models.

    Mistral-3 pairs a Pixtral vision tower (aspect-ratio aware ViT, see
    ``vision_config``) with the Mistral-Nemo / Mistral-Small text decoder
    (``text_config``). Image patches are concatenated by the projector via
    a ``spatial_merge_size``-by-``spatial_merge_size`` spatial-to-channel
    fold (default 2 collapses each ``2x2`` patch group into a single
    token), which is what allows the model to process *non-square* images
    without padding to a fixed grid. ``image_token_index`` is the
    placeholder text token whose positions are overwritten with projected
    vision tokens at call time. ``layer_types`` controls per-layer
    attention dispatch on the text side and is propagated automatically
    to the wrapped text decoder when the value is left ``None``.

    Composition flags:
        * ``sub_configs`` declares the two embedded configs so the EasyDeL
          factory knows how to recursively serialize/deserialize them.
        * ``is_composition = True`` opts this config out of the standard
          field flattening.
        * ``attribute_map`` aliases the HuggingFace name
          ``image_token_id`` to the internal ``image_token_index``.

    Attributes:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `PixtralVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MistralConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 10):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        multimodal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multimodal projector.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The downsampling factor for the spatial merge operation.
    """

    model_type = "mistral3"
    attribute_map: tp.ClassVar = {"image_token_id": "image_token_index"}
    sub_configs: tp.ClassVar = {"text_config": EasyDeLBaseConfig, "vision_config": EasyDeLBaseConfig}
    is_composition = True

    def __init__(
        self,
        vision_config: dict | EasyDeLBaseConfig | None = None,
        text_config: dict | EasyDeLBaseConfig | None = None,
        image_token_index: int = 10,
        projector_hidden_act: str = "gelu",
        vision_feature_layer: int | list[int] = -1,
        multimodal_projector_bias: bool = False,
        spatial_merge_size: int = 2,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the Mistral3 vision-language configuration.

        Args:
            vision_config (dict | EasyDeLBaseConfig | None, optional): Vision encoder
                configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
                ``None`` for a Pixtral default. Defaults to None.
            text_config (dict | EasyDeLBaseConfig | None, optional): Text decoder
                configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
                ``None`` for a Mistral default. Defaults to None.
            image_token_index (int, optional): Token id used as a placeholder for image
                features. Defaults to 10.
            projector_hidden_act (str, optional): Activation used by the multimodal
                projector. Defaults to "gelu".
            vision_feature_layer (int | list[int], optional): Index (or indices) of the
                vision encoder hidden layer(s) to extract features from. Defaults to -1.
            multimodal_projector_bias (bool, optional): Whether the projector uses
                bias. Defaults to False.
            spatial_merge_size (int, optional): Downsampling factor of the spatial
                merge step (e.g. 2 collapses each 2x2 patch group). Defaults to 2.
            layer_types (list[str] | None, optional): Per-layer attention type. If
                ``None``, defaults to ``"full_attention"`` for every text decoder
                layer. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to ``EasyDeLBaseConfig``.
        """
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config["model_type"] if "model_type" in vision_config else "pixtral"
            vision_config = registry.get_config(vision_config["model_type"])(**vision_config)
        elif vision_config is None:
            vision_config = registry.get_config("pixtral")(
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=1540,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                head_dim=64,
                hidden_act="gelu",
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "mistral"
            text_config = registry.get_config(text_config["model_type"])(**text_config)
        elif text_config is None:
            text_config = registry.get_config("mistral")(
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=5120,
                initializer_range=0.02,
                intermediate_size=32768,
                max_position_embeddings=131072,
                model_type="mistral",
                num_attention_heads=32,
                num_hidden_layers=40,
                num_key_value_heads=8,
                rms_norm_eps=1e-05,
                rope_theta=1000000000.0,
                sliding_window=None,
                use_cache=True,
                vocab_size=131072,
            )

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size
        self.layer_types = layer_types
        if self.layer_types is None and hasattr(self.text_config, "num_hidden_layers"):
            self.layer_types = ["full_attention"] * self.text_config.num_hidden_layers
