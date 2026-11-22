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

"""Configuration for the PaliGemma vision-language conditional generators.

Defines :class:`PaliGemmaConfig`, a composite config wrapping a SigLIP vision
tower config and a Gemma text-decoder config plus the single-linear multimodal
projector width (``projection_dim``). Sub-configs are auto-resolved through
the EasyDeL registry, so PaliGemma 1 (Gemma text) and PaliGemma 2 (Gemma 2
text) checkpoints both route through this class via ``model_type``.
"""

import typing

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config, registry

from ..auto import AutoEasyDeLConfig

logger = get_logger(__name__)


@register_config("paligemma")
class PaliGemmaConfig(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`PaliGemmaForConditionalGeneration`].
    It is used to instantiate a PaliGemma model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults yields a configuration similar to
    [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224).

    PaliGemma composes a SigLIP vision tower with a Gemma text decoder through a single linear
    multimodal projector. Image tokens and the text prefix attend bidirectionally while the suffix
    remains causal (driven by ``token_type_ids`` at runtime).

    Args:
        vision_config (`Union[EasyDeLBaseConfig, dict]`, *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[EasyDeLBaseConfig, dict]`, *optional*, defaults to `GemmaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index used as the placeholder for image features.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size used when constructing the default text config.
        projection_dim (`int`, *optional*, defaults to 2048):
            Output dimension of the multimodal projector (must match the text hidden size).
        hidden_size (`int`, *optional*, defaults to 2048):
            Hidden size of the text backbone (kept for HF config compatibility).
    """

    model_type = "paligemma"
    attribute_map: typing.ClassVar = {"image_token_id": "image_token_index"}
    sub_configs: typing.ClassVar = {"text_config": AutoEasyDeLConfig, "vision_config": AutoEasyDeLConfig}

    def __init__(
        self,
        vision_config: dict | EasyDeLBaseConfig | None = None,
        text_config: dict | EasyDeLBaseConfig | None = None,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        **kwargs,
    ):
        """Initialize the PaliGemma configuration.

        Args:
            vision_config (dict | EasyDeLBaseConfig | None, optional): Vision encoder
                configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
                ``None`` for the SigLIP So400m/14 224px default. Defaults to None.
            text_config (dict | EasyDeLBaseConfig | None, optional): Text decoder
                configuration. Accepts a dict, an ``EasyDeLBaseConfig`` instance, or
                ``None`` for the Gemma-2B default. Defaults to None.
            image_token_index (int, optional): Token id used as a placeholder for image
                features. Defaults to 256000.
            vocab_size (int, optional): Vocabulary size for the default text config.
                Defaults to 257152.
            projection_dim (int, optional): Multimodal projector output width.
                Defaults to 2048.
            hidden_size (int, optional): Text backbone hidden size (HF compatibility
                field). Defaults to 2048.
            **kwargs: Additional keyword arguments forwarded to ``EasyDeLBaseConfig``
                (e.g. ``tie_word_embeddings``, which defaults to True as in HF).
        """
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = registry.get_config(vision_config["model_type"])(**vision_config)
        elif vision_config is None:
            vision_config = registry.get_config("siglip_vision_model")(
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )
        self.vision_config = vision_config
        # The projector maps vision hidden states to `projection_dim` (HF stores it on the
        # vision sub-config; PaliGemma requires it to equal the text hidden size).
        self.vision_config.projection_dim = projection_dim

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "gemma")
            text_config = registry.get_config(text_config["model_type"])(**text_config)
        elif text_config is None:
            text_config = registry.get_config("gemma")(
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                vocab_size=self.vocab_size,
            )
        self.text_config = text_config
        # Mirrors HF PaliGemmaConfig.__post_init__ bookkeeping (unused by the forward path).
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2

        super().__init__(**kwargs)
