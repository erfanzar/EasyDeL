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

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.modules.glm4v.glm4v_configuration import Glm4vTextConfig, Glm4vVisionConfig


@register_config("glm46v")
class Glm46VConfig(EasyDeLBaseConfig):
    """
    Configuration class for the GLM-4.6V multimodal vision-language model.

    GLM-4.6V is an enhanced version of GLM4V that combines a vision encoder with a
    language model decoder for tasks like image understanding, visual question answering,
    and image-based conversation. It reuses the `Glm4vVisionConfig` and `Glm4vTextConfig`
    sub-configurations.

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
        video_start_token_id (`int`, *optional*, defaults to 151361):
            Token ID marking the start of a video sequence.
        video_end_token_id (`int`, *optional*, defaults to 151362):
            Token ID marking the end of a video sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.

    Example:
        ```python
        from easydel.modules.glm46v import Glm46VConfig

        # Load from pretrained
        config = Glm46VConfig.from_pretrained("THUDM/GLM-4.6V-9B")

        # Create custom config
        config = Glm46VConfig(
            text_config={"hidden_size": 4096, "num_hidden_layers": 40},
            vision_config={"hidden_size": 1536, "depth": 24},
        )
        ```
    """

    model_type = "glm46v"
    sub_configs: typing.ClassVar = {"vision_config": Glm4vVisionConfig, "text_config": Glm4vTextConfig}
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: typing.Mapping[str, typing.Any] | Glm4vTextConfig | None = None,
        vision_config: typing.Mapping[str, typing.Any] | Glm4vVisionConfig | None = None,
        image_token_id: int = 151343,
        video_token_id: int = 151344,
        image_start_token_id: int = 151339,
        image_end_token_id: int = 151340,
        video_start_token_id: int = 151361,
        video_end_token_id: int = 151362,
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

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding.

        Providing explicit partition rules is preferred over automatic sharding resolution,
        as it gives full control over parameter distribution across the device mesh.
        Returns ``None`` by default, which triggers automatic sharding via
        module-level ``craft_sharding`` hooks.

        Returns:
            Partition rules as ``tuple[tuple[str, PartitionSpec], ...] | None``.
        """
        return None


__all__ = ["Glm46VConfig"]
