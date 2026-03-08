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

"""Configuration classes for MoonshotAI Kimi-VL.

Mirrors https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct (trust_remote_code)
layout:
- `KimiVLConfig` holds `vision_config` (MoonViT) + `text_config` (DeepSeek-V3).
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

from ..deepseek_v3.deepseek_configuration import DeepseekV3Config


@register_config("moonvit")
class MoonViTConfig(EasyDeLBaseConfig):
    """Configuration for the MoonViT vision encoder used in Kimi-VL.

    MoonViT is a vision transformer that processes images into patch embeddings
    with learnable 2D positional embeddings and a spatial merge step that
    downsamples the patch grid.

    Args:
        patch_size (`int`, *optional*, defaults to 14):
            Size of each image patch for the vision transformer.
        init_pos_emb_height (`int`, *optional*, defaults to 64):
            Initial height of the 2D positional embedding grid.
        init_pos_emb_width (`int`, *optional*, defaults to 64):
            Initial width of the 2D positional embedding grid.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the vision encoder.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of transformer layers in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the hidden layers in the vision encoder.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Dimensionality of the MLP intermediate layer.
        merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
            Kernel size for spatial merge (downsampling) of patch features.
    """

    model_type = "moonvit"
    base_config_key = "vision_config"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] | list[int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)

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


@register_config("kimi_vl")
class KimiVLConfig(EasyDeLBaseConfig):
    """Top-level configuration for the Kimi-VL vision-language model.

    Combines a ``MoonViTConfig`` vision encoder with a ``DeepseekV3Config`` text
    decoder. The vision encoder output is projected into the text decoder's
    embedding space via a multi-modal projector.

    See: `moonshotai/Kimi-VL-A3B-Instruct <https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct>`_

    Args:
        vision_config (`dict` or `MoonViTConfig`, *optional*):
            Configuration for the MoonViT vision encoder. Defaults to
            ``MoonViTConfig()`` if not provided.
        text_config (`dict` or `DeepseekV3Config`, *optional*):
            Configuration for the DeepSeek-V3 text decoder. Defaults to
            ``DeepseekV3Config()`` if not provided.
        ignore_index (`int`, *optional*, defaults to -100):
            Label index to ignore in the cross-entropy loss.
        media_placeholder_token_id (`int`, *optional*, defaults to 163605):
            Token index used as a placeholder for image embeddings in the input.
        pad_token_id (`int`, *optional*, defaults to 0):
            Index of the padding token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output word embeddings.
    """

    model_type = "kimi_vl"
    sub_configs: tp.ClassVar = {"vision_config": MoonViTConfig, "text_config": DeepseekV3Config}
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vision_config: Mapping[str, tp.Any] | MoonViTConfig | None = None,
        text_config: Mapping[str, tp.Any] | DeepseekV3Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
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

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.vocab_size = getattr(self.text_config, "vocab_size", None)

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, decoder: bool = True) -> DeepseekV3Config:
        return self.text_config  # type: ignore

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


__all__ = ["KimiVLConfig", "MoonViTConfig"]
