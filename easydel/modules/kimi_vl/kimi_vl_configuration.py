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

"""Configuration classes for MoonshotAI Kimi-VL.

Mirrors https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct (trust_remote_code)
layout:
- `KimiVLConfig` holds `vision_config` (MoonViT) + `text_config` (DeepSeek-V3).
"""

from __future__ import annotations

import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

from ..deepseek_v3.deepseek_configuration import DeepseekV3Config


@register_config("moonvit")
class MoonViTConfig(EasyDeLBaseConfig):
    """Configuration for the MoonViT vision tower used by Kimi-VL."""

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

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for MoonViT vision tower.

        Returns:
            Tuple of partition rules for distributing MoonViT parameters.
        """
        pmag = self.partition_manager
        return (
            (r"patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
            (r"patch_embed/proj/bias", pmag.resolve(Replicated)),
            (r"patch_embed/pos_emb/kernel", pmag.resolve(Replicated)),
            (r"blocks/\d+/wqkv/kernel", pmag.resolve(ColumnWise)),
            (r"blocks/\d+/wqkv/bias", pmag.resolve(Replicated)),
            (r"blocks/\d+/wo/kernel", pmag.resolve(RowWise)),
            (r"blocks/\d+/wo/bias", pmag.resolve(Replicated)),
            (r"blocks/\d+/mlp/fc0/kernel", pmag.resolve(ColumnWise)),
            (r"blocks/\d+/mlp/fc0/bias", pmag.resolve(Replicated)),
            (r"blocks/\d+/mlp/fc1/kernel", pmag.resolve(RowWise)),
            (r"blocks/\d+/mlp/fc1/bias", pmag.resolve(Replicated)),
            (r"blocks/\d+/(norm0|norm1)/scale", pmag.resolve(Replicated)),
            (r"blocks/\d+/(norm0|norm1)/bias", pmag.resolve(Replicated)),
            (r"final_layernorm/scale", pmag.resolve(Replicated)),
            (r"final_layernorm/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


@register_config("kimi_vl")
class KimiVLConfig(EasyDeLBaseConfig):
    """Configuration for KimiVLForConditionalGeneration."""

    model_type = "kimi_vl"
    sub_configs: tp.ClassVar = {"vision_config": MoonViTConfig, "text_config": DeepseekV3Config}
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vision_config: tp.Mapping[str, tp.Any] | MoonViTConfig | None = None,
        text_config: tp.Mapping[str, tp.Any] | DeepseekV3Config | None = None,
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
        return self.text_config

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for the full KimiVL model.

        Combines partition rules from text_config (DeepSeek-V3) and vision_config (MoonViT),
        plus rules for the multi-modal projector.

        Returns:
            Tuple of partition rules for distributing all model parameters.
        """
        pmag = self.partition_manager
        tp_rules = (
            self.text_config.get_partition_rules(*args, **kwargs)
            if hasattr(self.text_config, "get_partition_rules")
            else ()
        )
        vp_rules = (
            self.vision_config.get_partition_rules(*args, **kwargs)
            if hasattr(self.vision_config, "get_partition_rules")
            else ()
        )
        projector_rules = (
            (r"vision_tower/.*", pmag.resolve(Replicated)),
            (r"multi_modal_projector/pre_norm/scale", pmag.resolve(Replicated)),
            (r"multi_modal_projector/pre_norm/bias", pmag.resolve(Replicated)),
            (r"multi_modal_projector/linear_1/kernel", pmag.resolve(ColumnWise)),
            (r"multi_modal_projector/linear_1/bias", pmag.resolve(Replicated)),
            (r"multi_modal_projector/linear_2/kernel", pmag.resolve(RowWise)),
            (r"multi_modal_projector/linear_2/bias", pmag.resolve(Replicated)),
        )
        return projector_rules + tp_rules + vp_rules


__all__ = ["KimiVLConfig", "MoonViTConfig"]
