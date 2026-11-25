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

"""Inkling 2 and Inkling 2 Loop model family package for EasyDeL."""

from .inkling2_configuration import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    Inkling2TextConfig,
    Inkling2VisionConfig,
)
from .modeling_inkling2 import (
    INKLING2_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_TEXT_KEY_RULES,
    INKLING2_LOOP_VLM_KEY_RULES,
    INKLING2_LOOP_VLM_MODEL_KEY_RULES,
    INKLING2_TEXT_KEY_RULES,
    INKLING2_VISION_KEY_RULES,
    INKLING2_VLM_KEY_RULES,
    INKLING2_VLM_MODEL_KEY_RULES,
    Inkling2Cache,
    Inkling2CacheConfig,
    Inkling2CacheView,
    Inkling2ForCausalLM,
    Inkling2ForConditionalGeneration,
    Inkling2KeyNormalizer,
    Inkling2LoopForCausalLM,
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopModel,
    Inkling2LoopTextModel,
    Inkling2Model,
    Inkling2ModelOutput,
    Inkling2TextModel,
    Inkling2VisionModel,
    Inkling2VisionModelOutput,
    create_inkling2_cache_config,
)


def _register_hf_auto_configs() -> None:
    """Register Inkling 2 configs with ``transformers.AutoConfig`` if not already shipped upstream."""
    try:
        from transformers import AutoConfig
    except Exception:
        return
    for model_type, cfg_cls in (
        ("inkling2_text", Inkling2TextConfig),
        ("inkling2_vision", Inkling2VisionConfig),
        ("inkling2", Inkling2Config),
        ("inkling2_loop_text", Inkling2LoopTextConfig),
        ("inkling2_loop", Inkling2LoopConfig),
    ):
        try:
            AutoConfig.register(model_type, cfg_cls, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type, cfg_cls)
            except ValueError:
                pass
        except ValueError:
            pass


_register_hf_auto_configs()

__all__ = [
    "INKLING2_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_TEXT_KEY_RULES",
    "INKLING2_LOOP_VLM_KEY_RULES",
    "INKLING2_LOOP_VLM_MODEL_KEY_RULES",
    "INKLING2_TEXT_KEY_RULES",
    "INKLING2_VISION_KEY_RULES",
    "INKLING2_VLM_KEY_RULES",
    "INKLING2_VLM_MODEL_KEY_RULES",
    "Inkling2Cache",
    "Inkling2CacheConfig",
    "Inkling2CacheView",
    "Inkling2Config",
    "Inkling2ForCausalLM",
    "Inkling2ForConditionalGeneration",
    "Inkling2KeyNormalizer",
    "Inkling2LoopConfig",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopModel",
    "Inkling2LoopTextConfig",
    "Inkling2LoopTextModel",
    "Inkling2Model",
    "Inkling2ModelOutput",
    "Inkling2TextConfig",
    "Inkling2TextModel",
    "Inkling2VisionConfig",
    "Inkling2VisionModel",
    "Inkling2VisionModelOutput",
    "create_inkling2_cache_config",
]
