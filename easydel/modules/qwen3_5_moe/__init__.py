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

"""Qwen3.5-MoE text and multimodal model family for EasyDeL.

Qwen3.5-MoE is the Mixture-of-Experts evolution of Qwen3.5. It pairs the
Qwen3-Next hybrid attention/linear-attention decoder (with sparse routed
MoE FFN blocks) with the Qwen3-VL-MoE vision tower for multimodal inputs.

This package re-exports the configuration classes and the registered model
wrappers used by ``AutoEasyDeLConfig`` / ``AutoEasyDeLModel``:

- :class:`Qwen3_5MoeConfig` — composite multimodal config (text + vision
  sub-configs plus image/video token ids).
- :class:`Qwen3_5MoeTextConfig` — MoE text-decoder hyperparameters
  (hybrid attention schedule, expert count, routing).
- :class:`Qwen3_5MoeVisionConfig` — Qwen3-VL-MoE vision encoder config.
- :class:`Qwen3_5MoeTextModel` — text-only base transformer (no LM head).
- :class:`Qwen3_5MoeForCausalLM` — text-only causal LM wrapper.
- :class:`Qwen3_5MoeModel` — multimodal base model fusing vision and text.
- :class:`Qwen3_5MoeForConditionalGeneration` — image/video-conditioned
  generation wrapper with LM head and MoE auxiliary loss support.
"""

from .modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextModel,
)
from .qwen3_5_moe_configuration import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)

__all__ = (
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeTextModel",
    "Qwen3_5MoeVisionConfig",
)
