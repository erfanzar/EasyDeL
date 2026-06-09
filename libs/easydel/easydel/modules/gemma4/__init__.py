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

"""Gemma4 model implementation for EasyDeL.

Gemma4 extends Gemma3 with:
- Per-layer input embeddings
- KV sharing across layers
- Mixture of Experts (MoE) blocks
- V-normalization (without learnable scale)
- Key-equals-value attention (k_eq_v)
- Different head dimensions for global vs local layers
- Double-wide MLP for KV-shared layers
- Per-layer scalar weighting
"""

from .gemma4_configuration import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from .modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4Model,
    Gemma4MultimodalEmbedder,
    Gemma4RMSNorm,
    Gemma4TextModel,
    Gemma4VisionModel,
)

__all__ = (
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4Model",
    "Gemma4MultimodalEmbedder",
    "Gemma4RMSNorm",
    "Gemma4TextConfig",
    "Gemma4TextModel",
    "Gemma4VisionConfig",
    "Gemma4VisionModel",
)
