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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma4 Assistant (MTP drafter) — speculative-decoding draft model.

Standalone small drafter model that pairs with a Gemma4 target via
Hugging Face's ``assistant_model=`` speculative-decoding API. Unlike
Qwen3.5's inline DeepSeek-V3-style MTP head, the Gemma4 assistant is
a separate model with:

- A small 4-layer Gemma4-style decoder (Q-only self-attention; K/V
  pulled from the target model's KV cache at runtime via the
  speculative-decoding controller).
- ``pre_projection`` that fuses the target's hidden state with the
  target's next-token embedding (both at ``backbone_hidden_size``).
- ``post_projection`` that maps the assistant's draft hidden back to
  ``backbone_hidden_size`` for the per-step feedback buffer.
- A centroid-clustered output head (``num_centroids=2048``,
  ``centroid_intermediate_top_k=32``) that scores only ~4096
  candidate tokens per step instead of the full 262K vocab.

See :class:`Gemma4AssistantForCausalLM` for the standalone draft
interface and the README of ``google/gemma-4-*-it-assistant`` for the
overall MTP design.
"""

from .configuration_gemma4_assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantTextConfig,
)
from .modeling_gemma4_assistant import (
    Gemma4AssistantCentroidHead,
    Gemma4AssistantForCausalLM,
    Gemma4AssistantModel,
    Gemma4AssistantOutput,
)

__all__ = (
    "Gemma4AssistantCentroidHead",
    "Gemma4AssistantConfig",
    "Gemma4AssistantForCausalLM",
    "Gemma4AssistantModel",
    "Gemma4AssistantOutput",
    "Gemma4AssistantTextConfig",
)
