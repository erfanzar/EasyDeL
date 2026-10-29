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

"""MiniMax M3 VL model family (composite VLM + self-contained text decoder)."""

from .minimax_m3_vl_configuration import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig, MiniMaxM3VLVisionConfig
from .modeling_minimax_m3_vl import (
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLForConditionalGeneration,
    MiniMaxM3VLModel,
    MiniMaxM3VLTextModel,
    MiniMaxM3VLVisionModel,
)

__all__ = [
    "MiniMaxM3VLConfig",
    "MiniMaxM3VLForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLTextConfig",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLVisionConfig",
    "MiniMaxM3VLVisionModel",
]
