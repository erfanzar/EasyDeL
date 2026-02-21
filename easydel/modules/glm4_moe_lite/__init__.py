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

"""GLM-4-MoE-Lite: lightweight MoE decoder with MLA attention.

This module implements the GLM-4 MoE Lite architecture with:
- Multi-head Latent Attention (MLA) using low-rank KV compression
- Grouped top-k MoE routing with shared experts
- Dense-to-sparse MLP schedule per layer
"""

from .glm4_moe_lite_configuration import Glm4MoeLiteConfig
from .modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM, Glm4MoeLiteModel

__all__ = ("Glm4MoeLiteConfig", "Glm4MoeLiteForCausalLM", "Glm4MoeLiteModel")
