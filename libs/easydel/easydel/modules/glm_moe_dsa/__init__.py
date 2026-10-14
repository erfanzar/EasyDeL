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

"""GLM-MoE-DSA: MoE decoder with MLA and dynamic sparse attention.

This module implements the GLM-MoE-DSA architecture with:
- Multi-head Latent Attention (MLA) using low-rank KV compression
- Dynamic sparse attention indexer with top-k token selection
- Grouped top-k MoE routing with shared experts
- Dense-to-sparse MLP schedule per layer
"""

from .glm_moe_dsa_configuration import GlmMoeDsaConfig
from .modeling_glm_moe_dsa import GlmMoeDsaForCausalLM, GlmMoeDsaModel

__all__ = ("GlmMoeDsaConfig", "GlmMoeDsaForCausalLM", "GlmMoeDsaModel")
