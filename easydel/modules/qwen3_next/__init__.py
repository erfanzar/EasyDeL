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

"""Qwen3Next - Hybrid attention model with GatedDeltaRule and MoE.

This module implements the Qwen3Next architecture, which combines:
- Full attention layers with sigmoid gating and partial RoPE
- Linear attention layers using GatedDeltaNet
- MoE FFN with routed and shared experts

Example:
    >>> from easydel.modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
    >>> config = Qwen3NextConfig(
    ...     hidden_size=2048,
    ...     num_hidden_layers=24,
    ...     num_attention_heads=16,
    ... )
    >>> model = Qwen3NextForCausalLM(config=config, dtype=jnp.bfloat16)
"""

from easydel.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextModel,
)
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

__all__ = (
    "Qwen3NextConfig",
    "Qwen3NextForCausalLM",
    "Qwen3NextModel",
)
