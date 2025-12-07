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

"""Kimi Linear model implementation.

Kimi Linear is a hybrid attention architecture from Moonshot AI that combines:
- MLA (Multi-Latent Attention) for full attention layers (like DeepSeek V3)
- KDA (Kernel Delta Attention) for linear attention layers
- MoE (Mixture of Experts) with sigmoid routing and shared experts

References:
    - Kimi Linear: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
"""

from .kimi_linear_configuration import KimiLinearConfig
from .modeling_kimi_linear import KimiLinearForCausalLM, KimiLinearModel

__all__ = ("KimiLinearConfig", "KimiLinearForCausalLM", "KimiLinearModel")
