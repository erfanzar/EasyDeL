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

from ._compute_fns import (
    apply_basic_rope,
    apply_phi3_rope,
    compute_basic_frequencies,
    compute_basic_inv_frequencies,
    compute_deepseek_frequencies,
    compute_dynamic_frequencies,
    compute_linear_frequencies,
    compute_llama3_frequencies,
    compute_llama3_inv_frequencies,
    compute_phi3_frequencies,
    compute_yarn_frequencies,
    compute_yarn_inv_frequencies,
)
from ._configs import RopeConfig
from ._modules import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MultiModalRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from ._rotary import get_frequencies, get_inv_frequencies, get_rope
from ._utils import yarn_get_mscale

__all__ = (
    "DeepseekScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "Llama3RotaryEmbedding",
    "MultiModalRotaryEmbedding",
    "Phi3LongRoPEScaledRotaryEmbedding",
    "RopeConfig",
    "RotaryEmbedding",
    "YaRNScalingRotaryEmbedding",
    "apply_basic_rope",
    "apply_phi3_rope",
    "compute_basic_frequencies",
    "compute_basic_inv_frequencies",
    "compute_deepseek_frequencies",
    "compute_dynamic_frequencies",
    "compute_linear_frequencies",
    "compute_llama3_frequencies",
    "compute_llama3_inv_frequencies",
    "compute_phi3_frequencies",
    "compute_yarn_frequencies",
    "compute_yarn_inv_frequencies",
    "compute_yarn_inv_frequencies",
    "get_frequencies",
    "get_inv_frequencies",
    "get_rope",
    "yarn_get_mscale",
)
