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

"""Rotary Position Embeddings (RoPE) module for EasyDeL.

This package provides comprehensive support for Rotary Position Embeddings (RoPE)
with various scaling methods commonly used in modern transformer architectures.
RoPE encodes position information by rotating query and key vectors in the
attention mechanism.

Supported RoPE Scaling Methods:
    - **Default/Basic**: Standard RoPE with no scaling, as introduced in RoFormer.
    - **Linear**: Linearly scaled positions for simple context extension.
    - **Dynamic NTK**: Dynamic Neural Tangent Kernel scaling that adjusts the
      base parameter for better extrapolation.
    - **YaRN**: Yet another RoPE extensioN method combining interpolation and
      extrapolation with frequency correction.
    - **Deepseek YaRN**: Variant of YaRN with additional mscale parameters.
    - **LongRoPE (Phi-3)**: Phi-3 style scaling with separate short/long factors.
    - **Llama-3**: Wavelength-based frequency adjustment scaling method.
    - **mRoPE**: Multi-modal RoPE for vision-language models (Qwen2/3-VL).

Main Components:
    - **Modules**: Neural network modules (RotaryEmbedding and variants) that
      can be integrated into transformer attention layers.
    - **Compute Functions**: Low-level functions for computing frequency caches
      and applying rotary transformations.
    - **Factory Functions**: High-level functions (get_rope, get_frequencies)
      for creating embeddings based on configuration dictionaries.
    - **Configuration**: RopeConfig dataclass for managing RoPE parameters.

Example:
    >>> import jax.numpy as jnp
    >>> from easydel.layers.rotary import (
    ...     get_rope,
    ...     get_frequencies,
    ...     RopeConfig,
    ... )
    >>> # Create a standard RoPE embedding
    >>> rope = get_rope(head_size=64, rotary_dim=64, max_position=2048, base=10000)
    >>> # Create YaRN-scaled RoPE for extended context
    >>> rope_scaling = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 2048}
    >>> rope_yarn = get_rope(head_size=64, rotary_dim=64, max_position=4096, base=10000, rope_scaling=rope_scaling)
    >>> # Apply RoPE to query and key tensors
    >>> query = jnp.ones((1, 128, 8, 64))  # [batch, seq, heads, head_dim]
    >>> key = jnp.ones((1, 128, 8, 64))
    >>> positions = jnp.arange(128)
    >>> q_rot, k_rot = rope(positions, query, key)

See Also:
    - RoFormer paper: https://arxiv.org/abs/2104.09864
    - YaRN paper: https://arxiv.org/abs/2309.00071
"""

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
