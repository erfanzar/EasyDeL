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

"""Mamba - Efficient state-space sequence model without attention.

This module implements the Mamba architecture, a state-space model (SSM) that
achieves competitive performance with transformers while maintaining linear-time
complexity for sequence processing. Mamba replaces the attention mechanism with
selective state-space models, enabling efficient long-range sequence modeling.

**Key Features**:
- Selective state-space models (S6) for adaptive sequence modeling
- Linear-time complexity O(L) vs quadratic O(L²) for attention
- Data-dependent selective mechanisms for input-aware processing
- Hardware-efficient implementation using specialized kernels
- No attention mechanism - purely recurrent state-space architecture

**Architecture Highlights**:
- Structured State-Space (S4) foundation with selective mechanisms
- Gated convolution layers for local context
- Time-step parameters that adapt to input content
- Efficient recurrent processing via parallel scan
- Residual connections with optional float32 computation

**State-Space Model Components**:
- Selective SSM blocks with learnable A, B, C, D matrices
- Input-dependent time-step discretization
- Convolution kernel for local pattern capture
- Projection layers for input/output transformations
- State size controls memory vs expressiveness trade-off

**Performance Characteristics**:
- Constant memory footprint during generation (vs growing KV cache)
- Parallelizable training via parallel scan
- Fast autoregressive inference with recurrent state
- Scales efficiently to very long sequences (100K+ tokens)

**Available Model Variants**:
- MambaModel: Base state-space model
- MambaForCausalLM: Model with language modeling head

Example:
    >>> from easydel.modules.mamba import (
    ...     MambaConfig,
    ...     MambaForCausalLM,
    ... )
    >>> config = MambaConfig(
    ...     hidden_size=768,
    ...     state_size=16,  # SSM state dimension
    ...     num_hidden_layers=32,
    ...     expand=2,  # Intermediate size expansion factor
    ...     conv_kernel=4,  # Local convolution window
    ... )
    >>> model = MambaForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .mamba_configuration import MambaConfig
from .modeling_mamba import MambaForCausalLM, MambaModel

__all__ = ("MambaConfig", "MambaForCausalLM", "MambaModel")
