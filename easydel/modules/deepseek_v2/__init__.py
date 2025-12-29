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

"""DeepSeek-V2 - Advanced MoE model with Multi-head Latent Attention.

This module implements the DeepSeek-V2 architecture, featuring innovative Multi-head
Latent Attention (MLA) and efficient Mixture-of-Experts (MoE) for scaling to very
large models with reduced computational cost. DeepSeek-V2 is designed for extreme
efficiency and performance in large-scale language modeling.

**Key Features**:
- Multi-head Latent Attention (MLA) for compressed KV cache
- Mixture-of-Experts (MoE) with routed and shared experts
- DeepSeekMoE architecture with fine-grained expert routing
- Low-rank adaptation (LoRA) for query and key-value projections
- Extreme parameter efficiency with sparse activation

**Architecture Highlights**:
- KV compression via low-rank projections (kv_lora_rank, q_lora_rank)
- Configurable expert parallelism (ep_size)
- Separate dimensions for RoPE and non-RoPE components
- Flexible MoE layer frequency and dense layer replacement
- Support for auxiliary loss and sequence-level expert routing

**MLA Components**:
- Query LoRA rank for compressed query representations
- KV LoRA rank for efficient key-value compression
- Separate head dimensions for QK-RoPE and value projections
- QK-nope (non-positional encoding) head dimension

**MoE Configuration**:
- Routed experts for sparse activation
- Shared experts active for all tokens
- Top-k expert selection per token
- Configurable expert groups and routing strategies

**Available Model Variants**:
- DeepseekV2Model: Base model with MLA and optional MoE layers
- DeepseekV2ForCausalLM: Model with language modeling head

Example:
    >>> from easydel.modules.deepseek_v2 import (
    ...     DeepseekV2Config,
    ...     DeepseekV2ForCausalLM,
    ... )
    >>> config = DeepseekV2Config(
    ...     hidden_size=4096,
    ...     num_hidden_layers=30,
    ...     num_attention_heads=32,
    ...     kv_lora_rank=512,  # MLA compression
    ...     q_lora_rank=1536,
    ...     n_routed_experts=64,  # MoE with 64 routed experts
    ...     n_shared_experts=2,
    ...     num_experts_per_tok=6,  # Top-6 routing
    ... )
    >>> model = DeepseekV2ForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .deepseek_configuration import DeepseekV2Config
from .modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2Model

__all__ = "DeepseekV2Config", "DeepseekV2ForCausalLM", "DeepseekV2Model"
