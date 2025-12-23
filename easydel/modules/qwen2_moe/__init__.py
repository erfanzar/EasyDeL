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

"""Qwen2-MoE: Sparse Mixture-of-Experts Transformer for EasyDeL.

This module implements the Qwen2-MoE architecture, a sparse mixture-of-experts (MoE)
transformer model developed by Alibaba Cloud. The architecture combines traditional
dense transformer layers with sparse MoE layers to achieve better performance and
efficiency through conditional computation.

Architecture Overview:
    - **Decoder-only transformer**: Autoregressive language model with causal masking
    - **Sparse MoE layers**: Alternating between dense and sparse expert layers based on
      `decoder_sparse_step` configuration
    - **Expert routing**: Top-k routing mechanism that selects `num_experts_per_tok` experts
      from a pool of `num_experts` for each token
    - **Shared expert**: Additional shared expert that processes all tokens alongside
      routed experts, controlled by a learned gating mechanism
    - **Sliding window attention**: Optional sliding window attention for long-range dependencies
    - **Grouped Query Attention (GQA)**: Configurable key-value heads for efficient attention

MoE Configuration:
    - **num_experts** (default: 60): Total number of expert networks in each MoE layer
    - **num_experts_per_tok** (default: 4): Number of experts activated per token (top-k routing)
    - **moe_intermediate_size** (default: 1408): Intermediate hidden size for expert MLPs
    - **shared_expert_intermediate_size** (default: 5632): Intermediate size for shared expert
    - **decoder_sparse_step** (default: 1): Frequency of MoE layers (e.g., 1 means every layer,
      2 means every other layer)
    - **router_aux_loss_coef** (default: 0.001): Coefficient for auxiliary load balancing loss

Key Features:
    - Dynamic expert selection during forward pass for computational efficiency
    - Load balancing through auxiliary loss to prevent expert collapse
    - Flexible parallelization with expert parallelism, tensor parallelism, and data parallelism
    - Support for gradient checkpointing to reduce memory usage
    - Compatible with various attention mechanisms (standard, flash, paged)

Available Models:
    - **Qwen2MoeConfig**: Configuration class with all model hyperparameters
    - **Qwen2MoeModel**: Base transformer model without task-specific head
    - **Qwen2MoeForCausalLM**: Model with language modeling head for text generation
    - **Qwen2MoeForSequenceClassification**: Model with classification head

Example Usage:
    ```python
    from easydel import Qwen2MoeConfig, Qwen2MoeForCausalLM
    from jax import random
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration with MoE settings
    config = Qwen2MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_experts=60,
        num_experts_per_tok=4,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        decoder_sparse_step=1,  # MoE every layer
    )

    # Create model for causal language modeling
    rngs = nn.Rngs(0)
    model = Qwen2MoeForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

    # Access router logits for expert routing analysis
    router_logits = outputs.router_logits  # Tuple of (num_layers,) router outputs
    ```

References:
    - Qwen2-MoE paper: https://arxiv.org/abs/2405.04434
    - Qwen technical report: https://arxiv.org/abs/2309.16609
"""

from .modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeForSequenceClassification, Qwen2MoeModel
from .qwen2_moe_configuration import Qwen2MoeConfig

__all__ = (
    "Qwen2MoeConfig",
    "Qwen2MoeForCausalLM",
    "Qwen2MoeForSequenceClassification",
    "Qwen2MoeModel",
)
