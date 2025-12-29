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

"""MiniMax model implementation for EasyDeL.

This module provides the MiniMax (MiniMax-Text-01) model architecture, which features
a hybrid attention mechanism combining Lightning Attention (linear attention with
exponential decay) and standard full attention across alternating layers.

Architecture Highlights:
    - Hybrid Attention: Alternates between Lightning Attention (linear complexity) and
      standard full attention layers for balanced efficiency and quality.
    - Sparse Mixture-of-Experts (MoE): Uses 8 local experts with top-2 routing per token
      for efficient parameter scaling.
    - Lightning Attention: Block-wise linear attention with exponential decay for
      efficient long-sequence processing with 128K+ context support.
    - RMSNorm: Root Mean Square Layer Normalization for stable training.
    - Sliding Window: Optional sliding window attention for memory efficiency.

Key Features:
    - Maximum position embeddings: 131,072 tokens (4096 * 32)
    - Grouped Query Attention (GQA): 32 query heads, 8 key-value heads
    - Rotary Position Embeddings (RoPE) with theta=1e6
    - Block size: 256 tokens for lightning attention computation
    - Router jitter noise and auxiliary loss for MoE load balancing

Usage Example:
    ```python
    from easydel import MiniMaxConfig, MiniMaxForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration
    config = MiniMaxConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=131072,
        sliding_window=4096,
    )

    # Create model
    model = MiniMaxForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 128), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 128, 32000)
    ```

Available Classes:
    - MiniMaxConfig: Configuration class for MiniMax models
    - MiniMaxModel: Base transformer model without LM head
    - MiniMaxForCausalLM: MiniMax model with causal language modeling head
"""

from .minimax_configuration import MiniMaxConfig
from .modeling_minimax import MiniMaxForCausalLM, MiniMaxModel

__all__ = (
    "MiniMaxConfig",
    "MiniMaxForCausalLM",
    "MiniMaxModel",
)
