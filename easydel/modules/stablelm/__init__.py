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

"""StableLM model implementation for EasyDeL.

This module provides the StableLM model architecture, developed by Stability AI.
StableLM is a decoder-only transformer with unique architectural features including
partial rotary embeddings, parallel residual connections, and QK layer normalization.

Architecture Highlights:
    - Decoder-only Transformer: Standard causal attention architecture
    - Partial Rotary Embeddings: Only 25% of head dimensions use RoPE (configurable)
    - Parallel Residual Connections: Optional parallel residual path for improved gradient flow
    - QK Layer Normalization: Optional LayerNorm on query and key states before attention
    - Grouped Query Attention (GQA): Configurable query and key-value head counts
    - Standard LayerNorm: Uses LayerNorm instead of RMSNorm (epsilon=1e-5)

Key Features:
    - Vocabulary size: 50,304 tokens (optimized vocabulary)
    - Context length: 4,096 tokens default
    - Partial rotary factor: 0.25 (RoPE applied to 25% of each head dimension)
    - Optional QKV bias: Configurable bias terms in attention projections
    - Parallel residual: Optional architecture variant with parallel attention/MLP paths

Architectural Innovations:
    - Partial RoPE: Reduces positional encoding overhead while maintaining performance
    - QK LayerNorm: Stabilizes attention computation with normalization before softmax
    - Parallel Residual: Enables better gradient flow and potentially faster training
    - Flexible attention bias: Supports both biased and bias-free attention variants

Usage Example:
    ```python
    from easydel import StableLmConfig, StableLmForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration (StableLM-3B style)
    config = StableLmConfig(
        vocab_size=50304,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # MHA (can use GQA with fewer KV heads)
        max_position_embeddings=4096,
        partial_rotary_factor=0.25,
        qk_layernorm=False,  # Enable for QK normalization variant
        use_parallel_residual=False,  # Enable for parallel residual variant
    )

    # Create model for causal LM
    model = StableLmForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 512), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 512, 50304)
    ```

Available Classes:
    - StableLmConfig: Configuration class for StableLM models
    - StableLmModel: Base transformer model without task-specific heads
    - StableLmForCausalLM: StableLM model with causal language modeling head
"""

from .modeling_stablelm import StableLmForCausalLM, StableLmModel
from .stablelm_configuration import StableLmConfig

__all__ = "StableLmConfig", "StableLmForCausalLM", "StableLmModel"
