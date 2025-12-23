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

"""GLM model implementation for EasyDeL.

This module provides the GLM (General Language Model) architecture, developed by
Tsinghua University's Knowledge Engineering Group. GLM is a decoder-only transformer
optimized for both natural language understanding and generation tasks.

Architecture Highlights:
    - Decoder-only Transformer: Standard transformer architecture with causal masking
    - Grouped Query Attention (GQA): 32 query heads with 2 key-value heads for efficiency
    - Partial Rotary Position Embeddings: 50% of head dimensions use RoPE (partial_rotary_factor=0.5)
    - Fixed Head Dimension: 128-dimensional attention heads regardless of hidden size
    - RMSNorm: Root Mean Square Layer Normalization for stable training
    - Gated MLP: SwiGLU activation (gate_up_proj design)

Key Features:
    - Maximum position embeddings: 131,072 tokens
    - Vocabulary size: 151,552 tokens (extended for multilingual support)
    - Attention bias: Supports bias terms in Q, K, V, and output projections
    - High precision RMSNorm: epsilon=1.5625e-07 for numerical stability
    - Flexible layer types: Configurable attention types per layer

Usage Example:
    ```python
    from easydel import GlmConfig, GlmForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration (GLM-4-9B style)
    config = GlmConfig(
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.5,
        max_position_embeddings=131072,
    )

    # Create model for causal LM
    model = GlmForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 512), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 512, 151552)

    # For sequence classification
    from easydel import GlmForSequenceClassification

    classifier = GlmForSequenceClassification(
        config=config,
        rngs=random.PRNGKey(0)
    )
    class_outputs = classifier(input_ids=input_ids)
    ```

Available Classes:
    - GlmConfig: Configuration class for GLM models
    - GlmModel: Base transformer model without task-specific heads
    - GlmForCausalLM: GLM model with causal language modeling head
    - GlmForSequenceClassification: GLM model with classification head
"""

from .glm_configuration import GlmConfig
from .modeling_glm import GlmForCausalLM, GlmForSequenceClassification, GlmModel

__all__ = ("GlmConfig", "GlmForCausalLM", "GlmForSequenceClassification", "GlmModel")
