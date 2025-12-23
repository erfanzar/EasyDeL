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

"""GLM-4 model implementation for EasyDeL.

This module provides the GLM-4 (General Language Model 4) architecture, the fourth
generation model from Tsinghua University's Knowledge Engineering Group. GLM-4 is an
enhanced decoder-only transformer with additional post-normalization layers for
improved training stability.

Architecture Highlights:
    - Enhanced Decoder Architecture: Standard transformer with dual post-normalization
    - Grouped Query Attention (GQA): 32 query heads with 2 key-value heads for efficiency
    - Partial Rotary Position Embeddings: 50% of head dimensions use RoPE
    - Fixed Head Dimension: 128-dimensional attention heads regardless of model size
    - Dual Post-Normalization: Additional RMSNorm after both attention and MLP outputs
    - Gated MLP: SwiGLU activation with gate_up_proj design

Key Features:
    - Maximum position embeddings: 131,072 tokens
    - Vocabulary size: 151,552 tokens (extended multilingual vocabulary)
    - Attention bias: Supports bias terms in attention projections
    - Ultra-high precision RMSNorm: epsilon=1.5625e-07
    - Post-layer normalization: Separate norms for attention and MLP outputs

Improvements over GLM:
    - Post-self-attention LayerNorm for better gradient flow
    - Post-MLP LayerNorm for enhanced stability
    - More robust training dynamics with dual normalization

Usage Example:
    ```python
    from easydel import Glm4Config, Glm4ForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration (GLM-4-9B style)
    config = Glm4Config(
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
    model = Glm4ForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 512), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 512, 151552)

    # For sequence classification
    from easydel import Glm4ForSequenceClassification

    classifier = Glm4ForSequenceClassification(
        config=config,
        rngs=random.PRNGKey(0)
    )
    class_outputs = classifier(input_ids=input_ids)
    ```

Available Classes:
    - Glm4Config: Configuration class for GLM-4 models
    - Glm4Model: Base transformer model without task-specific heads
    - Glm4ForCausalLM: GLM-4 model with causal language modeling head
    - Glm4ForSequenceClassification: GLM-4 model with classification head
"""

from .glm4_configuration import Glm4Config
from .modeling_glm4 import Glm4ForCausalLM, Glm4ForSequenceClassification, Glm4Model

__all__ = ("Glm4Config", "Glm4ForCausalLM", "Glm4ForSequenceClassification", "Glm4Model")
