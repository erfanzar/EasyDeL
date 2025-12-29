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

"""OLMo2 Model Implementation for EasyDeL.

This module provides the OLMo2 (Open Language Model 2) architecture, an advanced
decoder-only transformer model developed by the Allen Institute for AI (AI2).
OLMo2 features Query/Key normalization for improved training stability and uses
standard transformer components with Rotary Position Embeddings (RoPE).

Architecture Details:
    OLMo2 is a decoder-only causal language model with the following key components:

    - Token Embeddings: Learned token embeddings for vocabulary
    - Decoder Layers: Stack of transformer decoder layers with:
        * Multi-head self-attention with RoPE (Rotary Position Embeddings)
        * Query/Key RMSNorm for improved training stability
        * Grouped Query Attention (GQA) support for efficient inference
        * Gated Linear Unit (GLU) feed-forward networks with SiLU activation
        * Post-attention and post-feedforward RMSNorm layers
        * Residual connections after each sub-layer
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary for next-token prediction

Key Features:
    - Query/Key Normalization: RMSNorm applied to Q and K projections for stability
    - Grouped Query Attention (GQA): Configurable key-value head count for efficiency
    - Rotary Position Embeddings: RoPE for position encoding with optional scaling
    - Flash Attention Support: Efficient attention computation with various backends
    - KV Caching: Support for key-value caching during generation
    - Tensor Parallelism: Built-in support for distributed training
    - Gradient Checkpointing: Memory-efficient training with configurable checkpointing
    - Multiple Task Heads: Base model, causal LM, and sequence classification

Default Configuration (OLMo2-7B):
    - Vocabulary size: 50304
    - Hidden size: 4096
    - Intermediate size: 11008
    - Number of layers: 32
    - Attention heads: 32
    - Key-value heads: 32 (can be reduced for GQA)
    - Max position embeddings: 2048
    - RoPE theta: 10000.0
    - Hidden activation: SiLU

Usage Example:
    ```python
    from easydel import Olmo2Config, Olmo2ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration
    config = Olmo2Config(
        vocab_size=50304,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Using GQA
        max_position_embeddings=2048,
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = Olmo2ForCausalLM(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=rngs,
    )

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])  # Token IDs
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=False,
        output_attentions=False,
    )

    # Access logits for next token prediction
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    next_token_logits = logits[:, -1, :]  # Last position logits
    ```

Available Classes:
    - Olmo2Config: Configuration class for OLMo2 models
    - Olmo2Model: Base OLMo2 transformer model (no task head)
    - Olmo2ForCausalLM: OLMo2 model with causal language modeling head
    - Olmo2ForSequenceClassification: OLMo2 model with sequence classification head

Model Variants:
    - OLMo2-1B: 1 billion parameters
    - OLMo2-7B: 7 billion parameters
    - OLMo2-13B: 13 billion parameters

Reference:
    Based on the OLMo2 architecture from the Allen Institute for AI.
    See: https://huggingface.co/allenai/OLMo2-7B-1124-hf

Note:
    OLMo2 introduces Query/Key normalization as a key architectural innovation
    to improve training stability, especially at larger scales. This is implemented
    via RMSNorm applied to the Q and K projections before attention computation.
"""

from .modeling_olmo2 import Olmo2ForCausalLM, Olmo2ForSequenceClassification, Olmo2Model
from .olmo2_configuration import Olmo2Config

__all__ = ("Olmo2Config", "Olmo2ForCausalLM", "Olmo2ForSequenceClassification", "Olmo2Model")
