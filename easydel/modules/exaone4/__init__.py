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

"""EXAONE 4 Model Implementation for EasyDeL.

This module provides the EXAONE 4.0 architecture from LG AI Research, a decoder-only
transformer model featuring sliding window attention for efficient long-context processing.
EXAONE 4 is designed for multilingual and multimodal understanding with strong performance
on Korean and English language tasks.

Architecture Details:
    EXAONE 4.0 is a decoder-only causal language model with:

    - Token Embeddings: Learned token embeddings with expanded vocabulary (102400 tokens)
    - Decoder Layers: Transformer decoder layers with:
        * Multi-head self-attention with RoPE (Rotary Position Embeddings)
        * **Sliding Window Attention**: Configurable pattern for efficiency
        * Grouped Query Attention (GQA) support
        * Gated Linear Unit (GLU) feed-forward networks with SiLU activation
        * RMSNorm for layer normalization
        * Residual connections
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary

Key Features:
    - **Sliding Window Attention**: Configurable window size (default 4096) with
      pattern-based full attention layers for global context
    - Large Vocabulary: 102,400 tokens for improved multilingual coverage
    - Grouped Query Attention: Efficient KV caching for generation
    - Rotary Position Embeddings: RoPE with optional scaling
    - Flexible Attention Patterns: Per-layer sliding vs. full attention
    - Flash Attention Support: Optimized attention computation
    - Tensor Parallelism: Distributed training support
    - Gradient Checkpointing: Memory-efficient training
    - Multiple Task Heads: Base model, causal LM, and sequence classification

Default Configuration (EXAONE 4.0):
    - Vocabulary size: 102,400
    - Hidden size: 4096
    - Intermediate size: 16384
    - Number of layers: 32
    - Attention heads: 32
    - Key-value heads: Configurable (defaults to num_attention_heads)
    - Max position embeddings: 2048
    - Sliding window: 4096 tokens
    - Sliding window pattern: Full attention every 4th layer
    - RoPE theta: 10000.0
    - Hidden activation: SiLU

Usage Example:
    ```python
    from easydel import Exaone4Config, Exaone4ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration
    config = Exaone4Config(
        vocab_size=102400,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Using GQA
        max_position_embeddings=2048,
        sliding_window=4096,
        sliding_window_pattern=4,  # Full attention every 4th layer
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = Exaone4ForCausalLM(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=rngs,
    )

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    ```

Available Classes:
    - Exaone4Config: Configuration class for EXAONE 4.0 models
    - Exaone4Model: Base EXAONE 4.0 transformer model
    - Exaone4ForCausalLM: EXAONE 4.0 with causal language modeling head
    - Exaone4ForSequenceClassification: EXAONE 4.0 with classification head

Reference:
    EXAONE 4.0 by LG AI Research.
    See: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

Note:
    EXAONE 4.0 is optimized for multilingual understanding, particularly excelling
    in Korean and English tasks. The sliding window attention mechanism enables
    efficient processing of long documents while maintaining strong performance.
"""

from .exaone4_configuration import Exaone4Config
from .modeling_exaone4 import (
    Exaone4ForCausalLM,
    Exaone4ForSequenceClassification,
    Exaone4Model,
)

__all__ = (
    "Exaone4Config",
    "Exaone4ForCausalLM",
    "Exaone4ForSequenceClassification",
    "Exaone4Model",
)
