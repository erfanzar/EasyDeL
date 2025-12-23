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

"""OLMo3 Model Implementation for EasyDeL.

This module provides the OLMo3 (Open Language Model 3) architecture, an advanced
decoder-only transformer model developed by the Allen Institute for AI (AI2).
OLMo3 extends OLMo2 with sliding window attention for improved efficiency and longer
context handling, while maintaining Query/Key normalization for training stability.

Architecture Details:
    OLMo3 is a decoder-only causal language model that builds upon OLMo2 with:

    - Token Embeddings: Learned token embeddings for vocabulary
    - Decoder Layers: Stack of transformer decoder layers with:
        * Multi-head self-attention with RoPE (Rotary Position Embeddings)
        * Query/Key RMSNorm for improved training stability
        * **Sliding Window Attention**: Configurable per-layer attention patterns
        * Grouped Query Attention (GQA) support for efficient inference
        * Gated Linear Unit (GLU) feed-forward networks with SiLU activation
        * Post-attention and post-feedforward RMSNorm layers
        * Residual connections after each sub-layer
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary for next-token prediction

Key Features:
    - **Sliding Window Attention**: Default pattern uses sliding window (4096 tokens)
      for 3 out of 4 layers, with full attention every 4th layer for global context
    - Query/Key Normalization: RMSNorm applied to Q and K projections for stability
    - Grouped Query Attention (GQA): Configurable key-value head count
    - Rotary Position Embeddings: RoPE with optional scaling
    - Flexible Attention Patterns: Per-layer configuration of attention types
    - Flash Attention Support: Efficient attention computation
    - KV Caching: Support for efficient generation
    - Tensor Parallelism: Built-in distributed training support
    - Gradient Checkpointing: Memory-efficient training
    - Multiple Task Heads: Base model, causal LM, and sequence classification

Default Configuration (OLMo3-1B):
    - Vocabulary size: 50304
    - Hidden size: 4096
    - Intermediate size: 11008
    - Number of layers: 32
    - Attention heads: 32
    - Key-value heads: 32 (can be reduced for GQA)
    - Max position embeddings: 2048
    - Sliding window: 4096 tokens
    - RoPE theta: 10000.0
    - Hidden activation: SiLU
    - Attention pattern: Sliding for 75% of layers, full for every 4th layer

Usage Example:
    ```python
    from easydel import Olmo3Config, Olmo3ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration with custom attention pattern
    config = Olmo3Config(
        vocab_size=50304,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Using GQA
        max_position_embeddings=2048,
        sliding_window=4096,  # Sliding window size
        # Custom per-layer attention (optional)
        # layer_types=['sliding_attention'] * 24 + ['full_attention'] * 8
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = Olmo3ForCausalLM(
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
    - Olmo3Config: Configuration class for OLMo3 models
    - Olmo3Model: Base OLMo3 transformer model (no task head)
    - Olmo3ForCausalLM: OLMo3 model with causal language modeling head
    - Olmo3ForSequenceClassification: OLMo3 model with sequence classification head

Sliding Window Attention:
    The sliding window mechanism allows each token to attend to a fixed-size window
    of surrounding tokens (default 4096), reducing computational complexity while
    maintaining local context. Every 4th layer uses full attention to capture
    global dependencies. This hybrid approach balances efficiency and expressiveness.

Reference:
    Based on the OLMo3 architecture from the Allen Institute for AI.
    See: https://huggingface.co/allenai/OLMo-3-0725-1B

Note:
    OLMo3's key innovations are sliding window attention for efficiency and
    Query/Key normalization for training stability. The configurable per-layer
    attention patterns allow fine-tuning the trade-off between local and global
    context modeling.
"""

from .modeling_olmo3 import Olmo3ForCausalLM, Olmo3ForSequenceClassification, Olmo3Model
from .olmo3_configuration import Olmo3Config

__all__ = ("Olmo3Config", "Olmo3ForCausalLM", "Olmo3ForSequenceClassification", "Olmo3Model")
