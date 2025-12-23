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

"""Seed-OSS decoder-only transformer model implementation for EasyDeL.

Seed-OSS is a GPT-style language model with architectural enhancements for
improved performance and efficiency in text generation tasks.

Architecture:
    - Decoder-only transformer with pre-norm (Pre-LN) architecture
    - RMSNorm for layer normalization
    - Rotary Position Embeddings (RoPE) with optional scaling
    - SwiGLU gated activation in MLP layers
    - Optional sliding window attention per layer
    - Biased QKV projections, bias-free output projection

Key Features:
    - Layer-specific attention patterns (full vs sliding window)
    - Configurable sliding window via max_window_layers
    - RoPE scaling for extended context (up to 131K tokens)
    - Gradient checkpointing support for memory efficiency
    - Optional scan-based MLP for reduced memory footprint
    - Support for Grouped Query Attention (GQA)

Usage Example:
    ```python
    import jax
    from easydel import SeedOssConfig, SeedOssForCausalLM
    from flax import nnx as nn

    # Initialize configuration
    config = SeedOssConfig(
        vocab_size=200704,
        hidden_size=7168,
        num_hidden_layers=36,
        num_attention_heads=56,
        num_key_value_heads=56,  # Full MHA, use fewer for GQA
        intermediate_size=20480,
        max_position_embeddings=131072,
        use_sliding_window=True,
        sliding_window=4096,
        max_window_layers=28,  # First 28 layers use sliding window
    )

    # Create model for causal LM
    model = SeedOssForCausalLM(
        config=config,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs
    input_ids = jax.numpy.array([[1, 2, 3, 4, 5]])

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    ```

Available Models:
    - Seed-OSS models with various configurations
    - Optimized for long-context generation (up to 131K tokens)

Classes:
    - SeedOssConfig: Configuration class with sliding window and RoPE settings
    - SeedOssModel: Base model without task-specific head
    - SeedOssForCausalLM: Model with language modeling head
    - SeedOssForSequenceClassification: Model with classification head
"""

from .modeling_seed_oss import SeedOssForCausalLM, SeedOssForSequenceClassification, SeedOssModel
from .seed_oss_configuration import SeedOssConfig

__all__ = (
    "SeedOssConfig",
    "SeedOssForCausalLM",
    "SeedOssForSequenceClassification",
    "SeedOssModel",
)
