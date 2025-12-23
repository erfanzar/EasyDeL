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

"""InternLM2 model implementation for EasyDeL.

This module provides the InternLM2 (InternLM 2nd generation) model architecture,
developed by Shanghai AI Laboratory. InternLM2 is a decoder-only transformer
optimized for both English and Chinese language understanding and generation.

Architecture Highlights:
    - Decoder-only Transformer: Standard causal attention architecture
    - Grouped Query Attention (GQA): Configurable query and key-value heads for efficiency
    - Rotary Position Embeddings (RoPE): With configurable theta and scaling support
    - RMSNorm: Root Mean Square Layer Normalization
    - Gated MLP: SiLU activation with gated projections
    - Bias Support: Optional bias terms in attention and MLP layers

Key Features:
    - Vocabulary size: 103,168 tokens (optimized for Chinese + English)
    - Flexible context length: 2,048 default, supports RoPE scaling for longer sequences
    - Attention bias: Configurable bias in Q, K, V projections
    - Pretraining tensor parallelism: Supports pretraining_tp parameter
    - Scan layers: Optional layer scanning for memory efficiency

Technical Specifications:
    - RMSNorm epsilon: 1e-6 for normalization stability
    - RoPE theta: 10,000 (configurable for different context lengths)
    - Support for RoPE scaling strategies (linear, dynamic, etc.)
    - Gradient checkpointing support for memory-efficient training

Usage Example:
    ```python
    from easydel import InternLM2Config, InternLM2ForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration (InternLM2-7B style)
    config = InternLM2Config(
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA with 8 KV heads
        max_position_embeddings=2048,
        rope_theta=10000,
        bias=True,
    )

    # Create model for causal LM
    model = InternLM2ForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 256), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 256, 103168)

    # For sequence classification
    from easydel import InternLM2ForSequenceClassification

    classifier = InternLM2ForSequenceClassification(
        config=config,
        rngs=random.PRNGKey(0)
    )
    class_outputs = classifier(input_ids=input_ids)
    ```

Available Classes:
    - InternLM2Config: Configuration class for InternLM2 models
    - InternLM2Model: Base transformer model without task-specific heads
    - InternLM2ForCausalLM: InternLM2 model with causal language modeling head
    - InternLM2ForSequenceClassification: InternLM2 model with classification head
"""

from .internlm2_configuration import InternLM2Config
from .modeling_internlm2 import InternLM2ForCausalLM, InternLM2ForSequenceClassification, InternLM2Model

__all__ = (
    "InternLM2Config",
    "InternLM2ForCausalLM",
    "InternLM2ForSequenceClassification",
    "InternLM2Model",
)
