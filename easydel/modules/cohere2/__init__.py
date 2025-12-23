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

"""Cohere2: Advanced Command R2 Language Model for EasyDeL.

This module implements the Cohere Command R2 architecture, an advanced large language
model designed for enterprise applications with focus on retrieval-augmented generation
(RAG), tool use, and extended context understanding. Command R2 builds on Cohere's
proven architecture with enhancements for accuracy, efficiency, and controllability.

Architecture Overview:
    - **Decoder-only transformer**: Standard autoregressive language model with causal masking
    - **Grouped Query Attention (GQA)**: Efficient attention with multiple query heads sharing
      fewer key-value heads for reduced memory and faster inference
    - **Rotary Position Embeddings (RoPE)**: Position encoding through rotation for better
      length extrapolation and long-context understanding
    - **Layered normalization**: Strategic placement of LayerNorm for training stability
    - **Gated activations**: SwiGLU or similar gated activation functions for better expressiveness
    - **Extended context window**: Designed to handle very long contexts efficiently

Key Features:
    - **RAG optimization**: Architecture optimized for retrieval-augmented generation workflows
    - **Tool use capabilities**: Enhanced instruction following for function calling and tool use
    - **Long context**: Extended context window for processing long documents
    - **Multilingual support**: Strong performance across multiple languages
    - **Grounded generation**: Better factuality and citation capabilities
    - **Safety mechanisms**: Built-in safety features and controllability

Configuration:
    - **vocab_size** (default: varies): Size of vocabulary
    - **hidden_size**: Dimension of hidden states and embeddings
    - **num_hidden_layers**: Number of transformer decoder layers
    - **num_attention_heads**: Number of attention heads
    - **num_key_value_heads**: Number of key-value heads for GQA
    - **intermediate_size**: Dimension of feedforward intermediate layer
    - **hidden_act**: Activation function (typically silu or gated variants)
    - **max_position_embeddings**: Maximum sequence length supported
    - **rope_theta**: Base for RoPE frequency calculation
    - **attention_dropout**: Dropout probability for attention weights
    - **logit_scale**: Optional scaling factor for output logits

Available Models:
    - **Cohere2Config**: Configuration class with all hyperparameters
    - **Cohere2Model**: Base transformer model without task-specific head
    - **Cohere2ForCausalLM**: Model with language modeling head for text generation
    - **Cohere2ForSequenceClassification**: Model with classification head

Example Usage:
    ```python
    from easydel import Cohere2Config, Cohere2ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Create configuration for Command R2
    config = Cohere2Config(
        vocab_size=256000,
        hidden_size=8192,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA for efficiency
        intermediate_size=22528,
        max_position_embeddings=131072,  # Extended context
        rope_theta=10000.0,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = Cohere2ForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```

Use Cases:
    - **Retrieval-Augmented Generation**: Integrating external knowledge sources
    - **Long document understanding**: Processing and reasoning over long texts
    - **Tool use and function calling**: Executing structured actions based on text
    - **Multi-turn conversations**: Maintaining context across extended dialogues
    - **Grounded QA**: Answering questions with citations and source attribution
    - **Multilingual tasks**: Cross-lingual understanding and generation

Performance Characteristics:
    - **Extended context**: Efficient handling of 128k+ token contexts
    - **GQA efficiency**: Reduced KV cache size for faster long-context inference
    - **Instruction following**: Strong adherence to complex instructions
    - **Factual accuracy**: Improved grounding and reduced hallucinations
    - **Multilingual performance**: Competitive across many languages

References:
    - Cohere documentation: https://docs.cohere.com/
    - Command R models: https://cohere.com/command
    - GQA paper: https://arxiv.org/abs/2305.13245
"""

from .cohere2_configuration import Cohere2Config
from .modeling_cohere2 import Cohere2ForCausalLM, Cohere2ForSequenceClassification, Cohere2Model

__all__ = (
    "Cohere2Config",
    "Cohere2ForCausalLM",
    "Cohere2ForSequenceClassification",
    "Cohere2Model",
)
