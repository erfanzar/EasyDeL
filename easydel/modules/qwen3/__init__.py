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

"""Qwen3 model implementation for EasyDeL.

This module provides the Qwen3 model architecture including configuration,
base model, causal language modeling, and sequence classification variants.

Qwen3 is the third generation of Alibaba's Qwen (Tongyi Qianwen) series of large
language models, featuring improved architecture, extended context length, and
enhanced multilingual capabilities. It builds on the Qwen2 foundation with
optimizations for both efficiency and performance.

Key Architectural Features:
    - Grouped Query Attention (GQA): Uses GQA with configurable num_attention_heads
      and num_key_value_heads for memory-efficient attention, especially beneficial
      for large models.
    - Extended Context Window: Supports up to 128K tokens context length with
      YaRN-based RoPE scaling, enabling long-document understanding and generation.
    - Advanced RoPE Configuration: Implements YaRN (Yet another RoPE extensioN method)
      with configurable rope_scaling parameters for extending context beyond
      pre-training length.
    - Efficient Tokenization: Uses a large vocabulary (typically 152K tokens) with
      multilingual coverage for improved tokenization efficiency.
    - SwiGLU Activation: Employs SwiGLU (Swish-Gated Linear Unit) activation in
      feedforward networks for better performance.
    - Tie Word Embeddings: Optionally shares input and output embedding weights
      to reduce parameters while maintaining quality.

Model Variants:
    - Qwen3Model: Base decoder-only transformer for feature extraction.
    - Qwen3ForCausalLM: Language modeling head for next-token prediction and generation.
    - Qwen3ForSequenceClassification: Classification head for sequence-level tasks.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForCausalLM
    import jax.numpy as jnp

    # Load pretrained model
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # Prepare input tokens
    input_ids = jnp.array([[1, 2, 3, 4, 5]])

    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        params=params,
        max_length=100,
        temperature=0.7,
    )
    ```

Configuration Example:
    ```python
    from easydel.modules.qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=152064,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=131072,
        rope_theta=1000000.0,
        # YaRN RoPE scaling for extended context
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )
    ```

For more information on Qwen3, see:
    - Model Card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    - Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
"""

from .modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForSequenceClassification, Qwen3Model
from .qwen3_configuration import Qwen3Config

__all__ = (
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3ForSequenceClassification",
    "Qwen3Model",
)
