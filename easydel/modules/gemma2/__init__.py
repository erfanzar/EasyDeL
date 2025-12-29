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

"""Gemma2 model implementation for EasyDeL.

This module provides the Gemma2 model architecture including configuration,
base model, causal language modeling, and sequence classification variants.

Gemma2 is an advanced decoder-only transformer model developed by Google, building upon
the original Gemma architecture with significant improvements for stability and performance.
It is designed for efficient causal language modeling and text generation tasks.

Key Architectural Features:
    - Sliding Window Attention: Alternates between local sliding window attention (odd layers)
      and global full attention (even layers) to balance efficiency and context modeling.
      Default sliding window size is 4096 tokens.
    - Soft-Capping: Applies tanh-based soft-capping to attention logits and final logits
      to prevent extreme values and improve training stability. Configured via
      `attn_logit_softcapping` and `final_logit_softcapping` parameters.
    - Query Pre-Attention Scaling: Uses a custom scalar (default: 224) for query scaling
      before attention computation, replacing the standard head_dim**-0.5 scaling.
    - RoPE (Rotary Position Embeddings): Applies rotary embeddings for relative position
      encoding with configurable rope_theta (default: 10000.0).
    - Pre and Post Layer Normalization: Employs RMSNorm both before and after attention
      and feedforward blocks for enhanced training stability.
    - Gated Linear Units (GLU): Uses gated feedforward networks with configurable
      activation functions (default: gelu_pytorch_tanh).

Model Variants:
    - Gemma2Model: Base decoder-only transformer for feature extraction.
    - Gemma2ForCausalLM: Language modeling head for next-token prediction and generation.
    - Gemma2ForSequenceClassification: Classification head for sequence-level tasks.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForCausalLM, Gemma2Config
    import jax.numpy as jnp

    # Initialize model from pretrained weights
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
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
    from easydel.modules.gemma2 import Gemma2Config

    config = Gemma2Config(
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        max_position_embeddings=8192,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        query_pre_attn_scalar=224,
        rope_theta=10000.0,
    )
    ```

For more information on Gemma2, see:
    - Model Card: https://huggingface.co/google/gemma-2-9b
    - Technical Report: https://arxiv.org/abs/2408.00118
"""

from .gemma2_configuration import Gemma2Config
from .modeling_gemma2 import Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2Model

__all__ = (
    "Gemma2Config",
    "Gemma2ForCausalLM",
    "Gemma2ForSequenceClassification",
    "Gemma2Model",
)
