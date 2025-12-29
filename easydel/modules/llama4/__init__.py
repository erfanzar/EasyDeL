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

"""LLaMA 4 model implementation for EasyDeL.

This module provides the LLaMA 4 multimodal model architecture including
text, vision, and conditional generation variants.

LLaMA 4 (also known as Llama 3.3) is Meta's latest multimodal large language model,
extending the LLaMA architecture with vision capabilities and mixture-of-experts
(MoE) feedforward networks. It supports both text-only and vision-language tasks.

Key Architectural Features:
    - Mixture of Experts (MoE): Uses sparse MoE layers in feedforward networks with
      configurable expert count, shared experts, and top-k routing for efficient
      scaling to larger model capacities.
    - Grouped Query Attention (GQA): Employs GQA with separate query and key-value
      head counts for memory-efficient attention computation.
    - Vision Encoder: Integrated vision tower for processing images, with learned
      multi-modal projection layers to align visual and text representations.
    - RoPE Scaling: Advanced rotary position embeddings with configurable scaling
      strategies including linear, dynamic, yarn, and llama3-specific variants.
    - QK Normalization: Optional query-key normalization for improved training
      stability in attention layers.
    - Extended Context: Supports long-context modeling with RoPE theta scaling
      and configurable maximum position embeddings.

Model Variants:
    - Llama4TextModel: Text-only decoder transformer for language modeling.
    - Llama4VisionModel: Vision encoder for processing image inputs.
    - Llama4ForCausalLM: Text model with language modeling head for generation.
    - Llama4ForConditionalGeneration: Full multimodal model combining vision
      encoder and text decoder for vision-language tasks.
    - Llama4ForSequenceClassification: Classification head for sequence tasks.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForCausalLM
    import jax.numpy as jnp

    # Load text-only model
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
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
    from easydel.modules.llama4 import Llama4Config, Llama4TextConfig

    text_config = Llama4TextConfig(
        vocab_size=128256,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        # MoE configuration
        num_experts_per_tok=2,
        num_local_experts=8,
        num_shared_experts=2,
    )
    ```

For more information on LLaMA 4, see:
    - Model Card: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    - LLaMA 3 Technical Report: https://arxiv.org/abs/2407.21783
"""

from .llama4_configuration import Llama4Config, Llama4TextConfig, Llama4VisionConfig
from .modeling_llama4 import (
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4ForSequenceClassification,
    Llama4TextModel,
    Llama4VisionModel,
)

__all__ = (
    "Llama4Config",
    "Llama4ForCausalLM",
    "Llama4ForConditionalGeneration",
    "Llama4ForSequenceClassification",
    "Llama4TextConfig",
    "Llama4TextModel",
    "Llama4VisionConfig",
    "Llama4VisionModel",
)
