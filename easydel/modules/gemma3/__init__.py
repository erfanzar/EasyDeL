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

"""Gemma3 model implementation for EasyDeL.

This module provides the Gemma3 multimodal model architecture including
configuration, text model, and conditional generation variants.

Gemma3 is a vision-language multimodal model developed by Google that combines a
SigLIP vision encoder with a Gemma2-based text decoder, enabling image understanding
and vision-question answering capabilities. The architecture extends Gemma2 with
multimodal projection layers for processing both text and image inputs.

Key Architectural Features:
    - Multimodal Architecture: Integrates SigLIP vision encoder with Gemma2-style
      text decoder through a learned projection layer for vision-language alignment.
    - Mixed Attention Patterns: Like Gemma2, alternates between local sliding window
      attention and global full attention across layers with configurable patterns.
    - Extended Context: Supports up to 131,072 token context length with RoPE scaling
      for long-context modeling (rope_theta=1,000,000).
    - Grouped Query Attention (GQA): Uses separate num_attention_heads and
      num_key_value_heads for efficient attention computation.
    - Vision Token Processing: Projects vision tokens from SigLIP encoder to text
      embedding space via multimodal projector for joint reasoning.
    - Sliding Window Pattern: Configurable sliding window attention pattern with
      default window size of 4096 tokens and pattern interval of 6 layers.

Model Variants:
    - Gemma3TextModel: Text-only decoder transformer for language modeling.
    - Gemma3ForCausalLM: Text model with language modeling head for generation.
    - Gemma3ForConditionalGeneration: Full multimodal model with vision encoder
      and text decoder for vision-language tasks like VQA and image captioning.
    - Gemma3ForSequenceClassification: Classification head for sequence tasks.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForConditionalGeneration
    import jax.numpy as jnp
    from PIL import Image

    # Load multimodal model
    model, params = AutoEasyDeLModelForConditionalGeneration.from_pretrained(
        "google/gemma-3-2b-it",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # Prepare inputs (image + text)
    image = Image.open("example.jpg")
    text = "What is in this image?"

    # Process and generate
    outputs = model.generate(
        pixel_values=preprocess_image(image),
        input_ids=tokenize_text(text),
        params=params,
        max_length=100,
    )
    ```

Configuration Example:
    ```python
    from easydel.modules.gemma3 import Gemma3Config, Gemma3TextConfig

    text_config = Gemma3TextConfig(
        vocab_size=262208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=131072,
        sliding_window=4096,
        rope_theta=1000000.0,
    )

    config = Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        vocab_size=262208,
    )
    ```

For more information on Gemma3, see:
    - Model Card: https://huggingface.co/google/paligemma-3-mix-448
    - Related: Gemma2 paper (text decoder) and SigLIP (vision encoder)
"""

from .gemma3_configuration import Gemma3Config, Gemma3TextConfig
from .modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3ForSequenceClassification,
    Gemma3MultiModalProjector,
    Gemma3TextModel,
)

__all__ = (
    "Gemma3Config",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3ForSequenceClassification",
    "Gemma3MultiModalProjector",
    "Gemma3TextConfig",
    "Gemma3TextModel",
)
