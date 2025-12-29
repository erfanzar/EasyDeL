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

"""Kimi-VL: Vision-Language Model with DeepSeek-V3 Backbone.

This module implements the Kimi-VL architecture from Moonshot AI, which combines a custom
MoonViT vision encoder with the DeepSeek-V3 language model (featuring MLA attention and MoE)
for advanced multimodal understanding.

Architecture Overview:
    Kimi-VL consists of three main components:

    1. Vision Encoder (MoonViT):
       - Custom vision transformer architecture
       - Patch embedding with learnable position embeddings (init_pos_emb_height x init_pos_emb_width)
       - Standard transformer blocks with self-attention and MLP
       - Spatial merging via merge_kernel_size (default 2x2) for downsampling
       - Projects visual features to text hidden dimension

    2. Vision-Language Integration:
       - Visual embeddings merged at media_placeholder_token_id positions (default 163605)
       - Simple replacement strategy: visual features directly replace placeholder tokens
       - No additional position encoding beyond standard text positions

    3. Text Decoder (DeepSeek-V3):
       - Multi-head Latent Attention (MLA): Compressed KV cache for efficiency
       - Key-Dimension Attention (KDA): Additional attention mechanism
       - Mixture of Experts (MoE) in feedforward layers
       - Processes interleaved text and visual tokens

Key Features:
    - Multi-head Latent Attention (MLA): DeepSeek-V3's compressed attention mechanism
    - MoE Routing: Sparse expert activation for efficient scaling
    - MoonViT Vision Encoder: Custom vision architecture optimized for Kimi
    - Simple Multimodal Fusion: Direct token replacement without complex merging

DeepSeek-V3 Attention (MLA):
    - Compresses KV cache to latent dimension (kv_lora_rank)
    - Reduces memory footprint while maintaining model capacity
    - Key-Dimension Attention (KDA) for additional expressiveness
    - Query-absorbs-KV pattern for efficient computation

Usage Example:
    ```python
    from easydel import KimiVLForConditionalGeneration, KimiVLConfig
    import jax.numpy as jnp

    # Initialize model
    config = KimiVLConfig(
        text_config={
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
        vision_config={
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "patch_size": 14,
        },
    )

    model = KimiVLForConditionalGeneration(config, rngs=nnx.Rngs(0))

    # Forward pass with image
    input_ids = jnp.array([[1, 2, 163605, 3, 4]])  # 163605 is media_placeholder_token_id
    pixel_values = jnp.ones((1, 3, 896, 896))  # Variable resolution image

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
    )
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    ```

Configuration Classes:
    - KimiVLConfig: Main configuration combining MoonViT and DeepSeek-V3 configs
    - MoonViTConfig: Vision encoder configuration

Model Classes:
    - KimiVLForConditionalGeneration: Full model for image-to-text generation
"""

from .kimi_vl_configuration import KimiVLConfig, MoonViTConfig
from .modeling_kimi_vl import KimiVLForConditionalGeneration

__all__ = (
    "KimiVLConfig",
    "KimiVLForConditionalGeneration",
    "MoonViTConfig",
)
