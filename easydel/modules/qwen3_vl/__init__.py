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

"""Qwen3-VL: Next-Generation Vision-Language Transformer for EasyDeL.

This module implements the Qwen3-VL architecture, an advanced evolution of Qwen2-VL
that enhances multimodal understanding capabilities through improved vision encoding,
more efficient fusion mechanisms, and better handling of high-resolution images and
long videos.

Architecture Overview:
    - **Enhanced Vision Encoder**: Improved Vision Transformer with better feature extraction
      for both images and videos, supporting higher resolutions and longer temporal sequences
    - **Dynamic Patch Processing**: Adaptive patch embedding that efficiently handles variable
      input resolutions without sacrificing quality
    - **Advanced Spatial Merging**: Refined downsampling strategy that preserves fine-grained
      visual details while maintaining computational efficiency
    - **Qwen3 Text Decoder**: Latest Qwen3 transformer decoder with improved architecture
      (Q/K normalization, enhanced attention mechanisms)
    - **Optimized Multimodal Fusion**: More efficient projection layer for aligning vision
      and language representations
    - **3D M-RoPE**: Multi-dimensional Rotary Position Embedding for temporal-spatial-height-width
      position encoding with enhanced scaling support

Vision Configuration:
    - **depth** (default: 32): Number of vision transformer layers
    - **embed_dim** (default: 1280): Vision patch embedding dimension
    - **hidden_size** (default: 3584): Intermediate dimension in vision MLP
    - **num_heads** (default: 16): Number of attention heads in vision encoder
    - **patch_size** (default: 14): Spatial patch size for images
    - **temporal_patch_size** (default: 2): Temporal patch size for videos
    - **spatial_merge_size** (default: 2): Spatial downsampling merge factor

Text Configuration:
    - **vocab_size** (default: 152064): Text vocabulary size
    - **hidden_size** (default: 8192): Hidden dimension of text decoder
    - **num_hidden_layers** (default: 80): Number of decoder layers
    - **num_attention_heads** (default: 64): Number of attention heads
    - **rope_theta** (default: 1000000.0): Base for RoPE frequency calculation
    - **attention_bias** (default: False): Whether to use attention bias

Key Improvements over Qwen2-VL:
    - **Higher resolution support**: Better handling of high-resolution images and videos
    - **Improved temporal modeling**: Enhanced video understanding with longer sequences
    - **Q/K normalization**: Optional query-key normalization for training stability
    - **Attention bias flexibility**: Configurable attention bias for better expressiveness
    - **Enhanced RoPE**: Improved rotary position encoding with advanced scaling
    - **Better efficiency**: Optimized architecture for faster inference and training

Available Models:
    - **Qwen3VLVisionConfig**: Configuration for vision encoder
    - **Qwen3VLTextConfig**: Configuration for Qwen3 text decoder
    - **Qwen3VLConfig**: Combined configuration with vision and text sub-configs
    - **Qwen3VLModel**: Base multimodal model without task head
    - **Qwen3VLTextModel**: Text-only decoder component
    - **Qwen3VLForConditionalGeneration**: Full VLM for image/video-to-text tasks
    - **Qwen3VisionTransformerPretrainedModel**: Standalone vision encoder
    - **Qwen3VLCausalLMOutputWithPast**: Output dataclass with rope_deltas

Example Usage:
    ```python
    from easydel import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    import jax.numpy as jnp
    from flax import nnx as nn

    # Create configuration for Qwen3-VL
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 152064,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "attention_bias": False,  # Qwen3 improvement
        },
        vision_config={
            "depth": 32,
            "embed_dim": 1280,
            "patch_size": 14,
        },
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = Qwen3VLForConditionalGeneration(config, rngs=rngs)

    # Process multimodal input
    input_ids = jnp.array([[1, 151655, 2, 3]])  # Text with image token
    pixel_values = jnp.ones((1, 3, 448, 448))   # High-resolution image
    image_grid_thw = jnp.array([[1, 32, 32]])   # Grid dimensions

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```

Special Features:
    - **Video understanding**: Native support for video inputs with temporal modeling
    - **Multi-image inputs**: Process multiple images in a single sequence
    - **Dynamic resolution**: Handles variable image sizes through position encoding
    - **Efficient generation**: Optimized KV caching for fast autoregressive decoding

References:
    - Qwen3-VL technical blog: https://qwenlm.github.io/
    - Qwen-VL series: https://github.com/QwenLM/Qwen-VL
"""

from .modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
)
from .qwen3_vl_configuration import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

__all__ = [
    "Qwen3VLCausalLMOutputWithPast",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLTextConfig",
    "Qwen3VLTextModel",
    "Qwen3VLVisionConfig",
    "Qwen3VisionTransformerPretrainedModel",
]
