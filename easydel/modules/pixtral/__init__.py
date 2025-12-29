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

"""Pixtral Vision Model Implementation for EasyDeL.

This module provides the Pixtral vision encoder architecture, designed for
multimodal vision-language applications. Pixtral is a vision transformer that
processes images through patch embedding and multi-head self-attention layers
with Rotary Position Embeddings (RoPE).

Architecture Details:
    The Pixtral vision encoder is designed to process variable-resolution images
    by dividing them into patches and encoding them with a transformer architecture.
    Key architectural components include:

    - Patch Embedding: Convolutional layer that converts image patches into embeddings
    - Position Encoding: 2D Rotary Position Embeddings (RoPE) for spatial awareness
    - Transformer Blocks: Standard transformer architecture with:
        * Multi-head self-attention with RoPE
        * Gated Linear Unit (GLU) feed-forward networks
        * RMSNorm for layer normalization
        * Residual connections
    - Multi-Image Support: Block-diagonal attention masks for processing multiple images

Key Features:
    - Variable resolution image processing up to 1024x1024 pixels
    - 2D Rotary Position Embeddings for spatial structure preservation
    - Block-diagonal attention for efficient multi-image batching
    - Flexible attention mechanisms with dropout support
    - JAX/Flax implementation with automatic gradient checkpointing
    - Distributed training support via tensor parallelism
    - Configurable precision (bfloat16, float32, etc.)

Default Configuration (Pixtral-12B):
    - Hidden size: 1024
    - Intermediate size: 4096
    - Number of layers: 24
    - Attention heads: 16
    - Image size: 1024x1024
    - Patch size: 16x16
    - RoPE theta: 10000.0

Usage Example:
    ```python
    from easydel import PixtralVisionConfig, PixtralVisionModel
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration
    config = PixtralVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=1024,
        patch_size=16,
    )

    # Create model
    rngs = nn.Rngs(0)
    model = PixtralVisionModel(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=rngs,
    )

    # Process images (list of images with shape [channels, height, width])
    images = [
        jnp.zeros((3, 1024, 1024)),  # First image
        jnp.zeros((3, 512, 512)),     # Second image (different size)
    ]

    outputs = model(
        pixel_values=images,
        output_hidden_states=True,
        output_attentions=False,
    )

    # Access outputs
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch, num_patches, hidden_size)
    all_hidden_states = outputs.hidden_states      # Tuple of hidden states from all layers
    ```

Available Classes:
    - PixtralVisionConfig: Configuration class for the Pixtral vision model
    - PixtralVisionModel: Main vision encoder model implementation

Reference:
    Based on the Pixtral architecture from Mistral AI's multimodal models.
    See: https://huggingface.co/mistralai/Pixtral-12B
"""

from .modeling_pixtral import PixtralVisionModel
from .pixtral_configuration import PixtralVisionConfig

__all__ = "PixtralVisionConfig", "PixtralVisionModel"
