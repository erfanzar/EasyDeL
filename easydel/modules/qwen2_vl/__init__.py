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

"""Qwen2-VL: Vision-Language Transformer for Multimodal Understanding.

This module implements the Qwen2-VL architecture, a powerful vision-language model
developed by Alibaba Cloud that combines a vision encoder with a text decoder to
process both images and videos alongside text for multimodal understanding tasks.

Architecture Overview:
    - **Vision Encoder**: Vision Transformer (ViT) based encoder that processes images and videos
      into patch embeddings with temporal-spatial-height-width (TSHW) positioning
    - **Patch Embedding**: Converts images/videos into tokens using 3D convolution with configurable
      patch sizes (spatial: 14x14, temporal: 2 for videos)
    - **Spatial Merging**: Downsamples visual tokens by merging neighboring patches to reduce
      sequence length while preserving spatial information
    - **Text Decoder**: Qwen2-based transformer decoder for autoregressive text generation
    - **Multimodal Fusion**: Learnable projector that aligns vision encoder output with text
      decoder's embedding space
    - **M-RoPE**: Multi-dimensional Rotary Position Embedding that encodes temporal, height,
      and width dimensions separately for fine-grained position awareness

Vision Configuration:
    - **depth** (default: 32): Number of vision transformer layers
    - **embed_dim** (default: 1280): Dimension of vision patch embeddings
    - **hidden_size** (default: 3584): Intermediate dimension in vision MLP
    - **num_heads** (default: 16): Number of attention heads in vision encoder
    - **patch_size** (default: 14): Spatial patch size for images
    - **temporal_patch_size** (default: 2): Temporal patch size for videos
    - **spatial_merge_size** (default: 2): Merge factor for spatial downsampling

Text Configuration:
    - **vocab_size** (default: 152064): Text vocabulary size
    - **hidden_size** (default: 8192): Hidden dimension of text decoder
    - **num_hidden_layers** (default: 80): Number of decoder layers
    - **num_attention_heads** (default: 64): Number of attention heads
    - **rope_theta** (default: 1000000.0): Base for RoPE frequency calculation
    - **use_sliding_window** (default: False): Enable sliding window attention for long sequences

Key Features:
    - **Unified image and video processing**: Single model handles both modalities
    - **Dynamic resolution**: Supports variable image/video sizes through position encoding
    - **3D positional encoding**: Temporal-Height-Width RoPE for precise spatial-temporal modeling
    - **Efficient token merging**: Reduces visual token count while maintaining quality
    - **Flexible attention**: Supports standard, flash, and paged attention mechanisms
    - **Multi-scale vision**: Can process multiple images/videos in one sequence

Available Models:
    - **Qwen2VLVisionConfig**: Configuration for vision encoder
    - **Qwen2VLTextConfig**: Configuration for text decoder
    - **Qwen2VLConfig**: Combined configuration with vision and text sub-configs
    - **Qwen2VLModel**: Base multimodal model without task head
    - **Qwen2VLTextModel**: Text-only decoder (used within the full model)
    - **Qwen2VLForConditionalGeneration**: Full VLM for image/video captioning and VQA

Special Token IDs:
    - **image_token_id** (default: 151655): Token representing image placeholder
    - **video_token_id** (default: 151656): Token representing video placeholder
    - **vision_start_token_id** (default: 151652): Marks start of vision sequence
    - **vision_end_token_id** (default: 151653): Marks end of vision sequence

Example Usage:
    ```python
    from easydel import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    import jax.numpy as jnp
    from flax import nnx as nn

    # Create configuration with custom settings
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": 152064,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
        },
        vision_config={
            "depth": 32,
            "embed_dim": 1280,
            "patch_size": 14,
        },
    )

    # Initialize model for conditional generation
    rngs = nn.Rngs(0)
    model = Qwen2VLForConditionalGeneration(config, rngs=rngs)

    # Process image and text
    input_ids = jnp.array([[1, 151655, 2, 3, 4]])  # Text with image token
    pixel_values = jnp.ones((1, 3, 224, 224))  # Single image
    image_grid_thw = jnp.array([[1, 16, 16]])  # Grid dimensions (T, H, W)

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```

Input Format:
    The model expects vision inputs to be preprocessed into grid representations:
    - Images: (batch, 3, height, width) in RGB format
    - Videos: (batch, 3, temporal_frames, height, width)
    - Grid THW: (num_images/videos, 3) specifying [temporal, height_patches, width_patches]

References:
    - Qwen2-VL paper: https://arxiv.org/abs/2409.12191
    - Qwen technical blog: https://qwenlm.github.io/blog/qwen2-vl/
"""

from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel, Qwen2VLTextModel
from .qwen2_vl_configuration import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig

__all__ = (
    "Qwen2VLConfig",
    "Qwen2VLForConditionalGeneration",
    "Qwen2VLModel",
    "Qwen2VLTextConfig",
    "Qwen2VLTextModel",
    "Qwen2VLVisionConfig",
)
