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

"""GLM-4V: Vision-Language Model with Multi-dimensional RoPE.

This module implements the GLM-4V architecture, a vision-language model that combines
a vision transformer encoder with a text decoder using multi-dimensional rotary position
embeddings (mRoPE) for effective vision-language alignment.

Architecture Overview:
    GLM-4V consists of three main components:

    1. Vision Encoder (Glm4vVisionModel):
       - Standard vision transformer with patch embedding
       - Processes images through self-attention layers
       - Spatial patch merger downsamples visual features by spatial_merge_size (default 2x2)
       - Projects vision features to text hidden dimension via merger MLP
       - Supports both image and video inputs with temporal patch embedding

    2. Vision-Language Integration:
       - Visual embeddings merged into text sequence at special token positions
       - Uses multi-dimensional RoPE (mRoPE) for position encoding
       - mRoPE sections: [temporal, height, width] dimensions (default [8, 12, 12])
       - Handles variable resolution images through dynamic grid positioning

    3. Text Decoder (Glm4vTextModel):
       - Standard transformer decoder with mRoPE support
       - Processes interleaved text and visual tokens
       - Standard MLP (gate/up/down projections) without MoE
       - Grouped query attention (GQA) for efficient KV caching

Key Features:
    - Multi-dimensional RoPE (mRoPE): 3D positional encoding [temporal, height, width]
    - Spatial Patch Merging: 2x2 downsampling of visual features for efficiency
    - Flexible Image Resolution: Handles variable resolution through adaptive grid
    - Video Support: Temporal patch embedding for video frame sequences
    - Query-Key Normalization: Optional use_qk_norm for training stability

Position Encoding (mRoPE):
    Unlike standard RoPE which uses 1D positions, GLM-4V uses 3D mRoPE:
    - Temporal dimension (8): For video frame positions
    - Height dimension (12): For vertical spatial positions
    - Width dimension (12): For horizontal spatial positions
    - Text tokens use standard 1D positions broadcast to 3D

Usage Example:
    ```python
    from easydel import Glm4vForConditionalGeneration, Glm4vConfig
    import jax.numpy as jnp

    # Initialize model
    config = Glm4vConfig(
        text_config={
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        },
        vision_config={
            "hidden_size": 1536,
            "depth": 24,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
    )

    model = Glm4vForConditionalGeneration(config, rngs=nnx.Rngs(0))

    # Prepare inputs with image
    input_ids = jnp.array([[1, 2, 151859, 3, 4]])  # 151859 is boi_token_id
    pixel_values = jnp.ones((1, 3, 336, 336))  # Single image
    image_grid_thw = jnp.array([[1, 24, 24]])  # Grid after patch embed

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    ```

Configuration Classes:
    - Glm4vConfig: Main configuration combining vision and text configs
    - Glm4vVisionConfig: Vision encoder configuration
    - Glm4vTextConfig: Text decoder configuration with mRoPE

Model Classes:
    - Glm4vForConditionalGeneration: Full model for image-to-text generation
    - Glm4vModel: Base multimodal model without LM head
    - Glm4vTextModel: Text decoder with mRoPE
    - Glm4vVisionModel: Vision encoder
"""

from .glm4v_configuration import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from .modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vModel,
    Glm4vTextModel,
    Glm4vVisionModel,
)

__all__ = [
    "Glm4vConfig",
    "Glm4vForConditionalGeneration",
    "Glm4vModel",
    "Glm4vTextConfig",
    "Glm4vTextModel",
    "Glm4vVisionConfig",
    "Glm4vVisionModel",
]
