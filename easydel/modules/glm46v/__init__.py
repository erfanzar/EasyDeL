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

"""GLM-4.6V: Enhanced Vision-Language Model.

This module implements the GLM-4.6V architecture, an enhanced version of GLM-4V with
improved vision-language capabilities. It inherits the GLM-4V architecture (vision encoder,
text decoder, mRoPE) with refinements for better multimodal understanding.

Architecture Overview:
    GLM-4.6V builds on GLM-4V with the same core components:

    1. Vision Encoder:
       - Reuses Glm4vVisionModel architecture
       - Standard vision transformer with patch embedding
       - Spatial patch merger for feature downsampling
       - Supports both images and videos with temporal patches

    2. Text Decoder:
       - Reuses Glm4vTextModel architecture
       - Multi-dimensional RoPE (mRoPE) for position encoding
       - Processes interleaved text and visual tokens
       - Grouped query attention (GQA) for efficiency

    3. Vision-Language Integration:
       - Visual embeddings merged at special token positions
       - Image tokens: image_token_id (default 151343)
       - Video tokens: video_token_id (default 151344)
       - Start/end markers for image and video sequences
       - mRoPE sections [8, 12, 12] for temporal/spatial encoding

Key Differences from GLM-4V:
    - Different special token IDs for compatibility with GLM-4.6 series
    - Optimized for better instruction following
    - Enhanced training procedures (not reflected in architecture)

Usage Example:
    ```python
    from easydel import Glm46VForConditionalGeneration, Glm46VConfig
    import jax.numpy as jnp

    # Initialize model
    config = Glm46VConfig(
        text_config={"hidden_size": 4096, "num_hidden_layers": 40},
        vision_config={"hidden_size": 1536, "depth": 24},
    )

    model = Glm46VForConditionalGeneration(config, rngs=nnx.Rngs(0))

    # Forward pass with image
    input_ids = jnp.array([[1, 2, 151343, 3, 4]])  # 151343 is image_token_id
    pixel_values = jnp.ones((1, 3, 336, 336))
    image_grid_thw = jnp.array([[1, 24, 24]])

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits
    ```

Configuration:
    - Glm46VConfig: Main configuration (reuses Glm4vVisionConfig and Glm4vTextConfig)

Models:
    - Glm46VForConditionalGeneration: Full model for image-to-text generation
    - Glm46VModel: Base multimodal model without LM head
"""

from .glm46v_configuration import Glm46VConfig
from .modeling_glm46v import Glm46VForConditionalGeneration, Glm46VModel

__all__ = [
    "Glm46VConfig",
    "Glm46VForConditionalGeneration",
    "Glm46VModel",
]
