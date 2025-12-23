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

"""GLM-4V-MoE: Vision-Language Model with Mixture of Experts.

This module implements the GLM-4V-MoE architecture, combining the vision-language
capabilities of GLM-4V with the grouped mixture-of-experts routing from GLM-4-MoE
for efficient scaling of multimodal models.

Architecture Overview:
    GLM-4V-MoE integrates three components:

    1. Vision Encoder (Glm4vMoeVisionModel):
       - Reuses GLM-4V vision transformer architecture
       - Patch embedding with spatial merging
       - Standard self-attention layers (no MoE in vision encoder)
       - Projects visual features to text dimension

    2. Text Decoder with MoE (Glm4vMoeTextModel):
       - Combines GLM-4V text decoder with GLM-4-MoE routing
       - Multi-dimensional RoPE (mRoPE) for position encoding
       - Hybrid dense-sparse architecture:
         * First `first_k_dense_replace` layers use dense FFN
         * Remaining layers use grouped MoE routing
       - Shared + routed experts for common and specialized knowledge

    3. Vision-Language Integration:
       - Visual embeddings merged at special token positions
       - mRoPE handles both text and visual position encoding
       - 3D position encoding [temporal, height, width] for images/videos

Key Features:
    - Grouped MoE Routing: Hierarchical expert selection (groups -> experts within groups)
    - Shared + Routed Experts: n_shared_experts always active, n_routed_experts conditionally selected
    - Multi-dimensional RoPE: 3D positional encoding for vision-language alignment
    - Hybrid Dense-Sparse: Early layers dense for stability, deep layers sparse for capacity

MoE Configuration:
    - num_experts_per_tok: Number of routed experts per token (default: 8)
    - n_routed_experts: Total number of routed experts (default: 128)
    - n_shared_experts: Number of always-active shared experts (default: 1)
    - n_group: Number of expert groups for hierarchical routing (default: 1)
    - topk_group: Number of groups to select (default: 1)
    - first_k_dense_replace: Number of dense layers before MoE (default: 1)

Usage Example:
    ```python
    from easydel import Glm4vMoeForConditionalGeneration, Glm4vMoeConfig
    import jax.numpy as jnp

    # Initialize model with MoE
    config = Glm4vMoeConfig(
        text_config={
            "hidden_size": 4096,
            "num_hidden_layers": 46,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 1,
        },
        vision_config={
            "hidden_size": 1536,
            "depth": 24,
        },
    )

    model = Glm4vMoeForConditionalGeneration(config, rngs=nnx.Rngs(0))

    # Forward pass with router logits
    input_ids = jnp.array([[1, 2, 151859, 3, 4]])  # Special token for image
    pixel_values = jnp.ones((1, 3, 336, 336))
    image_grid_thw = jnp.array([[1, 24, 24]])

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_router_logits=True,
    )
    logits = outputs.logits
    router_logits = outputs.router_logits  # For load balancing loss
    ```

Configuration Classes:
    - Glm4vMoeConfig: Main configuration combining vision and MoE text configs
    - Glm4vMoeVisionConfig: Vision encoder configuration (same as GLM-4V)
    - Glm4vMoeTextConfig: Text decoder with MoE configuration

Model Classes:
    - Glm4vMoeForConditionalGeneration: Full model for image-to-text generation
    - Glm4vMoeModel: Base multimodal model without LM head
    - Glm4vMoeTextModel: Text decoder with MoE layers
    - Glm4vMoeVisionModel: Vision encoder
"""

from .glm4v_moe_configuration import Glm4vMoeConfig, Glm4vMoeTextConfig, Glm4vMoeVisionConfig
from .modeling_glm4v_moe import (
    Glm4vMoeForConditionalGeneration,
    Glm4vMoeModel,
    Glm4vMoeTextModel,
    Glm4vMoeVisionModel,
)

__all__ = [
    "Glm4vMoeConfig",
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",
    "Glm4vMoeTextConfig",
    "Glm4vMoeTextModel",
    "Glm4vMoeVisionConfig",
    "Glm4vMoeVisionModel",
]
