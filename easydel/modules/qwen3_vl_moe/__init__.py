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

"""Qwen3-VL-MoE: Vision-Language Model with Mixture of Experts.

This module implements the Qwen3-VL-MoE architecture, which combines vision-language
capabilities from Qwen3-VL with Mixture of Experts (MoE) architecture from Qwen3-MoE,
enabling efficient multimodal understanding and generation at scale.

Architecture Overview:
    The model consists of three main components:

    1. Vision Encoder (Qwen3VLMoeVisionTransformerPretrainedModel):
       - Processes images and videos through a vision transformer backbone
       - Uses 3D patch embedding with temporal support for video inputs
       - Employs spatial patch merger to downsample visual features
       - Applies rotary position embeddings (RoPE) for spatial awareness
       - Outputs visual features at multiple deepstack layers for integration

    2. Vision-Language Integration:
       - Merges visual embeddings into text token sequences at placeholder positions
       - Computes 3D position IDs (temporal, height, width) for multi-dimensional RoPE (mRoPE)
       - Supports both image tokens and video tokens with temporal scaling
       - Handles variable-resolution images through adaptive grid sizing

    3. Text Decoder with MoE (Qwen3VLMoeTextModel):
       - Standard transformer decoder with MoE feedforward layers
       - MoE layers appear at configurable frequency (decoder_sparse_step)
       - Routes tokens to top-k experts (num_experts_per_tok) from total pool (num_experts)
       - Uses grouped query attention (GQA) with optional sliding window attention
       - Supports optional dense MLP layers (mlp_only_layers) for specific positions

Key Features:
    - Multi-modal RoPE (mRoPE): 3D positional encoding for temporal/spatial vision tokens
    - Deepstack Visual Integration: Vision features injected at multiple decoder layers
    - Sparse MoE: Conditional computation via expert routing for efficiency
    - Video Support: Native temporal modeling with per-frame and temporal embeddings
    - Flexible Resolution: Dynamic image grid sizing with spatial merging

Usage Example:
    ```python
    from easydel import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeConfig
    import jax.numpy as jnp

    # Initialize model
    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_experts": 60,
            "num_experts_per_tok": 4,
        },
        vision_config={
            "hidden_size": 1152,
            "depth": 27,
        }
    )

    model = Qwen3VLMoeForConditionalGeneration(config, rngs=nnx.Rngs(0))

    # Prepare inputs
    input_ids = jnp.array([[1, 2, 151655, 3, 4]])  # 151655 is image_token_id
    pixel_values = jnp.ones((1, 3, 224, 224))
    image_grid_thw = jnp.array([[1, 14, 14]])  # temporal, height, width grid

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    ```

Configuration Classes:
    - Qwen3VLMoeConfig: Main configuration combining vision and text configs
    - Qwen3VLMoeVisionConfig: Vision encoder configuration
    - Qwen3VLMoeTextConfig: Text decoder with MoE configuration

Model Classes:
    - Qwen3VLMoeForConditionalGeneration: Full model for image/video-to-text generation
    - Qwen3VLMoeModel: Base multimodal model without LM head
    - Qwen3VLMoeTextModel: Text decoder with MoE
    - Qwen3VLMoeVisionTransformerPretrainedModel: Vision encoder

Output Classes:
    - Qwen3VLMoeCausalLMOutputWithPast: Model outputs with router logits and mRoPE deltas
"""

from .modeling_qwen3_vl_moe import (
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeVisionTransformerPretrainedModel,
)
from .qwen3_vl_moe_configuration import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)

__all__ = [
    "Qwen3VLMoeCausalLMOutputWithPast",
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeModel",
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeTextModel",
    "Qwen3VLMoeVisionConfig",
    "Qwen3VLMoeVisionTransformerPretrainedModel",
]
