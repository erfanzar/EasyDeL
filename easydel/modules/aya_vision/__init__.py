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

"""Aya Vision multimodal model implementation for EasyDeL.

AyaVision is a vision-language model that combines a vision encoder (SigLIP) with
a language decoder (Cohere2/LLaMA) through a multi-modal projector. It's designed
for image-to-text generation tasks and multimodal understanding.

Architecture:
    - Vision Tower: SigLIP vision encoder for processing images
    - Multi-Modal Projector: Pixel shuffle + gated linear projection
    - Language Model: Cohere2 or LLaMA-based text decoder
    - Image Token Integration: Merges image features into text embeddings

Key Features:
    - Vision feature selection strategies (default/full)
    - Pixel shuffling for spatial downsampling (default 2x)
    - Gated activation in projector for better feature fusion
    - Supports standard transformer caching for efficient generation

Usage Example:
    ```python
    import jax
    from easydel import AyaVisionConfig, AyaVisionForConditionalGeneration
    from flax import nnx as nn

    # Initialize configuration
    config = AyaVisionConfig.from_pretrained("CohereForAI/aya-vision-8b")

    # Create model
    model = AyaVisionForConditionalGeneration(
        config=config,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs
    input_ids = jax.numpy.array([[1, 2, 3, 255036, 4, 5]])  # 255036 is image token
    pixel_values = jax.numpy.ones((1, 3, 384, 384))  # Image input

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
    )
    logits = outputs.logits
    ```

Available Models:
    - CohereForAI/aya-vision-8b: 8B parameter vision-language model

Classes:
    - AyaVisionConfig: Configuration class for AyaVision models
    - AyaVisionModel: Base model without LM head
    - AyaVisionForConditionalGeneration: Full model with LM head for generation
"""

from .aya_vision_configuration import AyaVisionConfig
from .modeling_aya_vision import AyaVisionForConditionalGeneration, AyaVisionModel

__all__ = "AyaVisionConfig", "AyaVisionForConditionalGeneration", "AyaVisionModel"
