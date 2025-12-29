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

"""Mistral3 multimodal vision-language model implementation for EasyDeL.

Mistral3 is a vision-language model that combines Pixtral vision encoder with Mistral
text decoder, enabling image understanding and text generation in a unified architecture.

Architecture:
    - Vision Encoder: Pixtral for processing images
    - Multimodal Projector: GELU-activated projections with spatial merging
    - Text Decoder: Mistral transformer for language generation
    - Spatial Merge: Downsampling factor for reducing image tokens

Key Features:
    - Native Image Understanding: Direct image input processing
    - Spatial Merging: Reduces image tokens via 2D downsampling (default 2x2)
    - Multi-layer Vision Features: Can extract from multiple vision encoder layers
    - High Resolution: Supports images up to 1540x1540 pixels
    - Long Context: Text model supports 131K token context

Usage Example:
    ```python
    import jax
    import jax.numpy as jnp
    from easydel import Mistral3Config, Mistral3ForConditionalGeneration
    from flax import nnx as nn

    # Initialize configuration
    config = Mistral3Config(
        image_token_index=10,  # Special token for image placeholder
        projector_hidden_act="gelu",
        vision_feature_layer=-1,  # Use last vision layer
        spatial_merge_size=2,  # 2x2 spatial downsampling
    )

    # Create model
    model = Mistral3ForConditionalGeneration(
        config=config,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs
    input_ids = jnp.array([[1, 2, 10, 3, 4]])  # 10 is image token
    pixel_values = jnp.ones((1, 3, 1024, 1024))  # Image input

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
    )
    logits = outputs.logits
    ```

Available Models:
    - mistralai/Pixtral-12B-2409: 12B parameter vision-language model

Classes:
    - Mistral3Config: Configuration class for vision and text components
    - Mistral3Tokenizer: Custom tokenizer with image token support
    - Mistral3Model: Base model without LM head
    - Mistral3ForConditionalGeneration: Full model with LM head for generation
"""

from .mistral3_configuration import Mistral3Config
from .mistral3_tokenizer import Mistral3Tokenizer
from .modeling_mistral3 import Mistral3ForConditionalGeneration, Mistral3Model

__all__ = ("Mistral3Config", "Mistral3ForConditionalGeneration", "Mistral3Model", "Mistral3Tokenizer")
