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

"""CLIP (Contrastive Language-Image Pre-training) model implementation for EasyDeL.

CLIP is OpenAI's multimodal architecture that jointly trains vision and language encoders
using contrastive learning on image-text pairs. It enables zero-shot image classification,
image-text retrieval, and visual reasoning by learning aligned representations across modalities.

Key architectural features:

- Dual-Encoder Architecture: Consists of independent vision and text encoders that
  project inputs into a shared embedding space where semantically similar images and
  texts have high cosine similarity.

- Vision Transformer (ViT) Encoder: Uses a transformer architecture on image patches.
  Images are divided into fixed-size patches (default 16x16 or 32x32), linearly embedded,
  and processed through transformer layers with a special [CLS] token for global representation.

- Text Transformer Encoder: A causal (autoregressive) transformer that processes tokenized
  text. Uses learned position embeddings and extracts the final token's representation as
  the text embedding (end-of-sequence pooling).

- Contrastive Learning Objective: Trains by maximizing cosine similarity between correct
  image-text pairs while minimizing similarity for mismatched pairs, using a symmetric
  cross-entropy loss (InfoNCE).

- Learnable Temperature Parameter: Uses a learned temperature scaling parameter (logit_scale)
  to control the sharpness of the similarity distribution during training and inference.

- Projection Heads: Optional linear projections after encoders to map representations
  to a joint embedding space (enabled via CLIPTextModelWithProjection).

- Multi-Head Attention: Both encoders use standard multi-head self-attention with
  configurable head counts and hidden dimensions.

Usage Example:
    ```python
    from easydel.modules.clip import (
        CLIPConfig, CLIPTextConfig, CLIPVisionConfig,
        CLIPModel, CLIPTextModel, CLIPVisionModel
    )
    import jax
    import jax.numpy as jnp
    from flax import nnx as nn

    # Configure CLIP with ViT-B/32 vision and text encoders
    vision_config = CLIPVisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=32,
    )

    text_config = CLIPTextConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        vocab_size=49408,
        max_position_embeddings=77,
    )

    config = CLIPConfig(
        vision_config=vision_config,
        text_config=text_config,
        projection_dim=512,
    )

    # Initialize full CLIP model
    rngs = nn.Rngs(0)
    model = CLIPModel(config, rngs=rngs)

    # Encode image and text
    pixel_values = jnp.ones((1, 224, 224, 3))  # Batch of images
    input_ids = jnp.ones((1, 77), dtype=jnp.int32)  # Batch of text tokens

    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    image_embeds = outputs.image_embeds  # (batch, projection_dim)
    text_embeds = outputs.text_embeds    # (batch, projection_dim)

    # Compute similarity
    similarity = jnp.matmul(image_embeds, text_embeds.T)  # (batch, batch)
    ```
"""

from .clip_configuration import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from .modeling_clip import (
    CLIPForImageClassification,
    CLIPModel,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModel,
)

__all__ = (
    "CLIPConfig",
    "CLIPForImageClassification",
    "CLIPModel",
    "CLIPTextConfig",
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionConfig",
    "CLIPVisionModel",
)
