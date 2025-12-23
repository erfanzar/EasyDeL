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

"""SigLIP model implementation for EasyDeL.

SigLIP (Sigmoid Loss for Language-Image Pre-training) is a vision-language model
that improves upon CLIP by using a simpler and more effective sigmoid-based loss
function instead of softmax-based contrastive loss. It achieves better performance
and training efficiency for vision-language alignment tasks.

Architecture Overview
--------------------
SigLIP follows a dual-encoder architecture similar to CLIP but with key improvements:

1. **Vision Encoder**: A Vision Transformer (ViT) that processes images as sequences
   of patches. Each image is divided into fixed-size patches, linearly embedded, and
   processed through transformer layers to produce visual representations.

2. **Text Encoder**: A transformer-based text encoder that processes tokenized text
   sequences. It uses learned token embeddings and positional encodings to create
   contextual text representations.

3. **Sigmoid Loss**: Unlike CLIP's softmax contrastive loss that requires large batch
   sizes and pairwise comparisons, SigLIP uses a simpler sigmoid loss that treats each
   image-text pair independently. This enables more efficient training with smaller
   batches while achieving better performance.

Key Components
-------------
- **SiglipConfig**: Combined configuration for both vision and text encoders, including
  projection dimensions and model-specific parameters.

- **SiglipVisionConfig**: Configuration for the vision encoder, specifying image size,
  patch size, hidden dimensions, and number of transformer layers.

- **SiglipTextConfig**: Configuration for the text encoder, defining vocabulary size,
  sequence length, hidden size, and attention heads.

- **SiglipModel**: The complete dual-encoder model that processes both images and text,
  producing aligned embeddings in a shared vector space.

- **SiglipVisionModel**: Standalone vision encoder for extracting image features.

- **SiglipTextModel**: Standalone text encoder for extracting text features.

- **SiglipForImageClassification**: Vision model with a classification head for
  image classification tasks.

Improvements Over CLIP
----------------------
1. **Sigmoid Loss**: More sample-efficient than softmax contrastive loss
2. **Smaller Batches**: Can train effectively with much smaller batch sizes
3. **Better Performance**: Often achieves superior zero-shot and fine-tuned performance
4. **Simpler Training**: Fewer hyperparameters and more stable training
5. **Bias Terms**: Includes learnable bias terms for better calibration

Use Cases
---------
- **Zero-Shot Image Classification**: Classify images using natural language descriptions
- **Image-Text Retrieval**: Find relevant images for text queries or vice versa
- **Vision-Language Pretraining**: Foundation model for multimodal tasks
- **Feature Extraction**: Extract aligned visual and textual embeddings
- **Vision Backbones**: Vision encoder for multimodal models (e.g., LLaVA)

Usage Example
------------
```python
from easydel import SiglipConfig, SiglipModel
import jax.numpy as jnp
from flax import nnx as nn

# Initialize configuration
config = SiglipConfig(
    vision_config={
        "hidden_size": 768,
        "image_size": 224,
        "patch_size": 16,
        "num_hidden_layers": 12,
    },
    text_config={
        "hidden_size": 768,
        "vocab_size": 32000,
        "max_position_embeddings": 64,
        "num_hidden_layers": 12,
    },
)

# Create model
model = SiglipModel(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Prepare inputs
pixel_values = jnp.ones((1, 3, 224, 224))  # Image
input_ids = jnp.ones((1, 16), dtype=jnp.int32)  # Text tokens

# Forward pass
outputs = model(
    pixel_values=pixel_values,
    input_ids=input_ids,
)

# Get embeddings
image_embeds = outputs.image_embeds  # Shape: [batch, embed_dim]
text_embeds = outputs.text_embeds    # Shape: [batch, embed_dim]

# Compute similarity
similarity = jnp.matmul(image_embeds, text_embeds.T)
```

Training Paradigm
----------------
SigLIP uses pairwise sigmoid loss where each image-text pair is scored independently:

```
loss = -sum(y_ij * log(sigmoid(z_ij)) + (1 - y_ij) * log(1 - sigmoid(z_ij)))
```

where y_ij = 1 if image i matches text j, else 0, and z_ij is the similarity score.
This allows efficient training without requiring all-pairs comparisons.

References
---------
- Paper: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
- Model Hub: https://huggingface.co/google/siglip-base-patch16-224
"""

from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from .modeling_siglip import SiglipForImageClassification, SiglipModel, SiglipTextModel, SiglipVisionModel

__all__ = (
    "SiglipConfig",
    "SiglipForImageClassification",
    "SiglipModel",
    "SiglipTextConfig",
    "SiglipTextModel",
    "SiglipVisionConfig",
    "SiglipVisionModel",
)
