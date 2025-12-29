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

"""LLaVA model implementation for EasyDeL.

LLaVA (Large Language and Vision Assistant) is a multimodal model that combines
a vision encoder with a large language model to enable visual understanding and
reasoning capabilities. It can process images alongside text, enabling tasks like
visual question answering, image captioning, and visual reasoning.

Architecture Overview
--------------------
LLaVA uses a modular architecture that connects vision and language models:

1. **Vision Encoder**: Typically a CLIP ViT (Vision Transformer) that processes input
   images and produces visual feature embeddings. The encoder is pretrained on
   image-text pairs and provides rich visual representations.

2. **Multimodal Projector**: A learnable linear projection (or MLP) that bridges the
   vision and language modalities. It maps visual features from the vision encoder
   into the embedding space of the language model, enabling seamless integration
   of visual and textual information.

3. **Language Model**: A decoder-only transformer LLM (typically LLaMA or similar)
   that processes the combined visual and textual embeddings to generate responses.
   The LLM provides the reasoning and text generation capabilities.

Key Components
-------------
- **LlavaConfig**: Configuration class that combines settings for both the vision
  encoder and language model. Includes parameters for the multimodal projector,
  vision feature selection strategy, and special token IDs for image inputs.

- **LlavaModel**: The base multimodal model that integrates the vision encoder,
  multimodal projector, and language model backbone. Processes combined image-text
  inputs and outputs contextualized hidden states.

- **LlavaForConditionalGeneration**: The generative variant that adds a language
  modeling head for text generation tasks. Enables visual question answering,
  image captioning, and other vision-language generation tasks.

How It Works
-----------
1. **Image Processing**: Input images are encoded by the vision encoder (CLIP ViT)
   to produce a sequence of patch embeddings (e.g., 576 tokens for a 336x336 image
   with 14x14 patches).

2. **Visual Feature Selection**: The model can select specific layers from the
   vision encoder (e.g., second-to-last layer via `vision_feature_layer=-2`) or
   use different selection strategies (`default` or `full`).

3. **Projection**: Visual embeddings are projected into the language model's
   embedding space using the multimodal projector (typically a linear layer or
   2-layer MLP with GELU activation).

4. **Embedding Merge**: Visual embeddings are inserted into the text sequence at
   positions marked by special image tokens (identified by `image_token_id`).
   The final sequence contains both text tokens and projected visual tokens.

5. **Language Model Processing**: The combined sequence is processed by the language
   model, which attends across both modalities to generate contextual representations
   and (optionally) produce text outputs.

Special Features
---------------
- **Flexible Vision Encoders**: Supports CLIP and other vision transformer encodings
- **Configurable Projector**: Can use linear projection or multi-layer MLP
- **Layer Selection**: Choose which vision encoder layers to extract features from
- **Strategy Options**: `default` (CLS token) or `full` (all patch tokens) vision features
- **Pretrained Initialization**: Can load pretrained vision and language components

Usage Example
------------
```python
from easydel import LlavaConfig, LlavaForConditionalGeneration
import jax.numpy as jnp
from flax import nnx as nn

# Initialize configuration with vision and text components
config = LlavaConfig(
    vision_config={
        "model_type": "clip_vision_model",
        "hidden_size": 1024,
        "image_size": 336,
        "patch_size": 14,
    },
    text_config={
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    },
    image_token_id=32000,
    projector_hidden_act="gelu",
    vision_feature_select_strategy="default",
)

# Create model instance
model = LlavaForConditionalGeneration(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Prepare inputs
# Text: "Describe this image: <image>"
# Image token (32000) is a placeholder for visual features
input_ids = jnp.array([[1, 2, 3, 32000, 4, 5]])  # <image> at position 3
pixel_values = jnp.ones((1, 3, 336, 336))  # Image tensor

# Forward pass
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
)
logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

# Generate response to visual question
# The model will attend to visual features when generating text
```

Training Paradigm
----------------
LLaVA uses a two-stage training approach:

1. **Pretrain Projector**: Freeze vision encoder and LLM, train only the multimodal
   projector on image captioning data to align vision and language representations.

2. **Finetune End-to-End**: Optionally finetune the entire model (or just LLM +
   projector) on instruction-following vision-language datasets for downstream tasks.

References
---------
- Paper: "Visual Instruction Tuning" (Liu et al., 2023)
- Model Hub: https://huggingface.co/llava-hf
- GitHub: https://github.com/haotian-liu/LLaVA
"""

from .llava_configuration import LlavaConfig
from .modeling_llava import LlavaForConditionalGeneration, LlavaModel

__all__ = "LlavaConfig", "LlavaForConditionalGeneration", "LlavaModel"
