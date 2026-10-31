# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""InternVL model implementation for EasyDeL.

InternVL is a multimodal model family (InternVL3/3.5 HF ports under the
``internvl`` model type) that couples the InternViT vision tower with a
decoder-only language model (Qwen2 by default; Llama variants exist) through
pixel-shuffle downsampling and a LayerNorm+MLP multimodal projector.

Architecture Overview
---------------------
1. **Vision Encoder (InternViT)**: a timm/BEiT-style ViT with a CLS token,
   learned absolute position embeddings, per-layer layer-scale multipliers
   (``lambda_1``/``lambda_2``), configurable LayerNorm/RMSNorm and optional
   full-width RMSNorm on the query/key projections.

2. **Pixel Shuffle + Projector**: vision patch features (minus the CLS token)
   are reshaped to their spatial grid and downsampled by pixel shuffle
   (``downsample_ratio=0.5`` trades a 2x2 spatial block for a 4x channel
   expansion), then projected by ``LayerNorm -> Linear -> GELU -> Linear``
   into the language model's embedding space.

3. **Language Model**: an auto-resolved EasyDeL decoder (Qwen2/Llama/...)
   consumes the merged embedding sequence with projected vision tokens at
   positions marked by ``image_token_id``.

Key Components
--------------
- **InternVLVisionConfig / InternVLVisionModel**: the standalone vision tower
  (registered under ``internvl_vision``).
- **InternVLConfig**: composite configuration wrapping vision + text configs.
- **InternVLModel**: base multimodal trunk returning hidden states.
- **InternVLForConditionalGeneration**: adds the (optionally tied) LM head.

Usage Example
-------------
```python
import easydel as ed
import jax.numpy as jnp
import spectrax as spx

config = ed.InternVLConfig(
    vision_config={"model_type": "internvl_vision", "hidden_size": 1024},
    text_config={"model_type": "qwen2", "hidden_size": 896},
    image_token_id=151667,
)
model = ed.InternVLForConditionalGeneration(
    config=config,
    dtype=jnp.bfloat16,
    rngs=spx.Rngs(0),
)
```

References
----------
- Paper: "InternVL: Scaling up Vision Foundation Models and Aligning for
  Generic Visual-Linguistic Tasks" (Chen et al., 2023) and successors.
- Model Hub: https://huggingface.co/OpenGVLab
"""

from .internvl_configuration import InternVLConfig, InternVLVisionConfig
from .modeling_internvl import InternVLForConditionalGeneration, InternVLModel, InternVLVisionModel

__all__ = (
    "InternVLConfig",
    "InternVLForConditionalGeneration",
    "InternVLModel",
    "InternVLVisionConfig",
    "InternVLVisionModel",
)
