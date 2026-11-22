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

"""PaliGemma model implementation for EasyDeL.

PaliGemma is Google's open vision-language model family combining a SigLIP
vision encoder with a Gemma language model through a single linear multimodal
projector. It targets transfer-friendly vision-language tasks: captioning,
VQA, detection/segmentation-style structured outputs, and OCR.

Architecture Overview
--------------------
1. **Vision Encoder**: A SigLIP ViT (So400m/14 for the 3B models) encodes the
   input image into patch embeddings (e.g. 256 tokens for 224px at patch 14).
   SigLIP has no CLS token; all patch tokens are used.

2. **Multimodal Projector**: A single biased linear layer maps SigLIP hidden
   states to the Gemma embedding width (``projection_dim``).

3. **Language Model**: A Gemma decoder (Gemma 1 for PaliGemma, Gemma 2 for
   PaliGemma 2) consumes the merged sequence. PaliGemma's signature attention
   pattern makes the image tokens and the text prefix fully bidirectional
   among themselves while the suffix (the generated answer) stays causal;
   this is driven by ``token_type_ids`` (0 = image + prefix, 1 = suffix).
   Positions are 1-indexed.

Key Components
-------------
- **PaliGemmaConfig**: Composite configuration wrapping the SigLIP vision
  config and Gemma text config plus ``image_token_index``/``projection_dim``.
- **PaliGemmaModel**: Base multimodal trunk (tower + projector + LM, no head).
- **PaliGemmaForConditionalGeneration**: Adds the LM head (weight-tied to the
  input embeddings by default) for conditional generation.

Usage Example
------------
```python
import easydel as ed
import jax.numpy as jnp
import spectrax as spx

config = ed.PaliGemmaConfig(
    vision_config={"model_type": "siglip_vision_model", "image_size": 224, "patch_size": 14},
    text_config={"model_type": "gemma", "hidden_size": 2048, "num_hidden_layers": 18},
    image_token_index=256000,
    projection_dim=2048,
)
model = ed.PaliGemmaForConditionalGeneration(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    rngs=spx.Rngs(0),
)
outputs = model(
    input_ids=input_ids,  # contains 256000 placeholders for the image tokens
    pixel_values=pixel_values,
    token_type_ids=token_type_ids,  # 0 = image + prefix, 1 = suffix
)
```

References
---------
- Paper: "PaliGemma: A versatile 3B VLM for transfer" (Beyer et al., 2024)
- Model Hub: https://huggingface.co/google/paligemma-3b-pt-224
"""

from .modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaModel
from .paligemma_configuration import PaliGemmaConfig

__all__ = "PaliGemmaConfig", "PaliGemmaForConditionalGeneration", "PaliGemmaModel"
