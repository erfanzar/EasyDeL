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

"""HunYuanVL: Tencent's dense image-text vision-language model.

This module implements the open-source HunYuanVL architecture (the family
behind ``tencent/HunyuanOCR``): a SigLIP-style vision tower paired with a
HunYuan dense decoder for OCR and document-understanding style workloads.

Architecture Overview:
    - **Vision Tower**: Conv patch embedding with a learned position grid
      bilinearly interpolated to each image's patch shape, pre-LayerNorm
      encoder blocks (biased q/k/v/o projections, no rotary embeddings,
      block-diagonal attention per image), and a convolutional patch merger
      that downsamples by ``spatial_merge_size``, appends a newline token
      per merged row, projects into the text hidden size, and wraps each
      image in learned begin/end tokens.
    - **Text Decoder**: HunYuan dense transformer — grouped-query attention
      where rotary embeddings are applied *before* the per-head RMSNorms on
      queries/keys, SwiGLU MLP, RMSNorm pre-normalization, and a tied LM
      head.
    - **Multimodal RoPE**: 3-axis mRoPE — ``mrope_section`` partitions half
      of each head across ``(width, height, image_index)``; image spans get
      2-D grid positions plus a global image ordinal while text keeps 1-D
      positions on all axes.

Available Classes:
    - **HunYuanVLVisionConfig**: Vision tower configuration
    - **HunYuanVLTextConfig**: Text decoder configuration
    - **HunYuanVLConfig**: Composite configuration
    - **HunYuanVLVisionModel**: Vision tower
    - **HunYuanVLTextModel**: Language-model trunk
    - **HunYuanVLModel**: Composite base model
    - **HunYuanVLForConditionalGeneration**: IMAGE_TEXT_TO_TEXT task head
"""

from .hunyuan_vl_configuration import HunYuanVLConfig, HunYuanVLTextConfig, HunYuanVLVisionConfig
from .modeling_hunyuan_vl import (
    HunYuanVLForConditionalGeneration,
    HunYuanVLModel,
    HunYuanVLTextModel,
    HunYuanVLVisionModel,
)

__all__ = (
    "HunYuanVLConfig",
    "HunYuanVLForConditionalGeneration",
    "HunYuanVLModel",
    "HunYuanVLTextConfig",
    "HunYuanVLTextModel",
    "HunYuanVLVisionConfig",
    "HunYuanVLVisionModel",
)
