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

"""Multimodal support for the eSurge inference engine.

This package provides vision-language model support including:
- VisionEncoderCache: LRU cache for vision encoder outputs
- MultiModalManager: Manages vision encoding and caching
- PlaceholderRange: Tracks multimodal placeholder positions
- MultiModalFeature: Single multimodal feature with metadata
- BatchedMultiModalInputs: Batched inputs for model forward pass

Example:
    >>> from easydel.inference.esurge.multimodal import MultiModalManager
    >>> manager = MultiModalManager(processor=processor)
    >>> pixel_values, grid_thw = manager.process_images(images)
"""

from .cache import VisionEncoderCache
from .manager import MultiModalManager
from .types import BatchedMultiModalInputs, MultiModalFeature, PlaceholderRange

__all__ = [
    "BatchedMultiModalInputs",
    "MultiModalFeature",
    "MultiModalManager",
    "PlaceholderRange",
    "VisionEncoderCache",
]
