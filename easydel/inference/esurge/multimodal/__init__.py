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

This package provides comprehensive vision-language model (VLM) support for the
eSurge inference engine, enabling efficient processing of images and videos
alongside text in a batched inference pipeline.

The module is designed to work with various VLM architectures including:
    - GLM4V/GLM46V models
    - Qwen2-VL/Qwen3-VL models
    - Other vision-language models with flat-patch or standard vision encoders

Key Components:
    VisionEncoderCache:
        Thread-safe LRU cache for vision encoder outputs that uses content-based
        hashing to avoid redundant computation when the same images are processed
        multiple times across requests.

    MultiModalManager:
        Central manager for vision data processing that handles:
        - Image and video preprocessing with resolution bucketing
        - Integration with HuggingFace processors for tokenization
        - Multimodal message parsing and placeholder insertion
        - Vision encoder output caching coordination

    PlaceholderRange:
        Lightweight dataclass that tracks the position and extent of multimodal
        placeholder tokens in a token sequence, enabling proper alignment between
        vision embeddings and text tokens.

    MultiModalFeature:
        Represents a single image or video with its processed pixel values,
        grid information (for flat-patch models), and optional cached embeddings
        from the vision encoder.

    BatchedMultiModalInputs:
        Aggregates multiple features into batched tensors suitable for efficient
        model execution, with tracking of which batch positions have vision data.

Example:
    Basic usage with a vision-language model::

        >>> from easydel.inference.esurge.multimodal import MultiModalManager
        >>> from transformers import AutoProcessor
        >>>
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
        >>> manager = MultiModalManager(processor=processor)
        >>>
        >>> # Process images with automatic resolution bucketing
        >>> pixel_values, grid_thw = manager.process_images(images)
        >>>
        >>> # Extract media from OpenAI-style messages
        >>> images, videos = manager.extract_media_from_messages(messages)
        >>>
        >>> # Tokenize with proper placeholder insertion
        >>> token_ids = manager.tokenize_multimodal(messages, images=images)

    Using the caching system::

        >>> from easydel.inference.esurge.multimodal import VisionEncoderCache
        >>>
        >>> cache = VisionEncoderCache(capacity_mb=1024)
        >>> hash_key = cache.compute_hash(pixel_values)
        >>> cached = cache.get(hash_key)
        >>> if cached is None:
        ...     embeddings = vision_encoder(pixel_values)
        ...     cache.put(hash_key, embeddings)

See Also:
    - easydel.inference.esurge: The parent eSurge inference engine module
    - easydel.modules: VLM model implementations that consume these inputs
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
