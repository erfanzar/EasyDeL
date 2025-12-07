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

"""Multimodal processing manager for vision-language models.

This module provides the MultiModalManager class for handling image and video
preprocessing, resolution bucketing, and integration with vision-language models.

Classes:
    MultiModalManager: Manages vision data processing and caching

Example:
    >>> manager = MultiModalManager(processor=processor)
    >>> pixel_values, grid_thw = manager.process_images([image])
    >>> prompt_ids = manager.tokenize_multimodal(messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .cache import VisionEncoderCache
from .types import MultiModalFeature

if TYPE_CHECKING:
    pass


# Default resolution buckets for compilation efficiency
DEFAULT_RESOLUTION_BUCKETS = [
    (384, 384),
    (512, 512),
    (768, 768),
    (1024, 1024),
]


class MultiModalManager:
    """Manager for multimodal (vision-language) processing.

    Handles image and video preprocessing with resolution bucketing to minimize
    JAX recompilation. Integrates with HuggingFace processors for tokenization
    and vision encoding preparation.

    Attributes:
        processor: HuggingFace processor for the vision-language model.
        resolution_buckets: List of (height, width) tuples for bucketing.
        cache: Vision encoder output cache.

    Example:
        >>> manager = MultiModalManager(processor=processor)
        >>> # Process images with automatic resolution bucketing
        >>> pixel_values, grid_thw = manager.process_images(images)
        >>> # Process OpenAI-style messages
        >>> images, videos = manager.extract_media_from_messages(messages)
    """

    def __init__(
        self,
        processor: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        cache_capacity_mb: int = 1024,
        enable_cache: bool = True,
    ):
        """Initialize MultiModalManager.

        Args:
            processor: HuggingFace processor (AutoProcessor) for the VLM.
            resolution_buckets: List of (H, W) resolution tuples for bucketing.
                Defaults to [(384, 384), (512, 512), (768, 768), (1024, 1024)].
            cache_capacity_mb: Vision encoder cache capacity in MB.
            enable_cache: Whether to enable vision encoder output caching.
        """
        self.processor = processor
        self.resolution_buckets = resolution_buckets or DEFAULT_RESOLUTION_BUCKETS
        self.cache = VisionEncoderCache(cache_capacity_mb) if enable_cache else None

    def resize_to_bucket(self, image: Image.Image) -> Image.Image:
        """Resize image to nearest resolution bucket.

        Selects the bucket with total pixels closest to the original image
        to minimize quality loss while enabling compilation reuse.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image at bucket resolution.
        """
        w, h = image.size
        original_pixels = w * h

        target = min(self.resolution_buckets, key=lambda b: abs(b[0] * b[1] - original_pixels))

        if (w, h) == target:
            return image

        return image.resize(target, Image.Resampling.LANCZOS)

    def process_images(
        self,
        images: list[Image.Image] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process images with resolution bucketing.

        Resizes images to bucket resolutions and processes them using
        the configured processor.

        Args:
            images: List of PIL Images to process.

        Returns:
            Tuple of (pixel_values, image_grid_thw) numpy arrays.
            Returns (None, None) if images is None or empty.
        """
        if not images:
            return None, None

        if self.processor is None:
            raise ValueError("Processor not configured for image processing")

        bucketed_images = [self.resize_to_bucket(img) for img in images]

        processed = self.processor(images=bucketed_images, return_tensors="np")

        pixel_values = processed.get("pixel_values")
        image_grid_thw = processed.get("image_grid_thw")

        return pixel_values, image_grid_thw

    def process_videos(
        self,
        videos: list[np.ndarray] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process videos with spatial resolution bucketing.

        Resizes video frames to bucket resolutions and processes them
        using the configured processor.

        Args:
            videos: List of video arrays with shape (T, H, W, C).

        Returns:
            Tuple of (pixel_values_videos, video_grid_thw) numpy arrays.
            Returns (None, None) if videos is None or empty.
        """
        if not videos:
            return None, None

        if self.processor is None:
            raise ValueError("Processor not configured for video processing")

        processed_videos = []
        for video in videos:
            _T, H, W, _C = video.shape

            target_h, target_w = min(
                self.resolution_buckets,
                key=lambda b: abs(b[0] * b[1] - H * W),
            )

            if (H, W) != (target_h, target_w):
                resized = np.stack(
                    [
                        np.array(Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS))
                        for frame in video
                    ]
                )
                processed_videos.append(resized)
            else:
                processed_videos.append(video)

        processed = self.processor(videos=processed_videos, return_tensors="np")

        pixel_values_videos = processed.get("pixel_values_videos")
        video_grid_thw = processed.get("video_grid_thw")

        return pixel_values_videos, video_grid_thw

    def extract_media_from_messages(
        self,
        messages: list[dict],
    ) -> tuple[list[Image.Image], list[np.ndarray]]:
        """Extract images and videos from OpenAI-style messages.

        Parses messages with content arrays containing image/video/text items
        and extracts the media for processing.

        Args:
            messages: List of message dicts in OpenAI format with content arrays.

        Returns:
            Tuple of (images, videos) lists.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": [
            ...         {"type": "image", "image": pil_image},
            ...         {"type": "text", "text": "Describe this"}
            ...     ]}
            ... ]
            >>> images, videos = manager.extract_media_from_messages(messages)
        """
        images = []
        videos = []

        for message in messages:
            content = message.get("content", [])

            if isinstance(content, str):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")

                if item_type == "image":
                    image = item.get("image")
                    if image is not None:
                        if isinstance(image, str):
                            image = Image.open(image)
                        images.append(image)

                elif item_type == "video":
                    video = item.get("video")
                    if video is not None:
                        videos.append(video)

                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        if url.startswith("data:"):
                            import base64
                            import io

                            header, data = url.split(",", 1)
                            image_data = base64.b64decode(data)
                            image = Image.open(io.BytesIO(image_data))
                        else:
                            pass
                        if "image" in locals():
                            images.append(image)

        return images, videos

    def tokenize_multimodal(
        self,
        messages: list[dict],
        images: list[Image.Image] | None = None,
        videos: list[np.ndarray] | None = None,
    ) -> list[int]:
        """Tokenize multimodal messages with placeholder insertion.

        Uses the processor's chat template to convert messages to token IDs,
        inserting appropriate placeholder tokens for images and videos.

        Args:
            messages: OpenAI-style messages list.
            images: Preprocessed images (optional, extracts from messages if None).
            videos: Preprocessed videos (optional, extracts from messages if None).

        Returns:
            List of token IDs with image/video placeholders inserted.
        """
        if self.processor is None:
            raise ValueError("Processor not configured for tokenization")

        if images is None and videos is None:
            images, videos = self.extract_media_from_messages(messages)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="np",
            padding=False,
        )

        return inputs["input_ids"][0].tolist()

    def clear_cache(self) -> None:
        """Clear the vision encoder cache."""
        if self.cache is not None:
            self.cache.clear()

    def process_images_to_features(
        self,
        images: list[Image.Image] | None,
        request_idx: int = 0,
    ) -> list[MultiModalFeature]:
        """Process images and create MultiModalFeature objects.

        Processes images with resolution bucketing and creates feature objects
        with content-based hashing for cache lookups.

        Args:
            images: List of PIL Images to process.
            request_idx: Index of the request in a batch.

        Returns:
            List of MultiModalFeature objects with pixel values and hashes.
        """
        if not images:
            return []

        pixel_values, image_grid_thw = self.process_images(images)

        if pixel_values is None:
            return []

        features = []
        num_images = len(images)
        for i in range(num_images):
            if pixel_values.ndim > 3:
                single_pv = pixel_values[i : i + 1]
            else:
                single_pv = pixel_values

            single_grid = None
            if image_grid_thw is not None and i < len(image_grid_thw):
                single_grid = image_grid_thw[i : i + 1]

            feature = MultiModalFeature.from_image(
                pixel_values=single_pv,
                grid_thw=single_grid,
                request_idx=request_idx,
            )
            features.append(feature)

        return features

    def process_videos_to_features(
        self,
        videos: list[np.ndarray] | None,
        request_idx: int = 0,
    ) -> list[MultiModalFeature]:
        """Process videos and create MultiModalFeature objects.

        Processes videos with resolution bucketing and creates feature objects
        with content-based hashing for cache lookups.

        Args:
            videos: List of video arrays with shape (T, H, W, C).
            request_idx: Index of the request in a batch.

        Returns:
            List of MultiModalFeature objects with pixel values and hashes.
        """
        if not videos:
            return []

        pixel_values_videos, video_grid_thw = self.process_videos(videos)

        if pixel_values_videos is None:
            return []

        features = []
        num_videos = len(videos)
        for i in range(num_videos):
            if pixel_values_videos.ndim > 4:
                single_pv = pixel_values_videos[i : i + 1]
            else:
                single_pv = pixel_values_videos

            single_grid = None
            if video_grid_thw is not None and i < len(video_grid_thw):
                single_grid = video_grid_thw[i : i + 1]

            feature = MultiModalFeature.from_video(
                pixel_values=single_pv,
                grid_thw=single_grid,
                request_idx=request_idx,
            )
            features.append(feature)

        return features

    def get_cache_stats(self) -> dict | None:
        """Get vision encoder cache statistics.

        Returns:
            Dictionary with cache stats or None if cache is disabled.
        """
        if self.cache is not None:
            return self.cache.get_stats()
        return None
