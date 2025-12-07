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

"""Multimodal data structures for vision-language model inference.

This module defines the core data structures for handling multimodal inputs
(images, videos) throughout the eSurge inference pipeline. These structures
enable proper tracking of vision data, batching across multiple requests,
and caching of encoder outputs.

Classes:
    PlaceholderRange: Tracks multimodal placeholder positions in prompts
    MultiModalFeature: Single multimodal feature with metadata and caching
    BatchedMultiModalInputs: Batched inputs for model forward pass

Example:
    >>> feature = MultiModalFeature.from_image(pixel_values, grid_thw)
    >>> batched = BatchedMultiModalInputs.from_features([feature1, feature2])
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


@dataclass(frozen=True)
class PlaceholderRange:
    """Tracks multimodal placeholder positions in a token sequence.

    When images or videos are inserted into prompts, they are represented
    by placeholder tokens. This class tracks where those placeholders are
    located in the token sequence.

    Attributes:
        offset: Starting position (token index) in the sequence.
        length: Number of placeholder tokens.
        modality: Type of media ("image", "video", "audio").

    Example:
        >>> # Image placeholder at position 5 with 256 tokens
        >>> placeholder = PlaceholderRange(offset=5, length=256, modality="image")
    """

    offset: int
    length: int
    modality: str = "image"

    @property
    def end(self) -> int:
        """Return the end position (exclusive) of the placeholder."""
        return self.offset + self.length


@dataclass
class MultiModalFeature:
    """Single multimodal feature with metadata for caching and batching.

    Represents a single image or video with its processed pixel values,
    grid information, and optional cached embeddings from the vision encoder.

    Attributes:
        mm_hash: Content-based hash for caching lookups.
        modality: Type of media ("image" or "video").
        pixel_values: Processed pixel values as numpy array.
        grid_thw: Grid shape (T, H, W) for the media.
        placeholder_range: Position of this feature's placeholders in the prompt.
        cached_embeddings: Cached vision encoder output (None until computed).
        request_idx: Index of the request this feature belongs to in a batch.

    Example:
        >>> feature = MultiModalFeature.from_image(pixel_values, grid_thw)
        >>> if feature.has_cached_embeddings:
        ...     embeddings = feature.cached_embeddings
    """

    mm_hash: str
    modality: str
    pixel_values: np.ndarray | None
    grid_thw: np.ndarray | None
    placeholder_range: PlaceholderRange | None = None
    cached_embeddings: jax.Array | None = None
    request_idx: int = 0

    @classmethod
    def from_image(
        cls,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray | None = None,
        placeholder_range: PlaceholderRange | None = None,
        request_idx: int = 0,
    ) -> MultiModalFeature:
        """Create a feature from image data.

        Args:
            pixel_values: Processed image pixel values.
            grid_thw: Grid shape (T, H, W) for the image.
            placeholder_range: Position of image placeholders.
            request_idx: Index of the request in a batch.

        Returns:
            MultiModalFeature configured for image data.
        """
        mm_hash = cls._compute_hash(pixel_values)
        return cls(
            mm_hash=mm_hash,
            modality="image",
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            placeholder_range=placeholder_range,
            request_idx=request_idx,
        )

    @classmethod
    def from_video(
        cls,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray | None = None,
        placeholder_range: PlaceholderRange | None = None,
        request_idx: int = 0,
    ) -> MultiModalFeature:
        """Create a feature from video data.

        Args:
            pixel_values: Processed video pixel values.
            grid_thw: Grid shape (T, H, W) for the video.
            placeholder_range: Position of video placeholders.
            request_idx: Index of the request in a batch.

        Returns:
            MultiModalFeature configured for video data.
        """
        mm_hash = cls._compute_hash(pixel_values)
        return cls(
            mm_hash=mm_hash,
            modality="video",
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            placeholder_range=placeholder_range,
            request_idx=request_idx,
        )

    @staticmethod
    def _compute_hash(pixel_values: np.ndarray) -> str:
        """Compute content-based hash for caching.

        Args:
            pixel_values: Pixel values to hash.

        Returns:
            MD5 hash string including shape information.
        """
        shape_bytes = np.array(pixel_values.shape, dtype=np.int32).tobytes()
        if pixel_values.size > 1_000_000:
            sampled = pixel_values.flat[::100]
            content_bytes = sampled.tobytes()
        else:
            content_bytes = pixel_values.tobytes()
        return hashlib.md5(shape_bytes + content_bytes).hexdigest()

    @property
    def has_cached_embeddings(self) -> bool:
        """Check if embeddings have been cached."""
        return self.cached_embeddings is not None

    def set_cached_embeddings(self, embeddings: jax.Array) -> None:
        """Cache the vision encoder embeddings.

        Args:
            embeddings: Vision encoder output to cache.
        """
        self.cached_embeddings = embeddings

    def clear_pixel_values(self) -> None:
        """Clear pixel values after encoding to free memory."""
        self.pixel_values = None
        self.grid_thw = None


@dataclass
class BatchedMultiModalInputs:
    """Batched multimodal inputs for model forward pass.

    Aggregates multiple MultiModalFeature instances into batched tensors
    suitable for efficient model execution. Tracks which batch positions
    have vision data via feature mapping.

    Attributes:
        pixel_values: Concatenated image pixel values [total_images, ...].
        image_grid_thw: Concatenated grid shapes [total_images, 3].
        pixel_values_videos: Concatenated video pixel values.
        video_grid_thw: Concatenated video grid shapes.
        image_features: List of image features with their metadata.
        video_features: List of video features with their metadata.
        request_to_image_indices: Maps request_idx to list of image indices.
        request_to_video_indices: Maps request_idx to list of video indices.

    Example:
        >>> batched = BatchedMultiModalInputs.from_features(features)
        >>> # Pass to model
        >>> output = model(pixel_values=batched.pixel_values, ...)
    """

    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None
    pixel_values_videos: np.ndarray | None = None
    video_grid_thw: np.ndarray | None = None
    image_features: list[MultiModalFeature] = field(default_factory=list)
    video_features: list[MultiModalFeature] = field(default_factory=list)
    request_to_image_indices: dict[int, list[int]] = field(default_factory=dict)
    request_to_video_indices: dict[int, list[int]] = field(default_factory=dict)

    @classmethod
    def from_features(
        cls,
        features: list[MultiModalFeature],
    ) -> BatchedMultiModalInputs:
        """Create batched inputs from a list of features.

        Groups features by modality (image vs video) and concatenates
        their pixel values and grid shapes into batched arrays.

        Args:
            features: List of MultiModalFeature instances.

        Returns:
            BatchedMultiModalInputs with concatenated data.
        """
        image_features = []
        video_features = []
        request_to_image_indices: dict[int, list[int]] = {}
        request_to_video_indices: dict[int, list[int]] = {}

        for feat in features:
            if feat.modality == "image" and feat.pixel_values is not None:
                idx = len(image_features)
                image_features.append(feat)
                if feat.request_idx not in request_to_image_indices:
                    request_to_image_indices[feat.request_idx] = []
                request_to_image_indices[feat.request_idx].append(idx)
            elif feat.modality == "video" and feat.pixel_values is not None:
                idx = len(video_features)
                video_features.append(feat)
                if feat.request_idx not in request_to_video_indices:
                    request_to_video_indices[feat.request_idx] = []
                request_to_video_indices[feat.request_idx].append(idx)

        # Concatenate image data
        pixel_values = None
        image_grid_thw = None
        if image_features:
            pixel_values_list = [f.pixel_values for f in image_features if f.pixel_values is not None]
            if pixel_values_list:
                pixel_values = np.concatenate(pixel_values_list, axis=0)
            grid_list = [f.grid_thw for f in image_features if f.grid_thw is not None]
            if grid_list:
                image_grid_thw = np.concatenate(grid_list, axis=0)

        # Concatenate video data
        pixel_values_videos = None
        video_grid_thw = None
        if video_features:
            video_pv_list = [f.pixel_values for f in video_features if f.pixel_values is not None]
            if video_pv_list:
                pixel_values_videos = np.concatenate(video_pv_list, axis=0)
            video_grid_list = [f.grid_thw for f in video_features if f.grid_thw is not None]
            if video_grid_list:
                video_grid_thw = np.concatenate(video_grid_list, axis=0)

        return cls(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            image_features=image_features,
            video_features=video_features,
            request_to_image_indices=request_to_image_indices,
            request_to_video_indices=request_to_video_indices,
        )

    @classmethod
    def empty(cls) -> BatchedMultiModalInputs:
        """Create an empty BatchedMultiModalInputs."""
        return cls()

    @property
    def has_images(self) -> bool:
        """Check if batch contains any images."""
        return self.pixel_values is not None

    @property
    def has_videos(self) -> bool:
        """Check if batch contains any videos."""
        return self.pixel_values_videos is not None

    @property
    def has_vision(self) -> bool:
        """Check if batch contains any vision data."""
        return self.has_images or self.has_videos

    @property
    def num_images(self) -> int:
        """Return number of images in the batch."""
        return len(self.image_features)

    @property
    def num_videos(self) -> int:
        """Return number of videos in the batch."""
        return len(self.video_features)

    def get_request_image_count(self, request_idx: int) -> int:
        """Get number of images for a specific request."""
        return len(self.request_to_image_indices.get(request_idx, []))

    def get_request_video_count(self, request_idx: int) -> int:
        """Get number of videos for a specific request."""
        return len(self.request_to_video_indices.get(request_idx, []))
