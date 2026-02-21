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

"""Multimodal data structures for vision-language model inference.

This module defines the core data structures for handling multimodal inputs
(images, videos) throughout the eSurge inference pipeline. These structures
enable proper tracking of vision data, batching across multiple requests,
and caching of encoder outputs.

The type system is designed to support:
    - Tracking placeholder positions in token sequences for proper
      alignment between vision embeddings and text tokens
    - Content-based hashing for efficient caching of vision encoder outputs
    - Batching multiple images/videos across requests for efficient
      GPU utilization during inference
    - Separation of concerns between preprocessing and encoding stages

Architecture Overview:
    1. Images/videos are first processed into MultiModalFeature objects
    2. Features from multiple requests are aggregated into BatchedMultiModalInputs
    3. The batch is passed through the vision encoder
    4. Embeddings are cached for future reuse via the mm_hash field

Classes:
    PlaceholderRange:
        Immutable dataclass tracking the position and extent of multimodal
        placeholder tokens in a token sequence.

    MultiModalFeature:
        Mutable dataclass representing a single image or video with its
        processed pixel values, grid information, and optional cached
        embeddings from the vision encoder.

    BatchedMultiModalInputs:
        Aggregates multiple features into batched numpy arrays suitable
        for efficient model forward passes.

Example:
    Creating and batching features::

        >>> # Create features from processed images
        >>> feature1 = MultiModalFeature.from_image(pixel_values1, grid_thw1)
        >>> feature2 = MultiModalFeature.from_image(pixel_values2, grid_thw2)
        >>>
        >>> # Batch them together for model input
        >>> batched = BatchedMultiModalInputs.from_features([feature1, feature2])
        >>>
        >>> # Pass to model
        >>> outputs = model(
        ...     pixel_values=batched.pixel_values,
        ...     image_grid_thw=batched.image_grid_thw,
        ...     ...
        ... )

See Also:
    VisionEncoderCache: Cache for storing computed embeddings
    MultiModalManager: High-level manager that creates these structures
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
    by placeholder tokens (e.g., <image>, <video>). This immutable dataclass
    tracks where those placeholders are located in the token sequence,
    enabling proper alignment between vision embeddings and text tokens
    during model execution.

    The frozen=True ensures instances are hashable and can be used as
    dictionary keys or in sets.

    Attributes:
        offset (int): Starting position (0-indexed token index) in the
            token sequence where the placeholder begins.
        length (int): Number of consecutive placeholder tokens. For
            flat-patch models, this typically equals the number of
            vision tokens (num_patches // spatial_merge_size^2).
        modality (str): Type of media represented by this placeholder.
            Common values are "image", "video", or "audio". Defaults
            to "image".

    Example:
        >>> # Image placeholder at position 5 with 256 tokens
        >>> placeholder = PlaceholderRange(offset=5, length=256, modality="image")
        >>> print(f"Placeholder spans tokens {placeholder.offset} to {placeholder.end}")
        Placeholder spans tokens 5 to 261

        >>> # Video placeholder
        >>> video_ph = PlaceholderRange(offset=10, length=512, modality="video")

    Note:
        The placeholder length depends on the vision encoder architecture:
        - For Qwen2-VL style models: num_patches // (spatial_merge_size^2)
        - For CLIP-style models: typically fixed (e.g., 576 for 14x14 patches)
    """

    offset: int
    length: int
    modality: str = "image"

    @property
    def end(self) -> int:
        """Return the end position (exclusive) of the placeholder.

        The end position is calculated as offset + length, following the
        standard Python convention for slice endpoints.

        Returns:
            int: The token index immediately after the last placeholder
                token. Can be used directly in slicing operations like
                tokens[placeholder.offset:placeholder.end].

        Example:
            >>> ph = PlaceholderRange(offset=5, length=10, modality="image")
            >>> tokens[ph.offset:ph.end]  # Gets all placeholder tokens
        """
        return self.offset + self.length


@dataclass
class MultiModalFeature:
    """Single multimodal feature with metadata for caching and batching.

    Represents a single image or video with its processed pixel values,
    grid information (for flat-patch models), and optional cached embeddings
    from the vision encoder.

    This class serves as the intermediate representation between raw media
    and batched model inputs. It supports:
        - Content-based hashing for cache lookups (mm_hash)
        - Tracking placeholder positions for token alignment
        - Storing cached embeddings to avoid recomputation
        - Memory management via clear_pixel_values()

    The class is mutable to allow setting cached embeddings after
    construction and clearing pixel values after encoding.

    Attributes:
        mm_hash (str): Content-based MD5 hash of the pixel values, used for
            cache lookups. Computed automatically via factory methods.
        modality (str): Type of media, either "image" or "video".
        pixel_values (np.ndarray | None): Processed pixel values as numpy array.
            Shape depends on the model architecture:
            - Flat-patch (GLM/Qwen): [num_patches, patch_dim]
            - Standard: [C, H, W] or [1, C, H, W]
            Can be None after clear_pixel_values() is called.
        grid_thw (np.ndarray | None): Grid shape array with shape [1, 3] or [3,]
            containing (T, H, W) - temporal frames, grid height, grid width.
            Used by flat-patch models to reconstruct spatial structure.
            None for models that don't use grid-based encoding.
        placeholder_range (PlaceholderRange | None): Position of this feature's
            placeholder tokens in the prompt. Set during tokenization.
            None if not yet determined.
        cached_embeddings (jax.Array | None): Cached vision encoder output.
            None until set via set_cached_embeddings(). Once set, the pixel
            values can be cleared to free memory.
        request_idx (int): Index of the request this feature belongs to in a
            batch. Used for tracking which request each feature came from.
            Defaults to 0.

    Example:
        Creating features::

            >>> feature = MultiModalFeature.from_image(pixel_values, grid_thw)
            >>> if feature.has_cached_embeddings:
            ...     embeddings = feature.cached_embeddings
            ... else:
            ...     embeddings = encode(feature.pixel_values)
            ...     feature.set_cached_embeddings(embeddings)
            ...     feature.clear_pixel_values()  # Free memory

    See Also:
        BatchedMultiModalInputs: For batching multiple features together
        VisionEncoderCache: For external caching of embeddings
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

        Factory method that creates a MultiModalFeature configured for image
        data. Automatically computes the content hash for cache lookups.

        Args:
            pixel_values (np.ndarray): Processed image pixel values. Shape
                depends on the model architecture:
                - Flat-patch (GLM/Qwen): [num_patches, patch_dim]
                - Standard: [C, H, W] or [1, C, H, W]
            grid_thw (np.ndarray | None): Grid shape (T, H, W) for flat-patch
                models. Should have shape [1, 3] or [3,]. For images, T is
                typically 1. Defaults to None.
            placeholder_range (PlaceholderRange | None): Position of image
                placeholder tokens in the prompt. Can be set later during
                tokenization. Defaults to None.
            request_idx (int): Index of the request this image belongs to
                in a batch. Defaults to 0.

        Returns:
            MultiModalFeature: A new feature instance configured for image
                data with modality="image" and computed mm_hash.

        Example:
            >>> feature = MultiModalFeature.from_image(
            ...     pixel_values=pixel_values,
            ...     grid_thw=np.array([[1, 24, 24]]),
            ...     request_idx=0
            ... )
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

        Factory method that creates a MultiModalFeature configured for video
        data. Automatically computes the content hash for cache lookups.

        Args:
            pixel_values (np.ndarray): Processed video pixel values. Shape
                depends on the model architecture:
                - Flat-patch (GLM/Qwen): [num_patches, patch_dim] where
                  num_patches = T * H * W (flattened spatiotemporal patches)
                - Standard: [T, C, H, W] or [1, T, C, H, W]
            grid_thw (np.ndarray | None): Grid shape (T, H, W) for flat-patch
                models. Should have shape [1, 3] or [3,]. T represents the
                number of temporal patch groups. Defaults to None.
            placeholder_range (PlaceholderRange | None): Position of video
                placeholder tokens in the prompt. Can be set later during
                tokenization. Defaults to None.
            request_idx (int): Index of the request this video belongs to
                in a batch. Defaults to 0.

        Returns:
            MultiModalFeature: A new feature instance configured for video
                data with modality="video" and computed mm_hash.

        Example:
            >>> feature = MultiModalFeature.from_video(
            ...     pixel_values=pixel_values,
            ...     grid_thw=np.array([[8, 24, 24]]),  # 8 temporal groups
            ...     request_idx=1
            ... )
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

        Creates an MD5 hash from the pixel values that can be used as a
        cache key. The hash includes both the shape and content of the
        array to ensure uniqueness.

        For large arrays (>1M elements), samples every 100th element to
        speed up hashing while maintaining reasonable uniqueness for
        typical image/video data.

        Args:
            pixel_values (np.ndarray): Pixel values to hash. Can be any
                shape (e.g., [H, W, C], [num_patches, patch_dim], etc.).

        Returns:
            str: 32-character hexadecimal MD5 hash string that uniquely
                identifies the content and shape of the input array.

        Note:
            This is an internal method. The hash is automatically computed
            by the from_image() and from_video() factory methods.
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
        """Check if embeddings have been cached.

        Returns:
            bool: True if cached_embeddings is not None, indicating that
                vision encoder output has been stored for this feature.

        Example:
            >>> if not feature.has_cached_embeddings:
            ...     embeddings = encode(feature.pixel_values)
            ...     feature.set_cached_embeddings(embeddings)
        """
        return self.cached_embeddings is not None

    def set_cached_embeddings(self, embeddings: jax.Array) -> None:
        """Cache the vision encoder embeddings.

        Stores the vision encoder output so it can be reused without
        recomputation. After calling this method, you can safely call
        clear_pixel_values() to free memory used by the raw pixel data.

        Args:
            embeddings (jax.Array): Vision encoder output embeddings.
                Typically has shape [num_tokens, hidden_size] where
                num_tokens depends on the model architecture.

        Example:
            >>> embeddings = vision_encoder(feature.pixel_values, feature.grid_thw)
            >>> feature.set_cached_embeddings(embeddings)
            >>> feature.clear_pixel_values()  # Free memory
        """
        self.cached_embeddings = embeddings

    def clear_pixel_values(self) -> None:
        """Clear pixel values after encoding to free memory.

        Sets both pixel_values and grid_thw to None. Call this after
        the vision encoder has processed the data and embeddings have
        been cached (either via set_cached_embeddings or in an external
        VisionEncoderCache).

        This is useful for reducing memory footprint during inference,
        as raw pixel data can be large and is no longer needed once
        embeddings are computed.

        Example:
            >>> feature.set_cached_embeddings(embeddings)
            >>> feature.clear_pixel_values()
            >>> assert feature.pixel_values is None
        """
        self.pixel_values = None
        self.grid_thw = None


@dataclass
class BatchedMultiModalInputs:
    """Batched multimodal inputs for model forward pass.

    Aggregates multiple MultiModalFeature instances into batched numpy arrays
    suitable for efficient model execution. Separates images and videos into
    distinct batches and maintains mappings from request indices to feature
    indices for proper routing during inference.

    This class is the final representation before passing vision data to the
    model. It handles:
        - Concatenation of pixel values from multiple features
        - Concatenation of grid shapes for flat-patch models
        - Tracking which features belong to which requests
        - Providing convenient accessors for batch statistics

    The class uses dataclass default_factory for mutable defaults to avoid
    the common Python pitfall of shared mutable default arguments.

    Attributes:
        pixel_values (np.ndarray | None): Concatenated image pixel values.
            For flat-patch models: [total_patches, patch_dim]
            For standard models: [total_images, C, H, W]
            None if no images in the batch.
        image_grid_thw (np.ndarray | None): Concatenated grid shapes for
            images with shape [num_images, 3]. Each row contains (T, H, W).
            None if no images or model doesn't use grids.
        pixel_values_videos (np.ndarray | None): Concatenated video pixel
            values, similar format to pixel_values but for videos.
            None if no videos in the batch.
        video_grid_thw (np.ndarray | None): Concatenated grid shapes for
            videos with shape [num_videos, 3]. Each row contains (T, H, W).
            None if no videos or model doesn't use grids.
        image_features (list[MultiModalFeature]): List of image features
            with their metadata, preserving individual feature information.
        video_features (list[MultiModalFeature]): List of video features
            with their metadata, preserving individual feature information.
        request_to_image_indices (dict[int, list[int]]): Maps request_idx to
            the list of indices in image_features belonging to that request.
        request_to_video_indices (dict[int, list[int]]): Maps request_idx to
            the list of indices in video_features belonging to that request.

    Example:
        >>> # Batch features from multiple requests
        >>> batched = BatchedMultiModalInputs.from_features([
        ...     feature1,  # request_idx=0
        ...     feature2,  # request_idx=0
        ...     feature3,  # request_idx=1
        ... ])
        >>>
        >>> # Pass to model
        >>> output = model(
        ...     pixel_values=batched.pixel_values,
        ...     image_grid_thw=batched.image_grid_thw,
        ...     ...
        ... )
        >>>
        >>> # Check batch contents
        >>> if batched.has_vision:
        ...     print(f"Batch has {batched.num_images} images and {batched.num_videos} videos")

    See Also:
        MultiModalFeature: Individual features that are batched together
        MultiModalManager.process_images_to_features: Creates features for batching
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

        Factory method that groups features by modality (image vs video)
        and concatenates their pixel values and grid shapes into batched
        arrays suitable for model forward passes.

        Features with None pixel_values are filtered out. The method builds
        request-to-index mappings to track which features belong to which
        inference request.

        Args:
            features (list[MultiModalFeature]): List of MultiModalFeature
                instances to batch together. Can contain a mix of image
                and video features from multiple requests.

        Returns:
            BatchedMultiModalInputs: A new instance with:
                - Concatenated pixel_values/pixel_values_videos arrays
                - Concatenated grid_thw arrays (if features have grids)
                - Preserved feature lists for metadata access
                - Request-to-index mappings for routing

        Example:
            >>> features = [
            ...     MultiModalFeature.from_image(img1_pv, grid1, request_idx=0),
            ...     MultiModalFeature.from_image(img2_pv, grid2, request_idx=0),
            ...     MultiModalFeature.from_video(vid_pv, vid_grid, request_idx=1),
            ... ]
            >>> batched = BatchedMultiModalInputs.from_features(features)
            >>> print(f"Images: {batched.num_images}, Videos: {batched.num_videos}")
            Images: 2, Videos: 1

        Note:
            Features are concatenated along axis 0. Ensure that all features
            of the same modality have compatible shapes along other dimensions.
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
        """Create an empty BatchedMultiModalInputs.

        Factory method that creates a BatchedMultiModalInputs with no
        features. Useful as a default value or for text-only requests.

        Returns:
            BatchedMultiModalInputs: An empty instance with all arrays
                set to None and all lists/dicts empty.

        Example:
            >>> empty_batch = BatchedMultiModalInputs.empty()
            >>> assert not empty_batch.has_vision
        """
        return cls()

    @property
    def has_images(self) -> bool:
        """Check if batch contains any images.

        Returns:
            bool: True if pixel_values is not None, indicating that
                at least one image feature was included in the batch.
        """
        return self.pixel_values is not None

    @property
    def has_videos(self) -> bool:
        """Check if batch contains any videos.

        Returns:
            bool: True if pixel_values_videos is not None, indicating
                that at least one video feature was included in the batch.
        """
        return self.pixel_values_videos is not None

    @property
    def has_vision(self) -> bool:
        """Check if batch contains any vision data (images or videos).

        Returns:
            bool: True if the batch contains at least one image or video.
                Equivalent to (has_images or has_videos).

        Example:
            >>> if batched.has_vision:
            ...     embeddings = vision_encoder(batched)
        """
        return self.has_images or self.has_videos

    @property
    def num_images(self) -> int:
        """Return number of images in the batch.

        Returns:
            int: Count of image features in the batch. Note this is the
                number of individual images, not the total number of
                patches or tokens.
        """
        return len(self.image_features)

    @property
    def num_videos(self) -> int:
        """Return number of videos in the batch.

        Returns:
            int: Count of video features in the batch. Note this is the
                number of individual videos, not the total number of
                frames or patches.
        """
        return len(self.video_features)

    def get_request_image_count(self, request_idx: int) -> int:
        """Get number of images for a specific request.

        Args:
            request_idx (int): The request index to query.

        Returns:
            int: Number of images belonging to the specified request.
                Returns 0 if the request_idx is not found.

        Example:
            >>> count = batched.get_request_image_count(request_idx=0)
            >>> print(f"Request 0 has {count} images")
        """
        return len(self.request_to_image_indices.get(request_idx, []))

    def get_request_video_count(self, request_idx: int) -> int:
        """Get number of videos for a specific request.

        Args:
            request_idx (int): The request index to query.

        Returns:
            int: Number of videos belonging to the specified request.
                Returns 0 if the request_idx is not found.

        Example:
            >>> count = batched.get_request_video_count(request_idx=1)
            >>> print(f"Request 1 has {count} videos")
        """
        return len(self.request_to_video_indices.get(request_idx, []))
