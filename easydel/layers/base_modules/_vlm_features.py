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

"""Vision-Language Model (VLM) features for multimodal task modules.

This module provides reusable feature implementations for Vision-Language Models,
including vision encoder management, multimodal embedding merging, video processing,
and multi-dimensional RoPE position computation.

Features:
    VisionEncoderFeature: Vision tower management and feature extraction
    MultiModalMergeFeature: Embedding merge strategies for VLMs
    VideoProcessingFeature: Video grid computation and temporal processing
    MultiDimensionalRoPEFeature: mRoPE position computation for Qwen-style VLMs
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class VisionEncoderFeature:
    """Manages vision encoder feature extraction for VLM models.

    Handles selecting features from specific vision encoder layers and applying
    different feature selection strategies (skip CLS token, use all tokens, etc.).

    Attributes:
        vision_feature_layer: Which layer(s) to extract features from
        vision_feature_select_strategy: How to process the extracted features
    """

    def __init__(
        self,
        vision_feature_layer: int | list[int] = -1,
        vision_feature_select_strategy: str = "default",
    ):
        """Initialize vision encoder feature.

        Args:
            vision_feature_layer: Index of hidden state layer to use, or list of
                indices for multi-layer extraction. -1 means last layer.
            vision_feature_select_strategy: Strategy for processing features:
                - "default": Skip CLS token (first token), return rest
                - "full": Return all tokens including CLS
                - "pooled": Return only CLS token (first token)
        """
        valid_strategies = {"default", "full", "pooled"}
        if vision_feature_select_strategy not in valid_strategies:
            raise ValueError(
                f"vision_feature_select_strategy must be one of {valid_strategies}, got {vision_feature_select_strategy}"
            )

        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

    def extract_features(
        self,
        hidden_states: tuple[Array, ...],
        feature_layer: int | list[int] | None = None,
        select_strategy: str | None = None,
    ) -> Array:
        """Extract and process features from vision encoder hidden states.

        Args:
            hidden_states: Tuple of hidden states from vision encoder,
                one per layer plus embeddings.
            feature_layer: Override for which layer(s) to use.
            select_strategy: Override for selection strategy.

        Returns:
            Processed vision features ready for projection.
        """
        feature_layer = feature_layer if feature_layer is not None else self.vision_feature_layer
        select_strategy = select_strategy or self.vision_feature_select_strategy

        # Extract features from specified layer(s)
        if isinstance(feature_layer, int):
            selected_feature = hidden_states[feature_layer]
        else:
            # Multi-layer extraction: concatenate along feature dimension
            selected_feature = jnp.concatenate(
                [hidden_states[idx] for idx in feature_layer],
                axis=-1,
            )

        # Apply selection strategy
        if select_strategy == "default":
            # Skip CLS token (first token)
            return selected_feature[:, 1:]
        elif select_strategy == "full":
            return selected_feature
        elif select_strategy == "pooled":
            # Return only CLS token
            return selected_feature[:, 0:1]
        else:
            raise ValueError(f"Unknown selection strategy: {select_strategy}")

    def __repr__(self) -> str:
        return f"VisionEncoderFeature(layer={self.vision_feature_layer}, strategy={self.vision_feature_select_strategy})"


class MultiModalMergeFeature:
    """Handles multimodal embedding merge strategies for VLM models.

    Provides utilities for merging vision embeddings into text embeddings
    at placeholder token positions, supporting various merge strategies.

    Attributes:
        strategy: Merge strategy ("placeholder" or "mask")
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders
    """

    def __init__(
        self,
        strategy: str = "placeholder",
        image_token_id: int | None = None,
        video_token_id: int | None = None,
    ):
        """Initialize multimodal merge feature.

        Args:
            strategy: Merge strategy:
                - "placeholder": Replace placeholder tokens with vision embeddings
                - "mask": Use masking for conditional replacement
            image_token_id: Token ID used as placeholder for images
            video_token_id: Token ID used as placeholder for videos
        """
        valid_strategies = {"placeholder", "mask"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy}")

        self.strategy = strategy
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def get_placeholder_token_ids(self) -> list[int]:
        """Get list of all placeholder token IDs."""
        tokens = []
        if self.image_token_id is not None:
            tokens.append(self.image_token_id)
        if self.video_token_id is not None:
            tokens.append(self.video_token_id)
        return tokens

    def merge(
        self,
        input_ids: Int[Array, "batch seq_len"],
        inputs_embeds: Float[Array, "batch seq_len hidden"],
        multimodal_embeddings: Float[Array, "num_tokens hidden"],
        placeholder_token_id: int | list[int] | None = None,
    ) -> Float[Array, "batch seq_len hidden"]:
        """Merge multimodal embeddings into text embeddings.

        Uses efficient cumsum-based gathering to replace placeholder tokens
        with vision embeddings.

        Args:
            input_ids: Input token IDs
            inputs_embeds: Text embeddings
            multimodal_embeddings: Vision embeddings to merge
            placeholder_token_id: Override placeholder token ID(s)

        Returns:
            Merged embeddings with vision features at placeholder positions
        """
        if placeholder_token_id is None:
            placeholder_token_id = self.get_placeholder_token_ids()
            if not placeholder_token_id:
                return inputs_embeds
            if len(placeholder_token_id) == 1:
                placeholder_token_id = placeholder_token_id[0]

        # Create mask for multimodal positions
        if isinstance(placeholder_token_id, list):
            placeholder_token_id = jnp.array(placeholder_token_id)
            is_multimodal = jnp.isin(input_ids, placeholder_token_id)
        else:
            is_multimodal = input_ids == placeholder_token_id

        return self._merge_with_cumsum(inputs_embeds, is_multimodal, multimodal_embeddings)

    def _merge_with_cumsum(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden"],
        is_multimodal: Array,
        multimodal_embeddings: Float[Array, "num_tokens hidden"],
    ) -> Float[Array, "batch seq_len hidden"]:
        """Merge using efficient cumsum-based gathering.

        This implementation is based on Qwen's merge_multimodal_embeddings,
        which uses cumulative sum for efficient index gathering.

        Args:
            inputs_embeds: Text embeddings
            is_multimodal: Boolean mask indicating multimodal positions
            multimodal_embeddings: Vision embeddings to merge

        Returns:
            Merged embeddings
        """
        # Create dummy row for padding (index 0 maps to dummy)
        dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
        flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)

        # Use cumsum to create gather indices
        gather_indices = jnp.cumsum(is_multimodal)
        update_values = flattened_padded[gather_indices]

        # Conditionally replace embeddings
        condition = jnp.expand_dims(is_multimodal, axis=-1)
        return jnp.where(condition, update_values, inputs_embeds)

    def create_multimodal_mask(
        self,
        input_ids: Int[Array, "batch seq_len"],
    ) -> Array:
        """Create boolean mask indicating multimodal token positions.

        Args:
            input_ids: Input token IDs

        Returns:
            Boolean mask where True indicates multimodal positions
        """
        placeholder_ids = self.get_placeholder_token_ids()
        if not placeholder_ids:
            return jnp.zeros_like(input_ids, dtype=bool)

        placeholder_ids = jnp.array(placeholder_ids)
        return jnp.isin(input_ids, placeholder_ids)

    def __repr__(self) -> str:
        return (
            f"MultiModalMergeFeature(strategy={self.strategy}, "
            f"image_token={self.image_token_id}, video_token={self.video_token_id})"
        )


class VideoProcessingFeature:
    """Handles video-specific processing for VLM models.

    Provides utilities for computing video grids, temporal positions,
    and managing video token indices.

    Attributes:
        temporal_patch_size: Size of temporal patches
        tokens_per_second: Number of tokens generated per second of video
    """

    def __init__(
        self,
        temporal_patch_size: int = 2,
        tokens_per_second: float = 1.0,
    ):
        """Initialize video processing feature.

        Args:
            temporal_patch_size: Number of frames per temporal patch
            tokens_per_second: Temporal scaling factor for position computation
        """
        if temporal_patch_size <= 0:
            raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}")

        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second

    @property
    def is_enabled(self) -> bool:
        """Check if video processing is enabled."""
        return True

    def compute_video_grid(
        self,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int,
        spatial_merge_size: int = 1,
    ) -> tuple[int, int, int]:
        """Compute temporal, height, width grid for video.

        Args:
            num_frames: Number of video frames
            height: Frame height in pixels
            width: Frame width in pixels
            patch_size: Vision patch size
            spatial_merge_size: Spatial merge factor

        Returns:
            Tuple of (temporal_grid, height_grid, width_grid)
        """
        temporal_grid = num_frames // self.temporal_patch_size
        height_grid = (height // patch_size) // spatial_merge_size
        width_grid = (width // patch_size) // spatial_merge_size
        return (temporal_grid, height_grid, width_grid)

    def get_temporal_positions(
        self,
        video_grid_thw: Array,
        second_per_grid_ts: list[float] | None = None,
    ) -> Array:
        """Compute temporal position indices for video tokens.

        Args:
            video_grid_thw: Video grid shapes [num_videos, 3]
            second_per_grid_ts: Per-video seconds per temporal grid

        Returns:
            Temporal position indices
        """
        if second_per_grid_ts is None:
            second_per_grid_ts = [1.0] * len(video_grid_thw)

        all_positions = []
        for i, (t, h, w) in enumerate(video_grid_thw):
            t, h, w = int(t), int(h), int(w)
            temporal_scale = second_per_grid_ts[i] * self.tokens_per_second

            t_indices = (
                jnp.broadcast_to(
                    jnp.arange(t).reshape(-1, 1),
                    (t, h * w),
                )
                * temporal_scale
            )
            all_positions.append(t_indices.flatten().astype(jnp.int32))

        return jnp.concatenate(all_positions)

    def __repr__(self) -> str:
        return (
            f"VideoProcessingFeature(temporal_patch={self.temporal_patch_size}, "
            f"tokens_per_second={self.tokens_per_second})"
        )


class MultiDimensionalRoPEFeature:
    """Computes 3D mRoPE position IDs for multimodal tokens.

    Multi-dimensional Rotary Position Embedding (mRoPE) extends standard RoPE
    to handle 3D positions (temporal, height, width) for vision tokens.
    This feature computes the position IDs that are then used by
    MultiModalRotaryEmbedding for the actual embedding application.

    Note:
        Works with existing MultiModalRotaryEmbedding in
        easydel/layers/rotary_embedding.py which handles THW interleaving.

    Attributes:
        spatial_merge_size: Spatial merge factor for vision tokens
        mrope_section: Section sizes for T, H, W dimensions
    """

    def __init__(
        self,
        spatial_merge_size: int = 2,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
    ):
        """Initialize multi-dimensional RoPE feature.

        Args:
            spatial_merge_size: Spatial merge factor from vision config
            mrope_section: RoPE dimension allocation for (temporal, height, width)
        """
        self.spatial_merge_size = spatial_merge_size
        self.mrope_section = mrope_section

    @property
    def is_enabled(self) -> bool:
        """Check if mRoPE is enabled."""
        return True

    def compute_vision_positions(
        self,
        grid_thw: tuple[int, int, int],
        start_position: int = 0,
    ) -> tuple[Array, Array, Array]:
        """Compute 3D position indices for a single vision input.

        Args:
            grid_thw: Grid shape as (temporal, height, width)
            start_position: Starting position offset

        Returns:
            Tuple of (temporal_positions, height_positions, width_positions)
        """
        t, h, w = grid_thw

        # Temporal positions: same for all spatial positions within a frame
        t_pos = (
            jnp.broadcast_to(
                jnp.arange(t).reshape(t, 1, 1),
                (t, h, w),
            ).flatten()
            + start_position
        )

        # Height positions: repeated across width
        h_pos = (
            jnp.broadcast_to(
                jnp.arange(h).reshape(1, h, 1),
                (t, h, w),
            ).flatten()
            + start_position
        )

        # Width positions: repeated across height
        w_pos = (
            jnp.broadcast_to(
                jnp.arange(w).reshape(1, 1, w),
                (t, h, w),
            ).flatten()
            + start_position
        )

        return t_pos, h_pos, w_pos

    def get_default_position_ids(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> Array:
        """Get default position IDs for text-only sequences.

        For text tokens, all three dimensions use the same position index.

        Args:
            sequence_length: Length of the sequence
            batch_size: Batch size

        Returns:
            Position IDs of shape (3, batch_size, sequence_length)
        """
        text_positions = jnp.arange(sequence_length)
        text_positions = jnp.broadcast_to(
            text_positions[jnp.newaxis, :],
            (batch_size, sequence_length),
        )
        # Stack same positions for all 3 dimensions (T, H, W all same for text)
        return jnp.stack([text_positions, text_positions, text_positions], axis=0)

    def __repr__(self) -> str:
        return (
            f"MultiDimensionalRoPEFeature(spatial_merge={self.spatial_merge_size}, mrope_section={self.mrope_section})"
        )
