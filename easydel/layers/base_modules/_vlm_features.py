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

These features encapsulate the specialized functionality needed by VLMs like
LLaVA, Qwen2-VL, Qwen3-VL, Gemma3, AyaVision, Mistral3, and Llama4.

Features:
    VisionEncoderFeature: Vision tower management and feature extraction from
        specific layers with configurable selection strategies.
    MultiModalMergeFeature: Efficient embedding merge strategies for VLMs using
        cumsum-based gathering to replace placeholder tokens.
    VideoProcessingFeature: Video grid computation, temporal position calculation,
        and video token management for video-capable VLMs.
    MultiDimensionalRoPEFeature: 3D mRoPE position computation for Qwen-style
        VLMs that use separate positional embeddings for temporal, height,
        and width dimensions.

Design Philosophy:
    Each feature handles one aspect of VLM processing, allowing models to
    compose only the features they need. For example, a basic image-only VLM
    might use VisionEncoderFeature and MultiModalMergeFeature, while a
    Qwen-style VLM would additionally use VideoProcessingFeature and
    MultiDimensionalRoPEFeature.

Example:
    Using VLM features in a model:

    ```python
    class MyVLM(BaseVisionLanguageModule):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            # Features are initialized in parent class

        def get_image_features(self, pixel_values, **kwargs):
            # Use vision encoder feature to extract features
            vision_outputs = self.vision_tower(
                pixel_values, output_hidden_states=True
            )
            selected = self._vision_encoder_feature.extract_features(
                vision_outputs.hidden_states
            )
            return self.projector(selected)
    ```

See Also:
    - BaseVisionLanguageModule: Uses these features
    - _features: General task module features
    - easydel.layers.rotary_embedding.MultiModalRotaryEmbedding: mRoPE implementation
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class VisionEncoderFeature:
    """Manage vision encoder feature extraction for VLM models.

    Handles selecting features from specific vision encoder layers and applying
    different feature selection strategies (skip CLS token, use all tokens, etc.).

    Vision transformers produce hidden states at each layer. Different VLMs may
    want to use features from different layers (last layer, intermediate layers,
    or concatenation of multiple layers) and may want to process them differently
    (skip CLS token, use only CLS, etc.).

    Attributes:
        vision_feature_layer (int | list[int]): Which layer(s) to extract features
            from. -1 means last layer.
        vision_feature_select_strategy (str): How to process the extracted features.

    Example:
        ```python
        # Use second-to-last layer, skip CLS token
        feature = VisionEncoderFeature(
            vision_feature_layer=-2,
            vision_feature_select_strategy="default"
        )

        # In forward pass
        vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
        selected = feature.extract_features(vision_outputs.hidden_states)
        projected = projector(selected)
        ```

    Note:
        The hidden_states tuple typically includes the embedding layer output
        followed by each transformer layer's output, so hidden_states[-1] is
        the last layer and hidden_states[0] is the embedding output.
    """

    def __init__(
        self,
        vision_feature_layer: int | list[int] = -1,
        vision_feature_select_strategy: str = "default",
    ):
        """Initialize vision encoder feature.

        Args:
            vision_feature_layer: Index of hidden state layer to use, or list of
                indices for multi-layer extraction. Negative indices work like
                Python list indexing (-1 = last layer, -2 = second to last, etc.).
                When a list is provided, features from all specified layers are
                concatenated along the feature dimension. Defaults to -1 (last layer).
            vision_feature_select_strategy: Strategy for processing features after
                layer extraction. Options:
                    - "default": Skip CLS token (first token), return rest. This is
                      appropriate for most VLMs where CLS is not needed.
                    - "full": Return all tokens including CLS. Use when the full
                      sequence including CLS is needed.
                    - "pooled": Return only CLS token (first token). Use for
                      classification-style vision features.
                Defaults to "default".

        Raises:
            ValueError: If vision_feature_select_strategy is not one of the
                valid options ("default", "full", "pooled").

        Example:
            ```python
            # Standard LLaVA-style: last layer, skip CLS
            feature = VisionEncoderFeature(
                vision_feature_layer=-1,
                vision_feature_select_strategy="default"
            )

            # Multi-layer concatenation for richer features
            feature = VisionEncoderFeature(
                vision_feature_layer=[-4, -3, -2, -1],
                vision_feature_select_strategy="default"
            )

            # Use intermediate layer
            feature = VisionEncoderFeature(
                vision_feature_layer=12,
                vision_feature_select_strategy="default"
            )
            ```
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

        Selects features from the specified layer(s) and applies the selection
        strategy to produce the final vision features ready for projection.

        Args:
            hidden_states: Tuple of hidden states from vision encoder, one per
                layer plus the embedding output. Typically has shape
                (num_layers + 1,) where each element is
                (batch_size, num_patches + 1, hidden_size).
            feature_layer: Override for which layer(s) to use. If None, uses
                the value from initialization.
            select_strategy: Override for selection strategy. If None, uses
                the value from initialization.

        Returns:
            Processed vision features of shape:
                - (batch_size, num_patches, hidden_size) for "default" strategy
                - (batch_size, num_patches + 1, hidden_size) for "full" strategy
                - (batch_size, 1, hidden_size) for "pooled" strategy
            If multi-layer extraction is used, hidden_size is multiplied by
            the number of layers.

        Example:
            ```python
            feature = VisionEncoderFeature(vision_feature_layer=-1)

            # Standard extraction
            vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
            selected = feature.extract_features(vision_outputs.hidden_states)

            # Override layer at runtime
            selected_deep = feature.extract_features(
                vision_outputs.hidden_states,
                feature_layer=-4
            )
            ```
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
        """Return string representation of the feature.

        Returns:
            String showing feature class name, layer selection, and strategy.
        """
        return f"VisionEncoderFeature(layer={self.vision_feature_layer}, strategy={self.vision_feature_select_strategy})"


class MultiModalMergeFeature:
    """Handle multimodal embedding merge strategies for VLM models.

    Provides utilities for merging vision embeddings into text embeddings
    at placeholder token positions. This is the core operation that enables
    VLMs to process both text and images in a unified sequence.

    The merge operation replaces placeholder tokens (e.g., <image>, <video>)
    in the text embedding sequence with the corresponding vision embeddings.
    This is done efficiently using cumsum-based gathering.

    Attributes:
        strategy (str): Merge strategy ("placeholder" or "mask").
        image_token_id (int | None): Token ID for image placeholders.
        video_token_id (int | None): Token ID for video placeholders.

    Example:
        ```python
        merge_feature = MultiModalMergeFeature(
            strategy="placeholder",
            image_token_id=32000,  # <image> token
            video_token_id=32001,  # <video> token
        )

        # Merge vision features into text embeddings
        merged = merge_feature.merge(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            multimodal_embeddings=vision_features,
        )
        ```

    Note:
        The cumsum-based merging is more efficient than scatter operations
        for this use case and is JIT-compatible.
    """

    def __init__(
        self,
        strategy: str = "placeholder",
        image_token_id: int | None = None,
        video_token_id: int | None = None,
    ):
        """Initialize multimodal merge feature.

        Args:
            strategy: Merge strategy. Options:
                - "placeholder": Replace placeholder tokens with vision embeddings.
                  This is the standard approach where the tokenizer produces
                  placeholder tokens that mark where images should be inserted.
                - "mask": Use masking for conditional replacement. Less common
                  but useful for certain architectures.
                Defaults to "placeholder".
            image_token_id: Token ID used as placeholder for images. This is
                typically a special token added to the vocabulary (e.g., <image>).
                Set to None if images are not supported.
            video_token_id: Token ID used as placeholder for videos. Similar to
                image_token_id but for video inputs. Set to None if videos are
                not supported.

        Raises:
            ValueError: If strategy is not one of the valid options.

        Example:
            ```python
            # Image-only VLM
            merge = MultiModalMergeFeature(
                strategy="placeholder",
                image_token_id=32000,
            )

            # Image and video VLM
            merge = MultiModalMergeFeature(
                strategy="placeholder",
                image_token_id=151655,  # Qwen2-VL image token
                video_token_id=151656,  # Qwen2-VL video token
            )
            ```
        """
        valid_strategies = {"placeholder", "mask"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy}")

        self.strategy = strategy
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def get_placeholder_token_ids(self) -> list[int]:
        """Get list of all configured placeholder token IDs.

        Returns:
            List of token IDs that are used as placeholders for multimodal
            content. The list may be empty if no placeholder tokens are configured.

        Example:
            ```python
            merge = MultiModalMergeFeature(
                image_token_id=32000,
                video_token_id=32001,
            )
            tokens = merge.get_placeholder_token_ids()
            # tokens = [32000, 32001]
            ```
        """
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
        with vision embeddings. This method is JIT-compatible and vectorized.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Contains placeholder tokens at positions where vision features
                should be inserted.
            inputs_embeds: Text embeddings of shape (batch_size, sequence_length,
                hidden_dim). The embeddings at placeholder positions will be
                replaced with multimodal_embeddings.
            multimodal_embeddings: Vision embeddings of shape (num_tokens, hidden_dim)
                to merge into the text sequence. The number of tokens should match
                the total number of placeholder tokens in input_ids.
            placeholder_token_id: Override placeholder token ID(s). If None, uses
                the configured image_token_id and video_token_id. Can be a single
                int or list of ints for multiple placeholder types.

        Returns:
            Merged embeddings of shape (batch_size, sequence_length, hidden_dim)
            with vision features at placeholder positions and original text
            embeddings elsewhere.

        Example:
            ```python
            merge = MultiModalMergeFeature(image_token_id=32000)

            # input_ids: [101, 32000, 32000, 32000, 102]
            # (3 image tokens representing 3 patches)
            # vision_features: (3, 768) - 3 patch embeddings

            merged = merge.merge(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                multimodal_embeddings=vision_features,
            )
            # merged[0, 1:4] now contains the vision features
            ```

        Note:
            The cumsum-based algorithm works by:
            1. Creating a mask of multimodal positions
            2. Computing cumulative sum to create gather indices
            3. Using where() to conditionally replace embeddings
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
        which uses cumulative sum for efficient index gathering. The algorithm
        is vectorized and JIT-compatible.

        Args:
            inputs_embeds: Text embeddings of shape (batch_size, seq_len, hidden_dim).
            is_multimodal: Boolean mask of shape (batch_size, seq_len) indicating
                which positions contain multimodal placeholders.
            multimodal_embeddings: Vision embeddings of shape (num_tokens, hidden_dim)
                to merge at the True positions in is_multimodal.

        Returns:
            Merged embeddings with vision features at multimodal positions.

        Note:
            The algorithm:
            1. Prepends a dummy zero row to multimodal_embeddings (index 0)
            2. Computes cumsum of is_multimodal to get 1-indexed gather positions
            3. Gathers from padded array (non-multimodal positions get dummy row)
            4. Uses where() to select between original and gathered embeddings
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

        Useful for operations that need to know which positions contain
        vision placeholders without performing the actual merge.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).

        Returns:
            Boolean mask of shape (batch_size, sequence_length) where True
            indicates positions with multimodal placeholder tokens.

        Example:
            ```python
            merge = MultiModalMergeFeature(image_token_id=32000)

            mask = merge.create_multimodal_mask(input_ids)
            # mask[i, j] is True where input_ids[i, j] == 32000
            ```
        """
        placeholder_ids = self.get_placeholder_token_ids()
        if not placeholder_ids:
            return jnp.zeros_like(input_ids, dtype=bool)

        placeholder_ids = jnp.array(placeholder_ids)
        return jnp.isin(input_ids, placeholder_ids)

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing feature class name, strategy, and token IDs.
        """
        return (
            f"MultiModalMergeFeature(strategy={self.strategy}, "
            f"image_token={self.image_token_id}, video_token={self.video_token_id})"
        )


class VideoProcessingFeature:
    """Handle video-specific processing for VLM models.

    Provides utilities for computing video grids, temporal positions,
    and managing video token indices. Videos are processed as sequences
    of frames that are patchified and flattened into tokens.

    This feature handles the temporal dimension that distinguishes videos
    from images, computing the grid shape (temporal x height x width) and
    temporal position indices for each video token.

    Attributes:
        temporal_patch_size (int): Number of frames per temporal patch.
        tokens_per_second (float): Temporal scaling factor for positions.

    Example:
        ```python
        video_feature = VideoProcessingFeature(
            temporal_patch_size=2,  # Group 2 frames per temporal patch
            tokens_per_second=1.0,
        )

        # Compute grid for a 16-frame, 224x224 video with patch_size=14
        t, h, w = video_feature.compute_video_grid(
            num_frames=16,
            height=224,
            width=224,
            patch_size=14,
        )
        # t=8 (16/2), h=16 (224/14), w=16 (224/14)
        # Total tokens: 8 * 16 * 16 = 2048
        ```

    Note:
        This feature computes grid dimensions and temporal positions but
        does not perform the actual video encoding, which is handled by
        the vision tower.
    """

    def __init__(
        self,
        temporal_patch_size: int = 2,
        tokens_per_second: float = 1.0,
    ):
        """Initialize video processing feature.

        Args:
            temporal_patch_size: Number of video frames that are grouped into
                one temporal patch. For example, with temporal_patch_size=2,
                a 16-frame video produces 8 temporal positions. Must be positive.
                Defaults to 2.
            tokens_per_second: Temporal scaling factor for position computation.
                Higher values spread positions further apart, which can help
                the model distinguish frames that are far apart in time.
                Defaults to 1.0.

        Raises:
            ValueError: If temporal_patch_size is not positive.

        Example:
            ```python
            # Standard video processing
            feature = VideoProcessingFeature(
                temporal_patch_size=2,
                tokens_per_second=1.0,
            )

            # More aggressive temporal compression
            feature = VideoProcessingFeature(
                temporal_patch_size=4,
                tokens_per_second=0.5,
            )
            ```
        """
        if temporal_patch_size <= 0:
            raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}")

        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second

    @property
    def is_enabled(self) -> bool:
        """Check if video processing is enabled.

        Returns:
            Always True for this feature. Used for feature presence checks.
        """
        return True

    def compute_video_grid(
        self,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int,
        spatial_merge_size: int = 1,
    ) -> tuple[int, int, int]:
        """Compute temporal, height, width grid dimensions for video.

        Given video dimensions and processing parameters, computes the grid
        shape that determines how many tokens the video will produce.

        Args:
            num_frames: Number of frames in the video.
            height: Frame height in pixels.
            width: Frame width in pixels.
            patch_size: Vision patch size in pixels. Each patch becomes one
                spatial token.
            spatial_merge_size: Spatial merge factor. If >1, neighboring patches
                are merged, reducing spatial resolution. Defaults to 1 (no merge).

        Returns:
            Tuple of (temporal_grid, height_grid, width_grid) representing
            the number of tokens along each dimension. Total tokens =
            temporal_grid * height_grid * width_grid.

        Example:
            ```python
            feature = VideoProcessingFeature(temporal_patch_size=2)

            # 16-frame 224x224 video, patch_size=14
            t, h, w = feature.compute_video_grid(
                num_frames=16,
                height=224,
                width=224,
                patch_size=14,
            )
            # t=8, h=16, w=16 -> 2048 tokens

            # With spatial merge
            t, h, w = feature.compute_video_grid(
                num_frames=16,
                height=224,
                width=224,
                patch_size=14,
                spatial_merge_size=2,
            )
            # t=8, h=8, w=8 -> 512 tokens
            ```
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

        Generates temporal position indices that indicate when each token
        occurs in the video. These positions can be used for temporal
        positional embeddings.

        Args:
            video_grid_thw: Video grid shapes of shape (num_videos, 3) where
                each row is (temporal, height, width) for one video.
            second_per_grid_ts: Per-video seconds per temporal grid position.
                If provided, should have length equal to num_videos. Used to
                scale temporal positions based on actual video frame rate.
                Defaults to None (uses 1.0 for all videos).

        Returns:
            Concatenated temporal position indices for all video tokens.
            Shape is (total_tokens,) where total_tokens is the sum of
            t*h*w for all videos.

        Example:
            ```python
            feature = VideoProcessingFeature(
                temporal_patch_size=2,
                tokens_per_second=1.0,
            )

            # Two videos: 4x4x4 and 2x4x4 grids
            video_grid_thw = jnp.array([[4, 4, 4], [2, 4, 4]])

            positions = feature.get_temporal_positions(video_grid_thw)
            # First video: t=0,0,0,...,0 (16 times), t=1,1,1,...,1 (16 times), etc.
            # Shape: (64 + 32,) = (96,)
            ```
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
        """Return string representation of the feature.

        Returns:
            String showing feature class name and configuration parameters.
        """
        return (
            f"VideoProcessingFeature(temporal_patch={self.temporal_patch_size}, "
            f"tokens_per_second={self.tokens_per_second})"
        )


class MultiDimensionalRoPEFeature:
    """Compute 3D mRoPE position IDs for multimodal tokens.

    Multi-dimensional Rotary Position Embedding (mRoPE) extends standard RoPE
    to handle 3D positions (temporal, height, width) for vision tokens.
    This is used by Qwen-style VLMs where vision tokens have spatial and
    temporal structure that should be reflected in their positional embeddings.

    Unlike standard 1D position IDs, mRoPE uses separate position indices for
    each dimension. The RoPE frequencies are split into sections, with each
    section handling one dimension.

    This feature computes the position IDs that are then used by
    MultiModalRotaryEmbedding for the actual embedding application.

    Attributes:
        spatial_merge_size (int): Spatial merge factor for vision tokens.
        mrope_section (tuple[int, int, int]): Dimension allocation for T, H, W.

    Example:
        ```python
        mrope_feature = MultiDimensionalRoPEFeature(
            spatial_merge_size=2,
            mrope_section=(24, 20, 20),  # 64 total dims split across T, H, W
        )

        # Compute positions for a 4x8x8 vision grid
        t_pos, h_pos, w_pos = mrope_feature.compute_vision_positions(
            grid_thw=(4, 8, 8),
            start_position=0,
        )
        # Each has shape (256,) for 4*8*8 tokens
        ```

    Note:
        Works with existing MultiModalRotaryEmbedding in
        easydel/layers/rotary_embedding.py which handles the THW interleaving
        and actual embedding computation.

    See Also:
        - easydel.layers.rotary_embedding.MultiModalRotaryEmbedding
    """

    def __init__(
        self,
        spatial_merge_size: int = 2,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
    ):
        """Initialize multi-dimensional RoPE feature.

        Args:
            spatial_merge_size: Spatial merge factor from vision config. Controls
                how neighboring patches are merged, affecting the spatial grid
                dimensions. Defaults to 2.
            mrope_section: RoPE dimension allocation for (temporal, height, width).
                The sum determines how many RoPE dimensions are used for position
                encoding (remaining dimensions can be used for other purposes).
                Defaults to (24, 20, 20) which allocates 64 dimensions total:
                24 for temporal, 20 for height, 20 for width.

        Example:
            ```python
            # Standard Qwen2-VL configuration
            feature = MultiDimensionalRoPEFeature(
                spatial_merge_size=2,
                mrope_section=(24, 20, 20),
            )

            # Custom configuration with more temporal dimensions
            feature = MultiDimensionalRoPEFeature(
                spatial_merge_size=1,
                mrope_section=(32, 16, 16),
            )
            ```
        """
        self.spatial_merge_size = spatial_merge_size
        self.mrope_section = mrope_section

    @property
    def is_enabled(self) -> bool:
        """Check if mRoPE is enabled.

        Returns:
            Always True for this feature. Used for feature presence checks.
        """
        return True

    def compute_vision_positions(
        self,
        grid_thw: tuple[int, int, int],
        start_position: int = 0,
    ) -> tuple[Array, Array, Array]:
        """Compute 3D position indices for a single vision input.

        Generates separate position indices for temporal, height, and width
        dimensions. Each position array has the same length (t * h * w tokens)
        but contains different values reflecting the 3D structure.

        Args:
            grid_thw: Grid shape as (temporal, height, width). For images,
                temporal is typically 1.
            start_position: Starting position offset. Added to all positions
                to handle multiple images/videos in sequence. Defaults to 0.

        Returns:
            Tuple of (temporal_positions, height_positions, width_positions),
            each with shape (t * h * w,). The positions are integers suitable
            for use with RoPE embeddings.

        Example:
            ```python
            feature = MultiDimensionalRoPEFeature()

            # For a 2x4x4 grid (2 temporal, 4 height, 4 width)
            t_pos, h_pos, w_pos = feature.compute_vision_positions(
                grid_thw=(2, 4, 4),
                start_position=0,
            )
            # t_pos: [0,0,0,...,0, 1,1,1,...,1] (16 zeros, 16 ones)
            # h_pos: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 0,0,0,0, ...] repeated for each t
            # w_pos: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, ...] repeated
            ```

        Note:
            The positions are structured so that:
            - Temporal position is constant within each frame
            - Height position varies along rows within each frame
            - Width position varies along columns within each frame
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

        For text tokens (not vision), all three dimensions use the same
        position index since text has only 1D sequential structure.

        Args:
            sequence_length: Length of the text sequence.
            batch_size: Batch size. Defaults to 1.

        Returns:
            Position IDs of shape (3, batch_size, sequence_length) where
            all three dimensions contain identical sequential positions
            [0, 1, 2, ..., sequence_length-1].

        Example:
            ```python
            feature = MultiDimensionalRoPEFeature()

            pos_ids = feature.get_default_position_ids(
                sequence_length=10,
                batch_size=2,
            )
            # pos_ids.shape = (3, 2, 10)
            # pos_ids[0] == pos_ids[1] == pos_ids[2]
            # Each contains [[0,1,2,...,9], [0,1,2,...,9]]
            ```

        Note:
            For VLM forward passes, these default positions are typically
            modified to incorporate vision token positions at the appropriate
            placeholder locations.
        """
        text_positions = jnp.arange(sequence_length)
        text_positions = jnp.broadcast_to(
            text_positions[jnp.newaxis, :],
            (batch_size, sequence_length),
        )
        # Stack same positions for all 3 dimensions (T, H, W all same for text)
        return jnp.stack([text_positions, text_positions, text_positions], axis=0)

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing feature class name and configuration parameters.
        """
        return (
            f"MultiDimensionalRoPEFeature(spatial_merge={self.spatial_merge_size}, mrope_section={self.mrope_section})"
        )
