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

"""Request state management for eSurge inference.

This module provides the CachedRequestState class, which tracks the state of
individual requests throughout the inference process. It stores prompt tokens,
generated tokens, sampling parameters, page allocations, and optional
vision-language model data.

Classes:
    CachedRequestState: Represents the complete state of a single inference
        request, including tokens, sampling parameters, and VLM data.

Example:
    >>> state = CachedRequestState(
    ...     req_id="request_123",
    ...     prompt_token_ids=[1, 2, 3, 4],
    ...     sampling_params=SamplingParams(temperature=0.7),
    ...     generator=jax.random.PRNGKey(42),
    ...     page_ids=([0, 1, 2],),
    ...     num_computed_tokens=4,
    ...     output_token_ids=[],
    ... )
    >>> state.num_tokens
    4
    >>> state.get_token_id(2)
    3
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from eformer.pytree import auto_pytree, field

from ...sampling_params import SamplingParams


@auto_pytree
class CachedRequestState:
    """Represents the state of a single request, compatible with JAX PyTree.

    This class tracks all information needed to process a single inference
    request throughout its lifecycle, from initial prompt to completion.
    It is designed to work with JAX's PyTree transformations while keeping
    large data structures on the CPU.

    The class supports vision-language models by storing pixel values and
    grid shapes for images and videos. Vision data is cleared after prefill
    to free memory, but mm_features with cached embeddings are preserved
    for multi-turn conversation support.

    Attributes:
        req_id (str): Unique identifier for this request.
        prompt_token_ids (list[int]): Token IDs for the input prompt.
        sampling_params (SamplingParams): Sampling configuration including
            temperature, top_k, top_p, etc.
        generator (jax.random.PRNGKey): JAX random key for stochastic sampling.
            May be None if using global RNG.
        page_ids (tuple[list[int], ...]): KV cache page allocations. Each inner
            list contains page indices for one attention group.
        num_computed_tokens (int): Number of tokens already processed by the
            model (prompt prefill progress).
        output_token_ids (list[int]): Tokens generated so far by the model.
        num_prompt_tokens (int): Length of prompt_token_ids, computed in __post_init__.
        pixel_values (np.ndarray | None): Image pixel values for VLM requests.
            Shape depends on the model's vision encoder.
        image_grid_thw (np.ndarray | None): Grid shape (T, H, W) for each image.
        pixel_values_videos (np.ndarray | None): Video pixel values for VLM requests.
        video_grid_thw (np.ndarray | None): Grid shape (T, H, W) for each video.
        mm_features (list[Any]): List of MultiModalFeature objects for cached
            vision embeddings. Preserved across turns for multi-turn support.
        prefill_inputs_embeds (np.ndarray | None): Precomputed prompt embeddings
            for VLM requests. Stored on host to avoid JIT-incompatible codepaths.
        prefill_position_ids (np.ndarray | None): Precomputed mRoPE position IDs
            for VLM requests with mRoPE-style models.
        prefill_rope_deltas (np.ndarray | None): RoPE delta adjustments for
            position ID computation in decode phase.
        prefill_visual_pos_masks (np.ndarray | None): Boolean mask indicating
            visual token positions for DeepStack-style visual injection.
        prefill_deepstack_visual_embeds (list[np.ndarray] | None): Layer-wise
            visual embeddings for DeepStack models.

    Note:
        Fields marked with `pytree_node=False` are not traversed by JAX PyTree
        operations, keeping them as CPU-resident Python objects.

    Example:
        >>> state = CachedRequestState(
        ...     req_id="request_123",
        ...     prompt_token_ids=[1, 2, 3, 4],
        ...     sampling_params=SamplingParams(temperature=0.7),
        ...     generator=None,
        ...     page_ids=([0, 1, 2],),
        ...     num_computed_tokens=0,
        ...     output_token_ids=[],
        ... )
        >>> state.num_tokens  # Total tokens
        4
        >>> state.has_vision  # Check for VLM data
        False
    """

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    generator: jax.random.PRNGKey
    page_ids: tuple[list[int], ...]
    num_computed_tokens: int = field(pytree_node=False)
    output_token_ids: list[int]
    num_prompt_tokens: int = field(pytree_node=False, default=-1)
    # Vision-language model support
    pixel_values: np.ndarray | None = field(pytree_node=False, default=None)
    image_grid_thw: np.ndarray | None = field(pytree_node=False, default=None)
    pixel_values_videos: np.ndarray | None = field(pytree_node=False, default=None)
    video_grid_thw: np.ndarray | None = field(pytree_node=False, default=None)
    # List of MultiModalFeature objects (using Any to avoid circular import issues with auto_pytree)
    mm_features: list[Any] = field(pytree_node=False, default_factory=list)
    _vision_processed: bool = field(pytree_node=False, default=False)
    # Precomputed VLM helpers (stored on host to avoid JIT-incompatible codepaths)
    prefill_inputs_embeds: np.ndarray | None = field(pytree_node=False, default=None)
    prefill_position_ids: np.ndarray | None = field(pytree_node=False, default=None)
    prefill_rope_deltas: np.ndarray | None = field(pytree_node=False, default=None)
    prefill_visual_pos_masks: np.ndarray | None = field(pytree_node=False, default=None)
    prefill_deepstack_visual_embeds: list[np.ndarray] | None = field(pytree_node=False, default=None)

    def __post_init__(self):
        """Initialize computed fields after instance creation.

        Sets num_prompt_tokens based on the length of prompt_token_ids.
        """
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """Get total number of tokens (prompt + generated).

        Returns:
            int: Total token count including both prompt and generated tokens.
        """
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def has_vision(self) -> bool:
        """Check if request has vision data (images or videos).

        Returns:
            bool: True if the request contains pixel values, video data,
                or cached multimodal features.
        """
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    @property
    def vision_processed(self) -> bool:
        """Check if vision data has been processed (prefill complete).

        Returns:
            bool: True if vision embeddings have been computed and raw
                vision data has been cleared.
        """
        return self._vision_processed

    def clear_vision_data(self) -> None:
        """Clear raw vision data after prefill to free memory.

        Removes pixel values and grid shapes from the request state to
        reduce memory usage after vision embeddings have been computed.
        The mm_features list with cached embeddings is preserved to
        support multi-turn conversations.

        Side Effects:
            - Sets pixel_values, image_grid_thw, pixel_values_videos,
              and video_grid_thw to None.
            - Calls clear_pixel_values() on each feature in mm_features.
            - Sets _vision_processed to True.
        """
        self.pixel_values = None
        self.image_grid_thw = None
        self.pixel_values_videos = None
        self.video_grid_thw = None
        # Clear pixel values from features but preserve cached embeddings
        for feat in self.mm_features:
            if hasattr(feat, "clear_pixel_values"):
                feat.clear_pixel_values()
        self._vision_processed = True

    def get_token_id(self, idx: int) -> int:
        """Get token ID at a specific position.

        Provides unified access to both prompt and generated tokens using
        a single index that spans the entire sequence.

        Args:
            idx: Zero-based position in the full token sequence (prompt
                followed by generated tokens).

        Returns:
            int: The token ID at the specified position.

        Raises:
            IndexError: If idx is out of bounds for the combined sequence.

        Example:
            >>> state.prompt_token_ids = [1, 2, 3]
            >>> state.output_token_ids = [4, 5]
            >>> state.get_token_id(0)  # First prompt token
            1
            >>> state.get_token_id(3)  # First generated token
            4
        """
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]
