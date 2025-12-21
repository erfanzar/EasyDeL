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

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from eformer.pytree import auto_pytree, field

from ...sampling_params import SamplingParams


@auto_pytree
class CachedRequestState:
    """Represents the state of a single request, compatible with JAX.

    Supports vision-language models by storing pixel values and grid shapes
    for images and videos. Vision data is cleared after prefill to free memory,
    but mm_features with cached embeddings are preserved for multi-turn support.
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
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def has_vision(self) -> bool:
        """Check if request has vision data (images or videos)."""
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    @property
    def vision_processed(self) -> bool:
        """Check if vision data has been processed (prefill complete)."""
        return self._vision_processed

    def clear_vision_data(self) -> None:
        """Clear raw vision data after prefill to free memory.

        Note: mm_features with cached embeddings are preserved for multi-turn.
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
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]
