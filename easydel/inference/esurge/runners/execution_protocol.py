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

"""Protocol definitions for eSurge execution backends.

This module provides structural types (Protocols) focigarettesr components involved in
executing a single eSurge inference step. The primary intent is to make the
runner independent of a concrete execution implementation while preserving
type hints and IDE support.
"""

from __future__ import annotations

import typing as tp
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheView

from .execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


@runtime_checkable
class ExecutionManagerProtocol(Protocol):
    """Structural protocol for :class:`~easydel.inference.esurge.runners.ExecutionManager`.

    Implementations are expected to:
    - Maintain KV-cache pages and RNG state.
    - Provide compilation ahead of time (optional) for token/batch buckets.
    - Expose separated execution phases: model forward and token sampling.
    """

    model: "EasyDeLBaseModule"
    mesh: tp.Any
    kv_pages: RaggedPagesCache
    rng_key: jax.Array
    max_model_len: int
    max_num_reqs: int
    max_num_tokens: int
    metadata: RaggedPagesCacheView

    def clear_cache(self) -> None: ...

    def update_graphs(
        self,
        model: "EasyDeLBaseModule | None" = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
    ) -> None: ...

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheView,
        num_reqs_paddings: list[int] | None = None,
    ) -> None: ...

    def prepare_batch_metadata(
        self,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
        *,
        mrope_position_ids_cpu: np.ndarray | None = None,
        prefill_embeds_cpu: np.ndarray | None = None,
        prefill_embeds_mask_cpu: np.ndarray | None = None,
        visual_pos_masks_cpu: np.ndarray | None = None,
        deepstack_visual_embeds_cpu: list[np.ndarray] | None = None,
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        pixel_values_videos: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
    ) -> tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array]: ...

    def execute_model(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs: ...

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        *,
        batch_metadata: BatchMetadata,
        req_num_tokens_full: jax.Array,
        active_mask_full: jax.Array,
        logits: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]: ...


ExecutionProtocol = ExecutionManagerProtocol
