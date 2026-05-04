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

"""Protocol definitions for eSurge execution backends.

This module provides structural types (Protocols) for components involved in
executing a single eSurge inference step. The primary intent is to make the
runner independent of a concrete execution implementation while preserving
type hints and IDE support.

The protocol defines the interface that any execution backend must implement
to be compatible with the eSurge runner. This enables future implementations
of alternative execution strategies (e.g., specialized hardware backends)
without modifying the runner code.

Classes:
    ExecutionManagerProtocol: Protocol defining the interface for execution
        managers that handle model compilation and execution.

Example:
    >>> def run_step(executor: ExecutionManagerProtocol, inputs: StepFunctionInputs):
    ...     '''Works with any ExecutionManagerProtocol implementation.'''
    ...     outputs = executor.execute_model(
    ...         num_tokens=256,
    ...         padded_num_reqs=16,
    ...         inputs=inputs,
    ...     )
    ...     return outputs
"""

from __future__ import annotations

import typing as tp
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from easydel.caching import (
    HybridCache,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.infra.sharding import MeshLike

from .execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


@runtime_checkable
class ExecutionManagerProtocol(Protocol):
    """Structural protocol for execution manager implementations.

    This protocol defines the interface that execution managers must implement
    to be compatible with the eSurge runner. It enables type-safe duck typing,
    allowing different execution backends to be used interchangeably.

    Implementations are expected to:
        - Maintain KV-cache pages and RNG state across inference steps.
        - Provide optional ahead-of-time compilation for token/batch buckets.
        - Expose separated execution phases: model forward and token sampling.
        - Handle graph updates for weight changes without full recompilation.

    Attributes:
        model (EasyDeLBaseModule): The EasyDeL model instance being executed.
        mesh (Any): JAX sharding mesh for distributed execution across devices.
        kv_pages (HybridCache | RaggedPagesCache | UnifiedAttentionCache): Paged
            key-value cache storage for attention computation.
        rng_key (jax.Array): JAX random key for stochastic sampling operations.
        max_model_len (int): Maximum sequence length supported by the model.
        max_num_reqs (int): Maximum number of concurrent requests.
        max_num_tokens (int): Maximum tokens per batch.
        metadata (RaggedPagesCacheConfig | UnifiedAttentionCacheConfig): KV cache
            configuration containing page size and other parameters.

    Note:
        This protocol is runtime_checkable, enabling isinstance() checks.
        However, these checks only verify method signatures, not behavior.

    See Also:
        :class:`~easydel.inference.esurge.runners.ExecutionManager`: The default
        implementation of this protocol.
    """

    model: "EasyDeLBaseModule"
    mesh: MeshLike
    kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache
    rng_key: jax.Array
    max_model_len: int
    max_num_reqs: int
    max_num_tokens: int
    metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig

    def clear_cache(self) -> None:
        """Clear compiled function cache.

        Removes all cached compiled functions, forcing recompilation on
        subsequent calls. Useful when model weights change or when memory
        needs to be freed.
        """
        ...

    def update_graphs(
        self,
        model: "EasyDeLBaseModule | None" = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
    ) -> None:
        """Update the graph components (weights) used by the executor.

        This method allows updating model weights without full recompilation.
        The executor maintains separate graph definition, state, and auxiliary
        components that can be independently updated.

        Args:
            model: Optional EasyDeL module to source new graph parts from.
                When provided, graphdef/graphstate/graphother are pulled
                from this model unless explicitly overridden.
            graphdef: Optional graph definition replacement (static structure).
            graphstate: Optional graph state replacement (typically weights).
            graphother: Optional auxiliary graph data replacement.

        Raises:
            ValueError: If neither a model nor explicit graph components
                are provided.
        """
        ...

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        num_reqs_paddings: list[int] | None = None,
    ) -> None:
        """Pre-compile execution functions for various input configurations.

        Compiles functions for different combinations of token counts and
        request counts to avoid runtime compilation overhead during serving.

        Args:
            num_tokens_paddings: List of token count configurations to compile.
            num_reqs_max_model_len: Maximum number of requests at max model length.
            max_pages_per_req: Maximum number of KV cache pages per request.
            max_num_reqs: Maximum number of concurrent requests.
            metadata: KV cache configuration with page size and other parameters.
            num_reqs_paddings: Optional explicit list of request count paddings.
                If None, paddings are computed automatically using power-of-2
                bucketing.
        """
        ...

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
        frequency_penalties_cpu: np.ndarray,
        presence_penalties_cpu: np.ndarray,
        repetition_penalties_cpu: np.ndarray,
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
    ) -> tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Prepare batch metadata for model execution.

        Builds the BatchMetadata structure from CPU arrays, handling token
        packing, position computation, page table preparation, and optional
        VLM data. Transfers the prepared payload to device.

        Args:
            num_tokens_static: Static token count for compilation bucket selection.
            scheduled_full_cpu: Tokens scheduled per request (CPU array).
            active_mask_full_cpu: Boolean mask for active requests (CPU array).
            input_ids_buf: Device buffer for input token IDs.
            position_ids_buf: Device buffer for position IDs.
            token_ids_cpu: All token IDs for all requests (CPU array).
            num_computed_tokens_cpu: Tokens already computed per request (CPU array).
            temperature_cpu: Temperature sampling parameters (CPU array).
            top_p_cpu: Top-p sampling parameters (CPU array).
            top_k_cpu: Top-k sampling parameters (CPU array).
            min_p_cpu: Min-p sampling parameters (CPU array).
            frequency_penalties_cpu: Frequency penalties (CPU array).
            presence_penalties_cpu: Presence penalties (CPU array).
            repetition_penalties_cpu: Repetition penalties (CPU array).
            page_table_cpu: KV cache page table (CPU array).
            padded_num_reqs_in: Padded request count for bucketing.
            page_table_version: Optional version for cache invalidation.
            mrope_position_ids_cpu: Optional mRoPE position IDs for VLMs.
            prefill_embeds_cpu: Optional precomputed embeddings for VLM prefill.
            prefill_embeds_mask_cpu: Optional mask for prefill embeddings.
            visual_pos_masks_cpu: Optional visual position masks for DeepStack.
            deepstack_visual_embeds_cpu: Optional DeepStack visual embeddings.
            pixel_values: Optional image pixel values for VLMs.
            image_grid_thw: Optional image grid shape (T, H, W).
            pixel_values_videos: Optional video pixel values for VLMs.
            video_grid_thw: Optional video grid shape (T, H, W).

        Returns:
            Tuple of (BatchMetadata, input_ids_buf, position_ids_buf,
            scheduled_full, active_mask_full) with device-resident arrays.
        """
        ...

    def execute_model(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs:
        """Execute the model forward pass.

        Runs the compiled model step function, updating the KV cache and
        producing hidden states and logits.

        Args:
            num_tokens: Number of tokens in this batch (for bucket selection).
            padded_num_reqs: Padded request count (for bucket selection).
            inputs: Consolidated step function inputs.

        Returns:
            ModelStepOutputs containing updated KV cache, hidden states,
            and logits for sampling.
        """
        ...

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        *,
        sampler_padded_num_reqs: int,
        sampler_num_reqs: int,
        sampler_total_tokens: int,
        req_num_tokens_full_cpu: np.ndarray,
        logits: jax.Array,
        rng_key: jax.Array,
        gather_positions_cpu: np.ndarray,
        sampling_seeds_cpu: np.ndarray,
        scatter_positions_cpu: np.ndarray,
        compact_window_row_indices_cpu: np.ndarray,
        compact_scheduled_cpu: np.ndarray,
        compact_seq_lens_cpu: np.ndarray,
        compact_active_mask_cpu: np.ndarray,
        compact_temperature_cpu: np.ndarray,
        compact_top_p_cpu: np.ndarray,
        compact_top_k_cpu: np.ndarray,
        compact_min_p_cpu: np.ndarray,
        compact_frequency_penalties_cpu: np.ndarray,
        compact_presence_penalties_cpu: np.ndarray,
        compact_repetition_penalties_cpu: np.ndarray,
        need_penalties: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample tokens from logits.

        Runs the compiled sampler function to generate new tokens from a
        compacted sampler window derived from the model outputs.

        Args:
            num_tokens: Number of tokens (for bucket selection).
            padded_num_reqs: Model-side padded request count for the current window.
            sampler_padded_num_reqs: Sampler-side padded request count after compaction.
            sampler_num_reqs: Actual compacted sampler rows.
            sampler_total_tokens: Total scheduled tokens in the compacted sampler batch.
            req_num_tokens_full_cpu: Target token count per request. The
                sampler packs the compact slice into its existing metadata
                transfer, avoiding a separate device vector each token.
            logits: Model output logits [padded_num_reqs, vocab_size].
            rng_key: JAX random key for stochastic sampling.
            gather_positions_cpu: Model-row indices used to gather compact logits.
            sampling_seeds_cpu: Per-row RNG fold-in seeds preserving row identity.
            scatter_positions_cpu: Output positions for restoring compact samples.
            compact_window_row_indices_cpu: Global request-row indices for compact rows.
            compact_scheduled_cpu: Scheduled tokens for compact rows.
            compact_seq_lens_cpu: Sequence lengths after the forward pass.
            compact_active_mask_cpu: Active compact rows.
            compact_temperature_cpu: Temperature per compact row.
            compact_top_p_cpu: Top-p per compact row.
            compact_top_k_cpu: Top-k per compact row.
            compact_min_p_cpu: Min-p per compact row.
            compact_frequency_penalties_cpu: Frequency penalty per compact row.
            compact_presence_penalties_cpu: Presence penalty per compact row.
            compact_repetition_penalties_cpu: Repetition penalty per compact row.
            need_penalties: Whether any request in the batch needs penalties.

        Returns:
            Tuple of (updated_rng_key, sampled_tokens, valid_mask, updated_token_counts) where:
            - updated_rng_key: New RNG key for next step.
            - sampled_tokens: Generated token IDs [padded_num_reqs], -1 for invalid.
            - valid_mask: Boolean mask indicating valid samples [padded_num_reqs].
            - updated_token_counts: Exact device-side token counts [max_num_reqs, vocab_size].
        """
        ...


ExecutionProtocol = ExecutionManagerProtocol
"""Alias for ExecutionManagerProtocol for backward compatibility."""
