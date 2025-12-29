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

"""Typed PyTree structures for execution manager step functions.

This module defines strongly-typed input and output structures for the
execution manager's step function, consolidating long parameter lists into
clean PyTree-compatible structures that support automatic sharding and
enable better type safety.

Key structures:
    - StepFunctionInputs: Consolidated inputs for step execution
    - StepFunctionOutputs: Consolidated outputs from step execution
"""

from __future__ import annotations

import typing

import jax
from eformer.pytree import auto_pytree

from easydel.layers.caching import RaggedPagesCache

if typing.TYPE_CHECKING:
    pass


@auto_pytree(frozen=True)
class MinimalDeviceState:
    """Minimal device state for sampler updates only.

    Contains only the fields that must be updated on device during sampling.
    All other metadata stays on CPU in SequenceBuffer as NumPy arrays.

    Attributes:
        token_ids: Token IDs buffer [max_num_reqs, max_model_len] for sampler updates.
        num_tokens: Total token count per request [max_num_reqs] for sampler increments.
    """

    token_ids: jax.Array
    num_tokens: jax.Array


@auto_pytree(frozen=True)
class BatchMetadata:
    """Precomputed tensors describing the current batch layout.

    This metadata supports both v2 and v3 attention kernels:
    - v2: Uses slot_mapping and num_kv_update_slices for KV cache updates
    - v3: Uses request_distribution for optimized ragged attention

    Performance note:
        This structure is part of the step-function input PyTree. To keep host→device
        transfer overhead low, several small vectors/scalars are packed into a few
        dense arrays (rather than many independent leaves). Public attributes like
        `scheduled`, `seq_lens`, etc. are exposed as properties backed by those
        packed arrays.

    Vision-language model support:
    - pixel_values: Image pixel values for VLMs [batch, seq, channels]
    - image_grid_thw: Grid shape (T, H, W) for each image
    - pixel_values_videos: Video pixel values for VLMs
    - video_grid_thw: Grid shape (T, H, W) for each video
    """

    # Packed int32: [2, max_num_reqs+1] => (query_start_loc, seq_lens_padded)
    packed_qsl_seqlens: jax.Array

    # Packed per-request parameters (padded to batch bucket):
    # - int32: [3, padded_num_reqs] => (scheduled, logits_indices, top_k)
    # - float32: [3, padded_num_reqs] => (temperature, top_p, min_p)
    packed_i32_padded: jax.Array
    packed_f32_padded: jax.Array

    # Packed int32: [5] => (num_requests, padded_num_reqs, request_distribution[0:3])
    packed_misc_i32: jax.Array

    pages_tables: jax.Array
    input_ids_buf: jax.Array
    position_ids_buf: jax.Array

    # Total number of tokens in the batch (used by hybrid mode)
    num_tokens: jax.Array | None = None

    # v2-specific fields (optional, only populated when version="v2")
    slot_mapping: jax.Array | None = None
    num_kv_update_slices: jax.Array | None = None

    # Vision-language model data (None for text-only requests)
    pixel_values: jax.Array | None = None
    image_grid_thw: jax.Array | None = None
    pixel_values_videos: jax.Array | None = None
    video_grid_thw: jax.Array | None = None

    # VLM prefill helpers (kept explicit to avoid JIT-incompatible model codepaths)
    # - mrope_position_ids: 3D RoPE indices for mRoPE-style models [3, num_tokens]
    # - prefill_embeds: Optional per-token embedding overrides [num_tokens, hidden]
    # - prefill_embeds_mask: Mask selecting which rows in `prefill_embeds` to use [num_tokens]
    mrope_position_ids: jax.Array | None = None
    prefill_embeds: jax.Array | None = None
    prefill_embeds_mask: jax.Array | None = None

    # DeepStack-style visual injection (optional, model-specific)
    visual_pos_masks: jax.Array | None = None
    deepstack_visual_embeds: tuple[jax.Array, ...] | None = None

    @property
    def query_start_loc(self) -> jax.Array:
        return self.packed_qsl_seqlens[0]

    @property
    def seq_lens(self) -> jax.Array:
        # Packed row is padded to `max_num_reqs+1` to share a single buffer with query_start_loc.
        return self.packed_qsl_seqlens[1, :-1]

    @property
    def scheduled(self) -> jax.Array:
        return self.packed_i32_padded[0]

    @property
    def logits_indices(self) -> jax.Array:
        return self.packed_i32_padded[1]

    @property
    def top_k(self) -> jax.Array:
        return self.packed_i32_padded[2]

    @property
    def temperature(self) -> jax.Array:
        return self.packed_f32_padded[0]

    @property
    def top_p(self) -> jax.Array:
        return self.packed_f32_padded[1]

    @property
    def min_p(self) -> jax.Array:
        return self.packed_f32_padded[2]

    @property
    def num_requests(self) -> jax.Array:
        return self.packed_misc_i32[0]

    @property
    def padded_num_reqs(self) -> jax.Array:
        return self.packed_misc_i32[1]

    @property
    def request_distribution(self) -> jax.Array:
        return self.packed_misc_i32[2:5]


@auto_pytree(frozen=True)
class ModelStepOutputs:
    """Outputs returned from the pure model forward pass."""

    kv_pages: RaggedPagesCache
    hidden_states: jax.Array
    logits: jax.Array


@auto_pytree(frozen=True)
class StepFunctionInputs:
    """Consolidated inputs for fused step execution.

    This frozen PyTree replaces a 12-parameter function signature with a single
    typed structure, improving code maintainability and enabling automatic sharding
    via JAX pytree transformations.

    Attributes:
        kv_pages: Paged key-value cache storage for attention computation.
            Uses ragged paging for memory-efficient caching across variable-length
            sequences.
        scheduled_full: Number of tokens scheduled for processing per request
            [max_num_reqs]. Determines how many tokens from each request will be
            processed in this step.
        req_num_tokens_full: Total number of tokens required per request
            [max_num_reqs]. Used for tracking completion and memory allocation.
        active_mask_full: Boolean mask indicating which requests are active
            [max_num_reqs]. Inactive requests are skipped during processing.
        rng_key: JAX random key for stochastic sampling operations. Split and
            updated at each step for proper random state management.
        batch_metadata: Pre-computed batch metadata prepared from CPU arrays.

    Note:
        All arrays are device-resident. The structure is frozen for immutability
        and JIT safety. Shardings are applied at the PyTree level.

    Example:
        >>> inputs = StepFunctionInputs(
        ...     kv_pages=cache,
        ...     scheduled_full=jnp.array([4, 8, 2]),
        ...     req_num_tokens_full=jnp.array([512, 256, 128]),
        ...     active_mask_full=jnp.array([True, True, False]),
        ...     rng_key=jax.random.PRNGKey(0),
        ...     batch_metadata=metadata,
        ... )
    """

    kv_pages: RaggedPagesCache
    scheduled_full: jax.Array  # [max_num_reqs] int32
    req_num_tokens_full: jax.Array  # [max_num_reqs] int32
    active_mask_full: jax.Array  # [max_num_reqs] bool
    rng_key: jax.Array
    batch_metadata: BatchMetadata

    def print_status(self) -> None:
        """Print the shapes of all fields in this StepFunctionInputs structure."""
        lines = []

        lines.append("StepFunctionInputs Status")
        lines.append("\nMinimalDeviceState:")
        lines.append(f"  token_ids:       {self.device_state.token_ids.shape}")
        lines.append(f"  num_tokens:      {self.device_state.num_tokens.shape}")
        lines.append("\nRaggedPagesCache:")
        lines.append(f"  kv_pages:        {len(self.kv_pages.views)}x{self.kv_pages.views[-1].kv_pages.shape}")
        lines.append("\nRequest Arrays:")
        lines.append(f"  scheduled_full:      {self.scheduled_full.shape}")
        lines.append(f"  req_num_tokens_full: {self.req_num_tokens_full.shape}")
        lines.append(f"  active_mask_full:    {self.active_mask_full.shape}")
        lines.append(f"  rng_key:             {self.rng_key.shape}")
        lines.append("\nBatchMetadata:")
        lines.append(f"  scheduled:            {self.batch_metadata.scheduled.shape}")
        lines.append(f"  query_start_loc:      {self.batch_metadata.query_start_loc.shape}")
        lines.append(f"  seq_lens:             {self.batch_metadata.seq_lens.shape}")
        lines.append(f"  pages_tables:         {self.batch_metadata.pages_tables.shape}")
        lines.append(f"  padded_num_reqs:      {self.batch_metadata.padded_num_reqs.shape}")
        lines.append(f"  request_distribution: {self.batch_metadata.request_distribution.shape}")
        lines.append(f"  logits_indices:       {self.batch_metadata.logits_indices.shape}")
        lines.append(f"  input_ids_buf:        {self.batch_metadata.input_ids_buf.shape}")
        lines.append(f"  position_ids_buf:     {self.batch_metadata.position_ids_buf.shape}")
        lines.append(f"  num_requests:         {self.batch_metadata.num_requests.shape}")
        lines.append(f"  temperature:          {self.batch_metadata.temperature.shape}")
        lines.append(f"  top_p:                {self.batch_metadata.top_p.shape}")
        lines.append(f"  top_k:                {self.batch_metadata.top_k.shape}")
        lines.append(f"  min_p:                {self.batch_metadata.min_p.shape}")
        if self.batch_metadata.slot_mapping is not None:
            lines.append(f"  slot_mapping:         {self.batch_metadata.slot_mapping.shape}")
        if self.batch_metadata.num_kv_update_slices is not None:
            lines.append(f"  num_kv_update_slices: {self.batch_metadata.num_kv_update_slices.shape}")
        if self.batch_metadata.pixel_values is not None:
            lines.append("\nVision-Language Model Data:")
            lines.append(f"  pixel_values:         {self.batch_metadata.pixel_values.shape}")
        if self.batch_metadata.image_grid_thw is not None:
            lines.append(f"  image_grid_thw:       {self.batch_metadata.image_grid_thw.shape}")
        if self.batch_metadata.pixel_values_videos is not None:
            lines.append(f"  pixel_values_videos:  {self.batch_metadata.pixel_values_videos.shape}")
        if self.batch_metadata.video_grid_thw is not None:
            lines.append(f"  video_grid_thw:       {self.batch_metadata.video_grid_thw.shape}")
        print("\n".join(lines))


@auto_pytree(frozen=True)
class StepFunctionOutputs:
    """Consolidated outputs from fused step execution.

    This frozen PyTree packages all step execution results into a single typed
    structure, enabling clean return semantics and automatic sharding propagation.

    Attributes:
        device_state: Updated minimal device state (token_ids, num_tokens) reflecting
            new token generation from sampler.
        kv_pages: Updated key-value cache pages with newly computed attention states.
        input_ids_buf: Updated input token buffer [max_num_tokens]. May contain
            newly sampled tokens appended for next iteration.
        position_ids_buf: Updated position buffer [max_num_tokens]. Position indices
            incremented for decode steps.
        query_start_loc: Cumulative query start indices [max_num_reqs+1]. Used to
            slice contiguous token buffers into per-request segments. Last element
            contains total token count.
        seq_lens: Current sequence length for each request [max_num_reqs]. Updated
            after token generation to track progress toward max_tokens.
        pages_tables: Page table mapping for attention [num_reqs, max_pages].
            Maps logical KV cache positions to physical page indices.
        rng_key: Updated JAX random key for next sampling step. Must be threaded
            through for proper random state management.
        out_tokens: Newly sampled tokens [max_num_reqs]. Invalid positions masked
            with -1 (for finished or inactive requests).
        valid_mask: Boolean mask for valid outputs [max_num_reqs]. True for requests
            that generated valid tokens this step.
        hidden_states: Final layer hidden states [num_tokens, hidden_dim]. Raw
            transformer outputs before LM head projection.
        logits: Vocabulary logits from LM head [padded_num_reqs, vocab_size].
            Unfiltered probability distributions before sampling.

    Note:
        Output tensors maintain device placement from inputs. The structure is
        frozen to prevent accidental mutation. Use PyTree unpacking to extract
        individual fields.

    Example:
        >>> outputs = step_fn(inputs)
        >>> new_tokens = outputs.out_tokens
        >>> is_valid = outputs.valid_mask
        >>> updated_cache = outputs.kv_pages
    """

    device_state: MinimalDeviceState
    kv_pages: RaggedPagesCache
    input_ids_buf: jax.Array
    position_ids_buf: jax.Array
    query_start_loc: jax.Array
    seq_lens: jax.Array
    pages_tables: jax.Array
    rng_key: jax.Array
    out_tokens: jax.Array
    valid_mask: jax.Array
    hidden_states: jax.Array
    logits: jax.Array
