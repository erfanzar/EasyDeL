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

The structures in this module are designed to be:
    - Immutable (frozen) for JIT compatibility and safety
    - PyTree-compatible for automatic JAX transformations
    - Memory-efficient through packed arrays to reduce transfer overhead
    - Self-documenting with comprehensive attribute descriptions

Key structures:
    MinimalDeviceState: Minimal state for sampler updates (token_ids, num_tokens).
    BatchMetadata: Precomputed batch layout tensors with packed storage.
    ModelStepOutputs: Pure model forward pass outputs (cache, hidden, logits).
    StepFunctionInputs: Consolidated inputs for fused step execution.
    StepFunctionOutputs: Consolidated outputs from fused step execution.

Architecture Notes:
    The module uses array packing to minimize host-device transfer overhead.
    Instead of many small arrays (high latency), related values are packed
    into dense 2D arrays:

    - packed_qsl_seqlens: [2, max_num_reqs+1] for query_start_loc and seq_lens
    - packed_i32_padded: [3, padded_num_reqs] for scheduled, logits_indices, top_k
    - packed_f32_padded: [3, padded_num_reqs] for temperature, top_p, min_p
    - packed_misc_i32: [5] for num_requests, padded_num_reqs, request_distribution

    Properties provide unpacking for convenient access while maintaining
    transfer efficiency.

Example:
    >>> # Create step inputs
    >>> inputs = StepFunctionInputs(
    ...     kv_pages=cache,
    ...     scheduled_full=scheduled,
    ...     req_num_tokens_full=req_tokens,
    ...     active_mask_full=active_mask,
    ...     rng_key=rng_key,
    ...     batch_metadata=batch_metadata,
    ... )
    >>>
    >>> # Execute step and unpack outputs
    >>> outputs = step_fn(inputs)
    >>> new_tokens = outputs.out_tokens
    >>> updated_cache = outputs.kv_pages
"""

from __future__ import annotations

import typing

import jax
from eformer.pytree import auto_pytree

from easydel.layers.caching import HybridCache, RaggedPagesCache, UnifiedAttentionCache

if typing.TYPE_CHECKING:
    pass


@auto_pytree(frozen=True)
class MinimalDeviceState:
    """Minimal device state for sampler updates only.

    This structure contains only the fields that must be updated on device
    during the sampling phase of token generation. By keeping this minimal,
    we reduce host-device transfer overhead and maintain most metadata as
    CPU-resident NumPy arrays in the SequenceBuffer.

    The MinimalDeviceState is designed to be part of the step function's
    input/output PyTree, enabling efficient JAX transformations while
    minimizing the data that needs to cross the host-device boundary.

    Attributes:
        token_ids (jax.Array): Token IDs buffer with shape [max_num_reqs, max_model_len].
            Contains all tokens (prompt + generated) for each request. Updated
            by the sampler to append newly generated tokens.
        num_tokens (jax.Array): Total token count per request with shape [max_num_reqs].
            Tracks the current sequence length for each request, incremented
            after successful token generation.

    Note:
        This class is frozen (immutable) for JIT safety. Updates create new
        instances rather than modifying in place.

    Example:
        >>> state = MinimalDeviceState(
        ...     token_ids=jnp.zeros((32, 2048), dtype=jnp.int32),
        ...     num_tokens=jnp.zeros((32,), dtype=jnp.int32),
        ... )
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
        """Get cumulative query start locations for each request.

        Returns:
            jax.Array: Cumulative start indices with shape [max_num_reqs+1].
                Element i contains the starting index in the packed token buffer
                where request i's tokens begin. The last element contains the
                total number of tokens across all requests.
        """
        return self.packed_qsl_seqlens[0]

    @property
    def seq_lens(self) -> jax.Array:
        """Get current sequence lengths for each request.

        Returns:
            jax.Array: Sequence lengths with shape [max_num_reqs]. Each element
                contains the total number of tokens (prompt + generated) for
                the corresponding request after this step completes.

        Note:
            The packed row is padded to `max_num_reqs+1` to share a single
            buffer with query_start_loc, so we slice off the last element.
        """
        return self.packed_qsl_seqlens[1, :-1]

    @property
    def scheduled(self) -> jax.Array:
        """Get number of tokens scheduled for each request in this step.

        Returns:
            jax.Array: Scheduled token counts with shape [padded_num_reqs].
                Indicates how many tokens from each request are being processed
                in this step. Zero means the request is not participating.
        """
        return self.packed_i32_padded[0]

    @property
    def logits_indices(self) -> jax.Array:
        """Get indices for extracting logits from hidden states.

        Returns:
            jax.Array: Logits indices with shape [padded_num_reqs]. For each
                request, this is the index of the last token in the packed
                batch, which is used to extract logits for next-token prediction.
        """
        return self.packed_i32_padded[1]

    @property
    def top_k(self) -> jax.Array:
        """Get top-k sampling parameters for each request.

        Returns:
            jax.Array: Top-k values with shape [padded_num_reqs]. Zero means
                no top-k filtering is applied for that request.
        """
        return self.packed_i32_padded[2]

    @property
    def temperature(self) -> jax.Array:
        """Get temperature sampling parameters for each request.

        Returns:
            jax.Array: Temperature values with shape [padded_num_reqs]. Values
                <= 0 indicate greedy sampling. Higher values increase randomness.
        """
        return self.packed_f32_padded[0]

    @property
    def top_p(self) -> jax.Array:
        """Get top-p (nucleus) sampling parameters for each request.

        Returns:
            jax.Array: Top-p values with shape [padded_num_reqs]. Values of 1.0
                indicate no nucleus filtering. Lower values restrict sampling
                to tokens with highest cumulative probability.
        """
        return self.packed_f32_padded[1]

    @property
    def min_p(self) -> jax.Array:
        """Get min-p sampling parameters for each request.

        Returns:
            jax.Array: Min-p threshold values with shape [padded_num_reqs].
                Filters out tokens with probability below this fraction of
                the maximum token probability.
        """
        return self.packed_f32_padded[2]

    @property
    def num_requests(self) -> jax.Array:
        """Get the actual number of active requests (unpadded).

        Returns:
            jax.Array: Scalar containing the number of active requests in
                this batch, excluding padding.
        """
        return self.packed_misc_i32[0]

    @property
    def padded_num_reqs(self) -> jax.Array:
        """Get the padded number of requests for compilation efficiency.

        Returns:
            jax.Array: Scalar containing the padded request count. This is
                typically a power of 2 to reduce the number of unique
                compiled function variants.
        """
        return self.packed_misc_i32[1]

    @property
    def request_distribution(self) -> jax.Array:
        """Get request distribution for v3 attention kernel optimization.

        Returns:
            jax.Array: Distribution triple [decode_count, chunked_prefill_end, total].
                Used by ragged page attention v3 to optimize memory access patterns
                based on whether requests are in decode or prefill phase.
        """
        return self.packed_misc_i32[2:5]


@auto_pytree(frozen=True)
class ModelStepOutputs:
    """Outputs returned from the pure model forward pass.

    This frozen PyTree structure encapsulates all outputs from a single
    model forward step. It separates the model computation from sampling,
    allowing the caller to perform additional processing (e.g., logprob
    computation) before sampling.

    Attributes:
        kv_pages (HybridCache | RaggedPagesCache | UnifiedAttentionCache):
            Updated key-value cache pages containing newly computed attention
            states. The cache type depends on the configured attention mechanism
            (v2 uses RaggedPagesCache, v3 uses RaggedPagesCache with request
            distribution, unified uses UnifiedAttentionCache).
        hidden_states (jax.Array): Final layer hidden states with shape
            [num_tokens, hidden_dim]. These are the raw transformer outputs
            before the language model head projection, useful for tasks
            requiring access to token representations.
        logits (jax.Array): Vocabulary logits with shape [padded_num_reqs, vocab_size].
            Output from the language model head applied to the last token
            position of each request. These are unfiltered probability
            distributions ready for sampling.

    Note:
        The structure is frozen for immutability and JIT safety. Output tensors
        maintain device placement from inputs. The logits are extracted only
        for the last token position of each request (using logits_indices from
        BatchMetadata) to minimize computation.

    Example:
        >>> outputs = model_step_fn(graphstate, graphother, kv_pages, metadata)
        >>> new_cache = outputs.kv_pages
        >>> logits_for_sampling = outputs.logits
        >>> hidden_for_analysis = outputs.hidden_states
    """

    kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache
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

    kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache
    scheduled_full: jax.Array  # [max_num_reqs] int32
    req_num_tokens_full: jax.Array  # [max_num_reqs] int32
    active_mask_full: jax.Array  # [max_num_reqs] bool
    rng_key: jax.Array
    batch_metadata: BatchMetadata

    def print_status(self) -> None:
        """Print the shapes of all fields in this StepFunctionInputs structure.

        Outputs a formatted summary of all array shapes in this input structure,
        useful for debugging compilation issues or verifying input dimensions.
        The output includes sections for MinimalDeviceState, Paged KV Cache,
        Request Arrays, BatchMetadata, and optional VLM data.

        Note:
            This method accesses device arrays and should not be called in
            performance-critical code paths. It's intended for debugging only.
        """
        lines = []

        lines.append("StepFunctionInputs Status")
        lines.append("\nMinimalDeviceState:")
        lines.append(f"  token_ids:       {self.device_state.token_ids.shape}")
        lines.append(f"  num_tokens:      {self.device_state.num_tokens.shape}")
        lines.append("\nPaged KV Cache:")
        last_view = self.kv_pages.views[-1]
        if hasattr(last_view, "kv_pages"):
            lines.append(f"  kv_pages:        {len(self.kv_pages.views)}x{last_view.kv_pages.shape}")
        elif hasattr(last_view, "key_cache") and hasattr(last_view, "value_cache"):
            lines.append(f"  key_cache:       {len(self.kv_pages.views)}x{last_view.key_cache.shape}")
            lines.append(f"  value_cache:     {len(self.kv_pages.views)}x{last_view.value_cache.shape}")
        else:
            lines.append(f"  views:           {len(self.kv_pages.views)}x{type(last_view)}")
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
    kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache
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
