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
    """

    scheduled: jax.Array
    query_start_loc: jax.Array
    seq_lens: jax.Array
    pages_tables: jax.Array
    padded_num_reqs: jax.Array
    request_distribution: jax.Array
    logits_indices: jax.Array
    input_ids_buf: jax.Array
    position_ids_buf: jax.Array
    num_requests: jax.Array
    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    min_p: jax.Array
    positions: jax.Array

    # v2-specific fields (optional, only populated when version="v2")
    slot_mapping: jax.Array | None = None
    num_kv_update_slices: jax.Array | None = None


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
        device_state: Minimal device state containing only token_ids and num_tokens
            for sampler updates. All other metadata stays on CPU.
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
        ...     device_state=MinimalDeviceState(...),
        ...     kv_pages=cache,
        ...     scheduled_full=jnp.array([4, 8, 2]),
        ...     req_num_tokens_full=jnp.array([512, 256, 128]),
        ...     active_mask_full=jnp.array([True, True, False]),
        ...     rng_key=jax.random.PRNGKey(0),
        ...     batch_metadata=metadata,
        ... )
    """

    device_state: MinimalDeviceState
    kv_pages: RaggedPagesCache
    scheduled_full: jax.Array  # [max_num_reqs] int32
    req_num_tokens_full: jax.Array  # [max_num_reqs] int32
    active_mask_full: jax.Array  # [max_num_reqs] bool
    rng_key: jax.Array
    batch_metadata: BatchMetadata


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
