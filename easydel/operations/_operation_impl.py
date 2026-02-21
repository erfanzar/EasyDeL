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

"""Core attention operator implementation framework for EasyDeL.

This module provides the foundational classes and abstractions for implementing
various attention mechanisms in JAX. It includes:

- OperationOutput: Container for attention computation results
- OperationMetadata: Configuration and runtime metadata for attention operations
- OperationImpl: Abstract base class for specific attention implementations
- OperationRegistry: Plugin system for discovering and managing attention implementations

The module supports multiple attention backends (TPU, GPU, CPU) and provides
common utilities for mask handling, head repetition (for GQA/MQA), and sharding
specifications for distributed computation.

Key Design Principles:
1. Backend-agnostic interface with backend-specific optimizations
2. Support for various attention patterns (vanilla, flash, ring, etc.)
3. Efficient handling of different tensor layouts (BTHD vs BHTD)
4. Integration with JAX's sharding and parallelism features
5. Flexible metadata system for runtime configuration

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata, OperationRegistry
    >>>
    >>> # Create metadata for attention configuration
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.float16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim),
    ...     dropout_prob=0.1
    ... )
    >>>
    >>> # Get and instantiate a specific attention implementation
    >>> attn_impl = OperationRegistry.create("flash", metadata)
    >>>
    >>> # Use the attention implementation
    >>> output = attn_impl(query, key, value, mask=attention_mask)
"""

from __future__ import annotations

import einops
from eformer import common_types
from eformer import escale as es
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Bool, Float

from ._base_operation import BaseOperation, OperationRegistry
from ._operation_meta import OperationMetadata
from .requirements import ExecutionMode, OperationRequirements

OperationRegistry = OperationRegistry

logger = get_logger("EasyDeL-OperationOperator")


NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ


@auto_pytree
class OperationOutput:
    """
    This dataclass encapsulates the results computation
    """


class OperationImpl(BaseOperation):
    """
    Abstract Base Class for specific attention implementations.

    Inherits from `BaseOperation` to leverage backend-specific dispatching.
    Subclasses must implement the core attention logic (`forward_native`) and
    potentially provide optimized versions for TPU (`forward_tpu`), GPU (`forward_gpu`),
    etc. They also need to declare their name and associated metadata.

    Provides common helper methods for attention processing like mask manipulation,
    head repeating (for GQA/MQA), and determining runtime mode.
    """

    def __init__(self, metadata: OperationMetadata) -> None:
        """
        Initializes the attention implementation with its metadata.

        Args:
            metadata: An `OperationMetadata` instance containing configuration
                and context for this attention operation.
        """
        self.metadata = metadata

    def get_instance_requirements(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """
        Returns the operation requirements, applying instance-level overrides.

        This method wraps the class-level `get_requirements()` and applies
        any instance-level overrides from metadata (e.g., `requires_cache`).

        This is the preferred method to call when you need requirements that
        respect instance configuration, such as when determining cache needs
        for vision encoders that don't need KV cache.

        Args:
            mode: The execution mode (prefill, decode, or mixed).

        Returns:
            OperationRequirements with instance-level overrides applied.

        Example:
            >>> op = GatedDeltaRuleOp(metadata)
            >>> # Class default: requires_cache=True
            >>> class_reqs = op.get_requirements()
            >>> # Instance override: metadata.requires_cache=False
            >>> instance_reqs = op.get_instance_requirements()
        """
        # Get class-level requirements
        reqs = self.get_requirements(mode)

        # Apply instance-level requires_cache override from metadata
        if self.metadata is not None and self.metadata.requires_cache is not None:
            reqs = reqs.with_requires_cache(self.metadata.requires_cache)

        return reqs

    def get_mode(self, query: Float[Array, "batch ... num_heads head_dim"], BTHD: bool = True) -> RUNTIME_MODE_TYPES:  # type:ignore
        """
        Determines the runtime mode (normal or generation) based on query shape.

        Assumes generation mode if the query sequence length dimension is 1.

        Args:
            query: The query tensor.
            BTHD: Boolean indicating tensor layout (True for B, T, H, D; False for B, H, T, D).
        """
        ingeneration = query.shape[1] == 1 if BTHD else query.shape[2] == 1
        return common_types.MODE_DECODE if ingeneration else common_types.MODE_TRAIN

    @staticmethod
    def _split_attention_mask(
        attn_mask: Bool[Array, "... seq_len seq_len"],
    ) -> tuple[Bool[Array, "... seq_len"], Bool[Array, "... seq_len"]]:
        """
        Splits a combined attention mask into separate query and key-value masks.

        Assumes the input mask `attn_mask` might be 4D (batch, head, q_seq, kv_seq)
        or 3D (batch, q_seq, kv_seq). It derives the query mask by checking which
        query positions can attend to *any* key position, and the key-value mask
        by checking which key positions *can be attended to* by any query position.

        Args:
            attn_mask: The combined attention mask (3D or 4D). If 4D, the last head dim
                is used. Shape (..., q_seq, kv_seq).

        Returns:
            A tuple `(q_mask, kv_mask)`:
                - `q_mask`: Boolean array of shape (..., q_seq). True for valid query tokens.
                - `kv_mask`: Boolean array of shape (..., kv_seq). True for valid key/value tokens.
        """
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, -1, :, :]
        return jnp.any(attn_mask, axis=-1), jnp.any(attn_mask, axis=-2)

    @staticmethod
    def _combine_query_kv_masks(
        q_mask: Bool[Array, "... q_seq"], kv_mask: Bool[Array, "... kv_seq"]
    ) -> Bool[Array, "... q_seq kv_seq"]:
        """
        Combines separate query and key-value masks into a standard attention mask.

        Creates a broadcastable mask where `mask[b, i, j]` is True if both
        `q_mask[b, i]` and `kv_mask[b, j]` are True.

        Args:
            q_mask: Boolean array of shape (..., q_seq). True for valid query tokens.
            kv_mask: Boolean array of shape (..., kv_seq). True for valid key/value tokens.

        Returns:
            A boolean attention mask of shape (..., q_seq, kv_seq).
        """
        if kv_mask.ndim == 2:
            kv_mask = kv_mask[:, None, :]
        if q_mask.ndim == 2:
            q_mask = q_mask[:, :, None]
        return q_mask * kv_mask

    @staticmethod
    def _create_causal_mask(qseq: int) -> Bool[Array, "seq_len seq_len"]:
        """
        Creates a causal attention mask (lower triangular).

        Args:
            qseq: The sequence length .

        Returns:
            A boolean array of shape (qseq, qseq) where `mask[i, j]` is
            True if `j <= i`, representing causal visibility.
        """
        return jnp.tril(jnp.ones((qseq, qseq), dtype="b1"))

    @staticmethod
    def repeat_kv_heads(
        k: Float[Array, "batch seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch seq_len num_kv_heads head_dim"],
        num_reps: int,
    ) -> tuple[Float[Array, "batch seq_len num_q_heads head_dim"], Float[Array, "batch seq_len num_q_heads head_dim"]]:
        """
        Repeats Key and Value heads for Grouped Query Operation (GQA) or Multi-Query Operation (MQA).

        Expands the head dimension of K and V tensors to match the number of query heads.

        Args:
            k: Key tensor, assumes shape (batch, seq_len, num_kv_heads, head_dim).
            v: Value tensor, assumes shape (batch, seq_len, num_kv_heads, head_dim).
            num_reps: The number of times to repeat each KV head (num_q_heads // num_kv_heads).

        Returns:
            A tuple `(k_repeated, v_repeated)` with shapes
            (batch, seq_len, num_q_heads, head_dim).
        """
        return (
            einops.repeat(k, "b s h d -> b s (h r) d", r=num_reps),
            einops.repeat(v, "b s h d -> b s (h r) d", r=num_reps),
        )

    def _handle_kvhead(
        self,
        array: Float[Array, "batch heads q_seq kv_seq"] | None,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> Float[Array, "batch num_q_heads q_seq kv_seq"] | None:
        """
        Processes an attention bias or similar array based on head configuration (GQA/MQA).

        If the array's head dimension matches `num_kv_heads`, it repeats the heads
        to match `num_q_heads`. If it matches `num_q_heads` or is 1 (broadcastable),
        it's returned as is.

        Args:
            array: The input array, typically an attention bias. Assumes head dimension
                is at index 1. Shape (batch, num_heads, q_seq, kv_seq) or similar.
                Can be None.
            num_q_heads: The number of query heads.
            num_kv_heads: The number of key/value heads.

        Returns:
            The processed array with head dimension matching `num_q_heads`, or None
            if the input was None.

        Raises:
            ValueError: If the array's head dimension is incompatible.
        """
        if array is None:
            return None

        current_num_heads: int = array.shape[1]
        matches_q_heads: bool = current_num_heads == num_q_heads
        is_broadcastable: bool = current_num_heads == 1

        if matches_q_heads or is_broadcastable:
            return array

        matches_kv_heads: bool = current_num_heads == num_kv_heads
        if matches_kv_heads:
            repetitions: int = num_q_heads // current_num_heads
            repeated: Float[Array, "batch num_q_heads q_seq kv_seq"] = einops.repeat(
                array,
                "b h q k -> b (h r) q k",
                r=repetitions,
            )
            return repeated
        else:
            raise ValueError(
                f"Incompatible array shape. Got {current_num_heads} heads, expected {num_q_heads}, {num_kv_heads}, or 1"
            )

    def create_stable_sharding(
        self,
        state_ps: Ps | None = None,
        preserved_indices: list[int] | None = None,
        clone_ps: Ps | None = None,
        dep: Ps | bool | None = True,
        tensor: Float[Array, "..."] | None = None,
    ) -> Ps | None:
        """
        Helper to create a PartitionSpec, potentially preserving only certain axes.

        This might be used for ensuring intermediate tensors or states have compatible
        sharding, possibly replicating across axes not specified in `preserved_indices`.

        Args:
            state_ps: The base PartitionSpec to modify.
            preserved_indices: A list of dimension indices whose partitioning should be
                kept from `state_ps` (or `clone_ps` if provided). Other dimensions
                will be set to None (replicated). If None, `state_ps` is returned.
            clone_ps: An optional PartitionSpec to copy axis names from for the
                preserved indices, instead of using `state_ps`.
            dep: A dependency flag or PartitionSpec. If None, returns None. Defaults to True.
                (The exact purpose might be context-specific, potentially for control flow).
            tensor: Optional tensor to get corrected sharding for.

        Returns:
            A new PartitionSpec with only specified axes partitioned, or None based on `dep`.
            Returns `state_ps` directly if `preserved_indices` is None.
        """
        with self.metadata.mesh:
            if dep is None:
                return None

            if state_ps is None:
                return None

            if preserved_indices is None:
                if tensor is None:
                    return state_ps
                corrected: Ps = es.get_corrected_named_sharding(tensor.shape, state_ps).spec
                return corrected

            num_dims: int = len(state_ps)
            new_spec: list[str | None] = [None] * num_dims
            idx: int
            for idx in preserved_indices:
                source_ps: Ps = state_ps if clone_ps is None else clone_ps
                new_spec[idx] = source_ps[idx]

            sharding: Ps = Ps(*new_spec)

            if tensor is None:
                return sharding
            else:
                corrected_sharding: Ps = es.get_corrected_named_sharding(tensor.shape, sharding).spec
                return corrected_sharding
