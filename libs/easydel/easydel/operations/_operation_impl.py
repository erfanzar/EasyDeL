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
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Bool, Float
from spectrax import common_types

from ._base_operation import BaseOperation, OperationRegistry
from ._operation_meta import OperationMetadata
from .requirements import ExecutionMode, OperationRequirements

__all__ = ["OperationImpl", "OperationMetadata", "OperationOutput", "OperationRegistry"]

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
    """Base pytree-compatible container for operation results.

    Concrete operation outputs (such as ``AttentionOutput``) inherit from this
    class. The ``@auto_pytree`` decoration ensures subclasses can be used inside
    JAX transformations (``jit``, ``vmap``, ``scan``) without manual flatten/
    unflatten registration.

    Subclasses define operation-specific fields (e.g. attention weights, cache
    views) on top of this empty base.
    """


class OperationImpl(BaseOperation):
    """Attention-flavoured :class:`BaseOperation` with shared kernel helpers.

    Concrete attention operators in :mod:`easydel.operations.kernels`
    subclass this rather than :class:`BaseOperation` directly to inherit:

    * Backend-aware dispatch via :class:`BaseOperation.__call__`.
    * The mask/segment/head-repeat utilities defined below
      (:meth:`_split_attention_mask`, :meth:`_combine_query_kv_masks`,
      :meth:`_create_causal_mask`, :meth:`repeat_kv_heads`,
      :meth:`_handle_kvhead`).
    * Mode discrimination via :meth:`get_mode` (decode vs train) and
      sharding-spec construction via :meth:`create_stable_sharding`.

    Subclass contract:

    * MUST implement :meth:`forward_native` (inherited from
      :class:`BaseOperation`) and :meth:`BaseOperation.get_impl_name`.
    * MAY override :meth:`get_requirements` to declare per-operator
      metadata / cache requirements.
    * MAY override :meth:`forward_tpu`, :meth:`forward_gpu`, etc. for
      hardware-specialised paths.

    Attributes:
        metadata (OperationMetadata): Runtime configuration assigned in
            :meth:`__init__`; never ``None`` after construction.
    """

    def __init__(self, metadata: OperationMetadata) -> None:
        """Initialize the operator with its runtime metadata.

        Args:
            metadata: Runtime configuration carrying dtype, mesh, sharding
                policy, backend selection, and optional per-operation
                ejkernel configs.
        """
        self.metadata = metadata

    def get_impl_metadata(self) -> OperationMetadata:
        """Return the :class:`OperationMetadata` carried by this instance.

        Returns:
            The metadata supplied to :meth:`__init__`.

        Raises:
            RuntimeError: If ``self.metadata`` is ``None`` — this indicates
                the instance was constructed through a non-standard path
                that bypassed :meth:`__init__`.
        """
        if self.metadata is None:
            raise RuntimeError("metadata must be set before calling this method")
        return self.metadata

    def get_instance_requirements(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Return :meth:`get_requirements` with instance overrides applied.

        Wraps the class-level :meth:`BaseOperation.get_requirements` and
        layers on the instance-specific ``requires_cache`` override taken
        from :attr:`OperationMetadata.requires_cache`. Prefer this method
        whenever you need the requirements that will actually be honoured
        for a given instance — for example, when a vision-encoder
        operation has been constructed with ``requires_cache=False``.

        Args:
            mode: Execution mode (``PREFILL``, ``DECODE``, or ``MIXED``)
                forwarded to :meth:`get_requirements`.

        Returns:
            OperationRequirements that reflect both the class-level
            declaration and any instance-level overrides.

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
        """Infer the runtime mode (train vs single-step decode) from query shape.

        Treats a query sequence length of 1 as single-step generation; any
        other length is treated as training / prefill.

        Args:
            query: Query tensor in either ``(B, T, H, D)`` or ``(B, H, T, D)``
                layout depending on ``BTHD``.
            BTHD: If True, the sequence length is read from axis 1
                (``B, T, H, D`` layout). If False, it is read from axis 2
                (``B, H, T, D`` layout). Defaults to True.

        Returns:
            ``common_types.MODE_DECODE`` when the query length is 1, else
            ``common_types.MODE_TRAIN``.
        """
        in_generation = query.shape[1] == 1 if BTHD else query.shape[2] == 1
        return common_types.MODE_DECODE if in_generation else common_types.MODE_TRAIN

    @staticmethod
    def _split_attention_mask(
        attn_mask: Bool[Array, "... seq_len seq_len"],
    ) -> tuple[Bool[Array, "... seq_len"], Bool[Array, "... seq_len"]]:
        """Split a 2D attention mask into separate query and key-value masks.

        For 4D inputs ``(batch, head, q_seq, kv_seq)`` only the last head
        slice is used; 3D inputs ``(batch, q_seq, kv_seq)`` are consumed
        directly. The query mask marks positions that can attend to *any*
        key, and the kv mask marks positions that can be attended to by
        *any* query.

        Args:
            attn_mask: Combined attention mask, shape ``(..., q_seq, kv_seq)``,
                either 3D or 4D.

        Returns:
            A tuple ``(q_mask, kv_mask)``:

            * ``q_mask`` — bool array ``(..., q_seq)``, ``True`` for valid
              query tokens.
            * ``kv_mask`` — bool array ``(..., kv_seq)``, ``True`` for valid
              key/value tokens.
        """
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, -1, :, :]
        return jnp.any(attn_mask, axis=-1), jnp.any(attn_mask, axis=-2)

    @staticmethod
    def _combine_query_kv_masks(
        q_mask: Bool[Array, "... q_seq"], kv_mask: Bool[Array, "... kv_seq"]
    ) -> Bool[Array, "... q_seq kv_seq"]:
        """Combine separate query and key-value masks into a 2D attention mask.

        Produces the outer-product-style mask where
        ``mask[b, i, j] == q_mask[b, i] & kv_mask[b, j]``. Adds the missing
        singleton axis on either input as needed before the product, so
        callers can pass purely 2D ``q_mask``/``kv_mask`` without manual
        broadcasting.

        Args:
            q_mask: Bool array ``(..., q_seq)``. ``True`` for valid queries.
            kv_mask: Bool array ``(..., kv_seq)``. ``True`` for valid
                key/value tokens.

        Returns:
            Bool attention mask of shape ``(..., q_seq, kv_seq)``.
        """
        if kv_mask.ndim == 2:
            kv_mask = kv_mask[:, None, :]
        if q_mask.ndim == 2:
            q_mask = q_mask[:, :, None]
        return q_mask * kv_mask

    @staticmethod
    def _create_causal_mask(qseq: int) -> Bool[Array, "seq_len seq_len"]:
        """Build a square causal (lower-triangular) attention mask.

        Args:
            qseq: Sequence length on both axes.

        Returns:
            Bool array of shape ``(qseq, qseq)`` where ``mask[i, j]`` is
            ``True`` iff ``j <= i`` — that is, position ``i`` can attend
            to all positions up to and including itself.
        """
        return jnp.tril(jnp.ones((qseq, qseq), dtype="b1"))

    @staticmethod
    def repeat_kv_heads(
        k: Float[Array, "batch seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch seq_len num_kv_heads head_dim"],
        num_reps: int,
    ) -> tuple[Float[Array, "batch seq_len num_q_heads head_dim"], Float[Array, "batch seq_len num_q_heads head_dim"]]:
        """Repeat K/V heads to match the query head count (GQA / MQA).

        Each KV head is duplicated ``num_reps`` times along the head axis so
        the result has ``num_kv_heads * num_reps == num_q_heads`` heads.

        Args:
            k: Key tensor of shape ``(batch, seq_len, num_kv_heads, head_dim)``.
            v: Value tensor of shape ``(batch, seq_len, num_kv_heads, head_dim)``.
            num_reps: Repetition factor, typically
                ``num_q_heads // num_kv_heads``.

        Returns:
            Tuple ``(k_repeated, v_repeated)`` both of shape
            ``(batch, seq_len, num_q_heads, head_dim)``.
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
        """Normalise an attention-bias-shaped array to ``num_q_heads`` heads.

        Used by attention kernels to align an externally-supplied bias /
        mask tensor with the query head count under GQA / MQA. The head
        dimension is assumed to live at axis 1.

        * If the head dimension already equals ``num_q_heads`` or is 1
          (broadcastable), the input is returned unchanged.
        * If it equals ``num_kv_heads``, each head is repeated
          ``num_q_heads // num_kv_heads`` times.
        * Otherwise a :class:`ValueError` is raised.

        Args:
            array: Attention bias / mask of shape
                ``(batch, num_heads, q_seq, kv_seq)``, or ``None``.
            num_q_heads: Target number of query heads.
            num_kv_heads: Number of key/value heads (used as the fallback
                head count to expand from).

        Returns:
            Array with head dimension equal to ``num_q_heads``, or ``None``
            if the input was ``None``.

        Raises:
            ValueError: If the array's head dimension is not ``num_q_heads``,
                ``num_kv_heads``, or ``1``.
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
        """Construct an axis-selective :class:`PartitionSpec` for an intermediate.

        Used by attention kernels to derive a PartitionSpec for an internal
        buffer that only preserves a subset of the input's axes (the rest
        are replicated). Optionally corrects the resulting spec against a
        concrete tensor shape via
        :func:`spectrax.get_corrected_named_sharding`.

        Args:
            state_ps: The base PartitionSpec to slice axes out of.
            preserved_indices: Axis indices to retain from ``state_ps`` (or
                from ``clone_ps`` if given). All other axes are set to
                ``None`` (replicated). If ``None``, ``state_ps`` is returned
                unchanged (subject to ``tensor`` correction).
            clone_ps: Alternate PartitionSpec to source axis names from for
                the preserved indices. Defaults to ``state_ps``.
            dep: Dependency gate — if ``None``, the function returns
                ``None`` immediately. Lets callers conditionally suppress
                sharding without an outer ``if`` branch.
            tensor: Optional tensor used to correct the resulting spec for
                the actual shape (handles axis-name → mesh-axis mapping).

        Returns:
            A PartitionSpec with only the preserved axes partitioned, or
            ``None`` when ``dep`` / ``state_ps`` / the metadata's mesh is
            absent.
        """
        mesh = self.metadata.mesh
        if mesh is None:
            return state_ps
        with mesh:
            if dep is None:
                return None

            if state_ps is None:
                return None

            if preserved_indices is None:
                if tensor is None:
                    return state_ps
                corrected: Ps = spx.get_corrected_named_sharding(tensor.shape, state_ps).spec
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
                corrected_sharding: Ps = spx.get_corrected_named_sharding(tensor.shape, sharding).spec
                return corrected_sharding
