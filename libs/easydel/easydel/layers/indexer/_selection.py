# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Common selection contract for token, pooled-block and compressed indexers.

The strategy owns projections, compression and cache updates. Selection units
are static module metadata, not values carried through JIT/scan. Attention
consumers receive array-only selections, with an optional score surrogate in
that same index domain. Logical token offsets and compressed-entry offsets
must never be interchanged; physical page translation belongs to the cache
adapter. Group-specific indexers retain their group axis throughout selection.
"""

from __future__ import annotations

import dataclasses
import math
import typing as tp

import jax
import spectrax as spx
from eformer import common_types
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array

from ._primitives import topk_selection_mask


class IndexerScoreSharding(common_types.DynamicShardingAxes):
    """Row layout of ``[batch, query, ..., candidates]`` indexer scores.

    Rows follow the activation batch/query partition; candidates stay whole so
    each shard ranks complete rows.
    """

    axes: tp.ClassVar = [common_types.BATCH, common_types.QUERY_LENGTH, common_types.EMPTY]
    mode: tp.ClassVar = 1


def _score_row_layout(mesh_source: object | None, shape: tuple[int, ...]):
    """Resolve ``(mesh, spec)`` for row-parallel selection, or ``(None, None)``."""
    mesh = getattr(mesh_source, "mesh", mesh_source)
    mesh = getattr(mesh, "jax_mesh", mesh)
    resolver = getattr(mesh_source, "runtime_sharding_resolver", None)
    if mesh is None or resolver is None or len(shape) < 3:
        return None, None
    spec = resolver.with_mesh(mesh).resolve(dynamic_axes=IndexerScoreSharding, shape=(shape[0], shape[1], shape[-1]))
    sizes = dict(mesh.shape)
    rows = []
    for dim, entry in zip(shape[:2], spec[:2], strict=True):
        axes = entry if isinstance(entry, tuple) else (() if entry is None else (entry,))
        rows.append(entry if dim % math.prod(sizes[axis] for axis in axes) == 0 else None)
    return mesh, PartitionSpec(*rows, *([None] * (len(shape) - 2)))


def top_k_indices(scores: Array, k: int, mesh_source: object | None = None) -> Array:
    """Return ``jax.lax.top_k(scores, k)[1]``; see :func:`top_k_values_indices`."""
    return top_k_values_indices(scores, k, mesh_source)[1]


def top_k_values_indices(scores: Array, k: int, mesh_source: object | None = None) -> tuple[Array, Array]:
    """Return ``jax.lax.top_k(scores, k)``, via the exact ejkernel top-k on TPU.

    ``jax.lax.top_k`` lowers to a full sort of each row on TPU; the registered
    ejkernel operation bisects for the ``k``-th key instead and returns the
    same indices bit for bit, choosing XLA itself where that is faster. A
    Pallas call is not SPMD-partitionable, so on a multi-device mesh the rows
    are split explicitly with the layout resolved from ``mesh_source`` (an
    object exposing ``mesh`` and ``runtime_sharding_resolver``, e.g. a model
    config). Without one, multi-device selection keeps ``jax.lax.top_k``.

    Args:
        scores: Floating scores ``[..., candidates]``.
        k: Static selection width, at most the candidate count.
        mesh_source: Optional mesh/resolver provider for row-parallel layout.

    Returns:
        ``(values, indices)``, each ``[..., k]``, bit-identical to
        ``jax.lax.top_k``. Take the values from here rather than gathering
        them: on TPU an element gather of ``k`` per row costs as much as a
        scatter. Unused values are dead code under jit.
    """
    if jax.default_backend() != "tpu" or not jnp.issubdtype(scores.dtype, jnp.floating):
        return jax.lax.top_k(scores, k)
    from ejkernel.modules import topk  # pyright: ignore[reportMissingTypeStubs]

    def select(rows: Array) -> tuple[Array, Array]:
        return topk(rows, k)

    ambient = jax.sharding.get_abstract_mesh()
    if not ambient.empty and ambient.manual_axes:
        auto = [name for name in ambient.axis_names if name not in ambient.manual_axes and ambient.shape[name] > 1]
        return jax.lax.top_k(scores, k) if auto else select(scores)
    mesh, spec = _score_row_layout(mesh_source, scores.shape)
    if mesh is None:
        return select(scores) if jax.device_count() == 1 else jax.lax.top_k(scores, k)
    if mesh.size == 1:
        return select(scores)
    return jax.shard_map(select, mesh=mesh, in_specs=(spec,), out_specs=(spec, spec), check_vma=False)(scores)


@dataclasses.dataclass(frozen=True)
class SelectionSpec:
    """Static index-domain contract of an indexer strategy.

    Args:
        candidate_unit: Unit ranked by top-k: token, block or compressed entry.
        output_unit: Unit consumed by attention after optional block expansion.
        grouped: Whether selections retain an independent head/group axis.
    """

    candidate_unit: tp.Literal["token", "block", "compressed_entry"]
    output_unit: tp.Literal["token", "compressed_entry"]
    grouped: bool = False

    def __post_init__(self):
        if self.candidate_unit not in ("token", "block", "compressed_entry"):
            raise ValueError(f"Unknown candidate unit: {self.candidate_unit!r}")
        if self.output_unit not in ("token", "compressed_entry"):
            raise ValueError(f"Unknown output unit: {self.output_unit!r}")
        if (self.candidate_unit == "compressed_entry") != (self.output_unit == "compressed_entry"):
            raise ValueError("Compressed entries must remain in the compressed-entry index domain.")


class IndexerSelection(tp.NamedTuple):
    """Array-only selection exchanged between an indexer and attention.

    Attributes:
        indices: Integer offsets ``[..., selected]``; -1 denotes padding.
            Leading axes may be ``[B,Q]`` or ``[B,G,Q]`` for grouped selection.
        score_proxy: Optional float scores ``[..., domain_size]`` in the
            *output* index domain, used for zero-primal straight-through bias.
            Block scores must be expanded to token scores before supplying it.
        mask: Optional dense boolean form of ``indices`` over the candidate
            domain, ``[..., domain_size]``. Selectors that can derive it
            without a scatter supply it; :meth:`to_mask` returns it when its
            width matches the requested size.
    """

    indices: Array
    score_proxy: Array | None = None
    mask: Array | None = None

    @property
    def topk_indices(self) -> Array:
        """Selected offsets (compatibility with legacy indexer outputs)."""
        return self.indices

    def to_mask(self, size: int) -> Array:
        """Scatter selected offsets into ``[..., size]`` without a one-hot cube.

        Negative and out-of-range indices select nothing; duplicates collapse.
        ``size`` is a static token or compressed-entry capacity, according to
        the strategy's :class:`SelectionSpec`. Empty selections are supported.
        """
        if size < 0:
            raise ValueError("Selection domain size must be non-negative.")
        if self.indices.ndim < 1 or not jnp.issubdtype(self.indices.dtype, jnp.integer):
            raise ValueError("Selection indices must be an integer array with a selection axis.")
        if self.mask is not None and self.mask.shape[-1] == size:
            return self.mask
        shape = self.indices.shape[:-1]
        rows = math.prod(shape)
        indices = self.indices.reshape(rows, self.indices.shape[-1])
        valid = (indices >= 0) & (indices < size)
        safe = jnp.where(valid, indices, size)
        mask = jnp.zeros((rows, size + 1), dtype=jnp.bool_)
        mask = mask.at[jnp.arange(rows)[:, None], safe].set(True)
        return mask[:, :size].reshape(*shape, size)

    def to_bias(self, size: int, dtype: jnp.dtype = jnp.float32, mask_value: float = -jnp.inf) -> Array:
        """Build additive attention bias with optional selected-score gradients.

        The hard forward bias is zero on selected positions and ``mask_value``
        elsewhere. A supplied score proxy changes only gradients, not values.
        Consumers add/broadcast the attention-head axis themselves.
        """
        mask = self.to_mask(size)
        bias = jnp.where(mask, jnp.asarray(0.0, dtype), jnp.asarray(mask_value, dtype))
        if self.score_proxy is not None:
            if self.score_proxy.shape != mask.shape:
                raise ValueError("score_proxy must have the mask shape in the output index domain.")
            # Invalid candidates may carry -inf. Sanitize before subtracting
            # so an all-invalid row cannot create inf-inf NaNs in gradients.
            scores = self.score_proxy.astype(dtype)
            scores = jnp.where(mask & jnp.isfinite(scores), scores, 0.0)
            bias = bias + (scores - jax.lax.stop_gradient(scores))
        return bias


class BaseIndexer(spx.Module):
    """Shared selection layer contract, specialized by projection/cache strategy.

    Concrete layers preserve checkpoint-native parameter names and can expose
    family-specific projection or cached-forward methods, just as attention
    implementations do. Their common selection core accepts arbitrary leading
    axes and explicitly preserves the candidate domain. No nested module or
    parallel registry is introduced by this base class.
    """

    selection_spec: tp.ClassVar[SelectionSpec] = SelectionSpec("token", "token")

    #: Optional mesh/resolver provider (e.g. the model config) for row-parallel
    #: TPU selection; adapters assign it. Not a parameter or checkpoint leaf.
    mesh_source = None

    @staticmethod
    def select_candidates(
        scores: Array, k: int, valid: Array | None = None, mesh_source: object | None = None
    ) -> IndexerSelection:
        """Rank candidates and return clamped-width, -1-padded selections.

        Args:
            scores: Floating point scores ``[..., candidates]``.
            k: Non-negative static budget, clamped to candidate capacity.
            valid: Optional broadcastable validity mask applied *before*
                ranking. NaN and -inf scores are unselectable; +inf is allowed
                for strategies that promote local blocks within the budget.
            mesh_source: Optional mesh/resolver provider; see :func:`top_k_indices`.

        Returns:
            Selection in the candidate domain. Block strategies explicitly
            expand it to their output domain before passing it to attention.
        """
        if k < 0:
            raise ValueError("Selection budget must be non-negative.")
        eligible = ~jnp.isnan(scores) & (scores > -jnp.inf)
        if valid is not None:
            eligible = eligible & valid.astype(jnp.bool_)
        masked = jnp.where(eligible, scores, -jnp.inf)
        values, indices = top_k_values_indices(masked, min(k, scores.shape[-1]), mesh_source)
        # Eligible scores are never NaN or -inf, so the picked value tells
        # eligibility without gathering it.
        picked = values > -jnp.inf
        mask = None
        if indices.shape[-1] > 0:
            # Unused masks are dead code under jit.
            mask = topk_selection_mask(masked, values, indices) & eligible
        return IndexerSelection(jnp.where(picked, indices, -1).astype(jnp.int32), mask=mask)
