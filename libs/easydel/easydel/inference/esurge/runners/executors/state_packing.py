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

"""Leaf packing for compiled-call dispatch cost.

The SPMD hot path passes the model state as a flat tuple of leaves; every
compiled-call dispatch pays a fixed per-leaf host cost (~3.3us/leaf on TPU),
which for a 48-layer model (~700 leaves) is ~2.3ms per decode step — the
dominant host cost of the step.

:class:`StatePacker` reduces the leaf count by stacking groups of identical
leaves (same shape, dtype and sharding — e.g. one tensor kind across the 48
layers) into single ``[group, ...]`` device buffers.  Grouping is purely
structural, so every model family works unchanged: hybrid stacks per layer
type, quantized (codes, scales) pairs form their own groups, VLM towers
group among themselves, and any leaf without an identical sibling passes
through unpacked.

Packed copies coexist with the original state (the model and runner keep
their references), so packing is budgeted: groups are selected greedily by
leaves-eliminated-per-byte under ``max_packed_bytes``, which packs the many
small per-layer tensors (norms, projections, biases) and skips huge groups
(stacked MoE expert tables) whose dispatch share is small anyway.

Unpacking happens INSIDE the compiled function (static slices of an
invariant input — views, no copies), so consumers see the original leaf
tuple order exactly.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

_DEFAULT_MAX_PACKED_BYTES = int(1.5 * 1024**3)
"""Per-host budget for the extra device memory packed copies may occupy."""

_MIN_GROUP_LEAVES = 4
"""Groups smaller than this save too few dispatches to be worth a stack."""


@dataclasses.dataclass(frozen=True)
class _Group:
    """One packable class of identical leaves.

    Attributes:
        leaf_indices: Positions (in the original leaf tuple) stacked into
            this group, in stacking order.
        shape: Common per-leaf shape.
        dtype: Common leaf dtype (string form).
        sharding: The stacked buffer's sharding (leading axis unsharded).
    """

    leaf_indices: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    sharding: NamedSharding


class StatePacker:
    """Stack identical state leaves into grouped buffers for cheap dispatch.

    Built once from a concrete leaf tuple (shapes, dtypes and shardings are
    read from the live arrays); :meth:`pack` then converts any leaf tuple
    with the same structure into the packed argument tuple, and
    :meth:`unpack` (traceable) restores the original order inside a jit.
    """

    def __init__(self, leaves: tp.Sequence[jax.Array], *, max_packed_bytes: int = _DEFAULT_MAX_PACKED_BYTES):
        """Plan the packing from a concrete leaf tuple.

        Args:
            leaves: The flat state leaves (concrete, device-resident, carrying
                ``NamedSharding``); the plan keys on each leaf's
                (shape, dtype, sharding-spec).
            max_packed_bytes: Budget for the total bytes of packed copies
                (they coexist with the originals).  Groups are chosen greedily
                by leaves-eliminated-per-byte until the budget is exhausted.
        """
        self.num_leaves = len(leaves)
        by_key: dict[tuple, list[int]] = {}
        for idx, leaf in enumerate(leaves):
            sharding = getattr(leaf, "sharding", None)
            if not isinstance(sharding, NamedSharding):
                continue
            # ``sharding.mesh`` itself, not ``id(...)``: ``Mesh`` hashes and
            # compares structurally, so equal-but-distinct mesh objects (parts
            # of the state placed by different code paths) group together
            # instead of splitting below ``_MIN_GROUP_LEAVES`` and silently
            # losing the packing.
            key = (tuple(leaf.shape), str(leaf.dtype), str(sharding.spec), sharding.memory_kind, sharding.mesh)
            by_key.setdefault(key, []).append(idx)

        candidates: list[tuple[float, int, tuple, list[int]]] = []
        for key, idxs in by_key.items():
            if len(idxs) < _MIN_GROUP_LEAVES:
                continue
            shape, dtype = key[0], key[1]
            group_bytes = int(np.prod(shape, dtype=np.int64)) * jnp.dtype(dtype).itemsize * len(idxs)
            saved = len(idxs) - 1
            candidates.append((saved / max(group_bytes, 1), group_bytes, key, idxs))

        candidates.sort(key=lambda c: (-c[0], c[1]))
        groups: list[_Group] = []
        budget = int(max_packed_bytes)
        for _, group_bytes, key, idxs in candidates:
            if group_bytes > budget:
                continue
            budget -= group_bytes
            shape, dtype = key[0], key[1]
            leaf_sharding = leaves[idxs[0]].sharding
            stacked_sharding = NamedSharding(
                leaf_sharding.mesh,
                PartitionSpec(None, *tuple(leaf_sharding.spec)),
                memory_kind=leaf_sharding.memory_kind,
            )
            groups.append(_Group(leaf_indices=tuple(idxs), shape=tuple(shape), dtype=dtype, sharding=stacked_sharding))

        self.groups = tuple(groups)
        self._grouped_leaf_set = frozenset(i for g in self.groups for i in g.leaf_indices)
        self.passthrough_indices = tuple(i for i in range(self.num_leaves) if i not in self._grouped_leaf_set)
        self.packed_bytes = int(max_packed_bytes) - budget
        self.packed_arg_count = len(self.groups) + len(self.passthrough_indices)

    @property
    def is_effective(self) -> bool:
        """Whether packing meaningfully reduces the argument count."""
        return bool(self.groups) and self.packed_arg_count < self.num_leaves

    def pack(self, leaves: tp.Sequence[jax.Array]) -> tuple[jax.Array, ...]:
        """Stack grouped leaves into device buffers; pass singles through.

        Runs real device work (one stack per group) — call at load/weight-swap
        time, never on the hot path.

        Args:
            leaves: Leaf tuple with the same structure the packer was built
                from.

        Returns:
            ``(*stacked_groups, *passthrough_leaves)``.
        """
        if len(leaves) != self.num_leaves:
            raise ValueError(f"StatePacker built for {self.num_leaves} leaves, got {len(leaves)}.")
        packed: list[jax.Array] = []
        for group in self.groups:
            stacked = jnp.stack([leaves[i] for i in group.leaf_indices], axis=0)
            packed.append(jax.device_put(stacked, group.sharding))
        for idx in self.passthrough_indices:
            packed.append(leaves[idx])
        return tuple(packed)

    def unpack(self, packed: tp.Sequence[jax.Array]) -> tuple[jax.Array, ...]:
        """Restore the original leaf tuple from packed args (traceable).

        Inside a jit the per-leaf ``group[i]`` reads are static slices of an
        invariant input — XLA folds them into consumers with no copies.

        Args:
            packed: The tuple produced by :meth:`pack` (or matching tracers).

        Returns:
            Leaves in the original flat order.
        """
        expected = len(self.groups) + len(self.passthrough_indices)
        if len(packed) != expected:
            raise ValueError(f"StatePacker expected {expected} packed args, got {len(packed)}.")
        leaves: list[jax.Array | None] = [None] * self.num_leaves
        for g_idx, group in enumerate(self.groups):
            stacked = packed[g_idx]
            for slot, leaf_idx in enumerate(group.leaf_indices):
                leaves[leaf_idx] = stacked[slot]
        for p_idx, leaf_idx in enumerate(self.passthrough_indices):
            leaves[leaf_idx] = packed[len(self.groups) + p_idx]
        return tuple(leaves)  # type: ignore[return-value]

    def packed_shardings(self, leaf_shardings: tp.Sequence[tp.Any]) -> tuple[tp.Any, ...]:
        """Map per-leaf shardings onto the packed argument order.

        Args:
            leaf_shardings: Sharding per original leaf (same order/length).

        Returns:
            ``(*group_shardings, *passthrough_leaf_shardings)``.
        """
        if len(leaf_shardings) != self.num_leaves:
            raise ValueError(f"StatePacker built for {self.num_leaves} leaves, got {len(leaf_shardings)} shardings.")
        out: list[tp.Any] = [g.sharding for g in self.groups]
        out.extend(leaf_shardings[i] for i in self.passthrough_indices)
        return tuple(out)


__all__ = ("StatePacker",)
