# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`StagesArray` — a logical array that lives on a subset of MPMD groups.

Pipeline-parallel models produce values that exist on *some* but not
necessarily *all* pipeline stages. A stage-2 activation only exists on
stage 2; a tied embedding table might live on stages 0 and N-1 but not
the middle stages. :class:`StagesArray` gives those values a concrete
Python type instead of us passing around per-stage ``dict[int,
jax.Array]`` blobs.

In single-process runs (the default for local training / tests)
:class:`StagesArray` is a thin wrapper: each constituent :class:`jax.Array`
is fully addressable from the caller's process. In multi-process setups
some sub-arrays live in other processes and are *not* addressable here
— ``partially_addressable`` reports that.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ...core._typing import DType

__all__ = ["StagesArray", "abstract_stages_array"]


@dataclass(frozen=True)
class StagesArray:
    """An array that lives on an explicit subset of MPMD groups.

    Attributes:
        shards: Mapping from ``mpmd_idx`` to the :class:`jax.Array`
            that lives on that pipeline stage. All sub-arrays must
            share the same ``shape`` and ``dtype``.
        replicated: ``True`` if the sub-arrays hold identical values
            (e.g. a broadcasted constant). Used as a hint by reduction
            ops; not verified at construction time.
    """

    shards: Mapping[int, jax.Array]
    replicated: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate shard non-emptiness and shape/dtype homogeneity for arrays.

        When shards are plain :class:`jax.Array` s, validates that all
        have the same shape and dtype. When shards are pytrees (e.g.
        :class:`~spectrax.core.state.State` grad dicts), validation is
        skipped — the caller is responsible for structural consistency.
        """
        if not self.shards:
            raise ValueError("StagesArray must have at least one shard.")
        first = next(iter(self.shards.values()))
        if hasattr(first, "shape") and hasattr(first, "dtype"):
            shapes = {a.shape for a in self.shards.values()}
            dtypes = {a.dtype for a in self.shards.values()}
            if len(shapes) != 1:
                raise ValueError(f"StagesArray shards must have identical shapes; got {shapes}.")
            if len(dtypes) != 1:
                raise ValueError(f"StagesArray shards must have identical dtypes; got {dtypes}.")

    @property
    def mpmd_idxs(self) -> frozenset[int]:
        """Set of pipeline-stage indices this array lives on.

        Returns:
            Result described by this helper.
        """
        return frozenset(self.shards.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of each per-stage shard (all shards share the same shape).

        Returns:
            Result described by this helper.
        """
        first = next(iter(self.shards.values()))
        return first.shape

    @property
    def dtype(self) -> DType:
        """Dtype of each per-stage shard.

        Returns:
            Result described by this helper.
        """
        first = next(iter(self.shards.values()))
        return first.dtype

    @property
    def partially_addressable(self) -> bool:
        """``True`` iff some shard is not addressable from the caller's process.

        Single-process runs have every device local, so this is always
        ``False``. In multi-process mode some shards live on remote
        hosts and :attr:`shards` may only give you a subset.

        Returns:
            Result described by this helper.
        """
        local = set(jax.local_devices())
        for arr in self.shards.values():
            for dev in arr.devices():
                if dev not in local:
                    return True
        return False

    @property
    def process_index(self) -> int:
        """The calling process's index as reported by :func:`jax.process_index`.

        ``0`` in single-process runs. In multi-process runs it identifies
        which host is asking — useful when building cross-process
        placement decisions around this array's shards.

        Returns:
            Result described by this helper.
        """
        return int(jax.process_index())

    @property
    def local_shards(self) -> Mapping[int, jax.Array]:
        """Subset of :attr:`shards` whose devices are local to this process.

        In single-process mode this equals :attr:`shards`. In multi-process
        mode it returns only the entries callable code can inspect
        without crossing process boundaries.

        Returns:
            Result described by this helper.
        """
        local_devs = set(jax.local_devices())
        local: dict[int, jax.Array] = {}
        for idx, arr in self.shards.items():
            if any(d in local_devs for d in arr.devices()):
                local[idx] = arr
        return local

    @property
    def remote_mpmd_idxs(self) -> frozenset[int]:
        """Shard indices whose devices belong to other processes.

        Empty in single-process mode. When non-empty, those shards can
        only be consumed by the owning process; calling :meth:`__getitem__`
                with a remote index returns an array handle whose concrete
        values aren't addressable here.

        Returns:
            Result described by this helper.
        """
        local = set(self.local_shards.keys())
        return frozenset(self.shards.keys()) - local

    def gather_to_process(self, target_process: int) -> StagesArray:
        """Collect all shards onto devices owned by ``target_process``.

        Moves each shard whose device lies in another process onto one
        of ``target_process``'s local devices via :func:`jax.device_put`.
        JAX's unified device handling (under an initialized
        :func:`jax.distributed.initialize`) carries the bytes across
        the process boundary transparently; for single-process runs
        where ``target_process == 0`` the call is a no-op.

        Returns a new :class:`StagesArray` with every shard now resident
        on devices owned by ``target_process``. The original is
        untouched.

        Args:
                    target_process: Index of the destination process as
                        reported by :func:`jax.process_index`.

        Raises:
                    ValueError: If ``target_process`` owns no devices visible
                        to the caller (e.g. the caller holds no handle to any
                        of that process's devices and ``jax.distributed`` is
                        not initialized).

        Returns:
            Result described by this helper.
        """
        target_devs = [d for d in jax.devices() if int(d.process_index) == int(target_process)]
        if not target_devs:
            raise ValueError(
                f"no devices found for process_index={target_process}; "
                f"either the index is out of range or jax.distributed "
                f"has not been initialized so only local devices are visible."
            )
        anchor = target_devs[0]
        new_shards: dict[int, jax.Array] = {}
        for idx, arr in self.shards.items():
            if any(int(d.process_index) == int(target_process) for d in arr.devices()):
                new_shards[idx] = arr
            else:
                new_shards[idx] = jax.device_put(arr, anchor)
        return StagesArray(shards=new_shards, replicated=self.replicated)

    def __len__(self) -> int:
        """Return the number of shards.

        Returns:
            Integer length for the container.
        """
        return len(self.shards)

    def __iter__(self) -> Iterator[jax.Array]:
        """Iterate over shards in sorted index order.

        Returns:
            Iterator over the contained values.
        """
        for idx in sorted(self.shards.keys()):
            yield self.shards[idx]

    def __getitem__(self, mpmd_idx: int) -> jax.Array:
        """Return the shard at ``mpmd_idx``.

        Raises:
                    KeyError: If this :class:`StagesArray` does not live on stage
                        ``mpmd_idx``.

        Args:
            mpmd_idx: Mpmd idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        if mpmd_idx not in self.shards:
            raise KeyError(f"StagesArray has no shard at mpmd_idx={mpmd_idx}; lives on {sorted(self.mpmd_idxs)}.")
        return self.shards[mpmd_idx]

    def __contains__(self, mpmd_idx: int) -> bool:
        """Return ``True`` iff this array has a shard at ``mpmd_idx``.

        Args:
            mpmd_idx: Mpmd idx value consumed by this operation.

        Returns:
            Return ``True`` iff this array has a shard at ``mpmd_idx``.
        """
        return mpmd_idx in self.shards

    def with_shard(self, mpmd_idx: int, array: jax.Array) -> StagesArray:
        """Return a copy with shard ``mpmd_idx`` replaced by ``array``.

        Args:
            mpmd_idx: Mpmd idx value consumed by this operation.
            array: Array value consumed by this operation.

        Returns:
            Return a copy with shard ``mpmd_idx`` replaced by ``array``.
        """
        if array.shape != self.shape or array.dtype != self.dtype:
            raise ValueError(
                f"New shard has shape/dtype {array.shape}/{array.dtype}; expected {self.shape}/{self.dtype}."
            )
        new_shards = dict(self.shards)
        new_shards[mpmd_idx] = array
        return StagesArray(shards=new_shards, replicated=self.replicated)

    def to_local_array(self) -> jax.Array:
        """Return the first shard as a plain :class:`jax.Array`.

        Convenient for single-shard :class:`StagesArray` s — raises if
        the array lives on more than one stage.

        Returns:
            Return the first shard as a plain :class:`jax.Array`.
        """
        if len(self.shards) != 1:
            raise ValueError(
                f"to_local_array requires exactly one shard; have {len(self.shards)} on {sorted(self.mpmd_idxs)}."
            )
        return next(iter(self.shards.values()))

    def reduce_sum(self) -> jax.Array:
        """Host-side sum across all shards (expensive; for testing).

        Pulls every shard to host, sums, and returns a host-resident
        :class:`jax.Array`. Because shards typically live on different
        devices, a direct device-side add would fail — this routes
        through numpy on purpose.

        Returns:
            Result described by this helper.
        """
        acc: np.ndarray | None = None
        for arr in self.shards.values():
            host = np.asarray(jax.device_get(arr))
            acc = host if acc is None else acc + host
        assert acc is not None
        return jnp.asarray(acc)

    def replicated_value(self) -> jax.Array:
        """Return any shard, asserting ``replicated`` was set.

        Use when this :class:`StagesArray` was constructed to represent a
        constant that happens to be computed on multiple stages.

        Returns:
            Return any shard, asserting ``replicated`` was set.
        """
        if not self.replicated:
            raise ValueError("replicated_value() requires StagesArray(replicated=True).")
        return next(iter(self.shards.values()))


def _flatten_stages_array(arr: StagesArray):
    """PyTree flatten: ordered list of shard arrays + aux tuple of idxs.

    Args:
        arr: Arr value consumed by this operation.
    """
    idxs = tuple(sorted(arr.shards.keys()))
    leaves = tuple(arr.shards[i] for i in idxs)
    return leaves, (idxs, arr.replicated)


def _unflatten_stages_array(aux, leaves):
    """PyTree unflatten: rebuild StagesArray from leaves + aux.

    Args:
        aux: Aux value consumed by this operation.
        leaves: Leaves value consumed by this operation.
    """
    idxs, replicated = aux
    shards = dict(zip(idxs, leaves, strict=True))
    return StagesArray(shards=shards, replicated=replicated)


jax.tree_util.register_pytree_node(
    StagesArray,
    _flatten_stages_array,
    _unflatten_stages_array,
)


def _stages_array_no_ndarray_conversion(self):
    """Refuse implicit ndarray conversion.

    Letting ``jnp.asarray(stages_array)`` silently flatten an StagesArray
    to ``(N, *shape)`` would mask a class of bugs; raising a clear
    ``TypeError`` instead forces callers to pick a shard or reduce
    explicitly.
    """
    raise TypeError(
        "StagesArray cannot be converted to a single ndarray — use "
        "``arr[mpmd_idx]`` to pick a shard or ``arr.reduce_sum()`` "
        "to combine shards."
    )


StagesArray.__array__ = _stages_array_no_ndarray_conversion


def abstract_stages_array(
    shape: tuple[int, ...],
    dtype: DType,
    mpmd_idxs: frozenset[int] | set[int] | list[int] | tuple[int, ...],
    *,
    replicated: bool = False,
) -> StagesArray:
    """Build an :class:`StagesArray` of zeros with the given per-stage presence.

    Useful for multi-process placement planning: callers can declare
    "this pipeline intermediate lives on stages ``{0, 1, 3}``" without
    materializing real data. The returned array has ``len(mpmd_idxs)``
    shards of zeros, each local to the calling process by default.
    Multi-process callers can then :meth:`gather_to_process` / relocate
    individual shards as needed.

    This is a convenience constructor — it doesn't allocate on remote
    processes, so every shard comes back addressable in single-process
    mode. In multi-process mode, the caller is responsible for placing
    each shard on an appropriate device with :func:`jax.device_put`.

    Args:
        shape: Shape of each per-stage shard.
        dtype: Dtype of each per-stage shard.
        mpmd_idxs: Collection of stage indices on which this array
            should have a shard. Duplicates are dropped.
        replicated: Forwarded to :class:`StagesArray.replicated`.

    Returns:
        A new :class:`StagesArray` with a ``zeros(shape, dtype)`` shard at
        each requested stage.
    """
    idxs = tuple(sorted(set(int(i) for i in mpmd_idxs)))
    shards = {i: jnp.zeros(shape, dtype=dtype) for i in idxs}
    return StagesArray(shards=shards, replicated=replicated)
