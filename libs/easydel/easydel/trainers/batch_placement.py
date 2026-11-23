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

"""Batch-sharded placement of trainer step inputs.

A trainer step compiled with a replicated ``in_shardings`` entry for its batch
copies the whole global batch to every device before the in-step
``constrain_batch_sharding`` reshards it. :class:`BatchPlacedStep` instead puts
each batch leaf on the mesh with the same rank-aware spec the step constrains
to, and the step declares that argument ``None`` so jit keeps the placement.
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from .utils import is_numeric_dtype


def _jax_mesh(mesh: tp.Any) -> jax.sharding.Mesh:
    return mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh


def _restrict_spec_to_mesh(partition_spec: PartitionSpec, axis_names: set[str]) -> PartitionSpec:
    """Drop axis names the mesh does not define so ``NamedSharding`` accepts the spec."""

    def _entry(entry):
        if entry is None:
            return None
        names = tuple(entry) if isinstance(entry, (tuple, list)) else (entry,)
        kept = tuple(name for name in names if name in axis_names)
        if not kept:
            return None
        return kept if len(kept) > 1 else kept[0]

    return PartitionSpec(*(_entry(entry) for entry in partition_spec))


def batch_leaf_sharding(
    shape: tuple[int, ...],
    partition_spec: PartitionSpec,
    mesh: tp.Any,
    *,
    key: str | None = None,
) -> NamedSharding:
    """Return the sharding the step's ``constrain_batch_sharding`` would give a leaf of ``shape``.

    Args:
        shape: Global leaf shape.
        partition_spec: Trainer batch spec, e.g. ``P(("dp", "fsdp"), "sp")``.
        mesh: Mesh (JAX or SpectraX) the step runs on.
        key: Optional dictionary key name (used to disambiguate rank-3 mRoPE
            ``(3, B, S)`` ``position_ids`` from ``(B, S, D)`` tensors).

    Returns:
        A ``NamedSharding`` on ``mesh``; fully replicated when no batch axis divides the leaf.
    """
    from .training_utils import _leaf_batch_partition_spec

    jax_mesh = _jax_mesh(mesh)
    axis_sizes = {str(name): int(size) for name, size in dict(jax_mesh.shape).items()}
    spec = _restrict_spec_to_mesh(partition_spec, set(axis_sizes))
    return NamedSharding(jax_mesh, _leaf_batch_partition_spec(tuple(int(d) for d in shape), spec, axis_sizes, key=key))


def _batch_axes_for_mesh(partition_spec: PartitionSpec, axis_names: tp.Sequence[str]) -> tuple[str, ...]:
    """Return mesh axis names that shard the batch dimension (entry 0 of ``partition_spec``)."""
    entries = tuple(partition_spec)
    if not entries or entries[0] is None:
        return ()
    raw = tuple(entries[0]) if isinstance(entries[0], (tuple, list)) else (entries[0],)
    known = set(axis_names)
    return tuple(str(name) for name in raw if name in known)


def batch_shard_group_rows(
    global_batch_size: int,
    partition_spec: PartitionSpec,
    mesh: tp.Any,
    *,
    process_index: int | None = None,
) -> tuple[int, ...]:
    """Return sorted unique global batch row indices owned by this process's batch-shard group.

    On multi-host meshes where each batch-shard process group (see
    :func:`~easydel.trainers.data_lanes.batch_shard_process_groups`) owns a
    strict subset of the global batch rows, every process in the same group
    receives the exact same sorted tuple of row indices, and the tuples across
    all groups partition ``range(global_batch_size)`` without overlap.

    When the batch is replicated across all processes (no batch axes on
    ``mesh``, or ``global_batch_size`` is not divisible by the batch-axis
    product), returns ``tuple(range(global_batch_size))``.
    """
    global_batch_size = int(global_batch_size)
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}.")
    jax_mesh = _jax_mesh(mesh)
    devices = np.asarray(jax_mesh.devices)
    axis_names = tuple(str(name) for name in jax_mesh.axis_names)
    batch_axes = _batch_axes_for_mesh(partition_spec, axis_names)
    if not batch_axes:
        return tuple(range(global_batch_size))

    axis_sizes = {name: int(devices.shape[i]) for i, name in enumerate(axis_names)}
    batch_ways = 1
    for name in batch_axes:
        batch_ways *= axis_sizes[name]
    if batch_ways <= 1 or global_batch_size % batch_ways != 0:
        return tuple(range(global_batch_size))

    from .data_lanes import batch_shard_process_groups

    groups = batch_shard_process_groups(devices, axis_names, batch_axes)
    proc = jax.process_index() if process_index is None else int(process_index)
    my_group = next((group for group in groups if proc in group), None)
    if my_group is None:
        return tuple(range(global_batch_size))
    my_group_set = set(my_group)

    rows: set[int] = set()
    batch_entry: str | tuple[str, ...] = batch_axes if len(batch_axes) > 1 else batch_axes[0]
    if isinstance(jax_mesh, jax.sharding.Mesh):
        sharding = NamedSharding(jax_mesh, PartitionSpec(batch_entry))
        idx_map = sharding.devices_indices_map((global_batch_size,))
        for dev, slc in idx_map.items():
            if int(dev.process_index) in my_group_set:
                rows.update(range(*slc[0].indices(global_batch_size)))
    else:
        batch_dims = [axis_names.index(name) for name in batch_axes]
        batch_sizes = [axis_sizes[name] for name in batch_axes]
        shard_size = global_batch_size // batch_ways
        for pos in np.ndindex(devices.shape):
            if int(devices[pos].process_index) not in my_group_set:
                continue
            rank = 0
            for dim, size in zip(batch_dims, batch_sizes, strict=True):
                rank = rank * size + pos[dim]
            start = rank * shard_size
            rows.update(range(start, start + shard_size))
    return tuple(sorted(rows))


def _local_slice_for_global_index(
    host: np.ndarray,
    index: tuple[slice, ...],
    batch_axis: int,
    global_batch_size: int,
    global_to_local: dict[int, int],
) -> np.ndarray:
    """Slice a local host array ``host`` for the global shard ``index`` along ``batch_axis``."""
    g_rows = range(*index[batch_axis].indices(global_batch_size))
    l_rows = [global_to_local[g] for g in g_rows]
    if l_rows and l_rows[-1] - l_rows[0] + 1 == len(l_rows) and l_rows == list(range(l_rows[0], l_rows[-1] + 1)):
        local_index = (*index[:batch_axis], slice(l_rows[0], l_rows[-1] + 1), *index[batch_axis + 1 :])
        return host[local_index]
    other_index = (*index[:batch_axis], slice(None), *index[batch_axis + 1 :])
    return np.take(host[other_index], l_rows, axis=batch_axis)


def _is_mrope_position_key(key: tp.Any, shape: tuple[int, ...]) -> bool:
    return (
        isinstance(key, str)
        and (key == "position_ids" or key.endswith("_position_ids"))
        and len(shape) == 3
        and int(shape[0]) == 3
    )


_FLAT_PACKED_EMBED_KEYS = frozenset({"image_embeds", "image_embed_positions", "image_embed_mask"})


def place_batch(
    batch: tp.Any,
    partition_spec: PartitionSpec,
    mesh: tp.Any,
    *,
    global_batch_size: int | None = None,
) -> tp.Any:
    """Put every array leaf of ``batch`` on ``mesh`` with its batch-sharded placement.

    When ``global_batch_size`` is ``None`` (default), host arrays are treated as
    the global batch, the same contract the replicated ``in_shardings`` relied
    on: every process passes the full batch and only the addressable shards are
    transferred.

    When ``global_batch_size`` is set, host arrays whose batch dimension matches
    this process's batch-shard group size (see :func:`batch_shard_group_rows`)
    are lifted directly into the global sharded ``jax.Array`` of shape
    ``global_batch_size`` along their batch axis, avoiding redundant host-side
    loading of rows owned by other process groups. Host arrays already carrying
    ``global_batch_size`` rows continue to use direct global slicing.

    Device arrays already on the target sharding are returned unchanged; others
    are resharded. Non-array leaves (``None``, Python scalars) pass through.

    Args:
        batch: Batch pytree, usually a flat dict.
        partition_spec: Trainer batch spec.
        mesh: Mesh the step runs on.
        global_batch_size: Optional global batch size for lifting per-group
            local host batches into global sharded arrays.

    Returns:
        The batch with array leaves committed to their shardings.
    """
    from .training_utils import _NON_BATCH_LEADING_KEYS, _infer_batch_size

    group_rows: tuple[int, ...] | None = None
    global_to_local: dict[int, int] | None = None
    local_batch_size: int | None = None
    lifting_local = False

    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        group_rows = batch_shard_group_rows(global_batch_size, partition_spec, mesh)
        local_batch_size = len(group_rows)
        if local_batch_size < global_batch_size:
            host_leaves = [
                (path, np.asarray(leaf))
                for path, leaf in jax.tree_util.tree_leaves_with_path(batch)
                if not isinstance(leaf, jax.Array)
                and (isinstance(leaf, np.ndarray) or (hasattr(leaf, "__array__") and hasattr(leaf, "shape")))
            ]
            numeric_host = [
                (path, arr) for path, arr in host_leaves if arr.ndim > 0 and is_numeric_dtype(arr.dtype)
            ]
            if numeric_host:
                host_batch_view = {
                    (
                        next((e.key for e in reversed(path) if isinstance(e, jax.tree_util.DictKey)), f"_leaf_{i}")
                    ): arr
                    for i, (path, arr) in enumerate(numeric_host)
                }
                inferred = _infer_batch_size(host_batch_view)
                if inferred == local_batch_size:
                    lifting_local = True
                    global_to_local = {g: i for i, g in enumerate(group_rows)}
                    for path, _arr in numeric_host:
                        key = next((e.key for e in reversed(path) if isinstance(e, jax.tree_util.DictKey)), None)
                        if key in _FLAT_PACKED_EMBED_KEYS:
                            raise ValueError(
                                f"place_batch cannot lift local host batch containing flat packed embed key {key!r} "
                                f"(local_batch_size={local_batch_size} < global_batch_size={global_batch_size}); "
                                "disable `data_lane_local_batch` for flat packed multimodal embeds."
                            )
                elif inferred != global_batch_size:
                    raise ValueError(
                        f"place_batch expected host batch size {local_batch_size} (local group) or "
                        f"{global_batch_size} (global), got {inferred}."
                    )

    def _place(path: tuple[tp.Any, ...], leaf: tp.Any) -> tp.Any:
        key = next((entry.key for entry in reversed(path) if isinstance(entry, jax.tree_util.DictKey)), None)
        if isinstance(leaf, jax.Array):
            target = batch_leaf_sharding(tuple(leaf.shape), partition_spec, mesh, key=key)
            current = getattr(leaf, "sharding", None)
            if isinstance(current, NamedSharding) and current.is_equivalent_to(target, leaf.ndim):
                return leaf
            return jax.device_put(leaf, target)
        if isinstance(leaf, np.ndarray) or (hasattr(leaf, "__array__") and hasattr(leaf, "shape")):
            host = np.asarray(leaf)
            if not is_numeric_dtype(host.dtype):
                return leaf
            if (
                lifting_local
                and global_batch_size is not None
                and local_batch_size is not None
                and global_to_local is not None
                and host.ndim > 0
                and key not in _NON_BATCH_LEADING_KEYS
            ):
                if _is_mrope_position_key(key, tuple(host.shape)):
                    batch_axis = 1
                else:
                    batch_axis = 0
                if int(host.shape[batch_axis]) == local_batch_size:
                    global_shape = (
                        *host.shape[:batch_axis],
                        global_batch_size,
                        *host.shape[batch_axis + 1 :],
                    )
                    target = batch_leaf_sharding(global_shape, partition_spec, mesh, key=key)
                    if len(target.spec) > batch_axis and target.spec[batch_axis] is not None:
                        return jax.make_array_from_callback(
                            global_shape,
                            target,
                            lambda index, h=host, ax=batch_axis: _local_slice_for_global_index(
                                h, index, ax, global_batch_size, global_to_local
                            ),
                        )
            target = batch_leaf_sharding(tuple(host.shape), partition_spec, mesh, key=key)
            return jax.make_array_from_callback(host.shape, target, lambda index: host[index])
        return leaf

    return jax.tree_util.tree_map_with_path(_place, batch)


def replicated_batch_slot_spec(arguments: tp.Any, mesh: tp.Any, in_shardings: tp.Any) -> PartitionSpec | None:
    """Return the batch spec to place argument 1 with, or ``None`` to leave the step unchanged.

    Only primary trainer steps qualify: ``arguments`` carries a ``step_partition_spec``,
    the mesh is SPMD, and ``in_shardings[1]`` is the fully replicated batch slot.
    """
    if arguments is None or mesh is None or bool(getattr(mesh, "is_mpmd", False)):
        return None
    if not isinstance(in_shardings, (tuple, list)) or len(in_shardings) < 2:
        return None
    slot = in_shardings[1]
    if not isinstance(slot, NamedSharding) or len(tuple(slot.spec)) != 0:
        return None
    spec = getattr(arguments, "step_partition_spec", None)
    return spec if isinstance(spec, PartitionSpec) else None


def release_batch_slot(in_shardings: tp.Any, batch_spec: PartitionSpec | None) -> tp.Any:
    """Replace the replicated batch slot with ``None`` when the batch will be placed by the caller."""
    if batch_spec is None:
        return in_shardings
    return (in_shardings[0], None, *in_shardings[2:])


def wrap_batch_placed(
    fn: tp.Callable[..., tp.Any],
    batch_spec: PartitionSpec | None,
    mesh: tp.Any,
    *,
    global_batch_size: int | None = None,
):
    """Wrap ``fn`` in :class:`BatchPlacedStep` when ``batch_spec`` is set, else return it unchanged."""
    if batch_spec is None:
        return fn
    return BatchPlacedStep(
        fn,
        batch_argnum=1,
        partition_spec=batch_spec,
        mesh=mesh,
        global_batch_size=global_batch_size,
    )


class BatchPlacedStep:
    """Callable wrapper that places argument ``batch_argnum`` before calling a compiled step.

    The wrapped function must have been compiled with ``None`` as the
    ``in_shardings`` entry for that argument. ``lower`` places the batch too, so
    AOT compilation sees the same input shardings as the runtime call.
    Attribute reads and writes other than the wrapper's own fall through to the
    compiled function (e.g. the ``static_argnums_`` marker trainers set).
    """

    _own = frozenset({"_fn", "_batch_argnum", "_partition_spec", "_mesh", "_global_batch_size"})

    def __init__(
        self,
        fn: tp.Callable[..., tp.Any],
        *,
        batch_argnum: int,
        partition_spec: PartitionSpec,
        mesh,
        global_batch_size: int | None = None,
    ):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_batch_argnum", int(batch_argnum))
        object.__setattr__(self, "_partition_spec", partition_spec)
        object.__setattr__(self, "_mesh", mesh)
        object.__setattr__(
            self,
            "_global_batch_size",
            None if global_batch_size is None else int(global_batch_size),
        )

    def with_global_batch_size(self, global_batch_size: int | None) -> BatchPlacedStep:
        """Return a shallow wrapper around the same compiled function with ``global_batch_size`` set."""
        return BatchPlacedStep(
            self._fn,
            batch_argnum=self._batch_argnum,
            partition_spec=self._partition_spec,
            mesh=self._mesh,
            global_batch_size=global_batch_size,
        )

    def _placed(self, args: tuple[tp.Any, ...]) -> tuple[tp.Any, ...]:
        i = self._batch_argnum
        if len(args) <= i:
            return args
        if self._global_batch_size is None:
            placed = place_batch(args[i], self._partition_spec, self._mesh)
        else:
            placed = place_batch(
                args[i],
                self._partition_spec,
                self._mesh,
                global_batch_size=self._global_batch_size,
            )
        return (*args[:i], placed, *args[i + 1 :])

    def __call__(self, *args, **kwargs):
        return self._fn(*self._placed(args), **kwargs)

    def lower(self, *args, **kwargs):
        return self._fn.lower(*self._placed(args), **kwargs)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_fn"), name)

    def __setattr__(self, name: str, value: tp.Any) -> None:
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._fn, name, value)
