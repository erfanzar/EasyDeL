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


def batch_leaf_sharding(shape: tuple[int, ...], partition_spec: PartitionSpec, mesh: tp.Any) -> NamedSharding:
    """Return the sharding the step's ``constrain_batch_sharding`` would give a leaf of ``shape``.

    Args:
        shape: Global leaf shape.
        partition_spec: Trainer batch spec, e.g. ``P(("dp", "fsdp"), "sp")``.
        mesh: Mesh (JAX or SpectraX) the step runs on.

    Returns:
        A ``NamedSharding`` on ``mesh``; fully replicated when no batch axis divides the leaf.
    """
    from .training_utils import _leaf_batch_partition_spec

    jax_mesh = _jax_mesh(mesh)
    axis_sizes = {str(name): int(size) for name, size in dict(jax_mesh.shape).items()}
    spec = _restrict_spec_to_mesh(partition_spec, set(axis_sizes))
    return NamedSharding(jax_mesh, _leaf_batch_partition_spec(tuple(int(d) for d in shape), spec, axis_sizes))


def place_batch(batch: tp.Any, partition_spec: PartitionSpec, mesh: tp.Any) -> tp.Any:
    """Put every array leaf of ``batch`` on ``mesh`` with its batch-sharded placement.

    Host arrays are treated as the global batch, the same contract the replicated
    ``in_shardings`` relied on: every process passes the full batch and only the
    addressable shards are transferred. Device arrays already on the target
    sharding are returned unchanged; others are resharded. Non-array leaves
    (``None``, Python scalars) pass through.

    Args:
        batch: Batch pytree, usually a flat dict.
        partition_spec: Trainer batch spec.
        mesh: Mesh the step runs on.

    Returns:
        The batch with array leaves committed to their shardings.
    """

    def _place(leaf: tp.Any) -> tp.Any:
        if isinstance(leaf, jax.Array):
            target = batch_leaf_sharding(tuple(leaf.shape), partition_spec, mesh)
            current = getattr(leaf, "sharding", None)
            if isinstance(current, NamedSharding) and current.is_equivalent_to(target, leaf.ndim):
                return leaf
            return jax.device_put(leaf, target)
        if isinstance(leaf, np.ndarray) or (hasattr(leaf, "__array__") and hasattr(leaf, "shape")):
            host = np.asarray(leaf)
            if not is_numeric_dtype(host.dtype):
                return leaf
            target = batch_leaf_sharding(tuple(host.shape), partition_spec, mesh)
            return jax.make_array_from_callback(host.shape, target, lambda index: host[index])
        return leaf

    if isinstance(batch, dict):
        return {key: jax.tree_util.tree_map(_place, value) for key, value in batch.items()}
    return jax.tree_util.tree_map(_place, batch)


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


def wrap_batch_placed(fn: tp.Callable[..., tp.Any], batch_spec: PartitionSpec | None, mesh: tp.Any):
    """Wrap ``fn`` in :class:`BatchPlacedStep` when ``batch_spec`` is set, else return it unchanged."""
    if batch_spec is None:
        return fn
    return BatchPlacedStep(fn, batch_argnum=1, partition_spec=batch_spec, mesh=mesh)


class BatchPlacedStep:
    """Callable wrapper that places argument ``batch_argnum`` before calling a compiled step.

    The wrapped function must have been compiled with ``None`` as the
    ``in_shardings`` entry for that argument. ``lower`` places the batch too, so
    AOT compilation sees the same input shardings as the runtime call.
    Attribute reads and writes other than the wrapper's own fall through to the
    compiled function (e.g. the ``static_argnums_`` marker trainers set).
    """

    _own = frozenset({"_fn", "_batch_argnum", "_partition_spec", "_mesh"})

    def __init__(self, fn: tp.Callable[..., tp.Any], *, batch_argnum: int, partition_spec: PartitionSpec, mesh):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_batch_argnum", int(batch_argnum))
        object.__setattr__(self, "_partition_spec", partition_spec)
        object.__setattr__(self, "_mesh", mesh)

    def _placed(self, args: tuple[tp.Any, ...]) -> tuple[tp.Any, ...]:
        i = self._batch_argnum
        if len(args) <= i:
            return args
        return (*args[:i], place_batch(args[i], self._partition_spec, self._mesh), *args[i + 1 :])

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
