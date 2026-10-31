# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Microbatch reshape helpers for the SpectraX MPMD runtime."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from .sharding import _spec_axis_shape_mismatches


def _has_microbatch_axis(x: object) -> bool:
    """Return whether ``x`` has a leading axis that can be split.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Return whether ``x`` has a leading axis that can be split.
    """
    return hasattr(x, "shape") and getattr(x, "ndim", len(getattr(x, "shape", ()))) > 0


def _microbatch_sample(x: object, m: int) -> object:
    """Return microbatch 0 for batch leaves and pass shared leaves through.

    Args:
        x: Input value consumed by the operation.
        m: M value consumed by this operation.

    Returns:
        Return microbatch 0 for batch leaves and pass shared leaves through.
    """
    if not _has_microbatch_axis(x):
        return x
    return _microbatch(x, m)[0]


def _microbatch(x: jax.Array, m: int) -> jax.Array:
    """Reshape a ``(B, ...)`` array into ``(m, B // m, ...)`` microbatches.

    The leading batch axis is split into ``m`` microbatches of equal
    size; ``B`` must be evenly divisible by ``m``. Non-array or scalar
    leaves are returned unchanged as a safety net.

    Args:
        x: Input array with leading batch dimension.
        m: Number of microbatches.

    Returns:
        A new array shaped ``(m, B // m, *x.shape[1:])``, or ``x`` itself
        if it has no batch dimension.

    Raises:
        ValueError: If ``B`` is not a multiple of ``m``.
    """
    if not _has_microbatch_axis(x):
        return x
    b = x.shape[0]
    if b % m:
        raise ValueError(f"Batch size {b} not divisible by number of microbatches {m}.")
    return cast(jax.Array, _split_microbatch_stack(x, m, context="_microbatch"))


def _partition_spec_entry_axes(axis: object) -> tuple[object, ...]:
    """Return the concrete mesh-axis entries used by one ``PartitionSpec`` item."""
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(part for part in axis if part is not None)
    return (axis,)


def _merged_flattened_partition_axis(left: object, right: object) -> object:
    """Merge the sharding axes for dimensions collapsed by a reshape."""
    axes = _partition_spec_entry_axes(left) + _partition_spec_entry_axes(right)
    if not axes:
        return None
    if len(set(axes)) != len(axes):
        raise ValueError(
            "SpectraX cannot flatten a microbatch stack whose first two dimensions "
            f"reuse a mesh axis: left={left!r}, right={right!r}."
        )
    return axes[0] if len(axes) == 1 else axes


def _named_sharding_with_memory_kind(sharding: jax.sharding.NamedSharding, memory_kind: object) -> object:
    """Attach ``memory_kind`` to a ``NamedSharding`` when the backend exposes it."""
    if memory_kind is None or not hasattr(sharding, "with_memory_kind"):
        return sharding
    try:
        return sharding.with_memory_kind(memory_kind)
    except Exception:
        return sharding


def _reshape_with_named_shardings(
    value: object,
    out_shape: tuple[int, ...],
    *,
    in_sharding: jax.sharding.NamedSharding,
    out_sharding: object,
) -> object:
    """Run an eager reshape with explicit input/output shardings."""

    def _reshape(x):
        return jnp.reshape(x, out_shape)

    return jax.jit(_reshape, in_shardings=in_sharding, out_shardings=out_sharding)(value)


def _split_microbatch_stack(value: object, m: int, *, context: str) -> object:
    """Split ``(batch, ...)`` into ``(microbatch, batch_per_microbatch, ...)``.

    The reshape introduces a scheduler axis. That new axis is logical control
    flow, not model/data parallelism, so any existing sharding on the original
    batch dimension must move to the new per-microbatch batch dimension.
    """
    if not _has_microbatch_axis(value):
        return value
    shape = tuple(int(dim) for dim in value.shape)
    batch = shape[0]
    if batch % m:
        raise ValueError(f"Batch size {batch} not divisible by number of microbatches {m}.")

    out_shape = (int(m), batch // int(m), *shape[1:])
    source_sharding = getattr(value, "sharding", None)
    if not isinstance(source_sharding, jax.sharding.NamedSharding):
        return value.reshape(out_shape)

    source_parts = list(tuple(source_sharding.spec))
    while len(source_parts) < len(shape):
        source_parts.append(None)
    target_spec = jax.sharding.PartitionSpec(None, *source_parts[: len(shape)])
    mismatches = _spec_axis_shape_mismatches(target_spec, source_sharding.mesh, out_shape)
    if mismatches:
        raise ValueError(
            "SpectraX cannot preserve sharding while splitting a batch into microbatches. "
            f"context={context}, input_shape={shape}, output_shape={out_shape}, "
            f"source_axes={getattr(source_sharding.mesh, 'axis_names', None)}, "
            f"source_spec={source_sharding.spec}, target_spec={target_spec}, invalid_dims={mismatches}. "
            "Change the batch size, microbatch count, or sharding policy so the "
            "per-microbatch batch dimension is divisible by its mesh-axis product."
        )

    target_sharding = jax.sharding.NamedSharding(source_sharding.mesh, target_spec)
    target_sharding = _named_sharding_with_memory_kind(target_sharding, getattr(source_sharding, "memory_kind", None))
    return _reshape_with_named_shardings(
        value,
        out_shape,
        in_sharding=source_sharding,
        out_sharding=target_sharding,
    )


def _flatten_microbatch_stack(value: object, m: int, *, context: str) -> object:
    """Flatten ``(microbatch, batch, ...)`` while preserving the current layout.

    Forward-only MPMD auxiliary calls return full-batch tensors by collapsing the
    schedule microbatch axis back into the logical batch axis.  The reshape must
    carry over the value's existing sharding; otherwise XLA may choose a weaker
    layout such as TP-only and materialize a huge per-device buffer.
    """
    if not hasattr(value, "shape") or getattr(value, "ndim", len(getattr(value, "shape", ()))) < 2:
        return value
    shape = tuple(int(dim) for dim in value.shape)
    if shape[0] != int(m):
        return value

    out_shape = (shape[0] * shape[1], *shape[2:])
    source_sharding = getattr(value, "sharding", None)
    if not isinstance(source_sharding, jax.sharding.NamedSharding):
        return value.reshape(out_shape)

    source_spec = tuple(source_sharding.spec)
    source_parts = list(source_spec)
    while len(source_parts) < len(shape):
        source_parts.append(None)

    merged_axis = _merged_flattened_partition_axis(source_parts[0], source_parts[1])
    target_spec = jax.sharding.PartitionSpec(merged_axis, *source_parts[2 : 1 + len(out_shape)])
    mismatches = _spec_axis_shape_mismatches(target_spec, source_sharding.mesh, out_shape)
    if mismatches:
        raise ValueError(
            "SpectraX cannot preserve sharding while flattening a microbatch stack. "
            f"context={context}, input_shape={shape}, output_shape={out_shape}, "
            f"source_axes={getattr(source_sharding.mesh, 'axis_names', None)}, "
            f"source_spec={source_sharding.spec}, target_spec={target_spec}, invalid_dims={mismatches}."
        )

    target_sharding = jax.sharding.NamedSharding(source_sharding.mesh, target_spec)
    target_sharding = _named_sharding_with_memory_kind(target_sharding, getattr(source_sharding, "memory_kind", None))
    return _reshape_with_named_shardings(
        value,
        out_shape,
        in_sharding=source_sharding,
        out_sharding=target_sharding,
    )
