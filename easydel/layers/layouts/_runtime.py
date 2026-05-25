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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime helpers for activations of fused / TP-interleaved projections.

This module owns the pure-array (JAX-side) helpers used by
:class:`FusedColumnLayout` and the public ``split_fused_*`` builders to
unpack TP-interleaved activations at runtime, and to constrain newly-split
slices to the active tensor-parallel mesh axis.

The "interleaved" packing layout assumed throughout this module is the
one produced by :func:`torch_interleave_segments_for_tp` at checkpoint
load time: for segment sizes ``(A, B, C)`` and TP size ``T``, the fused
last axis is ordered ``[A_0, B_0, C_0, A_1, B_1, C_1, ..., A_{T-1},
B_{T-1}, C_{T-1}]``. Splitting it back to logical
``(A_0..A_{T-1}, B_0..B_{T-1}, C_0..C_{T-1})`` requires the TP size,
which is resolved from the model config.
"""

from __future__ import annotations

import typing as tp

import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import PartitionAxis, with_sharding_constraint

from easydel.infra.sharding import OptionalMesh, mesh_axis_size, resolve_stage_mesh

from ._types import Array, EasyDeLBaseConfig


def _partition_axis(config: EasyDeLBaseConfig | None = None) -> PartitionAxis:
    """Return the real :class:`PartitionAxis` from a config, with a fallback.

    Args:
        config: Owning model config; ``None`` triggers a default
            :class:`PartitionAxis()` instance.

    Returns:
        Resolved :class:`PartitionAxis` for sharding-axis lookups.
    """
    if config is not None and config.partition_axis is not None:
        return config.partition_axis
    return PartitionAxis()


def _mesh(config: EasyDeLBaseConfig | None = None) -> OptionalMesh:
    """Return the active mesh from a config, with a fallback.

    Args:
        config: Owning model config; ``None`` returns ``None`` so callers
            fall back to the ambient JAX mesh.

    Returns:
        The mesh attached to ``config`` or ``None`` when no config was
        supplied.
    """
    if config is not None:
        return config.mesh
    return None


def tensor_parallel_axis(config: EasyDeLBaseConfig | None = None) -> str | None:
    """Return the tensor-parallel mesh axis configured on ``config``.

    Args:
        config: Owning model config; ``None`` uses the default
            :class:`PartitionAxis`.

    Returns:
        Name of the tensor-parallel mesh axis or ``None`` when none is
        configured.
    """
    return _partition_axis(config).tensor_parallel_axis


def tensor_parallel_size(
    config: EasyDeLBaseConfig | None = None,
    *,
    tp_axis: str | None = None,
    arr: Array | None = None,
) -> int:
    """Resolve the active tensor-parallel size for a model / config.

    Args:
        config: Owning model config; the source of the TP axis name and
            mesh when ``tp_axis`` / ``arr`` do not pin them down.
        tp_axis: Optional explicit TP axis name override.
        arr: Optional array whose sharding is consulted to disambiguate
            the stage mesh (relevant inside MPMD pipeline stages).

    Returns:
        Active TP size; ``1`` when no TP axis is configured or no mesh
        is available.
    """
    tp_axis = tp_axis if tp_axis is not None else tensor_parallel_axis(config)
    if tp_axis is None:
        return 1

    stage_mesh = resolve_stage_mesh(_mesh(config), arr=arr)
    if stage_mesh is None:
        return 1
    return int(mesh_axis_size(stage_mesh, tp_axis))


def with_tp_last_axis_sharding(
    x: Array,
    config: EasyDeLBaseConfig | None = None,
    *,
    tp_axis: str | None = None,
) -> Array:
    """Constrain ``x`` so that its last axis is sharded on the TP mesh axis.

    No-op fast paths: returns ``x`` unchanged when no TP axis is
    configured, no mesh is available, the TP size is ``<=1``, ``x`` is
    a 0-D array, or the last axis is not divisible by the TP size.

    Args:
        x: Array to constrain.
        config: Owning model config used to resolve the TP axis and mesh.
        tp_axis: Optional explicit TP axis override.

    Returns:
        ``x`` with a :class:`PartitionSpec` constraint applied
        (when applicable), otherwise ``x`` unchanged.
    """
    tp_axis = tp_axis if tp_axis is not None else tensor_parallel_axis(config)
    if tp_axis is None:
        return x

    stage_mesh = resolve_stage_mesh(_mesh(config), arr=x)
    if stage_mesh is None:
        return x

    tp_size = int(mesh_axis_size(stage_mesh, tp_axis))
    if tp_size <= 1 or x.ndim < 1 or int(x.shape[-1]) % tp_size != 0:
        return x

    spec = PartitionSpec(*([None] * (x.ndim - 1)), tp_axis)
    return tp.cast(Array, with_sharding_constraint(x, spec, mesh=stage_mesh))


def normalize_segment_sizes(segment_sizes: tp.Sequence[int]) -> tuple[int, ...]:
    """Coerce a sequence of segment sizes into a canonical ``tuple[int, ...]``.

    Args:
        segment_sizes: Iterable of segment widths (potentially ``np.int64``
            or other numeric subtypes that need to be normalized to plain
            ``int``).

    Returns:
        ``tuple`` of Python ``int`` values in the original order.
    """
    return tuple(int(size) for size in segment_sizes)


def split_interleaved_segments_last_axis(
    x: Array,
    segment_sizes: tp.Sequence[int],
    *,
    tp_size: int | None = None,
    config: EasyDeLBaseConfig | None = None,
    apply_sharding: bool = True,
) -> tuple[Array, ...] | None:
    """Split ``x`` from TP-interleaved segment order along its last axis.

    For segment sizes ``(A, B, C)`` and ``tp_size=2`` the assumed
    last-axis layout is ``[A_0, B_0, C_0, A_1, B_1, C_1]``; this function
    returns the three tensors that concatenate the per-rank chunks back
    together: ``concat([A_0, A_1])``, ``concat([B_0, B_1])``,
    ``concat([C_0, C_1])`` — each in its logical segment-axis order.

    Returns ``None`` (the caller is expected to fall back to a contiguous
    split) when the operation cannot be performed safely, namely when
    ``tp_size <= 1``, ``x`` is 0-D, ``segment_sizes`` is empty, any
    segment is non-positive or not divisible by ``tp_size``, or the last
    axis does not match ``sum(segment_sizes)``.

    Args:
        x: Activation of shape ``[..., sum(segment_sizes)]``.
        segment_sizes: Logical (un-sharded) per-segment widths.
        tp_size: Optional explicit TP size; resolved from ``config`` when
            ``None``.
        config: Owning model config used to resolve the TP size.
        apply_sharding: When ``True`` re-applies the TP last-axis sharding
            constraint to each output.

    Returns:
        Tuple of per-segment arrays in original order, or ``None`` when
        a TP-aware split is not applicable.
    """
    segment_sizes = normalize_segment_sizes(segment_sizes)
    if tp_size is None:
        tp_size = tensor_parallel_size(config, arr=x)
    if tp_size <= 1 or x.ndim < 1 or not segment_sizes:
        return None
    if any(size <= 0 or size % tp_size != 0 for size in segment_sizes):
        return None
    if int(x.shape[-1]) != sum(segment_sizes):
        return None

    local_sizes = tuple(size // tp_size for size in segment_sizes)
    local_total = sum(local_sizes)
    interleaved = jnp.reshape(x, (*x.shape[:-1], tp_size, local_total))
    offsets = tuple(np.cumsum(local_sizes)[:-1].tolist())
    local_parts = jnp.split(interleaved, offsets, axis=-1)
    outputs = tuple(
        jnp.reshape(part, (*x.shape[:-1], size)) for part, size in zip(local_parts, segment_sizes, strict=True)
    )
    if apply_sharding:
        return tuple(with_tp_last_axis_sharding(part, config) for part in outputs)
    return outputs


def split_interleaved_pair_last_axis(
    x: Array,
    *,
    tp_size: int | None = None,
    config: EasyDeLBaseConfig | None = None,
    apply_sharding: bool = True,
) -> tuple[Array, Array]:
    """Split an equal-pair TP-interleaved fused projection (e.g. gate/up).

    Convenience wrapper around :func:`split_interleaved_segments_last_axis`
    for the two-equal-halves case. Always returns a 2-tuple, falling back
    to a plain ``jnp.split(x, 2, axis=-1)`` when TP packing does not apply
    (no TP mesh, odd last-axis size, or other ineligibility conditions).

    Args:
        x: Activation of shape ``[..., 2 * half]``.
        tp_size: Optional explicit TP size override.
        config: Owning model config used to resolve the TP size.
        apply_sharding: Whether to re-apply the TP last-axis sharding
            constraint to each output half.

    Returns:
        Tuple ``(first_half, second_half)`` each of shape ``[..., half]``.
    """
    fallback = tuple(jnp.split(x, 2, axis=-1))
    if int(x.shape[-1]) % 2 != 0:
        return tp.cast(tuple[Array, Array], fallback)

    half = int(x.shape[-1]) // 2
    parts = split_interleaved_segments_last_axis(
        x,
        (half, half),
        tp_size=tp_size,
        config=config,
        apply_sharding=apply_sharding,
    )
    if parts is None:
        return tp.cast(tuple[Array, Array], fallback)
    return tp.cast(tuple[Array, Array], parts)


def keep_interleaved_segments_last_axis(
    x: Array,
    segment_sizes: tp.Sequence[int],
    keep: tp.Sequence[int],
    *,
    tp_size: int | None = None,
    config: EasyDeLBaseConfig | None = None,
    apply_sharding: bool = True,
) -> Array | None:
    """Return a subset of segments from a TP-interleaved packed tensor.

    Splits ``x`` via :func:`split_interleaved_segments_last_axis`, then
    concatenates only the segments listed in ``keep``. Useful for paths
    that consume only a subset of a fused projection's outputs (e.g. an
    encoder that ignores Q from a fused QKV when only K/V are needed).

    Args:
        x: Activation of shape ``[..., sum(segment_sizes)]``.
        segment_sizes: Logical per-segment widths in the packed order.
        keep: Indices of segments to retain (preserving the supplied order).
        tp_size: Optional explicit TP size override.
        config: Owning model config used to resolve the TP size.
        apply_sharding: When ``True`` re-applies the TP last-axis sharding
            constraint to the concatenated output.

    Returns:
        Concatenated kept-segments array, or ``None`` when the TP-aware
        split is not applicable (callers can then fall back to a
        contiguous split + selection).
    """
    parts = split_interleaved_segments_last_axis(
        x,
        segment_sizes,
        tp_size=tp_size,
        config=config,
        apply_sharding=False,
    )
    if parts is None:
        return None
    kept = jnp.concatenate([parts[int(idx)] for idx in keep], axis=-1)
    if apply_sharding:
        return with_tp_last_axis_sharding(kept, config)
    return kept


def split_contiguous_segments_last_axis(x: Array, segment_sizes: tp.Sequence[int]) -> tuple[Array, ...]:
    """Plain (non-TP-aware) split of a packed tensor along its last axis.

    Mirrors the semantics of ``jnp.split`` (and ``np.split`` for numpy
    arrays) but takes per-segment sizes directly rather than the
    cumulative-offset form expected by ``split``. Used as the fallback
    when the TP-aware splitters decide they cannot proceed.

    Args:
        x: Tensor of shape ``[..., sum(segment_sizes)]``.
        segment_sizes: Per-segment widths in concatenation order.

    Returns:
        Tuple of per-segment slices in original order.
    """
    segment_sizes = normalize_segment_sizes(segment_sizes)
    if isinstance(x, np.ndarray):
        return tuple(np.split(x, np.cumsum(segment_sizes)[:-1], axis=-1))
    return tuple(jnp.split(x, np.cumsum(segment_sizes)[:-1], axis=-1))
