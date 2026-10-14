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

"""Torch-side helpers for TP-interleaved packing of checkpoint tensors.

These helpers are invoked by the checkpoint loader / exporter — *not* by
the JAX runtime — to fuse separate per-purpose Torch tensors (Q/K/V,
gate/up) into one TP-interleaved tensor (and back). They are written
torch-naively (the torch module is passed in by the caller) so that this
package keeps its hard dependency on torch optional.

Interleaving contract: for segment sizes ``(A, B, C)`` and TP size ``T``
the fused output along ``dim`` is ordered
``[A_0, B_0, C_0, A_1, B_1, C_1, ..., A_{T-1}, B_{T-1}, C_{T-1}]`` where
``X_i`` denotes the ``i``-th equal chunk of segment ``X`` split along
``dim``. The matching de-interleave functions undo exactly this layout.
"""

from __future__ import annotations

import typing as tp

import numpy as np

from ._runtime import normalize_segment_sizes


def torch_interleave_segments_for_tp(
    torch: tp.Any,
    segments: tp.Sequence[tp.Any],
    *,
    tp_size: int,
    dim: int = 0,
) -> tp.Any:
    """Pack separate torch tensors into a TP-interleaved fused tensor.

    Each segment is chunked into ``tp_size`` equal slices along ``dim``,
    and the slices are concatenated in rank-major order:
    ``[seg0_rank0, seg1_rank0, ..., seg0_rank1, seg1_rank1, ...]``.

    When ``tp_size <= 1`` or any segment is not divisible by ``tp_size``,
    falls back to a plain ``torch.cat(segments, dim=dim)`` so the result
    remains usable (just without the interleaved layout).

    Args:
        torch: Torch module reference (avoids a hard dependency in this
            package).
        segments: Per-purpose source tensors to fuse.
        tp_size: Active tensor-parallel size.
        dim: Axis along which to concatenate. Defaults to ``0``.

    Returns:
        Single torch tensor with the segments interleaved on ``dim``.
    """
    segments = tuple(segments)
    if tp_size <= 1 or not segments:
        return torch.cat(segments, dim=dim)
    if any(int(segment.shape[dim]) % tp_size != 0 for segment in segments):
        return torch.cat(segments, dim=dim)

    chunks_by_segment = [segment.chunk(tp_size, dim=dim) for segment in segments]
    chunks = [segment_chunks[rank_idx] for rank_idx in range(tp_size) for segment_chunks in chunks_by_segment]
    return torch.cat(chunks, dim=dim).contiguous()


def torch_deinterleave_segments_for_tp(
    torch: tp.Any,
    tensor: tp.Any,
    segment_sizes: tp.Sequence[int],
    *,
    tp_size: int,
    dim: int = 0,
) -> tuple[tp.Any, ...]:
    """Split a TP-interleaved torch tensor back into contiguous segments.

    Inverse of :func:`torch_interleave_segments_for_tp`. Each per-segment
    output is a contiguous tensor whose ``dim`` axis is the concatenation
    of the per-rank slices in rank order.

    Falls back to a plain ``torch.split`` when ``tp_size <= 1``, any
    segment is not divisible by ``tp_size``, ``dim`` is out of range, or
    the source ``dim`` size does not match ``sum(segment_sizes)``.

    Args:
        torch: Torch module reference.
        tensor: TP-interleaved fused source tensor.
        segment_sizes: Logical (un-sharded) per-segment widths.
        tp_size: Active tensor-parallel size.
        dim: Axis along which the tensor was interleaved. Defaults to ``0``.

    Returns:
        Tuple of per-segment torch tensors in the original packing order.
    """
    segment_sizes = normalize_segment_sizes(segment_sizes)
    if tp_size <= 1 or not segment_sizes:
        return tuple(torch.split(tensor, segment_sizes, dim=dim))
    if any(size <= 0 or size % tp_size != 0 for size in segment_sizes):
        return tuple(torch.split(tensor, segment_sizes, dim=dim))
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim or int(tensor.shape[dim]) != sum(segment_sizes):
        return tuple(torch.split(tensor, segment_sizes, dim=dim))

    local_sizes = tuple(size // tp_size for size in segment_sizes)
    local_total = sum(local_sizes)
    shape = tuple(tensor.shape)
    view_shape = (*shape[:dim], tp_size, local_total, *shape[dim + 1 :])
    packed = tensor.reshape(*view_shape)
    tuple(np.cumsum(local_sizes)[:-1].tolist())
    local_parts = torch.split(packed, local_sizes, dim=dim + 1)
    outputs = []
    for part, size in zip(local_parts, segment_sizes, strict=True):
        out_shape = (*shape[:dim], size, *shape[dim + 1 :])
        outputs.append(part.reshape(*out_shape).contiguous())
    return tuple(outputs)


def torch_interleave_axis_segments_for_tp(
    torch: tp.Any,
    tensor: tp.Any,
    segment_sizes: tp.Sequence[int],
    *,
    tp_size: int,
    dim: int,
) -> tp.Any:
    """Re-interleave segments inside one already-concatenated torch tensor.

    Differs from :func:`torch_interleave_segments_for_tp` in that the
    segments are *already* concatenated into ``tensor`` (not supplied
    separately). The tensor is first split into per-segment views and
    then re-stitched in rank-interleaved order so that the result matches
    the layout produced by :func:`torch_interleave_segments_for_tp`.

    Returns ``tensor`` unchanged when any of the usual ineligibility
    conditions hold (``tp_size <= 1``, indivisible segments, mismatched
    ``dim`` size).

    Args:
        torch: Torch module reference.
        tensor: Source tensor containing the segments concatenated along
            ``dim``.
        segment_sizes: Per-segment widths along ``dim``.
        tp_size: Active tensor-parallel size.
        dim: Axis along which to re-interleave.

    Returns:
        Tensor with the same shape but with TP-interleaved segment order
        along ``dim``.
    """
    segment_sizes = normalize_segment_sizes(segment_sizes)
    if tp_size <= 1 or not segment_sizes or any(size % tp_size != 0 for size in segment_sizes):
        return tensor
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim or int(tensor.shape[dim]) != sum(segment_sizes):
        return tensor

    parts = torch.split(tensor, segment_sizes, dim=dim)
    chunks = []
    for rank_idx in range(tp_size):
        for part, size in zip(parts, segment_sizes, strict=True):
            local_size = size // tp_size
            chunks.append(part.narrow(dim, rank_idx * local_size, local_size))
    return torch.cat(chunks, dim=dim).contiguous()


def torch_deinterleave_axis_segments_for_tp(
    torch: tp.Any,
    tensor: tp.Any,
    segment_sizes: tp.Sequence[int],
    *,
    tp_size: int,
    dim: int,
) -> tp.Any:
    """Inverse of :func:`torch_interleave_axis_segments_for_tp` for one tensor.

    De-interleaves ``tensor`` into per-segment views and re-concatenates
    them in contiguous (logical) segment order along ``dim`` so the
    result matches the original pre-interleaving layout.

    Args:
        torch: Torch module reference.
        tensor: Source tensor with TP-interleaved segment order along
            ``dim``.
        segment_sizes: Per-segment widths along ``dim``.
        tp_size: Active tensor-parallel size.
        dim: Axis along which the segments are interleaved.

    Returns:
        Tensor of the same shape with segments restored to contiguous
        per-segment order along ``dim``.
    """
    parts = torch_deinterleave_segments_for_tp(
        torch,
        tensor,
        segment_sizes,
        tp_size=tp_size,
        dim=dim,
    )
    return torch.cat(parts, dim=dim).contiguous()
