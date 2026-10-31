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

"""Helpers for aligning splash/mask metadata with QKV ``PartitionSpec`` axes.

The block-sparse and splash attention kernels accept a ``mask_info`` object
that carries names of the mesh axes used to shard the attention tensors. This
module provides a small utility to populate those axis names from the actual
``PartitionSpec`` of the query/key tensors so that the kernel sees a
self-consistent picture of how the tensors are distributed.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array


def topk_selection_mask(scores: Array, values: Array, indices: Array) -> Array:
    """Convert a ``lax.top_k`` result to its boolean selection mask without scatter.

    Args:
        scores: Ranked scores ``[..., n]``.
        values: Descending selected scores ``[..., k]``, with ``k >= 1``.
        indices: Selected indices of the same shape as ``values``.

    Returns:
        Boolean mask ``[..., n]`` matching the indices, including IEEE total-order ties.
    """

    def order_key(x):
        bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
        return jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)

    keys = order_key(scores)
    value_keys = order_key(values)
    kth = value_keys[..., -1:]
    tie_cut = jnp.max(jnp.where(value_keys == kth, indices, -1), axis=-1, keepdims=True)
    position = jax.lax.broadcasted_iota(jnp.int32, scores.shape, scores.ndim - 1)
    return (keys > kth) | ((keys == kth) & (position <= tie_cut))


def _spec_axis(spec: tp.Any, index: int) -> tp.Any:
    """Safely look up an axis name in a ``PartitionSpec``-like sequence.

    Args:
        spec: A ``PartitionSpec`` or any indexable that follows the same
            convention. May be ``None``.
        index: Position to read from ``spec``.

    Returns:
        Any: The axis label at ``index`` (or ``None`` when the entry is empty,
        the index is out of range, or ``spec`` is ``None``). Empty tuples
        ``()`` are normalized to ``None``.
    """
    if spec is None:
        return None
    try:
        if index >= len(spec):
            return None
        axis = spec[index]
    except Exception:
        return None
    return None if axis == () else axis


def align_mask_info_to_qkv_specs(
    mask_info: tp.Any,
    *,
    query_spec: tp.Any,
    key_spec: tp.Any | None = None,
    layout: tp.Literal["bthd", "bhtd"] = "bthd",
) -> tp.Any:
    """Stamp QKV sharding axis names onto a ``mask_info`` object.

    Reads the batch/head/sequence axis names from ``query_spec`` (and
    ``key_spec`` for KV heads) and calls ``mask_info.replace(...)`` to produce
    a copy whose axis-name fields agree with the actual sharding of the
    attention tensors.

    Args:
        mask_info: An attrs/dataclass-like object that exposes ``replace``,
            typically a splash-attention ``MaskInfo``. ``None`` is passed
            through unchanged.
        query_spec: The ``PartitionSpec`` (or compatible) used to shard the
            query tensor.
        key_spec: ``PartitionSpec`` for the key tensor. Defaults to
            ``query_spec`` when ``None``.
        layout: Tensor layout. ``"bthd"`` => ``(batch, time, heads, dim)``
            (sequence at index 1, heads at index 2); ``"bhtd"`` =>
            ``(batch, heads, time, dim)`` (sequence at index 2, heads at
            index 1).

    Returns:
        Any: A new ``mask_info`` with ``batch_axis_name``,
        ``qheads_axis_name``, ``kvheads_axis_name`` and
        ``sequence_axis_name`` populated. Returns the original object when it
        is ``None`` or has no callable ``replace``.
    """
    if mask_info is None:
        return None

    if layout == "bhtd":
        sequence_index = 2
        head_index = 1
    else:
        sequence_index = 1
        head_index = 2

    replace = getattr(mask_info, "replace", None)
    if not callable(replace):
        return mask_info

    key_spec = query_spec if key_spec is None else key_spec
    return replace(
        batch_axis_name=_spec_axis(query_spec, 0),
        qheads_axis_name=_spec_axis(query_spec, head_index),
        kvheads_axis_name=_spec_axis(key_spec, head_index),
        sequence_axis_name=_spec_axis(query_spec, sequence_index),
    )
