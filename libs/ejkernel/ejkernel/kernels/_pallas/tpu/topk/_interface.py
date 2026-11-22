# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Pallas TPU registration for the fused top-k operation."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import topk_threshold_tpu
from ._pallas_select_fwd import SELECT_MODES, select_block_rows, topk_select_tpu


def _select(operand: Array, k: int | Array, axis: int, mode: str, fallback) -> Array:
    """Route ``"exact_mask"`` / ``"kth"`` to the sort-free selection kernel.

    Args:
        operand: Input array.
        k: Static or per-row ``k`` (broadcastable to ``operand`` minus ``axis``).
        axis: Reduction axis.
        mode: ``"exact_mask"`` or ``"kth"``.
        fallback: The XLA reference, used for dtypes or widths the kernel does not serve.

    Returns:
        The mode's result, identical to the XLA reference.

    Raises:
        ValueError: If a static ``k`` is outside ``[1, n]`` under ``mode="kth"``.
    """
    axis_n = axis if axis >= 0 else operand.ndim + axis
    width = operand.shape[axis_n]
    if (
        not jnp.issubdtype(operand.dtype, jnp.floating)
        or select_block_rows(width, operand.dtype) is None
        or operand.size == 0
    ):
        return fallback(operand, k, axis=axis, mode=mode)
    if mode == "kth" and isinstance(k, int) and not 1 <= k <= width:
        raise ValueError(f"mode='kth' needs 1 <= k <= {width}, got {k}.")
    moved = operand if axis_n == operand.ndim - 1 else jnp.moveaxis(operand, axis_n, -1)
    lead = moved.shape[:-1]
    ks = jnp.asarray(k, jnp.int32)
    if ks.ndim == moved.ndim:
        ks = ks[..., 0]
    ks = jnp.broadcast_to(ks, lead).reshape(-1)
    ks = jnp.clip(ks, 1 if mode == "kth" else 0, width)
    out = topk_select_tpu(moved.reshape(-1, width), ks, mode=mode)
    if mode == "kth":
        return out.reshape(lead)
    out = out.reshape(moved.shape)
    return out if axis_n == operand.ndim - 1 else jnp.moveaxis(out, -1, axis_n)


@kernel_registry.register("topk", Platform.PALLAS, Backend.TPU)
def topk(
    operand: Array,
    k: int | Array,
    axis: int = -1,
    mode: str = "values",
    mask_fill: float | None = None,
) -> tuple[Array, Array] | Array:
    """Threshold-and-compact top-k on TPU; see the XLA reference for semantics.

    ``mode="values"`` with a static ``k`` on floating inputs runs the
    threshold-and-compact kernel; ``"exact_mask"`` and ``"kth"`` (static or
    per-row ``k``) run the sort-free selection kernel on floats of at most 32
    bits. ``mask``, ``filter`` and integer inputs delegate to the XLA reference.
    """
    from ejkernel.kernels._xla.topk._interface import topk as topk_xla

    if mode in SELECT_MODES:
        return _select(operand, k, axis, mode, topk_xla)

    if mode != "values" or not isinstance(k, int) or not jnp.issubdtype(operand.dtype, jnp.floating):
        return topk_xla(operand, k, axis=axis, mode=mode, mask_fill=mask_fill)

    axis_n = axis if axis >= 0 else operand.ndim + axis
    moved = operand if axis_n == operand.ndim - 1 else jnp.moveaxis(operand, axis_n, -1)
    lead = moved.shape[:-1]
    flat = moved.reshape(-1, moved.shape[-1])

    values, indices = topk_threshold_tpu(flat, k=int(k))
    values = values.reshape(*lead, int(k))
    indices = indices.reshape(*lead, int(k))
    if axis_n != operand.ndim - 1:
        values = jnp.moveaxis(values, -1, axis_n)
        indices = jnp.moveaxis(indices, -1, axis_n)
    return values, indices
