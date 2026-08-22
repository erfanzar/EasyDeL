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
from ._pallas_impl_fwd import topk_superset_tpu


@kernel_registry.register("topk", Platform.PALLAS, Backend.TPU)
def topk(
    operand: Array,
    k: int | Array,
    axis: int = -1,
    mode: str = "values",
    mask_fill: float | None = None,
) -> tuple[Array, Array] | Array:
    """Blockwise-superset top-k on TPU; see the XLA reference for semantics.

    Only ``mode="values"`` with a static ``k`` is accelerated here. ``mask``
    and ``filter`` need a per-row dynamic ``k`` and a threshold rather than a
    sorted result, which the superset does not produce, so they delegate to the
    XLA reference instead of pretending to be fused.
    """
    from ejkernel.kernels._xla.topk._interface import topk as topk_xla

    if mode != "values" or not isinstance(k, int):
        return topk_xla(operand, k, axis=axis, mode=mode, mask_fill=mask_fill)

    axis_n = axis if axis >= 0 else operand.ndim + axis
    moved = operand if axis_n == operand.ndim - 1 else jnp.moveaxis(operand, axis_n, -1)
    lead = moved.shape[:-1]
    flat = moved.reshape(-1, moved.shape[-1])

    values, indices = topk_superset_tpu(flat, k=int(k))
    values = values.reshape(*lead, int(k))
    indices = indices.reshape(*lead, int(k))
    if axis_n != operand.ndim - 1:
        values = jnp.moveaxis(values, -1, axis_n)
        indices = jnp.moveaxis(indices, -1, axis_n)
    return values, indices
