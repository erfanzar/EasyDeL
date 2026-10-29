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

"""TileLang reduce-scatter matmul — single-device fallback implementation.

This module provides a compatible API for the reduce-scatter-matmul collective
but does **not** implement a real multi-device ring.  The current implementation
performs a single-device dense matmul (``x @ y.T``) and returns the full result
without any scatter/reduce communication.

Limitations:
  * ``axis_name`` must be ``None`` or ``"__tp_dummy__"`` — real collective
    communication is not wired through TVM-FFI.
  * ``tp_size`` must be ``None`` or 1.
  * ``collective_id`` must be ``None`` or 0.
  * ``precision`` must be ``jax.lax.Precision.DEFAULT``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from .._dense_matmul import dense_matmul_tilelang


@kernel_registry.register("reduce_scatter_matmul", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def reduce_scatter_matmul(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m_local n"]:
    """Single-device matmul implementing the reduce-scatter-matmul interface.

    Registered as ``("reduce_scatter_matmul", Platform.TILELANG, Backend.GPU)``.

    Computes ``x @ y.T`` via :func:`~.._dense_matmul.dense_matmul_tilelang`,
    transposing ``y`` before the call.  Multi-device collective semantics
    (scatter/reduce) are **not** implemented.

    Args:
        x: ``[m, k_shard]`` float — left operand.
        y: ``[n, k_shard]`` float — right operand (transposed internally).
        axis_name: JAX parallel axis name for the collective.  Must be ``None``
            or ``"__tp_dummy__"`` — any other value raises an error.
        bm: Tile size in the ``m`` dimension (default 128); must be positive.
        bn: Tile size in the ``n`` dimension (default 128); must be positive.
        bk: Tile size in the ``k`` dimension (default 128); must be positive.
        tp_size: Tensor-parallelism degree.  Must be ``None`` or 1.
        collective_id: NCCL collective ID.  Must be ``None`` or 0.
        precision: JAX precision.  Must be ``jax.lax.Precision.DEFAULT``.

    Returns:
        ``[m, n]`` float — the full matmul result (no scatter applied).

    Raises:
        EjkernelRuntimeError: if any unsupported argument value is passed.
    """
    if bm <= 0 or bn <= 0 or bk <= 0:
        raise EjkernelRuntimeError("tile-lang reduce_scatter_matmul requires positive bm, bn and bk.")
    if collective_id not in (None, 0):
        raise EjkernelRuntimeError("tile-lang reduce_scatter_matmul does not support nonzero collective_id.")
    if precision != jax.lax.Precision.DEFAULT:
        raise EjkernelRuntimeError("tile-lang reduce_scatter_matmul does not yet support custom precision.")
    if tp_size not in (None, 1):
        raise EjkernelRuntimeError("tile-lang reduce_scatter_matmul v0 does not yet support tp_size > 1.")
    if axis_name not in (None, "__tp_dummy__"):
        raise EjkernelRuntimeError(
            "tile-lang reduce_scatter_matmul needs a native collective kernel for real axis_name."
        )

    partial = dense_matmul_tilelang(x, jnp.swapaxes(y, 0, 1).astype(x.dtype))
    return partial


__all__ = ["reduce_scatter_matmul"]
