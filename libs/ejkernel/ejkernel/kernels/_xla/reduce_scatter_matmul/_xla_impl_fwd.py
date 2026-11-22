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

"""XLA reduce-scatter matmul: forward implementation and custom VJP.

Computes ``reduce_scatter(x @ y.T, scatter_dim=0)`` via a local dot product
followed by ``lax.psum_scatter``.  A custom VJP reconstructs the full gradient
via ``all_gather`` then ``dot``.

Public entry point: ``reduce_scatter_matmul``.
Internal pieces:
    - ``_validate_inputs``: static shape/dtype checks.
    - ``_forward_impl``: the raw dot + psum_scatter computation.
    - ``_reduce_scatter_matmul_core``: ``custom_vjp`` wrapper.
    - ``_reduce_scatter_matmul_core_fwd`` / ``_reduce_scatter_matmul_core_bwd``:
      VJP rules registered on the core function.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ._xla_impl_bwd import reduce_scatter_matmul_backward


def _validate_inputs(x: jax.Array, y: jax.Array, *, tp_size: int | None) -> None:
    """Validate shapes, dtypes, and optional tp_size for reduce-scatter matmul.

    Args:
        x: Left operand (must be rank-2).
        y: Right operand (must be rank-2, same dtype as ``x``, with
            ``x.shape[1] == y.shape[1]``).
        tp_size: Optional tensor-parallel degree; must be >= 1 when provided.

    Raises:
        ValueError: On any shape, dtype, or parameter mismatch.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be rank-2 tensors, got {x.ndim=} and {y.ndim=}.")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must share dtype, got {x.dtype=} and {y.dtype=}.")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "Incompatible matmul shapes for reduce_scatter_matmul: "
            f"x.shape={x.shape}, y.shape={y.shape}, expected x.shape[1] == y.shape[1]."
        )
    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1 when provided, got {tp_size}.")


def _forward_impl(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    *,
    axis_name: str,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m_local n"]:
    partial_out = jnp.dot(x, y.T, precision=precision)
    return lax.psum_scatter(partial_out, axis_name=axis_name, scatter_dimension=0, tiled=True)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _reduce_scatter_matmul_core(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m_local n"]:
    return _forward_impl(x, y, axis_name=axis_name, precision=precision)


def _reduce_scatter_matmul_core_fwd(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    precision: jax.lax.PrecisionLike,
):
    out = _forward_impl(x, y, axis_name=axis_name, precision=precision)
    return out, (x, y)


def _reduce_scatter_matmul_core_bwd(
    axis_name: str,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m_local n"],
):
    x, y = residual
    grad_x, grad_y = reduce_scatter_matmul_backward(
        dy,
        x,
        y,
        axis_name=axis_name,
        precision=precision,
    )
    return grad_x, grad_y


_reduce_scatter_matmul_core.defvjp(_reduce_scatter_matmul_core_fwd, _reduce_scatter_matmul_core_bwd)


def reduce_scatter_matmul(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    *,
    axis_name: str,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m_local n"]:
    """Compute ``reduce_scatter(x @ y.T, scatter_dim=0)`` with an explicit custom VJP.

    Validates inputs, then delegates to ``_reduce_scatter_matmul_core`` which
    carries the custom backward rule.

    Note:
        ``bm``, ``bn``, ``bk``, and ``collective_id`` are accepted for API
        compatibility with hardware-specific backends but are silently ignored.

    Args:
        x: Left operand, shape ``[m, k_shard]``.
        y: Right operand, shape ``[n, k_shard]``.  Must share dtype with ``x``.
        axis_name: JAX device-mesh axis name for the ``psum_scatter``.
        bm: Ignored.
        bn: Ignored.
        bk: Ignored.
        tp_size: Optional tensor-parallel size for validation (must be >= 1).
        collective_id: Ignored.
        precision: Matmul precision.

    Returns:
        Scattered output shard, shape ``[m_local, n]``.
    """
    del bm, bn, bk, collective_id
    _validate_inputs(x, y, tp_size=tp_size)
    return _reduce_scatter_matmul_core(x, y, axis_name, precision)


__all__ = ("reduce_scatter_matmul",)
