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

"""Pallas TPU reduce-scatter matmul interface with explicit backward kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import reduce_scatter_matmul as _reduce_scatter_matmul_impl


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _reduce_scatter_matmul_core(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m_local n"]:
    """Differentiable core of reduce-scatter matmul with custom VJP.

    Computes ``x @ y.T`` distributed across devices: each device holds a
    K-shard of both ``x`` and ``y`` (along their last dimension), and the
    result is reduced and scattered so that device ``i`` receives rows
    ``[i * (m // tp_size), (i+1) * (m // tp_size))`` of the full product.

    The ``precision`` argument is forwarded to the backward pass only; the
    forward pass delegates entirely to the Pallas implementation which
    selects its own arithmetic precision.

    Args:
        x: Input matrix shard [m, k_shard].
        y: Weight matrix shard [n, k_shard].
        axis_name: JAX collective axis name for inter-device communication.
        bm: Block size along the M dimension for tiled matmul.
        bn: Block size along the N dimension for tiled matmul.
        bk: Block size along the K dimension for tiled matmul.
        tp_size: Tensor-parallel degree. ``None`` infers it from the axis size.
        collective_id: Identifier for the collective operation (used by the
            Pallas kernel for pipeline scheduling). ``None`` disables it.
        precision: Arithmetic precision for the backward gradient computation.

    Returns:
        Local output shard [m_local, n] where ``m_local = m // tp_size``.
    """
    del precision
    return _reduce_scatter_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        tp_size=tp_size,
        collective_id=collective_id,
        bm=bm,
        bn=bn,
        bk=bk,
    )


def _reduce_scatter_matmul_core_fwd(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
):
    """Custom VJP forward rule: run the Pallas kernel and save x, y as residuals.

    Saves ``(x, y)`` for use in the backward pass so that gradients can be
    computed with a standard dense matmul. The ``precision`` argument is
    discarded here (the Pallas kernel manages its own precision).

    Args:
        x: Input matrix shard [m, k_shard].
        y: Weight matrix shard [n, k_shard].
        axis_name: JAX collective axis name.
        bm: Block size along M for tiled matmul.
        bn: Block size along N for tiled matmul.
        bk: Block size along K for tiled matmul.
        tp_size: Tensor-parallel degree (``None`` infers from axis size).
        collective_id: Collective pipeline identifier.
        precision: Arithmetic precision (forwarded to backward only).

    Returns:
        Tuple of (output [m_local, n], residuals (x, y)).
    """
    del precision
    out = _reduce_scatter_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        tp_size=tp_size,
        collective_id=collective_id,
        bm=bm,
        bn=bn,
        bk=bk,
    )
    return out, (x, y)


def _reduce_scatter_matmul_core_bwd(
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m_local n"],
):
    """Custom VJP backward rule: reconstruct full dy via all-gather and compute grads.

    The forward pass scatters the M dimension across devices, so the upstream
    gradient ``dy`` has shape [m_local, n]. This function all-gathers ``dy``
    along the M axis to reconstruct the full [m, n] gradient, then computes
    dense matmul gradients for ``x`` and ``y``.

    Args:
        axis_name: JAX collective axis name used for ``lax.all_gather``.
        bm: Unused block-size parameter (kept for VJP signature compatibility).
        bn: Unused block-size parameter.
        bk: Unused block-size parameter.
        tp_size: Unused tensor-parallel degree.
        collective_id: Unused collective identifier.
        precision: Arithmetic precision for ``jnp.dot`` gradient computation.
        residual: Tuple ``(x, y)`` saved by the forward pass.
        dy: Upstream gradient w.r.t. output [m_local, n].

    Returns:
        Tuple ``(grad_x, grad_y)`` where:

        - **grad_x** - Gradient w.r.t. x [m, k_shard]: ``dy_full @ y``.
        - **grad_y** - Gradient w.r.t. y [n, k_shard]: ``dy_full.T @ x``.
    """
    del bm, bn, bk, tp_size, collective_id
    x, y = residual
    dy_full = lax.all_gather(dy, axis_name=axis_name, axis=0, tiled=True)
    grad_x = jnp.dot(dy_full, y, precision=precision)
    grad_y = jnp.dot(dy_full.T, x, precision=precision)
    return grad_x, grad_y


_reduce_scatter_matmul_core.defvjp(_reduce_scatter_matmul_core_fwd, _reduce_scatter_matmul_core_bwd)


@kernel_registry.register("reduce_scatter_matmul", Platform.PALLAS, Backend.TPU)
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
    """Bidirectional M-split reduce-scatter matmul for TPU with differentiable backward.

    Computes a distributed matrix product ``x @ y.T`` where both ``x`` and
    ``y`` hold a K-shard (last-dimension shard) of the full matrices, and the
    result is reduced and scattered so that device ``i`` receives a contiguous
    M-slice of the output.

    Internally, this wraps :func:`_reduce_scatter_matmul_core` which uses a
    JAX ``custom_vjp`` to provide an explicit backward pass: the forward pass
    runs the Pallas bidirectional M-split kernel, and the backward pass uses
    ``lax.all_gather`` followed by dense matmul to compute ``grad_x`` and
    ``grad_y``.

    For ``bfloat16``/``float16`` inputs the Pallas kernel falls back to
    ``lax.psum_scatter`` (no tiled pipeline).

    Args:
        x: Input activation shard [m, k_shard]. Each device holds a shard
            along the K dimension; the M dimension is not sharded on input.
        y: Weight matrix shard [n, k_shard]. Same K-sharding as ``x``.
        axis_name: JAX named axis for collective communication (e.g. ``"tp"``).
            Must be active (inside ``shard_map`` or ``pmap``) at call time.
        bm: Tile size along the M dimension. Default: ``128``.
        bn: Tile size along the N dimension. Default: ``128``.
        bk: Tile size along the K dimension. Default: ``128``.
        tp_size: Tensor-parallel degree (number of devices). ``None`` infers
            it from the collective axis size.
        collective_id: Pipeline collective identifier passed to the Pallas
            kernel for scheduling async collectives. Default: ``0``.
        precision: ``jax.lax.Precision`` used for backward gradient matmuls.
            The forward Pallas kernel manages its own precision internally.

    Returns:
        Local output shard [m_local, n] where ``m_local = m // tp_size``.
        Device ``i`` receives rows ``[i * m_local, (i+1) * m_local)``.
    """
    return _reduce_scatter_matmul_core(x, y, axis_name, bm, bn, bk, tp_size, collective_id, precision)


__all__ = ("reduce_scatter_matmul",)
