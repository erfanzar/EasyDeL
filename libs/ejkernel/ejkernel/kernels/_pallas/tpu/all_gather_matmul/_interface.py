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

"""Public interface and custom VJP for TPU all-gather matmul.

Wraps the low-level Pallas kernel (``_pallas_impl.all_gather_matmul``) with:
  - Helper utilities for resolving tensor-parallel world size.
  - A ``jax.custom_vjp`` rule so that the backward pass uses a fused
    reduce-scatter matmul (imported from ``reduce_scatter_matmul``) instead
    of materialising the full gathered gradient.
  - Kernel-registry registration under ``Platform.PALLAS / Backend.TPU``.

Backward pass:
    ``grad_x`` is computed via
    ``reduce_scatter_matmul(dy, y_for_dx, axis_name=..., tp_size=tp)``.
    ``grad_y`` is computed via ``lax.all_gather(x) @ dy`` (or ``.T`` variant).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import all_gather_matmul as _all_gather_matmul_impl


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active mapped context when available."""
    try:
        return jax.core.concrete_or_error(
            int,
            lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size using explicit value, axis context, then global device count."""
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def _largest_divisor_leq(x: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)) -> int:
    x = int(max(1, x))
    for candidate in candidates:
        if candidate <= x and x % candidate == 0:
            return candidate
    return 1


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _all_gather_matmul_core(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m n_local"]:
    """Forward-only dispatch to the Pallas all-gather matmul kernel.

    Defined with ``jax.custom_vjp`` (nondiff args: axis_name, rhs_transpose,
    bn, bk, tp_size, collective_id, precision) so that the VJP rule can use
    the fused reduce-scatter backward.  The ``precision`` argument is accepted
    for interface compatibility but ignored by the forward kernel; it is used
    in the backward pass gradient for ``y``.

    Args:
        x: Local LHS shard of shape ``[m_local, k]``.
        y: Local RHS shard of shape ``[k, n_local]`` or ``[n_local, k]``
            (when ``rhs_transpose=True``).
        axis_name: pmap/shard_map axis name for the collective.
        rhs_transpose: Whether ``y`` is stored transposed.
        bn: N-dimension block size (None → use full ``n_per_device``).
        bk: K-dimension block size (None → use full ``k``).
        tp_size: Tensor-parallel world size (None → inferred).
        collective_id: Integer semaphore allocation ID for the DMA barrier.
        precision: JAX precision hint used in the backward gradient for ``y``;
            ignored in the forward kernel.

    Returns:
        ``all_gather(x) @ y`` of shape ``[m, n_local]`` where ``m = m_local * tp_size``.
    """
    del precision
    return _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
    )


def _all_gather_matmul_core_fwd(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
):
    """Custom VJP forward pass: compute output and save residuals ``(x, y)``."""
    del precision
    out = _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
    )
    return out, (x, y)


def _all_gather_matmul_core_bwd(
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m n_local"],
):
    """Custom VJP backward pass for all-gather matmul.

    Computes:
      - ``grad_x`` via fused reduce-scatter matmul:
        ``reduce_scatter(dy @ y.T, axis_name)`` (block sizes chosen
        heuristically from ``m_per_device`` and ``y`` shape).
      - ``grad_y`` via ``all_gather(x).T @ dy`` (or transposed variant for
        ``rhs_transpose=True``), using ``precision`` for the dot.

    The ``bn`` and ``bk`` arguments (from the forward non-diff args) are
    deliberately ignored in the backward; block sizes are recomputed.

    Args:
        axis_name: Collective axis name.
        rhs_transpose: Whether RHS was stored transposed.
        bn: Ignored in backward.
        bk: Ignored in backward.
        tp_size: Tensor-parallel world size; resolved via
            ``_resolve_tp_size`` if ``None``.
        collective_id: DMA semaphore allocation ID.
        precision: JAX precision hint for the ``grad_y`` dot product.
        residual: Tuple ``(x, y)`` saved by the forward pass.
        dy: Output gradient of shape ``[m, n_local]``.

    Returns:
        Tuple ``(grad_x, grad_y)`` matching the differentiable inputs
        ``(x, y)``; shapes ``[m_local, k]`` and ``[k, n_local]`` (or
        ``[n_local, k]`` when ``rhs_transpose=True``).
    """
    del bn, bk
    x, y = residual
    tp = _resolve_tp_size(tp_size, axis_name)

    if rhs_transpose:
        y_for_dx = y.T
    else:
        y_for_dx = y

    m_total = int(dy.shape[0])
    n_local = int(dy.shape[1])
    m_block = m_total // int(tp)
    m_half_block = max(1, m_block // 2)

    bm = _largest_divisor_leq(m_half_block)
    rs_bn = _largest_divisor_leq(int(y_for_dx.shape[0]))
    rs_bk = _largest_divisor_leq(n_local)

    from ..reduce_scatter_matmul._pallas_impl import reduce_scatter_matmul as _reduce_scatter_matmul_impl

    grad_x = _reduce_scatter_matmul_impl(
        dy,
        y_for_dx,
        axis_name=axis_name,
        tp_size=tp,
        collective_id=collective_id,
        bm=bm,
        bn=rs_bn,
        bk=rs_bk,
    )

    x_full = lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
    if rhs_transpose:
        grad_y = jnp.dot(dy.T, x_full, precision=precision)
    else:
        grad_y = jnp.dot(x_full.T, dy, precision=precision)

    return grad_x, grad_y


_all_gather_matmul_core.defvjp(_all_gather_matmul_core_fwd, _all_gather_matmul_core_bwd)


@kernel_registry.register("all_gather_matmul", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def all_gather_matmul(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool = False,
    bn: int | None = None,
    bk: int | None = None,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n_local"]:
    """Bidirectional ring all-gather fused with matmul on TPU.

    Gathers the LHS shard ``x`` across all devices on ``axis_name`` using a
    bidirectional ring protocol that overlaps DMA transfers with MXU
    computation, then computes the full ``all_gather(x) @ y`` result without
    materialising the gathered LHS in HBM.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  The gradient uses a fused reduce-scatter
    matmul for ``grad_x`` and a standard gathered dot for ``grad_y``.

    Args:
        x: Local LHS shard of shape ``[m_local, k]``.  Each device holds a
            contiguous ``m_local = m / tp_size`` row slice.  ``m_local`` must
            be divisible by 2 and ``m_local // 2`` must be divisible by 8.
        y: Local RHS shard.  Shape ``[k, n_local]`` (default) or
            ``[n_local, k]`` when ``rhs_transpose=True``.  ``k`` must be
            divisible by 128; ``n = n_local * tp_size`` must be divisible by
            128.
        axis_name: pmap / shard_map axis name used for the ring collective.
        rhs_transpose: If ``True``, ``y`` is stored as ``[n_local, k]``
            (i.e. the K dimension is axis 1 instead of axis 0).
        bn: Block size in the N dimension.  Defaults to the full
            ``n_per_device`` when ``None``.
        bk: Block size in the K dimension.  Defaults to the full ``k`` when
            ``None``.
        tp_size: Tensor-parallel world size.  Inferred from the active
            ``axis_name`` context (``lax.psum``) or ``jax.device_count()``
            when ``None``.
        collective_id: Integer ID for TPU DMA barrier-semaphore allocation.
            Defaults to 0.  Use distinct IDs for concurrent collectives.
        precision: JAX dot-product precision hint.  Applied only in the
            backward pass for the ``grad_y`` computation.

    Returns:
        Output of shape ``[m, n_local]`` where ``m = m_local * tp_size``.

    Raises:
        ValueError: If ``tp_size < 1``, input ranks are not 2, dtypes differ,
            contracting dimensions are incompatible, or tiling constraints are
            violated (see ``_pallas_impl.validate_inputs``).

    Note:
        When ``tp_size == 1`` the Pallas kernel is bypassed and the result
        is computed with a plain ``jnp.dot``.
    """
    return _all_gather_matmul_core(x, y, axis_name, rhs_transpose, bn, bk, tp_size, collective_id, precision)


__all__ = ("all_gather_matmul",)
