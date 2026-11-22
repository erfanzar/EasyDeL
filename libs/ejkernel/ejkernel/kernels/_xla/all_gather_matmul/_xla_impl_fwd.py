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

"""XLA all-gather matmul — forward pass and custom VJP registration.

Algorithm
---------
Forward:
    1. ``lax.all_gather(x, axis_name, axis=0, tiled=True)`` → ``x_full [m, k]``
    2. ``x_full @ y`` (or ``x_full @ y.T`` when ``rhs_transpose=True``)
       → output ``[m, n_local]``

The gathered ``x_full`` is saved in the residual so the backward pass can
reuse it instead of re-gathering.

Backward (see ``_xla_impl_bwd``):
    - ``grad_y = x_full.T @ dy`` (transposed for ``rhs_transpose`` case)
    - ``grad_x_partial = dy @ y.T``
    - ``grad_x = lax.psum_scatter(grad_x_partial, axis=0)`` → ``[m_local, k]``
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ._xla_impl_bwd import all_gather_matmul_backward


def _validate_inputs(
    x: jax.Array,
    y: jax.Array,
    *,
    rhs_transpose: bool,
    tp_size: int | None,
) -> None:
    """Validate all-gather matmul inputs before entering JIT-compiled code.

    Runs at Python (trace) time; raises ``ValueError`` on shape/dtype/rank
    violations so errors are reported before compilation begins.

    Args:
        x: Local LHS shard; must be rank-2.
        y: RHS matrix; must be rank-2 and share dtype with ``x``.
        rhs_transpose: If True, ``y``'s *second* dimension must equal
            ``x.shape[1]``; otherwise ``y``'s *first* dimension must match.
        tp_size: Optional tensor-parallelism size; validated to be ``>= 1``.

    Raises:
        ValueError: On any shape/dtype/rank violation.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be rank-2 tensors, got {x.ndim=} and {y.ndim=}.")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must share dtype, got {x.dtype=} and {y.dtype=}.")

    k = x.shape[1]
    if rhs_transpose:
        k_from_y = y.shape[1]
    else:
        k_from_y = y.shape[0]

    if k != k_from_y:
        raise ValueError(
            "Incompatible matmul shapes for all_gather_matmul: "
            f"x.shape={x.shape}, y.shape={y.shape}, rhs_transpose={rhs_transpose}."
        )

    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1 when provided, got {tp_size}.")


def _forward_impl(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    *,
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "m n_local"], Float[Array, "m k"]]:
    """All-gather ``x`` and compute the matrix product.

    Args:
        x: Local LHS shard ``[m_local, k]``.
        y: RHS shard ``[k, n_local]`` (or ``[n_local, k]`` if ``rhs_transpose``).
        axis_name: Collective axis name passed to ``lax.all_gather``.
        rhs_transpose: Whether to transpose ``y`` before the dot.
        precision: JAX precision for the dot product.

    Returns:
        Tuple of:
            - ``out``: Product ``[m, n_local]``.
            - ``x_full``: Gathered ``[m, k]`` (saved as residual for the VJP).
    """
    x_full = lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
    if rhs_transpose:
        out = jnp.dot(x_full, y.T, precision=precision)
    else:
        out = jnp.dot(x_full, y, precision=precision)
    return out, x_full


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _all_gather_matmul_core(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m n_local"]:
    """Core all-gather matmul with custom VJP.

    Non-differentiable arguments ``axis_name``, ``rhs_transpose``, and
    ``precision`` (positions 2-4) are captured by the VJP rules via
    ``nondiff_argnums``.

    Args:
        x: Local LHS shard ``[m_local, k]``.
        y: RHS shard.
        axis_name: Name of the collective axis.
        rhs_transpose: Whether ``y`` is transposed before the dot.
        precision: JAX dot precision.

    Returns:
        All-gathered product ``[m, n_local]``.
    """
    out, _ = _forward_impl(x, y, axis_name=axis_name, rhs_transpose=rhs_transpose, precision=precision)
    return out


def _all_gather_matmul_core_fwd(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
):
    """VJP forward rule: compute the output and return residuals.

    Returns:
        Tuple of (output ``[m, n_local]``, residuals ``(x, y, x_full)``).
        ``x_full`` is the gathered LHS saved to avoid a second all-gather in
        the backward pass.
    """
    out, x_full = _forward_impl(x, y, axis_name=axis_name, rhs_transpose=rhs_transpose, precision=precision)
    return out, (x, y, x_full)


def _all_gather_matmul_core_bwd(
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m n_local"],
):
    """VJP backward rule: compute gradients via reduce-scatter for ``x``.

    Args:
        axis_name: Non-diff argument forwarded from ``custom_vjp``.
        rhs_transpose: Non-diff argument forwarded from ``custom_vjp``.
        precision: Non-diff argument forwarded from ``custom_vjp``.
        residual: Tuple ``(x_local, y, x_full)`` from the forward rule.
        dy: Upstream gradient w.r.t. the output, shape ``[m, n_local]``.

    Returns:
        Tuple ``(grad_x, grad_y)`` with shapes ``[m_local, k]`` and
        matching ``y``'s shape.
    """
    _, y, x_full = residual
    grad_x, grad_y = all_gather_matmul_backward(
        dy,
        y,
        x_full,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        precision=precision,
    )
    return grad_x, grad_y


_all_gather_matmul_core.defvjp(_all_gather_matmul_core_fwd, _all_gather_matmul_core_bwd)


def all_gather_matmul(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    *,
    axis_name: str,
    rhs_transpose: bool = False,
    bn: int | None = None,
    bk: int | None = None,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n_local"]:
    """Compute ``all_gather(x, axis=0) @ y`` with an explicit VJP.

    Validates inputs, drops arguments unused by this backend (``bn``, ``bk``,
    ``collective_id``), then delegates to ``_all_gather_matmul_core`` which
    carries the ``custom_vjp`` registration.

    Args:
        x: Local LHS shard ``[m_local, k]``.
        y: RHS shard ``[k, n_local]`` or ``[n_local, k]``.
        axis_name: Name of the collective mesh axis.
        rhs_transpose: Transpose ``y`` before the product. Default False.
        bn: Ignored (Triton/CUDA tile-size hint).
        bk: Ignored (Triton/CUDA tile-size hint).
        tp_size: Optional validation-only TP size (``>= 1`` if given).
        collective_id: Ignored (CUDA NCCL tag).
        precision: JAX dot precision. Default ``Precision.DEFAULT``.

    Returns:
        Output matrix ``[m, n_local]``.
    """
    del bn, bk, collective_id
    _validate_inputs(x, y, rhs_transpose=rhs_transpose, tp_size=tp_size)
    return _all_gather_matmul_core(x, y, axis_name, rhs_transpose, precision)


__all__ = ("all_gather_matmul",)
