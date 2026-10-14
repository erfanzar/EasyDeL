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

"""Tile-lang ``all_gather_matmul`` — single-device subset (v0).

This v0 implementation supports only the ``tp_size=1`` (or ``tp_size=None``)
degenerate case where no collective communication takes place.  The gathered
tensor is simply the local ``x`` and the result is ``dense_matmul_tilelang(x, y)``.

Gated parameters (raise ``EjkernelRuntimeError``):
    * ``tp_size > 1`` — real tensor-parallel all-gather not implemented.
    * ``collective_id != 0`` and ``!= None`` — not supported.
    * ``precision != DEFAULT`` — not supported.
    * ``axis_name`` other than ``None`` / ``"__tp_dummy__"`` — no native collective.
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


@kernel_registry.register("all_gather_matmul", Platform.TILELANG, Backend.GPU)
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
    """All-gather then matmul — v0 single-device path only.

    For ``tp_size=1`` (or ``tp_size=None``) this is equivalent to
    ``x @ y`` (or ``x @ y.T`` when ``rhs_transpose=True``) computed via
    :func:`dense_matmul_tilelang`.  No actual collective communication occurs.

    Args:
        x: local shard ``(m_local, k)``.
        y: weight shard ``(k, n_local)`` or ``(n_local, k)`` when
            ``rhs_transpose=True``.
        axis_name: name of the JAX collective axis.  Must be ``None`` or the
            sentinel ``"__tp_dummy__"`` in this v0 implementation.
        rhs_transpose: if True, ``y`` is in transposed ``(n_local, k)`` layout
            and will be transposed before the matmul.
        bn: ignored (tile-size hint for future native kernel).
        bk: ignored (tile-size hint for future native kernel).
        tp_size: tensor-parallel degree.  Must be ``None`` or ``1``; larger
            values raise ``EjkernelRuntimeError``.
        collective_id: must be ``None`` or ``0``.
        precision: must be ``jax.lax.Precision.DEFAULT``.

    Returns:
        ``(m, n_local)`` matmul result in ``x.dtype``.

    Raises:
        EjkernelRuntimeError: if any gated parameter is set to an unsupported
            value (see module docstring).
    """
    if bn is not None and bn <= 0:
        raise EjkernelRuntimeError("tile-lang all_gather_matmul requires bn > 0 when provided.")
    if bk is not None and bk <= 0:
        raise EjkernelRuntimeError("tile-lang all_gather_matmul requires bk > 0 when provided.")
    if collective_id not in (None, 0):
        raise EjkernelRuntimeError("tile-lang all_gather_matmul does not support nonzero collective_id.")
    if precision != jax.lax.Precision.DEFAULT:
        raise EjkernelRuntimeError("tile-lang all_gather_matmul does not yet support custom precision.")
    if tp_size not in (None, 1):
        raise EjkernelRuntimeError("tile-lang all_gather_matmul v0 does not yet support tp_size > 1.")
    if axis_name not in (None, "__tp_dummy__"):
        raise EjkernelRuntimeError("tile-lang all_gather_matmul needs a native collective kernel for real axis_name.")

    x_gathered = x

    y_eff = jnp.swapaxes(y, 0, 1) if rhs_transpose else y
    return dense_matmul_tilelang(x_gathered, y_eff.astype(x_gathered.dtype))


__all__ = ["all_gather_matmul"]
