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

"""Public interface and kernel-registry wrapper for XLA all-gather matmul.

Exposes ``all_gather_matmul`` to the ejkernel dispatch system under the
``Platform.XLA / Backend.ANY`` key. The implementation is in
``_xla_impl_fwd`` together with an explicit custom VJP that uses
``lax.psum_scatter`` in the backward pass.
"""

from __future__ import annotations

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import all_gather_matmul as _all_gather_matmul_impl


@kernel_registry.register("all_gather_matmul", Platform.XLA, Backend.ANY)
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
    """Compute ``all_gather(x, axis=0) @ y`` across a JAX device mesh.

    Gathers local row-shards of ``x`` along ``axis_name`` (producing the full
    ``[m, k]`` matrix), then multiplies by the local column-shard ``y``.  The
    custom VJP sends a reduce-scatter back through ``x`` so the gradient has the
    same ``[m_local, k]`` shape as the input.

    This is the XLA (pure-JAX / ``lax`` collective) reference implementation.
    The ``bn``, ``bk``, ``collective_id``, and ``tp_size`` arguments are
    accepted for API compatibility but are **ignored** by this backend.

    Args:
        x: Local row-shard of the LHS matrix, shape ``[m_local, k]``.
        y: Local column-shard of the RHS matrix.  Shape is ``[k, n_local]``
            when ``rhs_transpose=False``, or ``[n_local, k]`` when
            ``rhs_transpose=True``.
        axis_name: Name of the ``pmap``/``shard_map`` axis over which to
            all-gather ``x``.
        rhs_transpose: If ``True``, ``y`` is transposed before the matmul.
            Default ``False``.
        bn: Ignored (tile-size hint for Triton/CUDA backends). Default None.
        bk: Ignored (tile-size hint for Triton/CUDA backends). Default None.
        tp_size: Optional tensor-parallelism size.  When provided, must be
            ``>= 1``; used only for input validation. Default None.
        collective_id: Ignored (CUDA NCCL collective tag). Default 0.
        precision: JAX matrix-multiplication precision. Default
            ``jax.lax.Precision.DEFAULT``.

    Returns:
        Output matrix ``[m, n_local]`` where ``m = m_local * num_devices``.

    Raises:
        ValueError: If ``x`` or ``y`` is not rank-2, dtypes differ, inner
            dimensions are incompatible, or ``tp_size < 1``.
    """
    return _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
        precision=precision,
    )


__all__ = ("all_gather_matmul",)
