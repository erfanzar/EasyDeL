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

"""Public interface and kernel-registry entry for reduce-scatter matmul."""

from __future__ import annotations

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import reduce_scatter_matmul as _reduce_scatter_matmul_impl


@kernel_registry.register("reduce_scatter_matmul", Platform.XLA, Backend.ANY)
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
    """Compute ``reduce_scatter(x @ y.T, scatter_dim=0)`` over a device mesh.

    Each device holds a shard of ``k`` (the contraction dimension).  The
    kernel first computes the partial outer product ``x @ y.T`` locally, then
    applies ``psum_scatter`` along ``axis_name`` to sum partial results across
    devices and distribute row shards of the output.

    Registered under ``"reduce_scatter_matmul"`` for ``Platform.XLA``,
    ``Backend.ANY``.  Must be called inside a ``jax.pmap`` / ``jax.shard_map``
    context so that ``axis_name`` is defined.

    Note:
        ``bm``, ``bn``, ``bk``, and ``collective_id`` are accepted for API
        parity with hardware-specific backends (e.g., CUDA) but are ignored
        here.

    Args:
        x: Left operand, shape ``[m, k_shard]`` on each device.
        y: Right operand, shape ``[n, k_shard]`` on each device.
        axis_name: JAX device-mesh axis name for the collective operation.
        bm: Block size for M dimension.  Ignored in this backend.
        bn: Block size for N dimension.  Ignored in this backend.
        bk: Block size for K dimension.  Ignored in this backend.
        tp_size: Optional tensor-parallel degree used for validation only.
            Must be >= 1 when provided.
        collective_id: Ignored in this backend.
        precision: JAX ``lax.Precision`` for the matrix multiply.

    Returns:
        Output shard of shape ``[m_local, n]`` where
        ``m_local = m / num_devices_along_axis_name``.
    """
    return _reduce_scatter_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        bm=bm,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
        precision=precision,
    )


__all__ = ("reduce_scatter_matmul",)
