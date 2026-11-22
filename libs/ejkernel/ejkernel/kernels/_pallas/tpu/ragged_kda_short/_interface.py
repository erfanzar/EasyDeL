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

"""Pallas TPU registry entry for the short-row phase of ``ragged_kda`` (one kernel for every row of 1..W tokens)."""

from __future__ import annotations

from jaxtyping import Array

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.ragged_kda._xla_impl import RowPlan, short_phase
from ._pallas_impl import ragged_kda_short_pallas, supported


@kernel_registry.register("ragged_kda_short", Platform.PALLAS, Backend.TPU, priority=10)
def ragged_kda_short(
    out: Array,
    pools: tuple[Array | None, Array | None, Array | None, Array],
    *,
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    weights: tuple[Array, Array, Array] | None,
    plan: RowPlan,
    short_row_max: int,
    use_qk_l2norm: bool,
) -> tuple[Array, tuple[Array | None, Array | None, Array | None, Array]]:
    """Short rows of a ``ragged_kda`` step (Pallas TPU; conv fused, state in VMEM, pools updated in place).

    Same contract as the XLA reference ``short_phase``. Post-conv inputs (``weights=None``), head sizes that are not
    multiples of 128 and 1-byte float streams or pools run the reference instead.

    Args:
        out: ``[T, H, V]`` fp32 output buffer.
        pools: ``(q_conv_state, k_conv_state, v_conv_state, recurrent_state)``.
        query: Queries ``[T, H, K]`` (raw with ``weights``, post-conv without).
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        weights: ``(q, k, v)`` conv taps ``[dim, d_conv]``, or ``None``.
        plan: The step's row plan.
        short_row_max: ``W`` (static).
        use_qk_l2norm: L2-normalise queries and keys.

    Returns:
        ``(out, pools)`` with the short rows' outputs and captured slots written.
    """
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        weights=weights,
        plan=plan,
        short_row_max=short_row_max,
        use_qk_l2norm=use_qk_l2norm,
    )
    dtypes = (query.dtype, key.dtype, value.dtype, *(p.dtype for p in pools if p is not None))
    if weights is None or not supported(query.shape[-1], value.shape[-1], dtypes):
        return short_phase(out, pools, **kwargs)
    return ragged_kda_short_pallas(out, pools, **kwargs)


__all__ = ("ragged_kda_short",)
