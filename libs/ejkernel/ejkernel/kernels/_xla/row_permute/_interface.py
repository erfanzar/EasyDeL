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

"""XLA registry entry (priority 0 fallback) for the run-table row permutation."""

from __future__ import annotations

from jaxtyping import Array, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import row_permute_xla


@kernel_registry.register("row_permute", Platform.XLA, Backend.ANY)
def row_permute(
    x: Array,
    runs: Int[Array, "num_runs 3"],
    inverse_runs: Int[Array, "num_runs 3"] | None = None,
    init: Array | None = None,
    *,
    out_rows: int | None = None,
    chunk_rows: int = 128,
) -> Array:
    """``out[dst + j] = x[src + j]`` for every run ``(dst, src, length)``, as one gather.

    Args:
        x: ``[rows, ...]`` source rows.
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``; zero-length runs are padding, destination ranges
            must not overlap.
        inverse_runs: Pallas backward table; ignored here (autodiff of the gather is exact).
        init: Optional ``[out_rows, ...]`` values for rows no run covers (zeros when ``None``).
        out_rows: Destination rows (default ``x.shape[0]``, or ``init.shape[0]``).
        chunk_rows: Pallas streaming window; ignored here.

    Returns:
        ``[out_rows, ...]`` in ``x.dtype``.
    """
    # Not ``del``: the registry would treat a deleted parameter as unsupported, but both are accepted hints here
    # (the gather needs no window size, and its autodiff needs no inverse table).
    _unused = (inverse_runs, chunk_rows)
    return row_permute_xla(x, runs, init=init, out_rows=out_rows)


__all__ = ("row_permute",)
