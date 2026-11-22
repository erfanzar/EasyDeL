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

"""Registry entry (Pallas, TPU) for the run-table row permutation, differentiable via ``jax.custom_vjp``.

Shapes/dtypes the streaming kernel does not support (rank != 2, rows not a multiple of 16, width not a multiple of
128, 1-byte dtypes, 2-byte dtypes other than bfloat16) and non-TPU backends route to the XLA gather.
"""

from __future__ import annotations

import jax
from jaxtyping import Array, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.row_permute._xla_impl import resolve_out_rows, row_permute_xla
from ._pallas_impl import DEFAULT_CHUNK_ROWS, check_supported, permute_rows_runs


@kernel_registry.register("row_permute", Platform.PALLAS, Backend.TPU, priority=10)
def row_permute(
    x: Array,
    runs: Int[Array, "num_runs 3"],
    inverse_runs: Int[Array, "num_runs 3"] | None = None,
    init: Array | None = None,
    *,
    out_rows: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Array:
    """``out[dst + j] = x[src + j]`` for every run ``(dst, src, length)``, streamed through VMEM on TPU.

    Args:
        x: ``[rows, D]`` source rows.
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``; zero-length runs are padding, destination ranges
            must not overlap.
        inverse_runs: Inverse table for the backward (default: ``runs`` with ``dst``/``src`` swapped). The custom
            VJP equals the gather's gradient when no source row is read twice.
        init: Optional ``[out_rows, D]`` values for rows no run covers (zeros when ``None``).
        out_rows: Destination rows (default ``x.shape[0]``, or ``init.shape[0]``).
        chunk_rows: Source rows per streamed window (multiple of 16).

    Returns:
        ``[out_rows, D]`` in ``x.dtype``.
    """
    n_out = resolve_out_rows(x, init, out_rows)
    if jax.default_backend() != "tpu" or check_supported(tuple(x.shape), x.dtype, n_out) is not None:
        return row_permute_xla(x, runs, init=init, out_rows=n_out)
    return permute_rows_runs(x, runs, inverse_runs, init, out_rows=n_out, chunk_rows=chunk_rows)


__all__ = ("row_permute",)
