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

"""Pallas TPU registry entry for the experts-over-tp local expert FFN.

Same algorithm and custom VJP as the XLA entry (see ``ejkernel.kernels._xla.moe_local_ffn``), with TPU primitives:

* token-row dispatch and the output-cotangent gather: the ``moe_row_gather`` per-row DMA kernel;
* the token-major combine and the input-gradient gather: the ``moe_row_gather_sum`` kernel (fp32 accumulation);
* grouped matmuls: the Pallas grouped matmul with the whole contraction in one tile (:func:`gmm_tiling`);
* weight gradients: the transposed grouped matmul accumulating in place across windows and chunks
  (``existing_out``, :func:`tgmm_tiling`). Empty experts keep their running sum: the kernel visits every group and
  stores ``0 + existing``.

Both tilings keep the MXU busier than the :data:`GMM_TILE` / :data:`TGMM_TILE` fallbacks but need the raised
scoped-VMEM limit of :func:`local_ffn_vmem_limit`.

Shapes the DMA kernels cannot take (see :func:`local_ffn_unsupported_reason`) run the XLA entry after a one-time
``RuntimeWarning``.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.moe_local_ffn._xla_impl import (
    LocalFfnPrimitives,
    check_local_ffn_shapes,
    local_ffn,
    moe_local_ffn_xla,
    plan_local_ffn,
)
from ..grouped_matmul._pallas_impl import grouped_matmul as _pallas_gmm
from ..grouped_matmul._pallas_impl import transposed_grouped_matmul as _pallas_tgmm
from ..moe_row_gather._pallas_impl import (
    INDEX_BLOCK,
    gather_rows,
    gather_sum_block_tokens,
    gather_sum_rows,
    gather_sum_unsupported_reason,
    gather_unsupported_reason,
)

F32 = jnp.float32
GMM_TILE = (512, 1024, 1024)
"""Fallback ``(tm, tk, tn)`` of the forward/data-gradient grouped matmuls at the default scoped-VMEM limit."""
TGMM_TILE = (512, 512, 1024)
"""Fallback ``(tm, tk, tn)`` of the in-place weight-gradient transposed grouped matmul at the default limit."""
VMEM_LIMIT_BYTES = 80 * 2**20
"""Scoped-VMEM limit of the large local-FFN tiles (capped at 90% of the chip's VMEM)."""
_VMEM_HEADROOM = 0.75
"""A large tile is used only when its estimated buffers stay under this fraction of the limit."""
ROW_ALIGN = INDEX_BLOCK
"""Window rows are a multiple of the gather step (1024), which is also a multiple of ``tm``."""

_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def token_align(top_k: int) -> int:
    """Chunk-size multiple required by the gather-sum kernel's token step."""
    return gather_sum_block_tokens(top_k)


def local_ffn_unsupported_reason(
    x,
    weights,
    experts,
    w_gate,
    w_up,
    w_down,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
) -> str | None:
    """Why the Pallas entry would run the XLA implementation for these arguments (``None``: the kernels run).

    Only ``.shape`` / ``.dtype`` are read.
    """
    del weights, w_up, w_down
    if jax.default_backend() != "tpu":
        return f"backend is {jax.default_backend()}, not tpu"
    for name, arr in (("x", x), ("w_gate", w_gate)):
        if jnp.dtype(arr.dtype) != jnp.bfloat16:
            return f"{name} dtype {jnp.dtype(arr.dtype).name} (needs bfloat16)"
    tokens, hidden = x.shape
    top_k = experts.shape[1]
    inter = w_gate.shape[2]
    if inter % 128:
        return f"expert width {inter} is not a multiple of 128"
    tc, r, _, _ = plan_local_ffn(
        tokens, top_k, tokens_per_chunk, rows_per_window, token_align=token_align(top_k), row_align=ROW_ALIGN
    )
    if tc < INDEX_BLOCK:
        return f"{tc} tokens per chunk is fewer than {INDEX_BLOCK}"
    return gather_unsupported_reason((tc, hidden), x.dtype, r) or gather_sum_unsupported_reason(
        (r, hidden), x.dtype, tc, top_k, F32
    )


def _tiles(m: int, k: int, n: int, tile: tuple[int, int, int]) -> tuple[int, int, int]:
    return min(m, tile[0]), min(k, tile[1]), min(n, tile[2])


@functools.cache
def local_ffn_vmem_limit() -> int | None:
    """80 MiB capped at 90% of the local TPU's VMEM; ``None`` (default-limit tiles) when the capacity is unknown."""
    try:
        capacity = int(getattr(pltpu.get_tpu_info(), "vmem_capacity_bytes", 0) or 0)
    except Exception:  # off TPU or an older runtime
        return None
    return min(VMEM_LIMIT_BYTES, int(capacity * 0.9)) if capacity > 0 else None


def gmm_tiling(m: int, k: int, n: int, itemsize: int, vmem_limit: int | None) -> tuple[tuple[int, int, int], int | None]:
    """``((tm, tk, tn), vmem_limit_bytes)`` for a forward/data-gradient grouped matmul ``[m, k] x [G, k, n]``.

    ``tm = 256``, the whole contraction in one tile (no k-loop revisits the accumulator) and the output columns split
    in two, so both v5p TensorCores take a column half (the column axis is the kernel's only parallel grid axis).
    Falls back to :data:`GMM_TILE` at the default limit when there is no raised limit, the shape does not divide, or
    the double-buffered tiles would not fit.
    """
    if vmem_limit and m % 256 == 0 and k % 128 == 0 and n % 256 == 0:
        tm, tk, tn = 256, k, n // 2
        need = 2 * (tm * tk + tk * tn + tm * tn) * itemsize + tm * tn * 4
        if need <= _VMEM_HEADROOM * vmem_limit:
            return (tm, tk, tn), vmem_limit
    return _tiles(m, k, n, GMM_TILE), None


def tgmm_tiling(
    m: int, k: int, n: int, itemsize: int, acc_itemsize: int, vmem_limit: int | None
) -> tuple[tuple[int, int, int], int | None]:
    """``((tm, tk, tn), vmem_limit_bytes)`` for the in-place weight gradient ``acc[G, k, n] += lhs[m, k]^T rhs[m, n]``.

    ``(256, 2048, 1024)`` (capped at the operand sizes); the aliased accumulator block is double-buffered on both
    sides of the kernel on top of the fp32 scratch. Falls back to :data:`TGMM_TILE` at the default limit under the
    same conditions as :func:`gmm_tiling`.
    """
    if vmem_limit and m % 256 == 0 and k % 128 == 0 and n % 128 == 0:
        tm, tk, tn = 256, min(k, 2048), min(n, 1024)
        need = 2 * (tm * tk + tm * tn) * itemsize + 4 * tk * tn * acc_itemsize + tk * tn * 4
        if need <= _VMEM_HEADROOM * vmem_limit:
            return (tm, tk, tn), vmem_limit
    return _tiles(m, k, n, TGMM_TILE), None


def pallas_primitives() -> LocalFfnPrimitives:
    """TPU primitives: DMA row gathers, Pallas grouped matmul, in-place transposed grouped matmul."""
    vmem_limit = local_ffn_vmem_limit()

    def gather(src, rows):
        return gather_rows(src, rows)[0]

    def gather_sum(y, idx, w):
        # Negative slots (other windows' / devices' routes, ~(ep-1)/ep of all slots) issue no row DMA.
        return gather_sum_rows(y, idx, w, F32, skip_negative=True)

    def gmm(lhs, rhs, group_sizes, transpose_rhs=False):
        m, k = lhs.shape
        n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
        tiling, limit = gmm_tiling(m, k, n, jnp.dtype(lhs.dtype).itemsize, vmem_limit)
        return _pallas_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=tiling,
            transpose_rhs=transpose_rhs,
            vmem_limit_bytes=limit,
        )

    def tgmm(lhs, rhs, group_sizes, acc):
        m, k = lhs.shape
        n = rhs.shape[1]
        itemsize, acc_itemsize = jnp.dtype(lhs.dtype).itemsize, jnp.dtype(acc.dtype).itemsize
        tiling, limit = tgmm_tiling(m, k, n, itemsize, acc_itemsize, vmem_limit)
        return _pallas_tgmm(
            lhs.T,
            rhs,
            group_sizes,
            preferred_element_type=acc.dtype,
            tiling=tiling,
            existing_out=acc,
            vmem_limit_bytes=limit,
        )

    return LocalFfnPrimitives(gather=gather, gather_sum=gather_sum, gmm=gmm, tgmm=tgmm)


@kernel_registry.register("moe_local_ffn", Platform.PALLAS, Backend.TPU, priority=10)
def moe_local_ffn(
    x: Float[Array, "tokens hidden"],
    weights: Float[Array, "tokens k"],
    experts: Int[Array, "tokens k"],
    w_gate: Float[Array, "experts hidden inter"],
    w_up: Float[Array, "experts hidden inter"],
    w_down: Float[Array, "experts inter hidden"],
    activation: Callable[[Array, Array], Array] | None = None,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
) -> Float[Array, "tokens hidden"]:
    """This device's partial MoE output over its local experts (dropless dynamic row windows, custom VJP).

    Args:
        x: ``[tokens, hidden]`` token rows (replicated over the expert-sharding axis).
        weights: ``[tokens, k]`` routing (combine) weights; differentiable.
        experts: ``[tokens, k]`` local expert ids; ids outside ``[0, experts)`` are other devices' routes.
        w_gate: ``[experts, hidden, inter]`` gate projection.
        w_up: ``[experts, hidden, inter]`` up projection.
        w_down: ``[experts, inter, hidden]`` down projection.
        activation: ``(gate, up) -> h``; default ``silu(gate) * up``. Must close over no traced values.
        tokens_per_chunk: Tokens per outer chunk (default: all tokens); rounded up to the gather-sum token step.
        rows_per_window: Rows per inner window (default: every route of a chunk); rounded up to a multiple of 1024.
            Windows past the actual local row count are skipped, so no route is ever dropped.

    Returns:
        ``[tokens, hidden]`` in ``x.dtype``: ``sum_{j local} weights[t, j] * FFN_{experts[t, j]}(x[t])``.
    """
    check_local_ffn_shapes(x, weights, experts, w_gate, w_up, w_down)
    reason = local_ffn_unsupported_reason(x, weights, experts, w_gate, w_up, w_down, tokens_per_chunk, rows_per_window)
    if reason is not None:
        _warn_once(f"moe_local_ffn: the Pallas TPU kernels cannot run here ({reason}); using the XLA implementation.")
        return moe_local_ffn_xla(
            x, weights, experts, w_gate, w_up, w_down, activation, tokens_per_chunk, rows_per_window
        )
    return local_ffn(
        pallas_primitives(),
        x,
        weights,
        experts,
        w_gate,
        w_up,
        w_down,
        activation=activation,
        tokens_per_chunk=tokens_per_chunk,
        rows_per_window=rows_per_window,
        token_align=token_align(experts.shape[1]),
        row_align=ROW_ALIGN,
    )


__all__ = (
    "gmm_tiling",
    "local_ffn_unsupported_reason",
    "local_ffn_vmem_limit",
    "moe_local_ffn",
    "pallas_primitives",
    "tgmm_tiling",
    "token_align",
)
