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

"""TileLang entry point for Gated Linear Attention (GLA).

Registers ``gla`` under ``Platform.TILELANG / Backend.GPU``.  The
implementation delegates to the shared recurrent kernel
:func:`ejkernel.kernels._tilelang.recurrent._impl.recurrent_tilelang` with
key-side gating supplied through the ``g`` / ``g_gamma`` arguments.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ..recurrent._impl import recurrent_tilelang


@kernel_registry.register("gla", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent_gla(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Run Gated Linear Attention (GLA) on GPU via TileLang.

    Delegates to the shared recurrent kernel with optional key-side gate
    decay (``g``), optional global per-head slope (``g_gamma``), and
    optional packed-sequence mode (``cu_seqlens``).

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        key: ``[batch, seq_len, num_kv_heads, qk_head_dim]`` float tensor.
            Multi-query attention (``num_kv_heads < num_heads``) is supported
            by the underlying recurrent kernel.
        value: ``[batch, seq_len, num_kv_heads, v_head_dim]`` float tensor.
        g: Optional ``[batch, seq_len, num_heads, qk_head_dim]`` per-element
            key-gate in log-space (applied as ``exp(g)`` decay).  Mutually
            exclusive with ``g_gamma`` (pass at most one).
        g_gamma: Optional ``[..., num_heads]`` static per-head decay slope.
            Mutually exclusive with ``g``.
        softmax_scale: Optional scalar scale for attention logits.  Defaults
            to ``1 / sqrt(qk_head_dim)`` inside the recurrent kernel.
        initial_state: Optional ``[..., num_heads, qk_head_dim, v_head_dim]``
            float32 initial hidden state.
        reverse: If ``True``, run the scan in reverse time order.
        cu_seqlens: Optional ``int32`` cumulative sequence-length vector of
            shape ``[num_seqs + 1]`` for packed (variable-length) sequences.
            Requires ``batch == 1`` and ``initial_state`` (when given) to
            have first dimension equal to ``num_seqs``.
        block_k: Accepted for operation-level config parity with Triton; the
            current TileLang recurrent kernel uses its internal tile schedule.
        block_v: Accepted for operation-level config parity with Triton; the
            current TileLang recurrent kernel uses its internal tile schedule.
        num_warps: Accepted for operation-level config parity with Triton.
        num_stages: Accepted for operation-level config parity with Triton.

    Returns:
        A 2-tuple ``(output, final_state)`` where:

        - ``output``: ``[batch, seq_len, num_heads, v_head_dim]``.
        - ``final_state``: ``[..., num_heads, qk_head_dim, v_head_dim]``
          float32.

    Raises:
        EjkernelRuntimeError: If packed-sequence constraints are violated.
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise EjkernelRuntimeError("tile-lang gla packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang gla packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise EjkernelRuntimeError("tile-lang gla packed initial_state must have one state per sequence.")
    return recurrent_tilelang(
        query,
        key,
        value,
        initial_state=initial_state,
        softmax_scale=softmax_scale,
        g=g,
        g_gamma=g_gamma,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["recurrent_gla"]
