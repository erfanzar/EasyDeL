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

"""TileLang linear-attention recurrence — public interface.

Registers :func:`recurrent` with the ejkernel kernel registry under
``("recurrent", Platform.TILELANG, Backend.GPU)`` and delegates to
:func:`~._impl.recurrent_tilelang`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._impl import recurrent_tilelang


@kernel_registry.register("recurrent", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
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
    """TileLang linear-attention recurrence with optional decay gates.

    Computes the gated linear-attention state update:

        h_{t+1} = h_t * (gamma * exp(g) * exp(gk)) * exp(gv) + k_t ⊗ v_t
        o_t = h_{t+1} @ q_t * softmax_scale

    where ``⊗`` is an outer product.  Only the gates supplied as non-``None``
    are applied; omitted gates default to 1.0.

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float.
        key: ``[batch, seq_len, num_kv_heads, qk_head_dim]`` float.
            ``num_kv_heads`` must divide ``num_heads`` (GQA).
        value: ``[batch, seq_len, num_kv_heads, v_head_dim]`` float.
        g: Optional per-element per-head log-space decay
            ``[batch, seq_len, num_heads, qk_head_dim]``; applied to the full
            state before the outer-product update.
        g_gamma: Optional per-head static (or per-batch) scalar decay.
            Accepted shapes: ``(num_heads,)`` or ``(batch, num_heads)``; in
            packed mode ``(num_seqs, num_heads)`` or ``(1, num_heads)``.
        gk: Optional per-element per-head Q/K-axis gate
            ``[batch, seq_len, num_heads, qk_head_dim]``.
        gv: Optional per-element per-head V-axis gate
            ``[batch, seq_len, num_heads, v_head_dim]``.
        softmax_scale: Multiplier applied to ``o_t``; defaults to
            ``1/sqrt(qk_head_dim)``.
        initial_state: Optional fp32 initial state
            ``[batch, num_heads, qk_head_dim, v_head_dim]`` or
            ``[num_seqs, num_heads, qk_head_dim, v_head_dim]`` in packed mode.
            Defaults to zeros.
        reverse: If ``True``, scan the sequence right-to-left.
        cu_seqlens: Optional int32 CSR pointer array ``[num_seqs + 1]`` enabling
            packed (varlen) mode with ``batch == 1``.
        block_k: Accepted for API compatibility with Triton; ignored by TileLang.
        block_v: Accepted for API compatibility with Triton; ignored by TileLang.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``(output, final_state)`` where:

        * ``output``: ``[batch, seq_len, num_heads, v_head_dim]`` in the input dtype.
        * ``final_state``: ``[batch, num_heads, qk_head_dim, v_head_dim]`` fp32.
          In packed mode the shapes are ``[1, ...]`` and ``[num_seqs, ...]``
          respectively.

    Raises:
        ValueError: on shape or dtype validation failures.
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError("tile-lang recurrent packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise ValueError("tile-lang recurrent packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise ValueError("tile-lang recurrent packed initial_state must have one state per sequence.")
    return recurrent_tilelang(
        query,
        key,
        value,
        initial_state=initial_state,
        softmax_scale=softmax_scale,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["recurrent"]
