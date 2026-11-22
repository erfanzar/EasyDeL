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

"""Tile-lang RWKV-6 forward."""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import rwkv6_tilelang


@kernel_registry.register("rwkv6", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Tile-lang RWKV-6 forward (DPLR scan, forward + backward via tile-lang).

    Registered as ``"rwkv6"`` on ``Platform.TILELANG / Backend.GPU``.

    Args:
        r: queries (receptor), ``(batch, seq_len, num_heads, qk_head_dim)``.
        k: keys, same shape as ``r``.
        v: values, ``(batch, seq_len, num_heads, v_head_dim)``.
        w: per-step time-decay (log-space), same shape as ``r``.
        u: per-head bonus, ``(num_heads, qk_head_dim)``.
        softmax_scale: attention scale; defaults to ``1/sqrt(qk_head_dim)``
            when ``None``.
        initial_state: optional fp32 ``(batch_or_seqs, num_heads, qk_head_dim,
            v_head_dim)`` initial hidden state; defaults to all-zeros.
        reverse: run the recurrence in reverse time (default ``False``).
        cu_seqlens: optional int32 ``(num_seqs+1,)`` cumulative offsets for
            packed-sequence mode.  When given, batch must equal 1 and
            ``initial_state`` (if provided) must have ``num_seqs`` entries.

    Returns:
        ``(O, Hf)`` — ``O`` is ``(batch, seq_len, num_heads, v_head_dim)`` in
        the input dtype; ``Hf`` is fp32 ``(batch_or_seqs, num_heads,
        qk_head_dim, v_head_dim)``.

    Raises:
        EjkernelRuntimeError: for shape / dtype violations.
    """
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise EjkernelRuntimeError("tile-lang rwkv6 packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang rwkv6 packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise EjkernelRuntimeError("tile-lang rwkv6 packed initial_state must have one state per sequence.")
    qk_head_dim = r.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_head_dim)
    return rwkv6_tilelang(
        r,
        k,
        v,
        w,
        u,
        initial_state=initial_state,
        softmax_scale=scale,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["rwkv6"]
