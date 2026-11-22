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

"""Tile-lang RWKV-7 (DPLR) forward."""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import rwkv7_tilelang


@kernel_registry.register("rwkv7", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Tile-lang RWKV-7 DPLR scan (standard parameterisation).

    Registered as ``"rwkv7"`` on ``Platform.TILELANG / Backend.GPU``.

    Implements the standard ``(a, b)`` variant of the DPLR update::

        hb       = sum_i b[t,i] * h[:, i]
        h_next   = h * exp(w_t) + a_t outer hb + k_t outer v_t
        o_t      = r_t @ h_next

    Args:
        r: queries ``(batch, seq_len, num_heads, qk_head_dim)``.
        w: per-step time-decay (log-space), same shape as ``r``.
        k: keys, same shape as ``r``.
        v: values ``(batch, seq_len, num_heads, v_head_dim)``.
        a: DPLR ``a`` coefficients, same shape as ``r``.
        b: DPLR ``b`` coefficients, same shape as ``r``.
        softmax_scale: attention scale; defaults to ``1/sqrt(qk_head_dim)``.
        initial_state: optional fp32 initial state ``(batch_or_seqs,
            num_heads, qk_head_dim, v_head_dim)``; defaults to zeros.
        reverse: run in reverse time (default ``False``).
        cu_seqlens: optional int32 ``(num_seqs+1,)`` packed-sequence offsets;
            batch must be 1 when provided.
        block_v: Accepted for API compatibility with Triton; ignored by TileLang.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``(O, Hf)`` — ``O`` is ``(batch, seq_len, num_heads, v_head_dim)``
        in the input dtype; ``Hf`` is fp32.

    Raises:
        EjkernelRuntimeError: for shape / dtype violations.
    """
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise EjkernelRuntimeError("tile-lang rwkv7 packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang rwkv7 packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise EjkernelRuntimeError("tile-lang rwkv7 packed initial_state must have one state per sequence.")
    qk_head_dim = r.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_head_dim)
    o, hf = rwkv7_tilelang(
        r,
        w,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        softmax_scale=scale,
        reverse=reverse,
        mul_variant=False,
        cu_seqlens=cu_seqlens,
    )
    return o, hf


@kernel_registry.register("rwkv7_mul", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Tile-lang RWKV-7 multiplicative parameterisation (kk/a variant).

    Registered as ``"rwkv7_mul"`` on ``Platform.TILELANG / Backend.GPU``.

    The ``mul_variant=True`` kernel re-parameterises the DPLR update as::

        a_loc = kk * a        (element-wise)
        b_loc = -kk
        hb       = sum_i b_loc[i] * h[:, i]
        h_next   = h * exp(w_t) + a_loc outer hb + k_t outer v_t
        o_t      = r_t @ h_next

    Args:
        r: queries ``(batch, seq_len, num_heads, qk_head_dim)``.
        w: per-step time-decay (log-space), same shape as ``r``.
        k: keys, same shape as ``r``.
        v: values ``(batch, seq_len, num_heads, v_head_dim)``.
        kk: the ``kk`` factor (maps to ``a_loc = kk * a``), same shape as ``r``.
        a: the ``a`` factor, same shape as ``r``.
        softmax_scale: attention scale; defaults to ``1/sqrt(qk_head_dim)``.
        initial_state: optional fp32 initial state; defaults to zeros.
        reverse: run in reverse time (default ``False``).
        cu_seqlens: optional int32 ``(num_seqs+1,)`` packed-sequence offsets.
        block_v: Accepted for API compatibility with Triton; ignored by TileLang.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``(O, Hf)`` — same shapes as :func:`rwkv7`.

    Raises:
        EjkernelRuntimeError: for shape / dtype violations.
    """
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise EjkernelRuntimeError("tile-lang rwkv7_mul packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang rwkv7_mul packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise EjkernelRuntimeError("tile-lang rwkv7_mul packed initial_state must have one state per sequence.")
    qk_head_dim = r.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_head_dim)
    return rwkv7_tilelang(
        r,
        w,
        k,
        v,
        kk,
        a,
        initial_state=initial_state,
        softmax_scale=scale,
        reverse=reverse,
        mul_variant=True,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["rwkv7", "rwkv7_mul"]
