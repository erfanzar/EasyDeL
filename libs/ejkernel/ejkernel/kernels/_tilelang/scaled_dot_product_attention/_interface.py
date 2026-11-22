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

"""Tile-lang scaled_dot_product_attention.

SDPA on GPU *is* FlashAttention-2 — materialising the full N×N attention
matrix is not viable. This surface forwards directly onto the
feature-complete tile-lang FlashAttention kernels: ``attention_mask``,
``bias`` / ``init_bias``, ``sliding_window``, ``causal`` and ``softmax_scale``
are all applied natively. Ragged ``cum_seqlens`` is gated with an explicit
error (the FlashAttention kernel rejects it too).
"""

from __future__ import annotations

from collections.abc import Callable

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ..flash_attention._impl import flash_attention_tilelang


@kernel_registry.register("scaled_dot_product_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """SDPA via the feature-complete tile-lang FlashAttention-2 kernels.

    Every argument is honoured natively. ``init_bias`` is materialised when
    ``bias`` is not supplied.

    Raises:
        EjkernelRuntimeError: if ``cum_seqlens_q`` / ``cum_seqlens_k`` is
            supplied — ragged-packed attention is not implemented (the
            FlashAttention kernel rejects it for the same reason).
    """
    if cum_seqlens_q is not None or cum_seqlens_k is not None:
        raise EjkernelRuntimeError(
            "tile-lang scaled_dot_product_attention: ragged cum_seqlens is not supported; "
            "pass an explicit attention_mask instead."
        )
    if bias is None and init_bias is not None:
        bias = init_bias()
    return flash_attention_tilelang(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )


__all__ = ["scaled_dot_product_attention"]
