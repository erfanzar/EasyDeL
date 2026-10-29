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

"""Tile-lang ring_attention.

Ring attention is FlashAttention with a ring-style coordination over a
distributed mesh axis. The single-device path (``axis_name is None``)
reduces exactly to FlashAttention and forwards onto the feature-complete
tile-lang FA kernels with every score-space argument applied natively
(segment ids, bias, sliding window, soft cap, attention sinks). The
native multi-device ring (``axis_name`` set) and the opaque ``mask_builder``
callback are gated with explicit errors rather than silently ignored.
Non-default contraction precision delegates to the registered XLA ring
implementation before native capability checks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ..flash_attention._impl import flash_attention_tilelang

Mask = Any


@kernel_registry.register("ring_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "sink_heads num_sinks"] | None = None,
    bias: Float[Array, "bias_batch bias_heads bias_seq_len_q bias_seq_len_k"] | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    fused_backward: bool = False,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Single-device tile-lang ring-attention (forward + backward).

    Forwards onto the feature-complete FlashAttention kernels. ``chunk_size``
    and ``fused_backward`` are scheduling hints — the tile-lang kernels are
    already fully fused and the backward is always the split dQ/dKdV design,
    so they impose no behaviour change. Causal / sliding-window masking is by
    sequence position; ``q_position_ids`` / ``kv_position_ids`` participate
    through that ordering. Explicit non-default ``precision`` uses the
    registered XLA ring implementation, which honors contraction precision
    and accepts broadcast bias and shared or per-head attention sinks.
    ``Precision.DEFAULT`` (or ``"default"``) preserves the native TileLang path.

    Raises:
        EjkernelRuntimeError: on the native DEFAULT-precision path if
            ``axis_name`` is set (multi-device ring coordination is not yet
            bridged through TVM-FFI) or if ``mask_builder`` is supplied (pass
            an explicit ``bias`` / segment ids instead of an opaque callback).
    """
    if precision not in (lax.Precision.DEFAULT, "default"):
        return kernel_registry.get("ring_attention", platform=Platform.XLA, backend=Backend.ANY)(
            query=query,
            key=key,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_position_ids=q_position_ids,
            kv_position_ids=kv_position_ids,
            softmax_aux=softmax_aux,
            bias=bias,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            axis_name=axis_name,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            fused_backward=fused_backward,
            precision=precision,
        )
    if axis_name is not None:
        raise EjkernelRuntimeError(
            "tile-lang ring_attention: multi-device ring coordination (axis_name) is not "
            "implemented; run on a single device or use the XLA backend."
        )
    if mask_builder is not None:
        raise EjkernelRuntimeError(
            "tile-lang ring_attention: mask_builder callbacks are not supported; "
            "pass an explicit bias / q_segment_ids / kv_segment_ids instead."
        )
    _scheduling_hints = (chunk_size, fused_backward, q_position_ids, kv_position_ids)
    return flash_attention_tilelang(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )


__all__ = ["ring_attention"]
