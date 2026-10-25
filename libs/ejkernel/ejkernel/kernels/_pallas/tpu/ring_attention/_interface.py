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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

from collections.abc import Callable

import jaxtyping
from beartype import beartype
from jax import lax

from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import Array, BwdParams, Float, FwdParams, Int
from ._pallas_impl_fwd import ring_attention as _ring_attention_impl


@kernel_registry.register("ring_attention", Platform.PALLAS, Backend.TPU)
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
    """Computes ring attention using Splash Attention kernels on TPU.

    This implementation uses JAX's splash attention with ring communication
    topology for distributed attention computation across devices.

    Args:
        query: Query tensor [batch, q_len, num_heads, head_dim].
        key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
        q_segment_ids: Optional query segment IDs [batch, q_len]. Used for
            cross-document masking in packed-sequence training.
        kv_segment_ids: Optional KV segment IDs [batch, kv_len].
        q_position_ids: Optional query position IDs [batch, q_len]. Forwarded
            to the XLA ring-attention fallback; not used by the Pallas splash
            path when ``axis_name`` is set.
        kv_position_ids: Optional KV position IDs [batch, kv_len]. Forwarded
            to the XLA ring-attention fallback; not used by the Pallas splash
            path when ``axis_name`` is set.
        softmax_aux: Optional attention sink logits. 1-D ``[num_sinks]`` (shared) is
            reduced via ``logsumexp`` and passed to the splash kernel as ``sinks``.
            2-D ``[num_heads | num_kv_heads | 1, num_sinks]`` is reduced exactly to
            per-head sinks (``logsumexp`` over the sink axis) and also uses the
            splash kernel when ``axis_name`` is set; without a ring axis 2-D sinks
            use the XLA ring kernel, which consumes them natively.
        bias: Optional attention bias (not supported in the Pallas splash path;
            only passed through to the XLA fallback).
        mask_builder: Optional custom mask builder function. Forces the XLA
            ring-attention fallback when ``axis_name`` is ``None``.
        sliding_window: Sliding window size as int (symmetric) or
            ``(left, right)`` tuple for local attention.
        chunk_size: Chunk size for chunked causal attention (Llama4-style).
        causal: Whether to use causal masking.
        logits_soft_cap: Soft cap for attention logits.
        softmax_scale: Scaling factor for attention scores. Defaults to
            ``head_dim ** -0.5`` when ``None``.
        axis_name: Name of the ring communication axis. ``None`` routes to
            the XLA single-device path.
        fwd_params: Forward pass block size parameters (``q_blocksize``,
            ``kv_blocksize``).
        bwd_params: Backward pass block size parameters (``q_blocksize``,
            ``kv_blocksize``).
        fused_backward: Whether to use fused backward kernel (forwarded to the
            XLA fallback only; the Pallas path manages its own backward via
            ``ring_splash_attention``).
        precision: JAX contraction precision. Explicit non-default precision
            uses the registered XLA ring implementation because Splash does
            not expose this control. The default retains the Splash path.

    Returns:
        Attention output [batch, q_len, num_heads, head_dim].

    Raises:
        NotImplementedError: If ``bias`` is provided when ``axis_name`` is set
            (bias is not supported by the Pallas splash path).
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
    return _ring_attention_impl(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        softmax_aux,
        bias,
        mask_builder,
        sliding_window,
        chunk_size,
        causal,
        logits_soft_cap,
        softmax_scale,
        axis_name,
        fwd_params,
        bwd_params,
        fused_backward,
    )


__all__ = ("ring_attention",)
