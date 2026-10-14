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

"""Ragged decode attention interface for variable-length decoding.

This module provides the public API for attention during decoding with
variable-length sequences. Supports MQA/GQA configurations with sliding
window and attention sink capabilities.
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_decode_mqa_xla


@kernel_registry.register("ragged_decode_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Ragged MQA/GQA decode-phase attention over a flat KV cache (XLA reference).

    Implements attention for a single decode step per sequence where each sequence
    in the batch may have a different valid KV range.  Supports Multi-Query
    Attention (MQA, ``num_kv_heads=1``) and Grouped-Query Attention (GQA,
    ``num_kv_heads < num_q_heads``).

    Attention is computed with an online-softmax scan over KV blocks of size
    ``fwd_params.kv_blocksize`` (default 256), so the full KV tensor never
    needs to be materialised all at once.

    Registered under ``"ragged_decode_attention"`` for ``Platform.XLA``,
    ``Backend.ANY``.  This is the numerical reference; other backends must
    match its output.

    Args:
        query: Single-token query per sequence.
            Shape: ``[batch, num_q_heads, head_dim]``.
        key: Full KV-cache key tensor for the batch.
            Shape: ``[batch, seq_len, num_kv_heads, head_dim]``.
        value: Full KV-cache value tensor for the batch.
            Shape: ``[batch, seq_len, num_kv_heads, head_dim]``.
        sequence_start: First valid KV position (inclusive) per sequence.
            Shape: ``[batch]``, dtype ``int32``.
        sequence_end: One-past-last valid KV position (exclusive) per sequence.
            Shape: ``[batch]``, dtype ``int32``.  The query token is assumed to
            sit at position ``sequence_end[i] - 1``.
        softmax_scale: Multiplicative scale applied to QK^T logits.
            Defaults to ``1 / sqrt(head_dim)`` when ``None``.
        fwd_params: Optional ``FwdParams`` dataclass.  Only ``kv_blocksize`` is
            used; all other fields are ignored.  Defaults to ``FwdParams()``
            (i.e. ``kv_blocksize=256``).
        sliding_window: Optional ``(left, right)`` window sizes for local
            attention.  ``left`` tokens to the left and ``right`` tokens to the
            right of the current query position are attended to.  ``None`` means
            full attention over ``[sequence_start, sequence_end)``.
        logits_soft_cap: Optional tanh soft-capping value.  Applied as
            ``logits_soft_cap * tanh(logits / logits_soft_cap)`` before softmax.
        softmax_aux: Optional attention-sink auxiliary logits.
            Shape ``[num_sinks]`` or ``[num_q_heads, num_sinks]`` or
            ``[num_kv_heads, num_sinks]``.  These logits seed the online-softmax
            running maximum and normaliser so that the model can absorb
            probability mass into "sink" tokens even when those tokens are not
            present in the KV window.

    Returns:
        Attention output of shape ``[batch, num_q_heads, head_dim]``.

    Note:
        This is a pure XLA/JAX implementation and runs on CPU, GPU, and TPU.
        For TPU production workloads prefer the Pallas implementation.
    """
    return ragged_decode_mqa_xla(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
