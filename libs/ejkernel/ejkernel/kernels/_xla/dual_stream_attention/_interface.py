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

"""Dual-Stream Hybrid Attention interface (XLA reference, differentiable)."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import dual_stream_attention_xla


@kernel_registry.register("dual_stream_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def dual_stream_attention(
    query: Float[Array, "batch num_q_heads q_len head_dim"],
    k_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    v_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    k_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    v_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    g_q_seg: Int[Array, "batch q_len"] | None = None,
    g_kv_seg: Int[Array, "batch kv_g_len"] | None = None,
    l_q_seg: Int[Array, "batch q_len"] | None = None,
    l_kv_seg: Int[Array, "batch kv_l_len"] | None = None,
    global_mask: Bool[Array, "batch mask_heads q_len kv_g_len"] | None = None,
    local_mask: Bool[Array, "batch mask_heads q_len kv_l_len"] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int = 128,
    overlap: bool = True,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    q_chunk_size: int = 512,
    kv_chunk_size: int = 1024,
) -> Float[Array, "batch num_q_heads q_len head_dim"]:
    """Dual-Stream Hybrid Attention (XLA fallback and numerical reference).

    Registered under ``"dual_stream_attention"`` for ``Platform.XLA`` /
    ``Backend.ANY``. Jointly normalizes attention over a global KV stream and a
    local sliding-window KV stream under a single fp32 softmax denominator.

    Args:
        query: Queries ``[batch, num_q_heads, q_len, head_dim]``.
        k_global: Global-stream keys ``[batch, num_kv_heads, kv_g_len, head_dim]``.
        v_global: Global-stream values ``[batch, num_kv_heads, kv_g_len, head_dim]``.
        k_local: Local-stream keys ``[batch, num_kv_heads, kv_l_len, head_dim]``.
        v_local: Local-stream values ``[batch, num_kv_heads, kv_l_len, head_dim]``.
        g_q_seg: Optional global-stream query segment IDs ``[batch, q_len]``.
        g_kv_seg: Optional global-stream KV segment IDs ``[batch, kv_g_len]``.
        l_q_seg: Optional local-stream query segment IDs ``[batch, q_len]``.
        l_kv_seg: Optional local-stream KV segment IDs ``[batch, kv_l_len]``.
        global_mask: Optional explicit boolean mask ``[batch, 1|Hq, q_len, kv_g_len]``.
        local_mask: Optional explicit boolean mask ``[batch, 1|Hq, q_len, kv_l_len]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        sliding_window: Width ``W`` of the local sliding window.
        overlap: If ``True``, global stream attends causally ``k <= q`` (overlapping
            with the local window); if ``False``, global attends disjointly ``k <= q - W``.
        fwd_params: Accepted for interface parity with Pallas; unused by XLA.
        bwd_params: Accepted for interface parity with Pallas; unused by XLA.
        q_chunk_size: Query block size for memory-bounded chunked XLA execution.
        kv_chunk_size: Global-KV block size for causal-skip XLA execution.

    Returns:
        Attention output ``[batch, num_q_heads, q_len, head_dim]``.
    """
    del fwd_params, bwd_params
    return dual_stream_attention_xla(
        query=query,
        k_global=k_global,
        v_global=v_global,
        k_local=k_local,
        v_local=v_local,
        g_q_seg=g_q_seg,
        g_kv_seg=g_kv_seg,
        l_q_seg=l_q_seg,
        l_kv_seg=l_kv_seg,
        global_mask=global_mask,
        local_mask=local_mask,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        overlap=overlap,
        q_chunk_size=q_chunk_size,
        kv_chunk_size=kv_chunk_size,
    )
