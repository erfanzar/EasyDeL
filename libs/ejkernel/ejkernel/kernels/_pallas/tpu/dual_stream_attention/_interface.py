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

"""Dual-Stream Hybrid Attention interface (Pallas TPU)."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import dual_stream_attention_tpu


@kernel_registry.register("dual_stream_attention", Platform.PALLAS, Backend.TPU)
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
    """Dual-Stream Hybrid Attention via TPU Pallas SplashAttention + exact custom VJP.

    Registered under ``"dual_stream_attention"`` for ``Platform.PALLAS`` /
    ``Backend.TPU``. Matches the XLA reference in value and gradient and falls
    back to XLA automatically for shapes outside SplashAttention alignment.
    """
    return dual_stream_attention_tpu(
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
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        q_chunk_size=q_chunk_size,
        kv_chunk_size=kv_chunk_size,
    )
